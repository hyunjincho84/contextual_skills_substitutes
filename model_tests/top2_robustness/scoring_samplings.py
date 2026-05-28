# -*- coding: utf-8 -*-

import os

CACHE_DIR = "/home/jovyan/LEM_data2/hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CACHE_DIR, "hub")

import re
import glob
import argparse
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional

import torch
import pandas as pd
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== CONFIG =====
BASE_DATA_DIR = os.environ.get("BASE_DATA_DIR", "/home/jovyan/LEM_data2/data")
DEFAULT_INPUT_GLOB = os.path.join(BASE_DATA_DIR, "bert_sampling_variants_from_summary", "*", "sv_summary_llama_full_bert_top10_*_sampling.csv.gz")
DEFAULT_LLAMA_CHECKPOINT = "meta-llama/Llama-3.2-3B"

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 512
WINDOW_SIZE = 256
SAMPLE_PER_FILE = None
RANDOM_SEED = 42
ANCHOR_SEARCH_RADIUS = 2
MIN_LLAMA3_TRANSFORMERS_VERSION = (4, 45, 0)

SAMPLING_COLUMNS = [
    "sample_top2_like",
    "sample_top10",
    "sample_thresh_0005_sampling",
    "sample_thresh_0001_sampling",
    "sample_temp_15",
    "sample_temp_20",
]

torch.set_float32_matmul_precision("high")


# ---------------- Helpers ----------------
def parse_version_tuple(version: str) -> Tuple[int, int, int]:
    parts = [int(x) for x in re.findall(r"\d+", version)[:3]]
    parts.extend([0] * (3 - len(parts)))
    return tuple(parts[:3])


def assert_llama_checkpoint_supported(llama_ckpt: str) -> None:
    ckpt_name = llama_ckpt.lower()
    needs_new_rope = re.search(r"llama[-_/]?3\.[123]", ckpt_name) is not None
    if not needs_new_rope:
        return

    installed = parse_version_tuple(transformers.__version__)
    if installed >= MIN_LLAMA3_TRANSFORMERS_VERSION:
        return

    min_version = ".".join(str(x) for x in MIN_LLAMA3_TRANSFORMERS_VERSION)
    raise RuntimeError(
        f"{llama_ckpt} needs transformers>={min_version}; "
        f"this Python environment has transformers=={transformers.__version__}. "
        "Update the environment used by run_top2_pipeline.sh, for example:\n"
        "/home/jovyan/.venv/torch2.3.0-py3.11-cuda12.1/bin/python -m pip install "
        f"'transformers>={min_version},<5'"
    )


def extract_year_month(path: str) -> str:
    m = re.search(r"(\d{4}-\d{2})", os.path.basename(path))
    if not m:
        raise ValueError(f"Could not parse YYYY-MM from {path}")
    return m.group(1)


def list_sampling_files(input_glob: str) -> Dict[str, str]:
    files = sorted(glob.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {input_glob}")
    ym2path = {}
    for fp in files:
        ym = extract_year_month(fp)
        ym2path[ym] = fp
    return ym2path


def normalize_for_match(text: str) -> str:
    if text is None:
        return ""
    s = str(text).lower().strip()
    s = s.replace("▁", "")
    s = s.replace("Ġ", "")
    s = s.replace("##", "")
    s = re.sub(r"^[^a-z0-9]+", "", s)
    s = re.sub(r"[^a-z0-9]+$", "", s)
    return s


def get_skill_first_word(skill: str) -> str:
    skill = str(skill).strip()
    if not skill:
        return ""
    return normalize_for_match(skill.split()[0])


def token_matches_skill_start(token_str: str, skill_first_word: str) -> bool:
    tok = normalize_for_match(token_str)
    if not tok or not skill_first_word:
        return False
    return tok == skill_first_word or tok.startswith(skill_first_word) or skill_first_word.startswith(tok)


def filter_intersection_nonempty_rows(df: pd.DataFrame, methods: List[str]) -> pd.DataFrame:
    use_methods = [m for m in methods if m in df.columns]
    if not use_methods:
        return df.iloc[0:0].copy()

    mask = pd.Series(True, index=df.index)
    for m in use_methods:
        mask &= df[m].notna()
        mask &= df[m].astype(str).str.strip().ne("")

    return df[mask].reset_index(drop=True)


# ---------------- Old anchor helper ----------------
def _anchor_by_offset(sent: str, char_start: int, tokenizer):
    enc = tokenizer(
        sent,
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    if len(input_ids) > 0 and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if len(offsets) > 0 and isinstance(offsets[0], list) and len(offsets[0]) == 2 and isinstance(offsets[0][0], int):
        pass
    elif len(offsets) > 0 and isinstance(offsets[0], list) and isinstance(offsets[0][0], tuple):
        offsets = offsets[0]

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    k = -1
    for i, (s_char, e_char) in enumerate(offsets):
        if s_char <= char_start < e_char:
            k = i
            break

    if k < 0:
        for i, (s_char, e_char) in enumerate(offsets):
            if s_char >= char_start:
                k = i
                break

    return tokens, k


def refine_anchor_nearby(
    tokens: List[str],
    base_idx: int,
    target_skill: str,
    radius: int = 2,
) -> int:
    if base_idx < 0 or base_idx >= len(tokens):
        return base_idx

    skill_first_word = get_skill_first_word(target_skill)
    if not skill_first_word:
        return base_idx

    left = max(0, base_idx - radius)
    right = min(len(tokens) - 1, base_idx + radius)

    candidates = []
    for i in range(left, right + 1):
        if token_matches_skill_start(tokens[i], skill_first_word):
            candidates.append(i)

    if not candidates:
        return base_idx

    candidates.sort(key=lambda i: (abs(i - base_idx), i))
    return candidates[0]


# ---------------- Build sentences: OLD METHOD ----------------
def build_original_window(truth: str, masked: str, tokenizer, window_size: int, max_len: int, anchor_search_radius: int):
    truth = str(truth)
    masked = str(masked)

    prefix, sep, suffix = masked.partition("[MASK]")
    if sep == "":
        tokens = tokenizer.tokenize(masked)
        text = tokenizer.convert_tokens_to_string(tokens)
        return text, tokens, -1

    sent = prefix + truth + suffix
    char_start = len(prefix)

    tokens, k = _anchor_by_offset(sent, char_start, tokenizer)
    if k < 0:
        return sent, tokens, -1

    k = refine_anchor_nearby(tokens, k, truth, radius=anchor_search_radius)

    truth_token_len = len(tokenizer.tokenize(truth))
    start = max(0, k - window_size)
    end = min(len(tokens), k + window_size + truth_token_len)

    win = tokens[start:end]
    k_local = k - start
    text = tokenizer.convert_tokens_to_string(win)

    return text, win, k_local


def build_fill_window(fill_skill: str, masked: str, tokenizer, window_size: int, max_len: int, anchor_search_radius: int):
    fill_skill = str(fill_skill)
    masked = str(masked)

    prefix, sep, suffix = masked.partition("[MASK]")
    if sep == "":
        tokens = tokenizer.tokenize(masked)
        text = tokenizer.convert_tokens_to_string(tokens)
        return text, tokens, -1

    sent = prefix + fill_skill + suffix
    char_start = len(prefix)

    tokens, idx = _anchor_by_offset(sent, char_start, tokenizer)
    if idx < 0:
        return sent, tokens, -1

    idx = refine_anchor_nearby(tokens, idx, fill_skill, radius=anchor_search_radius)

    fill_token_len = len(tokenizer.tokenize(fill_skill))
    start = max(0, idx - window_size)
    end = min(len(tokens), idx + window_size + fill_token_len)

    win = tokens[start:end]
    fill_local = idx - start
    text = tokenizer.convert_tokens_to_string(win)

    return text, win, fill_local


# ---------------- LLaMA SV computation ----------------
@torch.no_grad()
def compute_sv_batch(
    texts_orig: List[str],
    texts_sub: List[str],
    k_indices: List[int],
    tokenizer,
    model,
    sent_batch: int,
    device: str,
):
    out = []
    use_amp = device.startswith("cuda")

    for s in range(0, len(texts_orig), sent_batch):
        o_batch = texts_orig[s:s + sent_batch]
        p_batch = texts_sub[s:s + sent_batch]
        k_batch = k_indices[s:s + sent_batch]

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            enc_o = tokenizer(
                o_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN
            ).to(device)
            outputs_o = model(
                **enc_o,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=False
            )

            enc_p = tokenizer(
                p_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN
            ).to(device)
            outputs_p = model(
                **enc_p,
                output_attentions=False,
                output_hidden_states=True,
                use_cache=False
            )

        hs_o = outputs_o.hidden_states
        hs_p = outputs_p.hidden_states

        top4_o = torch.cat(hs_o[-4:], dim=-1)
        top4_p = torch.cat(hs_p[-4:], dim=-1)

        att_stack = torch.stack(outputs_o.attentions, dim=0)
        mean_att = att_stack.mean(dim=(0, 2))

        B = top4_o.size(0)
        for b in range(B):
            k_loc = k_batch[b]
            if k_loc < 0:
                out.append(float("nan"))
                continue

            T_o = int(enc_o["attention_mask"][b].sum())
            T_p = int(enc_p["attention_mask"][b].sum())
            T = min(T_o, T_p)

            H_o = top4_o[b, :T, :]
            H_p = top4_p[b, :T, :]

            cos = (H_o * H_p).sum(dim=-1) / (
                H_o.norm(dim=-1) * H_p.norm(dim=-1)
            ).clamp_min(1e-12)

            k_use = min(k_loc, T - 1)

            w = mean_att[b, :T, k_use].clamp_min(0.0)
            if w.sum() > 0:
                w = w / w.sum()
            else:
                w = torch.ones_like(w) / T

            sv = (w * cos).sum().item()
            out.append(sv)

    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


# ---------------- Per-method evaluation ----------------
def evaluate_one_method_on_df(
    method_name: str,
    df: pd.DataFrame,
    tokenizer,
    model,
    window_size: int,
    sent_batch: int,
    anchor_search_radius: int,
    debug_fail_reason: bool = False,
    device: str = DEFAULT_DEVICE,
) -> Tuple[pd.DataFrame, dict]:
    needed = {"truth", "masked_sentence", method_name}
    if not needed.issubset(df.columns):
        raise ValueError(f"[{method_name}] missing columns {needed - set(df.columns)}")

    work = df.copy()
    work["truth"] = work["truth"].astype(str)
    work["masked_sentence"] = work["masked_sentence"].astype(str)
    work[method_name] = work[method_name].astype(str).str.strip()

    res = pd.DataFrame(index=work.index)
    res[f"{method_name}__chosen_substitute"] = work[method_name]
    res[f"{method_name}__sv_llama"] = pd.NA
    res[f"{method_name}__anchor1_text"] = ""
    res[f"{method_name}__anchor2_text"] = ""

    orig_texts, sub_texts = [], []
    k_indices = []
    valid_index = []
    anchor1_texts, anchor2_texts = [], []

    fail_stats = {
        "total_input_rows": len(work),
        "valid_rows": 0,
        "fail_orig_none": 0,
        "fail_sub_none": 0,
        "fail_k_loc": 0,
        "fail_fill_loc": 0,
    }

    for i, row in work.iterrows():
        truth = str(row["truth"]).strip()
        masked = str(row["masked_sentence"])
        chosen = str(row[method_name]).strip()

        orig_text, orig_toks, k_loc = build_original_window(
            truth=truth,
            masked=masked,
            tokenizer=tokenizer,
            window_size=window_size,
            max_len=MAX_LEN,
            anchor_search_radius=anchor_search_radius,
        )
        if k_loc < 0:
            fail_stats["fail_orig_none"] += 1
            if debug_fail_reason:
                prefix, _, suffix = masked.partition("[MASK]")
                print("[ORIG FAIL]")
                print("truth:", truth)
                print("prefix tail:", prefix[-250:])
                print("suffix head:", suffix[:250])
            continue

        sub_text, sub_toks, fill_loc = build_fill_window(
            fill_skill=chosen,
            masked=masked,
            tokenizer=tokenizer,
            window_size=window_size,
            max_len=MAX_LEN,
            anchor_search_radius=anchor_search_radius,
        )
        if fill_loc < 0:
            fail_stats["fail_sub_none"] += 1
            if debug_fail_reason:
                prefix, _, suffix = masked.partition("[MASK]")
                print("[SUB FAIL]")
                print("chosen:", chosen)
                print("prefix tail:", prefix[-250:])
                print("suffix head:", suffix[:250])
            continue

        valid_index.append(i)
        orig_texts.append(orig_text)
        sub_texts.append(sub_text)
        k_indices.append(k_loc)

        anchor1_texts.append(
            tokenizer.convert_tokens_to_string([orig_toks[k_loc]])
            if 0 <= k_loc < len(orig_toks) else ""
        )
        anchor2_texts.append(
            tokenizer.convert_tokens_to_string([sub_toks[fill_loc]])
            if 0 <= fill_loc < len(sub_toks) else ""
        )

    fail_stats["valid_rows"] = len(valid_index)

    if len(valid_index) > 0:
        sv_vals = compute_sv_batch(
            texts_orig=orig_texts,
            texts_sub=sub_texts,
            k_indices=k_indices,
            tokenizer=tokenizer,
            model=model,
            sent_batch=sent_batch,
            device=device,
        )

        res.loc[valid_index, f"{method_name}__sv_llama"] = sv_vals
        res.loc[valid_index, f"{method_name}__anchor1_text"] = anchor1_texts
        res.loc[valid_index, f"{method_name}__anchor2_text"] = anchor2_texts

    if debug_fail_reason:
        fail_total = (
            fail_stats["fail_orig_none"]
            + fail_stats["fail_sub_none"]
            + fail_stats["fail_k_loc"]
            + fail_stats["fail_fill_loc"]
        )
        print(
            f"[{method_name}] fail_breakdown "
            f"(orig_none={fail_stats['fail_orig_none']}, "
            f"sub_none={fail_stats['fail_sub_none']}, "
            f"k_loc={fail_stats['fail_k_loc']}, "
            f"fill_loc={fail_stats['fail_fill_loc']}, "
            f"total_fail={fail_total}, "
            f"valid={fail_stats['valid_rows']}, "
            f"input={fail_stats['total_input_rows']})"
        )

    return res.reset_index(drop=True), fail_stats


# ---------------- Main ----------------
def parse_gpus(gpus: str) -> List[str]:
    spec = str(gpus).strip().lower()
    if spec in ("", "auto"):
        if torch.cuda.is_available():
            return [str(i) for i in range(torch.cuda.device_count())]
        return ["cpu"]

    vals = [g.strip() for g in str(gpus).split(",") if g.strip()]
    if not vals:
        return ["cpu"]
    if not torch.cuda.is_available():
        return ["cpu"]

    max_gpu = torch.cuda.device_count()
    usable = []
    for g in vals:
        if g.lower() == "cpu":
            usable.append("cpu")
            continue
        try:
            gpu_idx = int(g)
        except ValueError:
            continue
        if 0 <= gpu_idx < max_gpu:
            usable.append(str(gpu_idx))
        else:
            print(f"[WARN] Ignoring unavailable GPU id {g}; visible GPU count is {max_gpu}.")
    return usable or ["cpu"]


def resolve_device(gpu_id: str) -> str:
    if torch.cuda.is_available() and str(gpu_id).lower() != "cpu":
        return f"cuda:{gpu_id}"
    return "cpu"


def load_llama(args, device: str, worker_label: str):
    print(f"[{worker_label}] Loading LLaMA from: {args.llama_ckpt} on {device}")
    os.makedirs(args.cache_dir, exist_ok=True)
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN", None)

    tokenizer = AutoTokenizer.from_pretrained(
        args.llama_ckpt,
        token=hf_token,
        cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {
        "torch_dtype": "auto",
        "token": hf_token,
        "cache_dir": args.cache_dir,
    }
    if device.startswith("cuda"):
        model_kwargs["device_map"] = {"": device}

    model = AutoModelForCausalLM.from_pretrained(
        args.llama_ckpt,
        **model_kwargs,
    ).eval()
    if device == "cpu":
        model = model.to(device)

    try:
        model.set_attn_implementation("eager")
        print(f"[{worker_label}] Set attention implementation to 'eager' for LLaMA.")
    except Exception:
        print(f"[{worker_label}] WARNING: eager attention not available.")

    return tokenizer, model


def output_path_for(csv_path: str, ym: str) -> str:
    return os.path.join(
        os.path.dirname(csv_path),
        f"sv_summary_llama_all_methods_{ym}.csv.gz"
    )



def process_one_month(ym: str, csv_path: str, args, tokenizer, model, device: str, worker_label: str):
    out_path = output_path_for(csv_path, ym)

    if args.resume and os.path.exists(out_path):
        print(f"[{worker_label}][SKIP][resume] already exists: {out_path}")
        return

    df = pd.read_csv(csv_path)

    if "truth" not in df.columns or "masked_sentence" not in df.columns:
        print(f"[{worker_label}][WARN] Missing truth/masked_sentence in {csv_path}, skip.")
        return

    method_cols = [m for m in list(args.methods) if m in df.columns]
    if not method_cols:
        print(f"[{worker_label}][WARN] No requested sampling columns found in {csv_path}, skip.")
        return

    df_intersection = filter_intersection_nonempty_rows(df, method_cols)

    print(
        f"[{worker_label}][{ym}] total_rows={len(df):,}, "
        f"intersection_rows={len(df_intersection):,}"
    )

    if len(df_intersection) == 0:
        print(f"[{worker_label}][WARN] No rows remain after intersection filtering in {csv_path}, skip.")
        return

    sample_per_file = args.sample_per_file
    if args.debug and args.debug_rows is not None:
        sample_per_file = args.debug_rows

    if sample_per_file is not None and len(df_intersection) > sample_per_file:
        df_eval = df_intersection.sample(
            n=sample_per_file,
            random_state=args.random_seed
        ).reset_index(drop=True)
    else:
        df_eval = df_intersection.reset_index(drop=True)

    print(f"[{worker_label}][{ym}] eval_rows_after_sampling={len(df_eval):,}")
    if args.debug:
        print(f"[{worker_label}][{ym}][DEBUG] methods={method_cols}")
        print(f"[{worker_label}][{ym}][DEBUG] input columns={list(df.columns)}")

    out_df = df_eval.copy()

    for method_name in method_cols:
        method_res, fail_stats = evaluate_one_method_on_df(
            method_name=method_name,
            df=df_eval,
            tokenizer=tokenizer,
            model=model,
            window_size=args.window_size,
            sent_batch=args.sent_batch,
            anchor_search_radius=args.anchor_search_radius,
            debug_fail_reason=(args.debug_fail_reason or args.debug),
            device=device,
        )

        out_df = pd.concat([out_df, method_res], axis=1)

        n_valid = pd.to_numeric(out_df[f"{method_name}__sv_llama"], errors="coerce").notna().sum()
        print(
            f"[{worker_label}][{method_name}] valid_sv_rows={n_valid}, total_rows={len(out_df)}, "
            f"fail_orig_none={fail_stats['fail_orig_none']}, "
            f"fail_sub_none={fail_stats['fail_sub_none']}, "
            f"fail_k_loc={fail_stats['fail_k_loc']}, "
            f"fail_fill_loc={fail_stats['fail_fill_loc']}"
        )

    out_df.to_csv(out_path, index=False, compression="gzip")
    print(f"[{worker_label}][ALL METHODS] saved: {out_path} (rows={len(out_df)})")


def process_months(month_items: List[Tuple[str, str]], args, gpu_id: str, worker_idx: int):
    device = resolve_device(gpu_id)
    worker_label = f"worker{worker_idx}:gpu{gpu_id}"
    if device.startswith("cuda"):
        torch.cuda.set_device(device)

    tokenizer, model = load_llama(args, device, worker_label)

    for ym, csv_path in tqdm(
        month_items,
        desc=f"[{worker_label}] SV(LLaMA)",
        unit="month",
        position=worker_idx,
    ):
        process_one_month(ym, csv_path, args, tokenizer, model, device, worker_label)


def split_round_robin(items: List[Tuple[str, str]], n: int) -> List[List[Tuple[str, str]]]:
    chunks = [[] for _ in range(n)]
    for i, item in enumerate(items):
        chunks[i % n].append(item)
    return chunks


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB)
    ap.add_argument("--llama-ckpt", default=DEFAULT_LLAMA_CHECKPOINT)
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    ap.add_argument("--sent-batch", type=int, default=4)
    ap.add_argument(
        "--sample-per-file",
        default=SAMPLE_PER_FILE,
        help="Rows to sample per month. Use -1, none, or all for all rows.",
    )
    ap.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--anchor-search-radius", type=int, default=ANCHOR_SEARCH_RADIUS)
    ap.add_argument("--gpus", default="auto", help="Comma-separated GPU ids, 'auto' to use all visible GPUs, or 'cpu'.")
    ap.add_argument("--no-parallel", action="store_true", help="Run on a single device even when multiple GPUs are listed.")
    ap.add_argument("--debug", action="store_true", help="Debug mode: run a small subset with verbose diagnostics on one device.")
    ap.add_argument("--debug-months", type=int, default=1, help="Number of months to process in --debug mode.")
    ap.add_argument("--debug-rows", type=int, default=10, help="Rows per month to process in --debug mode.")
    ap.add_argument(
        "--methods",
        nargs="*",
        default=SAMPLING_COLUMNS,
        help="Sampling columns to evaluate.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip months whose final output file already exists.",
    )
    ap.add_argument(
        "--debug-fail-reason",
        action="store_true",
        help="Print detailed failure breakdown.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    assert_llama_checkpoint_supported(args.llama_ckpt)

    if isinstance(args.sample_per_file, str):
        spf = args.sample_per_file.strip().lower()
        if spf in ("", "none", "all", "-1"):
            args.sample_per_file = None
        else:
            args.sample_per_file = int(spf)

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Requested GPUs: {args.gpus}")
    print(f"HF cache dir: {args.cache_dir}")
    print(f"Sample per file: {args.sample_per_file if args.sample_per_file is not None else 'ALL'}")
    print(f"Anchor search radius: {args.anchor_search_radius}")
    print(f"Debug mode: {args.debug}")

    ym2path = list_sampling_files(args.input_glob)
    months = sorted(ym2path.keys())
    if args.debug:
        months = months[:args.debug_months]
    month_items = [(ym, ym2path[ym]) for ym in months]
    print(f"Months found: {months}")

    gpus = parse_gpus(args.gpus)
    if args.debug:
        gpus = gpus[:1]

    use_parallel = (not args.no_parallel and not args.debug and len(gpus) > 1 and torch.cuda.is_available())
    if not use_parallel:
        process_months(month_items, args, gpus[0], 0)
        return

    chunks = split_round_robin(month_items, len(gpus))
    ctx = mp.get_context("spawn")
    procs = []
    for worker_idx, (gpu_id, chunk) in enumerate(zip(gpus, chunks)):
        if not chunk:
            continue
        p = ctx.Process(target=process_months, args=(chunk, args, gpu_id, worker_idx))
        p.start()
        procs.append(p)

    failed = False
    for p in procs:
        p.join()
        if p.exitcode != 0:
            failed = True
            print(f"[ERROR] worker pid={p.pid} exited with code {p.exitcode}")

    if failed:
        raise RuntimeError("At least one GPU worker failed.")


if __name__ == "__main__":
    main()
    
        
"""
python3 scoring_samplings.py \
  --input-glob "/home/jovyan/LEM_data2/data/bert_sampling_variants_from_summary/*/sv_summary_llama_full_bert_top10_*_sampling.csv.gz" \
  --cache-dir /home/jovyan/LEM_data2/hf_cache \
  --sample-per-file 200
"""
