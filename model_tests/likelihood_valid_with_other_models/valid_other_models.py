# -*- coding: utf-8 -*-

import os

CACHE_DIR = "/home/jovyan/LEM_data2/hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CACHE_DIR, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR, "transformers")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import re
import glob
import argparse
from typing import List, Dict, Optional, Set, Tuple

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ===== CONFIG =====
BASE_DATA_DIR = os.environ.get("BASE_DATA_DIR", "/home/jovyan/LEM_data2/data")
DEFAULT_BERT_PRED_DIR = os.path.join(BASE_DATA_DIR, "bert_pred_new", "pred")
DEFAULT_W2V_PRED_DIR = os.path.join(BASE_DATA_DIR, "skill2vec_pred_new", "pred")
DEFAULT_COND_PRED_DIR = os.path.join(BASE_DATA_DIR, "condprob_pred_new", "pred")

DEFAULT_JUDGE_CHECKPOINT = "google/gemma-2-9b-it"

OUTPUT_ROOT = os.environ.get("VALID_OUTPUT_ROOT", os.path.join(BASE_DATA_DIR, "gemma_eval_full"))
JUDGE_NAME = "gemma"
SV_COLUMN = "sv_gemma"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 512
WINDOW_SIZE = 256
ANCHOR_SEARCH_RADIUS = 2

torch.set_float32_matmul_precision("high")


# ---------------- List prediction files ----------------
def list_monthly_prediction_files(base_dir: str, years: List[str]) -> Dict[str, str]:
    ym2path = {}
    if not os.path.isdir(base_dir):
        return ym2path

    year_dirs = sorted([d for d in os.listdir(base_dir) if d.isdigit()])
    if years:
        want = set(map(str, years))
        year_dirs = [y for y in year_dirs if y in want]

    for y in year_dirs:
        ypath = os.path.join(base_dir, y)
        files = sorted(glob.glob(os.path.join(ypath, "predictions_*[0-9]-[0-9][0-9].csv.gz")))
        if not files:
            files = sorted(glob.glob(os.path.join(ypath, "predictions_*[0-9]-[0-9][0-9].csv")))
        for fp in files:
            m = re.search(r"(\d{4}-\d{2})", os.path.basename(fp))
            if not m:
                continue
            ym = m.group(1)
            ym2path[ym] = fp
    return ym2path


# ---------------- Output path helper ----------------
def get_output_dir(model_name: str, ym: Optional[str] = None) -> str:
    mapping = {
        "bert": os.path.join(OUTPUT_ROOT, "bert"),
        "conditional": os.path.join(OUTPUT_ROOT, "cond"),
        "w2v": os.path.join(OUTPUT_ROOT, "skill2_vec"),
    }
    out_dir = mapping[model_name]
    if ym and re.match(r"^\d{4}-\d{2}$", str(ym)):
        out_dir = os.path.join(out_dir, str(ym)[:4])
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


SCORING_ADDED_COLUMNS = [
    "sv_llama",
    "anchor1",
    "anchor2",
    "anchor1_text",
    "anchor2_text",
]


def get_scoring_output_path(pred_csv_path: str, model_name: str, ym: str) -> str:
    return os.path.join(
        os.path.dirname(pred_csv_path),
        f"sv_summary_llama_full_{model_name}_{ym}.csv.gz",
    )


def read_scoring_rows(scoring_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        scoring_path,
        compression="gzip" if scoring_path.endswith(".gz") else None,
    )
    drop_cols = [c for c in SCORING_ADDED_COLUMNS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def infer_model_and_ym_from_scoring_path(scoring_path: str):
    base = os.path.basename(scoring_path)
    m = re.search(r"sv_summary_llama_full_(bert|w2v|conditional)_(\d{4}-\d{2})\.csv(\.gz)?$", base)
    if not m:
        return None, None
    return m.group(1), m.group(2)


# ---------------- Anchor helpers: working old offset-based version ----------------
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


# ---------------- Build sentences: SAME AS WORKING BASELINE ----------------
def build_original_sentence(row, tokenizer, window_size, anchor_search_radius):
    truth = str(row["truth"]).strip()
    masked = str(row["masked_sentence"])

    prefix, sep, suffix = masked.partition("[MASK]")
    if sep == "":
        sent = masked
        tokens = tokenizer.tokenize(sent)
        return tokenizer.convert_tokens_to_string(tokens), tokens, -1

    sent = prefix + truth + suffix
    char_start = len(prefix)

    tokens, k = _anchor_by_offset(sent, char_start, tokenizer)
    if k < 0:
        return tokenizer.convert_tokens_to_string(tokens), tokens, -1

    k = refine_anchor_nearby(tokens, k, truth, radius=anchor_search_radius)

    truth_token_len = len(tokenizer.tokenize(truth))
    start = max(0, k - window_size)
    end = min(len(tokens), k + window_size + truth_token_len)

    win = tokens[start:end]
    k_local = k - start

    return tokenizer.convert_tokens_to_string(win), win, k_local


def build_sentence_with_fill(row, fill_skill, tokenizer, window_size, anchor_search_radius):
    fill_skill = str(fill_skill).strip()
    masked = str(row["masked_sentence"])

    prefix, sep, suffix = masked.partition("[MASK]")
    if sep == "":
        sent = masked
        tokens = tokenizer.tokenize(sent)
        return tokenizer.convert_tokens_to_string(tokens), tokens, -1

    sent = prefix + fill_skill + suffix
    char_start = len(prefix)

    tokens, idx = _anchor_by_offset(sent, char_start, tokenizer)
    if idx < 0:
        return tokenizer.convert_tokens_to_string(tokens), tokens, -1

    idx = refine_anchor_nearby(tokens, idx, fill_skill, radius=anchor_search_radius)

    fill_token_len = len(tokenizer.tokenize(fill_skill))
    start = max(0, idx - window_size)
    end = min(len(tokens), idx + window_size + fill_token_len)

    win = tokens[start:end]
    fill_start = idx - start

    return tokenizer.convert_tokens_to_string(win), win, fill_start


# ---------------- Gemma-based SV computation ----------------
@torch.no_grad()
def compute_sv_batch(texts_orig, texts_sub, k_indices, tokenizer, model, max_len, sent_batch):
    out = []
    batch_size = max(1, sent_batch)
    s = 0

    while s < len(texts_orig):
        cur_batch = min(batch_size, len(texts_orig) - s)
        o_batch = texts_orig[s:s + cur_batch]
        p_batch = texts_sub[s:s + cur_batch]
        k_batch = k_indices[s:s + cur_batch]

        try:
            with torch.cuda.amp.autocast(enabled=False):
                enc_o = tokenizer(
                    o_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_len
                ).to(DEVICE)
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
                    max_length=max_len
                ).to(DEVICE)
                outputs_p = model(
                    **enc_p,
                    output_attentions=False,
                    output_hidden_states=True,
                    use_cache=False
                )

            hs_o = outputs_o.hidden_states
            hs_p = outputs_p.hidden_states

            mean_att = None
            for att in outputs_o.attentions:
                layer_att = att.mean(dim=1)
                mean_att = layer_att if mean_att is None else mean_att + layer_att
            mean_att = mean_att / len(outputs_o.attentions)

            B = hs_o[-1].size(0)
            for b in range(B):
                k_loc = k_batch[b]
                if k_loc < 0:
                    out.append(float("nan"))
                    continue

                T_o = int(enc_o["attention_mask"][b].sum())
                T_p = int(enc_p["attention_mask"][b].sum())
                T = min(T_o, T_p)

                dot = None
                norm_o_sq = None
                norm_p_sq = None
                for H_o_layer, H_p_layer in zip(hs_o[-4:], hs_p[-4:]):
                    H_o = H_o_layer[b, :T, :]
                    H_p = H_p_layer[b, :T, :]
                    layer_dot = (H_o * H_p).sum(dim=-1)
                    layer_norm_o_sq = H_o.pow(2).sum(dim=-1)
                    layer_norm_p_sq = H_p.pow(2).sum(dim=-1)
                    dot = layer_dot if dot is None else dot + layer_dot
                    norm_o_sq = layer_norm_o_sq if norm_o_sq is None else norm_o_sq + layer_norm_o_sq
                    norm_p_sq = layer_norm_p_sq if norm_p_sq is None else norm_p_sq + layer_norm_p_sq

                cos = dot / (norm_o_sq.sqrt() * norm_p_sq.sqrt()).clamp_min(1e-12)
                k_use = min(k_loc, T - 1)

                w = mean_att[b, :T, k_use].clamp_min(0.0)
                if w.sum() > 0:
                    w = w / w.sum()
                else:
                    w = torch.ones_like(w) / T

                sv = (w * cos).sum().item()
                out.append(sv)

            del enc_o, enc_p, outputs_o, outputs_p, hs_o, hs_p, mean_att
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            s += cur_batch

        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if cur_batch == 1:
                raise
            batch_size = max(1, cur_batch // 2)
            print(f"[OOM] Reducing sent_batch from {cur_batch} to {batch_size} and retrying at row offset {s}.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# ---------------- Compute SV for one model/month ----------------
def compute_sv_for_df(
    model_name: str,
    df: pd.DataFrame,
    ym: str,
    tokenizer,
    model,
    window_size: int,
    sent_batch: int,
    anchor_search_radius: int,
    out_path: Optional[str] = None,
):
    needed = {"truth", "pred_top1", "pred_top5", "masked_sentence"}
    if not needed.issubset(df.columns):
        print(f"[{model_name}] SKIP {ym}: missing columns.")
        return

    df = df.copy()
    df["truth"] = df["truth"].astype(str)
    df["pred_top1"] = df["pred_top1"].astype(str)
    df["pred_top5"] = df["pred_top5"].astype(str)
    df["masked_sentence"] = df["masked_sentence"].astype(str)

    orig_texts, sub_texts = [], []
    k_indices, fill_indices = [], []
    anchor1_texts, anchor2_texts = [], []
    chosen_substitutions = []

    for _, row in df.iterrows():
        truth = row["truth"]
        top1 = row["pred_top1"]
        top5_list = row["pred_top5"].split("|") if row["pred_top5"] else []

        if model_name == "bert":
            chosen = top5_list[1] if (truth == top1 and len(top5_list) > 1) else top1
        else:
            chosen = top1
        chosen_substitutions.append(chosen)

        x_text, x_tokens, k_loc = build_original_sentence(
            row=row,
            tokenizer=tokenizer,
            window_size=window_size,
            anchor_search_radius=anchor_search_radius,
        )
        x_sub, x_tokens_sub, fill_loc = build_sentence_with_fill(
            row=row,
            fill_skill=chosen,
            tokenizer=tokenizer,
            window_size=window_size,
            anchor_search_radius=anchor_search_radius,
        )

        orig_texts.append(x_text)
        sub_texts.append(x_sub)
        k_indices.append(k_loc)
        fill_indices.append(fill_loc)

        if 0 <= k_loc < len(x_tokens):
            anchor1_text = tokenizer.convert_tokens_to_string([x_tokens[k_loc]])
        else:
            anchor1_text = ""
        anchor1_texts.append(anchor1_text)

        if 0 <= fill_loc < len(x_tokens_sub):
            anchor2_text = tokenizer.convert_tokens_to_string([x_tokens_sub[fill_loc]])
        else:
            anchor2_text = ""
        anchor2_texts.append(anchor2_text)
    
    sv_vals = compute_sv_batch(
        orig_texts, sub_texts, k_indices, tokenizer, model, MAX_LEN, sent_batch
    )

    out_df = df.copy()
    out_df["chosen_substitution"] = chosen_substitutions
    out_df[SV_COLUMN] = sv_vals
    out_df["anchor1"] = k_indices
    out_df["anchor2"] = fill_indices
    out_df["anchor1_text"] = anchor1_texts
    out_df["anchor2_text"] = anchor2_texts

    if out_path is None:
        out_dir = get_output_dir(model_name, ym)
        out_path = os.path.join(
            out_dir,
            f"sv_summary_{JUDGE_NAME}_full_{model_name}_{ym}.csv.gz"
        )

    compression = "gzip" if out_path.endswith(".gz") else None
    out_df.to_csv(out_path, index=False, compression=compression)
    print(f"[{model_name}] saved: {out_path} (rows={len(out_df)})")


# ---------------- Main ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert_pred_dir", default=DEFAULT_BERT_PRED_DIR)
    ap.add_argument("--w2v_pred_dir", default=DEFAULT_W2V_PRED_DIR)
    ap.add_argument("--cond_pred_dir", default=DEFAULT_COND_PRED_DIR)
    ap.add_argument("--judge_ckpt", default=DEFAULT_JUDGE_CHECKPOINT)
    ap.add_argument("--judge-name", default=None)
    ap.add_argument("--output-root", default=OUTPUT_ROOT)
    ap.add_argument("--years", nargs="*", default=[])
    ap.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    ap.add_argument("--anchor-search-radius", type=int, default=ANCHOR_SEARCH_RADIUS)
    ap.add_argument("--sent-batch", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--task-rank",
        type=int,
        default=0,
        help="Worker rank for sharding full evaluation tasks across multiple processes.",
    )
    ap.add_argument(
        "--task-world-size",
        type=int,
        default=1,
        help="Number of workers used to shard full evaluation tasks.",
    )
    ap.add_argument(
        "--debug-file",
        default=None,
        help="Run only a few rows from one scoring.py output file and save to ./ for anchor checks.",
    )
    ap.add_argument(
        "--debug-model",
        choices=["bert", "w2v", "conditional"],
        default=None,
        help="Model name for --debug-file. Inferred from filename when omitted.",
    )
    ap.add_argument("--debug-n", type=int, default=10)
    ap.add_argument("--debug-output", default=None)
    return ap.parse_args()


def infer_judge_name(judge_ckpt: str) -> str:
    ckpt = judge_ckpt.lower()
    if "qwen2.5" in ckpt or "qwen" in ckpt:
        return "qwen25"
    if "gemma" in ckpt:
        return "gemma"
    return re.sub(r"[^a-z0-9]+", "_", judge_ckpt.lower()).strip("_")


def validate_task_shard_args(task_rank: int, task_world_size: int):
    if task_world_size < 1:
        raise ValueError("--task-world-size must be >= 1")
    if task_rank < 0 or task_rank >= task_world_size:
        raise ValueError("--task-rank must satisfy 0 <= rank < task_world_size")


def build_full_eval_tasks(
    model_to_ym2path: Dict[str, Dict[str, str]],
    common_months: List[str],
) -> List[Tuple[str, str, str, str, str]]:
    tasks = []
    model_names = list(model_to_ym2path.keys())
    for month_idx, ym in enumerate(common_months):
        shift = month_idx % len(model_names)
        rotated_model_names = model_names[shift:] + model_names[:shift]
        for model_name in rotated_model_names:
            ym2path = model_to_ym2path[model_name]
            pred_csv_path = ym2path[ym]
            scoring_path = get_scoring_output_path(pred_csv_path, model_name, ym)
            out_dir = get_output_dir(model_name, ym)
            out_path = os.path.join(
                out_dir,
                f"sv_summary_{JUDGE_NAME}_full_{model_name}_{ym}.csv.gz",
            )
            tasks.append((model_name, ym, pred_csv_path, scoring_path, out_path))
    return tasks


def main():
    args = parse_args()
    validate_task_shard_args(args.task_rank, args.task_world_size)

    global OUTPUT_ROOT, JUDGE_NAME, SV_COLUMN
    OUTPUT_ROOT = args.output_root
    JUDGE_NAME = args.judge_name or infer_judge_name(args.judge_ckpt)
    SV_COLUMN = f"sv_{JUDGE_NAME}"

    print(f"Device: {DEVICE}")
    print(f"Judge name: {JUDGE_NAME}")
    print(f"Task shard: rank={args.task_rank}, world_size={args.task_world_size}")
    if torch.cuda.is_available():
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<all>")
        print(f"CUDA_VISIBLE_DEVICES: {visible_devices}")
        print(f"CUDA device count visible to this process: {torch.cuda.device_count()}")
    print(f"Loading judge model: {args.judge_ckpt}")

    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN", None)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.judge_ckpt,
        token=hf_token,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModel.from_pretrained(
        args.judge_ckpt,
        torch_dtype="auto",
        token=hf_token,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(DEVICE).eval()

    if args.debug_file is not None:
        inferred_model, inferred_ym = infer_model_and_ym_from_scoring_path(args.debug_file)
        debug_model = args.debug_model or inferred_model
        debug_ym = inferred_ym or "debug"
        if debug_model is None:
            raise ValueError("Could not infer model name from --debug-file. Pass --debug-model.")

        df_debug = read_scoring_rows(args.debug_file).head(args.debug_n).reset_index(drop=True)
        if len(df_debug) == 0:
            print(f"[DEBUG] Empty file, skip: {args.debug_file}")
            return

        out_path = args.debug_output or os.path.abspath(
            f"debug_anchor_{JUDGE_NAME}_{debug_model}_{debug_ym}.csv"
        )
        print(f"[DEBUG] model={debug_model}, ym={debug_ym}, rows={len(df_debug)}, output={out_path}")
        compute_sv_for_df(
            model_name=debug_model,
            df=df_debug,
            ym=debug_ym,
            tokenizer=tokenizer,
            model=model,
            window_size=args.window_size,
            sent_batch=args.sent_batch,
            anchor_search_radius=args.anchor_search_radius,
            out_path=out_path,
        )
        return

    model_to_ym2path = {}

    if os.path.isdir(args.bert_pred_dir):
        model_to_ym2path["bert"] = list_monthly_prediction_files(args.bert_pred_dir, args.years)
    if os.path.isdir(args.w2v_pred_dir):
        model_to_ym2path["w2v"] = list_monthly_prediction_files(args.w2v_pred_dir, args.years)
    if os.path.isdir(args.cond_pred_dir):
        model_to_ym2path["conditional"] = list_monthly_prediction_files(args.cond_pred_dir, args.years)

    if not model_to_ym2path:
        print("No prediction directories found. Check paths.")
        return

    common_months: Optional[Set[str]] = None
    for ym2path in model_to_ym2path.values():
        months = set(ym2path.keys())
        if common_months is None:
            common_months = months
        else:
            common_months &= months

    if not common_months:
        print("No common months across selected models.")
        return

    common_months = sorted(common_months)
    print(f"Common months across models: {common_months}")


    all_tasks = build_full_eval_tasks(model_to_ym2path, common_months)
    tasks = [
        task for task_idx, task in enumerate(all_tasks)
        if task_idx % args.task_world_size == args.task_rank
    ]
    print(
        f"Worker {args.task_rank}/{args.task_world_size}: "
        f"{len(tasks)} of {len(all_tasks)} tasks assigned."
    )

    desc = f"[worker {args.task_rank}] SV({JUDGE_NAME}, scoring rows)"
    for model_name, ym, _pred_csv_path, scoring_path, out_path in tqdm(tasks, desc=desc, unit="task"):
        if not os.path.exists(scoring_path):
            print(f"[{model_name}] SKIP {ym}: scoring output not found: {scoring_path}")
            continue

        if args.resume and os.path.exists(out_path):
            print(f"[SKIP][resume] already exists: {out_path}")
            continue

        df_scoring_rows = read_scoring_rows(scoring_path)

        if len(df_scoring_rows) == 0:
            print(f"[{model_name}] Empty scoring output, skip: {scoring_path}")
            continue

        print(f"[{model_name}][{ym}] Using rows from scoring output: {len(df_scoring_rows)}")

        compute_sv_for_df(
            model_name=model_name,
            df=df_scoring_rows,
            ym=ym,
            tokenizer=tokenizer,
            model=model,
            window_size=args.window_size,
            sent_batch=args.sent_batch,
            anchor_search_radius=args.anchor_search_radius,
            out_path=out_path,
        )


if __name__ == "__main__":
    main()
    
"""
python3 valid_other_models.py \
  --bert_pred_dir /home/jovyan/LEM_data2/data/bert_pred_new/pred \
  --w2v_pred_dir /home/jovyan/LEM_data2/data/skill2vec_pred_new/pred \
  --cond_pred_dir /home/jovyan/LEM_data2/data/condprob_pred_new/pred \
  --judge_ckpt google/gemma-2-9b-it \
  --anchor-search-radius 2 \
  --resume

# anchor debug example:
python3 valid_other_models.py \
  --debug-file /path/to/sv_summary_llama_full_bert_2016-01.csv.gz \
  --debug-n 10
"""
