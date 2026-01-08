# -*- coding: utf-8 -*-
"""
SV (Validation Score) — LLaMA-based contextual validator (FULL version + DEBUG option)

Default mode:
    - From each model’s predictions_YYYY-MM.csv[.gz], use only samples where
      (masked_sentence, truth) are common across ALL models
    - Limit the number of samples per month to cap (--cap), based on shared keys

Debug mode (--debug-file):
    - For a single specified predictions file,
      compute SV on sample_n rows (default: 100) sampled with random_state=42

Input directories (per model):
    * BERT_BASE_DIR = bert_pred_new/pred
    * W2V_BASE_DIR  = skill2vec_pred_new/pred
    * COND_BASE_DIR = condprob_pred_new/pred

Each predictions file must contain:
    truth, pred_top1, pred_top5, masked_sentence

Outputs:
    Default mode:
        base_dir/YYYY/sv_summary_llama_full_{model_name}_{YYYY-MM}.csv.gz
    Debug mode:
        sv_summary_llama_debug_{model_name}.csv (saved in the same folder)
"""

import os
import re
import glob
import argparse
import random
from typing import List, Tuple, Optional, Dict, Set

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== CONFIG =====
DEFAULT_BERT_PRED_DIR         = "/home/jovyan/LEM_data2/hyunjincho/bert_pred_new/pred"
DEFAULT_W2V_PRED_DIR          = "/home/jovyan/LEM_data2/hyunjincho/skill2vec_pred_new/pred"
DEFAULT_COND_PRED_DIR         = "/home/jovyan/LEM_data2/hyunjincho/condprob_pred_new/pred"

DEFAULT_LLAMA_CHECKPOINT = "meta-llama/Llama-3.2-3B"

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN       = 512
WINDOW_SIZE   = 256

torch.set_float32_matmul_precision("high")


# ---------------- List prediction files ----------------
def list_monthly_prediction_files(base_dir: str, years: List[str]) -> Dict[str, str]:
    """
    Collect prediction files under base_dir/{YEAR}/predictions_*.csv[.gz].

    Returns:
        dict mapping 'YYYY-MM' -> full file path
    """
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
            ym = m.group(1)  # YYYY-MM
            ym2path[ym] = fp
    return ym2path


# ---------------- Offset-based anchor helper ----------------
def _anchor_by_offset(sent: str, char_start: int, tokenizer):
    enc = tokenizer(
        sent,
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    input_ids = enc["input_ids"]
    offsets   = enc["offset_mapping"]

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


# ---------------- Build sentences (offset-based anchor) ----------------
def build_original_sentence(row, tokenizer, window_size):
    truth  = str(row["truth"])
    masked = str(row["masked_sentence"])

    prefix, sep, suffix = masked.partition("[MASK]")
    if sep == "":
        sent   = masked
        tokens = tokenizer.tokenize(sent)
        return tokenizer.convert_tokens_to_string(tokens), tokens, -1

    sent = prefix + truth + suffix
    char_start = len(prefix)

    tokens, k = _anchor_by_offset(sent, char_start, tokenizer)
    if k < 0:
        return tokenizer.convert_tokens_to_string(tokens), tokens, -1

    truth_token_len = len(tokenizer.tokenize(truth))
    start = max(0, k - window_size)
    end   = min(len(tokens), k + window_size + truth_token_len)

    win = tokens[start:end]
    k_local = k - start

    return tokenizer.convert_tokens_to_string(win), win, k_local


def build_sentence_with_fill(row, fill_skill, tokenizer, window_size):
    fill_skill = str(fill_skill)
    masked     = str(row["masked_sentence"])

    prefix, sep, suffix = masked.partition("[MASK]")
    if sep == "":
        sent   = masked
        tokens = tokenizer.tokenize(sent)
        return tokenizer.convert_tokens_to_string(tokens), tokens, -1

    sent = prefix + fill_skill + suffix
    char_start = len(prefix)

    tokens, idx = _anchor_by_offset(sent, char_start, tokenizer)
    if idx < 0:
        return tokenizer.convert_tokens_to_string(tokens), tokens, -1

    fill_token_len = len(tokenizer.tokenize(fill_skill))
    start = max(0, idx - window_size)
    end   = min(len(tokens), idx + window_size + fill_token_len)

    win = tokens[start:end]
    fill_start = idx - start

    return tokenizer.convert_tokens_to_string(win), win, fill_start


# ---------------- LLaMA SV computation ----------------
@torch.no_grad()
def compute_sv_batch(texts_orig, texts_sub, k_indices, tokenizer, model, max_len, sent_batch):

    out = []
    use_amp = (DEVICE == "cuda")

    for s in range(0, len(texts_orig), sent_batch):
        o_batch = texts_orig[s:s+sent_batch]
        p_batch = texts_sub[s:s+sent_batch]
        k_batch = k_indices[s:s+sent_batch]

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
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
            T   = min(T_o, T_p)

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

    torch.cuda.empty_cache()
    return out


# ---------------- Compute SV for a shared DF (core logic extracted) ----------------
def compute_sv_for_df(
    model_name: str,
    df: pd.DataFrame,
    out_base_path: str,
    tokenizer,
    model,
    window_size: int,
    sent_batch: int,
    debug: bool = False,
):
    year  = os.path.basename(os.path.dirname(out_base_path))
    fname = os.path.basename(out_base_path)

    needed = {"truth", "pred_top1", "pred_top5", "masked_sentence"}
    if not needed.issubset(df.columns):
        print(f"[{model_name}] SKIP {year}/{fname}: missing columns.")
        return

    df["truth"]           = df["truth"].astype(str)
    df["pred_top1"]       = df["pred_top1"].astype(str)
    df["pred_top5"]       = df["pred_top5"].astype(str)
    df["masked_sentence"] = df["masked_sentence"].astype(str)

    orig_texts, sub_texts = [], []
    k_indices, fill_indices = [], []
    anchor1_texts, anchor2_texts = [], []

    for _, row in df.iterrows():
        truth = row["truth"]
        top1  = row["pred_top1"]
        top5_list = row["pred_top5"].split("|") if row["pred_top5"] else []

        if model_name == "bert":
            if truth == top1 and len(top5_list) > 1:
                chosen = top5_list[1]
            else:
                chosen = top1
        else:
            chosen = top1

        x_text, x_tokens, k_loc = build_original_sentence(row, tokenizer, window_size)
        x_sub,  x_tokens_sub, fill_loc = build_sentence_with_fill(row, chosen, tokenizer, window_size)

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
    out_df["sv_llama"]     = sv_vals
    out_df["anchor1"]      = k_indices
    out_df["anchor2"]      = fill_indices
    out_df["anchor1_text"] = anchor1_texts
    out_df["anchor2_text"] = anchor2_texts

    if debug:
        out_path = os.path.join(
            os.path.dirname(out_base_path),
            f"sv_summary_llama_debug_{model_name}.csv"
        )
    else:
        ym = re.findall(r"(\d{4}-\d{2})", os.path.basename(out_base_path))
        postfix = ym[0] if ym else "month"
        out_path = os.path.join(
            os.path.dirname(out_base_path),
            f"sv_summary_llama_full_{model_name}_{postfix}.csv.gz"
        )

    out_df.to_csv(out_path, index=False,
                  compression=("gzip" if not debug else None))
    print(f"[{model_name}] saved: {out_path} (rows={len(out_df)})")


# ---------------- SV for a single CSV (debug/standalone helper) ----------------
def evaluate_one_month_csv(
    model_name: str,
    csv_path: str,
    tokenizer,
    model,
    window_size: int,
    sent_batch: int,
    sample_n: Optional[int] = None,
    debug: bool = False,
):
    df = pd.read_csv(csv_path)

    if sample_n is not None and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
        print(f"[{model_name}] DEBUG: using {len(df)} rows sampled from {csv_path}")

    compute_sv_for_df(
        model_name=model_name,
        df=df,
        out_base_path=csv_path,
        tokenizer=tokenizer,
        model=model,
        window_size=window_size,
        sent_batch=sent_batch,
        debug=debug,
    )


# ---------------- Main ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-from", type=str, default="all",
                    help="all 또는 콤마로 구분된 모델 리스트 (bert,bert_freezed,w2v,conditional)")
    ap.add_argument("--bert_pred_dir",          default=DEFAULT_BERT_PRED_DIR)
    ap.add_argument("--w2v_pred_dir",           default=DEFAULT_W2V_PRED_DIR)
    ap.add_argument("--cond_pred_dir",          default=DEFAULT_COND_PRED_DIR)
    ap.add_argument("--llama_ckpt",             default=DEFAULT_LLAMA_CHECKPOINT)
    ap.add_argument("--years", nargs="*", default=[],
                    help="특정 연도만 처리하고 싶으면 예: --years 2015 2016 (미지정 시 전체)")
    ap.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    ap.add_argument("--sent-batch",  type=int, default=4)

    # 🔧 cap: maximum number of shared samples per month
    ap.add_argument(
        "--cap",
        type=int,
        default=8_000,
        help="Max number of shared samples per month (based on masked_sentence, truth pairs)."
    )

    # 🔧 debug options (single-file mode)
    ap.add_argument(
        "--debug-file",
        type=str,
        default=None,
        help="Run in debug mode on a single predictions CSV (use only sample_n rows from that file)."
    )
    ap.add_argument(
        "--debug-n",
        type=int,
        default=100,
        help="Number of samples to use in --debug-file mode (default: 100)."
    )

    return ap.parse_args()


def main():
    args = parse_args()

    print(f"Device: {DEVICE}")
    print(f"Loading LLaMA from HF Hub: {args.llama_ckpt}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.llama_ckpt,
        use_auth_token=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.llama_ckpt,
        torch_dtype="auto",
        device_map="auto",
        use_auth_token=True
    ).eval()

    try:
        model.set_attn_implementation("eager")
        print("Set attention implementation to 'eager' for LLaMA.")
    except Exception:
        print("WARNING: eager attention not available.")

    # ── Debug mode: a single file + N samples ─────────────────────
    if args.debug_file is not None:
        model_name = "bert"
        print(f"[DEBUG] Running on single file: {args.debug_file} (n={args.debug_n})")
        evaluate_one_month_csv(
            model_name=model_name,
            csv_path=args.debug_file,
            tokenizer=tokenizer,
            model=model,
            window_size=args.window_size,
            sent_batch=args.sent_batch,
            sample_n=args.debug_n,
            debug=True,
        )
        return

    # ── Default mode: use samples shared across ALL models by (masked_sentence, truth) ────────
    if args.start_from == "all":
        wanted = {"bert", "w2v", "conditional"}
    else:
        wanted = set(x.strip().lower() for x in args.start_from.split(","))

    # Build mapping: model -> { YYYY-MM -> file_path }
    model_to_ym2path = {}

    if "bert" in wanted and os.path.isdir(args.bert_pred_dir):
        model_to_ym2path["bert"] = list_monthly_prediction_files(args.bert_pred_dir, args.years)
    if "w2v" in wanted and os.path.isdir(args.w2v_pred_dir):
        model_to_ym2path["w2v"] = list_monthly_prediction_files(args.w2v_pred_dir, args.years)
    if "conditional" in wanted and os.path.isdir(args.cond_pred_dir):
        model_to_ym2path["conditional"] = list_monthly_prediction_files(args.cond_pred_dir, args.years)

    if not model_to_ym2path:
        print("No prediction directories found. Check paths.")
        return

    # Use only months (YYYY-MM) that exist for all selected models
    common_months: Optional[Set[str]] = None
    for ym2path in model_to_ym2path.values():
        months = set(ym2path.keys())
        if common_months is None:
            common_months = months
        else:
            common_months = common_months & months

    if not common_months:
        print("No common months across selected models.")
        return

    common_months = sorted(common_months)
    print(f"Common months across models: {common_months}")

    random.seed(42)

    for ym in tqdm(common_months, desc="[ALL MODELS] SV(LLaMA, full by month)", unit="month"):
        key_sets: Dict[str, Set[str]] = {}
        for model_name, ym2path in model_to_ym2path.items():
            csv_path = ym2path.get(ym)
            if csv_path is None:
                break

            try:
                df_small = pd.read_csv(csv_path, usecols=["masked_sentence", "truth"])
            except Exception as e:
                print(f"[{model_name}] Failed to read for month {ym}: {e}")
                break

            df_small["masked_sentence"] = df_small["masked_sentence"].astype(str)
            df_small["truth"] = df_small["truth"].astype(str)
            df_small["__key__"] = df_small["masked_sentence"] + "||" + df_small["truth"]
            key_sets[model_name] = set(df_small["__key__"].unique())

        if len(key_sets) != len(model_to_ym2path):
            print(f"[WARN] Some model missing or failed for {ym}, skip this month.")
            continue

        shared_keys = None
        for s in key_sets.values():
            if shared_keys is None:
                shared_keys = set(s)
            else:
                shared_keys &= s

        if not shared_keys:
            print(f"[WARN] No shared (masked_sentence, truth) keys in {ym}, skip.")
            continue

        shared_keys = list(shared_keys)
        if args.cap is not None and len(shared_keys) > args.cap:
            shared_keys = random.sample(shared_keys, args.cap)
        shared_keys_set = set(shared_keys)

        print(f"[{ym}] Shared keys: {len(shared_keys_set)} (after cap={args.cap})")

        for model_name, ym2path in model_to_ym2path.items():
            csv_path = ym2path[ym]
            df_full = pd.read_csv(csv_path)

            df_full["masked_sentence"] = df_full["masked_sentence"].astype(str)
            df_full["truth"] = df_full["truth"].astype(str)
            df_full["__key__"] = df_full["masked_sentence"] + "||" + df_full["truth"]

            df_sub = df_full[df_full["__key__"].isin(shared_keys_set)].drop(columns="__key__").reset_index(drop=True)

            if len(df_sub) == 0:
                print(f"[{model_name}] No rows remain after key filter in {ym}, skip.")
                continue

            compute_sv_for_df(
                model_name=model_name,
                df=df_sub,
                out_base_path=csv_path,
                tokenizer=tokenizer,
                model=model,
                window_size=args.window_size,
                sent_batch=args.sent_batch,
                debug=False,
            )


if __name__ == "__main__":
    main()