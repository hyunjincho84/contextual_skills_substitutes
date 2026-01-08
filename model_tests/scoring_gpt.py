#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute LLaMA-based SV(sv_llama) for GPT predictions (BATCH over many per-year GPT output files).

Inputs (default):
  /home/jovyan/LEM_data2/hyunjincho/gpt_samples/20*/*with_gpt_pred*.csv.*

Each input CSV must contain:
  - truth
  - masked_sentence
  - row_idx          (MUST already exist; we DO NOT regenerate it)
  - pred_top1 (optional if pred_top5 exists)
  - pred_top5 (optional)

Output (per input file, next to it):
  - <input_stem>_with_sv_llama.csv.gz
    (always gzip, even if input is .csv)

Adds columns:
  - sv_llama, anchor1, anchor2, anchor1_text, anchor2_text
  - pred_used  (the actual prediction used for fill: top1 or top2)

Behavior:
  - If pred_top1 == truth, use pred_top5's 2nd item as pred_used (top2) when available.
  - Else use pred_top1.

Crash-safe:
  - writes incrementally per chunk to the output file (append mode)

Resume:
  - --resume will skip rows whose row_idx <= max(row_idx) already written in output.
"""

import os
import re
import glob
import argparse
from typing import List, Optional

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")

# ---------- DEFAULTS ----------
DEFAULT_IN_PATTERN = "/home/jovyan/LEM_data2/hyunjincho/gpt_samples/20*/*with_gpt_pred.csv*"
DEFAULT_LLAMA_CKPT = "meta-llama/Llama-3.2-3B"


# ---------------- offset 기반 anchor 헬퍼 ----------------
def _anchor_by_offset(sent: str, char_start: int, tokenizer):
    enc = tokenizer(sent, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    # tokenizer가 batch 형태로 주는 경우 방어
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


# ---------------- Build sentences (offset 기반 anchor) ----------------
def build_original_sentence(truth: str, masked_sentence: str, tokenizer, window_size: int):
    truth = str(truth)
    masked = str(masked_sentence)

    prefix, sep, suffix = masked.partition("[MASK]")
    if sep == "":
        tokens = tokenizer.tokenize(masked)
        return tokenizer.convert_tokens_to_string(tokens), tokens, -1

    sent = prefix + truth + suffix
    char_start = len(prefix)

    tokens, k = _anchor_by_offset(sent, char_start, tokenizer)
    if k < 0:
        return tokenizer.convert_tokens_to_string(tokens), tokens, -1

    truth_token_len = len(tokenizer.tokenize(truth))
    start = max(0, k - window_size)
    end = min(len(tokens), k + window_size + truth_token_len)

    win = tokens[start:end]
    k_local = k - start
    return tokenizer.convert_tokens_to_string(win), win, k_local


def build_sentence_with_fill(fill_skill: str, masked_sentence: str, tokenizer, window_size: int):
    fill_skill = str(fill_skill)
    masked = str(masked_sentence)

    prefix, sep, suffix = masked.partition("[MASK]")
    if sep == "":
        tokens = tokenizer.tokenize(masked)
        return tokenizer.convert_tokens_to_string(tokens), tokens, -1

    sent = prefix + fill_skill + suffix
    char_start = len(prefix)

    tokens, idx = _anchor_by_offset(sent, char_start, tokenizer)
    if idx < 0:
        return tokenizer.convert_tokens_to_string(tokens), tokens, -1

    fill_token_len = len(tokenizer.tokenize(fill_skill))
    start = max(0, idx - window_size)
    end = min(len(tokens), idx + window_size + fill_token_len)

    win = tokens[start:end]
    fill_start = idx - start
    return tokenizer.convert_tokens_to_string(win), win, fill_start


# ---------------- LLaMA SV computation ----------------
@torch.no_grad()
def compute_sv_batch(
    texts_orig: List[str],
    texts_sub: List[str],
    k_indices: List[int],
    tokenizer,
    model,
    max_len: int,
    sent_batch: int,
):
    out = []
    use_amp = (DEVICE == "cuda")

    for s in range(0, len(texts_orig), sent_batch):
        o_batch = texts_orig[s : s + sent_batch]
        p_batch = texts_sub[s : s + sent_batch]
        k_batch = k_indices[s : s + sent_batch]

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            enc_o = tokenizer(
                o_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            ).to(DEVICE)
            outputs_o = model(
                **enc_o,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=False,
            )

            enc_p = tokenizer(
                p_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            ).to(DEVICE)
            outputs_p = model(
                **enc_p,
                output_attentions=False,
                output_hidden_states=True,
                use_cache=False,
            )

        hs_o = outputs_o.hidden_states
        hs_p = outputs_p.hidden_states

        top4_o = torch.cat(hs_o[-4:], dim=-1)
        top4_p = torch.cat(hs_p[-4:], dim=-1)

        att_stack = torch.stack(outputs_o.attentions, dim=0)  # [L,B,H,T,T]
        mean_att = att_stack.mean(dim=(0, 2))  # [B,T,T]

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

            denom = (H_o.norm(dim=-1) * H_p.norm(dim=-1)).clamp_min(1e-12)
            cos = (H_o * H_p).sum(dim=-1) / denom

            k_use = min(k_loc, T - 1)

            w = mean_att[b, :T, k_use].clamp_min(0.0)
            if w.sum() > 0:
                w = w / w.sum()
            else:
                w = torch.ones_like(w) / T

            sv = (w * cos).sum().item()
            out.append(sv)

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


def ensure_pred_top1(df: pd.DataFrame) -> pd.DataFrame:
    if "pred_top1" in df.columns:
        df["pred_top1"] = df["pred_top1"].fillna("").astype(str)
        return df

    if "pred_top5" in df.columns:
        def first_from_top5(x):
            x = "" if pd.isna(x) else str(x)
            parts = [p.strip() for p in x.split("|") if p.strip()]
            return parts[0] if parts else ""
        df["pred_top1"] = df["pred_top5"].apply(first_from_top5).astype(str)
        return df

    raise ValueError("Input must contain either pred_top1 or pred_top5.")


def choose_pred_used(row: pd.Series) -> str:
    """
    pred_used selection:
      - 기본: pred_top1
      - if pred_top1 == truth and pred_top5 has >=2: use 2nd item (top2)
    """
    truth = str(row.get("truth", ""))
    pred1 = str(row.get("pred_top1", "")).strip()

    pred_used = pred1

    # truth == top1 -> top2
    if pred_used != "" and pred_used == truth:
        if "pred_top5" in row and pd.notna(row["pred_top5"]):
            parts = [p.strip() for p in str(row["pred_top5"]).split("|") if p.strip()]
            if len(parts) >= 2:
                pred_used = parts[1].strip()

    return pred_used


def compute_sv_for_chunk(
    df: pd.DataFrame,
    tokenizer,
    model,
    window_size: int,
    max_len: int,
    sent_batch: int,
):
    df = df.copy()

    # required
    for col in ["truth", "masked_sentence", "row_idx"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["truth"] = df["truth"].astype(str)
    df["masked_sentence"] = df["masked_sentence"].astype(str)

    df = ensure_pred_top1(df)

    # pred_used column
    df["pred_used"] = df.apply(choose_pred_used, axis=1).astype(str)

    orig_texts, sub_texts = [], []
    k_indices, fill_indices = [], []
    anchor1_texts, anchor2_texts = [], []

    for _, row in df.iterrows():
        truth = row["truth"]
        masked = row["masked_sentence"]
        pred_used = (row["pred_used"] or "").strip()

        x_text, x_tokens, k_loc = build_original_sentence(truth, masked, tokenizer, window_size)

        # pred_used 비면 NaN 처리되게
        if pred_used == "":
            x_sub, x_tokens_sub, fill_loc = x_text, x_tokens, -1
            k_loc_use = -1
        else:
            x_sub, x_tokens_sub, fill_loc = build_sentence_with_fill(pred_used, masked, tokenizer, window_size)
            k_loc_use = k_loc

        orig_texts.append(x_text)
        sub_texts.append(x_sub)
        k_indices.append(k_loc_use)
        fill_indices.append(fill_loc)

        if 0 <= k_loc < len(x_tokens):
            anchor1_texts.append(tokenizer.convert_tokens_to_string([x_tokens[k_loc]]))
        else:
            anchor1_texts.append("")

        if 0 <= fill_loc < len(x_tokens_sub):
            anchor2_texts.append(tokenizer.convert_tokens_to_string([x_tokens_sub[fill_loc]]))
        else:
            anchor2_texts.append("")

    sv_vals = compute_sv_batch(orig_texts, sub_texts, k_indices, tokenizer, model, max_len, sent_batch)

    # add outputs (keep existing row_idx / pred_top5 그대로 유지)
    df["sv_llama"] = sv_vals
    df["anchor1"] = k_indices
    df["anchor2"] = fill_indices
    df["anchor1_text"] = anchor1_texts
    df["anchor2_text"] = anchor2_texts
    return df


def infer_year_from_path(path: str) -> Optional[str]:
    m = re.search(r"(20\d{2})", path)
    return m.group(1) if m else None


def infer_out_path_next_to_input(in_path: str) -> str:
    base = os.path.basename(in_path)
    if base.endswith(".csv.gz"):
        stem = base[:-len(".csv.gz")]
    elif base.endswith(".csv"):
        stem = base[:-len(".csv")]
    else:
        stem = os.path.splitext(base)[0]
    out_name = f"{stem}_with_sv_llama.csv.gz"
    return os.path.join(os.path.dirname(in_path), out_name)


def append_chunk_gzip(out_path: str, df_chunk: pd.DataFrame, write_header: bool):
    mode = "wt" if write_header else "at"
    df_chunk.to_csv(
        out_path,
        index=False,
        compression="gzip",
        mode=mode,
        header=write_header,
    )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-pattern", type=str, default=DEFAULT_IN_PATTERN,
                    help="Glob for GPT output files (with_gpt_pred).")
    ap.add_argument("--years", nargs="*", default=None,
                    help="Optional filter: only these years (e.g., 2014 2015).")
    ap.add_argument("--llama-ckpt", type=str, default=DEFAULT_LLAMA_CKPT)
    ap.add_argument("--window-size", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--sent-batch", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=50_000)
    ap.add_argument("--use-auth-token", action="store_true", help="HF gated model이면 필요")
    ap.add_argument("--resume", action="store_true",
                    help="Resume append (skip already-written rows based on existing output max row_idx).")
    return ap.parse_args()


def main():
    args = parse_args()

    files = sorted(glob.glob(args.in_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.in_pattern}")

    # years filter
    if args.years:
        want = set(map(str, args.years))
        files2 = []
        for fp in files:
            y = infer_year_from_path(fp)
            if y in want:
                files2.append(fp)
        files = files2

    if not files:
        raise FileNotFoundError("After --years filtering, no files remain.")

    print(f"[INFO] Matched {len(files)} files.")
    for fp in files[:5]:
        print(f"  - {fp}")
    if len(files) > 5:
        print("  ...")

    print(f"Device: {DEVICE}")
    print(f"Loading LLaMA: {args.llama_ckpt}")

    tok = AutoTokenizer.from_pretrained(args.llama_ckpt, use_auth_token=args.use_auth_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.llama_ckpt,
        torch_dtype="auto",
        device_map="auto",
        use_auth_token=args.use_auth_token,
    ).eval()

    try:
        model.set_attn_implementation("eager")
        print("Set attention implementation to 'eager'.")
    except Exception:
        print("WARNING: eager attention not available.")

    grand_in_rows = 0
    grand_out_rows = 0

    for in_path in files:
        out_path = infer_out_path_next_to_input(in_path)
        print(f"\n[FILE] {in_path}")
        print(f"  -> OUT: {out_path}")

        header = pd.read_csv(in_path, nrows=0)
        cols = set(header.columns)

        required = {"truth", "masked_sentence", "row_idx"}
        if not required.issubset(cols):
            print(f"  [SKIP] Missing required: {required - cols}")
            continue
        if ("pred_top1" not in cols) and ("pred_top5" not in cols):
            print("  [SKIP] Missing pred_top1/pred_top5")
            continue

        # resume support: skip already written row_idx
        start_row_idx_exclusive = None
        if args.resume and os.path.exists(out_path):
            try:
                existing = pd.read_csv(out_path, usecols=["row_idx"])
                if len(existing) > 0:
                    start_row_idx_exclusive = int(existing["row_idx"].max())
                print(f"  [RESUME] max_written_row_idx={start_row_idx_exclusive}")
            except Exception as e:
                print(f"  [RESUME WARN] failed to read existing out, will overwrite. err={e}")
                start_row_idx_exclusive = None

        # overwrite if not resume (or resume failed)
        if (not args.resume) or (start_row_idx_exclusive is None):
            if os.path.exists(out_path):
                os.remove(out_path)

        chunk_iter = pd.read_csv(in_path, chunksize=args.chunk_size)

        wrote_any = os.path.exists(out_path)
        in_rows_this = 0
        out_rows_this = 0

        for chunk in tqdm(chunk_iter, desc="Scoring chunks", unit="chunk"):
            chunk = chunk.reset_index(drop=True)

            # IMPORTANT: do NOT regenerate row_idx
            if "row_idx" not in chunk.columns:
                raise ValueError(f"[FATAL] row_idx missing in chunk for {in_path}")

            # resume filter: keep only row_idx > max_written
            if args.resume and (start_row_idx_exclusive is not None):
                chunk = chunk[pd.to_numeric(chunk["row_idx"], errors="coerce") > start_row_idx_exclusive].copy()
                if chunk.empty:
                    continue

            in_rows_this += len(chunk)

            scored = compute_sv_for_chunk(
                chunk,
                tokenizer=tok,
                model=model,
                window_size=args.window_size,
                max_len=args.max_len,
                sent_batch=args.sent_batch,
            )

            append_chunk_gzip(out_path, scored, write_header=(not wrote_any))
            wrote_any = True
            out_rows_this += len(scored)

        print(f"  [DONE] in_rows={in_rows_this:,} | out_rows={out_rows_this:,}")
        grand_in_rows += in_rows_this
        grand_out_rows += out_rows_this

    print("\n===== ALL DONE =====")
    print(f"Total processed input rows : {grand_in_rows:,}")
    print(f"Total written output rows  : {grand_out_rows:,}")


if __name__ == "__main__":
    main()