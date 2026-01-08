#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run BERT skill prediction on the sampled GPT-eval CSVs produced by your sampler script.

Input (per-year gzip CSV):
  {IN_ROOT}/{year}/gpt_unique_samples_{year}_global*.csv.gz
Expected columns:
  - row_idx
  - year
  - file
  - truth
  - masked_sentence

What this script does:
  1) Load the sampled rows (streaming by chunks).
  2) Keep rows that contain [MASK] and whose truth is in-vocab.
  3) Run your fine-tuned BERTForSkillPrediction to get top-K predictions.
  4) Save per-year outputs to:
       {OUT_ROOT}/{year}/bert_preds_{year}_top{K}.csv.gz

Output columns include:
  - year, file, row_idx, truth, masked_sentence
  - pred_top1, pred_topk, pred_topk_probs
"""

import os
import re
import glob
import json
import argparse
from typing import Dict, List, Optional, Iterable, Tuple

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizerFast

# If your model.py is not on PYTHONPATH, add it via --model-py-dir
# from model import BERTForSkillPrediction


# ------------------------- helpers -------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_vocab(vocab_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        skill2idx_raw = json.load(f)
    # ensure int indices
    skill2idx = {str(k): int(v) for k, v in skill2idx_raw.items()}
    idx2skill = {int(v): str(k) for k, v in skill2idx.items()}
    return skill2idx, idx2skill

def list_sample_files(in_root: str, years: Optional[List[str]] = None) -> List[str]:
    # expects: {in_root}/{year}/gpt_unique_samples_{year}_global*.csv.gz
    pattern = os.path.join(in_root, "20*", "gpt_unique_samples_*_global*0.csv.gz")
    files = sorted(glob.glob(pattern))
    if years:
        want = set(str(y) for y in years)
        files = [fp for fp in files if os.path.basename(os.path.dirname(fp)) in want]
    return files

def get_year_from_sample_path(fp: str) -> Optional[str]:
    # parent dir is the year
    year = os.path.basename(os.path.dirname(fp))
    return year if re.fullmatch(r"20\d{2}", year) else None

def write_gz_append(out_path: str, rows: List[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if os.path.exists(out_path):
        df.to_csv(out_path, mode="a", header=False, index=False, compression="gzip")
    else:
        df.to_csv(out_path, mode="w", header=True, index=False, compression="gzip")


# ------------------------- main inference -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", type=str, default="/home/jovyan/LEM_data2/hyunjincho/gpt_samples")
    ap.add_argument("--out-root", type=str, default="/home/jovyan/LEM_data2/hyunjincho/bert_pred_new/gpt_sample")

    ap.add_argument("--years", nargs="*", default=None, help="e.g., --years 2020 2021")

    ap.add_argument("--model-name", type=str, required=True,
                    help="HF checkpoint directory, e.g. /.../bert_pretrained/checkpoint-165687")
    ap.add_argument("--best-model-pt", type=str, required=True,
                    help="Fine-tuned weights, e.g. /.../best_model.pt")
    ap.add_argument("--vocab-path", type=str, required=True,
                    help="skill2idx.json path")

    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--topk", type=int, default=5)

    ap.add_argument("--chunk-size", type=int, default=200_000)
    ap.add_argument("--flush-every", type=int, default=50_000)

    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--model-py-dir", type=str, default=None,
                    help="Directory that contains model.py (optional). If set, it will be added to sys.path.")

    args = ap.parse_args()

    if args.model_py_dir:
        import sys
        sys.path.insert(0, os.path.abspath(args.model_py_dir))

    from model import BERTForSkillPrediction  # now safe to import

    device = torch.device(args.device)

    # Load tokenizer / vocab / model
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    skill2idx, idx2skill = load_vocab(args.vocab_path)
    num_skills = len(skill2idx)

    model = BERTForSkillPrediction(args.model_name, num_skills=num_skills).to(device)
    state = torch.load(args.best_model_pt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    files = list_sample_files(args.in_root, args.years)
    if not files:
        raise FileNotFoundError(f"No sample files found under: {args.in_root}")

    print(f"[INFO] device={device} | files={len(files)} | topk={args.topk}")
    print(f"[INFO] in-root={args.in_root}")
    print(f"[INFO] out-root={args.out_root}")

    req_cols = ["row_idx", "year", "file", "truth", "masked_sentence"]

    torch.backends.cudnn.benchmark = True

    with torch.inference_mode():
        for fp in tqdm(files, desc="Sample files", unit="file"):
            year = get_year_from_sample_path(fp) or "unknown"
            out_dir = os.path.join(args.out_root, year)
            ensure_dir(out_dir)
            out_path = os.path.join(out_dir, f"bert_preds_{year}_top{args.topk}.csv.gz")

            # We'll stream the input in chunks, but do batching in-memory per chunk
            # to avoid complicated DataLoader glue.
            buffer_rows: List[dict] = []
            pending_texts: List[str] = []
            pending_meta: List[dict] = []

            def flush_pending_batch():
                nonlocal buffer_rows, pending_texts, pending_meta
                if not pending_texts:
                    return

                enc = tokenizer(
                    pending_texts,
                    padding=True,
                    truncation=True,
                    max_length=args.max_len,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(device, non_blocking=True)
                attn_mask = enc["attention_mask"].to(device, non_blocking=True)

                mask_id = tokenizer.mask_token_id
                mask_positions = (input_ids == mask_id)
                mask_idx = torch.argmax(mask_positions.int(), dim=1)

                logits = model(input_ids, attn_mask, mask_idx)
                probs = F.softmax(logits, dim=-1)

                topk = torch.topk(probs, k=args.topk, dim=-1)
                topk_idx = topk.indices.detach().cpu().tolist()
                topk_val = topk.values.detach().cpu().tolist()

                for meta, idxs, vals in zip(pending_meta, topk_idx, topk_val):
                    pred_names = [idx2skill[int(i)] for i in idxs]
                    buffer_rows.append({
                        **meta,
                        "pred_top1": pred_names[0],
                        f"pred_top{args.topk}": "|".join(pred_names),
                        f"pred_top{args.topk}_probs": "|".join([f"{float(v):.6f}" for v in vals]),
                    })

                pending_texts = []
                pending_meta = []

            total_in = 0
            total_used = 0

            for chunk in pd.read_csv(fp, usecols=req_cols, chunksize=args.chunk_size, compression="gzip"):
                if chunk is None or len(chunk) == 0:
                    continue
                total_in += len(chunk)

                # basic cleanup
                sub = chunk.dropna(subset=["truth", "masked_sentence"]).copy()
                if len(sub) == 0:
                    continue

                sub["truth"] = sub["truth"].astype(str).str.lower().str.strip()
                sub["masked_sentence"] = sub["masked_sentence"].astype(str)

                # keep only rows with [MASK]
                sub = sub[sub["masked_sentence"].str.contains(r"\[MASK\]", regex=True)]
                if len(sub) == 0:
                    continue

                # keep only in-vocab truths (so "truth" is meaningful in your label space)
                sub = sub[sub["truth"].isin(skill2idx)]
                if len(sub) == 0:
                    continue

                total_used += len(sub)

                # batch them
                for row in sub.itertuples(index=False):
                    meta = {
                        "year": str(row.year),
                        "file": str(row.file),
                        "row_idx": row.row_idx,
                        "truth": str(row.truth),
                        "masked_sentence": str(row.masked_sentence),
                    }
                    pending_meta.append(meta)
                    pending_texts.append(meta["masked_sentence"])

                    if len(pending_texts) >= args.batch_size:
                        flush_pending_batch()

                    if len(buffer_rows) >= args.flush_every:
                        write_gz_append(out_path, buffer_rows)
                        buffer_rows = []

            # flush remaining
            flush_pending_batch()
            if buffer_rows:
                write_gz_append(out_path, buffer_rows)
                buffer_rows = []

            print(f"[OK] {os.path.basename(fp)} -> {out_path}")
            print(f"     rows_in={total_in:,} | rows_used={total_used:,}")

    print("[DONE] All sample files processed.")


if __name__ == "__main__":
    main()