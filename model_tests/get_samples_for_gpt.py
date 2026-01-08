#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a reproducible GPT evaluation sample set from BERT prediction outputs.

What this script does:
1) Scan yearly compressed CSV files under `IN_ROOT` that contain at least:
   - truth
   - masked_sentence
2) Build a GLOBAL-unique pool of (truth, masked_sentence) pairs across ALL files/years.
3) From that global-unique pool, keep a fixed fraction (default 1%) using a stable hash
   (so the sample is reproducible given the same seed).
4) Save the sampled rows into yearly gzip CSV files under `OUT_ROOT/{year}/...`.

Output columns:
- row_idx: row index within the original file (0-based within that file)
- year: extracted from the file path
- file: relative path of the source file
- truth
- masked_sentence
"""

import os
import re
import glob
import argparse
from typing import Optional, List, Dict

import pandas as pd
from tqdm import tqdm

IN_ROOT  = "/home/jovyan/LEM_data2/hyunjincho/bert_pred_new/pred"
OUT_ROOT = "/home/jovyan/LEM_data2/gpt_samples"

CHUNK_SIZE = 200_000
KEY_SEP = "|||SEP|||"
REQ_COLS = ["truth", "masked_sentence"]

# ---------------- utils ----------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def extract_year_from_path(fp: str) -> Optional[str]:
    m = re.search(r"/(20\d{2})/", fp)
    return m.group(1) if m else None

def list_input_files(in_root: str, years: Optional[List[str]] = None) -> List[str]:
    pattern = os.path.join(in_root, "20*", "sv_summary_llama_full_bert_*.csv.gz")
    files = sorted(glob.glob(pattern))
    if years:
        want = set(str(y) for y in years)
        files = [fp for fp in files if (extract_year_from_path(fp) in want)]
    return files

def stable_hash_01(s: str, seed: int) -> float:
    """
    Stable, reproducible hash mapped to [0, 1).
    Do NOT use Python built-in hash() because it can vary across sessions.
    """
    import hashlib
    h = hashlib.blake2b((str(seed) + "||" + s).encode("utf-8"), digest_size=8).digest()
    x = int.from_bytes(h, byteorder="big", signed=False)
    return x / float(2**64)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", type=str, default=IN_ROOT)
    ap.add_argument("--out-root", type=str, default=OUT_ROOT)
    ap.add_argument("--years", nargs="*", default=None, help="e.g., --years 2020 2021")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--sample-frac", type=float, default=0.01, help="Fraction of GLOBAL-unique pairs to keep (default=0.01)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = list_input_files(args.in_root, args.years)
    if not files:
        print(f"[ERROR] No input files found: {args.in_root}/20*/sv_summary_llama_full_bert_*.csv.gz")
        return

    years_found = sorted({extract_year_from_path(fp) for fp in files if extract_year_from_path(fp)})
    print(f"[INFO] Found {len(files)} files across years: {years_found}")
    print(f"[INFO] sample-frac (global unique): {args.sample_frac}  seed={args.seed}")
    print(f"[INFO] Output root: {args.out_root}")

    # Track GLOBAL unique keys across all files/years
    seen_keys = set()

    # Global stats: total unique and total sampled
    global_unique_cnt = 0
    global_sample_cnt = 0

    # Per-year stats
    per_year_unique: Dict[str, int] = {y: 0 for y in years_found}
    per_year_sample: Dict[str, int] = {y: 0 for y in years_found}

    # Accumulate sampled rows by year (kept fraction is small, so feasible)
    per_year_chunks: Dict[str, List[pd.DataFrame]] = {y: [] for y in years_found}

    for fp in tqdm(files, desc="Scanning files", unit="file"):
        year = extract_year_from_path(fp)
        if year is None:
            continue

        # Header check
        try:
            header = pd.read_csv(fp, nrows=0)
        except Exception as e:
            print(f"[WARN] Failed to read header: {fp} | {e}")
            continue
        if not set(REQ_COLS).issubset(set(header.columns)):
            print(f"[WARN] Skip (missing {REQ_COLS}): {fp}")
            continue

        row_base = 0
        try:
            for chunk in pd.read_csv(fp, usecols=REQ_COLS, chunksize=args.chunk_size):
                chunk = chunk.copy()
                chunk["row_idx"] = range(row_base, row_base + len(chunk))
                row_base += len(chunk)

                chunk = chunk.dropna(subset=REQ_COLS)
                if chunk.empty:
                    continue

                chunk["truth"] = chunk["truth"].astype(str)
                chunk["masked_sentence"] = chunk["masked_sentence"].astype(str)

                keys = chunk["truth"] + KEY_SEP + chunk["masked_sentence"]

                # (1) GLOBAL-unique filter
                is_new = ~keys.isin(seen_keys)
                if not is_new.any():
                    continue

                new_chunk = chunk.loc[is_new, ["row_idx", "truth", "masked_sentence"]].copy()
                new_keys = keys[is_new]

                seen_keys.update(new_keys.tolist())

                # Update unique counts
                n_new = len(new_chunk)
                global_unique_cnt += n_new
                per_year_unique[year] += n_new

                # (2) Sample a fraction from GLOBAL-unique pool (reproducible by key hash)
                keep_mask = new_keys.map(lambda k: stable_hash_01(k, args.seed) < args.sample_frac).to_numpy()
                if keep_mask.any():
                    sampled = new_chunk.loc[keep_mask].copy()
                    sampled.insert(1, "year", year)
                    sampled.insert(2, "file", os.path.relpath(fp, args.in_root))

                    per_year_chunks[year].append(sampled)

                    n_keep = len(sampled)
                    global_sample_cnt += n_keep
                    per_year_sample[year] += n_keep

        except Exception as e:
            print(f"[WARN] Failed during chunk read: {fp} | {e}")
            continue

    # ---- Save ----
    for year in years_found:
        out_year_dir = os.path.join(args.out_root, year)
        ensure_dir(out_year_dir)

        if not per_year_chunks[year]:
            print(f"[{year}] sampled=0 (unique_total={per_year_unique[year]:,})")
            continue

        out_df = pd.concat(per_year_chunks[year], ignore_index=True)
        out_path = os.path.join(out_year_dir, f"gpt_unique_samples_{year}_global{args.sample_frac:.4f}.csv.gz")
        out_df.to_csv(out_path, index=False, compression="gzip")
        print(f"[{year}] saved: {out_path} (rows={len(out_df):,})")

    # ---- Print stats ----
    print("\n===== SUMMARY (GLOBAL UNIQUE → FRACTION SAMPLE) =====")
    for year in years_found:
        print(f"{year}: sampled={per_year_sample[year]:,} / unique_total={per_year_unique[year]:,}")

    print("\n----- GLOBAL -----")
    print(f"GLOBAL sampled     = {global_sample_cnt:,}")
    print(f"GLOBAL unique_total= {global_unique_cnt:,}")
    if global_unique_cnt > 0:
        print(f"GLOBAL sampled frac= {global_sample_cnt / global_unique_cnt:.4%}")

if __name__ == "__main__":
    main()