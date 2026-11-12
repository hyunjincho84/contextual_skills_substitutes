#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create reproducible yearly GPT evaluation samples from BERT prediction outputs.

What this script does:
1) Scan yearly `predictions_*.csv.gz` files under `IN_ROOT/pred/{year}` containing:
   - truth
   - masked_sentence
2) Build a GLOBAL-unique pool of (truth, masked_sentence) pairs across ALL files/years.
3) Reproducibly sample 1,000 rows per year.
4) Combine all sampled years and save one `bert.csv` file.

Output columns:
- row_idx: row index within the original file (0-based within that file)
- year: extracted from the file path
- file: relative path of the source file
- truth
- masked_sentence
- bert_pred_top5: BERT's ordered Top-5 predictions
- bert_substitution: highest-probability BERT prediction excluding truth
"""

import os
import re
import glob
import argparse
import random
from typing import Optional, List, Dict

import pandas as pd
from tqdm import tqdm

BASE_DATA_DIR = os.environ.get("BASE_DATA_DIR", "/home/jovyan/LEM_data3/hyunjincho/LEM")
IN_ROOT = os.environ.get("BERT_PRED_DIR", os.path.join(BASE_DATA_DIR, "predictions_clean", "bert_pred"))
OUT_PATH = os.environ.get(
    "GPT_BERT_SAMPLES_PATH", os.path.join(BASE_DATA_DIR, "predictions_clean", "gpt_samplings", "bert.csv")
)

CHUNK_SIZE = 200_000
INPUT_GLOB = "pred/20*/predictions_*.csv.gz"
KEY_SEP = "|||SEP|||"
PAIR_COLS = ["truth", "masked_sentence"]
REQ_COLS = PAIR_COLS + ["pred_top5", "substitution"]
SAMPLES_PER_YEAR = 1_000
PARAMETER = "text"

# ---------------- utils ----------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def extract_year_from_path(fp: str) -> Optional[str]:
    m = re.search(r"/(20\d{2})/", fp)
    return m.group(1) if m else None

def list_input_files(in_root: str, input_glob: str, years: Optional[List[str]] = None) -> List[str]:
    pattern = os.path.join(in_root, input_glob)
    files = sorted(glob.glob(pattern))
    if years:
        want = set(str(y) for y in years)
        files = [fp for fp in files if (extract_year_from_path(fp) in want)]
    return files

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", type=str, default=IN_ROOT)
    ap.add_argument("--out-path", type=str, default=OUT_PATH)
    ap.add_argument("--input-glob", type=str, default=INPUT_GLOB)
    ap.add_argument("--years", nargs="*", default=None, help="e.g., --years 2020 2021")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--samples-per-year", type=int, default=SAMPLES_PER_YEAR)
    ap.add_argument(
        "--parameter",
        choices=("text",),
        default=PARAMETER,
        help="Sampling identity; text uses unique (truth, masked_sentence) pairs.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if os.path.exists(args.out_path):
        raise FileExistsError(f"Refusing to overwrite existing sample: {args.out_path}")

    files = list_input_files(args.in_root, args.input_glob, args.years)
    if not files:
        print(f"[ERROR] No input files found: {os.path.join(args.in_root, args.input_glob)}")
        return

    years_found = sorted({extract_year_from_path(fp) for fp in files if extract_year_from_path(fp)})
    print(f"[INFO] Found {len(files)} files across years: {years_found}")
    if args.samples_per_year <= 0:
        raise ValueError("--samples-per-year must be positive")
    print(f"[INFO] samples-per-year: {args.samples_per_year}")
    print(f"[INFO] parameter={args.parameter}  seed={args.seed}")
    print(f"[INFO] Output path: {args.out_path}")

    # Track GLOBAL unique keys across all files/years
    seen_keys = set()

    # Stats and one deterministic reservoir per year.
    global_unique_cnt = 0
    per_year_unique: Dict[str, int] = {y: 0 for y in years_found}
    per_year_samples: Dict[str, List[dict]] = {y: [] for y in years_found}
    per_year_rng = {
        y: random.Random(f"{args.seed}:{args.parameter}:{y}") for y in years_found
    }

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

                chunk = chunk.dropna(subset=PAIR_COLS)
                if chunk.empty:
                    continue

                chunk["truth"] = chunk["truth"].astype(str)
                chunk["masked_sentence"] = chunk["masked_sentence"].astype(str)

                keys = chunk["truth"] + KEY_SEP + chunk["masked_sentence"]

                # (1) GLOBAL-unique filter
                is_new = ~keys.isin(seen_keys)
                if not is_new.any():
                    continue

                new_chunk = chunk.loc[is_new, ["row_idx"] + REQ_COLS].copy()
                new_keys = keys[is_new]

                # keys.isin(seen_keys) does not remove duplicates within this chunk.
                within_chunk_new = ~new_keys.duplicated()
                new_chunk = new_chunk.loc[within_chunk_new].copy()
                new_keys = new_keys.loc[within_chunk_new]

                seen_keys.update(new_keys.tolist())

                # (2) Uniform reservoir sampling over unique text pairs in each year.
                source_file = os.path.relpath(fp, args.in_root)
                samples = per_year_samples[year]
                rng = per_year_rng[year]
                for row in new_chunk.to_dict("records"):
                    global_unique_cnt += 1
                    per_year_unique[year] += 1
                    seen_count = per_year_unique[year]

                    row = {
                        "row_idx": row["row_idx"],
                        "year": year,
                        "file": source_file,
                        "truth": row["truth"],
                        "masked_sentence": row["masked_sentence"],
                        "bert_pred_top5": row["pred_top5"],
                        "bert_substitution": row["substitution"],
                    }
                    if len(samples) < args.samples_per_year:
                        samples.append(row)
                    else:
                        replace_idx = rng.randrange(seen_count)
                        if replace_idx < args.samples_per_year:
                            samples[replace_idx] = row

        except Exception as e:
            print(f"[WARN] Failed during chunk read: {fp} | {e}")
            continue

    # ---- Save ----
    all_samples = []
    for year in years_found:
        sampled_count = len(per_year_samples[year])
        if sampled_count < args.samples_per_year:
            print(f"[WARN] {year}: requested={args.samples_per_year:,}, sampled={sampled_count:,}")
        all_samples.extend(per_year_samples[year])

    out_dir = os.path.dirname(os.path.abspath(args.out_path))
    ensure_dir(out_dir)
    out_df = pd.DataFrame(all_samples)
    out_df.to_csv(args.out_path, index=False)
    print(f"[INFO] Saved combined sample: {args.out_path} (rows={len(out_df):,})")

    # ---- Print stats ----
    print("\n===== SUMMARY (GLOBAL UNIQUE → PER-YEAR SAMPLE) =====")
    for year in years_found:
        print(f"{year}: sampled={len(per_year_samples[year]):,} / unique_total={per_year_unique[year]:,}")

    print("\n----- GLOBAL -----")
    global_sample_cnt = sum(len(rows) for rows in per_year_samples.values())
    print(f"GLOBAL sampled     = {global_sample_cnt:,}")
    print(f"GLOBAL unique_total= {global_unique_cnt:,}")

if __name__ == "__main__":
    main()