#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build (truth, subs, year_month, count) from BERT prediction files.

- Reads: predictions_2025-*.csv.gz
- subs = top-ranked WRONG label from pred_top5
- Aggregates counts by (truth, subs, year_month)

Output example:
truth,subs,year_month,count
linux,unix,202505,53
"""

import os
import re
import glob
import json
import ast
import argparse
from typing import List, Optional

import pandas as pd
import numpy as np


# --------------------------------------------------
# Utils
# --------------------------------------------------

def parse_year_month_from_filename(path: str) -> Optional[str]:
    """
    Extract YYYYMM from filename like:
      predictions_2025-01.csv.gz
      predictions_202505.csv.gz
    """
    base = os.path.basename(path)
    m = re.search(r"(\d{4})-(\d{1,2})", base)
    if m:
        return f"{m.group(1)}{int(m.group(2)):02d}"
    m = re.search(r"(\d{6})", base)
    if m:
        return m.group(1)
    return None


def safe_parse_topk(x) -> List[str]:
    """
    Parse pred_top5 which may be:
      - "['a','b','c']"
      - '["a","b"]'
      - "a|b|c"
      - "a, b, c"
      - already a list
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []

    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]

    s = str(x).strip()
    if not s:
        return []

    # JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(v).strip() for v in obj if str(v).strip()]
    except Exception:
        pass

    # Python literal
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return [str(v).strip() for v in obj if str(v).strip()]
    except Exception:
        pass

    # Delimiters
    if "|" in s:
        return [p.strip() for p in s.split("|") if p.strip()]
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]

    return [s]


def pick_top_wrong(truth: str, topk: List[str]) -> Optional[str]:
    """Return first candidate != truth."""
    t = str(truth).strip()
    for c in topk:
        c = str(c).strip()
        if c and c != t:
            return c
    return None


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred-dir",
        required=True,
        help="Directory containing predictions_2025-*.csv.gz"
    )
    ap.add_argument(
        "--out-csv",
        required=True,
        help="Output CSV path (counts_by_pair.csv)"
    )
    ap.add_argument(
        "--glob",
        default="predictions_2025-*.csv.gz",
        help="Filename glob pattern"
    )
    ap.add_argument(
        "--encoding",
        default="utf-8"
    )
    args = ap.parse_args()

    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, args.glob)))
    if not pred_files:
        raise FileNotFoundError(f"No files matched in {args.pred_dir}")

    rows = []

    for pf in pred_files:
        ym = parse_year_month_from_filename(pf)
        if ym is None:
            raise ValueError(f"Cannot infer year_month from {pf}")

        print(f"[LOAD] {pf} -> year_month={ym}")

        df = pd.read_csv(
            pf,
            compression="gzip" if pf.endswith(".gz") else None,
            encoding=args.encoding,
            low_memory=False
        )

        required = {"truth", "pred_top5"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{pf} missing columns: {missing}")

        for truth, top5 in zip(df["truth"], df["pred_top5"]):
            topk = safe_parse_topk(top5)
            subs = pick_top_wrong(truth, topk)
            if subs is None:
                continue

            rows.append({
                "truth": str(truth).strip(),
                "subs": str(subs).strip(),
                "year_month": ym
            })

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        print("❌ No valid (truth, subs) pairs found.")
        return

    counts = (
        long_df
        .groupby(["truth", "subs", "year_month"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )

    counts.to_csv(args.out_csv, index=False, encoding=args.encoding)

    print("==============================================")
    print(f"Saved: {args.out_csv}")
    print(f"Unique pairs: {len(counts):,}")
    print("Top 10:")
    print(counts.head(10))
    print("==============================================")


if __name__ == "__main__":
    main()
    
    
"""
python3 get_count_by_pair.py \
  --pred-dir /home/jovyan/LEM_data2/hyunjincho/bert_pred_new/pred/2025 \
  --out-csv ./counts_by_pair_2025.csv
"""