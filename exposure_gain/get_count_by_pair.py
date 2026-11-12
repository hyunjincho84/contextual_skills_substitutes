#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build monthly unordered skill-pair counts from BERT predictions."""

import argparse
import glob
import os
import re
from typing import Optional

import pandas as pd


def parse_year_month_from_filename(path: str) -> Optional[str]:
    """Extract YYYYMM from predictions_YYYY-MM.csv[.gz]."""
    base = os.path.basename(path)
    match = re.search(r"(\d{4})-(\d{1,2})", base)
    if match:
        return f"{match.group(1)}{int(match.group(2)):02d}"
    match = re.search(r"(\d{6})", base)
    return match.group(1) if match else None


def canonicalize_pairs(
    truth: pd.Series,
    subs: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Return a deterministic (skill_a, skill_b) order for each pair."""
    truth_key = truth.str.casefold()
    subs_key = subs.str.casefold()
    truth_first = truth_key.lt(subs_key) | (
        truth_key.eq(subs_key) & truth.le(subs)
    )
    return truth.where(truth_first, subs), subs.where(truth_first, truth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred-dir",
        required=True,
        help="Directory containing monthly BERT prediction files",
    )
    parser.add_argument(
        "--out-csv",
        required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--glob",
        default="predictions_2025-*.csv.gz",
        help="Filename glob inside --pred-dir",
    )
    parser.add_argument(
        "--candidate-col",
        default="substitution",
        help="Truth-excluded substitute column",
    )
    parser.add_argument("--encoding", default="utf-8")
    args = parser.parse_args()

    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, args.glob)))
    if not pred_files:
        raise FileNotFoundError(
            f"No files matched: {os.path.join(args.pred_dir, args.glob)}"
        )

    monthly_counts = []
    total_rows = 0
    valid_rows = 0

    for pred_path in pred_files:
        year_month = parse_year_month_from_filename(pred_path)
        if year_month is None:
            raise ValueError(f"Cannot infer year_month from {pred_path}")

        header = pd.read_csv(pred_path, nrows=0)
        required = {"truth", args.candidate_col}
        missing = required.difference(header.columns)
        if missing:
            raise ValueError(f"{pred_path} missing columns: {sorted(missing)}")

        df = pd.read_csv(
            pred_path,
            usecols=["truth", args.candidate_col],
            compression="gzip" if pred_path.endswith(".gz") else None,
            encoding=args.encoding,
        )
        total_rows += len(df)

        truth = df["truth"].fillna("").astype(str).str.strip()
        subs = df[args.candidate_col].fillna("").astype(str).str.strip()
        valid = (
            truth.ne("")
            & subs.ne("")
            & truth.str.casefold().ne(subs.str.casefold())
        )

        pair_truth, pair_subs = canonicalize_pairs(truth[valid], subs[valid])
        pairs = pd.DataFrame({
            # Retain these names for get_trend.py compatibility, but store
            # the two skills in deterministic, unordered order.
            "truth": pair_truth,
            "subs": pair_subs,
            "year_month": year_month,
        })
        valid_rows += len(pairs)
        if not pairs.empty:
            monthly_counts.append(
                pairs.groupby(
                    ["truth", "subs", "year_month"],
                    as_index=False,
                ).size().rename(columns={"size": "count"})
            )

        print(
            f"[LOAD] {pred_path} -> year_month={year_month}, "
            f"rows={len(df):,}, valid_pairs={len(pairs):,}"
        )

    if not monthly_counts:
        raise RuntimeError("No valid (truth, substitute) pairs found.")

    counts = (
        pd.concat(monthly_counts, ignore_index=True)
        .groupby(["truth", "subs", "year_month"], as_index=False)["count"]
        .sum()
        .sort_values(
            ["year_month", "count", "truth", "subs"],
            ascending=[True, False, True, True],
        )
        .reset_index(drop=True)
    )

    output_dir = os.path.dirname(os.path.abspath(args.out_csv))
    os.makedirs(output_dir, exist_ok=True)
    counts.to_csv(args.out_csv, index=False, encoding=args.encoding)

    print("==============================================")
    print(f"Saved: {args.out_csv}")
    print(f"Input rows: {total_rows:,}")
    print(f"Valid substitute rows: {valid_rows:,}")
    print(f"Dropped invalid rows: {total_rows - valid_rows:,}")
    print(f"Unique monthly unordered pairs: {len(counts):,}")
    print(counts.head(10))
    print("==============================================")


if __name__ == "__main__":
    main()
