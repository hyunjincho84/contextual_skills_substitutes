# -*- coding: utf-8 -*-
"""
Join partial Google Trends results with sample_1000_scored.csv
using (truth, subs, year_month) as the key.

- Attach truth_trend, subs_trend, and trend_diff where keys exist
- Compute margin = trend_diff - sv_loss_from_norm

Default paths:
  trends_partial: ./counts_by_pair_with_trends_monthly.csv.partial
  sample_csv    : ./sample_1000_scored.csv
  out_csv       : ./sample_1000_scored_with_trends.csv

Example usage:
  python3 get_margin.py \
    --partial ./counts_by_pair_with_trends_monthly.csv.partial \
    --sample  ./sample_1000_scored.csv \
    --out     ./sample_1000_scored_with_trends.csv
"""

import argparse
import os
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--partial",
        default="./counts_by_pair_with_trends_monthly.csv.partial",
        help="Path to partial trends file (.partial)"
    )
    ap.add_argument(
        "--sample",
        default="./sample_1000_scored.csv",
        help="Sample score CSV (must include truth, subs, year_month, sv_loss_from_norm)"
    )
    ap.add_argument(
        "--out",
        default="./sample_1000_scored_with_trends.csv",
        help="Output CSV path"
    )
    ap.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding for input/output"
    )
    args = ap.parse_args()

    # 1) Load trends input
    if not os.path.exists(args.partial):
        # If .partial does not exist, fall back to the final CSV
        alt = args.partial.replace(".partial", "")
        if os.path.exists(alt):
            print(f"⚠️ .partial not found — falling back to final file: {alt}")
            partial_path = alt
        else:
            raise FileNotFoundError(
                f"Trends file not found: {args.partial} (or {alt})"
            )
    else:
        partial_path = args.partial

    print(f"📥 Loading trends from: {partial_path}")
    df_tr = pd.read_csv(partial_path, encoding=args.encoding)

    required_tr_cols = {
        "truth", "subs", "year_month",
        "truth_trend", "subs_trend", "trend_diff"
    }
    if not required_tr_cols.issubset(df_tr.columns):
        raise ValueError(
            f"Missing required columns in trends file: "
            f"{required_tr_cols - set(df_tr.columns)}"
        )

    # Deduplicate by key (truth, subs, year_month)
    # If duplicates exist, keep the last row (assuming append order in .partial)
    df_tr["__order__"] = range(len(df_tr))
    df_tr = (
        df_tr.sort_values("__order__")
             .drop_duplicates(
                 subset=["truth", "subs", "year_month"],
                 keep="last"
             )
             .drop(columns="__order__")
    )

    # 2) Load sample file
    print(f"📥 Loading sample from: {args.sample}")
    df_samp = pd.read_csv(args.sample, encoding=args.encoding)

    required_s_cols = {"truth", "subs", "year_month", "sv_loss_from_norm"}
    missing_s = required_s_cols - set(df_samp.columns)
    if missing_s:
        raise ValueError(
            f"Missing required columns in sample file: {missing_s}"
        )

    # Normalize key columns as strings (trim whitespace)
    for c in ["truth", "subs", "year_month"]:
        df_tr[c] = df_tr[c].astype(str).str.strip()
        df_samp[c] = df_samp[c].astype(str).str.strip()

    # 3) Join (left join on sample; keep only successfully matched rows)
    merged = df_samp.merge(
        df_tr[
            ["truth", "subs", "year_month",
             "truth_trend", "subs_trend", "trend_diff"]
        ],
        on=["truth", "subs", "year_month"],
        how="left",
        validate="many_to_one"
    )

    before = len(merged)
    matched = merged["truth_trend"].notna().sum()
    print(
        f"🔗 Join result: {matched:,} matched rows "
        f"out of {before:,} total"
    )

    # Keep only matched rows
    merged = merged[merged["truth_trend"].notna()].copy()

    # 4) Compute margin = trend_diff - sv_loss_from_norm
    for col in ["truth_trend", "subs_trend", "trend_diff", "sv_loss_from_norm"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["margin"] = merged["trend_diff"] - merged["sv_loss_from_norm"]

    # 5) Reorder columns for readability
    front_cols = [
        "truth", "subs", "year_month",
        "truth_trend", "subs_trend", "trend_diff",
        "sv_loss_from_norm", "margin"
    ]
    other_cols = [c for c in merged.columns if c not in front_cols]
    merged = merged[front_cols + other_cols]

    # 6) Save output
    merged.to_csv(args.out, index=False, encoding=args.encoding)
    print(f"✅ Saved: {args.out}")
    print(f"   ➤ Number of matched rows: {len(merged):,}")


if __name__ == "__main__":
    main()