#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize job-posting–level margin ratio distribution
(positive margins only, log-binned x-axis with bar edges).

Definitions:
- One job posting = (row_idx, file_path)
- margin_ratio = (subs_trend - truth_trend) / truth_trend
- For each posting, the maximum margin_ratio across rows is used
- Report the count and ratio of postings with margin_ratio > 0
- Log binning on x-axis only (10^-1 ~ max)
- Histogram bars with edge color and linewidth
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def main():
    # --------------------------------------------------
    # Arguments
    # --------------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-csv",
        default="./sample_1000_scored_with_trends.csv",
        help="Input CSV containing trend-augmented scoring results"
    )
    ap.add_argument(
        "--out-dir",
        default="./margin_ratio_distribution",
        help="Output path for the histogram image"
    )
    ap.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding for input CSV"
    )
    args = ap.parse_args()

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    df = pd.read_csv(args.in_csv, encoding=args.encoding)

    # --------------------------------------------------
    # Required columns check
    # --------------------------------------------------
    required_cols = {"truth_trend", "subs_trend", "row_idx", "file_path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --------------------------------------------------
    # Remove NaNs and avoid division by zero
    # --------------------------------------------------
    df = df.dropna(subset=["truth_trend", "subs_trend"]).copy()
    df = df[df["truth_trend"] != 0]

    # --------------------------------------------------
    # Compute margin ratio (percentage)
    # --------------------------------------------------
    df["margin_ratio"] = (df["subs_trend"] - df["truth_trend"]) / df["truth_trend"] * 100

    # --------------------------------------------------
    # Aggregate at job-posting level (max margin per posting)
    # --------------------------------------------------
    group_cols = ["row_idx", "file_path"]
    postings = (
        df.groupby(group_cols, as_index=False)["margin_ratio"]
          .max()
          .rename(columns={"margin_ratio": "margin_ratio_max"})
    )

    # --------------------------------------------------
    # Print statistics
    # --------------------------------------------------
    total_posts = len(postings)
    positive_posts = (postings["margin_ratio_max"] > 0).sum()
    ratio_posts = positive_posts / total_posts * 100 if total_posts > 0 else 0.0

    print(
        f"[INFO] Job postings: {total_posts:,} total | "
        f"margin_ratio > 0: {positive_posts:,} ({ratio_posts:.2f}%)"
    )

    # --------------------------------------------------
    # Keep positive margins only
    # --------------------------------------------------
    postings_pos = postings.loc[postings["margin_ratio_max"] > 0].copy()
    if postings_pos.empty:
        print("[WARN] No postings with positive margin_ratio. Nothing to plot.")
        return

    # --------------------------------------------------
    # Define log-spaced bins
    # --------------------------------------------------
    min_val = 1e-1
    max_val = postings_pos["margin_ratio_max"].max()
    bins = np.logspace(np.log10(min_val), np.log10(max_val), 40)

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    sns.set(style="ticks", font_scale=1.4)
    plt.figure(figsize=(8, 5))

    sns.histplot(
        postings_pos["margin_ratio_max"],
        bins=bins,
        color="#55C0C2",
        edgecolor="#333333",
        linewidth=0.6,
        alpha=0.85,
    )

    plt.xscale("log")
    plt.margins(x=0)
    plt.grid(False)

    plt.xlabel("Exposure Gain (%)", fontsize=24, labelpad=10)
    plt.ylabel("Count", fontsize=24, labelpad=10)

    plt.tight_layout()
    plt.savefig(args.out_dir, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved histogram → {args.out_dir}")


if __name__ == "__main__":
    main()