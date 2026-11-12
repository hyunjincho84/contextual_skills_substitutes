#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compute posting-level Google Trends margin ratios and plot their distribution."""

import argparse
import glob
import os
import re
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DATA_DIR = os.environ.get("BASE_DATA_DIR", "/home/jovyan/LEM_data3/hyunjincho/LEM")
ANALYSIS_YEAR = os.environ.get("ANALYSIS_YEAR", "2025")
EXPOSURE_GAIN_DIR = os.path.join(BASE_DATA_DIR, "exposure_gain")
DEFAULT_PRED_DIR = os.path.join(
    BASE_DATA_DIR,
    "predictions_clean",
    "bert_pred",
    "pred",
    ANALYSIS_YEAR,
)
DEFAULT_PRED_GLOB = f"predictions_{ANALYSIS_YEAR}-*.csv.gz"
DEFAULT_MARGIN_CSV = os.path.join(
    EXPOSURE_GAIN_DIR, f"trends_by_pair_{ANALYSIS_YEAR}.csv"
)
DEFAULT_OUT_FIG = os.path.join(
    EXPOSURE_GAIN_DIR,
    f"posting_margin_ratio_hist_{ANALYSIS_YEAR}_clean.png",
)


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def parse_year_month_from_filename(path: str) -> str:
    """Extract YYYYMM from predictions_YYYY-MM.csv[.gz]."""
    base = os.path.basename(path)
    match = re.search(r"(\d{4})-(\d{1,2})", base)
    if match:
        return f"{match.group(1)}{int(match.group(2)):02d}"
    match = re.search(r"(\d{6})", base)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot parse year_month from {path}")


def drop_bidirectional(df_long: pd.DataFrame, posting_cols: List[str]) -> pd.DataFrame:
    """Drop both directions when A->B and B->A occur in one posting."""
    work = df_long.copy()
    truth = work["truth"].astype(str)
    subs = work["subs"].astype(str)
    work["_a"] = truth.where(truth <= subs, subs)
    work["_b"] = subs.where(truth <= subs, truth)
    work["_dir"] = (work["truth"] != work["_a"]).astype(int)

    grouped = work.groupby(posting_cols + ["_a", "_b"])["_dir"].nunique()
    bidirectional = grouped[grouped == 2].reset_index()[posting_cols + ["_a", "_b"]]
    marked = work.merge(
        bidirectional,
        on=posting_cols + ["_a", "_b"],
        how="left",
        indicator=True,
    )
    to_drop = marked["_merge"].eq("both")
    out = marked.loc[~to_drop].drop(columns=["_a", "_b", "_dir", "_merge"])

    print("==============================================")
    print("[Bidirectional-swap removal]")
    print(f"Posting id columns: {posting_cols}")
    print(f"Rows removed: {int(to_drop.sum()):,}")
    print(f"Remaining long rows: {len(out):,}")
    print("==============================================")
    return out


def load_margin_table(path: str, encoding: str) -> pd.DataFrame:
    trends = pd.read_csv(path, encoding=encoding, low_memory=False)
    required = {"truth", "subs", "year_month", "truth_trend", "subs_trend"}
    missing = required.difference(trends.columns)
    if missing:
        raise ValueError(
            f"[margin-csv] Missing columns: {sorted(missing)}. "
            f"Columns={list(trends.columns)}"
        )

    for col in ["truth", "subs", "year_month"]:
        trends[col] = trends[col].fillna("").astype(str).str.strip()
    trends["year_month"] = (
        trends["year_month"].str.replace(r"\.0$", "", regex=True).str.zfill(6)
    )
    trends["truth_trend"] = pd.to_numeric(trends["truth_trend"], errors="coerce")
    trends["subs_trend"] = pd.to_numeric(trends["subs_trend"], errors="coerce")

    truth_key = trends["truth"].str.casefold()
    subs_key = trends["subs"].str.casefold()
    truth_first = truth_key.lt(subs_key) | (
        truth_key.eq(subs_key) & trends["truth"].le(trends["subs"])
    )

    # Normalize both legacy directional outputs and new unordered outputs.
    canonical = pd.DataFrame(
        {
            "skill_a": trends["truth"].where(truth_first, trends["subs"]),
            "skill_b": trends["subs"].where(truth_first, trends["truth"]),
            "year_month": trends["year_month"],
            "trend_a": trends["truth_trend"].where(truth_first, trends["subs_trend"]),
            "trend_b": trends["subs_trend"].where(truth_first, trends["truth_trend"]),
        }
    ).drop_duplicates(subset=["skill_a", "skill_b", "year_month"], keep="last")

    # A single unordered Trends query provides both directional margins.
    forward = canonical.rename(
        columns={
            "skill_a": "truth",
            "skill_b": "subs",
            "trend_a": "truth_trend",
            "trend_b": "subs_trend",
        }
    )
    reverse = canonical.rename(
        columns={
            "skill_b": "truth",
            "skill_a": "subs",
            "trend_b": "truth_trend",
            "trend_a": "subs_trend",
        }
    )
    directional = pd.concat([forward, reverse], ignore_index=True)

    zero_or_missing = directional["truth_trend"].isna() | directional["truth_trend"].eq(
        0
    )
    if zero_or_missing.any():
        print(
            "[MARGIN] Dropping directional rows with missing/zero truth_trend: "
            f"{int(zero_or_missing.sum()):,}"
        )
    directional = directional.loc[~zero_or_missing].copy()
    directional["margin_ratio"] = (
        (directional["subs_trend"] - directional["truth_trend"])
        / directional["truth_trend"]
        * 100.0
    )
    return directional[["truth", "subs", "year_month", "margin_ratio"]].drop_duplicates(
        subset=["truth", "subs", "year_month"], keep="last"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred-dir",
        default=DEFAULT_PRED_DIR,
        help="Directory containing monthly predictions",
    )
    parser.add_argument(
        "--pred-glob",
        default=DEFAULT_PRED_GLOB,
        help="Filename glob inside --pred-dir",
    )
    parser.add_argument(
        "--candidate-col",
        default="substitution",
        help="Truth-excluded substitute column in prediction files",
    )
    parser.add_argument(
        "--posting-source-col",
        default="file_path",
        help="Column paired with row_idx to identify a posting",
    )
    parser.add_argument(
        "--preprocessed-root",
        default=None,
        help="Deprecated; current prediction files already contain posting ids",
    )
    parser.add_argument(
        "--margin-csv",
        default=DEFAULT_MARGIN_CSV,
        help="CSV containing pair/month Google Trends values",
    )
    parser.add_argument("--out-fig", default=DEFAULT_OUT_FIG)
    parser.add_argument("--drop-bidirectional", action="store_true")
    parser.add_argument("--encoding", default="utf-8")
    args = parser.parse_args()

    trends = load_margin_table(args.margin_csv, args.encoding)
    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, args.pred_glob)))
    if not pred_files:
        raise FileNotFoundError(
            f"No prediction files matched: {os.path.join(args.pred_dir, args.pred_glob)}"
        )

    all_rows = []
    for pred_path in pred_files:
        year_month = parse_year_month_from_filename(pred_path)
        print(f"[LOAD] {pred_path} -> year_month={year_month}")

        header = pd.read_csv(pred_path, nrows=0)
        required = {
            "row_idx",
            "truth",
            args.candidate_col,
            args.posting_source_col,
        }
        missing = required.difference(header.columns)
        if missing:
            raise ValueError(
                f"[pred] Missing {sorted(missing)} in {pred_path}. "
                f"Columns={list(header.columns)}"
            )

        pred = pd.read_csv(
            pred_path,
            usecols=sorted(required),
            compression="gzip" if pred_path.endswith(".gz") else None,
            encoding=args.encoding,
            low_memory=False,
        )
        truth = pred["truth"].fillna("").astype(str).str.strip()
        subs = pred[args.candidate_col].fillna("").astype(str).str.strip()
        posting_source = (
            pred[args.posting_source_col].fillna("").astype(str).str.strip()
        )
        valid = (
            pred["row_idx"].notna()
            & posting_source.ne("")
            & posting_source.str.casefold().ne("nan")
            & truth.ne("")
            & subs.ne("")
            & truth.str.casefold().ne(subs.str.casefold())
        )

        rows = pd.DataFrame(
            {
                "row_idx": pred.loc[valid, "row_idx"],
                "posting_source": posting_source[valid],
                "year_month": year_month,
                "truth": truth[valid],
                "subs": subs[valid],
            }
        ).reset_index(drop=True)
        print(
            f"[PAIR] valid={len(rows):,}, dropped={len(pred) - len(rows):,}, "
            f"candidate_col={args.candidate_col}"
        )

        rows = rows.merge(
            trends,
            on=["truth", "subs", "year_month"],
            how="left",
            validate="many_to_one",
        )
        rows["margin_ratio"] = pd.to_numeric(rows["margin_ratio"], errors="coerce")
        all_rows.append(rows)

    long_df = pd.concat(all_rows, ignore_index=True)
    matched = int(long_df["margin_ratio"].notna().sum())
    print(
        f"[MERGE] Trends matched: {matched:,}/{len(long_df):,} rows "
        f"({matched / len(long_df) * 100.0 if len(long_df) else 0.0:.2f}%)"
    )

    posting_cols = ["row_idx", "posting_source"]
    if args.drop_bidirectional:
        long_df = drop_bidirectional(long_df, posting_cols)

    long_df["_pos"] = long_df["margin_ratio"].gt(0).fillna(False)
    posting = long_df.groupby(posting_cols, as_index=False).agg(
        has_positive_margin=("_pos", "any"),
        max_margin_ratio=("margin_ratio", "max"),
    )

    total = len(posting)
    positive = int(posting["has_positive_margin"].sum())
    percentage = positive / total * 100.0 if total else 0.0
    print("==============================================")
    print("Posting-level positive-margin substitution rate")
    print(f"Posting id = {posting_cols}")
    print(f"Total postings: {total:,}")
    print(
        "Postings with >=1 positive-margin substitution: "
        f"{positive:,} ({percentage:.2f}%)"
    )
    print("==============================================")

    plot_values = posting.loc[
        posting["max_margin_ratio"] > 0, "max_margin_ratio"
    ].dropna()
    if plot_values.empty:
        print("[WARN] No positive max_margin_ratio values to plot.")
        return

    min_value = 1e-1
    max_value = float(plot_values.max())
    if max_value <= min_value:
        bins = 20
        use_log = False
    else:
        bins = np.logspace(np.log10(min_value), np.log10(max_value), 40)
        use_log = True

    sns.set(style="ticks", font_scale=1.4)
    plt.figure(figsize=(8, 5))
    sns.histplot(
        plot_values,
        bins=bins,
        color="#55C0C2",
        edgecolor="#333333",
        linewidth=0.6,
        alpha=0.85,
    )
    if use_log:
        plt.xscale("log")
    plt.margins(x=0)
    plt.grid(False)
    plt.xlabel("Potential Candidate-Pool Gain (%)", fontsize=20)
    plt.ylabel("Count", fontsize=20)
    plt.tight_layout()
    ensure_dir(os.path.dirname(os.path.abspath(args.out_fig)))
    plt.savefig(args.out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved histogram -> {args.out_fig}")


if __name__ == "__main__":
    main()
