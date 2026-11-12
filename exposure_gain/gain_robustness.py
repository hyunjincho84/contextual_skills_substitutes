#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Posting-level robustness checks for Google-Trends exposure gain.

This is the clean-data counterpart of
``rebuttal_kdd/exposure_gain/get_graph_multi_metric.py``.  Clean prediction
files already contain ``row_idx``, a posting source path, and the highest
truth-excluded candidate (``substitution``), so no positional realignment to
the old preprocessed files is needed.

The appendix statistics follow the reference scopes:
  * median original trend x: the directional Trends table;
  * alternative gain means: posting-level maxima after prediction/Trend merge.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_COLUMNS = (
    "margin_ratio_pct",
    "absolute_gain",
    "clipped_margin_ratio_pct",
    "log_ratio_gain",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge clean predictions with monthly Trends values and compute "
            "posting-level robustness metrics."
        )
    )
    parser.add_argument(
        "--pred-dir",
        required=True,
        type=Path,
        help="Clean BERT prediction directory for one year, e.g. predictions_clean/bert_pred/pred/2025.",
    )
    parser.add_argument("--pred-glob", default="predictions_2025-*.csv.gz")
    parser.add_argument(
        "--candidate-col",
        default="substitution",
        help="Highest-probability truth-excluded candidate in clean predictions.",
    )
    parser.add_argument(
        "--posting-source-col",
        default="file_path",
        help="Column paired with row_idx to identify a posting.",
    )
    parser.add_argument("--margin-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="exposure_gain_robustness")
    parser.add_argument(
        "--drop-bidirectional",
        action="store_true",
        help="Drop both A->B and B->A when they occur in the same posting.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="epsilon in log((y+epsilon)/(x+epsilon)); reference value is 1.0.",
    )
    parser.add_argument(
        "--clip-c",
        type=float,
        default=10.0,
        help="floor c in (y-x)/max(x,c); reference value is 10.",
    )
    parser.add_argument("--hist-bins", type=int, default=50)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--encoding", default="utf-8")
    args = parser.parse_args()
    if args.epsilon <= 0:
        parser.error("--epsilon must be positive")
    if args.clip_c <= 0:
        parser.error("--clip-c must be positive")
    if args.hist_bins < 1:
        parser.error("--hist-bins must be at least 1")
    return args


def parse_year_month_from_filename(path: str) -> str:
    base = os.path.basename(path)
    match = re.search(r"(\d{4})-(\d{1,2})", base)
    if match:
        return f"{match.group(1)}{int(match.group(2)):02d}"
    match = re.search(r"(\d{6})", base)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot parse year_month from prediction file: {path}")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def atomic_csv(frame: pd.DataFrame, path: Path, encoding: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, encoding=encoding)
    os.replace(tmp, path)


def load_trends(
    path: Path, encoding: str, epsilon: float, clip_c: float
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Trends CSV not found: {path}")
    trends = pd.read_csv(path, encoding=encoding, low_memory=False)
    required = {"truth", "subs", "year_month", "truth_trend", "subs_trend"}
    missing = required.difference(trends.columns)
    if missing:
        raise ValueError(
            f"Trends CSV is missing {sorted(missing)}; columns={list(trends.columns)}"
        )

    for column in ("truth", "subs", "year_month"):
        trends[column] = trends[column].fillna("").astype(str).str.strip()
    trends["year_month"] = (
        trends["year_month"].str.replace(r"\.0$", "", regex=True).str.zfill(6)
    )
    trends["truth_trend"] = pd.to_numeric(trends["truth_trend"], errors="coerce")
    trends["subs_trend"] = pd.to_numeric(trends["subs_trend"], errors="coerce")
    trends = trends.dropna(subset=["truth_trend", "subs_trend"]).copy()

    negative = trends["truth_trend"].lt(0) | trends["subs_trend"].lt(0)
    if negative.any():
        raise ValueError(
            f"Found {int(negative.sum()):,} negative Google Trends scores."
        )

    if "trend_diff" in trends.columns:
        trend_diff = pd.to_numeric(trends["trend_diff"], errors="coerce")
        fallback = trends["subs_trend"] - trends["truth_trend"]
        trends["absolute_gain"] = trend_diff.fillna(fallback)
    else:
        trends["absolute_gain"] = trends["subs_trend"] - trends["truth_trend"]

    x = trends["truth_trend"]
    y = trends["subs_trend"]
    trends["margin_ratio_pct"] = np.where(
        x.ne(0), trends["absolute_gain"] / x * 100.0, np.nan
    )
    trends["clipped_margin_ratio_pct"] = (
        trends["absolute_gain"] / np.maximum(x, clip_c) * 100.0
    )
    trends["log_ratio_gain"] = np.log((y + epsilon) / (x + epsilon))

    columns = [
        "truth",
        "subs",
        "year_month",
        "truth_trend",
        "subs_trend",
        *METRIC_COLUMNS,
    ]
    return trends[columns].drop_duplicates(
        subset=["truth", "subs", "year_month"], keep="last"
    )


def load_clean_predictions(args: argparse.Namespace, trends: pd.DataFrame) -> pd.DataFrame:
    pred_dir = args.pred_dir.expanduser().resolve()
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Clean prediction directory not found: {pred_dir}")
    pred_files = sorted(glob.glob(str(pred_dir / args.pred_glob)))
    if not pred_files:
        raise FileNotFoundError(f"No prediction files matched: {pred_dir / args.pred_glob}")

    required = {
        "row_idx",
        "truth",
        args.candidate_col,
        args.posting_source_col,
    }
    parts: list[pd.DataFrame] = []
    for pred_path in pred_files:
        year_month = parse_year_month_from_filename(pred_path)
        header = pd.read_csv(pred_path, nrows=0)
        missing = required.difference(header.columns)
        if missing:
            raise ValueError(
                f"Prediction file {pred_path} is missing {sorted(missing)}; "
                f"columns={list(header.columns)}"
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
                "row_idx": pred.loc[valid, "row_idx"].to_numpy(),
                "posting_source": posting_source.loc[valid].to_numpy(),
                "year_month": year_month,
                "truth": truth.loc[valid].to_numpy(),
                "subs": subs.loc[valid].to_numpy(),
            }
        )
        rows = rows.merge(
            trends,
            on=["truth", "subs", "year_month"],
            how="left",
            validate="many_to_one",
        )
        matched = int(rows["truth_trend"].notna().sum())
        print(
            f"[MERGE] {os.path.basename(pred_path)}: valid={len(rows):,}, "
            f"Trends matched={matched:,}"
        )
        parts.append(rows)
    return pd.concat(parts, ignore_index=True)


def drop_bidirectional(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    work = frame.copy()
    truth = work["truth"].astype(str)
    subs = work["subs"].astype(str)
    work["_a"] = truth.where(truth <= subs, subs)
    work["_b"] = subs.where(truth <= subs, truth)
    work["_direction"] = work["truth"].ne(work["_a"]).astype(np.int8)
    keys = ["row_idx", "posting_source", "_a", "_b"]
    directions = work.groupby(keys, dropna=False)["_direction"].nunique()
    bidirectional = directions[directions.eq(2)].reset_index()[keys]
    marked = work.merge(bidirectional, on=keys, how="left", indicator=True)
    to_drop = marked["_merge"].eq("both")
    output = marked.loc[~to_drop].drop(
        columns=["_a", "_b", "_direction", "_merge"]
    )
    return output.reset_index(drop=True), int(to_drop.sum())


def aggregate_postings(frame: pd.DataFrame) -> pd.DataFrame:
    posting_keys = ["row_idx", "posting_source"]
    aggregations: dict[str, tuple[str, Any]] = {
        "prediction_rows": ("truth", "size"),
        "matched_rows": ("truth_trend", "count"),
    }
    for metric in METRIC_COLUMNS:
        aggregations[f"max_{metric}"] = (metric, "max")
        aggregations[f"has_positive_{metric}"] = (
            metric,
            lambda values: pd.to_numeric(values, errors="coerce").gt(0).any(),
        )
    return frame.groupby(posting_keys, as_index=False).agg(**aggregations)


def make_summary(trends: pd.DataFrame, posting: pd.DataFrame) -> pd.DataFrame:
    specifications = [
        (
            "Median original trend score x",
            float(trends["truth_trend"].median()),
            "trend score",
            "Trends table",
            int(trends["truth_trend"].count()),
            "median(x)",
        ),
        (
            "Mean absolute gain (y-x)",
            float(posting["max_absolute_gain"].mean()),
            "trend-score points",
            "posting-level maximum",
            int(posting["max_absolute_gain"].count()),
            "mean(max_posting(y-x))",
        ),
        (
            "Mean clipped relative gain",
            float(posting["max_clipped_margin_ratio_pct"].mean()),
            "percent",
            "posting-level maximum",
            int(posting["max_clipped_margin_ratio_pct"].count()),
            "mean(max_posting(100*(y-x)/max(x,c)))",
        ),
        (
            "Mean log-ratio gain",
            float(posting["max_log_ratio_gain"].mean()),
            "log points",
            "posting-level maximum",
            int(posting["max_log_ratio_gain"].count()),
            "mean(max_posting(log((y+epsilon)/(x+epsilon))))",
        ),
    ]
    return pd.DataFrame(
        specifications,
        columns=["metric", "value", "unit", "analysis_scope", "n", "definition"],
    )


def make_x_diagnostics(trends: pd.DataFrame, clip_c: float) -> pd.DataFrame:
    x = trends["truth_trend"].dropna()
    rows = [
        ("count", float(x.count()), "rows"),
        ("mean", float(x.mean()), "trend score"),
        ("min", float(x.min()), "trend score"),
        ("q01", float(x.quantile(0.01)), "trend score"),
        ("q05", float(x.quantile(0.05)), "trend score"),
        ("q10", float(x.quantile(0.10)), "trend score"),
        ("q25", float(x.quantile(0.25)), "trend score"),
        ("median", float(x.median()), "trend score"),
        ("q75", float(x.quantile(0.75)), "trend score"),
        ("q90", float(x.quantile(0.90)), "trend score"),
        ("q95", float(x.quantile(0.95)), "trend score"),
        ("q99", float(x.quantile(0.99)), "trend score"),
        ("max", float(x.max()), "trend score"),
        ("share_x_equal_zero", float(x.eq(0).mean() * 100.0), "percent"),
        ("share_x_below_clip_c", float(x.lt(clip_c).mean() * 100.0), "percent"),
    ]
    return pd.DataFrame(rows, columns=["diagnostic", "value", "unit"])


def plot_x_distribution(trends: pd.DataFrame, path: Path, bins: int) -> None:
    x = trends["truth_trend"].dropna().to_numpy(dtype=np.float64)
    median = float(np.median(x))
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.hist(x, bins=bins, color="#4C78A8", alpha=0.85, edgecolor="white")
    ax.axvline(
        median,
        color="#D62728",
        linestyle="--",
        linewidth=2,
        label=f"Median x = {median:.2f}",
    )
    ax.set_xlabel("Original Google Trends score x")
    ax.set_ylabel("Directional pair-month count")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(summary: pd.DataFrame) -> None:
    values = dict(zip(summary["metric"], summary["value"]))
    print("==============================================")
    print("Posting-level exposure-gain robustness")
    print("==============================================")
    print(f"Median original trend score x: {values['Median original trend score x']:.2f}")
    print(f"Mean absolute gain (y-x): {values['Mean absolute gain (y-x)']:.2f}")
    print(f"Mean clipped relative gain: {values['Mean clipped relative gain']:.2f}%")
    print(f"Mean log-ratio gain: {values['Mean log-ratio gain']:.2f}")
    print("==============================================")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trends_path = args.margin_csv.expanduser().resolve()
    trends = load_trends(trends_path, args.encoding, args.epsilon, args.clip_c)
    matched = load_clean_predictions(args, trends)

    rows_before_bidirectional = len(matched)
    bidirectional_rows_removed = 0
    if args.drop_bidirectional:
        matched, bidirectional_rows_removed = drop_bidirectional(matched)
        print(f"[FILTER] Bidirectional rows removed: {bidirectional_rows_removed:,}")

    posting = aggregate_postings(matched)
    summary = make_summary(trends, posting)
    diagnostics = make_x_diagnostics(trends, args.clip_c)

    summary_path = output_dir / f"{args.prefix}_summary.csv"
    diagnostics_path = output_dir / f"{args.prefix}_x_diagnostics.csv"
    matched_path = output_dir / f"{args.prefix}_matched_rows.csv.gz"
    posting_path = output_dir / f"{args.prefix}_posting_level_metrics.csv.gz"
    plot_path = output_dir / f"{args.prefix}_x_distribution.png"
    metadata_path = output_dir / f"{args.prefix}_metadata.json"

    atomic_csv(summary, summary_path, args.encoding)
    atomic_csv(diagnostics, diagnostics_path, args.encoding)
    matched.to_csv(matched_path, index=False, encoding=args.encoding, compression="gzip")
    posting.to_csv(posting_path, index=False, encoding=args.encoding, compression="gzip")
    if not args.no_plot:
        plot_x_distribution(trends, plot_path, args.hist_bins)

    metadata = {
        "bidirectional_rows_removed": bidirectional_rows_removed,
        "candidate_col": args.candidate_col,
        "clip_c": args.clip_c,
        "drop_bidirectional": args.drop_bidirectional,
        "epsilon": args.epsilon,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "margin_csv": str(trends_path),
        "matched_rows_after_filter": int(len(matched)),
        "matched_rows_before_filter": int(rows_before_bidirectional),
        "matplotlib_version": matplotlib.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "platform": platform.platform(),
        "posting_source_col": args.posting_source_col,
        "postings": int(len(posting)),
        "pred_dir": str(args.pred_dir.expanduser().resolve()),
        "pred_glob": args.pred_glob,
        "python_version": sys.version,
        "reference_code": "/home/jovyan/hyunjincho/rebuttal_kdd/exposure_gain/get_graph_multi_metric.py",
        "trend_rows": int(len(trends)),
    }
    atomic_json(metadata_path, metadata)

    print_summary(summary)
    print(f"[SAVED] {summary_path}")
    print(f"[SAVED] {diagnostics_path}")
    print(f"[SAVED] {matched_path}")
    print(f"[SAVED] {posting_path}")
    if not args.no_plot:
        print(f"[SAVED] {plot_path}")
    print(f"[SAVED] {metadata_path}")


if __name__ == "__main__":
    main()
