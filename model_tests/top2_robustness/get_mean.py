# -*- coding: utf-8 -*-

import argparse
import glob
import os
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

BASE_DATA_DIR = os.environ.get("BASE_DATA_DIR", "/home/jovyan/LEM_data2/data")
DEFAULT_INPUT_GLOB = os.path.join(
    BASE_DATA_DIR,
    "bert_sampling_variants_from_summary",
    "*",
    "sv_summary_llama_all_methods_*.csv.gz",
)

SAMPLING_METHODS = [
    "sample_top2_like",
    "sample_top10",
    "sample_thresh_0005_sampling",
    "sample_thresh_0001_sampling",
    "sample_temp_15",
    "sample_temp_20",
]


def sv_col(method_name: str) -> str:
    return f"{method_name}__sv_llama"


def detect_methods(files: List[str]) -> List[str]:
    methods = []
    seen = set()

    for fp in tqdm(files, desc="Detecting methods", unit="file"):
        header = pd.read_csv(fp, nrows=0)
        for col in header.columns:
            if not col.endswith("__sv_llama"):
                continue
            method = col[: -len("__sv_llama")]
            if method not in seen:
                seen.add(method)
                methods.append(method)

    ordered = [m for m in SAMPLING_METHODS if m in seen]
    ordered.extend([m for m in methods if m not in set(ordered)])
    return ordered


def summarize_file(fp: str, methods: List[str]) -> Tuple[int, Dict[str, Tuple[int, float]]]:
    header = pd.read_csv(fp, nrows=0)
    score_cols = [sv_col(m) for m in methods if sv_col(m) in header.columns]

    if not score_cols:
        return 0, {m: (0, 0.0) for m in methods}

    df = pd.read_csv(fp, usecols=score_cols)
    stats = {}
    for method in methods:
        col = sv_col(method)
        if col not in df.columns:
            stats[method] = (0, 0.0)
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        stats[method] = (int(vals.shape[0]), float(vals.sum()))

    return len(df), stats


def print_summary(label: str, total_rows: int, counts: Dict[str, int], sums: Dict[str, float]) -> None:
    print(f"[{label}] rows={total_rows:,}")
    for method in counts:
        if counts[method] == 0:
            mean_text = "nan"
        else:
            mean_text = f"{sums[method] / counts[method]:.6f}"
        print(
            f"[{label}] {method}: "
            f"mean_sv_llama={mean_text}, valid_rows={counts[method]:,}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Print row counts and mean SV scores for sampling methods."
    )
    parser.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Sampling method names. Defaults to detected *_sv_llama columns.",
    )
    parser.add_argument(
        "--by-file",
        action="store_true",
        help="Also print per-file summaries before the final total.",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.input_glob}")

    methods = list(args.methods) if args.methods else detect_methods(files)
    if not methods:
        raise ValueError("No *__sv_llama score columns found.")

    total_rows = 0
    total_counts = {m: 0 for m in methods}
    total_sums = {m: 0.0 for m in methods}

    print(f"Found files: {len(files):,}")
    print(f"Methods: {methods}")

    for fp in tqdm(files, desc="Reading score files", unit="file"):
        row_count, file_stats = summarize_file(fp, methods)
        total_rows += row_count

        file_counts = {}
        file_sums = {}
        for method in methods:
            count, score_sum = file_stats[method]
            total_counts[method] += count
            total_sums[method] += score_sum
            file_counts[method] = count
            file_sums[method] = score_sum

        if args.by_file:
            print_summary(os.path.basename(fp), row_count, file_counts, file_sums)

    print_summary("TOTAL", total_rows, total_counts, total_sums)


if __name__ == "__main__":
    main()
