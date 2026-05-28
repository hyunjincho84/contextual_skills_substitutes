# -*- coding: utf-8 -*-

import argparse
import glob
import os
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

BASE_DATA_DIR = os.environ.get("BASE_DATA_DIR", "/home/jovyan/LEM_data2/data")
DEFAULT_QWEN_ROOT = os.path.join(BASE_DATA_DIR, "qwen25_3b_eval_full")
DEFAULT_GEMMA_ROOT = os.path.join(BASE_DATA_DIR, "gemma2_2b_eval_full")

BASELINE_DIR_TO_NAME = {
    "bert": "bert",
    "cond": "conditional",
    "skill2_vec": "w2v",
}


def list_result_files(root: str) -> List[str]:
    patterns = [
        os.path.join(root, "*", "*.csv.gz"),
        os.path.join(root, "*", "*", "*.csv.gz"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    return sorted(set(files))


def baseline_from_path(fp: str, root: str) -> str:
    rel_parts = os.path.relpath(fp, root).split(os.sep)
    dirname = rel_parts[0] if rel_parts else os.path.basename(os.path.dirname(fp))
    return BASELINE_DIR_TO_NAME.get(dirname, dirname)


def summarize_judge(judge_name: str, root: str) -> List[Dict[str, object]]:
    score_col = f"sv_{judge_name}"
    files = list_result_files(root)
    if not files:
        print(f"[WARN] No files found for {judge_name}: {root}")
        return []

    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    file_counts: Dict[str, int] = {}

    for fp in tqdm(files, desc=f"[{judge_name}] mean files", unit="file"):
        baseline = baseline_from_path(fp, root)
        try:
            df = pd.read_csv(fp, usecols=[score_col])
        except ValueError:
            print(f"[WARN] Missing column {score_col}, skip: {fp}")
            continue

        vals = pd.to_numeric(df[score_col], errors="coerce").dropna()
        sums[baseline] = sums.get(baseline, 0.0) + float(vals.sum())
        counts[baseline] = counts.get(baseline, 0) + int(len(vals))
        file_counts[baseline] = file_counts.get(baseline, 0) + 1

    rows = []
    for baseline in sorted(counts):
        total_count = counts[baseline]
        mean_value = sums[baseline] / total_count if total_count else float("nan")
        rows.append(
            {
                "judge": judge_name,
                "baseline": baseline,
                "file_count": file_counts.get(baseline, 0),
                "row_count": total_count,
                "mean": mean_value,
            }
        )
    return rows


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-root", default=DEFAULT_QWEN_ROOT)
    ap.add_argument("--qwen-judge", default="qwen25")
    ap.add_argument("--gemma-root", default=DEFAULT_GEMMA_ROOT)
    ap.add_argument("--gemma-judge", default="gemma")
    ap.add_argument("--output-csv", default=None)
    return ap.parse_args()


def main():
    args = parse_args()

    rows = []
    rows.extend(summarize_judge(args.qwen_judge, args.qwen_root))
    rows.extend(summarize_judge(args.gemma_judge, args.gemma_root))

    if not rows:
        raise FileNotFoundError("No valid result files found for qwen or gemma.")

    out_df = pd.DataFrame(rows)

    print("\n==============================")
    print("MEAN SV BY JUDGE / BASELINE")
    print("==============================")
    for row in rows:
        print(
            f"{row['judge']}\t{row['baseline']}\t"
            f"files={row['file_count']}\trows={row['row_count']}\tmean={row['mean']}"
        )

    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        out_df.to_csv(args.output_csv, index=False)
        print(f"\nSaved summary: {args.output_csv}")


if __name__ == "__main__":
    main()
