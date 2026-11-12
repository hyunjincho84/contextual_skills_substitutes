#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Summarize paired BERT/GPT sample scores for four SV judges.

The input files are produced by ``scoring_gpt.py`` (BERT judge) and
``scoring_gpt_other_models.py`` (LLaMA, Gemma, and Qwen judges). For each
judge, BERT and GPT rows are aligned by the same identity used by
``get_mean.py``. Summary statistics are computed only on rows for which both
scores are finite, so every BERT/GPT comparison uses an identical sample.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


BASE_DATA_DIR = os.environ.get(
    "BASE_DATA_DIR", "/home/jovyan/LEM_data3/hyunjincho/LEM"
)
DEFAULT_SAMPLE_ROOT = os.environ.get(
    "GPT_SAMPLINGS_DIR",
    os.path.join(BASE_DATA_DIR, "predictions_clean", "gpt_samplings"),
)
DEFAULT_OTHER_MODELS_ROOT = os.environ.get(
    "GPT_OTHER_MODELS_SV_ROOT",
    os.path.join(BASE_DATA_DIR, "evaluation_sv", "gpt_other_models"),
)
DEFAULT_OUTPUT_CSV = os.path.join(
    DEFAULT_OTHER_MODELS_ROOT,
    "gpt_bert_mean_summary_raw.csv",
)
SOURCE_ORDER = ("bert", "gpt")
JUDGE_ORDER = ("llama", "gemma", "qwen", "bert")
IDENTITY_COLS = ["row_idx", "truth", "masked_sentence"]
JUDGE_CONFIGS = {
    "llama": {"file_tag": "llama", "score_col": "sv_llama"},
    "gemma": {"file_tag": "gemma", "score_col": "sv_gemma"},
    "qwen": {"file_tag": "qwen25", "score_col": "sv_qwen25"},
    "bert": {"file_tag": "bert", "score_col": "sv_bert"},
}


class RunningStats:
    """Numerically stable streaming summary, matching ``get_mean.py``."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return
        batch_count = int(values.size)
        batch_mean = float(values.mean())
        batch_m2 = float(np.square(values - batch_mean).sum())
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            delta = batch_mean - self.mean
            combined = self.count + batch_count
            self.mean += delta * batch_count / combined
            self.m2 += (
                batch_m2
                + delta * delta * self.count * batch_count / combined
            )
            self.count = combined
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))


def default_score_path(
    *,
    judge: str,
    source: str,
    sample_root: str,
    other_models_root: str,
) -> str:
    config = JUDGE_CONFIGS[judge]
    file_tag = str(config["file_tag"])
    if judge == "bert":
        return os.path.join(sample_root, f"{source}_with_sv_bert.csv.gz")
    return os.path.join(
        other_models_root,
        file_tag,
        f"{source}_with_sv_{file_tag}.csv.gz",
    )


def resolve_score_paths(
    args: argparse.Namespace,
    judge: str,
) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    for source in SOURCE_ORDER:
        override = getattr(args, f"{judge}_{source}_file")
        paths[source] = os.path.abspath(
            override
            or default_score_path(
                judge=judge,
                source=source,
                sample_root=args.sample_root,
                other_models_root=args.other_models_root,
            )
        )
    return paths


def _filter_years(
    frame: pd.DataFrame,
    years: Optional[Sequence[str]],
    path: str,
) -> pd.DataFrame:
    if not years:
        return frame
    if "year" not in frame.columns:
        raise ValueError(f"--years requires a year column: {path}")
    wanted = {str(year) for year in years}
    numeric = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    return frame.loc[numeric.astype(str).isin(wanted)].copy()


def _single_metadata_value(
    frame: pd.DataFrame,
    column: str,
    path: str,
) -> str:
    if column not in frame.columns:
        return ""
    values = sorted(
        {
            str(value).strip()
            for value in frame[column].tolist()
            if str(value).strip()
        }
    )
    if len(values) > 1:
        raise ValueError(f"Multiple {column} values in {path}: {values}")
    return values[0] if values else ""


def load_score_frame(
    *,
    path: str,
    source: str,
    judge: str,
    years: Optional[Sequence[str]],
) -> Tuple[pd.DataFrame, str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[{judge}/{source}] score file not found: {path}")

    config = JUDGE_CONFIGS[judge]
    score_col = str(config["score_col"])
    header = pd.read_csv(path, nrows=0)
    required = set(IDENTITY_COLS + [score_col])
    if years:
        required.add("year")
    missing = sorted(required.difference(header.columns))
    if missing:
        raise ValueError(
            f"[{judge}/{source}] missing columns {missing}: {path}"
        )

    optional = [
        column
        for column in (
            "year",
            "prediction_source",
            "judge_name",
            "judge_checkpoint",
        )
        if column in header.columns
    ]
    usecols = list(dict.fromkeys(IDENTITY_COLS + [score_col] + optional))
    frame = pd.read_csv(
        path,
        usecols=usecols,
        keep_default_na=False,
        low_memory=False,
    )
    frame = _filter_years(frame, years, path)

    metadata_source = _single_metadata_value(
        frame, "prediction_source", path
    )
    if metadata_source and metadata_source != source:
        raise ValueError(
            f"[{judge}/{source}] prediction_source={metadata_source!r}: {path}"
        )
    metadata_judge = _single_metadata_value(frame, "judge_name", path)
    expected_judge = str(config["file_tag"])
    if metadata_judge and metadata_judge != expected_judge:
        raise ValueError(
            f"[{judge}/{source}] judge_name={metadata_judge!r}, "
            f"expected {expected_judge!r}: {path}"
        )
    checkpoint = _single_metadata_value(frame, "judge_checkpoint", path)

    result = frame.loc[:, IDENTITY_COLS + [score_col]].copy()
    return result.rename(columns={score_col: f"{source}_sv"}), checkpoint


def match_sources(
    *,
    judge: str,
    frames: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Align BERT and GPT rows exactly, including duplicate occurrences."""
    bert = frames["bert"]
    gpt = frames["gpt"]
    if (
        len(bert) == len(gpt)
        and bert[IDENTITY_COLS].equals(gpt[IDENTITY_COLS])
    ):
        matched = bert[IDENTITY_COLS + ["bert_sv"]].copy()
        matched["gpt_sv"] = gpt["gpt_sv"].to_numpy()
        return matched

    match_cols = IDENTITY_COLS + ["_identity_occurrence"]
    working: Dict[str, pd.DataFrame] = {}
    for source in SOURCE_ORDER:
        frame = frames[source].copy()
        frame["_identity_occurrence"] = frame.groupby(
            IDENTITY_COLS,
            sort=False,
            dropna=False,
        ).cumcount()
        working[source] = frame

    matched = working["bert"].merge(
        working["gpt"][match_cols + ["gpt_sv"]],
        on=match_cols,
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    source_counts = {source: len(frames[source]) for source in SOURCE_ORDER}
    if any(len(matched) != count for count in source_counts.values()):
        raise ValueError(
            f"[{judge}] BERT/GPT identity mismatch: "
            f"matched={len(matched):,}, source_counts={source_counts}"
        )
    return matched


def compute_paired_stats(
    matched: pd.DataFrame,
) -> Tuple[Dict[str, RunningStats], int]:
    scores = {
        source: pd.to_numeric(
            matched[f"{source}_sv"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        for source in SOURCE_ORDER
    }
    finite_inputs = np.logical_and.reduce(
        [np.isfinite(scores[source]) for source in SOURCE_ORDER]
    )
    excluded = int((~finite_inputs).sum())
    stats = {source: RunningStats() for source in SOURCE_ORDER}
    for source in SOURCE_ORDER:
        stats[source].update(scores[source][finite_inputs])
    if not stats["bert"].count:
        raise ValueError("No finite paired BERT/GPT scores were found.")
    return stats, excluded


def summary_values(stats: RunningStats) -> Dict[str, float]:
    if stats.count > 1:
        variance = stats.m2 / (stats.count - 1)
        std = math.sqrt(max(variance, 0.0))
        se = std / math.sqrt(stats.count)
        ci95_lower = stats.mean - 1.96 * se
        ci95_upper = stats.mean + 1.96 * se
    else:
        variance = std = se = ci95_lower = ci95_upper = float("nan")
    return {
        "variance": variance,
        "std": std,
        "se": se,
        "ci95_lower": ci95_lower,
        "ci95_upper": ci95_upper,
    }


def make_summary_row(
    *,
    judge: str,
    source: str,
    stats: RunningStats,
    score_path: str,
    score_root: str,
    total_aligned_rows: int,
    excluded_nonfinite_input_rows: int,
    years: Optional[Sequence[str]],
    checkpoint: str,
    mean_difference: float,
) -> Dict[str, object]:
    intervals = summary_values(stats)
    return {
        "evaluation": judge,
        "model": source,
        "normalization_model": "none",
        "file_count": 1,
        "N": stats.count,
        "mean": stats.mean,
        "variance": intervals["variance"],
        "std": intervals["std"],
        "se": intervals["se"],
        "ci95_lower": intervals["ci95_lower"],
        "ci95_upper": intervals["ci95_upper"],
        "95% CI": (
            f"[{intervals['ci95_lower']:.6f}, "
            f"{intervals['ci95_upper']:.6f}]"
        ),
        "min": stats.minimum,
        "max": stats.maximum,
        "score_column": JUDGE_CONFIGS[judge]["score_col"],
        "normalization": "raw_sv",
        "total_aligned_rows": total_aligned_rows,
        "excluded_nonfinite_input_rows": excluded_nonfinite_input_rows,
        "excluded_zero_or_small_denominator_rows": 0,
        "excluded_nonfinite_ratio_rows": 0,
        "min_abs_denominator": 0.0,
        "identity_columns": "|".join(IDENTITY_COLS),
        "distribution_plot": "",
        "evaluation_set": "paired_bert_gpt_sampling_rows",
        "selected_months": "",
        "selected_years": "|".join(years or []),
        "score_data_dir": os.path.dirname(os.path.abspath(score_path)),
        "score_data_root": os.path.abspath(score_root),
        "input_file": os.path.abspath(score_path),
        "judge_file_tag": JUDGE_CONFIGS[judge]["file_tag"],
        "judge_checkpoint": checkpoint,
        "paired_mean_difference_gpt_minus_bert": mean_difference,
    }


def print_summary(row: Dict[str, object]) -> None:
    print(f"\n===== [{row['evaluation']} / {row['model']}] RAW SV =====")
    print(f"Matched valid rows    : {row['N']:,}")
    print(f"Mean raw score        : {row['mean']:.6f}")
    print(f"Variance              : {row['variance']:.6f}")
    print(f"Standard deviation    : {row['std']:.6f}")
    print(f"Standard error        : {row['se']:.6f}")
    print(f"95% CI                : {row['95% CI']}")
    print(f"Full minimum          : {row['min']:.6f}")
    print(f"Full maximum          : {row['max']:.6f}")


def save_summary(rows: List[Dict[str, object]], output_csv: str) -> None:
    output_path = os.path.abspath(output_csv)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary = f"{output_path}.tmp.{os.getpid()}"
    try:
        pd.DataFrame(rows).to_csv(temporary, index=False)
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute paired raw-SV summaries for BERT and GPT predictions "
            "under LLaMA, Gemma, Qwen, and BERT judges."
        )
    )
    parser.add_argument(
        "--judges",
        nargs="+",
        choices=JUDGE_ORDER,
        default=list(JUDGE_ORDER),
        help="Judges to summarize (default: llama gemma qwen bert).",
    )
    parser.add_argument("--sample-root", default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument(
        "--other-models-root", default=DEFAULT_OTHER_MODELS_ROOT
    )
    parser.add_argument(
        "--years",
        nargs="*",
        default=None,
        help="Optional year filter, for example: --years 2018 2019",
    )
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)

    for judge in JUDGE_ORDER:
        for source in SOURCE_ORDER:
            parser.add_argument(
                f"--{judge}-{source}-file",
                dest=f"{judge}_{source}_file",
                default=None,
                help=f"Override the {judge} judge / {source} score file.",
            )

    args = parser.parse_args()
    if len(set(args.judges)) != len(args.judges):
        parser.error("--judges must not contain duplicates")
    return args


def main() -> None:
    args = parse_args()
    print(f"Selected judges : {', '.join(args.judges)}")
    print("Models          : bert, gpt")
    print("Normalization   : none (raw SV)")
    print(f"Identity columns: {IDENTITY_COLS}")
    print("Finite filter   : keep a row only when both BERT and GPT are finite")

    all_rows: List[Dict[str, object]] = []
    for judge in args.judges:
        paths = resolve_score_paths(args, judge)
        score_root = (
            args.sample_root if judge == "bert" else args.other_models_root
        )
        print(f"\n================ [{judge}] ================")
        print(f"BERT score file: {paths['bert']}")
        print(f"GPT score file : {paths['gpt']}")
        print(f"SV column      : {JUDGE_CONFIGS[judge]['score_col']}")

        frames: Dict[str, pd.DataFrame] = {}
        checkpoints: Dict[str, str] = {}
        for source in SOURCE_ORDER:
            frames[source], checkpoints[source] = load_score_frame(
                path=paths[source],
                source=source,
                judge=judge,
                years=args.years,
            )
        nonempty_checkpoints = {
            value for value in checkpoints.values() if value
        }
        if len(nonempty_checkpoints) > 1:
            raise ValueError(
                f"[{judge}] judge checkpoint mismatch: {checkpoints}"
            )
        checkpoint = next(iter(nonempty_checkpoints), "")

        matched = match_sources(judge=judge, frames=frames)
        stats, excluded = compute_paired_stats(matched)
        mean_difference = stats["gpt"].mean - stats["bert"].mean
        rows = [
            make_summary_row(
                judge=judge,
                source=source,
                stats=stats[source],
                score_path=paths[source],
                score_root=score_root,
                total_aligned_rows=len(matched),
                excluded_nonfinite_input_rows=excluded,
                years=args.years,
                checkpoint=checkpoint,
                mean_difference=mean_difference,
            )
            for source in SOURCE_ORDER
        ]
        for row in rows:
            print_summary(row)

        print(f"\n===== [{judge}] PAIRED COMPARISON =====")
        print(f"Total aligned rows             : {len(matched):,}")
        print(f"Excluded non-finite input rows: {excluded:,}")
        print(f"Paired valid rows              : {stats['bert'].count:,}")
        print(f"BERT mean                      : {stats['bert'].mean:.6f}")
        print(f"GPT mean                       : {stats['gpt'].mean:.6f}")
        print(f"Delta (GPT - BERT)             : {mean_difference:.6f}")
        all_rows.extend(rows)

    save_summary(all_rows, args.output_csv)
    print(f"\nSaved paired raw summary: {os.path.abspath(args.output_csv)}")


if __name__ == "__main__":
    main()
