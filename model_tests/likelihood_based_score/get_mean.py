# -*- coding: utf-8 -*-
"""Summarize raw SV scores on the pre-cleaned evaluation sets.

Unlike ``get_score_mean_w2v_one.py``, this script does not normalize scores.
It aligns BERT, Skill2Vec, and conditional-probability rows month by month and
reports the same summary statistics and audit columns for their raw evaluator
scores. Supported evaluators are LLaMA, BERT, Qwen, and Gemma.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Self-contained helpers for locating, aligning, and summarizing score files.
MODEL_NAMES = ("bert", "w2v", "conditional")
INTERNAL_TO_CLI_MODEL = {
    "bert": "bert",
    "w2v": "w2v",
    "conditional": "condprob",
}
IDENTITY_COLS = ["row_idx", "truth", "masked_sentence"]


class RunningStats:
    """Numerically stable streaming summary for score arrays."""

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
            self.m2 += batch_m2 + delta * delta * self.count * batch_count / combined
            self.count = combined
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))


def _parse_month(path: str, model: str, file_tag: str) -> Optional[str]:
    pattern = (
        rf"^sv_summary_{re.escape(file_tag)}_full_{re.escape(model)}_"
        rf"(\d{{4}}-\d{{2}})\.csv(?:\.gz)?$"
    )
    match = re.match(pattern, os.path.basename(path))
    return match.group(1) if match else None


def list_monthly_files(
    directory: str,
    model: str,
    file_tag: str,
    years: Optional[Sequence[str]],
    months: Optional[Sequence[str]],
) -> Dict[str, str]:
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"[{model}] directory not found: {directory}")
    year_filter = {str(year) for year in years} if years else None
    month_filter = {str(month) for month in months} if months else None
    pattern = os.path.join(
        directory, "*", f"sv_summary_{file_tag}_full_{model}_*.csv*"
    )
    result: Dict[str, str] = {}
    for score_path in sorted(glob.glob(pattern)):
        month = _parse_month(score_path, model, file_tag)
        if month is None:
            continue
        if year_filter is not None and month[:4] not in year_filter:
            continue
        if month_filter is not None and month not in month_filter:
            continue
        if month in result:
            raise ValueError(
                f"[{model}] duplicate monthly output for {month}: "
                f"{result[month]} and {score_path}"
            )
        result[month] = score_path
    if not result:
        raise FileNotFoundError(f"[{model}] no selected score files matching {pattern}")
    return result


def require_identical_months(
    model_files: Dict[str, Dict[str, str]], *, allow_incomplete: bool
) -> List[str]:
    month_sets = {model: set(files) for model, files in model_files.items()}
    common = set.intersection(*month_sets.values())
    union = set.union(*month_sets.values())
    if not common:
        raise FileNotFoundError("No common months across selected model outputs.")
    if common != union:
        details = "; ".join(
            f"{model} missing {sorted(union - available)}"
            for model, available in month_sets.items()
            if union - available
        )
        if not allow_incomplete:
            raise ValueError(f"Monthly outputs differ across models: {details}")
        print(f"[WARN] Using common months only; {details}")
    return sorted(common)


def validate_headers(
    model_files: Dict[str, Dict[str, str]], months: Sequence[str], sv_col: str
) -> None:
    required = set(IDENTITY_COLS + [sv_col])
    for model, files in model_files.items():
        for month in months:
            score_path = files[month]
            missing = sorted(required - set(pd.read_csv(score_path, nrows=0).columns))
            if missing:
                raise ValueError(
                    f"[{model} / {month}] missing columns {missing}: {score_path}"
                )


def _load_score_frame(path: str, model: str, sv_col: str) -> pd.DataFrame:
    required = IDENTITY_COLS + [sv_col]
    try:
        frame = pd.read_csv(
            path, usecols=required, keep_default_na=False, low_memory=False
        )
    except ValueError as exc:
        raise ValueError(f"[{model}] missing required columns in {path}") from exc
    return frame.rename(columns={sv_col: f"{model}_sv"})


def match_month(
    *, model_files: Dict[str, Dict[str, str]], month: str, sv_col: str
) -> pd.DataFrame:
    frames = {
        model: _load_score_frame(model_files[model][month], model, sv_col)
        for model in MODEL_NAMES
    }
    reference = frames["bert"]
    same_order = all(
        len(reference) == len(frames[model])
        and reference[IDENTITY_COLS].equals(frames[model][IDENTITY_COLS])
        for model in MODEL_NAMES[1:]
    )
    if same_order:
        matched = reference[IDENTITY_COLS + ["bert_sv"]].copy()
        for model in MODEL_NAMES[1:]:
            matched[f"{model}_sv"] = frames[model][f"{model}_sv"].to_numpy()
        return matched

    match_cols = IDENTITY_COLS + ["_identity_occurrence"]
    source_counts = {model: len(frame) for model, frame in frames.items()}
    for frame in frames.values():
        frame["_identity_occurrence"] = frame.groupby(
            IDENTITY_COLS, sort=False, dropna=False
        ).cumcount()
    matched = frames["bert"]
    for model in MODEL_NAMES[1:]:
        matched = matched.merge(
            frames[model][match_cols + [f"{model}_sv"]],
            on=match_cols,
            how="inner",
            validate="one_to_one",
        )
    if any(len(matched) != count for count in source_counts.values()):
        raise ValueError(
            f"[{month}] cross-model identity mismatch: "
            f"matched={len(matched):,}, source_counts={source_counts}"
        )
    return matched


def save_summary(rows: List[Dict[str, object]], output_csv: str) -> None:
    output_path = os.path.abspath(output_csv)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary_path = f"{output_path}.tmp.{os.getpid()}"
    try:
        pd.DataFrame(rows).to_csv(temporary_path, index=False)
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

BASE_DATA_DIR = os.environ.get("BASE_DATA_DIR", "/home/jovyan/LEM_data3/hyunjincho/LEM")
EVALUATION_ROOT = os.path.join(BASE_DATA_DIR, "evaluation_sv")
EVALUATION_CONFIGS = {
    "llama_clean": {
        "root": os.path.join(EVALUATION_ROOT, "llama_clean"),
        "file_tag": "llama",
        "sv_col": "sv_llama",
        "legacy_prefix": "llama",
    },
    "bert_clean": {
        "root": os.path.join(EVALUATION_ROOT, "bert_clean"),
        "file_tag": "bert",
        "sv_col": "sv_bert",
        "legacy_prefix": "bert",
    },
    "qwen_clean": {
        "root": os.path.join(EVALUATION_ROOT, "qwen_clean"),
        "file_tag": "qwen25",
        "sv_col": "sv_qwen25",
        "legacy_prefix": "qwen",
    },
    "gemma_clean": {
        "root": os.path.join(EVALUATION_ROOT, "gemma_clean"),
        "file_tag": "gemma",
        "sv_col": "sv_gemma",
        "legacy_prefix": "gemma",
    },
}
DEFAULT_OUTPUT_CSV = os.path.join(
    EVALUATION_ROOT,
    "all_evaluations_mean_summary_raw_clean.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute unnormalized raw SV summary statistics for clean "
            "evaluation sets across BERT, Skill2Vec, and CondProb predictions."
        )
    )
    parser.add_argument(
        "--evaluations",
        "--judges",
        dest="evaluations",
        nargs="+",
        choices=tuple(EVALUATION_CONFIGS),
        default=list(EVALUATION_CONFIGS),
        help=(
            "Clean evaluation sets (default: llama_clean bert_clean "
            "qwen_clean gemma_clean)."
        ),
    )
    for evaluation_name, config in EVALUATION_CONFIGS.items():
        cli_prefix = evaluation_name.replace("_", "-")
        legacy_prefix = config["legacy_prefix"]
        root_options = [f"--{cli_prefix}-root"]
        judge_options = [f"--{cli_prefix}-judge"]
        sv_col_options = [f"--{cli_prefix}-sv-col"]
        if legacy_prefix:
            root_options.append(f"--{legacy_prefix}-root")
            judge_options.append(f"--{legacy_prefix}-judge")
            sv_col_options.append(f"--{legacy_prefix}-sv-col")

        parser.add_argument(
            *root_options,
            dest=f"{evaluation_name}_root",
            default=config["root"],
            help=f"Root containing {evaluation_name}/{{bert,w2v,conditional}} scores.",
        )
        parser.add_argument(
            *judge_options,
            dest=f"{evaluation_name}_judge",
            default=config["file_tag"],
            help=f"Filename judge tag (default: {config['file_tag']}).",
        )
        parser.add_argument(
            *sv_col_options,
            dest=f"{evaluation_name}_sv_col",
            default=None,
            help=(
                f"Score column override (default: {config['sv_col']}; "
                "or sv_<judge tag> when the tag is overridden)."
            ),
        )

    parser.add_argument(
        "--sv-col",
        default=None,
        help="Single-evaluation compatibility override for the score column.",
    )
    parser.add_argument(
        "--years",
        nargs="*",
        default=None,
        help="Optional years, for example: --years 2018 2019",
    )
    parser.add_argument(
        "--months",
        nargs="*",
        default=None,
        help="Optional exact months, for example: --months 2018-01 2018-02",
    )
    parser.add_argument(
        "--strict-months",
        action="store_true",
        help="Fail when the three prediction models do not have identical months.",
    )
    parser.add_argument(
        "--bert-pred-dir",
        "--bert_pred_dir",
        dest="bert_pred_dir",
        default=None,
        help="Single-evaluation override for the BERT prediction directory.",
    )
    parser.add_argument(
        "--w2v-pred-dir",
        "--w2v_pred_dir",
        dest="w2v_pred_dir",
        default=None,
        help="Single-evaluation override for the Skill2Vec prediction directory.",
    )
    parser.add_argument(
        "--cond-pred-dir",
        "--cond_pred_dir",
        dest="cond_pred_dir",
        default=None,
        help="Single-evaluation override for the CondProb prediction directory.",
    )
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()

    if len(set(args.evaluations)) != len(args.evaluations):
        parser.error("--evaluations must not contain duplicates")
    single_evaluation_options = (
        args.bert_pred_dir,
        args.w2v_pred_dir,
        args.cond_pred_dir,
        args.sv_col,
    )
    if len(args.evaluations) != 1 and any(single_evaluation_options):
        parser.error(
            "--bert-pred-dir/--w2v-pred-dir/--cond-pred-dir/--sv-col "
            "require exactly one --evaluations value"
        )
    return args


def build_selected_configs(
    args: argparse.Namespace,
) -> Dict[str, Dict[str, object]]:
    selected: Dict[str, Dict[str, object]] = {}
    for evaluation_key in args.evaluations:
        default_config = EVALUATION_CONFIGS[evaluation_key]
        root = str(getattr(args, f"{evaluation_key}_root"))
        file_tag = str(getattr(args, f"{evaluation_key}_judge"))
        explicit_sv_col = getattr(args, f"{evaluation_key}_sv_col")
        if explicit_sv_col:
            sv_col = str(explicit_sv_col)
        elif file_tag == default_config["file_tag"]:
            sv_col = str(default_config["sv_col"])
        else:
            sv_col = f"sv_{file_tag}"

        directory_overrides: Dict[str, Optional[str]] = {
            "bert": None,
            "w2v": None,
            "conditional": None,
        }
        if len(args.evaluations) == 1:
            directory_overrides = {
                "bert": args.bert_pred_dir,
                "w2v": args.w2v_pred_dir,
                "conditional": args.cond_pred_dir,
            }
            sv_col = args.sv_col or sv_col

        selected[evaluation_key] = {
            "root": root,
            "file_tag": file_tag,
            "sv_col": sv_col,
            "directory_overrides": directory_overrides,
        }
    return selected


def prepare_files(
    *,
    root: str,
    file_tag: str,
    sv_col: str,
    years: Optional[Sequence[str]],
    months: Optional[Sequence[str]],
    allow_incomplete: bool,
    directory_overrides: Dict[str, Optional[str]],
) -> Tuple[Dict[str, Dict[str, str]], List[str], Dict[str, str]]:
    score_dirs = {
        "bert": directory_overrides["bert"] or os.path.join(root, "bert"),
        "w2v": directory_overrides["w2v"] or os.path.join(root, "w2v"),
        "conditional": (
            directory_overrides["conditional"] or os.path.join(root, "conditional")
        ),
    }
    model_files = {
        model: list_monthly_files(
            score_dirs[model],
            model,
            file_tag,
            years,
            months,
        )
        for model in MODEL_NAMES
    }
    selected_months = require_identical_months(
        model_files,
        allow_incomplete=allow_incomplete,
    )
    validate_headers(model_files, selected_months, sv_col)
    return model_files, selected_months, score_dirs


def compute_raw_stats(
    *,
    evaluation_key: str,
    model_files: Dict[str, Dict[str, str]],
    months: Sequence[str],
    sv_col: str,
) -> Tuple[Dict[str, RunningStats], int, int]:
    stats = {model: RunningStats() for model in MODEL_NAMES}
    total_aligned_rows = 0
    excluded_nonfinite_input_rows = 0

    for month in tqdm(months, desc=f"[{evaluation_key}] raw SV", unit="month"):
        matched = match_month(
            model_files=model_files,
            month=month,
            sv_col=sv_col,
        )
        total_aligned_rows += len(matched)
        scores = {
            model: pd.to_numeric(
                matched[f"{model}_sv"],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            for model in MODEL_NAMES
        }
        finite_inputs = np.logical_and.reduce(
            [np.isfinite(scores[model]) for model in MODEL_NAMES]
        )
        excluded_nonfinite_input_rows += int((~finite_inputs).sum())
        for model in MODEL_NAMES:
            stats[model].update(scores[model][finite_inputs])
        print(
            f"[{month}] aligned={len(matched):,}, "
            f"non-finite excluded={int((~finite_inputs).sum()):,}"
        )

    if not total_aligned_rows:
        raise ValueError("No aligned rows were found.")
    if not stats["bert"].count:
        raise ValueError("No finite aligned raw scores were found.")
    return stats, total_aligned_rows, excluded_nonfinite_input_rows


def make_summary_row(
    *,
    evaluation_key: str,
    stats: RunningStats,
    model: str,
    file_count: int,
    total_aligned_rows: int,
    excluded_nonfinite_input_rows: int,
    sv_col: str,
    months: Sequence[str],
    score_dir: str,
    score_root: str,
) -> Dict[str, object]:
    if stats.count > 1:
        variance = stats.m2 / (stats.count - 1)
        std = math.sqrt(max(variance, 0.0))
        se = std / math.sqrt(stats.count)
        ci95_lower = stats.mean - 1.96 * se
        ci95_upper = stats.mean + 1.96 * se
    else:
        variance = std = se = ci95_lower = ci95_upper = float("nan")

    return {
        "evaluation": evaluation_key,
        "model": INTERNAL_TO_CLI_MODEL[model],
        "normalization_model": "none",
        "file_count": file_count,
        "N": stats.count,
        "mean": stats.mean,
        "variance": variance,
        "std": std,
        "se": se,
        "ci95_lower": ci95_lower,
        "ci95_upper": ci95_upper,
        "95% CI": f"[{ci95_lower:.6f}, {ci95_upper:.6f}]",
        "min": stats.minimum,
        "max": stats.maximum,
        "score_column": sv_col,
        "normalization": "raw_sv",
        "total_aligned_rows": total_aligned_rows,
        "excluded_nonfinite_input_rows": excluded_nonfinite_input_rows,
        "excluded_zero_or_small_denominator_rows": 0,
        "excluded_nonfinite_ratio_rows": 0,
        "min_abs_denominator": 0.0,
        "identity_columns": "|".join(IDENTITY_COLS),
        "distribution_plot": "",
        "evaluation_set": "precleaned_aligned_rows_no_additional_deduplication",
        "selected_months": "|".join(months),
        "score_data_dir": os.path.abspath(score_dir),
        "score_data_root": os.path.abspath(score_root),
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


def main() -> None:
    args = parse_args()
    configs = build_selected_configs(args)

    prepared = {}
    for evaluation_key, config in configs.items():
        root = os.path.abspath(str(config["root"]))
        directory_overrides = config["directory_overrides"]
        assert isinstance(directory_overrides, dict)
        model_files, months, score_dirs = prepare_files(
            root=root,
            file_tag=str(config["file_tag"]),
            sv_col=str(config["sv_col"]),
            years=args.years,
            months=args.months,
            allow_incomplete=not args.strict_months,
            directory_overrides=directory_overrides,
        )
        prepared[evaluation_key] = (config, model_files, months, score_dirs)

    print(f"Selected evaluations : {', '.join(args.evaluations)}")
    print("Normalization        : none (raw SV)")

    all_rows: List[Dict[str, object]] = []
    for evaluation_key in args.evaluations:
        config, model_files, months, score_dirs = prepared[evaluation_key]
        root = os.path.abspath(str(config["root"]))
        sv_col = str(config["sv_col"])

        print(f"\n================ [{evaluation_key}] ================")
        print(f"Filename judge tag   : {config['file_tag']}")
        print(f"Common monthly files : {len(months)}")
        print(f"Month range          : {months[0]} .. {months[-1]}")
        print(f"Identity columns     : {IDENTITY_COLS}")
        print(f"SV column            : {sv_col}")
        print("Input filtering      : none (uses the pre-cleaned files as-is)")

        stats, total_aligned_rows, excluded_nonfinite_input_rows = compute_raw_stats(
            evaluation_key=evaluation_key,
            model_files=model_files,
            months=months,
            sv_col=sv_col,
        )
        rows = [
            make_summary_row(
                evaluation_key=evaluation_key,
                stats=stats[model],
                model=model,
                file_count=len(months),
                total_aligned_rows=total_aligned_rows,
                excluded_nonfinite_input_rows=excluded_nonfinite_input_rows,
                sv_col=sv_col,
                months=months,
                score_dir=score_dirs[model],
                score_root=root,
            )
            for model in MODEL_NAMES
        ]
        for row in rows:
            print_summary(row)

        print(f"\n===== [{evaluation_key}] MATCH AUDIT =====")
        print(f"Total aligned rows             : {total_aligned_rows:,}")
        print(
            "Excluded non-finite input rows: "
            f"{excluded_nonfinite_input_rows:,}"
        )
        all_rows.extend(rows)

    save_summary(all_rows, args.output_csv)
    print(f"\nSaved combined raw summary: {args.output_csv}")


if __name__ == "__main__":
    main()
