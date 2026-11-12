#!/usr/bin/env python3
"""Run paired t-tests for GPT/BERT samples or the legacy Qwen analysis.

The default ``gpt_bert`` task aligns the 16,000 sampled rows underlying
``gpt_bert_mean_summary_raw.csv`` and tests the within-row difference
``Ours (BERT predictions) - LLM (GPT predictions)``.  Use
``--task qwen_clean`` to run the original Ours-versus-Conditional Qwen test.
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
from scipy import stats
from tqdm import tqdm


BASE_DATA_DIR = os.environ.get(
    "BASE_DATA_DIR", "/home/jovyan/LEM_data3/hyunjincho/LEM"
)
DEFAULT_QWEN_ROOT = os.path.join(BASE_DATA_DIR, "evaluation_sv", "qwen")
DEFAULT_CLEAN_PRED_DIR = os.path.join(
    BASE_DATA_DIR, "predictions_clean", "bert_pred", "pred"
)
EXPECTED_FULL_CLEAN_ROWS = 4_492_710
DEFAULT_OUTPUT_CSV = os.path.join(
    DEFAULT_QWEN_ROOT,
    "paired_t_test_ours_vs_conditional_test_clean.csv",
)

DEFAULT_GPT_SAMPLE_ROOT = os.path.join(
    BASE_DATA_DIR, "predictions_clean", "gpt_samplings"
)
DEFAULT_GPT_OTHER_MODELS_ROOT = os.path.join(
    BASE_DATA_DIR, "evaluation_sv", "gpt_other_models"
)
DEFAULT_GPT_BERT_SUMMARY = os.path.join(
    DEFAULT_GPT_OTHER_MODELS_ROOT, "gpt_bert_mean_summary_raw.csv"
)
GPT_BERT_IDENTITY_COLS = ["row_idx", "truth", "masked_sentence"]
GPT_BERT_JUDGE_CONFIGS = {
    "llama": {"file_tag": "llama", "score_col": "sv_llama"},
    "gemma": {"file_tag": "gemma", "score_col": "sv_gemma"},
    "qwen": {"file_tag": "qwen25", "score_col": "sv_qwen25"},
    "bert": {"file_tag": "bert", "score_col": "sv_bert"},
}

MODEL_NAMES = ("bert", "conditional", "w2v")
IDENTITY_COLS = ["year", "file", "row_idx", "truth", "masked_sentence"]
CLEAN_KEY_COLS = IDENTITY_COLS
HASH_DTYPE = np.dtype([("hash_1", "<u8"), ("hash_2", "<u8")])
HASH_KEYS = ("sentpairhash0001", "sentpairhash0002")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the sampled GPT-versus-BERT paired t-test (default), or the "
            "legacy Qwen Ours-versus-Conditional clean test."
        )
    )
    parser.add_argument(
        "--task",
        choices=("gpt_bert", "qwen_clean"),
        default="gpt_bert",
        help="gpt_bert is the 16,000-row sampled test used by the mean summary.",
    )
    parser.add_argument(
        "--judge",
        choices=tuple(GPT_BERT_JUDGE_CONFIGS),
        default="bert",
        help="SV judge for --task gpt_bert (default: bert).",
    )
    parser.add_argument("--sample-root", default=DEFAULT_GPT_SAMPLE_ROOT)
    parser.add_argument(
        "--other-models-root", default=DEFAULT_GPT_OTHER_MODELS_ROOT
    )
    parser.add_argument(
        "--summary-csv",
        default=DEFAULT_GPT_BERT_SUMMARY,
        help="Mean summary used to verify N and source means before testing.",
    )
    parser.add_argument(
        "--skip-summary-check",
        action="store_true",
        help="Skip validation against --summary-csv.",
    )
    parser.add_argument(
        "--ours-file",
        default=None,
        help="Override the BERT-prediction score file for --task gpt_bert.",
    )
    parser.add_argument(
        "--llm-file",
        default=None,
        help="Override the GPT-prediction score file for --task gpt_bert.",
    )
    parser.add_argument("--qwen-root", default=DEFAULT_QWEN_ROOT)
    parser.add_argument(
        "--clean-pred-dir",
        default=DEFAULT_CLEAN_PRED_DIR,
        help=(
            "Monthly predictions_clean BERT files used as the exact test-clean "
            "row reference."
        ),
    )
    parser.add_argument(
        "--no-clean-filter",
        action="store_true",
        help="Disable test-clean filtering and use all selected Qwen rows.",
    )
    parser.add_argument(
        "--expected-clean-rows",
        type=int,
        default=EXPECTED_FULL_CLEAN_ROWS,
        help=(
            "Expected reference/valid row count for a full clean run; "
            "0 disables the check."
        ),
    )
    parser.add_argument("--ours-dir", default=None)
    parser.add_argument("--conditional-dir", default=None)
    parser.add_argument("--w2v-dir", default=None)
    parser.add_argument("--judge-tag", default="qwen25")
    parser.add_argument("--score-col", default="sv_qwen25")
    parser.add_argument(
        "--score-mode",
        choices=("w2v_ratio", "raw"),
        default="w2v_ratio",
        help=(
            "w2v_ratio tests the values shown in the W2V=1 table; raw tests "
            "the original Qwen SV scores."
        ),
    )
    parser.add_argument(
        "--unit",
        choices=("masked_sentence", "row"),
        default="row",
        help=(
            "masked_sentence averages duplicate exact sentences before the "
            "test; row treats every test-clean row as one pair (default)."
        ),
    )
    parser.add_argument("--years", nargs="*", default=None)
    parser.add_argument("--months", nargs="*", default=None)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Use common months instead of failing when model months differ.",
    )
    parser.add_argument(
        "--min-abs-w2v",
        type=float,
        default=0.0,
        help="For w2v_ratio, exclude abs(w2v) <= this value.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV; a task-specific default is used when omitted.",
    )
    parser.add_argument(
        "--audit-csv",
        default=None,
        help="Optional monthly matching/filter audit CSV.",
    )
    args = parser.parse_args()
    if args.output_csv is None:
        if args.task == "gpt_bert":
            args.output_csv = os.path.join(
                DEFAULT_GPT_OTHER_MODELS_ROOT,
                f"paired_t_test_{args.judge}_judge_ours_vs_llm_raw.csv",
            )
        else:
            args.output_csv = DEFAULT_OUTPUT_CSV

    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must be between 0 and 1")
    if args.min_abs_w2v < 0.0:
        parser.error("--min-abs-w2v must be non-negative")
    if args.expected_clean_rows < 0:
        parser.error("--expected-clean-rows must be non-negative")
    if args.months:
        invalid = [
            month
            for month in args.months
            if re.fullmatch(r"\d{4}-\d{2}", str(month)) is None
        ]
        if invalid:
            parser.error(f"invalid --months values: {invalid}")
    return args


def parse_month(path: str, model: str, judge_tag: str) -> Optional[str]:
    pattern = (
        rf"^sv_summary_{re.escape(judge_tag)}_full_{re.escape(model)}_"
        rf"(\d{{4}}-\d{{2}})\.csv(?:\.gz)?$"
    )
    match = re.match(pattern, os.path.basename(path))
    return match.group(1) if match else None


def list_monthly_files(
    directory: str,
    model: str,
    judge_tag: str,
    years: Optional[Sequence[str]],
    months: Optional[Sequence[str]],
) -> Dict[str, str]:
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"[{model}] directory not found: {directory}")

    year_filter = {str(year) for year in years} if years else None
    month_filter = {str(month) for month in months} if months else None
    paths = sorted(
        glob.glob(
            os.path.join(
                directory,
                "*",
                f"sv_summary_{judge_tag}_full_{model}_*.csv*",
            )
        )
    )
    result: Dict[str, str] = {}
    for path in paths:
        month = parse_month(path, model, judge_tag)
        if month is None:
            continue
        if year_filter is not None and month[:4] not in year_filter:
            continue
        if month_filter is not None and month not in month_filter:
            continue
        if month in result:
            raise ValueError(
                f"[{model}] duplicate monthly output for {month}: "
                f"{result[month]} and {path}"
            )
        result[month] = path
    if not result:
        raise FileNotFoundError(f"[{model}] no selected Qwen files in {directory}")
    return result


def list_clean_prediction_files(
    directory: str,
    years: Optional[Sequence[str]],
    months: Optional[Sequence[str]],
) -> Dict[str, str]:
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"[clean reference] directory not found: {directory}"
        )

    year_filter = {str(year) for year in years} if years else None
    month_filter = {str(month) for month in months} if months else None
    paths = sorted(
        glob.glob(os.path.join(directory, "*", "predictions_*.csv*"))
    )
    result: Dict[str, str] = {}
    pattern = re.compile(r"^predictions_(\d{4}-\d{2})\.csv(?:\.gz)?$")
    for path in paths:
        match = pattern.match(os.path.basename(path))
        if match is None:
            continue
        month = match.group(1)
        if year_filter is not None and month[:4] not in year_filter:
            continue
        if month_filter is not None and month not in month_filter:
            continue
        if month in result:
            raise ValueError(
                f"[clean reference] duplicate file for {month}: "
                f"{result[month]} and {path}"
            )
        result[month] = path
    if not result:
        raise FileNotFoundError(
            f"[clean reference] no selected prediction files in {directory}"
        )
    return result


def select_months(
    model_files: Dict[str, Dict[str, str]], allow_incomplete: bool
) -> List[str]:
    required_models = list(model_files)
    month_sets = {model: set(files) for model, files in model_files.items()}
    common = set.intersection(*(month_sets[model] for model in required_models))
    union = set.union(*(month_sets[model] for model in required_models))
    if not common:
        raise FileNotFoundError("No common months across selected model outputs.")
    if common != union:
        details = "; ".join(
            f"{model} missing {sorted(union - month_sets[model])}"
            for model in required_models
            if union - month_sets[model]
        )
        if not allow_incomplete:
            raise ValueError(
                f"Monthly outputs differ across models: {details}. "
                "Pass --allow-incomplete to use only common months."
            )
        print(f"[WARN] Monthly outputs differ: {details}")
    return sorted(common)


def load_score_frame(path: str, model: str, score_col: str) -> pd.DataFrame:
    required = IDENTITY_COLS + [score_col]
    try:
        frame = pd.read_csv(
            path,
            usecols=required,
            keep_default_na=False,
            low_memory=False,
        )
    except ValueError as exc:
        raise ValueError(f"[{model}] missing required columns in {path}") from exc
    return frame.rename(columns={score_col: f"{model}_sv"})


def identity_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return len(left) == len(right) and left[IDENTITY_COLS].equals(
        right[IDENTITY_COLS]
    )


def align_month(
    paths: Dict[str, str], score_col: str, include_w2v: bool
) -> Tuple[pd.DataFrame, Dict[str, int], str]:
    models = ["bert", "conditional"] + (["w2v"] if include_w2v else [])
    frames = {
        model: load_score_frame(paths[model], model, score_col)
        for model in models
    }
    source_counts = {model: len(frame) for model, frame in frames.items()}

    reference = frames["bert"]
    if all(identity_equal(reference, frames[model]) for model in models[1:]):
        matched = reference[IDENTITY_COLS + ["bert_sv"]].copy()
        for model in models[1:]:
            matched[f"{model}_sv"] = frames[model][f"{model}_sv"].to_numpy()
        return matched, source_counts, "row_order"

    match_cols = IDENTITY_COLS + ["_identity_occurrence"]
    for model, frame in frames.items():
        frame["_identity_occurrence"] = frame.groupby(
            IDENTITY_COLS, sort=False, dropna=False
        ).cumcount()

    matched = frames["bert"]
    for model in models[1:]:
        matched = matched.merge(
            frames[model][match_cols + [f"{model}_sv"]],
            on=match_cols,
            how="inner",
            validate="one_to_one",
        )

    if any(len(matched) != count for count in source_counts.values()):
        raise ValueError(
            "Cross-model identity mismatch: "
            f"matched={len(matched):,}, source_counts={source_counts}"
        )
    return matched, source_counts, "identity_merge"


def filter_to_clean_pairs(
    matched: pd.DataFrame, clean_path: str
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    try:
        clean = pd.read_csv(
            clean_path,
            usecols=CLEAN_KEY_COLS,
            keep_default_na=False,
            low_memory=False,
        )
    except ValueError as exc:
        raise ValueError(
            f"[clean reference] missing columns in {clean_path}: "
            f"{CLEAN_KEY_COLS}"
        ) from exc

    empty_key_rows = int(
        clean["masked_sentence"].eq("").sum() + clean["truth"].eq("").sum()
    )
    if empty_key_rows:
        raise ValueError(
            f"[clean reference] empty pairing keys in {clean_path}: "
            f"{empty_key_rows:,}"
        )

    clean_duplicate_rows = int(
        clean.duplicated(CLEAN_KEY_COLS, keep=False).sum()
    )
    if clean_duplicate_rows:
        raise ValueError(
            f"[clean reference] duplicate identity rows in "
            f"{clean_path}: {clean_duplicate_rows:,}"
        )

    qwen_duplicate_rows_removed = int(
        matched.duplicated(CLEAN_KEY_COLS, keep="first").sum()
    )
    qwen_unique = matched.drop_duplicates(CLEAN_KEY_COLS, keep="first")
    filtered = clean.merge(
        qwen_unique,
        on=CLEAN_KEY_COLS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing_rows = int(filtered["_merge"].ne("both").sum())
    if missing_rows:
        examples = filtered.loc[
            filtered["_merge"].ne("both"), CLEAN_KEY_COLS
        ].head(3)
        raise ValueError(
            f"Qwen outputs do not contain {missing_rows:,} clean rows from "
            f"{clean_path}. Examples: {examples.to_dict('records')}"
        )
    filtered = filtered.drop(columns="_merge")
    return filtered, {
        "clean_reference_rows": len(clean),
        "clean_matched_rows": len(filtered),
        "qwen_duplicate_pair_rows_removed": qwen_duplicate_rows_removed,
    }


def hash_sentences(sentences: pd.Series) -> np.ndarray:
    if bool(sentences.eq("").any()):
        raise ValueError("Encountered an empty masked_sentence pairing unit.")
    frame = sentences.to_frame(name="masked_sentence")
    result = np.empty(len(frame), dtype=HASH_DTYPE)
    for field, key in zip(HASH_DTYPE.names, HASH_KEYS):
        result[field] = pd.util.hash_pandas_object(
            frame,
            index=False,
            hash_key=key,
            categorize=True,
        ).to_numpy(dtype=np.uint64, copy=False)
    return result


def collapse_by_sentence(
    hashes: np.ndarray, ours: np.ndarray, conditional: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int]:
    _, inverse = np.unique(hashes, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    ours_by_sentence = np.bincount(inverse, weights=ours) / counts
    cond_by_sentence = np.bincount(inverse, weights=conditional) / counts
    duplicates_collapsed = int(len(hashes) - len(counts))
    return ours_by_sentence, cond_by_sentence, duplicates_collapsed


def safe_variance(values: np.ndarray) -> float:
    return float(np.var(values, ddof=1)) if len(values) > 1 else float("nan")


def compute_statistics(
    ours: np.ndarray,
    conditional: np.ndarray,
    *,
    alpha: float,
) -> Dict[str, object]:
    if len(ours) != len(conditional):
        raise ValueError("Paired arrays have different lengths.")
    if len(ours) < 2:
        raise ValueError("At least two valid pairs are required.")

    difference = ours - conditional
    n_pairs = int(len(difference))
    degrees_freedom = n_pairs - 1
    mean_ours = float(ours.mean())
    mean_cond = float(conditional.mean())
    mean_difference = float(difference.mean())
    variance_difference = safe_variance(difference)
    std_difference = math.sqrt(max(variance_difference, 0.0))
    se_difference = std_difference / math.sqrt(n_pairs)

    if se_difference == 0.0:
        if mean_difference == 0.0:
            t_statistic = 0.0
            p_two_sided = 1.0
            p_ours_greater = 0.5
            p_ours_less = 0.5
        else:
            t_statistic = math.copysign(math.inf, mean_difference)
            p_two_sided = 0.0
            p_ours_greater = 0.0 if mean_difference > 0 else 1.0
            p_ours_less = 0.0 if mean_difference < 0 else 1.0
        ci_lower = ci_upper = mean_difference
        cohens_dz = math.copysign(math.inf, mean_difference) if mean_difference else 0.0
    else:
        t_statistic = mean_difference / se_difference
        p_two_sided = float(2.0 * stats.t.sf(abs(t_statistic), degrees_freedom))
        p_ours_greater = float(stats.t.sf(t_statistic, degrees_freedom))
        p_ours_less = float(stats.t.cdf(t_statistic, degrees_freedom))
        critical = float(stats.t.ppf(1.0 - alpha / 2.0, degrees_freedom))
        ci_lower = mean_difference - critical * se_difference
        ci_upper = mean_difference + critical * se_difference
        cohens_dz = mean_difference / std_difference

    if ci_lower > 0.0:
        conclusion = "ours_significantly_higher"
    elif ci_upper < 0.0:
        conclusion = "conditional_significantly_higher"
    else:
        conclusion = "no_significant_difference"

    return {
        "n_pairs": n_pairs,
        "mean_ours": mean_ours,
        "variance_ours": safe_variance(ours),
        "std_ours": float(np.std(ours, ddof=1)),
        "mean_conditional": mean_cond,
        "variance_conditional": safe_variance(conditional),
        "std_conditional": float(np.std(conditional, ddof=1)),
        "mean_difference_ours_minus_conditional": mean_difference,
        "variance_difference": variance_difference,
        "std_difference": std_difference,
        "se_difference": se_difference,
        "t_statistic": t_statistic,
        "degrees_freedom": degrees_freedom,
        "p_value_two_sided": p_two_sided,
        "p_value_one_sided_ours_greater": p_ours_greater,
        "p_value_one_sided_ours_less": p_ours_less,
        "ci_level": 1.0 - alpha,
        "ci_lower_difference": ci_lower,
        "ci_upper_difference": ci_upper,
        "cohens_dz": cohens_dz,
        "paired_correlation": float(np.corrcoef(ours, conditional)[0, 1]),
        "ours_higher_pairs": int(np.count_nonzero(difference > 0.0)),
        "conditional_higher_pairs": int(np.count_nonzero(difference < 0.0)),
        "tied_pairs": int(np.count_nonzero(difference == 0.0)),
        "conclusion_at_alpha": conclusion,
    }


def atomic_to_csv(frame: pd.DataFrame, path: str) -> None:
    absolute_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
    temporary_path = f"{absolute_path}.tmp.{os.getpid()}"
    try:
        frame.to_csv(temporary_path, index=False)
        os.replace(temporary_path, absolute_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def gpt_bert_score_paths(args: argparse.Namespace) -> Dict[str, str]:
    config = GPT_BERT_JUDGE_CONFIGS[args.judge]
    file_tag = str(config["file_tag"])
    if args.judge == "bert":
        defaults = {
            source: os.path.join(
                args.sample_root, f"{source}_with_sv_{file_tag}.csv.gz"
            )
            for source in ("bert", "gpt")
        }
    else:
        defaults = {
            source: os.path.join(
                args.other_models_root,
                file_tag,
                f"{source}_with_sv_{file_tag}.csv.gz",
            )
            for source in ("bert", "gpt")
        }
    return {
        "ours": os.path.abspath(args.ours_file or defaults["bert"]),
        "llm": os.path.abspath(args.llm_file or defaults["gpt"]),
    }


def load_gpt_bert_score(
    path: str, label: str, score_col: str
) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[{label}] score file not found: {path}")
    required = GPT_BERT_IDENTITY_COLS + [score_col]
    try:
        frame = pd.read_csv(
            path,
            usecols=required,
            keep_default_na=False,
            low_memory=False,
        )
    except ValueError as exc:
        raise ValueError(
            f"[{label}] missing columns {required} in {path}"
        ) from exc
    return frame.rename(columns={score_col: f"{label}_sv"})


def align_gpt_bert_scores(
    ours: pd.DataFrame, llm: pd.DataFrame
) -> Tuple[pd.DataFrame, str]:
    if (
        len(ours) == len(llm)
        and ours[GPT_BERT_IDENTITY_COLS].equals(
            llm[GPT_BERT_IDENTITY_COLS]
        )
    ):
        matched = ours.copy()
        matched["llm_sv"] = llm["llm_sv"].to_numpy()
        return matched, "row_order"

    occurrence = "_identity_occurrence"
    match_cols = GPT_BERT_IDENTITY_COLS + [occurrence]
    ours = ours.copy()
    llm = llm.copy()
    for frame in (ours, llm):
        frame[occurrence] = frame.groupby(
            GPT_BERT_IDENTITY_COLS, sort=False, dropna=False
        ).cumcount()
    matched = ours.merge(
        llm[match_cols + ["llm_sv"]],
        on=match_cols,
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(matched) != len(ours) or len(matched) != len(llm):
        raise ValueError(
            "Ours/LLM identity mismatch: "
            f"matched={len(matched):,}, ours={len(ours):,}, llm={len(llm):,}"
        )
    return matched, "identity_merge"


def validate_against_mean_summary(
    *,
    summary_path: str,
    judge: str,
    total_aligned_rows: int,
    excluded_nonfinite_rows: int,
    ours: np.ndarray,
    llm: np.ndarray,
) -> None:
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"Mean summary not found: {summary_path}")
    summary = pd.read_csv(summary_path, keep_default_na=False)
    selected = summary.loc[summary["evaluation"].astype(str).eq(judge)]
    rows = {
        model: selected.loc[selected["model"].astype(str).eq(model)]
        for model in ("bert", "gpt")
    }
    if any(len(frame) != 1 for frame in rows.values()):
        raise ValueError(
            f"Expected one summary row per model for judge={judge}: "
            f"counts={{'bert': {len(rows['bert'])}, 'gpt': {len(rows['gpt'])}}}"
        )

    expected_n = len(ours)
    for model, values, label in (
        ("bert", ours, "Ours"),
        ("gpt", llm, "LLM"),
    ):
        row = rows[model].iloc[0]
        if int(row["N"]) != expected_n:
            raise ValueError(
                f"{label} N differs from summary: {expected_n} != {row['N']}"
            )
        if not np.isclose(
            float(values.mean()), float(row["mean"]), rtol=1e-12, atol=1e-12
        ):
            raise ValueError(
                f"{label} mean differs from summary: "
                f"{values.mean():.17g} != {row['mean']}"
            )
        if int(row["total_aligned_rows"]) != total_aligned_rows:
            raise ValueError(f"{label} total_aligned_rows differs from summary")
        if int(row["excluded_nonfinite_input_rows"]) != excluded_nonfinite_rows:
            raise ValueError(f"{label} excluded row count differs from summary")

    difference_gpt_minus_bert = float(llm.mean() - ours.mean())
    recorded_difference = float(
        rows["bert"].iloc[0]["paired_mean_difference_gpt_minus_bert"]
    )
    if not np.isclose(
        difference_gpt_minus_bert,
        recorded_difference,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("Paired mean difference differs from mean summary")


def run_gpt_bert_test(args: argparse.Namespace) -> None:
    config = GPT_BERT_JUDGE_CONFIGS[args.judge]
    score_col = str(config["score_col"])
    paths = gpt_bert_score_paths(args)
    ours_frame = load_gpt_bert_score(paths["ours"], "ours", score_col)
    llm_frame = load_gpt_bert_score(paths["llm"], "llm", score_col)
    matched, alignment_method = align_gpt_bert_scores(ours_frame, llm_frame)

    ours_raw = pd.to_numeric(matched["ours_sv"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    llm_raw = pd.to_numeric(matched["llm_sv"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    finite = np.isfinite(ours_raw) & np.isfinite(llm_raw)
    excluded_nonfinite_rows = int((~finite).sum())
    ours = ours_raw[finite]
    llm = llm_raw[finite]
    if len(ours) < 2:
        raise ValueError("At least two finite paired rows are required.")

    summary_checked = not args.skip_summary_check
    if summary_checked:
        validate_against_mean_summary(
            summary_path=os.path.abspath(args.summary_csv),
            judge=args.judge,
            total_aligned_rows=len(matched),
            excluded_nonfinite_rows=excluded_nonfinite_rows,
            ours=ours,
            llm=llm,
        )

    valid_aligned_rows = len(ours)
    duplicates_collapsed = 0
    if args.unit == "masked_sentence":
        sentence_hashes = hash_sentences(
            matched.loc[finite, "masked_sentence"].reset_index(drop=True)
        )
        ours, llm, duplicates_collapsed = collapse_by_sentence(
            sentence_hashes, ours, llm
        )

    raw = compute_statistics(ours, llm, alpha=args.alpha)
    conclusion = str(raw["conclusion_at_alpha"]).replace(
        "conditional", "llm"
    )
    result = {
        "test": "paired_t_test",
        "judge": args.judge,
        "score_column": score_col,
        "score_mode": "raw_sv",
        "difference_definition": "ours_bert_minus_llm_gpt",
        "pairing_unit": args.unit,
        "n_pairs": raw["n_pairs"],
        "degrees_freedom": raw["degrees_freedom"],
        "mean_ours_bert": raw["mean_ours"],
        "variance_ours_bert": raw["variance_ours"],
        "std_ours_bert": raw["std_ours"],
        "mean_llm_gpt": raw["mean_conditional"],
        "variance_llm_gpt": raw["variance_conditional"],
        "std_llm_gpt": raw["std_conditional"],
        "mean_difference_ours_minus_llm": raw[
            "mean_difference_ours_minus_conditional"
        ],
        "variance_difference": raw["variance_difference"],
        "std_difference": raw["std_difference"],
        "se_difference": raw["se_difference"],
        "t_statistic": raw["t_statistic"],
        "p_value_two_sided": raw["p_value_two_sided"],
        "p_value_one_sided_ours_greater": raw[
            "p_value_one_sided_ours_greater"
        ],
        "p_value_one_sided_llm_greater": raw[
            "p_value_one_sided_ours_less"
        ],
        "ci_level": raw["ci_level"],
        "ci_lower_difference": raw["ci_lower_difference"],
        "ci_upper_difference": raw["ci_upper_difference"],
        "cohens_dz": raw["cohens_dz"],
        "paired_correlation": raw["paired_correlation"],
        "ours_higher_pairs": raw["ours_higher_pairs"],
        "llm_higher_pairs": raw["conditional_higher_pairs"],
        "tied_pairs": raw["tied_pairs"],
        "conclusion_at_alpha": conclusion,
        "alpha": args.alpha,
        "identity_columns_for_alignment": "|".join(
            GPT_BERT_IDENTITY_COLS
        ),
        "alignment_method": alignment_method,
        "ours_source_rows": len(ours_frame),
        "llm_source_rows": len(llm_frame),
        "total_aligned_rows": len(matched),
        "valid_aligned_score_rows": valid_aligned_rows,
        "excluded_nonfinite_rows": excluded_nonfinite_rows,
        "duplicate_sentence_rows_collapsed": duplicates_collapsed,
        "summary_check_passed": summary_checked,
        "summary_csv": (
            os.path.abspath(args.summary_csv) if summary_checked else ""
        ),
        "ours_file": paths["ours"],
        "llm_file": paths["llm"],
    }
    atomic_to_csv(pd.DataFrame([result]), args.output_csv)

    print("================ GPT/BERT PAIRED T-TEST ================")
    print(f"Judge                    : {args.judge} ({score_col})")
    print(f"Pairing unit             : {args.unit}")
    print(f"Difference               : Ours(BERT) - LLM(GPT)")
    print(f"Alignment                : {alignment_method}")
    print(f"Total aligned rows       : {len(matched):,}")
    print(f"Excluded non-finite rows : {excluded_nonfinite_rows:,}")
    print(f"Paired units             : {raw['n_pairs']:,}")
    print(f"Mean Ours (BERT)         : {raw['mean_ours']:.10f}")
    print(f"Mean LLM (GPT)           : {raw['mean_conditional']:.10f}")
    print(
        f"Mean paired difference   : "
        f"{raw['mean_difference_ours_minus_conditional']:.10f}"
    )
    print(
        f"t({raw['degrees_freedom']:,})                 : "
        f"{raw['t_statistic']:.10f}"
    )
    print(f"Two-sided p-value        : {raw['p_value_two_sided']:.12g}")
    print(
        f"{100 * (1 - args.alpha):.1f}% CI of difference     : "
        f"[{raw['ci_lower_difference']:.10f}, "
        f"{raw['ci_upper_difference']:.10f}]"
    )
    print(f"Cohen's dz               : {raw['cohens_dz']:.10f}")
    print(f"Paired correlation       : {raw['paired_correlation']:.10f}")
    print(f"Conclusion               : {conclusion}")
    print(f"Summary check            : {'passed' if summary_checked else 'skipped'}")
    print(f"Output CSV               : {os.path.abspath(args.output_csv)}")


def main() -> None:
    args = parse_args()
    if args.task == "gpt_bert":
        run_gpt_bert_test(args)
        return
    qwen_root = os.path.abspath(args.qwen_root)
    clean_filter_enabled = not args.no_clean_filter
    clean_pred_dir = os.path.abspath(args.clean_pred_dir)
    directories = {
        "bert": os.path.abspath(args.ours_dir or os.path.join(qwen_root, "bert")),
        "conditional": os.path.abspath(
            args.conditional_dir or os.path.join(qwen_root, "conditional")
        ),
    }
    include_w2v = args.score_mode == "w2v_ratio"
    if include_w2v:
        directories["w2v"] = os.path.abspath(
            args.w2v_dir or os.path.join(qwen_root, "w2v")
        )

    model_files = {
        model: list_monthly_files(
            directory,
            model,
            args.judge_tag,
            args.years,
            args.months,
        )
        for model, directory in directories.items()
    }
    selection_files = dict(model_files)
    clean_files: Dict[str, str] = {}
    if clean_filter_enabled:
        clean_files = list_clean_prediction_files(
            clean_pred_dir,
            args.years,
            args.months,
        )
        selection_files["clean_reference"] = clean_files
    months = select_months(selection_files, args.allow_incomplete)

    print("================ QWEN PAIRED T-TEST ================")
    print(f"Score mode       : {args.score_mode}")
    print(f"Pairing unit     : {args.unit}")
    print(f"Difference       : Ours - Conditional")
    print(f"Selected months  : {months[0]} .. {months[-1]} (n={len(months)})")
    print(f"Score column     : {args.score_col}")
    print(
        f"Clean filter     : "
        f"{clean_pred_dir if clean_filter_enabled else 'disabled'}"
    )

    ours_parts: List[np.ndarray] = []
    cond_parts: List[np.ndarray] = []
    sentence_hash_parts: List[np.ndarray] = []
    audit_rows: List[Dict[str, object]] = []
    total_qwen_aligned_rows = 0
    total_clean_reference_rows = 0
    total_aligned_rows = 0
    total_invalid_rows = 0
    total_small_w2v_rows = 0
    total_negative_w2v_rows = 0

    for month in tqdm(months, desc="Qwen paired rows", unit="month"):
        paths = {model: model_files[model][month] for model in model_files}
        matched, source_counts, alignment_method = align_month(
            paths,
            args.score_col,
            include_w2v,
        )
        qwen_aligned_rows = len(matched)
        total_qwen_aligned_rows += qwen_aligned_rows
        clean_audit = {
            "clean_reference_rows": 0,
            "clean_matched_rows": qwen_aligned_rows,
            "qwen_duplicate_pair_rows_removed": 0,
        }
        if clean_filter_enabled:
            matched, clean_audit = filter_to_clean_pairs(
                matched,
                clean_files[month],
            )
            total_clean_reference_rows += clean_audit["clean_reference_rows"]
        aligned_rows = len(matched)
        total_aligned_rows += aligned_rows

        ours_raw = pd.to_numeric(matched["bert_sv"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        cond_raw = pd.to_numeric(
            matched["conditional_sv"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        valid = np.isfinite(ours_raw) & np.isfinite(cond_raw)
        small_w2v_rows = 0
        negative_w2v_rows = 0

        if include_w2v:
            w2v = pd.to_numeric(matched["w2v_sv"], errors="coerce").to_numpy(
                dtype=np.float64
            )
            valid &= np.isfinite(w2v)
            denominator_ok = np.abs(w2v) > args.min_abs_w2v
            small_w2v_rows = int(np.count_nonzero(valid & ~denominator_ok))
            negative_w2v_rows = int(np.count_nonzero(valid & (w2v < 0.0)))
            valid &= denominator_ok
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                ours_score = ours_raw / w2v
                cond_score = cond_raw / w2v
            valid &= np.isfinite(ours_score) & np.isfinite(cond_score)
        else:
            ours_score = ours_raw
            cond_score = cond_raw

        valid_rows = int(valid.sum())
        invalid_rows = int(aligned_rows - valid_rows)
        total_invalid_rows += invalid_rows
        total_small_w2v_rows += small_w2v_rows
        total_negative_w2v_rows += negative_w2v_rows
        ours_parts.append(ours_score[valid])
        cond_parts.append(cond_score[valid])
        if args.unit == "masked_sentence":
            sentence_hash_parts.append(
                hash_sentences(matched.loc[valid, "masked_sentence"])
            )

        audit_rows.append(
            {
                "month": month,
                "alignment_method": alignment_method,
                "qwen_aligned_rows_before_clean_filter": qwen_aligned_rows,
                **clean_audit,
                "aligned_rows": aligned_rows,
                "valid_score_rows": valid_rows,
                "excluded_invalid_rows": invalid_rows,
                "excluded_zero_or_small_w2v_rows": small_w2v_rows,
                "negative_w2v_rows": negative_w2v_rows,
                **{f"source_{model}_rows": count for model, count in source_counts.items()},
            }
        )

    full_clean_run = (
        clean_filter_enabled and not args.years and not args.months
    )
    if (
        full_clean_run
        and args.expected_clean_rows
        and (
            total_clean_reference_rows != args.expected_clean_rows
            or total_aligned_rows != args.expected_clean_rows
        )
    ):
        raise ValueError(
            "Full test-clean row count mismatch: "
            f"expected={args.expected_clean_rows:,}, "
            f"reference={total_clean_reference_rows:,}, "
            f"matched={total_aligned_rows:,}"
        )

    ours = np.concatenate(ours_parts)
    conditional = np.concatenate(cond_parts)
    valid_aligned_rows = int(len(ours))
    if (
        full_clean_run
        and args.expected_clean_rows
        and valid_aligned_rows != args.expected_clean_rows
    ):
        raise ValueError(
            "Invalid scores reduced the full test-clean paired sample: "
            f"expected={args.expected_clean_rows:,}, "
            f"valid={valid_aligned_rows:,}"
        )
    duplicates_collapsed = 0
    if args.unit == "masked_sentence":
        sentence_hashes = np.concatenate(sentence_hash_parts)
        ours, conditional, duplicates_collapsed = collapse_by_sentence(
            sentence_hashes,
            ours,
            conditional,
        )

    result = compute_statistics(ours, conditional, alpha=args.alpha)
    result.update(
        {
            "judge": args.judge_tag,
            "score_column": args.score_col,
            "score_mode": args.score_mode,
            "score_definition_ours": (
                "bert_sv / w2v_sv" if include_w2v else "bert_sv"
            ),
            "score_definition_conditional": (
                "conditional_sv / w2v_sv" if include_w2v else "conditional_sv"
            ),
            "difference_definition": "ours - conditional",
            "pairing_unit": args.unit,
            "identity_columns_for_alignment": "|".join(IDENTITY_COLS),
            "selected_month_count": len(months),
            "selected_months": "|".join(months),
            "clean_filter_enabled": clean_filter_enabled,
            "clean_reference_directory": (
                clean_pred_dir if clean_filter_enabled else ""
            ),
            "expected_full_clean_rows": (
                args.expected_clean_rows if clean_filter_enabled else 0
            ),
            "qwen_aligned_rows_before_clean_filter": total_qwen_aligned_rows,
            "clean_reference_rows": total_clean_reference_rows,
            "total_aligned_rows": total_aligned_rows,
            "valid_aligned_score_rows": valid_aligned_rows,
            "excluded_invalid_rows": total_invalid_rows,
            "excluded_zero_or_small_w2v_rows": total_small_w2v_rows,
            "negative_w2v_rows": total_negative_w2v_rows,
            "duplicate_sentence_rows_collapsed": duplicates_collapsed,
            "min_abs_w2v": args.min_abs_w2v if include_w2v else float("nan"),
            "alpha": args.alpha,
            "hash_bits_for_sentence_unit": 128 if args.unit == "masked_sentence" else 0,
            "ours_directory": directories["bert"],
            "conditional_directory": directories["conditional"],
            "w2v_directory": directories.get("w2v", ""),
        }
    )

    summary = pd.DataFrame([result])
    atomic_to_csv(summary, args.output_csv)
    audit_path = args.audit_csv
    if audit_path is None:
        base, extension = os.path.splitext(os.path.abspath(args.output_csv))
        audit_path = f"{base}_monthly_audit{extension or '.csv'}"
    atomic_to_csv(pd.DataFrame(audit_rows), audit_path)

    print("\n================ RESULT ================")
    print(f"Qwen rows before filter  : {total_qwen_aligned_rows:,}")
    if clean_filter_enabled:
        print(f"Clean reference rows     : {total_clean_reference_rows:,}")
    print(f"Valid aligned rows       : {valid_aligned_rows:,}")
    print(f"Paired units (M)         : {result['n_pairs']:,}")
    print(f"Duplicate rows collapsed : {duplicates_collapsed:,}")
    print(f"Mean Ours                : {result['mean_ours']:.10f}")
    print(f"Mean Conditional         : {result['mean_conditional']:.10f}")
    print(
        "Mean difference (Ours-Cond): "
        f"{result['mean_difference_ours_minus_conditional']:.10f}"
    )
    print(f"SD of differences        : {result['std_difference']:.10f}")
    print(f"SE of difference         : {result['se_difference']:.10f}")
    print(
        f"t({result['degrees_freedom']:,})                 : "
        f"{result['t_statistic']:.10f}"
    )
    print(f"Two-sided p-value        : {result['p_value_two_sided']:.12g}")
    print(
        "One-sided p (Ours > Cond): "
        f"{result['p_value_one_sided_ours_greater']:.12g}"
    )
    print(
        f"{100 * (1 - args.alpha):.1f}% CI of difference     : "
        f"[{result['ci_lower_difference']:.10f}, "
        f"{result['ci_upper_difference']:.10f}]"
    )
    print(f"Cohen's dz               : {result['cohens_dz']:.10f}")
    print(f"Paired correlation       : {result['paired_correlation']:.10f}")
    print(f"Conclusion               : {result['conclusion_at_alpha']}")
    print(f"Summary CSV              : {os.path.abspath(args.output_csv)}")
    print(f"Monthly audit CSV        : {os.path.abspath(audit_path)}")


if __name__ == "__main__":
    main()
