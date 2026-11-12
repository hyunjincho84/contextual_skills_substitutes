#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""BERT semantic validation for GPT and matched BERT predictions.

Both ``gpt.csv`` and its source sample ``bert.csv`` are scored with the same
BERT judge and span-aligned attention-weighted definition as ``scoring_bert.py``.
The GPT file uses its GPT candidates; the BERT file uses its BERT candidates.
In both cases the true skill is explicitly excluded before selecting the
highest-ranked substitute.

After scoring ``bert.csv``, each sample is located in the existing monthly
``evaluation_sv/bert_clean/bert`` output by ``file`` and zero-based
``iloc[row_idx]``. Identity fields, the selected substitute, and ``sv_bert``
are compared, and a per-row reference-check CSV is written next to the scored
BERT sample.
"""

import os
import re
import glob
import argparse
import math
import random
from typing import Dict, List, Optional, Set, Tuple

try:
    from .scoring_bert import (
        build_aligned_windows,
        compute_sv_batch as compute_aligned_sv_batch,
        load_model as load_bert_model,
    )
except ImportError:  # Direct script execution.
    from scoring_bert import (
        build_aligned_windows,
        compute_sv_batch as compute_aligned_sv_batch,
        load_model as load_bert_model,
    )

import torch
import pandas as pd
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")

SAMPLE_ROOT = os.environ.get(
    "GPT_SAMPLINGS_DIR",
    "/home/jovyan/LEM_data3/hyunjincho/LEM/predictions_clean/gpt_samplings",
)
DEFAULT_GPT_INPUT = os.path.join(SAMPLE_ROOT, "gpt.csv")
DEFAULT_BERT_INPUT = os.path.join(SAMPLE_ROOT, "bert.csv")
DEFAULT_REFERENCE_ROOT = os.environ.get(
    "BERT_SV_REFERENCE_ROOT",
    "/home/jovyan/LEM_data3/hyunjincho/LEM/evaluation_sv/bert_clean/bert",
)
DEFAULT_MODEL_NAME = os.environ.get("BERT_BASE_MODEL", "bert-base-uncased")
DEFAULT_CACHE_DIR = os.environ.get(
    "BERT_HF_CACHE_DIR",
    os.environ.get(
        "HF_CACHE_DIR", "/home/jovyan/LEM_data2/hf_cache/transformers"
    ),
)


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _pipe_candidates(value: object) -> List[str]:
    return [
        part.strip()
        for part in _clean_text(value).split("|")
        if part.strip()
    ]


def choose_pred_used(row: pd.Series, prediction_source: str) -> str:
    """Return the highest-ranked source candidate after excluding truth."""
    truth = _clean_text(row.get("truth")).casefold()
    if prediction_source == "bert":
        candidates = [
            _clean_text(row.get("bert_substitution")),
            *_pipe_candidates(row.get("bert_pred_top5")),
        ]
    elif prediction_source == "gpt":
        candidates = [
            _clean_text(row.get("gpt_substitution")),
            _clean_text(row.get("pred_used")),
            _clean_text(row.get("pred_top1")),
            *_pipe_candidates(row.get("gpt_pred_top5")),
            *_pipe_candidates(row.get("pred_top5")),
        ]
    else:
        raise ValueError(f"Unknown prediction source: {prediction_source}")

    seen: Set[str] = set()
    for candidate in candidates:
        normalized = candidate.casefold()
        if not candidate or normalized in seen:
            continue
        seen.add(normalized)
        if normalized != truth:
            return candidate
    return ""


def compute_sv_for_chunk(
    df: pd.DataFrame,
    prediction_source: str,
    tokenizer,
    model,
    window_size: int,
    max_len: int,
    sent_batch: int,
    device: str,
    strict: bool = False,
):
    """Score a chunk while retaining row order and invalid rows."""
    df = df.copy()
    for col in ["truth", "masked_sentence", "row_idx"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    candidate_columns = (
        {"bert_substitution", "bert_pred_top5"}
        if prediction_source == "bert"
        else {
            "gpt_substitution",
            "gpt_pred_top5",
            "pred_used",
            "pred_top1",
            "pred_top5",
        }
    )
    if not candidate_columns.intersection(df.columns):
        raise ValueError(f"Missing {prediction_source.upper()} substitution/candidate columns")

    df["pred_used"] = df.apply(
        choose_pred_used, axis=1, prediction_source=prediction_source
    ).astype(str)
    originals, substitutes = [], []
    valid_positions = []
    failure_count = 0

    for position, (_, row) in enumerate(df.iterrows()):
        truth = _clean_text(row.get("truth"))
        masked = _clean_text(row.get("masked_sentence"))
        pred_used = _clean_text(row.get("pred_used"))
        try:
            if not truth:
                raise ValueError("truth is empty")
            if not pred_used:
                raise ValueError(f"no truth-excluded {prediction_source.upper()} candidate")
            original, substitute = build_aligned_windows(
                masked,
                truth,
                pred_used,
                tokenizer,
                window_size,
                max_len,
            )
            originals.append(original)
            substitutes.append(substitute)
            valid_positions.append(position)
        except (TypeError, ValueError) as exc:
            failure_count += 1
            if strict:
                raise ValueError(f"row {position}: {exc}") from exc

    scores = [float("nan")] * len(df)
    anchor_texts = [""] * len(df)
    if originals:
        valid_scores, valid_anchors = compute_aligned_sv_batch(
            originals,
            substitutes,
            tokenizer,
            model,
            max_len,
            sent_batch,
            device,
        )
        for position, score, anchor_text in zip(
            valid_positions, valid_scores, valid_anchors
        ):
            scores[position] = score
            anchor_texts[position] = anchor_text

    df["substitution"] = df["pred_used"]
    df["anchor_text"] = anchor_texts
    df["sv_bert"] = scores
    df.attrs["failure_count"] = failure_count
    return df


def infer_year_from_path(path: str) -> Optional[str]:
    m = re.search(r"(20\d{2})", path)
    return m.group(1) if m else None


def infer_out_path_next_to_input(in_path: str) -> str:
    base = os.path.basename(in_path)
    if base.endswith(".csv.gz"):
        stem = base[:-len(".csv.gz")]
    elif base.endswith(".csv"):
        stem = base[:-len(".csv")]
    else:
        stem = os.path.splitext(base)[0]
    out_name = f"{stem}_with_sv_bert.csv.gz"
    return os.path.join(os.path.dirname(in_path), out_name)


def append_chunk_gzip(out_path: str, df_chunk: pd.DataFrame, write_header: bool):
    mode = "wt" if write_header else "at"
    df_chunk.to_csv(
        out_path,
        index=False,
        compression="gzip",
        mode=mode,
        header=write_header,
    )


def filter_chunk_years(
    chunk: pd.DataFrame,
    years: Optional[List[str]],
    source_path: str,
) -> pd.DataFrame:
    if not years:
        return chunk
    wanted = set(map(str, years))
    if "year" in chunk.columns:
        numeric_year = pd.to_numeric(chunk["year"], errors="coerce").astype("Int64")
        return chunk[numeric_year.astype(str).isin(wanted)].copy()
    path_year = infer_year_from_path(source_path)
    if path_year is None:
        raise ValueError(
            "--years requires a year column or year in the input path: "
            f"{source_path}"
        )
    return chunk if path_year in wanted else chunk.iloc[:0]


def expected_output_columns(input_columns: List[str]) -> List[str]:
    columns = list(input_columns)
    for column in ("pred_used", "substitution", "anchor_text", "sv_bert"):
        if column not in columns:
            columns.append(column)
    return columns


def count_filtered_rows(
    path: str,
    chunk_size: int,
    years: Optional[List[str]],
) -> int:
    total = 0
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        total += len(filter_chunk_years(chunk, years, path))
    return total


def validate_resume_output(
    input_path: str,
    output_path: str,
    chunk_size: int,
    years: Optional[List[str]],
):
    """Only accept a complete output with the current schema and row count."""
    if not os.path.isfile(output_path):
        return False, "output does not exist"
    try:
        input_header = pd.read_csv(input_path, nrows=0)
        output_header = pd.read_csv(output_path, nrows=0)
        expected = expected_output_columns(list(input_header.columns))
        if list(output_header.columns) != expected:
            return False, "output schema/order does not match the current scorer"
        input_rows = count_filtered_rows(input_path, chunk_size, years)
        output_rows = sum(
            len(chunk)
            for chunk in pd.read_csv(
                output_path, usecols=["sv_bert"], chunksize=100_000
            )
        )
        if input_rows != output_rows:
            return False, f"row-count mismatch: input={input_rows}, output={output_rows}"
    except Exception as exc:
        return False, f"unreadable output: {exc}"
    return True, f"validated {output_rows:,} rows"


def _year_text(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return _clean_text(value)
    return str(int(numeric))


def reference_path(reference_root: str, year: object, source_file: object) -> str:
    basename = os.path.basename(_clean_text(source_file))
    month_match = re.search(r"(20\d{2}-\d{2})", basename)
    if month_match is None:
        raise ValueError(f"Could not infer month from source file: {source_file}")
    month = month_match.group(1)
    return os.path.join(
        reference_root,
        _year_text(year),
        f"sv_summary_bert_full_bert_{month}.csv.gz",
    )


def reference_check_path(scored_bert_path: str) -> str:
    suffix = ".csv.gz"
    stem = (
        scored_bert_path[: -len(suffix)]
        if scored_bert_path.endswith(suffix)
        else os.path.splitext(scored_bert_path)[0]
    )
    return f"{stem}_reference_check.csv.gz"


def write_frame_atomically(frame: pd.DataFrame, path: str) -> None:
    temporary = f"{path}.tmp.{os.getpid()}"
    try:
        frame.to_csv(temporary, index=False, compression="gzip")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def compare_bert_with_reference(
    scored_path: str,
    reference_root: str,
    atol: float,
    rtol: float,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Compare sampled BERT scores with monthly outputs by zero-based iloc."""
    scored = pd.read_csv(scored_path).reset_index(names="sample_position")
    required = {
        "year",
        "file",
        "row_idx",
        "truth",
        "masked_sentence",
        "substitution",
        "sv_bert",
    }
    missing = sorted(required.difference(scored.columns))
    if missing:
        raise ValueError(f"BERT scored sample is missing columns: {missing}")

    records = []
    stats = {
        "rows": len(scored),
        "reference_found": 0,
        "identity_match": 0,
        "substitution_match": 0,
        "score_exact": 0,
        "score_match": 0,
        "all_match": 0,
    }
    for (year, source_file), group in scored.groupby(
        ["year", "file"], sort=False, dropna=False
    ):
        try:
            path = reference_path(reference_root, year, source_file)
            reference = pd.read_csv(
                path,
                usecols=["truth", "masked_sentence", "substitution", "sv_bert"],
            )
            reference_error = ""
        except FileNotFoundError:
            path = reference_path(reference_root, year, source_file)
            reference = None
            reference_error = "reference_file_missing"
        except (TypeError, ValueError) as exc:
            path = ""
            reference = None
            reference_error = f"reference_error: {exc}"

        for row in group.itertuples(index=False):
            record = {
                "sample_position": row.sample_position,
                "year": _year_text(row.year),
                "file": row.file,
                "row_idx": row.row_idx,
                "truth": row.truth,
                "substitution": row.substitution,
                "sv_bert": row.sv_bert,
                "reference_path": path,
                "reference_truth": "",
                "reference_substitution": "",
                "reference_sv_bert": float("nan"),
                "identity_match": False,
                "substitution_match": False,
                "score_abs_diff": float("nan"),
                "score_exact": False,
                "score_match": False,
                "all_match": False,
                "status": reference_error,
            }
            if reference is None:
                records.append(record)
                continue

            try:
                position = int(row.row_idx)
            except (TypeError, ValueError):
                record["status"] = "invalid_row_idx"
                records.append(record)
                continue
            if not 0 <= position < len(reference):
                record["status"] = "row_idx_out_of_range"
                records.append(record)
                continue

            original = reference.iloc[position]
            stats["reference_found"] += 1
            identity_match = (
                _clean_text(row.truth) == _clean_text(original["truth"])
                and _clean_text(row.masked_sentence)
                == _clean_text(original["masked_sentence"])
            )
            substitution_match = (
                _clean_text(row.substitution)
                == _clean_text(original["substitution"])
            )
            computed_score = pd.to_numeric(
                pd.Series([row.sv_bert]), errors="coerce"
            ).iloc[0]
            reference_score = pd.to_numeric(
                pd.Series([original["sv_bert"]]), errors="coerce"
            ).iloc[0]
            both_scores = not pd.isna(computed_score) and not pd.isna(reference_score)
            score_abs_diff = (
                abs(float(computed_score) - float(reference_score))
                if both_scores
                else float("nan")
            )
            score_exact = bool(
                both_scores and float(computed_score) == float(reference_score)
            )
            score_match = bool(
                both_scores
                and math.isclose(
                    float(computed_score),
                    float(reference_score),
                    rel_tol=rtol,
                    abs_tol=atol,
                )
            )
            all_match = identity_match and substitution_match and score_match

            record.update(
                {
                    "reference_truth": original["truth"],
                    "reference_substitution": original["substitution"],
                    "reference_sv_bert": reference_score,
                    "identity_match": identity_match,
                    "substitution_match": substitution_match,
                    "score_abs_diff": score_abs_diff,
                    "score_exact": score_exact,
                    "score_match": score_match,
                    "all_match": all_match,
                    "status": "match" if all_match else "mismatch",
                }
            )
            stats["identity_match"] += int(identity_match)
            stats["substitution_match"] += int(substitution_match)
            stats["score_exact"] += int(score_exact)
            stats["score_match"] += int(score_match)
            stats["all_match"] += int(all_match)
            records.append(record)

        del reference

    report = pd.DataFrame.from_records(records)
    if not report.empty:
        report.sort_values("sample_position", inplace=True)
        report.drop(columns=["sample_position"], inplace=True)
        report.reset_index(drop=True, inplace=True)
    return report, stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score GPT and matched BERT substitutes with BERT semantic validation."
    )
    parser.add_argument("--gpt-input", default=DEFAULT_GPT_INPUT)
    parser.add_argument("--bert-input", default=DEFAULT_BERT_INPUT)
    parser.add_argument(
        "--in-pattern",
        default=None,
        help="Legacy GPT input glob; when set, replaces --gpt-input.",
    )
    parser.add_argument("--reference-root", default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--years", nargs="*", default=None)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=320)
    parser.add_argument("--sent-batch", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=8_000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-bert-control", action="store_true")
    parser.add_argument("--skip-reference-check", action="store_true")
    parser.add_argument("--score-atol", type=float, default=5e-4)
    parser.add_argument("--score-rtol", type=float, default=1e-5)
    parser.add_argument("--require-reference-match", action="store_true")
    args = parser.parse_args()

    if args.window_size < 0 or args.max_len < 1:
        parser.error("window-size and max-len must be valid")
    if args.sent_batch < 1 or args.chunk_size < 1:
        parser.error("sent-batch and chunk-size must be positive")
    if args.score_atol < 0 or args.score_rtol < 0:
        parser.error("score tolerances must be non-negative")
    return args


def main():
    args = parse_args()
    if args.device == "auto":
        args.device = DEVICE
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    gpt_files = (
        sorted(glob.glob(args.in_pattern))
        if args.in_pattern
        else [args.gpt_input]
    )
    tasks = [("gpt", path) for path in gpt_files]
    if not args.skip_bert_control:
        tasks.append(("bert", args.bert_input))
    tasks = [
        (source_name, path)
        for source_name, path in tasks
        if "_with_sv_bert" not in os.path.basename(path)
    ]
    missing_inputs = [path for _, path in tasks if not os.path.isfile(path)]
    if missing_inputs:
        raise FileNotFoundError(f"Missing input files: {missing_inputs}")

    print(f"Judge: BERT ({args.model_name})")
    print(f"Device: {args.device}")
    print(
        f"Scoring config: window_size={args.window_size}, max_len={args.max_len}, "
        f"sent_batch={args.sent_batch}, chunk_size={args.chunk_size}"
    )
    print(f"Tasks: {[(name, path) for name, path in tasks]}")
    print(
        "Alignment: shared context IDs + mean-pooled skill spans; "
        "attention A[query=i, key=anchor] without weight renormalization"
    )
    tokenizer, model = load_bert_model(args)

    grand_in_rows = 0
    grand_out_rows = 0
    grand_failed_rows = 0
    bert_outputs: List[str] = []

    for prediction_source, in_path in tasks:
        out_path = infer_out_path_next_to_input(in_path)
        if prediction_source == "bert":
            bert_outputs.append(out_path)
        print(f"\n[{prediction_source.upper()}] {in_path}")
        print(f"  -> OUT: {out_path}")

        header = pd.read_csv(in_path, nrows=0)
        cols = set(header.columns)
        required = {"truth", "masked_sentence", "row_idx"}
        if not required.issubset(cols):
            raise ValueError(f"Missing required columns in {in_path}: {required - cols}")
        candidate_cols = (
            {"bert_substitution", "bert_pred_top5"}
            if prediction_source == "bert"
            else {
                "gpt_substitution",
                "gpt_pred_top5",
                "pred_used",
                "pred_top1",
                "pred_top5",
            }
        )
        if not candidate_cols.intersection(cols):
            raise ValueError(
                f"Missing {prediction_source.upper()} candidate columns: {in_path}"
            )

        if args.resume:
            complete, reason = validate_resume_output(
                in_path, out_path, args.chunk_size, args.years
            )
            if complete:
                print(f"  [RESUME] skip: {reason}")
                continue
            if os.path.exists(out_path):
                print(f"  [RESUME] recompute: {reason}")

        temporary = f"{out_path}.tmp.{os.getpid()}"
        wrote_any = False
        in_rows_this = 0
        out_rows_this = 0
        failed_rows_this = 0
        try:
            for chunk in tqdm(
                pd.read_csv(in_path, chunksize=args.chunk_size),
                desc=f"Scoring {prediction_source}",
                unit="chunk",
            ):
                chunk = filter_chunk_years(chunk, args.years, in_path)
                if chunk.empty:
                    continue
                chunk = chunk.reset_index(drop=True)
                in_rows_this += len(chunk)
                scored = compute_sv_for_chunk(
                    chunk,
                    prediction_source=prediction_source,
                    tokenizer=tokenizer,
                    model=model,
                    window_size=args.window_size,
                    max_len=args.max_len,
                    sent_batch=args.sent_batch,
                    device=args.device,
                    strict=args.strict,
                )
                failed_rows_this += int(scored.attrs.get("failure_count", 0))
                append_chunk_gzip(temporary, scored, write_header=(not wrote_any))
                wrote_any = True
                out_rows_this += len(scored)

            if not wrote_any:
                empty = header.copy()
                for column in expected_output_columns(list(header.columns)):
                    if column not in empty.columns:
                        empty[column] = pd.Series(dtype="object")
                append_chunk_gzip(temporary, empty, write_header=True)
            os.replace(temporary, out_path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

        print(
            f"  [DONE] input={in_rows_this:,} | output={out_rows_this:,} "
            f"| failed={failed_rows_this:,}"
        )
        grand_in_rows += in_rows_this
        grand_out_rows += out_rows_this
        grand_failed_rows += failed_rows_this

    if bert_outputs and not args.skip_reference_check:
        for scored_path in bert_outputs:
            print(f"\n[REFERENCE CHECK] {scored_path}")
            report, stats = compare_bert_with_reference(
                scored_path,
                args.reference_root,
                args.score_atol,
                args.score_rtol,
            )
            report_path = reference_check_path(scored_path)
            write_frame_atomically(report, report_path)
            max_diff = pd.to_numeric(
                report["score_abs_diff"], errors="coerce"
            ).max()
            print(
                f"  reference_found={stats['reference_found']:,}/{stats['rows']:,}, "
                f"identity_match={stats['identity_match']:,}, "
                f"substitution_match={stats['substitution_match']:,}, "
                f"score_exact={stats['score_exact']:,}, "
                f"score_match={stats['score_match']:,}, "
                f"all_match={stats['all_match']:,}"
            )
            print(f"  max_abs_score_diff={max_diff} | report={report_path}")
            if args.require_reference_match and stats["all_match"] != stats["rows"]:
                raise RuntimeError(
                    "BERT reference verification failed: "
                    f"{stats['all_match']:,}/{stats['rows']:,} rows matched"
                )

    print("\n===== ALL DONE =====")
    print(f"Total processed input rows : {grand_in_rows:,}")
    print(f"Total written output rows  : {grand_out_rows:,}")
    print(f"Total failed rows          : {grand_failed_rows:,}")


if __name__ == "__main__":
    main()