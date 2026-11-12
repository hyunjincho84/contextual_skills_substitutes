#!/usr/bin/env python3
"""Predict global conditional-probability substitutes on the full test split.

Each processed test row is an independent masked-sentence instance. For its
normalized ``true_skill == s1``, candidates are ranked by the train-only table
``P(s2 | s1)`` and the normalized truth is explicitly removed before Top-k
selection.

The test split contains many processed files per month. Workers therefore
write collision-free temporary gzip shards; the parent concatenates those
shards, in deterministic source-file order, into exactly one atomic monthly
output. The input tree is discovered recursively and all original columns are
preserved, including the posting identity ``(file_path, row_idx)`` and context
columns such as ``soc_2_name`` and ``masked_sentence``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator, TextIO

import pandas as pd


BASE_DATA_DIR = Path(
    os.environ.get("BASE_DATA_DIR", "/home/jovyan/LEM_data3/hyunjincho/LEM")
)
DEFAULT_COOC_PROBS = Path(
    os.environ.get(
        "CONDPROB_TABLE",
        str(BASE_DATA_DIR / "condprob_new" / "cooc_probs.csv.gz"),
    )
)
DEFAULT_TEST_ROOT = Path(
    os.environ.get(
        "TEST_SPLIT_ROOT",
        str(BASE_DATA_DIR / "all_train_test" / "test"),
    )
)
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get(
        "CONDPROB_OUT_DIR",
        str(BASE_DATA_DIR / "condprob_pred_new"),
    )
)
DEFAULT_PATTERN = "preprocessed_*.csv.gz"
DEFAULT_CHUNK_SIZE = 200_000
DEFAULT_TOP_K = 5
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)
DEFAULT_PROGRESS_EVERY = 25
DEFAULT_COMPRESSION_LEVEL = 1
MAX_WORKERS = 8
MANIFEST_SCHEMA_VERSION = 2
NORMALIZATION_VERSION = "preprocess.normalize_plang+lower+strip.v1"

_PLANG_TAIL_RE = re.compile(
    r"\s*\(programming language\)\s*$",
    flags=re.IGNORECASE,
)
_MONTH_RE = re.compile(r"^(20\d{2})-(0[1-9]|1[0-2])$")
_MONTH_IN_NAME_RE = re.compile(r"(?<!\d)(20\d{2})-(0[1-9]|1[0-2])(?!\d)")
_YEAR_RE = re.compile(r"^20\d{2}$")
REQUIRED_INPUT_COLUMNS = {
    "file_path",
    "row_idx",
    "true_skill",
    "masked_sentence",
    "soc_2_name",
}
ADDED_OUTPUT_COLUMNS = (
    "year",
    "file",
    "split_file",
    "truth",
    "pred_top1",
    "pred_top5",
    "pred_top5_probs",
    "pred_top5_count12",
    "prediction_k",
)
FILE_STAT_COLUMNS = (
    "year",
    "month",
    "split_file",
    "source_path",
    "output_path",
    "run_status",
    "input_rows",
    "prediction_rows",
    "skipped_missing_masked_sentence",
    "skipped_invalid_truth",
    "skipped_s1_not_in_cond",
    "skipped_no_candidates",
)

_TOP1_BY_TRUTH: dict[str, str] = {}
_TOPK_BY_TRUTH: dict[str, str] = {}
_TOPK_PROBS_BY_TRUTH: dict[str, str] = {}
_TOPK_COUNTS_BY_TRUTH: dict[str, str] = {}
_TOPK_LENGTH_BY_TRUTH: dict[str, int] = {}
_CONDITIONAL_SOURCES: set[str] = set()
_WORKER_CHUNK_SIZE = DEFAULT_CHUNK_SIZE
_WORKER_COMPRESSION_LEVEL = DEFAULT_COMPRESSION_LEVEL


def normalize_skill(value: object) -> str | None:
    """Exactly match preprocessing/training normalization (spaces retained)."""
    if not isinstance(value, str):
        return None
    normalized = _PLANG_TAIL_RE.sub("", value.strip()).strip().lower()
    return normalized or None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def open_csv_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def format_probability(value: float) -> str:
    return format(value, ".12g")


def load_conditional_table(
    path: Path,
    top_k: int,
) -> tuple[
    dict[str, tuple[str, str, str, str, int]],
    set[str],
    dict[str, int],
]:
    """Stream a source-grouped training table and retain only Top-k per s1."""
    predictions: dict[str, tuple[str, str, str, str, int]] = {}
    sources: set[str] = set()
    closed_sources: set[str] = set()
    current_source: str | None = None
    candidates: dict[str, tuple[float, int]] = {}
    stats = {
        "probability_rows_read": 0,
        "normalized_duplicate_pairs": 0,
        "self_edges_explicitly_excluded": 0,
        "conditional_source_skills": 0,
        "conditional_sources_with_candidates": 0,
    }

    def finish_source() -> None:
        nonlocal candidates
        if current_source is None:
            return
        ranked = sorted(
            candidates.items(),
            key=lambda item: (-item[1][0], -item[1][1], item[0]),
        )[:top_k]
        if ranked:
            skills = [skill for skill, _ in ranked]
            probabilities = [values[0] for _, values in ranked]
            counts = [values[1] for _, values in ranked]
            predictions[current_source] = (
                skills[0],
                "|".join(skills),
                "|".join(format_probability(value) for value in probabilities),
                "|".join(str(value) for value in counts),
                len(skills),
            )
        closed_sources.add(current_source)
        candidates = {}

    with open_csv_text(path) as handle:
        reader = csv.DictReader(handle)
        required = {"s1", "s2", "count12", "p_cond"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must contain columns {sorted(required)}; "
                f"found {reader.fieldnames}"
            )

        for line_number, row in enumerate(reader, start=2):
            stats["probability_rows_read"] += 1
            s1 = normalize_skill(row["s1"])
            s2 = normalize_skill(row["s2"])
            if s1 is None or s2 is None:
                raise ValueError(
                    f"{path}:{line_number} has an empty normalized skill"
                )
            try:
                count12 = int(row["count12"])
                probability = float(row["p_cond"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{line_number} has invalid count12/p_cond"
                ) from exc
            if count12 < 1:
                raise ValueError(f"{path}:{line_number} has count12 < 1")
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"{path}:{line_number} has p_cond outside [0, 1]"
                )

            if current_source != s1:
                if current_source is not None:
                    finish_source()
                if s1 in closed_sources:
                    raise ValueError(
                        f"{path} is not grouped contiguously by normalized s1; "
                        f"{s1!r} reappears at line {line_number}"
                    )
                current_source = s1
                sources.add(s1)

            if s2 == s1:
                stats["self_edges_explicitly_excluded"] += 1
                continue

            old = candidates.get(s2)
            new = (probability, count12)
            if old is not None:
                stats["normalized_duplicate_pairs"] += 1
            if old is None or new > old:
                candidates[s2] = new

    finish_source()
    stats["conditional_source_skills"] = len(sources)
    stats["conditional_sources_with_candidates"] = len(predictions)
    return predictions, sources, stats


def parse_test_month(relative_path: Path) -> tuple[str, str]:
    directory_months = {
        match.group(0)
        for part in relative_path.parts[:-1]
        if (match := _MONTH_RE.fullmatch(part)) is not None
    }
    if len(directory_months) > 1:
        raise ValueError(
            f"ambiguous month directories in test path: {relative_path}"
        )
    if directory_months:
        month = next(iter(directory_months))
    else:
        filename_months = {
            f"{year}-{number}"
            for year, number in _MONTH_IN_NAME_RE.findall(relative_path.name)
        }
        if len(filename_months) != 1:
            raise ValueError(
                f"cannot determine one YYYY-MM from test path: {relative_path}"
            )
        month = next(iter(filename_months))

    year = month[:4]
    directory_years = {
        part for part in relative_path.parts[:-1] if _YEAR_RE.fullmatch(part)
    }
    if directory_years and directory_years != {year}:
        raise ValueError(
            f"year/month directory conflict in test path: {relative_path}"
        )
    return year, month


def input_fingerprint(path: Path, relative_path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "split_file": relative_path.as_posix(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def initialize_worker(
    predictions: dict[str, tuple[str, str, str, str, int]],
    sources: set[str],
    chunk_size: int,
    compression_level: int,
) -> None:
    global _TOP1_BY_TRUTH
    global _TOPK_BY_TRUTH
    global _TOPK_PROBS_BY_TRUTH
    global _TOPK_COUNTS_BY_TRUTH
    global _TOPK_LENGTH_BY_TRUTH
    global _CONDITIONAL_SOURCES
    global _WORKER_CHUNK_SIZE
    global _WORKER_COMPRESSION_LEVEL

    _TOP1_BY_TRUTH = {skill: values[0] for skill, values in predictions.items()}
    _TOPK_BY_TRUTH = {skill: values[1] for skill, values in predictions.items()}
    _TOPK_PROBS_BY_TRUTH = {
        skill: values[2] for skill, values in predictions.items()
    }
    _TOPK_COUNTS_BY_TRUTH = {
        skill: values[3] for skill, values in predictions.items()
    }
    _TOPK_LENGTH_BY_TRUTH = {
        skill: values[4] for skill, values in predictions.items()
    }
    _CONDITIONAL_SOURCES = sources
    _WORKER_CHUNK_SIZE = chunk_size
    _WORKER_COMPRESSION_LEVEL = compression_level


@contextmanager
def deterministic_gzip_text_writer(
    path: Path,
    compression_level: int,
) -> Iterator[TextIO]:
    """Write one reproducible gzip member without embedding a temp filename."""
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=compression_level,
            fileobj=raw_handle,
            mtime=0,
        ) as gzip_handle:
            with io.TextIOWrapper(
                gzip_handle,
                encoding="utf-8",
                newline="",
            ) as text_handle:
                yield text_handle


def process_test_file(task: dict[str, object]) -> dict[str, object]:
    source_path = Path(str(task["source_path"]))
    split_file = str(task["split_file"])
    shard_path = Path(str(task["shard_path"]))
    year = str(task["year"])
    month = str(task["month"])

    try:
        header = pd.read_csv(source_path, compression="infer", nrows=0)
        input_columns = header.columns.tolist()
        if len(input_columns) != len(set(input_columns)):
            raise ValueError("duplicate CSV column names")
        missing = REQUIRED_INPUT_COLUMNS.difference(input_columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        collisions = set(ADDED_OUTPUT_COLUMNS).intersection(input_columns)
        if collisions:
            raise ValueError(
                f"input columns collide with prediction columns: {sorted(collisions)}"
            )
        output_columns = input_columns + list(ADDED_OUTPUT_COLUMNS)

        stats: dict[str, object] = {
            "year": year,
            "month": month,
            "split_file": split_file,
            "source_path": str(source_path),
            "input_rows": 0,
            "prediction_rows": 0,
            "skipped_missing_masked_sentence": 0,
            "skipped_invalid_truth": 0,
            "skipped_s1_not_in_cond": 0,
            "skipped_no_candidates": 0,
        }

        shard_path.parent.mkdir(parents=True, exist_ok=True)
        with deterministic_gzip_text_writer(
            shard_path,
            _WORKER_COMPRESSION_LEVEL,
        ) as shard_handle:
            chunks = pd.read_csv(
                source_path,
                compression="infer",
                chunksize=_WORKER_CHUNK_SIZE,
                low_memory=False,
            )
            for chunk in chunks:
                chunk_rows = len(chunk)
                stats["input_rows"] = int(stats["input_rows"]) + chunk_rows
                if chunk_rows == 0:
                    continue

                masked = chunk["masked_sentence"]
                masked_ok = (
                    masked.notna()
                    & masked.astype("string").str.strip().ne("")
                )
                normalized_truth = chunk["true_skill"].map(normalize_skill)
                truth_ok = normalized_truth.notna()
                source_known = normalized_truth.isin(_CONDITIONAL_SOURCES)
                has_candidates = normalized_truth.isin(_TOP1_BY_TRUTH)

                missing_mask = ~masked_ok
                invalid_truth = masked_ok & ~truth_ok
                unknown_source = masked_ok & truth_ok & ~source_known
                no_candidates = (
                    masked_ok & truth_ok & source_known & ~has_candidates
                )
                valid = masked_ok & truth_ok & source_known & has_candidates

                stats["skipped_missing_masked_sentence"] = (
                    int(stats["skipped_missing_masked_sentence"])
                    + int(missing_mask.sum())
                )
                stats["skipped_invalid_truth"] = (
                    int(stats["skipped_invalid_truth"])
                    + int(invalid_truth.sum())
                )
                stats["skipped_s1_not_in_cond"] = (
                    int(stats["skipped_s1_not_in_cond"])
                    + int(unknown_source.sum())
                )
                stats["skipped_no_candidates"] = (
                    int(stats["skipped_no_candidates"])
                    + int(no_candidates.sum())
                )

                valid_count = int(valid.sum())
                stats["prediction_rows"] = (
                    int(stats["prediction_rows"]) + valid_count
                )
                if valid_count == 0:
                    continue

                output = chunk.loc[valid].copy()
                valid_truth = normalized_truth.loc[valid]
                output["year"] = year
                output["file"] = source_path.name
                output["split_file"] = split_file
                output["truth"] = valid_truth.to_numpy()
                output["pred_top1"] = valid_truth.map(_TOP1_BY_TRUTH).to_numpy()
                output["pred_top5"] = valid_truth.map(_TOPK_BY_TRUTH).to_numpy()
                output["pred_top5_probs"] = (
                    valid_truth.map(_TOPK_PROBS_BY_TRUTH).to_numpy()
                )
                output["pred_top5_count12"] = (
                    valid_truth.map(_TOPK_COUNTS_BY_TRUTH).to_numpy()
                )
                output["prediction_k"] = (
                    valid_truth.map(_TOPK_LENGTH_BY_TRUTH).to_numpy()
                )
                output.to_csv(
                    shard_handle,
                    columns=output_columns,
                    index=False,
                    header=False,
                    lineterminator="\n",
                )

        accounted = (
            int(stats["prediction_rows"])
            + int(stats["skipped_missing_masked_sentence"])
            + int(stats["skipped_invalid_truth"])
            + int(stats["skipped_s1_not_in_cond"])
            + int(stats["skipped_no_candidates"])
        )
        if accounted != int(stats["input_rows"]):
            raise RuntimeError(
                f"row accounting mismatch: input={stats['input_rows']}, "
                f"accounted={accounted}"
            )
        stats["input_columns"] = input_columns
        stats["output_columns"] = output_columns
        stats["shard_path"] = str(shard_path)
        return stats
    except Exception as exc:
        raise RuntimeError(f"failed to predict {source_path}: {exc}") from exc


def temporary_path_for(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path_for(path)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path_for(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def merge_month_shards(
    output_path: Path,
    ordered_results: list[dict[str, object]],
    compression_level: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path_for(output_path)
    columns = ordered_results[0]["output_columns"]
    if any(result["output_columns"] != columns for result in ordered_results):
        raise ValueError(
            f"input schemas differ within month {ordered_results[0]['month']}"
        )

    try:
        with deterministic_gzip_text_writer(
            temporary,
            compression_level,
        ) as output_handle:
            csv.writer(output_handle, lineterminator="\n").writerow(columns)

        with temporary.open("ab") as destination:
            for result in ordered_results:
                with Path(str(result["shard_path"])).open("rb") as source:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def month_output_path(output_dir: Path, year: str, month: str) -> Path:
    return output_dir / "pred" / year / f"predictions_{month}.csv.gz"


def month_manifest_path(output_dir: Path, year: str, month: str) -> Path:
    return output_dir / "_manifests" / year / f"predictions_{month}.json"


def load_current_manifest(
    path: Path,
    output_path: Path,
    month_signature: str,
) -> dict[str, object] | None:
    if not path.is_file() or not output_path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        output_stat = output_path.stat()
        if (
            payload.get("schema_version") != MANIFEST_SCHEMA_VERSION
            or payload.get("month_signature") != month_signature
            or payload.get("output_size") != output_stat.st_size
            or payload.get("output_mtime_ns") != output_stat.st_mtime_ns
            or not isinstance(payload.get("file_stats"), list)
        ):
            return None
        return payload
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def aggregate_file_stats(
    file_stats: list[dict[str, object]],
) -> dict[str, int]:
    keys = (
        "input_rows",
        "prediction_rows",
        "skipped_missing_masked_sentence",
        "skipped_invalid_truth",
        "skipped_s1_not_in_cond",
        "skipped_no_candidates",
    )
    return {
        key: sum(int(row.get(key, 0)) for row in file_stats)
        for key in keys
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream the processed test split and write truth-excluded "
            "conditional-probability Top-k predictions, one atomic file per "
            "month."
        )
    )
    parser.add_argument("--cooc-probs", type=Path, default=DEFAULT_COOC_PROBS)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Print progress after this many completed/reused months.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=DEFAULT_COMPRESSION_LEVEL,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild monthly outputs even when their manifests are current.",
    )
    parser.add_argument(
        "--max-print-per-reason",
        type=int,
        default=10,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.cooc_probs = args.cooc_probs.expanduser().resolve()
    args.test_root = args.test_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.cooc_probs.is_file():
        raise FileNotFoundError(
            f"conditional-probability table not found: {args.cooc_probs}"
        )
    if not args.test_root.is_dir():
        raise FileNotFoundError(f"test root not found: {args.test_root}")
    if not args.pattern:
        raise ValueError("--pattern must not be empty")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if not 1 <= args.workers <= MAX_WORKERS:
        raise ValueError(f"--workers must be between 1 and {MAX_WORKERS}")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be >= 1")
    if not 1 <= args.compression_level <= 9:
        raise ValueError("--compression-level must be between 1 and 9")
    if args.max_print_per_reason < 0:
        raise ValueError("--max-print-per-reason must be >= 0")
    try:
        args.output_dir.relative_to(args.test_root)
    except ValueError:
        pass
    else:
        raise ValueError("--output-dir must not be inside --test-root")


def process_month_tasks(
    pool: Pool | None,
    tasks: list[dict[str, object]],
) -> list[dict[str, object]]:
    if pool is None:
        return [process_test_file(task) for task in tasks]
    return list(pool.imap_unordered(process_test_file, tasks, chunksize=1))


def main() -> int:
    args = parse_args()
    validate_args(args)

    files = sorted(
        path for path in args.test_root.rglob(args.pattern) if path.is_file()
    )
    if not files:
        raise FileNotFoundError(
            f"no files matching {args.pattern!r} under {args.test_root}"
        )

    grouped_files: dict[tuple[str, str], list[tuple[Path, Path]]] = defaultdict(list)
    seen_relative_paths: set[str] = set()
    for path in files:
        relative = path.relative_to(args.test_root)
        relative_text = relative.as_posix()
        if relative_text in seen_relative_paths:
            raise RuntimeError(f"duplicate test path discovered: {relative_text}")
        seen_relative_paths.add(relative_text)
        year, month = parse_test_month(relative)
        grouped_files[(year, month)].append((path, relative))
    for values in grouped_files.values():
        values.sort(key=lambda item: item[1].as_posix())

    predictions, sources, table_stats = load_conditional_table(
        args.cooc_probs,
        args.top_k,
    )
    if not predictions:
        raise RuntimeError(
            "conditional-probability table has no non-truth candidates"
        )

    script_hash = sha256_file(Path(__file__).resolve())
    table_hash = sha256_file(args.cooc_probs)
    config_payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "script_sha256": script_hash,
        "cooc_probs_sha256": table_hash,
        "top_k": args.top_k,
        "compression_level": args.compression_level,
        "normalization_version": NORMALIZATION_VERSION,
        "truth_excluded_explicitly": True,
        "output_layout": "pred/{year}/predictions_{YYYY-MM}.csv.gz",
    }
    config_signature = stable_json_hash(config_payload)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary_parent = args.output_dir / ".tmp"
    temporary_parent.mkdir(parents=True, exist_ok=True)
    run_temporary = Path(
        tempfile.mkdtemp(prefix="conditional_prob_", dir=temporary_parent)
    )

    print(f"test split root: {args.test_root}")
    print(f"input files:     {len(files):,}")
    print(f"months:          {len(grouped_files):,}")
    print(f"conditional:     {args.cooc_probs}")
    print(f"source skills:   {len(sources):,}")
    print(f"top-k:           {args.top_k}")
    print(f"workers:         {args.workers}")

    all_file_stats: list[dict[str, object]] = []
    month_stats: list[dict[str, object]] = []
    processed_months = 0
    reused_months = 0
    pool: Pool | None = None
    processing_succeeded = False

    try:
        if args.workers > 1:
            pool = Pool(
                processes=args.workers,
                initializer=initialize_worker,
                initargs=(
                    predictions,
                    sources,
                    args.chunk_size,
                    args.compression_level,
                ),
            )
        else:
            initialize_worker(
                predictions,
                sources,
                args.chunk_size,
                args.compression_level,
            )

        for month_number, ((year, month), month_files) in enumerate(
            sorted(grouped_files.items()),
            start=1,
        ):
            output_path = month_output_path(args.output_dir, year, month)
            manifest_path = month_manifest_path(args.output_dir, year, month)
            fingerprints = [
                input_fingerprint(path, relative)
                for path, relative in month_files
            ]
            month_signature = stable_json_hash(
                {
                    "config_signature": config_signature,
                    "inputs": fingerprints,
                }
            )

            current_manifest = None
            if not args.force:
                current_manifest = load_current_manifest(
                    manifest_path,
                    output_path,
                    month_signature,
                )
            if current_manifest is not None:
                reused_months += 1
                current_file_stats = []
                for stored in current_manifest["file_stats"]:
                    row = dict(stored)
                    row["run_status"] = "reused_up_to_date"
                    current_file_stats.append(row)
                all_file_stats.extend(current_file_stats)
                aggregate = aggregate_file_stats(current_file_stats)
                month_stats.append(
                    {
                        "year": year,
                        "month": month,
                        "run_status": "reused_up_to_date",
                        "input_files": len(month_files),
                        "output_path": str(output_path),
                        **aggregate,
                    }
                )
            else:
                month_temporary = run_temporary / month
                month_temporary.mkdir(parents=True, exist_ok=False)
                tasks = []
                for file_order, (path, relative) in enumerate(month_files):
                    tasks.append(
                        {
                            "source_path": str(path),
                            "split_file": relative.as_posix(),
                            "year": year,
                            "month": month,
                            "file_order": file_order,
                            "shard_path": str(
                                month_temporary / f"{file_order:06d}.csv.gz"
                            ),
                        }
                    )

                results = process_month_tasks(pool, tasks)
                if len(results) != len(tasks):
                    raise RuntimeError(
                        f"month {month}: expected {len(tasks)} worker results, "
                        f"got {len(results)}"
                    )
                order_by_split_file = {
                    relative.as_posix(): index
                    for index, (_, relative) in enumerate(month_files)
                }
                results.sort(
                    key=lambda row: order_by_split_file[str(row["split_file"])]
                )
                if len({str(row["split_file"]) for row in results}) != len(results):
                    raise RuntimeError(f"month {month}: duplicate worker result")

                merge_month_shards(
                    output_path,
                    results,
                    args.compression_level,
                )
                processed_months += 1

                stored_file_stats = []
                for result in results:
                    stored = {
                        key: value
                        for key, value in result.items()
                        if key not in {
                            "input_columns",
                            "output_columns",
                            "shard_path",
                        }
                    }
                    stored["output_path"] = str(output_path)
                    stored["run_status"] = "processed"
                    stored_file_stats.append(stored)
                aggregate = aggregate_file_stats(stored_file_stats)
                output_stat = output_path.stat()
                manifest_payload = {
                    "schema_version": MANIFEST_SCHEMA_VERSION,
                    "month_signature": month_signature,
                    "config_signature": config_signature,
                    "year": year,
                    "month": month,
                    "output_path": str(output_path),
                    "output_size": output_stat.st_size,
                    "output_mtime_ns": output_stat.st_mtime_ns,
                    "input_fingerprints": fingerprints,
                    "file_stats": stored_file_stats,
                    "aggregate": aggregate,
                }
                atomic_write_json(manifest_path, manifest_payload)
                all_file_stats.extend(stored_file_stats)
                month_stats.append(
                    {
                        "year": year,
                        "month": month,
                        "run_status": "processed",
                        "input_files": len(month_files),
                        "output_path": str(output_path),
                        **aggregate,
                    }
                )
                shutil.rmtree(month_temporary)

            if (
                month_number % args.progress_every == 0
                or month_number == len(grouped_files)
            ):
                cumulative_rows = sum(
                    int(row["prediction_rows"]) for row in month_stats
                )
                print(
                    f"months {month_number:,}/{len(grouped_files):,}; "
                    f"predictions={cumulative_rows:,}; "
                    f"processed={processed_months:,}; reused={reused_months:,}"
                )
        processing_succeeded = True
    finally:
        if pool is not None:
            if processing_succeeded:
                pool.close()
            else:
                pool.terminate()
            pool.join()
        if run_temporary.exists():
            shutil.rmtree(run_temporary)

    all_file_stats.sort(key=lambda row: str(row["split_file"]))
    month_stats.sort(key=lambda row: str(row["month"]))
    aggregate = aggregate_file_stats(all_file_stats)
    accounted = (
        aggregate["prediction_rows"]
        + aggregate["skipped_missing_masked_sentence"]
        + aggregate["skipped_invalid_truth"]
        + aggregate["skipped_s1_not_in_cond"]
        + aggregate["skipped_no_candidates"]
    )
    if accounted != aggregate["input_rows"]:
        raise RuntimeError(
            f"global row accounting mismatch: input={aggregate['input_rows']}, "
            f"accounted={accounted}"
        )

    files_summary_path = args.output_dir / "prediction_files.csv"
    months_summary_path = args.output_dir / "prediction_months.csv"
    summary_path = args.output_dir / "conditional_prob_prediction_summary.json"
    year_counts_path = args.output_dir / "pred" / "year_counts_conditional.csv"

    atomic_write_csv(files_summary_path, all_file_stats, FILE_STAT_COLUMNS)
    month_fields = (
        "year",
        "month",
        "output_path",
        "run_status",
        "input_files",
        "input_rows",
        "prediction_rows",
        "skipped_missing_masked_sentence",
        "skipped_invalid_truth",
        "skipped_s1_not_in_cond",
        "skipped_no_candidates",
    )
    atomic_write_csv(months_summary_path, month_stats, month_fields)

    per_year: dict[str, int] = defaultdict(int)
    for row in month_stats:
        per_year[str(row["year"])] += int(row["prediction_rows"])
    atomic_write_csv(
        year_counts_path,
        [
            {"year": year, "evaluated_samples": count}
            for year, count in sorted(per_year.items())
        ],
        ("year", "evaluated_samples"),
    )

    summary: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_root": str(args.test_root),
        "pattern": args.pattern,
        "input_layout": "test/{year}/{YYYY-MM}/preprocessed_*.csv.gz",
        "output_layout": "pred/{year}/predictions_{YYYY-MM}.csv.gz",
        "cooc_probs": str(args.cooc_probs),
        "cooc_probs_sha256": table_hash,
        "script_sha256": script_hash,
        "normalization_version": NORMALIZATION_VERSION,
        "truth_excluded_explicitly": True,
        "ranking": "descending p_cond, then count12, then normalized s2",
        "top_k_requested": args.top_k,
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "compression_level": args.compression_level,
        "test_files": len(files),
        "test_months": len(grouped_files),
        "processed_months_this_run": processed_months,
        "reused_months_this_run": reused_months,
        "all_input_columns_preserved": True,
        "posting_identity_columns_preserved": ["file_path", "row_idx"],
        "context_columns_preserved": ["soc_2_name", "masked_sentence"],
        "prediction_files_csv": str(files_summary_path),
        "prediction_months_csv": str(months_summary_path),
        "year_counts_csv": str(year_counts_path),
        "python_version": sys.version.split()[0],
        "pandas_version": pd.__version__,
        **table_stats,
        **aggregate,
    }
    atomic_write_json(summary_path, summary)

    print(f"input rows:       {aggregate['input_rows']:,}")
    print(f"prediction rows:  {aggregate['prediction_rows']:,}")
    print(f"skipped rows:     {aggregate['input_rows'] - aggregate['prediction_rows']:,}")
    print(f"monthly outputs:  {args.output_dir / 'pred'}")
    print(f"file summary:     {files_summary_path}")
    print(f"month summary:    {months_summary_path}")
    print(f"run summary:      {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
