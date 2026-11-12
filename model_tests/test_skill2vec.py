#!/usr/bin/env python3
"""Predict Skill2Vec cosine neighbors for every masked row in the test split.

Test CSV files are discovered recursively. Neighbors are precomputed once for
each mapped skill, with the truth explicitly excluded. Up to eight workers
stream source files into bounded shards, which are atomically merged into one
downstream-compatible gzip CSV per month.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec

BASE_DATA_DIR = Path(
    os.environ.get("BASE_DATA_DIR", "/home/jovyan/LEM_data3/hyunjincho/LEM")
)
DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "SKILL2VEC_MODEL_PATH",
        str(
            BASE_DATA_DIR
            / "skill2vec_new"
            / "skill2vec_norm_sg1_d300_win50_neg10_ep5.model"
        ),
    )
)
DEFAULT_TEST_ROOT = Path(
    os.environ.get("TEST_ROOT", str(BASE_DATA_DIR / "all_train_test" / "test"))
)
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("SKILL2VEC_OUT_DIR", str(BASE_DATA_DIR / "skill2vec_pred_new"))
)
DEFAULT_PATTERN = "preprocessed_*.csv.gz"
DEFAULT_CHUNK_SIZE = 200_000
DEFAULT_TOP_K = 5
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)
DEFAULT_PROGRESS_EVERY = 100
DEFAULT_COMPRESSION_LEVEL = 1
DEFAULT_MAX_PRINT_PER_REASON = 10
MAX_WORKERS = 8

OUTPUT_COLUMNS = (
    "year",
    "file",
    "row_idx",
    "file_path",
    "soc_2_name",
    "truth",
    "pred_top1",
    "pred_top5",
    "pred_top5_probs",
    "masked_sentence",
)
REQUIRED_INPUT_COLUMNS = (
    "true_skill",
    "masked_sentence",
    "row_idx",
    "file_path",
    "soc_2_name",
)
SKIP_REASONS = (
    "missing_mask",
    "empty_truth",
    "truth_not_in_token_map",
    "truth_not_in_model",
    "no_candidates",
)
SUMMARY_SCHEMA_VERSION = 1
MARKER_SCHEMA_VERSION = 1
NORMALIZATION_VERSION = "preprocess_plang_lower_strip_v1"
_PLANG_TAIL = re.compile(r"\s*\(programming language\)\s*$", re.IGNORECASE)
_YEAR_MONTH = re.compile(r"(?<!\d)(20\d{2})-(0[1-9]|1[0-2])(?!\d)")


@dataclass(frozen=True)
class FileTask:
    sequence: int
    input_path: str
    shard_dir: str
    year: str
    chunk_size: int
    compression_level: int


@dataclass
class FileResult:
    sequence: int
    input_path: str
    shard_path: str
    input_rows: int
    evaluated_rows: int
    skipped: dict[str, int]
    examples: dict[str, list[str]]


@dataclass
class TokenMapping:
    skill_to_token: dict[str, str]
    token_to_skill: dict[str, str]
    ordered_skills: list[str]
    mode: str
    source_path: Path | None
    joiner: str


_WORKER_LOOKUP: dict[str, tuple[str, str, str]] | None = None
_WORKER_SKILL_TO_TOKEN: dict[str, str] | None = None
_WORKER_MODEL_SKILLS: set[str] | None = None
_WORKER_MAX_EXAMPLES = 0


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict Skill2Vec cosine neighbors for all recursively discovered "
            "filtered test rows."
        )
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--token-map-path",
        type=Path,
        default=None,
        help=(
            "Trainer-produced *.skill_tokens.json. By default the model's "
            "sibling map is used; otherwise legacy underscore decoding is used "
            "with a warning."
        ),
    )
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel CSV I/O workers (1-{MAX_WORKERS}).",
    )
    parser.add_argument(
        "--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=DEFAULT_COMPRESSION_LEVEL,
    )
    parser.add_argument(
        "--max-print-per-reason",
        type=int,
        default=DEFAULT_MAX_PRINT_PER_REASON,
    )
    parser.add_argument(
        "--joiner",
        default="_",
        help="Only used by the legacy token-map fallback.",
    )
    parser.add_argument(
        "--overwrite",
        type=parse_bool,
        default=False,
        help="Rebuild months even when a matching completion marker exists.",
    )
    args = parser.parse_args()

    args.model_path = args.model_path.expanduser().resolve()
    args.test_root = args.test_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.token_map_path is not None:
        args.token_map_path = args.token_map_path.expanduser().resolve()

    if not args.model_path.is_file():
        parser.error(f"model not found: {args.model_path}")
    if not args.test_root.is_dir():
        parser.error(f"test root not found: {args.test_root}")
    if not args.pattern:
        parser.error("--pattern must not be empty")
    if args.chunk_size < 1 or args.top_k < 1:
        parser.error("--chunk-size and --top-k must be >= 1")
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be between 1 and {MAX_WORKERS}")
    if args.progress_every < 1:
        parser.error("--progress-every must be >= 1")
    if not 1 <= args.compression_level <= 9:
        parser.error("--compression-level must be between 1 and 9")
    if args.max_print_per_reason < 0:
        parser.error("--max-print-per-reason must be >= 0")
    if not args.joiner or any(char.isspace() for char in args.joiner):
        parser.error("--joiner must be non-empty and contain no whitespace")
    if args.token_map_path is not None and not args.token_map_path.is_file():
        parser.error(f"token map not found: {args.token_map_path}")
    return args


def normalize_dataset_skill(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = _PLANG_TAIL.sub("", value.strip()).strip().lower()
    return normalized or None


def discover_test_files(root: Path, pattern: str) -> list[Path]:
    files = sorted(path.resolve() for path in root.rglob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(f"no files matching {pattern!r} under {root}")
    return files


def extract_year_month(path: Path, root: Path) -> tuple[str, str]:
    relative = path.relative_to(root).as_posix()
    matches = {match.groups() for match in _YEAR_MONTH.finditer(relative)}
    if not matches:
        raise ValueError(f"cannot infer YYYY-MM from test path: {path}")
    if len(matches) != 1:
        raise ValueError(f"conflicting YYYY-MM values in test path: {path}")
    year, month = matches.pop()
    return year, f"{year}-{month}"


def group_files_by_month(
    files: Iterable[Path], root: Path
) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        _, year_month = extract_year_month(path, root)
        grouped[year_month].append(path)
    return {year_month: sorted(grouped[year_month]) for year_month in sorted(grouped)}


def load_keyed_vectors(model_path: Path) -> KeyedVectors:
    errors: list[str] = []
    primary = (
        ((lambda: KeyedVectors.load(str(model_path), mmap="r")), "KeyedVectors")
        if model_path.suffix == ".kv"
        else ((lambda: Word2Vec.load(str(model_path)).wv), "Word2Vec")
    )
    fallback = (
        ((lambda: Word2Vec.load(str(model_path)).wv), "Word2Vec")
        if model_path.suffix == ".kv"
        else ((lambda: KeyedVectors.load(str(model_path), mmap="r")), "KeyedVectors")
    )
    for loader, label in (primary, fallback):
        try:
            keyed_vectors = loader()
            if not keyed_vectors.index_to_key:
                raise ValueError("model vocabulary is empty")
            return keyed_vectors
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    raise RuntimeError(
        f"unable to load Skill2Vec model {model_path}; " + "; ".join(errors)
    )


def sibling_token_map_path(model_path: Path) -> Path:
    return model_path.with_suffix(".skill_tokens.json")


def load_explicit_token_map(path: Path) -> TokenMapping:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("skills") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} must contain a non-empty 'skills' list")

    parsed: list[tuple[int, str, str]] = []
    seen_indices: set[int] = set()
    skill_to_token: dict[str, str] = {}
    token_to_skill: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"invalid token-map record in {path}: {record!r}")
        skill = record.get("skill")
        token = record.get("word2vec_token")
        index = record.get("skill_index")
        if not isinstance(skill, str) or not isinstance(token, str):
            raise ValueError(f"invalid token-map skill/token in {path}: {record!r}")
        if not isinstance(index, int) or index < 0:
            raise ValueError(f"invalid token-map skill_index in {path}: {record!r}")
        normalized = normalize_dataset_skill(skill)
        if normalized != skill:
            raise ValueError(f"token-map skill is not normalized: {skill!r}")
        if not token or any(char.isspace() for char in token):
            raise ValueError(f"invalid Word2Vec token in {path}: {token!r}")
        if skill in skill_to_token or token in token_to_skill or index in seen_indices:
            raise ValueError(f"duplicate skill, token, or index in {path}: {record!r}")
        skill_to_token[skill] = token
        token_to_skill[token] = skill
        seen_indices.add(index)
        parsed.append((index, skill, token))

    parsed.sort()
    joiner = payload.get("joiner", "_")
    if not isinstance(joiner, str) or not joiner:
        joiner = "_"
    return TokenMapping(
        skill_to_token=skill_to_token,
        token_to_skill=token_to_skill,
        ordered_skills=[skill for _, skill, _ in parsed],
        mode="trainer_token_map",
        source_path=path,
        joiner=joiner,
    )


def load_legacy_token_map(wv: KeyedVectors, joiner: str) -> TokenMapping:
    warnings.warn(
        "No trainer-produced *.skill_tokens.json was found. Falling back to "
        f"legacy {joiner!r}->space decoding, which is ambiguous when a skill "
        "itself contains the joiner.",
        RuntimeWarning,
        stacklevel=2,
    )
    skill_to_token: dict[str, str] = {}
    token_to_skill: dict[str, str] = {}
    ordered_skills: list[str] = []
    for raw_token in wv.index_to_key:
        token = str(raw_token)
        skill = normalize_dataset_skill(token.replace(joiner, " "))
        if skill is None:
            raise ValueError(f"cannot decode legacy Word2Vec token: {token!r}")
        previous = skill_to_token.get(skill)
        if previous is not None and previous != token:
            raise ValueError(
                f"ambiguous legacy token decoding: {previous!r}, {token!r} -> {skill!r}"
            )
        skill_to_token[skill] = token
        token_to_skill[token] = skill
        ordered_skills.append(skill)
    return TokenMapping(
        skill_to_token=skill_to_token,
        token_to_skill=token_to_skill,
        ordered_skills=ordered_skills,
        mode="legacy_joiner_fallback",
        source_path=None,
        joiner=joiner,
    )


def resolve_token_mapping(
    model_path: Path,
    requested_path: Path | None,
    wv: KeyedVectors,
    joiner: str,
) -> TokenMapping:
    if requested_path is not None:
        return load_explicit_token_map(requested_path)
    sibling = sibling_token_map_path(model_path)
    if sibling.is_file():
        return load_explicit_token_map(sibling)
    return load_legacy_token_map(wv, joiner)


def precompute_neighbors(
    wv: KeyedVectors,
    mapping: TokenMapping,
    top_k: int,
) -> tuple[dict[str, tuple[str, str, str]], set[str], dict[str, int]]:
    model_tokens = {str(token) for token in wv.index_to_key}
    unmapped_model_tokens = sorted(model_tokens - set(mapping.token_to_skill))
    if unmapped_model_tokens:
        examples = ", ".join(repr(token) for token in unmapped_model_tokens[:10])
        raise ValueError(
            "Word2Vec vocabulary contains tokens absent from the token map: "
            f"{len(unmapped_model_tokens):,}; examples: {examples}"
        )

    candidate_skills = [
        skill
        for skill in mapping.ordered_skills
        if mapping.skill_to_token[skill] in wv.key_to_index
    ]
    if len(candidate_skills) < 2:
        raise ValueError(
            "Skill2Vec must contain at least two mapped skills to exclude truth"
        )
    candidate_tokens = [mapping.skill_to_token[skill] for skill in candidate_skills]
    vectors = np.stack(
        [wv.get_vector(token, norm=True) for token in candidate_tokens]
    ).astype(np.float32, copy=False)
    if not np.isfinite(vectors).all():
        raise ValueError("Skill2Vec contains non-finite normalized vectors")

    effective_top_k = min(top_k, len(candidate_skills) - 1)
    similarities = vectors @ vectors.T
    np.fill_diagonal(similarities, -np.inf)
    ranked = np.argsort(-similarities, axis=1, kind="mergesort")[
        :, :effective_top_k
    ]

    lookup: dict[str, tuple[str, str, str]] = {}
    for truth_position, truth_skill in enumerate(candidate_skills):
        positions = ranked[truth_position]
        predicted_skills = [candidate_skills[int(index)] for index in positions]
        scores = [float(similarities[truth_position, int(index)]) for index in positions]
        if truth_skill in predicted_skills:
            raise AssertionError(f"truth was not excluded for {truth_skill!r}")
        lookup[truth_skill] = (
            predicted_skills[0],
            "|".join(predicted_skills),
            "|".join(f"{score:.6f}" for score in scores),
        )

    metadata = {
        "token_map_skills": len(mapping.skill_to_token),
        "model_candidate_skills": len(candidate_skills),
        "token_map_skills_missing_from_model": (
            len(mapping.skill_to_token) - len(candidate_skills)
        ),
        "requested_top_k": top_k,
        "effective_top_k": effective_top_k,
    }
    return lookup, set(candidate_skills), metadata


def initialize_worker(
    lookup: dict[str, tuple[str, str, str]],
    skill_to_token: dict[str, str],
    model_skills: set[str],
    max_examples: int,
) -> None:
    global _WORKER_LOOKUP, _WORKER_SKILL_TO_TOKEN, _WORKER_MODEL_SKILLS
    global _WORKER_MAX_EXAMPLES
    _WORKER_LOOKUP = lookup
    _WORKER_SKILL_TO_TOKEN = skill_to_token
    _WORKER_MODEL_SKILLS = model_skills
    _WORKER_MAX_EXAMPLES = max_examples


def empty_skip_counts() -> dict[str, int]:
    return {reason: 0 for reason in SKIP_REASONS}


def validate_row_accounting(
    label: str,
    input_rows: int,
    evaluated_rows: int,
    skipped: dict[str, int],
) -> None:
    skipped_rows = sum(skipped.values())
    if input_rows != evaluated_rows + skipped_rows:
        raise RuntimeError(
            f"{label}: row accounting mismatch: input={input_rows:,}, "
            f"evaluated={evaluated_rows:,}, skipped={skipped_rows:,}"
        )


def add_skip(
    skipped: dict[str, int],
    examples: dict[str, list[str]],
    reason: str,
    example: str,
) -> None:
    skipped[reason] += 1
    if len(examples[reason]) < _WORKER_MAX_EXAMPLES:
        examples[reason].append(example)


def evaluate_file(task: FileTask) -> FileResult:
    if (
        _WORKER_LOOKUP is None
        or _WORKER_SKILL_TO_TOKEN is None
        or _WORKER_MODEL_SKILLS is None
    ):
        raise RuntimeError("prediction worker was not initialized")

    input_path = Path(task.input_path)
    shard_path = Path(task.shard_dir) / f"part-{task.sequence:08d}.csv.gz"
    skipped = empty_skip_counts()
    examples = {reason: [] for reason in SKIP_REASONS}
    input_rows = 0
    evaluated_rows = 0

    header = pd.read_csv(input_path, nrows=0, compression="infer")
    columns = set(header.columns)
    missing = sorted(set(REQUIRED_INPUT_COLUMNS) - columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {missing}")
    selected = list(REQUIRED_INPUT_COLUMNS)

    try:
        with gzip.open(
            shard_path,
            "wt",
            encoding="utf-8",
            newline="",
            compresslevel=task.compression_level,
        ) as shard_handle:
            writer = csv.writer(shard_handle, lineterminator="\n")
            chunks = pd.read_csv(
                input_path,
                usecols=selected,
                dtype=str,
                na_filter=False,
                chunksize=task.chunk_size,
                compression="infer",
            )
            for chunk in chunks:
                input_rows += len(chunk)
                truths = chunk["true_skill"].to_numpy(copy=False)
                masked_sentences = chunk["masked_sentence"].to_numpy(copy=False)
                row_indices = (
                    chunk["row_idx"].to_numpy(copy=False)
                    if "row_idx" in chunk
                    else [""] * len(chunk)
                )
                source_paths = (
                    chunk["file_path"].to_numpy(copy=False)
                    if "file_path" in chunk
                    else [""] * len(chunk)
                )
                soc_names = (
                    chunk["soc_2_name"].to_numpy(copy=False)
                    if "soc_2_name" in chunk
                    else [""] * len(chunk)
                )

                for truth_raw, masked, row_idx, source_path, soc_name in zip(
                    truths,
                    masked_sentences,
                    row_indices,
                    source_paths,
                    soc_names,
                ):
                    example_location = f"{input_path.name}:row_idx={row_idx}"
                    if "[MASK]" not in masked:
                        add_skip(
                            skipped, examples, "missing_mask", example_location
                        )
                        continue
                    truth = normalize_dataset_skill(truth_raw)
                    if truth is None:
                        add_skip(
                            skipped, examples, "empty_truth", example_location
                        )
                        continue
                    if truth not in _WORKER_SKILL_TO_TOKEN:
                        add_skip(
                            skipped,
                            examples,
                            "truth_not_in_token_map",
                            f"{example_location}:{truth}",
                        )
                        continue
                    if truth not in _WORKER_MODEL_SKILLS:
                        add_skip(
                            skipped,
                            examples,
                            "truth_not_in_model",
                            f"{example_location}:{truth}",
                        )
                        continue
                    prediction = _WORKER_LOOKUP.get(truth)
                    if prediction is None:
                        add_skip(
                            skipped,
                            examples,
                            "no_candidates",
                            f"{example_location}:{truth}",
                        )
                        continue

                    pred_top1, pred_top5, pred_top5_probs = prediction
                    writer.writerow(
                        (
                            task.year,
                            input_path.name,
                            row_idx,
                            source_path,
                            soc_name,
                            truth,
                            pred_top1,
                            pred_top5,
                            pred_top5_probs,
                            masked,
                        )
                    )
                    evaluated_rows += 1
    except BaseException:
        shard_path.unlink(missing_ok=True)
        raise

    validate_row_accounting(
        str(input_path), input_rows, evaluated_rows, skipped
    )
    return FileResult(
        sequence=task.sequence,
        input_path=str(input_path),
        shard_path=str(shard_path),
        input_rows=input_rows,
        evaluated_rows=evaluated_rows,
        skipped=skipped,
        examples=examples,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def month_signature(
    files: list[Path],
    test_root: Path,
    prediction_signature: str,
) -> str:
    payload = {
        "prediction_signature": prediction_signature,
        "sources": [source_identity(path, test_root) for path in files],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def temporary_path_for(path: Path) -> Path:
    return path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = temporary_path_for(path)
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_year_counts(path: Path, counts: dict[str, int]) -> None:
    temporary_path = temporary_path_for(path)
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("year", "evaluated_samples"))
            for year in sorted(counts):
                writer.writerow((year, counts[year]))
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def completion_marker_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".complete.json")


def load_matching_marker(
    marker_path: Path,
    output_path: Path,
    signature: str,
) -> dict[str, Any] | None:
    if not marker_path.is_file() or not output_path.is_file():
        return None
    try:
        with marker_path.open("r", encoding="utf-8") as handle:
            marker = json.load(handle)
        if marker.get("schema_version") != MARKER_SCHEMA_VERSION:
            return None
        if marker.get("source_signature") != signature:
            return None
        if marker.get("output_columns") != list(OUTPUT_COLUMNS):
            return None
        output_stat = output_path.stat()
        if marker.get("output_size") != output_stat.st_size:
            return None
        if marker.get("output_mtime_ns") != output_stat.st_mtime_ns:
            return None
        if not isinstance(marker.get("stats"), dict):
            return None
        return marker
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def merge_file_result(
    result: FileResult,
    month_stats: dict[str, Any],
    max_examples: int,
) -> None:
    month_stats["input_rows"] += result.input_rows
    month_stats["evaluated_rows"] += result.evaluated_rows
    for reason in SKIP_REASONS:
        month_stats["skipped"][reason] += result.skipped[reason]
        remaining = max_examples - len(month_stats["examples"][reason])
        if remaining > 0:
            month_stats["examples"][reason].extend(result.examples[reason][:remaining])


def process_month(
    year_month: str,
    files: list[Path],
    args: argparse.Namespace,
    pool: Any,
    prediction_signature: str,
) -> tuple[dict[str, Any], str, Path]:
    year = year_month[:4]
    year_dir = args.output_dir / "pred" / year
    year_dir.mkdir(parents=True, exist_ok=True)
    output_path = year_dir / f"predictions_{year_month}.csv.gz"
    marker_path = completion_marker_path(output_path)
    signature = month_signature(files, args.test_root, prediction_signature)

    if not args.overwrite:
        marker = load_matching_marker(marker_path, output_path, signature)
        if marker is not None:
            return marker["stats"], "reused", output_path

    month_stats: dict[str, Any] = {
        "input_files": len(files),
        "input_rows": 0,
        "evaluated_rows": 0,
        "skipped": empty_skip_counts(),
        "examples": {reason: [] for reason in SKIP_REASONS},
    }
    output_temporary = temporary_path_for(output_path)
    batch_size = max(1, args.workers * 2)

    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{year_month}.skill2vec-shards-", dir=year_dir
        ) as shard_dir:
            # The header is its own gzip member. Worker shards are already
            # compressed, so append their bytes directly without serial
            # decompression and recompression in the parent.
            with gzip.open(
                output_temporary,
                "wt",
                encoding="utf-8",
                newline="",
                compresslevel=args.compression_level,
            ) as header_handle:
                writer = csv.writer(header_handle, lineterminator="\n")
                writer.writerow(OUTPUT_COLUMNS)

            with output_temporary.open("ab") as output_handle:
                for start in range(0, len(files), batch_size):
                    batch_files = files[start : start + batch_size]
                    tasks = [
                        FileTask(
                            sequence=start + offset,
                            input_path=str(input_path),
                            shard_dir=shard_dir,
                            year=year,
                            chunk_size=args.chunk_size,
                            compression_level=args.compression_level,
                        )
                        for offset, input_path in enumerate(batch_files)
                    ]
                    results = (
                        pool.map(evaluate_file, tasks, chunksize=1)
                        if pool is not None
                        else [evaluate_file(task) for task in tasks]
                    )
                    results.sort(key=lambda result: result.sequence)
                    for result in results:
                        shard_path = Path(result.shard_path)
                        with shard_path.open("rb") as shard_handle:
                            shutil.copyfileobj(shard_handle, output_handle)
                        shard_path.unlink()
                        merge_file_result(
                            result, month_stats, args.max_print_per_reason
                        )

                    completed = min(start + batch_size, len(files))
                    if (
                        completed == len(files)
                        or completed % args.progress_every < batch_size
                    ):
                        print(
                            f"[{year_month}] merged {completed:,}/{len(files):,} "
                            f"files; predictions={month_stats['evaluated_rows']:,}",
                            flush=True,
                        )

        validate_row_accounting(
            year_month,
            month_stats["input_rows"],
            month_stats["evaluated_rows"],
            month_stats["skipped"],
        )
        os.replace(output_temporary, output_path)
        output_stat = output_path.stat()
        marker = {
            "schema_version": MARKER_SCHEMA_VERSION,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_signature": signature,
            "output_path": str(output_path),
            "output_size": output_stat.st_size,
            "output_mtime_ns": output_stat.st_mtime_ns,
            "output_columns": list(OUTPUT_COLUMNS),
            "stats": month_stats,
        }
        atomic_write_json(marker_path, marker)
        return month_stats, "written", output_path
    finally:
        output_temporary.unlink(missing_ok=True)


def add_month_to_totals(
    totals: dict[str, Any],
    year: str,
    stats: dict[str, Any],
    max_examples: int,
) -> None:
    validate_row_accounting(
        year,
        int(stats["input_rows"]),
        int(stats["evaluated_rows"]),
        {
            reason: int(stats.get("skipped", {}).get(reason, 0))
            for reason in SKIP_REASONS
        },
    )
    totals["input_files"] += int(stats["input_files"])
    totals["input_rows"] += int(stats["input_rows"])
    totals["evaluated_rows"] += int(stats["evaluated_rows"])
    totals["per_year"][year] += int(stats["evaluated_rows"])
    skipped = stats.get("skipped", {})
    examples = stats.get("examples", {})
    for reason in SKIP_REASONS:
        totals["skipped"][reason] += int(skipped.get(reason, 0))
        remaining = max_examples - len(totals["examples"][reason])
        if remaining > 0:
            totals["examples"][reason].extend(examples.get(reason, [])[:remaining])


def build_prediction_signature(
    args: argparse.Namespace,
    mapping: TokenMapping,
    precompute_metadata: dict[str, int],
    lookup: dict[str, tuple[str, str, str]],
) -> tuple[str, dict[str, Any]]:
    model_stat = args.model_path.stat()
    model_identity = {
        "path": str(args.model_path),
        "size": model_stat.st_size,
        "sha256": sha256_file(args.model_path),
    }
    script_path = Path(__file__).resolve()
    script_identity = {
        "path": str(script_path),
        "sha256": sha256_file(script_path),
    }
    if mapping.source_path is not None:
        token_map_identity: dict[str, Any] = {
            "path": str(mapping.source_path),
            "sha256": sha256_file(mapping.source_path),
        }
    else:
        token_map_identity = {
            "path": None,
            "legacy_joiner": mapping.joiner,
        }
    ordered_lookup = [(truth, *lookup[truth]) for truth in sorted(lookup)]
    lookup_bytes = json.dumps(
        ordered_lookup, ensure_ascii=False, separators=(",", ":")
    ).encode()
    payload = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "model": model_identity,
        "script": script_identity,
        "compression_level": args.compression_level,
        "neighbor_lookup_sha256": hashlib.sha256(lookup_bytes).hexdigest(),
        "token_mapping": {
            "mode": mapping.mode,
            **token_map_identity,
        },
        "requested_top_k": args.top_k,
        "effective_top_k": precompute_metadata["effective_top_k"],
        "output_columns": list(OUTPUT_COLUMNS),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    files = discover_test_files(args.test_root, args.pattern)
    grouped = group_files_by_month(files, args.test_root)

    print(f"test root:        {args.test_root}")
    print(f"test files:       {len(files):,}")
    print(f"test months:      {len(grouped):,}")
    print(f"workers:          {args.workers}")
    print(f"model:            {args.model_path}")

    wv = load_keyed_vectors(args.model_path)
    mapping = resolve_token_mapping(
        args.model_path, args.token_map_path, wv, args.joiner
    )
    lookup, model_skills, precompute_metadata = precompute_neighbors(
        wv, mapping, args.top_k
    )
    del wv
    prediction_signature, identity = build_prediction_signature(
        args, mapping, precompute_metadata, lookup
    )
    print(f"token mapping:    {mapping.mode}")
    print(
        "precomputed:      "
        f"{precompute_metadata['model_candidate_skills']:,} truth skills x "
        f"top-{precompute_metadata['effective_top_k']}"
    )
    print(
        "map skills absent from model: "
        f"{precompute_metadata['token_map_skills_missing_from_model']:,}"
    )

    totals: dict[str, Any] = {
        "input_files": 0,
        "input_rows": 0,
        "evaluated_rows": 0,
        "skipped": empty_skip_counts(),
        "examples": {reason: [] for reason in SKIP_REASONS},
        "per_year": defaultdict(int),
    }
    monthly_outputs: list[dict[str, Any]] = []
    reused_months = 0
    written_months = 0
    initializer_args = (
        lookup,
        mapping.skill_to_token,
        model_skills,
        args.max_print_per_reason,
    )
    pool: Any = None

    try:
        if args.workers == 1:
            initialize_worker(*initializer_args)
        else:
            pool = mp.Pool(
                processes=args.workers,
                initializer=initialize_worker,
                initargs=initializer_args,
            )

        for month_number, (year_month, month_files) in enumerate(
            grouped.items(), start=1
        ):
            stats, status, output_path = process_month(
                year_month,
                month_files,
                args,
                pool,
                prediction_signature,
            )
            reused_months += int(status == "reused")
            written_months += int(status == "written")
            add_month_to_totals(
                totals, year_month[:4], stats, args.max_print_per_reason
            )
            monthly_outputs.append(
                {
                    "year_month": year_month,
                    "status": status,
                    "input_files": int(stats["input_files"]),
                    "input_rows": int(stats["input_rows"]),
                    "evaluated_rows": int(stats["evaluated_rows"]),
                    "output_path": str(output_path),
                }
            )
            print(
                f"months {month_number:,}/{len(grouped):,}: {year_month} "
                f"{status}; total predictions={totals['evaluated_rows']:,}",
                flush=True,
            )
    except BaseException:
        if pool is not None:
            pool.terminate()
        raise
    else:
        if pool is not None:
            pool.close()
    finally:
        if pool is not None:
            pool.join()

    validate_row_accounting(
        "global",
        totals["input_rows"],
        totals["evaluated_rows"],
        totals["skipped"],
    )
    pred_root = args.output_dir / "pred"
    pred_root.mkdir(parents=True, exist_ok=True)
    year_counts_path = pred_root / "year_counts.csv"
    atomic_write_year_counts(year_counts_path, dict(totals["per_year"]))

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "test_root": str(args.test_root),
        "pattern": args.pattern,
        "output_root": str(pred_root),
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "compression_level": args.compression_level,
        "prediction_identity": identity,
        "precompute": precompute_metadata,
        "test_files": len(files),
        "months": len(grouped),
        "months_written": written_months,
        "months_reused": reused_months,
        "input_rows": totals["input_rows"],
        "evaluated_rows": totals["evaluated_rows"],
        "skipped": totals["skipped"],
        "skip_examples": totals["examples"],
        "per_year_evaluated_rows": dict(sorted(totals["per_year"].items())),
        "monthly_outputs": monthly_outputs,
        "year_counts_path": str(year_counts_path),
    }
    summary_path = pred_root / "skill2vec_prediction_summary.json"
    atomic_write_json(summary_path, summary)

    print("\n========== Skill2Vec prediction summary ==========")
    print(f"input rows:        {totals['input_rows']:,}")
    print(f"saved predictions: {totals['evaluated_rows']:,}")
    for reason in SKIP_REASONS:
        print(f"skipped {reason}: {totals['skipped'][reason]:,}")
        for example in totals["examples"][reason]:
            print(f"  example: {example}")
    print(f"monthly outputs:   {pred_root}")
    print(f"year counts:       {year_counts_path}")
    print(f"summary:           {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130) from None
