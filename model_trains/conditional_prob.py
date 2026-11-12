#!/usr/bin/env python3
"""Train a posting-level conditional-probability skill baseline.

The processed train split contains one row per masked-skill instance. This
script groups those rows by the posting identity (file_path, row_idx), counts
each distinct true_skill at most once per posting, and estimates:

    P(s2 | s1) = count_train_postings(s1, s2) / count_train_postings(s1)

Only all_train_test/train is read. The held-out test split is never used to
fit counts. Singleton-skill postings contribute to count(s1), as required.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DATA_DIR = Path(
    os.environ.get("BASE_DATA_DIR", "/home/jovyan/LEM_data3/hyunjincho/LEM")
)
DEFAULT_TRAIN_ROOT = Path(
    os.environ.get(
        "TRAIN_SPLIT_ROOT",
        str(BASE_DATA_DIR / "all_train_test" / "train"),
    )
)
DEFAULT_VOCAB_PATH = Path(
    os.environ.get(
        "SKILL_VOCAB_PATH",
        str(BASE_DATA_DIR / "all_filterd" / "skill2idx.json"),
    )
)
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("CONDPROB_DIR", str(BASE_DATA_DIR / "condprob_new"))
)
DEFAULT_PATTERN = "preprocessed_*.csv.gz"
DEFAULT_CHUNK_SIZE = 200_000
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)
DEFAULT_PROGRESS_EVERY = 100
DEFAULT_COMPRESSION_LEVEL = 1
MAX_WORKERS = 8
REQUIRED_COLUMNS = ("file_path", "row_idx", "true_skill")

_PLANG_TAIL_RE = re.compile(
    r"\s*\(programming language\)\s*$",
    flags=re.IGNORECASE,
)
_SKILL_TO_ID: dict[str, int] = {}
_VOCAB_SIZE = 0
_CHUNK_SIZE = DEFAULT_CHUNK_SIZE


@dataclass(frozen=True)
class FileCountResult:
    path: str
    source_path: str | None
    input_rows: int
    postings: int
    singleton_postings: int
    multi_skill_postings: int
    skill_counts: dict[int, int]
    pair_counts: dict[int, int]


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def normalize_skill(value: object) -> str | None:
    """Use the exact skill normalization from data_preprocess/preprocess.py."""
    if not isinstance(value, str):
        return None
    normalized = _PLANG_TAIL_RE.sub("", value.strip()).strip().lower()
    return normalized or None


def load_vocab(path: Path) -> tuple[list[str], dict[str, int]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(
            f"{path} must contain a non-empty skill-to-index JSON object"
        )

    skills_by_id = [""] * len(payload)
    seen_ids: set[int] = set()
    for skill, skill_id in payload.items():
        if not isinstance(skill, str) or not isinstance(skill_id, int):
            raise ValueError(
                f"{path} contains a non-string skill or non-integer index"
            )
        if skill_id < 0 or skill_id >= len(payload) or skill_id in seen_ids:
            raise ValueError(
                f"{path} indices must be unique and exactly 0..{len(payload)-1}"
            )
        normalized = normalize_skill(skill)
        if normalized != skill:
            raise ValueError(
                f"vocabulary key {skill!r} is not preprocessing-normalized"
            )
        seen_ids.add(skill_id)
        skills_by_id[skill_id] = skill

    if seen_ids != set(range(len(payload))):
        raise ValueError(
            f"{path} indices must be exactly 0..{len(payload)-1}"
        )
    return skills_by_id, {
        skill: skill_id for skill_id, skill in enumerate(skills_by_id)
    }


def initialize_worker(
    skill_to_id: dict[str, int],
    chunk_size: int,
) -> None:
    global _SKILL_TO_ID, _VOCAB_SIZE, _CHUNK_SIZE
    _SKILL_TO_ID = skill_to_id
    _VOCAB_SIZE = len(skill_to_id)
    _CHUNK_SIZE = chunk_size


def parse_row_idx(value: object, path: Path) -> int:
    try:
        row_idx = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} has invalid row_idx={value!r}") from exc
    if row_idx < 0:
        raise ValueError(f"{path} has negative row_idx={row_idx}")
    return row_idx


def count_split_file(path_text: str) -> FileCountResult:
    """Count one processed file; worker memory is bounded to that file."""
    path = Path(path_text)
    compression = "gzip" if path.suffix == ".gz" else None
    # One processed file is required to contain exactly one file_path, so row_idx
    # is sufficient as the local key. The main process also rejects repeated
    # file_path values across input files.
    posting_skills: dict[int, set[int]] = {}
    input_rows = 0
    source_path: str | None = None

    try:
        chunks = pd.read_csv(
            path,
            compression=compression,
            usecols=list(REQUIRED_COLUMNS),
            dtype={
                "file_path": "string",
                "row_idx": "Int64",
                "true_skill": "string",
            },
            chunksize=_CHUNK_SIZE,
            low_memory=False,
        )
        for chunk in chunks:
            input_rows += len(chunk)
            for raw_source, raw_row_idx, raw_skill in zip(
                chunk["file_path"],
                chunk["row_idx"],
                chunk["true_skill"],
            ):
                if pd.isna(raw_source):
                    raise ValueError("empty file_path")
                row_source = str(raw_source)
                if not row_source:
                    raise ValueError("empty file_path")
                if source_path is None:
                    source_path = row_source
                elif row_source != source_path:
                    raise ValueError(
                        "one processed file contains multiple file_path values"
                    )

                if pd.isna(raw_row_idx):
                    raise ValueError("null row_idx")
                row_idx = parse_row_idx(raw_row_idx, path)

                if pd.isna(raw_skill):
                    raise ValueError("null true_skill")
                skill = normalize_skill(str(raw_skill))
                if skill is None:
                    raise ValueError("empty true_skill")
                skill_id = _SKILL_TO_ID.get(skill)
                if skill_id is None:
                    raise ValueError(
                        f"true_skill {skill!r} is absent from filtered vocabulary"
                    )
                posting_skills.setdefault(row_idx, set()).add(skill_id)
    except (OSError, EOFError, pd.errors.ParserError, ValueError) as exc:
        raise RuntimeError(f"failed to count {path}: {exc}") from exc

    skill_counts: Counter[int] = Counter()
    pair_counts: Counter[int] = Counter()
    singleton_postings = 0
    multi_skill_postings = 0

    for skill_ids_set in posting_skills.values():
        skill_ids = sorted(skill_ids_set)
        skill_counts.update(skill_ids)
        if len(skill_ids) == 1:
            singleton_postings += 1
        elif len(skill_ids) >= 2:
            multi_skill_postings += 1
            pair_counts.update(
                first * _VOCAB_SIZE + second
                for first, second in combinations(skill_ids, 2)
            )

    return FileCountResult(
        path=path_text,
        source_path=source_path,
        input_rows=input_rows,
        postings=len(posting_skills),
        singleton_postings=singleton_postings,
        multi_skill_postings=multi_skill_postings,
        skill_counts=dict(skill_counts),
        pair_counts=dict(pair_counts),
    )


def merge_sparse(target: np.ndarray, values: dict[int, int]) -> None:
    if not values:
        return
    indices = np.fromiter(values.keys(), dtype=np.int64, count=len(values))
    counts = np.fromiter(values.values(), dtype=np.uint64, count=len(values))
    target[indices] += counts


def train_counts(
    files: list[Path],
    skill_to_id: dict[str, int],
    workers: int,
    chunk_size: int,
    progress_every: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    vocab_size = len(skill_to_id)
    skill_counts = np.zeros(vocab_size, dtype=np.uint64)
    # Only upper-triangle cells are populated. For 1,377 skills: about 15 MB.
    pair_counts_flat = np.zeros(vocab_size * vocab_size, dtype=np.uint64)
    stats = {
        "train_split_files": len(files),
        "nonempty_train_split_files": 0,
        "empty_train_split_files": 0,
        "train_masked_rows": 0,
        "train_postings": 0,
        "singleton_skill_postings": 0,
        "multi_skill_postings": 0,
    }
    seen_source_paths: set[str] = set()

    with Pool(
        processes=workers,
        initializer=initialize_worker,
        initargs=(skill_to_id, chunk_size),
    ) as pool:
        results = pool.imap_unordered(
            count_split_file,
            (str(path) for path in files),
            chunksize=1,
        )
        for completed, result in enumerate(results, start=1):
            if result.source_path is None:
                stats["empty_train_split_files"] += 1
            else:
                if result.source_path in seen_source_paths:
                    raise RuntimeError(
                        "the same source file occurs in multiple processed train "
                        f"files: {result.source_path}"
                    )
                seen_source_paths.add(result.source_path)
                stats["nonempty_train_split_files"] += 1

            merge_sparse(skill_counts, result.skill_counts)
            merge_sparse(pair_counts_flat, result.pair_counts)
            stats["train_masked_rows"] += result.input_rows
            stats["train_postings"] += result.postings
            stats["singleton_skill_postings"] += result.singleton_postings
            stats["multi_skill_postings"] += result.multi_skill_postings

            if completed % progress_every == 0 or completed == len(files):
                print(
                    f"counted train files {completed:,}/{len(files):,}; "
                    f"postings={stats['train_postings']:,}; "
                    f"rows={stats['train_masked_rows']:,}"
                )

    if (
        stats["singleton_skill_postings"]
        + stats["multi_skill_postings"]
        != stats["train_postings"]
    ):
        raise RuntimeError("posting skill-count categories do not sum to postings")

    return (
        skill_counts,
        pair_counts_flat.reshape((vocab_size, vocab_size)),
        stats,
    )


def temporary_path_for(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def write_skill_counts(
    path: Path,
    skills_by_id: list[str],
    counts: np.ndarray,
) -> None:
    temporary_path = temporary_path_for(path)
    try:
        with temporary_path.open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["skill", "count"])
            order = sorted(
                range(len(skills_by_id)),
                key=lambda index: (-int(counts[index]), skills_by_id[index]),
            )
            writer.writerows(
                (skills_by_id[index], int(counts[index]))
                for index in order
            )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def neighbors_for(
    skill_id: int,
    pair_counts: np.ndarray,
) -> list[tuple[int, int]]:
    neighbors: list[tuple[int, int]] = []
    for other_id in range(skill_id):
        count = int(pair_counts[other_id, skill_id])
        if count:
            neighbors.append((other_id, count))
    for other_id in range(skill_id + 1, pair_counts.shape[0]):
        count = int(pair_counts[skill_id, other_id])
        if count:
            neighbors.append((other_id, count))
    return neighbors


def write_probabilities(
    path: Path,
    skills_by_id: list[str],
    skill_counts: np.ndarray,
    pair_counts: np.ndarray,
    min_edge_count: int,
    topn: int,
    smooth_alpha: float,
    compression_level: int,
) -> int:
    temporary_path = temporary_path_for(path)
    rows_written = 0
    try:
        with gzip.open(
            temporary_path,
            "wt",
            encoding="utf-8",
            newline="",
            compresslevel=compression_level,
        ) as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["s1", "s2", "count12", "p_cond"])

            for skill_id, skill in enumerate(skills_by_id):
                denominator = int(skill_counts[skill_id])
                if denominator == 0:
                    continue
                all_neighbors = neighbors_for(skill_id, pair_counts)
                smoothing_width = max(1, len(all_neighbors))
                neighbors = [
                    item for item in all_neighbors
                    if item[1] >= min_edge_count
                ]
                neighbors.sort(
                    key=lambda item: (-item[1], skills_by_id[item[0]])
                )
                if topn:
                    neighbors = neighbors[:topn]

                for other_id, count12 in neighbors:
                    if smooth_alpha:
                        probability = (count12 + smooth_alpha) / (
                            denominator + smooth_alpha * smoothing_width
                        )
                    else:
                        probability = count12 / denominator
                    writer.writerow(
                        [
                            skill,
                            skills_by_id[other_id],
                            count12,
                            repr(probability),
                        ]
                    )
                    rows_written += 1

        os.replace(temporary_path, path)
        return rows_written
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary_path = temporary_path_for(path)
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train posting-level P(s2|s1) from grouped true_skill rows in the "
            "processed train split. Test data is never read."
        )
    )
    parser.add_argument(
        "--train-root", type=Path, default=DEFAULT_TRAIN_ROOT
    )
    parser.add_argument(
        "--vocab-path", type=Path, default=DEFAULT_VOCAB_PATH
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS
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
        "--min-edge-count-to-save", type=int, default=1
    )
    parser.add_argument(
        "--topn-per-s1-to-save",
        type=int,
        default=0,
        help="0 saves all qualifying directed edges.",
    )
    parser.add_argument("--smooth-alpha", type=float, default=0.0)

    # Backward-compatible parsing for the former raw-skills command. These
    # options are intentionally ignored because the new corpus is explicit.
    parser.add_argument("--data-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--train-index-csv", help=argparse.SUPPRESS)
    parser.add_argument("--skill-column", help=argparse.SUPPRESS)
    parser.add_argument("--target-skills-csv", help=argparse.SUPPRESS)
    parser.add_argument(
        "--lowercase",
        type=parse_bool,
        default=True,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.data_root is not None and args.train_root == DEFAULT_TRAIN_ROOT:
        args.train_root = args.data_root / "train"
    args.train_root = args.train_root.expanduser().resolve()
    args.vocab_path = args.vocab_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    if not args.lowercase:
        raise ValueError(
            "--lowercase=false is incompatible with the filtered vocabulary"
        )
    if not args.train_root.is_dir():
        raise FileNotFoundError(f"train root not found: {args.train_root}")
    if not args.vocab_path.is_file():
        raise FileNotFoundError(f"vocabulary not found: {args.vocab_path}")
    if not args.pattern:
        raise ValueError("--pattern must not be empty")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")
    if not 1 <= args.workers <= MAX_WORKERS:
        raise ValueError(f"--workers must be between 1 and {MAX_WORKERS}")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be >= 1")
    if not 1 <= args.compression_level <= 9:
        raise ValueError("--compression-level must be between 1 and 9")
    if args.min_edge_count_to_save < 1:
        raise ValueError("--min-edge-count-to-save must be >= 1")
    if args.topn_per_s1_to_save < 0:
        raise ValueError("--topn-per-s1-to-save must be >= 0")
    if args.smooth_alpha < 0:
        raise ValueError("--smooth-alpha must be >= 0")

    for option, value in (
        ("--train-index-csv", args.train_index_csv),
        ("--skill-column", args.skill_column),
        ("--target-skills-csv", args.target_skills_csv),
    ):
        if value:
            print(f"warning: {option} is obsolete and ignored", file=sys.stderr)


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        path
        for path in args.train_root.rglob(args.pattern)
        if path.is_file()
    )
    if not files:
        raise FileNotFoundError(
            f"no files matching {args.pattern!r} under {args.train_root}"
        )

    skills_by_id, skill_to_id = load_vocab(args.vocab_path)
    print(f"train split root: {args.train_root}")
    print(f"input files:      {len(files):,}")
    print(f"vocabulary:       {args.vocab_path} ({len(skills_by_id):,})")
    print(f"workers:          {args.workers}")
    print("test postings:    0 (test split is never read)")

    skill_counts, pair_counts, stats = train_counts(
        files,
        skill_to_id,
        args.workers,
        args.chunk_size,
        args.progress_every,
    )
    if stats["train_postings"] == 0:
        raise RuntimeError("the train split contains no postings")

    skill_counts_path = args.output_dir / "skill_counts.csv"
    probabilities_path = args.output_dir / "cooc_probs.csv.gz"
    summary_path = args.output_dir / "conditional_prob_summary.json"

    write_skill_counts(skill_counts_path, skills_by_id, skill_counts)
    probability_rows = write_probabilities(
        probabilities_path,
        skills_by_id,
        skill_counts,
        pair_counts,
        args.min_edge_count_to_save,
        args.topn_per_s1_to_save,
        args.smooth_alpha,
        args.compression_level,
    )

    observed_skills = int(np.count_nonzero(skill_counts))
    undirected_pairs = int(np.count_nonzero(pair_counts))
    summary: dict[str, object] = {
        "train_root": str(args.train_root),
        "pattern": args.pattern,
        "posting_identity": ["file_path", "row_idx"],
        "skill_source": "processed train split true_skill",
        "one_count_per_unique_skill_per_posting": True,
        "vocab_path": str(args.vocab_path),
        "vocab_size": len(skills_by_id),
        "test_split_read": False,
        "test_postings_used": 0,
        "count_s_definition": (
            "train postings containing s; singleton postings included"
        ),
        "count_s1_s2_definition": (
            "train postings containing both s1 and s2"
        ),
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "min_edge_count_to_save": args.min_edge_count_to_save,
        "topn_per_s1_to_save": args.topn_per_s1_to_save,
        "smooth_alpha": args.smooth_alpha,
        "observed_skills": observed_skills,
        "undirected_skill_pairs": undirected_pairs,
        "directed_probability_rows": probability_rows,
        "skill_counts_csv": str(skill_counts_path),
        "cooc_probs_csv": str(probabilities_path),
        **stats,
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "python_version": sys.version.split()[0],
    }
    atomic_write_json(summary_path, summary)

    print(f"train postings:   {stats['train_postings']:,}")
    print(f"train rows:       {stats['train_masked_rows']:,}")
    print(f"observed skills:  {observed_skills:,}")
    print(f"undirected pairs: {undirected_pairs:,}")
    print(f"saved edges:      {probability_rows:,}")
    print(f"skill counts:     {skill_counts_path}")
    print(f"probabilities:    {probabilities_path}")
    print(f"summary:          {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
