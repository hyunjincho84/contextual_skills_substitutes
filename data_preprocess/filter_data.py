#!/usr/bin/env python3
"""Globally deduplicate masked-skill rows and filter the skill vocabulary.

The deduplication key is the exact ``(masked_sentence, true_skill)`` pair.
Exactly one representative row is retained for every pair across every input
file. The representative is selected deterministically with a seeded hash so
sorted input paths do not systematically favor earlier years.

Filtering order
---------------
1. Select one global representative for each exact pair.
2. Count ``true_skill`` among the globally unique pairs.
3. Keep candidate skills whose unique-pair frequency meets the configured
   threshold and which are not explicitly excluded.
4. Among those candidates, keep only skills that share at least one
   ``(file_path, row_idx)`` posting with a different candidate skill.
5. Drop every selected row whose skill is absent from that final vocabulary.
6. Preserve ``posting_fingerprint`` for leakage-safe splitting and omit a
   ``body`` column if an input happens to contain one.

The selected row identity is ``(file_id, sample_idx)``. ``row_idx`` cannot be
used because one posting can produce multiple masked-skill samples with the
same ``row_idx``. A disk-backed SQLite index bounds memory use.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sqlite3
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator, TextIO

from tqdm import tqdm

DEFAULT_INPUT_ROOT = Path("/home/jovyan/LEM_data3/hyunjincho/LEM/all")
DEFAULT_OUTPUT_ROOT = Path("/home/jovyan/LEM_data3/hyunjincho/LEM/filtered")
DEFAULT_PATTERN = "preprocessed_*.csv.gz"
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)
DEFAULT_MIN_SKILL_FREQUENCY = 100
DEFAULT_DUPLICATE_EXAMPLE_LIMIT = 100
DEFAULT_DEDUP_SEED = 42

PAIR_COLUMNS = ("masked_sentence", "true_skill")
REQUIRED_OUTPUT_COLUMNS = (
    "masked_sentence",
    "true_skill",
    "posting_fingerprint",
    "row_idx",
    "file_path",
)

_READ_CONN: sqlite3.Connection | None = None
_OUTPUT_ROOT: Path | None = None
_SKILL_FREQUENCY: dict[str, int] = {}
_EXCLUDED_SKILLS: frozenset[str] = frozenset()
_SKILLS_WITH_COOCCURRENCE: frozenset[str] = frozenset()
_MIN_SKILL_FREQUENCY = DEFAULT_MIN_SKILL_FREQUENCY
_COMPRESSION_LEVEL = 1


@dataclass(frozen=True)
class InputFile:
    file_id: int
    path: Path
    rel_path: str


@dataclass(frozen=True)
class PairRecord:
    sample_idx: int
    pair_hash: bytes
    priority: bytes
    posting_fingerprint: str
    true_skill: str
    masked_sentence: str
    source_path: str
    row_idx: int

@dataclass(frozen=True)
class PairScanResult:
    file_id: int
    rel_path: str
    input_rows: int
    rows_missing_masked_sentence: int
    rows_missing_true_skill: int
    records: list[PairRecord]


@dataclass(frozen=True)
class DuplicateExample:
    masked_sentence: str
    true_skill: str
    selected_relative_path: str
    selected_sample_idx: int
    selected_file_path: str
    selected_row_idx: int
    duplicate_relative_path: str
    duplicate_sample_idx: int
    duplicate_file_path: str
    duplicate_row_idx: int


@dataclass(frozen=True)
class FilteredFileResult:
    file_id: int
    rel_path: str
    input_rows: int
    selected_pair_rows: int
    rows_dropped_excluded_skill: int
    rows_dropped_rare_skill: int
    rows_dropped_no_cooccurrence_skill: int
    output_rows: int
    output_postings: int


def normalize_skill_key(value: str) -> str:
    """Normalize skills for exclusion matching only."""
    return str(value).strip().casefold()


def iter_input_files(root: Path, pattern: str) -> list[InputFile]:
    paths = sorted(path for path in root.rglob(pattern) if path.is_file())
    return [
        InputFile(file_id=i, path=path, rel_path=str(path.relative_to(root)))
        for i, path in enumerate(paths)
    ]


def open_input_csv(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="\n")
    return path.open("r", encoding="utf-8", newline="\n")


def normalized_csv_lines(handle: TextIO) -> Iterator[str]:
    for line in handle:
        yield line.replace("\r\n", "\n").replace("\r", " ")


def open_output_csv(path: Path, compression_level: int) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(
            path,
            "wt",
            encoding="utf-8",
            newline="",
            compresslevel=compression_level,
        )
    return path.open("w", encoding="utf-8", newline="")


def header_positions(
    header: list[str], required: tuple[str, ...], path: Path
) -> dict[str, int]:
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    return {column: header.index(column) for column in required}


def parse_row_idx(value: str, path: Path) -> int:
    try:
        row_idx = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} has invalid row_idx={value!r}") from exc
    if row_idx < 0:
        raise ValueError(f"{path} has negative row_idx={row_idx}")
    return row_idx


def exact_pair_hash(masked_sentence: str, true_skill: str) -> bytes:
    """Hash an unambiguous length-prefixed encoding of the exact pair."""
    sentence_bytes = masked_sentence.encode("utf-8")
    skill_bytes = true_skill.encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(sentence_bytes).to_bytes(8, "big"))
    digest.update(sentence_bytes)
    digest.update(len(skill_bytes).to_bytes(8, "big"))
    digest.update(skill_bytes)
    return digest.digest()


def representative_priority(seed: int, rel_path: str, sample_idx: int) -> bytes:
    """Return a stable pseudo-random priority for one physical input row."""
    digest = hashlib.blake2b(digest_size=16, person=b"LEMpairpick-v1")
    digest.update(str(seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(rel_path.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(sample_idx).encode("ascii"))
    return digest.digest()


def scan_pair_file(args: tuple[InputFile, int]) -> PairScanResult:
    input_file, dedup_seed = args
    records: list[PairRecord] = []
    input_rows = 0
    rows_missing_masked_sentence = 0
    rows_missing_true_skill = 0

    with open_input_csv(input_file.path) as handle:
        reader = csv.reader(normalized_csv_lines(handle))
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{input_file.path} is empty and has no header") from exc
        positions = header_positions(
            header, REQUIRED_OUTPUT_COLUMNS, input_file.path
        )
        max_position = max(positions.values())

        for sample_idx, row in enumerate(reader):
            if len(row) <= max_position:
                raise ValueError(f"{input_file.path} contains a short CSV row")
            input_rows += 1
            masked_sentence = row[positions["masked_sentence"]]
            true_skill = row[positions["true_skill"]]
            posting_fingerprint = row[positions["posting_fingerprint"]].strip()
            if not posting_fingerprint:
                raise ValueError(
                    f"{input_file.path} contains an empty posting_fingerprint"
                )
            if not masked_sentence:
                rows_missing_masked_sentence += 1
                continue
            if not true_skill:
                rows_missing_true_skill += 1
                continue
            source_path = row[positions["file_path"]]
            if not source_path:
                raise ValueError(f"{input_file.path} contains an empty file_path")
            row_idx = parse_row_idx(row[positions["row_idx"]], input_file.path)
            records.append(
                PairRecord(
                    sample_idx=sample_idx,
                    pair_hash=exact_pair_hash(masked_sentence, true_skill),
                    priority=representative_priority(
                        dedup_seed, input_file.rel_path, sample_idx
                    ),
                    posting_fingerprint=posting_fingerprint,
                    true_skill=true_skill,
                    masked_sentence=masked_sentence,
                    source_path=source_path,
                    row_idx=row_idx,
                )
            )

    return PairScanResult(
        file_id=input_file.file_id,
        rel_path=input_file.rel_path,
        input_rows=input_rows,
        rows_missing_masked_sentence=rows_missing_masked_sentence,
        rows_missing_true_skill=rows_missing_true_skill,
        records=records,
    )


def connect_index(path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        uri = path.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        connection.execute("PRAGMA query_only = ON")
        return connection
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    connection.execute("PRAGMA temp_store = MEMORY")
    connection.execute("PRAGMA locking_mode = EXCLUSIVE")
    return connection


def initialize_index(path: Path) -> sqlite3.Connection:
    if path.exists():
        if not path.is_file():
            raise ValueError(f"--db exists and is not a file: {path}")
        path.unlink()
    connection = connect_index(path)
    connection.execute(
        """
        CREATE TABLE pair_choice (
            pair_hash BLOB PRIMARY KEY,
            priority BLOB NOT NULL,
            file_id INTEGER NOT NULL,
            sample_idx INTEGER NOT NULL,
            posting_fingerprint TEXT NOT NULL,
            true_skill TEXT NOT NULL,
            source_path TEXT NOT NULL,
            row_idx INTEGER NOT NULL
        ) WITHOUT ROWID
        """
    )
    connection.commit()
    return connection


def index_global_pairs(
    db_path: Path,
    input_files: list[InputFile],
    workers: int,
    dedup_seed: int,
    duplicate_example_limit: int,
) -> tuple[dict[str, int], Counter[str], list[DuplicateExample]]:
    connection = initialize_index(db_path)
    skill_frequency: Counter[str] = Counter()
    duplicate_examples: list[DuplicateExample] = []
    stats = {
        "input_rows": 0,
        "rows_missing_masked_sentence": 0,
        "rows_missing_true_skill": 0,
        "valid_pair_rows": 0,
        "unique_pairs": 0,
        "duplicate_pair_rows": 0,
        "representative_replacements": 0,
    }

    jobs = [(input_file, dedup_seed) for input_file in input_files]
    if workers == 1:
        results: Iterator[PairScanResult] = map(scan_pair_file, jobs)
        pool = None
    else:
        pool = Pool(processes=workers)
        results = pool.imap(scan_pair_file, jobs, chunksize=1)

    try:
        for result in tqdm(
            results,
            total=len(input_files),
            desc="Indexing exact masked-skill rows",
            unit="file",
        ):
            stats["input_rows"] += result.input_rows
            stats["rows_missing_masked_sentence"] += result.rows_missing_masked_sentence
            stats["rows_missing_true_skill"] += result.rows_missing_true_skill
            stats["valid_pair_rows"] += len(result.records)

            for record in result.records:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO pair_choice(
                        pair_hash, priority, file_id, sample_idx,
                        posting_fingerprint, true_skill, source_path, row_idx
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.pair_hash,
                        record.priority,
                        result.file_id,
                        record.sample_idx,
                        record.posting_fingerprint,
                        record.true_skill,
                        record.source_path,
                        record.row_idx,
                    ),
                )
                if cursor.rowcount == 1:
                    stats["unique_pairs"] += 1
                    skill_frequency[record.true_skill] += 1
                    continue

                stats["duplicate_pair_rows"] += 1
                previous = connection.execute(
                    """
                    SELECT priority, file_id, sample_idx, source_path, row_idx
                    FROM pair_choice WHERE pair_hash = ?
                    """,
                    (record.pair_hash,),
                ).fetchone()
                if previous is None:
                    raise RuntimeError("duplicate pair has no selected row")
                (
                    previous_priority,
                    previous_file_id,
                    previous_sample_idx,
                    previous_source_path,
                    previous_row_idx,
                ) = previous
                if len(duplicate_examples) < duplicate_example_limit:
                    duplicate_examples.append(
                        DuplicateExample(
                            masked_sentence=record.masked_sentence,
                            true_skill=record.true_skill,
                            selected_relative_path=input_files[
                                int(previous_file_id)
                            ].rel_path,
                            selected_sample_idx=int(previous_sample_idx),
                            selected_file_path=str(previous_source_path),
                            selected_row_idx=int(previous_row_idx),
                            duplicate_relative_path=result.rel_path,
                            duplicate_sample_idx=record.sample_idx,
                            duplicate_file_path=record.source_path,
                            duplicate_row_idx=record.row_idx,
                        )
                    )
                if record.priority < previous_priority:
                    connection.execute(
                        """
                        UPDATE pair_choice
                        SET priority = ?, file_id = ?, sample_idx = ?,
                            posting_fingerprint = ?, source_path = ?, row_idx = ?
                        WHERE pair_hash = ?
                        """,
                        (
                            record.priority,
                            result.file_id,
                            record.sample_idx,
                            record.posting_fingerprint,
                            record.source_path,
                            record.row_idx,
                            record.pair_hash,
                        ),
                    )
                    stats["representative_replacements"] += 1
            connection.commit()
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
        raise
    else:
        if pool is not None:
            pool.close()
            pool.join()
    finally:
        connection.close()

    connection = connect_index(db_path)
    connection.execute(
        "CREATE UNIQUE INDEX selected_input_row "
        "ON pair_choice(file_id, sample_idx)"
    )
    stats["unique_pair_postings"] = int(
        connection.execute(
            "SELECT COUNT(DISTINCT posting_fingerprint) FROM pair_choice"
        ).fetchone()[0]
    )
    connection.commit()
    connection.close()
    return stats, skill_frequency, duplicate_examples


def find_skills_with_cooccurrence(
    db_path: Path,
    input_files: list[InputFile],
    skill_frequency: Counter[str],
    excluded_skills: frozenset[str],
    min_skill_frequency: int,
) -> tuple[set[str], dict[str, int]]:
    """Return skills sharing a selected posting with another candidate skill."""
    eligible_skills = {
        skill
        for skill, frequency in skill_frequency.items()
        if frequency >= min_skill_frequency
        and normalize_skill_key(skill) not in excluded_skills
    }
    skills_with_cooccurrence: set[str] = set()
    stats = {
        "postings_checked_for_cooccurrence": 0,
        "singleton_candidate_skill_postings": 0,
        "multi_candidate_skill_postings": 0,
    }
    connection = connect_index(db_path, readonly=True)
    try:
        for input_file in tqdm(
            input_files,
            total=len(input_files),
            desc="Checking skill cooccurrence",
            unit="file",
        ):
            posting_skills: dict[tuple[str, int], set[str]] = {}
            for source_path, row_idx, true_skill in connection.execute(
                """
                SELECT source_path, row_idx, true_skill
                FROM pair_choice
                WHERE file_id = ?
                """,
                (input_file.file_id,),
            ):
                skill = str(true_skill)
                if skill not in eligible_skills:
                    continue
                posting_skills.setdefault(
                    (str(source_path), int(row_idx)), set()
                ).add(skill)

            for skills in posting_skills.values():
                stats["postings_checked_for_cooccurrence"] += 1
                if len(skills) >= 2:
                    stats["multi_candidate_skill_postings"] += 1
                    skills_with_cooccurrence.update(skills)
                else:
                    stats["singleton_candidate_skill_postings"] += 1

    finally:
        connection.close()
    return skills_with_cooccurrence, stats


def write_duplicate_examples(path: Path, examples: list[DuplicateExample]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, quoting=csv.QUOTE_ALL, lineterminator="\n")
            writer.writerow(
                [
                    "masked_sentence",
                    "true_skill",
                    "selected_relative_path_at_detection",
                    "selected_sample_idx_at_detection",
                    "selected_file_path",
                    "selected_row_idx",
                    "duplicate_relative_path",
                    "duplicate_sample_idx",
                    "duplicate_file_path",
                    "duplicate_row_idx",
                ]
            )
            for example in examples:
                writer.writerow(
                    [
                        example.masked_sentence,
                        example.true_skill,
                        example.selected_relative_path,
                        example.selected_sample_idx,
                        example.selected_file_path,
                        example.selected_row_idx,
                        example.duplicate_relative_path,
                        example.duplicate_sample_idx,
                        example.duplicate_file_path,
                        example.duplicate_row_idx,
                    ]
                )
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return len(examples)


def init_filter_worker(
    db_path: str,
    output_root: str,
    skill_frequency: dict[str, int],
    excluded_skills: frozenset[str],
    skills_with_cooccurrence: frozenset[str],
    min_skill_frequency: int,
    compression_level: int,
) -> None:
    global _READ_CONN, _OUTPUT_ROOT, _SKILL_FREQUENCY, _EXCLUDED_SKILLS
    global _MIN_SKILL_FREQUENCY, _COMPRESSION_LEVEL, _SKILLS_WITH_COOCCURRENCE
    if _READ_CONN is not None:
        _READ_CONN.close()
    _READ_CONN = connect_index(Path(db_path), readonly=True)
    _OUTPUT_ROOT = Path(output_root)
    _SKILL_FREQUENCY = skill_frequency
    _EXCLUDED_SKILLS = excluded_skills
    _SKILLS_WITH_COOCCURRENCE = skills_with_cooccurrence
    _MIN_SKILL_FREQUENCY = min_skill_frequency
    _COMPRESSION_LEVEL = compression_level


def selected_sample_indices(file_id: int) -> set[int]:
    if _READ_CONN is None:
        raise RuntimeError("filter worker was not initialized")
    return {
        int(row[0])
        for row in _READ_CONN.execute(
            "SELECT sample_idx FROM pair_choice WHERE file_id = ?", (file_id,)
        )
    }


def filter_one_file(input_file: InputFile) -> FilteredFileResult:
    if _OUTPUT_ROOT is None:
        raise RuntimeError("filter worker was not initialized")
    selected_indices = selected_sample_indices(input_file.file_id)
    output_path = _OUTPUT_ROOT / input_file.rel_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(
        f"{output_path.stem}.tmp.{os.getpid()}{output_path.suffix}"
    )

    input_rows = 0
    selected_pair_rows = 0
    rows_dropped_excluded_skill = 0
    rows_dropped_no_cooccurrence_skill = 0
    rows_dropped_rare_skill = 0
    output_rows = 0
    output_posting_ids: set[str] = set()
    try:
        with open_input_csv(input_file.path) as input_handle, open_output_csv(
            temp_path, _COMPRESSION_LEVEL
        ) as output_handle:
            reader = csv.reader(normalized_csv_lines(input_handle))
            writer = csv.writer(output_handle, lineterminator="\n")
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError(f"{input_file.path} has no header") from exc
            positions = header_positions(
                header, REQUIRED_OUTPUT_COLUMNS, input_file.path
            )
            output_indices = [
                i for i, column in enumerate(header) if column != "body"
            ]
            writer.writerow([header[i] for i in output_indices])
            max_position = max(positions.values())
            for sample_idx, row in enumerate(reader):
                if len(row) <= max_position:
                    raise ValueError(f"{input_file.path} contains a short CSV row")
                input_rows += 1
                if sample_idx not in selected_indices:
                    continue
                selected_pair_rows += 1
                true_skill = row[positions["true_skill"]]
                if normalize_skill_key(true_skill) in _EXCLUDED_SKILLS:
                    rows_dropped_excluded_skill += 1
                    continue
                if _SKILL_FREQUENCY.get(true_skill, 0) < _MIN_SKILL_FREQUENCY:
                    rows_dropped_rare_skill += 1
                    continue
                if true_skill not in _SKILLS_WITH_COOCCURRENCE:
                    rows_dropped_no_cooccurrence_skill += 1
                    continue
                writer.writerow([row[i] for i in output_indices])
                output_rows += 1
                output_posting_ids.add(
                    row[positions["posting_fingerprint"]]
                )
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    if selected_pair_rows != len(selected_indices):
        raise RuntimeError(
            f"selected-row accounting failed for {input_file.path}: "
            f"index={len(selected_indices):,}, read={selected_pair_rows:,}"
        )
    return FilteredFileResult(
        file_id=input_file.file_id,
        rel_path=input_file.rel_path,
        input_rows=input_rows,
        selected_pair_rows=selected_pair_rows,
        rows_dropped_excluded_skill=rows_dropped_excluded_skill,
        rows_dropped_rare_skill=rows_dropped_rare_skill,
        rows_dropped_no_cooccurrence_skill=rows_dropped_no_cooccurrence_skill,
        output_rows=output_rows,
        output_postings=len(output_posting_ids),
    )


def write_filtered_files(
    db_path: Path,
    input_files: list[InputFile],
    output_root: Path,
    skill_frequency: Counter[str],
    skills_with_cooccurrence: frozenset[str],
    excluded_skills: frozenset[str],
    min_skill_frequency: int,
    compression_level: int,
    workers: int,
) -> list[FilteredFileResult]:
    init_args = (
        str(db_path),
        str(output_root),
        dict(skill_frequency),
        excluded_skills,
        skills_with_cooccurrence,
        min_skill_frequency,
        compression_level,
    )
    if workers == 1:
        init_filter_worker(*init_args)
        results: Iterator[FilteredFileResult] = map(filter_one_file, input_files)
        pool = None
    else:
        pool = Pool(
            processes=workers,
            initializer=init_filter_worker,
            initargs=init_args,
        )
        results = pool.imap_unordered(filter_one_file, input_files, chunksize=1)

    collected: list[FilteredFileResult] = []
    try:
        for result in tqdm(
            results,
            total=len(input_files),
            desc="Writing filtered files",
            unit="file",
        ):
            collected.append(result)
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
        raise
    else:
        if pool is not None:
            pool.close()
            pool.join()
    return sorted(collected, key=lambda item: item.file_id)


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        temp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_skill_frequency(
    path: Path,
    skill_frequency: Counter[str],
    excluded_skills: frozenset[str],
    skills_with_cooccurrence: frozenset[str],
    min_skill_frequency: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                [
                    "true_skill",
                    "unique_pair_frequency",
                    "has_other_skill_cooccurrence",
                    "kept",
                ]
            )
            for skill in sorted(skill_frequency):
                frequency = skill_frequency[skill]
                writer.writerow(
                    [
                        skill,
                        frequency,
                        int(skill in skills_with_cooccurrence),
                        int(
                            frequency >= min_skill_frequency
                            and normalize_skill_key(skill) not in excluded_skills
                            and skill in skills_with_cooccurrence
                        ),
                    ]
                )
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_file_manifest(
    path: Path,
    input_root: Path,
    output_root: Path,
    results: list[FilteredFileResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                [
                    "relative_path",
                    "input_path",
                    "output_path",
                    "input_rows",
                    "selected_unique_pair_rows",
                    "rows_dropped_excluded_skill",
                    "rows_dropped_rare_skill",
                    "output_rows",
                    "rows_dropped_no_cooccurrence_skill",
                    "output_postings",
                ]
            )
            for result in results:
                writer.writerow(
                    [
                        result.rel_path,
                        input_root / result.rel_path,
                        output_root / result.rel_path,
                        result.input_rows,
                        result.selected_pair_rows,
                        result.rows_dropped_excluded_skill,
                        result.rows_dropped_rare_skill,
                        result.output_rows,
                        result.rows_dropped_no_cooccurrence_skill,
                        result.output_postings,
                    ]
                )
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def validate_separate_roots(input_root: Path, output_root: Path) -> None:
    input_resolved = input_root.resolve()
    output_resolved = output_root.resolve()
    if (
        input_resolved == output_resolved
        or input_resolved in output_resolved.parents
        or output_resolved in input_resolved.parents
    ):
        raise ValueError(
            "--output-root must be separate from --input-root and cannot be "
            "its parent or child"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Globally deduplicate exact (masked_sentence, true_skill) pairs "
            "and filter globally rare true-skill rows."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument(
        "--dedup-seed",
        type=int,
        default=DEFAULT_DEDUP_SEED,
        help="Seed for deterministic representative selection (default: 42).",
    )
    parser.add_argument(
        "--min-skill-frequency",
        type=int,
        default=DEFAULT_MIN_SKILL_FREQUENCY,
        help="Minimum globally unique-pair frequency (default: 100).",
    )
    parser.add_argument(
        "--exclude-skill",
        action="append",
        default=[],
        metavar="SKILL",
        help="Drop this normalized skill; repeat to exclude multiple skills.",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--compression-level", type=int, default=1,
        help="gzip output compression level from 1 to 9 (default: 1)."
    )
    parser.add_argument(
        "--db", type=Path,
        help="SQLite index (default: <output-root>/filter_data_index.sqlite)."
    )
    parser.add_argument(
        "--summary-json", type=Path,
        help="Summary path (default: <output-root>/filter_summary.json)."
    )
    parser.add_argument(
        "--duplicate-examples-csv", type=Path,
        help="Examples path (default: <output-root>/duplicate_examples.csv)."
    )
    parser.add_argument(
        "--duplicate-example-limit",
        type=int,
        default=DEFAULT_DUPLICATE_EXAMPLE_LIMIT,
        help="Maximum duplicate pair examples to save (default: 100).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.input_root = args.input_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.exclude_skill = frozenset(
        normalize_skill_key(skill) for skill in args.exclude_skill
    )
    if "" in args.exclude_skill:
        raise ValueError("--exclude-skill must not be empty")
    if not args.input_root.is_dir():
        raise FileNotFoundError(f"input root not found: {args.input_root}")
    validate_separate_roots(args.input_root, args.output_root)
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.min_skill_frequency < 0:
        raise ValueError("--min-skill-frequency must be >= 0")
    if args.dedup_seed < 0:
        raise ValueError("--dedup-seed must be >= 0")
    if args.duplicate_example_limit < 0:
        raise ValueError("--duplicate-example-limit must be >= 0")
    if not 1 <= args.compression_level <= 9:
        raise ValueError("--compression-level must be between 1 and 9")

    input_files = iter_input_files(args.input_root, args.pattern)
    if not input_files:
        raise FileNotFoundError(
            f"no files matching {args.pattern!r} under {args.input_root}"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    db_path = (
        args.db.expanduser().resolve()
        if args.db is not None
        else args.output_root / "filter_data_index.sqlite"
    )
    summary_path = (
        args.summary_json.expanduser().resolve()
        if args.summary_json is not None
        else args.output_root / "filter_summary.json"
    )
    duplicate_examples_path = (
        args.duplicate_examples_csv.expanduser().resolve()
        if args.duplicate_examples_csv is not None
        else args.output_root / "duplicate_examples.csv"
    )

    print(f"input root:              {args.input_root}")
    print(f"output root:             {args.output_root}")
    print(f"input files:             {len(input_files):,}")
    print(f"dedup columns:           {PAIR_COLUMNS}")
    print(f"dedup seed:              {args.dedup_seed}")
    print(
        "skill frequency rule:    keep unique-pair frequency >= "
        f"{args.min_skill_frequency}"
    )
    print(
        "excluded skills:         "
        + (", ".join(sorted(args.exclude_skill)) or "none")
    )
    print(f"workers:                 {args.workers}")
    print(f"SQLite index:            {db_path}")

    pair_stats, skill_frequency, duplicate_examples = index_global_pairs(
        db_path=db_path,
        input_files=input_files,
        workers=args.workers,
        dedup_seed=args.dedup_seed,
        duplicate_example_limit=args.duplicate_example_limit,
    )
    skills_with_cooccurrence, cooccurrence_stats = find_skills_with_cooccurrence(
        db_path=db_path,
        input_files=input_files,
        skill_frequency=skill_frequency,
        excluded_skills=args.exclude_skill,
        min_skill_frequency=args.min_skill_frequency,
    )
    duplicate_examples_written = write_duplicate_examples(
        duplicate_examples_path, duplicate_examples
    )
    file_results = write_filtered_files(
        db_path=db_path,
        input_files=input_files,
        output_root=args.output_root,
        skill_frequency=skill_frequency,
        excluded_skills=args.exclude_skill,
        skills_with_cooccurrence=frozenset(skills_with_cooccurrence),
        min_skill_frequency=args.min_skill_frequency,
        compression_level=args.compression_level,
        workers=args.workers,
    )

    kept_skills = sorted(skills_with_cooccurrence)
    skill2idx = {skill: index for index, skill in enumerate(kept_skills)}
    write_skill_frequency(
        args.output_root / "skill_frequency.csv",
        skill_frequency,
        args.exclude_skill,
        frozenset(skills_with_cooccurrence),
        args.min_skill_frequency,
    )
    atomic_write_json(args.output_root / "skill2idx.json", skill2idx)
    write_file_manifest(
        args.output_root / "filtered_files.csv",
        args.input_root,
        args.output_root,
        file_results,
    )

    output_rows = sum(result.output_rows for result in file_results)
    selected_rows = sum(result.selected_pair_rows for result in file_results)
    rows_dropped_excluded_skill = sum(
        result.rows_dropped_excluded_skill for result in file_results
    )
    rows_dropped_rare_skill = sum(
        result.rows_dropped_rare_skill for result in file_results
    )
    rows_dropped_no_cooccurrence_skill = sum(
        result.rows_dropped_no_cooccurrence_skill for result in file_results
    )
    if selected_rows != pair_stats["unique_pairs"]:
        raise RuntimeError("selected unique-pair accounting does not balance")
    if selected_rows != (
        output_rows
        + rows_dropped_excluded_skill
        + rows_dropped_rare_skill
        + rows_dropped_no_cooccurrence_skill
    ):
        raise RuntimeError(
            "output row accounting does not balance: "
            f"selected={selected_rows}, output={output_rows}, "
            f"excluded={rows_dropped_excluded_skill}, rare={rows_dropped_rare_skill}, "
            f"no_cooccurrence={rows_dropped_no_cooccurrence_skill}"
        )

    summary = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "pattern": args.pattern,
        "input_files": len(input_files),
        "dedup_columns": list(PAIR_COLUMNS),
        "pair_hash": "sha256_length_prefixed_utf8_exact_values",
        "representative_policy": "minimum_seeded_blake2b_priority",
        "dedup_seed": args.dedup_seed,
        "row_identity": ["input_file_id", "sample_idx"],
        "min_skill_frequency": args.min_skill_frequency,
        "skill_frequency_population": "globally_unique_pairs",
        "excluded_skills": sorted(args.exclude_skill),
        "sqlite_index": str(db_path),
        "duplicate_examples_csv": str(duplicate_examples_path),
        "duplicate_example_limit": args.duplicate_example_limit,
        "duplicate_examples_written": duplicate_examples_written,
        **pair_stats,
        **cooccurrence_stats,
        "cooccurrence_scope": (
            "globally_unique_pairs_grouped_by_(file_path,row_idx)"
        ),
        "distinct_skills_before_frequency_filter": len(skill_frequency),
        "skills_kept": len(kept_skills),
        "skills_dropped_no_other_skill_cooccurrence": (
            sum(
                frequency >= args.min_skill_frequency
                and normalize_skill_key(skill) not in args.exclude_skill
                for skill, frequency in skill_frequency.items()
            )
            - len(kept_skills)
        ),
        "rows_dropped_excluded_skill": rows_dropped_excluded_skill,
        "rows_dropped_rare_skill": rows_dropped_rare_skill,
        "rows_dropped_no_other_skill_cooccurrence": (
            rows_dropped_no_cooccurrence_skill
        ),
        "output_rows": output_rows,
        "output_postings": sum(result.output_postings for result in file_results),
    }
    atomic_write_json(summary_path, summary)

    print()
    print(f"input rows:                  {summary['input_rows']:,}")
    print(f"valid pair rows:             {summary['valid_pair_rows']:,}")
    print(f"globally unique pairs:       {summary['unique_pairs']:,}")
    print(
        f"postings after pair dedup:    {summary['unique_pair_postings']:,}"
    )
    print(f"duplicate pair rows dropped: {summary['duplicate_pair_rows']:,}")
    print(f"rare-skill rows dropped:     {summary['rows_dropped_rare_skill']:,}")
    print(f"excluded-skill rows dropped: {summary['rows_dropped_excluded_skill']:,}")
    print(
        "no-cooccurrence rows dropped: "
        f"{summary['rows_dropped_no_other_skill_cooccurrence']:,}"
    )
    print(f"output rows:                 {summary['output_rows']:,}")
    print(f"output postings:             {summary['output_postings']:,}")
    print(f"kept skills:                 {summary['skills_kept']:,}")
    print(f"summary:                     {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
