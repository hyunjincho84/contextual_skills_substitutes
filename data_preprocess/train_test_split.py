#!/usr/bin/env python3
"""Split filtered masked-skill data into train/test sets by posting.

The posting identity is the normalized-body ``posting_fingerprint``. A seeded
every distinct posting a deterministic random order, and the first globally
selected 20% (rounded to the nearest posting) become the test set.  Therefore,
all masked-skill rows from one posting always remain in the same split.

Input paths are mirrored below both output directories::

    <output-root>/train/<input-relative-path>
    <output-root>/test/<input-relative-path>

The global posting index is kept in a temporary SQLite database so the full
dataset does not need to fit in memory.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from multiprocessing import Pool
from pathlib import Path
from typing import TextIO
from tqdm import tqdm


DEFAULT_INPUT_ROOT = Path("/home/jovyan/LEM_data3/hyunjincho/LEM/all_filterd")
DEFAULT_OUTPUT_ROOT = Path(
    "/home/jovyan/LEM_data3/hyunjincho/LEM/all_train_test"
)
DEFAULT_PATTERN = "preprocessed_*.csv.gz"
DEFAULT_TEST_RATIO = Decimal("0.2")
DEFAULT_SEED = 42
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)
DEFAULT_COMPRESSION_LEVEL = 1
REQUIRED_COLUMNS = ("posting_fingerprint",)
RANDOM_ORDER_METHOD = "blake2b_64(seed,posting_fingerprint)"


@dataclass(frozen=True)
class InputFile:
    file_id: int
    path: Path
    rel_path: str


@dataclass(frozen=True)
class IndexedFile:
    file_id: int
    path: Path
    rel_path: str
    input_rows: int
    input_postings: int


@dataclass(frozen=True)
class SplitFileResult:
    file_id: int
    rel_path: str
    input_rows: int
    input_postings: int
    train_rows: int
    train_postings: int
    test_rows: int
    test_postings: int


_READ_CONN: sqlite3.Connection | None = None
_TRAIN_ROOT: Path | None = None
_TEST_ROOT: Path | None = None
_COMPRESSION_LEVEL = DEFAULT_COMPRESSION_LEVEL


def iter_input_files(root: Path, pattern: str) -> list[InputFile]:
    paths = sorted(path for path in root.rglob(pattern) if path.is_file())
    return [
        InputFile(
            file_id=file_id,
            path=path,
            rel_path=str(path.relative_to(root)),
        )
        for file_id, path in enumerate(paths)
    ]


def open_input_csv(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


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
    header: list[str],
    required: tuple[str, ...],
    path: Path,
) -> dict[str, int]:
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    return {column: header.index(column) for column in required}



def validate_posting_fingerprint(value: str, path: Path) -> str:
    fingerprint = str(value).strip().lower()
    if len(fingerprint) != 64:
        raise ValueError(
            f"{path} has invalid posting_fingerprint length: {value!r}"
        )
    try:
        bytes.fromhex(fingerprint)
    except ValueError as exc:
        raise ValueError(
            f"{path} has non-hex posting_fingerprint: {value!r}"
        ) from exc
    return fingerprint


def posting_random_key(seed: int, posting_fingerprint: str) -> bytes:
    digest = hashlib.blake2b(digest_size=8, person=b"LEMsplit-v2")
    digest.update(str(seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(posting_fingerprint.encode("ascii"))
    return digest.digest()


def initialize_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        """
        CREATE TABLE postings (
            posting_fingerprint TEXT PRIMARY KEY,
            random_key BLOB NOT NULL,
            is_test INTEGER NOT NULL DEFAULT 0 CHECK (is_test IN (0, 1))
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE file_postings (
            file_id INTEGER NOT NULL,
            posting_fingerprint TEXT NOT NULL,
            PRIMARY KEY (file_id, posting_fingerprint)
        ) WITHOUT ROWID
        """
    )
    return connection


def index_input_file(
    connection: sqlite3.Connection,
    input_file: InputFile,
    seed: int,
) -> IndexedFile:
    posting_fingerprints: set[str] = set()
    input_rows = 0

    with open_input_csv(input_file.path) as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(
                f"{input_file.path} is empty and has no CSV header"
            ) from exc

        positions = header_positions(header, REQUIRED_COLUMNS, input_file.path)
        max_position = max(positions.values())

        for row in reader:
            if len(row) <= max_position:
                raise ValueError(f"{input_file.path} contains a short CSV row")
            input_rows += 1
            posting_fingerprints.add(
                validate_posting_fingerprint(
                    row[positions["posting_fingerprint"]],
                    input_file.path,
                )
            )

    records = [
        (
            posting_fingerprint,
            posting_random_key(seed, posting_fingerprint),
        )
        for posting_fingerprint in sorted(posting_fingerprints)
    ]
    with connection:
        connection.executemany(
            """
            INSERT OR IGNORE INTO postings (
                posting_fingerprint, random_key
            ) VALUES (?, ?)
            """,
            records,
        )
        connection.executemany(
            """
            INSERT INTO file_postings (
                file_id, posting_fingerprint
            ) VALUES (?, ?)
            """,
            [
                (input_file.file_id, posting_fingerprint)
                for posting_fingerprint in sorted(posting_fingerprints)
            ],
        )

    return IndexedFile(
        file_id=input_file.file_id,
        path=input_file.path,
        rel_path=input_file.rel_path,
        input_rows=input_rows,
        input_postings=len(posting_fingerprints),
    )

def nearest_split_count(total: int, ratio: Decimal) -> int:
    if total == 0:
        return 0
    count = int(Decimal(total) * ratio + Decimal("0.5"))
    if total >= 2:
        count = max(1, min(total - 1, count))
    return count


def assign_test_postings(
    connection: sqlite3.Connection,
    test_ratio: Decimal,
) -> tuple[int, int]:
    total_postings = int(
        connection.execute("SELECT COUNT(*) FROM postings").fetchone()[0]
    )
    test_postings = nearest_split_count(total_postings, test_ratio)

    connection.execute(
        """
        CREATE INDEX posting_random_order
        ON postings(random_key, posting_fingerprint)
        """
    )
    if test_postings:
        with connection:
            connection.execute(
                """
                UPDATE postings
                SET is_test = 1
                WHERE rowid IN (
                    SELECT rowid
                    FROM postings
                    ORDER BY random_key, posting_fingerprint
                    LIMIT ?
                )
                """,
                (test_postings,),
            )

    assigned_test = int(
        connection.execute(
            "SELECT COUNT(*) FROM postings WHERE is_test = 1"
        ).fetchone()[0]
    )
    if assigned_test != test_postings:
        raise RuntimeError(
            f"expected {test_postings} test postings, assigned {assigned_test}"
        )
    return total_postings, test_postings


def init_split_worker(
    db_path: str,
    train_root: str,
    test_root: str,
    compression_level: int,
) -> None:
    global _READ_CONN, _TRAIN_ROOT, _TEST_ROOT, _COMPRESSION_LEVEL

    db_uri = f"file:{Path(db_path).as_posix()}?mode=ro&immutable=1"
    _READ_CONN = sqlite3.connect(db_uri, uri=True)
    _TRAIN_ROOT = Path(train_root)
    _TEST_ROOT = Path(test_root)
    _COMPRESSION_LEVEL = compression_level


def temporary_output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.tmp.{os.getpid()}{path.suffix}")


def split_input_file(indexed_file: IndexedFile) -> SplitFileResult:
    if _READ_CONN is None or _TRAIN_ROOT is None or _TEST_ROOT is None:
        raise RuntimeError("split worker was not initialized")

    train_path = _TRAIN_ROOT / indexed_file.rel_path
    test_path = _TEST_ROOT / indexed_file.rel_path
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    train_temp = temporary_output_path(train_path)
    test_temp = temporary_output_path(test_path)

    assignments = {
        str(posting_fingerprint): int(is_test)
        for posting_fingerprint, is_test in _READ_CONN.execute(
            """
            SELECT fp.posting_fingerprint, p.is_test
            FROM file_postings AS fp
            JOIN postings AS p USING (posting_fingerprint)
            WHERE fp.file_id = ?
            """,
            (indexed_file.file_id,),
        )
    }

    input_rows = 0
    train_rows = 0
    test_rows = 0
    train_postings: set[str] = set()
    test_postings: set[str] = set()

    try:
        with (
            open_input_csv(indexed_file.path) as input_handle,
            open_output_csv(train_temp, _COMPRESSION_LEVEL) as train_handle,
            open_output_csv(test_temp, _COMPRESSION_LEVEL) as test_handle,
        ):
            reader = csv.reader(input_handle)
            train_writer = csv.writer(train_handle, lineterminator="\n")
            test_writer = csv.writer(test_handle, lineterminator="\n")

            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError(
                    f"{indexed_file.path} is empty and has no CSV header"
                ) from exc

            positions = header_positions(
                header,
                REQUIRED_COLUMNS,
                indexed_file.path,
            )
            max_position = max(positions.values())
            train_writer.writerow(header)
            test_writer.writerow(header)

            for row in reader:
                if len(row) <= max_position:
                    raise ValueError(
                        f"{indexed_file.path} contains a short CSV row"
                    )

                input_rows += 1
                posting_fingerprint = validate_posting_fingerprint(
                    row[positions["posting_fingerprint"]],
                    indexed_file.path,
                )
                is_test = assignments.get(posting_fingerprint)
                if is_test is None:
                    raise RuntimeError(
                        "missing split assignment for posting_fingerprint="
                        f"{posting_fingerprint}"
                    )

                if is_test:
                    test_writer.writerow(row)
                    test_rows += 1
                    test_postings.add(posting_fingerprint)
                else:
                    train_writer.writerow(row)
                    train_rows += 1
                    train_postings.add(posting_fingerprint)

        if input_rows != indexed_file.input_rows:
            raise RuntimeError(
                f"{indexed_file.path} row count changed between passes: "
                f"{indexed_file.input_rows} -> {input_rows}"
            )
        if train_rows + test_rows != input_rows:
            raise RuntimeError(f"row accounting failed for {indexed_file.path}")
        if len(train_postings) + len(test_postings) != indexed_file.input_postings:
            raise RuntimeError(f"posting accounting failed for {indexed_file.path}")

        os.replace(train_temp, train_path)
        os.replace(test_temp, test_path)
    finally:
        for temp_path in (train_temp, test_temp):
            if temp_path.exists():
                temp_path.unlink()

    return SplitFileResult(
        file_id=indexed_file.file_id,
        rel_path=indexed_file.rel_path,
        input_rows=input_rows,
        input_postings=indexed_file.input_postings,
        train_rows=train_rows,
        train_postings=len(train_postings),
        test_rows=test_rows,
        test_postings=len(test_postings),
    )

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


def reject_unexpected_output_files(
    output_root: Path,
    pattern: str,
    expected_relative_paths: set[str],
) -> None:
    unexpected: list[Path] = []
    for split_name in ("train", "test"):
        split_root = output_root / split_name
        if not split_root.is_dir():
            continue
        for path in split_root.rglob(pattern):
            if path.is_file() and str(path.relative_to(split_root)) not in (
                expected_relative_paths
            ):
                unexpected.append(path)

    if unexpected:
        preview = "\n".join(f"  - {path}" for path in unexpected[:10])
        suffix = "" if len(unexpected) <= 10 else "\n  - ..."
        raise RuntimeError(
            "output contains split files that are not in the current input; "
            "remove or move them before running to avoid mixing datasets:\n"
            f"{preview}{suffix}"
        )


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_file_report(
    path: Path,
    input_root: Path,
    output_root: Path,
    results: list[SplitFileResult],
) -> None:
    temp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                [
                    "relative_path",
                    "input_path",
                    "train_path",
                    "test_path",
                    "input_rows",
                    "input_postings",
                    "train_rows",
                    "train_postings",
                    "test_rows",
                    "test_postings",
                ]
            )
            for result in results:
                writer.writerow(
                    [
                        result.rel_path,
                        input_root / result.rel_path,
                        output_root / "train" / result.rel_path,
                        output_root / "test" / result.rel_path,
                        result.input_rows,
                        result.input_postings,
                        result.train_rows,
                        result.train_postings,
                        result.test_rows,
                        result.test_postings,
                    ]
                )
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Globally split filtered data 80:20 by posting."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument(
        "--test-ratio",
        type=Decimal,
        default=DEFAULT_TEST_RATIO,
        help="Global test-posting ratio (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for deterministic posting randomization (default: 42).",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--compression-level",
        type=int,
        default=DEFAULT_COMPRESSION_LEVEL,
        help="gzip output compression level from 1 to 9 (default: 1).",
    )
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    args.input_root = args.input_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()

    if not args.input_root.is_dir():
        raise FileNotFoundError(f"input root not found: {args.input_root}")
    validate_separate_roots(args.input_root, args.output_root)
    if not Decimal("0") < args.test_ratio < Decimal("1"):
        raise ValueError("--test-ratio must be between 0 and 1")
    if args.seed < 0:
        raise ValueError("--seed must be >= 0")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if not 1 <= args.compression_level <= 9:
        raise ValueError("--compression-level must be between 1 and 9")

    input_files = iter_input_files(args.input_root, args.pattern)
    if not input_files:
        raise FileNotFoundError(
            f"no files matching {args.pattern!r} under {args.input_root}"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    train_root = args.output_root / "train"
    test_root = args.output_root / "test"
    train_root.mkdir(parents=True, exist_ok=True)
    test_root.mkdir(parents=True, exist_ok=True)

    expected_relative_paths = {item.rel_path for item in input_files}
    reject_unexpected_output_files(
        args.output_root,
        args.pattern,
        expected_relative_paths,
    )

    print(f"input root:       {args.input_root}")
    print(f"output root:      {args.output_root}")
    print(f"input files:      {len(input_files):,}")
    print("posting identity: posting_fingerprint")
    print(f"test ratio:       {args.test_ratio}")
    print(f"seed:             {args.seed}")
    print(f"workers:          {args.workers}")

    with tempfile.TemporaryDirectory(
        prefix=".train_test_split_work_",
        dir=args.output_root,
    ) as work_dir:
        db_path = Path(work_dir) / "posting_assignments.sqlite"
        connection = initialize_database(db_path)
        indexed_files: list[IndexedFile] = []

        try:
            for input_file in tqdm(
                input_files,
                total=len(input_files),
                desc="Indexing posting fingerprints",
                unit="file",
            ):
                indexed_files.append(
                    index_input_file(connection, input_file, args.seed)
                )

            total_postings, test_postings = assign_test_postings(
                connection,
                args.test_ratio,
            )
            connection.commit()
        finally:
            connection.close()

        train_postings = total_postings - test_postings
        print(f"global postings:  {total_postings:,}")
        print(f"train postings:   {train_postings:,}")
        print(f"test postings:    {test_postings:,}")

        results: list[SplitFileResult] = []
        with Pool(
            processes=args.workers,
            initializer=init_split_worker,
            initargs=(
                str(db_path),
                str(train_root),
                str(test_root),
                args.compression_level,
            ),
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(
                    split_input_file,
                    indexed_files,
                    chunksize=1,
                ),
                total=len(indexed_files),
                desc="Writing train/test files",
                unit="file",
            ):
                results.append(result)

    results.sort(key=lambda item: item.file_id)
    input_rows = sum(result.input_rows for result in results)
    train_rows = sum(result.train_rows for result in results)
    test_rows = sum(result.test_rows for result in results)
    if input_rows != train_rows + test_rows:
        raise RuntimeError("global row accounting failed")

    posting_references = sum(result.input_postings for result in results)
    duplicate_posting_references = posting_references - total_postings
    if duplicate_posting_references < 0:
        raise RuntimeError("global posting accounting failed")

    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config_payload = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "pattern": args.pattern,
        "posting_identity": ["posting_fingerprint"],
        "random_order_method": RANDOM_ORDER_METHOD,
        "seed": args.seed,
        "test_ratio": str(args.test_ratio),
    }
    configuration_hash = hashlib.sha256(
        json.dumps(
            config_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    summary: dict[str, object] = {
        **config_payload,
        "configuration_sha256": configuration_hash,
        "script_sha256": script_hash,
        "python_version": sys.version.split()[0],
        "sqlite_version": sqlite3.sqlite_version,
        "input_files": len(results),
        "input_rows": input_rows,
        "global_postings": total_postings,
        "posting_references_across_files": posting_references,
        "duplicate_posting_references_across_files": (
            duplicate_posting_references
        ),
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_postings": train_postings,
        "test_postings": test_postings,
        "actual_train_posting_ratio": (
            train_postings / total_postings if total_postings else 0.0
        ),
        "actual_test_posting_ratio": (
            test_postings / total_postings if total_postings else 0.0
        ),
        "actual_train_row_ratio": train_rows / input_rows if input_rows else 0.0,
        "actual_test_row_ratio": test_rows / input_rows if input_rows else 0.0,
        "train_root": str(train_root),
        "test_root": str(test_root),
    }

    report_path = args.output_root / "split_files.csv"
    summary_path = args.output_root / "split_summary.json"
    write_file_report(report_path, args.input_root, args.output_root, results)
    atomic_write_json(summary_path, summary)

    print(f"train rows:       {train_rows:,}")
    print(f"test rows:        {test_rows:,}")
    print(f"file report:      {report_path}")
    print(f"summary:          {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
