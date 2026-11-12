#!/usr/bin/env python3
"""Pretrain BERT MLM on original body text from train postings only.

Train split CSVs are used only to recover unique (file_path, row_idx) posting
IDs. Each ID is then resolved against its original raw CSV. Only body text
after the first case-insensitive ``job description:`` marker is fed to MLM
once per epoch. Test split files are never read.
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
import traceback
from pathlib import Path
from typing import Iterator, TextIO

import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info
from transformers import (
    BertForMaskedLM,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


BASE_DATA_DIR = Path(
    os.environ.get("BASE_DATA_DIR", "/home/jovyan/LEM_data2/data")
)
DEFAULT_TRAIN_ROOT = Path(
    os.environ.get(
        "TRAIN_SPLIT_ROOT",
        "/home/jovyan/LEM_data3/hyunjincho/LEM/all_train_test/train",
    )
)
DEFAULT_MODEL_NAME = os.environ.get("BERT_BASE_MODEL", "bert-base-uncased")
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("BERT_PRETRAIN_DIR", str(BASE_DATA_DIR / "bert_pretrained"))
)
DEFAULT_PATTERN = "preprocessed_*.csv.gz"
DEFAULT_MAX_LENGTH = 256
DEFAULT_RAW_CHUNK_SIZE = 100_000
DEFAULT_BODY_START_MARKER = "job description:"
DEFAULT_MLM_PROBABILITY = 0.15
DEFAULT_EPOCHS = 1.0
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_SEED = 42
DEFAULT_SAVE_STEPS = 5_000
DEFAULT_LOGGING_STEPS = 500
DEFAULT_DATALOADER_WORKERS = 16
DEFAULT_PROGRESS_EVERY = 500
DEFAULT_DATALOADER_PREFETCH_FACTOR = 4
REQUIRED_COLUMNS = ("row_idx", "file_path")


def open_csv(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def header_positions(header: list[str], path: Path) -> dict[str, int]:
    missing = [column for column in REQUIRED_COLUMNS if column not in header]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    return {column: header.index(column) for column in REQUIRED_COLUMNS}


def parse_row_idx(value: str, path: Path) -> int:
    try:
        row_idx = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} has invalid row_idx={value!r}") from exc
    if row_idx < 0:
        raise ValueError(f"{path} has negative row_idx={row_idx}")
    return row_idx


def scan_train_file(path: Path) -> tuple[str | None, int, list[int]]:
    raw_path: str | None = None
    masked_rows = 0
    posting_indices: set[int] = set()

    with open_csv(path) as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path} is empty and has no header") from exc
        positions = header_positions(header, path)
        max_position = max(positions.values())

        for row in reader:
            if len(row) <= max_position:
                raise ValueError(f"{path} contains a short CSV row")
            masked_rows += 1
            row_raw_path = row[positions["file_path"]]
            if not row_raw_path:
                raise ValueError(f"{path} contains an empty file_path")
            if raw_path is None:
                raw_path = row_raw_path
            elif row_raw_path != raw_path:
                raise ValueError(f"{path} contains more than one file_path")
            posting_indices.add(
                parse_row_idx(row[positions["row_idx"]], path)
            )

    return raw_path, masked_rows, sorted(posting_indices)


def initialize_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.executescript(
        """
        CREATE TABLE source_files (
            file_id INTEGER PRIMARY KEY,
            relative_path TEXT NOT NULL UNIQUE,
            split_path TEXT NOT NULL UNIQUE,
            raw_path TEXT NOT NULL UNIQUE,
            masked_rows INTEGER NOT NULL,
            train_postings INTEGER NOT NULL,
            split_size INTEGER NOT NULL,
            split_mtime_ns INTEGER NOT NULL,
            raw_size INTEGER NOT NULL,
            raw_mtime_ns INTEGER NOT NULL
        );

        CREATE TABLE postings (
            file_id INTEGER NOT NULL,
            row_idx INTEGER NOT NULL,
            PRIMARY KEY (file_id, row_idx),
            FOREIGN KEY (file_id) REFERENCES source_files(file_id)
        );

        PRAGMA user_version = 1;
        """
    )
    return connection


def signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime_ns)


def build_posting_index(
    train_root: Path,
    pattern: str,
    database_path: Path,
    progress_every: int,
) -> dict[str, int]:
    split_paths = sorted(
        path for path in train_root.rglob(pattern) if path.is_file()
    )
    if not split_paths:
        raise FileNotFoundError(
            f"no files matching {pattern!r} under {train_root}"
        )

    database_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{database_path.name}.",
        suffix=".tmp",
        dir=database_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    connection: sqlite3.Connection | None = None
    source_files = 0
    empty_files = 0
    total_masked_rows = 0
    total_postings = 0

    try:
        connection = initialize_database(temporary_path)
        for number, split_path in enumerate(split_paths, start=1):
            raw_path_text, masked_rows, posting_indices = scan_train_file(
                split_path
            )
            total_masked_rows += masked_rows

            if raw_path_text is None:
                empty_files += 1
            else:
                raw_path = Path(raw_path_text)
                if not raw_path.is_file():
                    raise FileNotFoundError(
                        f"raw source referenced by {split_path} not found: "
                        f"{raw_path}"
                    )
                split_size, split_mtime_ns = signature(split_path)
                raw_size, raw_mtime_ns = signature(raw_path)
                file_id = number - 1

                try:
                    with connection:
                        connection.execute(
                            """
                            INSERT INTO source_files VALUES (
                                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                            )
                            """,
                            (
                                file_id,
                                str(split_path.relative_to(train_root)),
                                str(split_path),
                                str(raw_path),
                                masked_rows,
                                len(posting_indices),
                                split_size,
                                split_mtime_ns,
                                raw_size,
                                raw_mtime_ns,
                            ),
                        )
                        connection.executemany(
                            "INSERT INTO postings VALUES (?, ?)",
                            (
                                (file_id, row_idx)
                                for row_idx in posting_indices
                            ),
                        )
                except sqlite3.IntegrityError as exc:
                    raise RuntimeError(
                        "the train split references one raw source from "
                        f"multiple files: {raw_path}"
                    ) from exc

                source_files += 1
                total_postings += len(posting_indices)

            if number % progress_every == 0 or number == len(split_paths):
                print(
                    f"indexed train files: {number:,}/{len(split_paths):,}"
                )

        connection.commit()
        connection.close()
        connection = None
        os.replace(temporary_path, database_path)
    finally:
        if connection is not None:
            connection.close()
        if temporary_path.exists():
            temporary_path.unlink()

    return {
        "train_split_files": len(split_paths),
        "train_split_files_with_postings": source_files,
        "empty_train_split_files": empty_files,
        "train_masked_rows_used_as_membership_index": total_masked_rows,
        "train_postings": total_postings,
    }


def read_index_counts(database_path: Path) -> tuple[int, int]:
    connection = sqlite3.connect(database_path)
    try:
        source_files = int(
            connection.execute("SELECT COUNT(*) FROM source_files").fetchone()[0]
        )
        postings = int(
            connection.execute("SELECT COUNT(*) FROM postings").fetchone()[0]
        )
    finally:
        connection.close()
    return source_files, postings


def validate_posting_index(
    train_root: Path,
    pattern: str,
    database_path: Path,
    progress_every: int,
) -> dict[str, int]:
    if not database_path.is_file():
        raise FileNotFoundError(
            f"posting index does not exist: {database_path}"
        )

    split_paths = sorted(
        path for path in train_root.rglob(pattern) if path.is_file()
    )
    if not split_paths:
        raise FileNotFoundError(
            f"no files matching {pattern!r} under {train_root}"
        )
    current_split_paths = {str(path) for path in split_paths}
    database_uri = f"file:{database_path.as_posix()}?mode=ro&immutable=1"
    connection = sqlite3.connect(database_uri, uri=True)
    try:
        user_version = int(
            connection.execute("PRAGMA user_version").fetchone()[0]
        )
        if user_version != 1:
            raise ValueError(
                f"unsupported posting-index version: {user_version}"
            )
        source_rows = list(
            connection.execute(
                """
                SELECT
                    relative_path, split_path, raw_path,
                    masked_rows, train_postings,
                    split_size, split_mtime_ns,
                    raw_size, raw_mtime_ns
                FROM source_files
                ORDER BY relative_path
                """
            )
        )
        posting_rows = int(
            connection.execute("SELECT COUNT(*) FROM postings").fetchone()[0]
        )
    finally:
        connection.close()

    indexed_splits: set[str] = set()
    total_masked_rows = 0
    total_postings = 0
    for number, row in enumerate(source_rows, start=1):
        (
            relative_path,
            split_path_text,
            raw_path_text,
            masked_rows,
            train_postings,
            split_size,
            split_mtime_ns,
            raw_size,
            raw_mtime_ns,
        ) = row
        split_path = Path(split_path_text)
        raw_path = Path(raw_path_text)
        if split_path_text not in current_split_paths:
            raise ValueError(
                f"posting index references a missing split: {split_path}"
            )
        if str(split_path.relative_to(train_root)) != relative_path:
            raise ValueError(
                f"relative split path changed for {split_path}"
            )
        if signature(split_path) != (split_size, split_mtime_ns):
            raise ValueError(f"train split changed: {split_path}")
        if not raw_path.is_file():
            raise FileNotFoundError(
                f"raw source referenced by index is missing: {raw_path}"
            )
        if signature(raw_path) != (raw_size, raw_mtime_ns):
            raise ValueError(f"raw source changed: {raw_path}")

        indexed_splits.add(split_path_text)
        total_masked_rows += int(masked_rows)
        total_postings += int(train_postings)
        if number % progress_every == 0 or number == len(source_rows):
            print(
                f"validated indexed files: {number:,}/{len(source_rows):,}"
            )

    empty_split_paths = [
        path for path in split_paths if str(path) not in indexed_splits
    ]
    for split_path in empty_split_paths:
        raw_path_text, masked_rows, posting_indices = scan_train_file(split_path)
        if raw_path_text is not None or masked_rows != 0 or posting_indices:
            raise ValueError(
                "existing posting index is incomplete; non-empty split is "
                f"not indexed: {split_path}"
            )

    if total_postings != posting_rows:
        raise ValueError(
            f"posting index count mismatch: metadata={total_postings:,}, "
            f"table={posting_rows:,}"
        )
    return {
        "train_split_files": len(split_paths),
        "train_split_files_with_postings": len(source_rows),
        "empty_train_split_files": len(empty_split_paths),
        "train_masked_rows_used_as_membership_index": total_masked_rows,
        "train_postings": total_postings,
    }


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_used_files(database_path: Path, output_path: Path) -> None:
    connection = sqlite3.connect(database_path)
    temporary_path = output_path.with_name(
        f"{output_path.name}.tmp.{os.getpid()}"
    )
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                [
                    "file_id",
                    "train_split_relative_path",
                    "train_split_path",
                    "raw_file_path",
                    "masked_rows",
                    "train_postings",
                    "split_size",
                    "split_mtime_ns",
                    "raw_size",
                    "raw_mtime_ns",
                ]
            )
            writer.writerows(
                connection.execute(
                    """
                    SELECT
                        file_id, relative_path, split_path, raw_path,
                        masked_rows, train_postings,
                        split_size, split_mtime_ns, raw_size, raw_mtime_ns
                    FROM source_files
                    ORDER BY relative_path
                    """
                )
            )
        os.replace(temporary_path, output_path)
    finally:
        connection.close()
        if temporary_path.exists():
            temporary_path.unlink()


def iter_selected_bodies(
    raw_path: Path,
    posting_indices: list[int],
    chunk_size: int,
    body_start_marker: str,
) -> Iterator[str]:
    if not posting_indices:
        return

    normalized_marker = body_start_marker.lower()

    compression = "gzip" if raw_path.suffix == ".gz" else None
    columns = set(
        pd.read_csv(raw_path, compression=compression, nrows=0).columns
    )
    if "body" not in columns:
        raise ValueError(f"{raw_path} does not contain a body column")

    target_position = 0
    row_offset = 0
    chunks = pd.read_csv(
        raw_path,
        compression=compression,
        usecols=["body"],
        chunksize=chunk_size,
        low_memory=False,
    )

    for chunk in chunks:
        chunk_end = row_offset + len(chunk)
        while (
            target_position < len(posting_indices)
            and posting_indices[target_position] < chunk_end
        ):
            row_idx = posting_indices[target_position]
            if row_idx < row_offset:
                raise RuntimeError(
                    f"posting indices are not increasing for {raw_path}"
                )
            body = chunk.iloc[row_idx - row_offset]["body"]
            if pd.isna(body):
                raise ValueError(
                    f"train posting ({raw_path}, {row_idx}) has null body"
                )
            body_text = str(body).replace("\n", " ")
            marker_start = body_text.lower().find(normalized_marker)
            if marker_start == -1:
                raise ValueError(
                    f"train posting ({raw_path}, {row_idx}) does not contain "
                    f"body start marker {body_start_marker!r}"
                )
            description = body_text[
                marker_start + len(body_start_marker):
            ].strip()
            if not description:
                raise ValueError(
                    f"train posting ({raw_path}, {row_idx}) has no text after "
                    f"body start marker {body_start_marker!r}"
                )
            yield description
            target_position += 1

        row_offset = chunk_end
        if target_position == len(posting_indices):
            break

    if target_position != len(posting_indices):
        missing_row_idx = posting_indices[target_position]
        raise IndexError(
            f"raw row {missing_row_idx} referenced by train split is missing "
            f"from {raw_path}"
        )


class TrainPostingBodyDataset(IterableDataset):
    """Yield one original train-posting body per posting per epoch."""

    def __init__(
        self,
        database_path: Path,
        tokenizer: BertTokenizer,
        max_length: int,
        raw_chunk_size: int,
        body_start_marker: str,
    ) -> None:
        self.database_path = database_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.raw_chunk_size = raw_chunk_size
        self.body_start_marker = body_start_marker
        _, self.total_postings = read_index_counts(database_path)

    def __len__(self) -> int:
        return self.total_postings

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        database_uri = (
            f"file:{self.database_path.as_posix()}?mode=ro&immutable=1"
        )
        connection = sqlite3.connect(database_uri, uri=True)

        try:
            source_rows = connection.execute(
                """
                SELECT file_id, raw_path
                FROM source_files
                WHERE (file_id % ?) = ?
                ORDER BY relative_path
                """,
                (worker_count, worker_id),
            )
            for file_id, raw_path_text in source_rows:
                posting_indices = [
                    int(row[0])
                    for row in connection.execute(
                        """
                        SELECT row_idx
                        FROM postings
                        WHERE file_id = ?
                        ORDER BY row_idx
                        """,
                        (file_id,),
                    )
                ]
                for body in iter_selected_bodies(
                    Path(raw_path_text),
                    posting_indices,
                    self.raw_chunk_size,
                    self.body_start_marker,
                ):
                    encoded = self.tokenizer(
                        body,
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_length,
                        return_special_tokens_mask=True,
                        return_tensors="pt",
                    )
                    yield {
                        key: value.squeeze(0)
                        for key, value in encoded.items()
                    }
        finally:
            connection.close()



def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"expected true/false, got {value!r}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pretrain BERT MLM on original body text from train postings only."
        )
    )
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument(
        "--posting-index",
        type=Path,
        help="SQLite index (default: <output-dir>/train_postings.sqlite).",
    )
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--raw-chunk-size",
        type=int,
        default=DEFAULT_RAW_CHUNK_SIZE,
    )
    parser.add_argument(
        "--body-start-marker",
        default=DEFAULT_BODY_START_MARKER,
        help=(
            "Use original body text after the first case-insensitive marker "
            "match (default: job description:)."
        ),
    )
    parser.add_argument(
        "--mlm-probability",
        type=float,
        default=DEFAULT_MLM_PROBABILITY,
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=DEFAULT_EPOCHS,
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--save-steps", type=int, default=DEFAULT_SAVE_STEPS
    )
    parser.add_argument(
        "--mixed-precision",
        choices=("no", "bf16", "fp16"),
        default="bf16",
        help="Training precision (default: bf16, recommended for A100).",
    )
    parser.add_argument(
        "--optim",
        default="adamw_torch_fused",
        help="Transformers optimizer name (default: adamw_torch_fused).",
    )
    parser.add_argument("--tf32", type=parse_bool, default=True)
    parser.add_argument(
        "--reuse-posting-index",
        action="store_true",
        help=(
            "Reuse an existing posting index after validating every split "
            "and raw-file signature."
        ),
    )
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument(
        "--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS
    )
    parser.add_argument(
        "--dataloader-workers", type=int, default=DEFAULT_DATALOADER_WORKERS
    )
    parser.add_argument(
        "--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Build and validate the train corpus without model training.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.01,
        help="Linear-scheduler warmup fraction (default: 0.01).",
    )
    parser.add_argument(
        "--dataloader-prefetch-factor",
        type=int,
        default=DEFAULT_DATALOADER_PREFETCH_FACTOR,
        help="Batches prefetched per DataLoader worker (default: 4).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.train_root.is_dir():
        raise FileNotFoundError(f"train root not found: {args.train_root}")
    if args.max_length < 3:
        raise ValueError("--max-length must be >= 3")
    if args.raw_chunk_size < 1:
        raise ValueError("--raw-chunk-size must be >= 1")
    if not args.body_start_marker:
        raise ValueError("--body-start-marker must not be empty")
    if not args.optim.strip():
        raise ValueError("--optim must not be empty")
    if not 0.0 < args.mlm_probability < 1.0:
        raise ValueError("--mlm-probability must be between 0 and 1")
    if args.num_train_epochs <= 0:
        raise ValueError("--num-train-epochs must be > 0")
    if args.per_device_train_batch_size < 1:
        raise ValueError("--per-device-train-batch-size must be >= 1")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be > 0")
    if args.seed < 0:
        raise ValueError("--seed must be >= 0")
    if args.save_steps < 1 or args.save_total_limit < 1 or args.logging_steps < 1:
        raise ValueError(
            "--save-steps, --save-total-limit, and --logging-steps must be >= 1"
        )
    if args.dataloader_workers < 0:
        raise ValueError("--dataloader-workers must be >= 0")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be >= 1")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if args.dataloader_prefetch_factor < 1:
        raise ValueError(
            "--dataloader-prefetch-factor must be >= 1"
        )


def make_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.mixed_precision == "bf16",
        fp16=args.mixed_precision == "fp16",
        warmup_ratio=args.warmup_ratio,
        dataloader_prefetch_factor=args.dataloader_prefetch_factor,
        tf32=args.tf32,
        optim=args.optim,
        dataloader_pin_memory=True,
        logging_steps=args.logging_steps,
        logging_dir=str(args.output_dir / "logs"),
        dataloader_num_workers=args.dataloader_workers,
        seed=args.seed,
        data_seed=args.seed,
        ddp_timeout=86_400,
        ddp_find_unused_parameters=False,
        accelerator_config={
            "dispatch_batches": True,
            "split_batches": False,
        },
        report_to=[],
    )


def distributed_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        if dist.get_backend() == "nccl":
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def run() -> int:
    args = parse_args()
    args.train_root = args.train_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    validate_args(args)
    training_args = (
        None if args.index_only else make_training_arguments(args)
    )
    is_main = (
        True if training_args is None else training_args.process_index == 0
    )
    world_size = 1 if training_args is None else training_args.world_size
    if is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    distributed_barrier()

    database_path = (
        args.posting_index.expanduser().resolve()
        if args.posting_index is not None
        else args.output_dir / "train_postings.sqlite"
    )
    if args.train_root in database_path.parents:
        raise ValueError("--posting-index must not be inside --train-root")

    if is_main:
        print(f"train split root:  {args.train_root}")
        print(f"output directory:  {args.output_dir}")
        print(f"base model:        {args.model_name}")
        print(f"epochs:            {args.num_train_epochs}")
        print(f"max length:        {args.max_length}")
        print(f"body start marker: {args.body_start_marker!r}")
        print(f"MLM probability:   {args.mlm_probability}")
        print(f"precision:         {args.mixed_precision}")
        print(f"optimizer:         {args.optim}")
        print(f"TF32:              {args.tf32}")
        print(f"warmup ratio:      {args.warmup_ratio}")
        print(
            f"reuse index:       {args.reuse_posting_index}"
        )
        print(f"world size:        {world_size}")
        print(
            f"batch/GPU, global: {args.per_device_train_batch_size} / "
            f"{args.per_device_train_batch_size * world_size}"
        )

    index_message: list[object] = [None]
    if is_main:
        try:
            if args.reuse_posting_index and database_path.is_file():
                print(f"validating reusable posting index: {database_path}")
                stats = validate_posting_index(
                    args.train_root,
                    args.pattern,
                    database_path,
                    args.progress_every,
                )
                stats["posting_index_reused"] = True
                print("posting index validation complete; rebuild skipped")
            else:
                if args.reuse_posting_index:
                    print("reusable posting index not found; building it")
                stats = build_posting_index(
                    args.train_root,
                    args.pattern,
                    database_path,
                    args.progress_every,
                )
                stats["posting_index_reused"] = False
            index_message[0] = stats
        except BaseException:
            index_message[0] = {"error": traceback.format_exc()}
    if world_size > 1:
        dist.broadcast_object_list(index_message, src=0)
    distributed_barrier()
    index_stats = index_message[0]
    if not isinstance(index_stats, dict):
        raise RuntimeError("rank 0 did not broadcast posting-index statistics")
    if "error" in index_stats:
        raise RuntimeError(
            "rank 0 failed while preparing the posting index:\n"
            f"{index_stats['error']}"
        )
    source_files, train_postings = read_index_counts(database_path)
    if train_postings != index_stats["train_postings"]:
        raise RuntimeError("train posting count differs from SQLite index")
    if train_postings == 0:
        raise RuntimeError("the train split contains no postings")

    used_files_path = args.output_dir / "used_files.csv"
    summary_path = args.output_dir / "pretrain_corpus_summary.json"
    if is_main:
        write_used_files(database_path, used_files_path)

    configuration = {
        "train_root": str(args.train_root),
        "output_dir": str(args.output_dir),
        "model_name": args.model_name,
        "pattern": args.pattern,
        "posting_identity": ["file_path", "row_idx"],
        "text_column": "body",
        "body_source": "original_raw_posting",
        "body_start_marker": args.body_start_marker,
        "body_slice_policy": "after_first_case_insensitive_marker",
        "test_postings_used": 0,
        "one_body_per_unique_train_posting_per_epoch": True,
        "body_tokenization": "truncate_and_pad_to_max_length",
        "max_length": args.max_length,
        "raw_chunk_size": args.raw_chunk_size,
        "mlm_probability": args.mlm_probability,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "world_size": world_size,
        "global_train_batch_size": args.per_device_train_batch_size * world_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "logging_steps": args.logging_steps,
        "dataloader_workers": args.dataloader_workers,
        "dataloader_prefetch_factor": args.dataloader_prefetch_factor,
        "mixed_precision": args.mixed_precision,
        "optimizer": args.optim,
        "tf32": args.tf32,
        "warmup_ratio": args.warmup_ratio,
        "reuse_posting_index": args.reuse_posting_index,
        "posting_index_reused": bool(index_stats["posting_index_reused"]),
        "progress_every": args.progress_every,
    }
    configuration_hash = hashlib.sha256(
        json.dumps(
            configuration,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    summary: dict[str, object] = {
        **configuration,
        "configuration_sha256": configuration_hash,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": sys.version.split()[0],
        "sqlite_version": sqlite3.sqlite_version,
        "posting_index": str(database_path),
        "used_files_csv": str(used_files_path),
        **index_stats,
        "train_split_files_with_postings": source_files,
        "train_postings": train_postings,
        "samples_per_epoch": train_postings,
        "training_completed": False,
    }
    if is_main:
        atomic_write_json(summary_path, summary)
        print(f"train source files: {source_files:,}")
        print(f"train postings:     {train_postings:,}")
        print(f"posting index:      {database_path}")
        print(f"used files:         {used_files_path}")
        print(f"corpus summary:     {summary_path}")

    if args.index_only:
        if is_main:
            print("index-only mode complete; model training was not started")
        return 0

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    dataset = TrainPostingBodyDataset(
        database_path,
        tokenizer,
        args.max_length,
        args.raw_chunk_size,
        args.body_start_marker,
    )
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )
    model = BertForMaskedLM.from_pretrained(args.model_name)

    assert training_args is not None
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    if is_main:
        print("starting train-posting MLM pretraining")
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(args.output_dir)
    distributed_barrier()

    if is_main:
        tokenizer.save_pretrained(args.output_dir)
        summary["training_completed"] = True
        atomic_write_json(summary_path, summary)
        print(f"model saved:        {args.output_dir}")
    return 0



def main() -> int:
    try:
        return run()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
