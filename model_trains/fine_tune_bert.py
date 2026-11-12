#!/usr/bin/env python3
"""Fine-tune BERT on masked_sentence -> true_skill rows.

Only the posting-safe train split is consumed. Nested gzip CSV files are
streamed by disjoint DataLoader worker file shards, so the full dataset is
never held in memory and each row appears exactly once per epoch.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import tempfile
import traceback
from dataclasses import dataclass
from itertools import chain, islice
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TextIO

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm
from transformers import BertTokenizerFast

from model import BERTForSkillPrediction


BASE_DATA_DIR = Path(
    os.environ.get(
        "BASE_DATA_DIR", "/home/jovyan/LEM_data3/hyunjincho/LEM"
    )
)
DEFAULT_DATA_ROOT = Path(
    os.environ.get("DATA_ROOT", str(BASE_DATA_DIR / "all_train_test"))
)
DEFAULT_VOCAB_PATH = Path(
    os.environ.get(
        "SKILL2IDX_PATH",
        str(BASE_DATA_DIR / "all_filterd" / "skill2idx.json"),
    )
)
DEFAULT_MODEL_NAME = os.environ.get(
    "BERT_MODEL_NAME", str(BASE_DATA_DIR / "bert_pretrained")
)
DEFAULT_CHECKPOINT_DIR = Path(
    os.environ.get(
        "BERT_FINETUNE_DIR",
        os.environ.get(
            "CKPT_DIR",
            str(BASE_DATA_DIR / "models" / "bert_fine_tuned_model"),
        ),
    )
)
DEFAULT_PATTERN = "preprocessed_*.csv.gz"
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 4e-5
DEFAULT_WEIGHT_DECAY = 0.01
MAX_WORKERS = 32
DEFAULT_NUM_WORKERS = min(16, os.cpu_count() or 1)
DEFAULT_SHUFFLE_BUFFER_SIZE = 10_000
DEFAULT_PREFETCH_FACTOR = 4
DEFAULT_PIN_MEMORY = True
DEFAULT_MIXED_PRECISION = "bf16"
DEFAULT_FUSED_OPTIMIZER = True
DEFAULT_TF32 = True
DEFAULT_ATTENTION_IMPLEMENTATION = "sdpa"
DEFAULT_SEED = 42
DEFAULT_LOSS_LOG_STEPS = 500
DEFAULT_LOSS_PLOT_EVERY_LOGS = 10
DEFAULT_LOSS_SMOOTHING_WINDOW = 20
FILE_ROW_MANIFEST_VERSION = 1
REQUIRED_COLUMNS = ("masked_sentence", "true_skill")
LOSS_METRIC_COLUMNS = (
    "epoch",
    "global_step",
    "epoch_step",
    "rows_seen_in_epoch",
    "interval_avg_loss",
    "epoch_avg_loss",
    "epoch_accuracy",
)


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
            "Fine-tune BERT on masked_sentence -> true_skill using only "
            "the posting-safe train split."
        )
    )
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=None,
        help="Default: <data-root>/train",
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=DEFAULT_VOCAB_PATH,
        help="Filtered skill2idx.json (default: %(default)s)",
    )
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR
    )
    parser.add_argument(
        "--best-model-path",
        type=Path,
        default=None,
        help="Default: <checkpoint-dir>/best_model.pt",
    )
    parser.add_argument(
        "--loss-metrics-path",
        type=Path,
        default=None,
        help="Default: <checkpoint-dir>/training_metrics.csv",
    )
    parser.add_argument(
        "--loss-plot-path",
        type=Path,
        default=None,
        help="Default: <checkpoint-dir>/training_loss.png",
    )
    parser.add_argument(
        "--loss-log-steps",
        type=int,
        default=DEFAULT_LOSS_LOG_STEPS,
        help="Record interval-average training loss every N steps.",
    )
    parser.add_argument(
        "--loss-plot-every-logs",
        type=int,
        default=DEFAULT_LOSS_PLOT_EVERY_LOGS,
        help="Refresh the PNG after this many new CSV records.",
    )
    parser.add_argument(
        "--loss-smoothing-window",
        type=int,
        default=DEFAULT_LOSS_SMOOTHING_WINDOW,
        help="Number of recorded loss points in the moving average.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Epoch checkpoint to resume from; omitted starts fresh.",
    )
    parser.add_argument(
        "--max-length", type=int, default=DEFAULT_MAX_LENGTH
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Batch size per process/GPU. Under torchrun, global batch size "
            "is this value multiplied by WORLD_SIZE."
        ),
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_LEARNING_RATE
    )
    parser.add_argument(
        "--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=(
            f"Parallel CSV/tokenization workers, 0-{MAX_WORKERS} "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=DEFAULT_SHUFFLE_BUFFER_SIZE,
        help="Bounded per-worker row shuffle buffer; 1 disables it.",
    )
    parser.add_argument(
        "--prefetch-factor", type=int, default=DEFAULT_PREFETCH_FACTOR
    )
    parser.add_argument(
        "--expected-train-rows",
        type=int,
        default=None,
        help=(
            "Expected rows per epoch. By default read from "
            "split_summary.json; otherwise counted once."
        ),
    )
    parser.add_argument(
        "--pin-memory", type=parse_bool, default=DEFAULT_PIN_MEMORY
    )
    parser.add_argument(
        "--mixed-precision",
        choices=("no", "bf16", "fp16"),
        default=DEFAULT_MIXED_PRECISION,
        help="Training precision (default: bf16 for A100).",
    )
    parser.add_argument(
        "--fused-optimizer",
        type=parse_bool,
        default=DEFAULT_FUSED_OPTIMIZER,
    )
    parser.add_argument("--tf32", type=parse_bool, default=DEFAULT_TF32)
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa"),
        default=DEFAULT_ATTENTION_IMPLEMENTATION,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device such as auto, cuda, cuda:0, or cpu.",
    )
    args = parser.parse_args()

    if args.train_dir is None:
        args.train_dir = args.data_root / "train"
    if args.max_length < 3:
        parser.error("--max-length must be >= 3 for [CLS], text, and [SEP]")
    if args.batch_size < 1 or args.epochs < 1:
        parser.error("--batch-size and --epochs must be >= 1")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be > 0")
    if args.weight_decay < 0:
        parser.error("--weight-decay must be >= 0")
    if args.loss_log_steps < 1:
        parser.error("--loss-log-steps must be >= 1")
    if args.loss_plot_every_logs < 1:
        parser.error("--loss-plot-every-logs must be >= 1")
    if args.loss_smoothing_window < 1:
        parser.error("--loss-smoothing-window must be >= 1")
    if not 0 <= args.num_workers <= MAX_WORKERS:
        parser.error(
            f"--num-workers must be between 0 and {MAX_WORKERS}"
        )
    if args.shuffle_buffer_size < 1:
        parser.error("--shuffle-buffer-size must be >= 1")
    if args.prefetch_factor < 1:
        parser.error("--prefetch-factor must be >= 1")
    if (
        args.expected_train_rows is not None
        and args.expected_train_rows < 1
    ):
        parser.error("--expected-train-rows must be >= 1")
    if args.seed < 0:
        parser.error("--seed must be >= 0")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device requested but unavailable: {requested}"
        )
    return device


def discover_files(root: Path, pattern: str) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"train directory not found: {root}")
    files = sorted(
        path for path in root.rglob(pattern) if path.is_file()
    )
    if not files:
        raise FileNotFoundError(
            f"no files matching {pattern!r} recursively under {root}"
        )
    return files


def open_csv(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(
            path, "rt", encoding="utf-8", newline="\n"
        )
    return path.open("r", encoding="utf-8", newline="\n")


def normalized_csv_lines(handle: TextIO) -> Iterator[str]:
    """Neutralize legacy bare CR characters without breaking quoted LF."""
    for line in handle:
        yield line.replace("\r\n", "\n").replace("\r", " ")


def header_positions(
    header: Sequence[str], path: Path
) -> dict[str, int]:
    missing = [
        column for column in REQUIRED_COLUMNS if column not in header
    ]
    if missing:
        raise ValueError(
            f"{path} is missing columns: {', '.join(missing)}"
        )
    return {
        column: header.index(column) for column in REQUIRED_COLUMNS
    }


def count_csv_rows(path: Path) -> int:
    with open_csv(path) as handle:
        reader = csv.reader(normalized_csv_lines(handle))
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(
                f"{path} is empty and has no CSV header"
            ) from exc
        header_positions(header, path)
        return sum(1 for _ in reader)


def infer_rows_from_summary(train_root: Path) -> int | None:
    summary_path = train_root.parent / "split_summary.json"
    if not summary_path.is_file():
        return None
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    summary_train_root = summary.get("train_root")
    if summary_train_root is not None:
        if (
            Path(summary_train_root).expanduser().resolve()
            != train_root.resolve()
        ):
            return None
    train_rows = summary.get("train_rows")
    if not isinstance(train_rows, int) or train_rows < 1:
        return None
    return train_rows


def determine_expected_rows(
    train_root: Path,
    files: Sequence[Path],
    explicit_rows: int | None,
    num_workers: int,
) -> tuple[int, str]:
    if explicit_rows is not None:
        return explicit_rows, "--expected-train-rows"

    summary_rows = infer_rows_from_summary(train_root)
    if summary_rows is not None:
        summary_path = train_root.parent / "split_summary.json"
        return summary_rows, str(summary_path)

    print(
        "split_summary.json unavailable; counting CSV rows once ...",
        flush=True,
    )
    worker_count = min(num_workers, len(files))
    if worker_count <= 1:
        total = sum(count_csv_rows(path) for path in files)
    else:
        with Pool(processes=worker_count) as pool:
            total = sum(
                pool.imap_unordered(
                    count_csv_rows, files, chunksize=16
                )
            )
    if total < 1:
        raise ValueError(f"no data rows found below {train_root}")
    return total, "CSV scan"


def load_skill_vocabulary(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(
            f"skill vocabulary not found: {path}"
        )
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            f"{path} must contain a non-empty JSON object"
        )

    skill2idx: dict[str, int] = {}
    for skill, index in raw.items():
        normalized = str(skill).strip().lower()
        if not normalized:
            raise ValueError(f"{path} contains an empty skill name")
        if normalized in skill2idx:
            raise ValueError(
                f"{path} contains duplicate normalized skill "
                f"{normalized!r}"
            )
        if not isinstance(index, int):
            raise ValueError(
                f"non-integer index for {skill!r}: {index!r}"
            )
        skill2idx[normalized] = index

    if set(skill2idx.values()) != set(range(len(skill2idx))):
        raise ValueError(
            f"{path} indices must be unique and contiguous from 0 "
            f"through {len(skill2idx) - 1}"
        )
    return skill2idx


Sample = tuple[str, int, str]
Batch = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


def iter_file_samples(
    path: Path,
    skill2idx: dict[str, int],
    start_row: int = 0,
    stop_row: int | None = None,
) -> Iterator[Sample]:
    with open_csv(path) as handle:
        reader = csv.reader(normalized_csv_lines(handle))
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(
                f"{path} is empty and has no CSV header"
            ) from exc
        positions = header_positions(header, path)
        max_position = max(positions.values())

        for csv_row_number, row in enumerate(reader, start=2):
            data_row = csv_row_number - 2
            if data_row < start_row:
                continue
            if stop_row is not None and data_row >= stop_row:
                break

            origin = f"{path}:{csv_row_number}"
            if len(row) <= max_position:
                raise ValueError(
                    f"{origin} contains a short CSV row"
                )
            sentence = row[positions["masked_sentence"]]
            if not sentence:
                raise ValueError(
                    f"{origin} has an empty masked_sentence"
                )
            if "[MASK]" not in sentence:
                raise ValueError(
                    f"{origin} masked_sentence has no [MASK]"
                )
            skill = row[positions["true_skill"]].strip().lower()
            if not skill:
                raise ValueError(
                    f"{origin} has an empty true_skill"
                )
            try:
                label = skill2idx[skill]
            except KeyError as exc:
                raise ValueError(
                    f"{origin} true_skill {skill!r} is absent from "
                    "the filtered vocabulary"
                ) from exc
            yield sentence, label, origin


def buffered_shuffle(
    samples: Iterable[Sample],
    buffer_size: int,
    rng: random.Random,
) -> Iterator[Sample]:
    if buffer_size <= 1:
        yield from samples
        return

    iterator = iter(samples)
    buffer = list(islice(iterator, buffer_size))
    for sample in iterator:
        position = rng.randrange(len(buffer))
        yield buffer[position]
        buffer[position] = sample
    rng.shuffle(buffer)
    yield from buffer


class MaskedSkillIterableDataset(IterableDataset):
    """Stream exact, balanced rank shards with bounded row shuffling."""

    def __init__(
        self,
        files: Sequence[Path],
        skill2idx: dict[str, int],
        expected_rows: int,
        shuffle_buffer_size: int,
        seed: int,
        epoch: int,
        rank: int = 0,
        world_size: int = 1,
        file_row_counts: dict[Path, int] | None = None,
    ) -> None:
        super().__init__()
        self.files = tuple(files)
        self.skill2idx = skill2idx
        self.expected_rows = expected_rows
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.epoch = epoch
        self.rank = rank
        self.world_size = world_size
        self.file_row_counts = file_row_counts
        self.rows_per_rank = math.ceil(expected_rows / world_size)
        if world_size > 1:
            if file_row_counts is None:
                raise ValueError("DDP requires exact per-file row counts")
            counted_rows = sum(file_row_counts[path] for path in self.files)
            if counted_rows != expected_rows:
                raise ValueError(
                    f"file row counts total {counted_rows:,}, expected "
                    f"{expected_rows:,}"
                )

    def __len__(self) -> int:
        return self.rows_per_rank

    def rank_segments(
        self, shuffled_files: Sequence[Path]
    ) -> tuple[list[tuple[Path, int, int]], int]:
        if self.file_row_counts is None:
            raise RuntimeError("rank_segments called without file row counts")

        rank_start = self.rank * self.rows_per_rank
        rank_stop = min(
            rank_start + self.rows_per_rank,
            self.expected_rows,
        )
        segments: list[tuple[Path, int, int]] = []
        cursor = 0
        for path in shuffled_files:
            file_rows = self.file_row_counts[path]
            file_stop = cursor + file_rows
            overlap_start = max(rank_start, cursor)
            overlap_stop = min(rank_stop, file_stop)
            if overlap_start < overlap_stop:
                segments.append(
                    (
                        path,
                        overlap_start - cursor,
                        overlap_stop - cursor,
                    )
                )
            cursor = file_stop
            if cursor >= rank_stop:
                break

        unique_rows = max(0, rank_stop - rank_start)
        padding_rows = self.rows_per_rank - unique_rows
        return segments, padding_rows

    def __iter__(self) -> Iterator[Sample]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers

        file_rng = random.Random(
            self.seed + self.epoch * 1_000_003
        )
        shuffled_files = list(self.files)
        file_rng.shuffle(shuffled_files)
        if self.world_size == 1:
            worker_files = shuffled_files[worker_id::worker_count]
            samples: Iterable[Sample] = chain.from_iterable(
                iter_file_samples(path, self.skill2idx)
                for path in worker_files
            )
            padding_rows = 0
        else:
            segments, padding_rows = self.rank_segments(shuffled_files)
            worker_segments = segments[worker_id::worker_count]
            samples = chain.from_iterable(
                iter_file_samples(
                    path,
                    self.skill2idx,
                    start_row,
                    stop_row,
                )
                for path, start_row, stop_row in worker_segments
            )

        row_rng = random.Random(
            self.seed
            + self.epoch * 1_000_003
            + self.rank * 1_000_033
            + worker_id * 97_409
        )
        shuffled_samples = buffered_shuffle(
            samples,
            self.shuffle_buffer_size,
            row_rng,
        )

        first_sample: Sample | None = None
        for sample in shuffled_samples:
            if first_sample is None:
                first_sample = sample
            yield sample

        if worker_id == 0 and padding_rows:
            if first_sample is None:
                raise RuntimeError(
                    f"rank {self.rank} cannot pad an empty data shard"
                )
            for _ in range(padding_rows):
                yield first_sample


class TokenizeBatch:
    """Pickle-safe vectorized DataLoader collator."""

    def __init__(
        self, tokenizer: BertTokenizerFast, max_length: int
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: Sequence[Sample]) -> Batch:
        sentences, labels, origins = zip(*batch)
        encoded = self.tokenizer(
            list(sentences),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        mask_locations = input_ids.eq(
            self.tokenizer.mask_token_id
        )
        missing_mask = ~mask_locations.any(dim=1)
        if missing_mask.any():
            index = int(
                missing_mask.nonzero(as_tuple=False)[0].item()
            )
            raise ValueError(
                f"{origins[index]} lost [MASK] after tokenization; "
                "--max-length differs from preprocessing"
            )
        mask_indices = mask_locations.to(torch.int64).argmax(dim=1)
        return (
            input_ids,
            attention_mask,
            mask_indices,
            torch.tensor(labels, dtype=torch.long),
        )


def concatenate_batches(batches: Sequence[Batch]) -> Batch:
    columns = zip(*batches)
    return tuple(
        torch.cat(parts, dim=0) for parts in columns
    )  # type: ignore[return-value]


def iter_globally_batched(
    loader: DataLoader, batch_size: int
) -> Iterator[Batch]:
    """Merge worker tail batches for an exact global step count."""
    pending: Batch | None = None

    for batch in loader:
        rows = int(batch[0].shape[0])
        if pending is None and rows == batch_size:
            yield batch
            continue
        if pending is not None:
            batch = concatenate_batches((pending, batch))
            pending = None

        rows = int(batch[0].shape[0])
        if rows >= batch_size:
            yield tuple(
                tensor[:batch_size] for tensor in batch
            )  # type: ignore[misc]
        if rows > batch_size:
            pending = tuple(
                tensor[batch_size:] for tensor in batch
            )  # type: ignore[assignment]
        elif rows < batch_size:
            pending = batch

    if pending is not None:
        yield pending


def make_loader(
    dataset: MaskedSkillIterableDataset,
    collator: TokenizeBatch,
    args: argparse.Namespace,
    pin_memory: bool,
) -> DataLoader:
    kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collator,
        "drop_last": False,
    }
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(**kwargs)


def atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def read_loss_metrics(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(LOSS_METRIC_COLUMNS):
            raise ValueError(
                f"unexpected loss-metrics columns in {path}: "
                f"{reader.fieldnames!r}"
            )
        return list(reader)


def atomic_write_loss_metrics(
    path: Path, rows: Sequence[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=LOSS_METRIC_COLUMNS
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    finally:
        temporary_path.unlink(missing_ok=True)


def save_loss_plot(
    metrics_path: Path,
    plot_path: Path,
    smoothing_window: int,
) -> None:
    # Import lazily so DataLoader workers do not import pyplot.
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    rows = read_loss_metrics(metrics_path)
    figure, axis = plt.subplots(figsize=(11, 6), dpi=150)

    if rows:
        steps = np.asarray(
            [int(row["global_step"]) for row in rows],
            dtype=np.int64,
        )
        losses = np.asarray(
            [float(row["interval_avg_loss"]) for row in rows],
            dtype=np.float64,
        )
        epochs = np.asarray(
            [int(row["epoch"]) for row in rows],
            dtype=np.int64,
        )
        window = min(smoothing_window, len(losses))
        weights = np.full(window, 1.0 / window)
        smoothed = np.convolve(losses, weights, mode="valid")
        smooth_steps = steps[window - 1 :]

        axis.plot(
            steps,
            losses,
            color="#8ab4f8",
            linewidth=0.8,
            alpha=0.35,
            label="Interval average",
        )
        axis.plot(
            smooth_steps,
            smoothed,
            color="#1659a5",
            linewidth=2.0,
            label=f"Moving average ({window} points)",
        )

        epoch_changes = np.flatnonzero(epochs[1:] != epochs[:-1]) + 1
        for change_index in epoch_changes:
            axis.axvline(
                steps[change_index],
                color="#6b7280",
                linestyle="--",
                linewidth=0.9,
                alpha=0.7,
            )
            axis.text(
                steps[change_index],
                axis.get_ylim()[1],
                f" Epoch {epochs[change_index]}",
                color="#4b5563",
                fontsize=9,
                va="top",
            )
        axis.legend(frameon=False)
    else:
        axis.text(
            0.5,
            0.5,
            "Waiting for the first loss record",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="#6b7280",
        )

    axis.set_title("BERT Fine-tuning Loss")
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Cross-entropy loss")
    axis.grid(True, alpha=0.2)
    figure.tight_layout()

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{plot_path.name}.",
        suffix=".png",
        dir=plot_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        figure.savefig(
            temporary_path,
            format="png",
            bbox_inches="tight",
        )
        os.replace(temporary_path, plot_path)
    finally:
        plt.close(figure)
        temporary_path.unlink(missing_ok=True)


class LossHistory:
    """Append resume-safe loss records and periodically refresh a PNG."""

    def __init__(
        self,
        metrics_path: Path,
        plot_path: Path,
        smoothing_window: int,
        plot_every_logs: int,
        resume: bool,
        completed_steps: int,
    ) -> None:
        self.metrics_path = metrics_path
        self.plot_path = plot_path
        self.smoothing_window = smoothing_window
        self.plot_every_logs = plot_every_logs
        self.logs_since_plot = 0
        self.closed = False

        retained: list[dict[str, object]] = []
        if resume:
            retained = [
                row
                for row in read_loss_metrics(metrics_path)
                if int(row["global_step"]) <= completed_steps
            ]
        atomic_write_loss_metrics(metrics_path, retained)
        save_loss_plot(
            metrics_path, plot_path, smoothing_window
        )

        self.handle = metrics_path.open(
            "a", encoding="utf-8", newline=""
        )
        self.writer = csv.DictWriter(
            self.handle, fieldnames=LOSS_METRIC_COLUMNS
        )

    def record(
        self,
        epoch: int,
        global_step: int,
        epoch_step: int,
        rows_seen: int,
        interval_avg_loss: float,
        epoch_avg_loss: float,
        epoch_accuracy: float,
    ) -> None:
        self.writer.writerow(
            {
                "epoch": epoch,
                "global_step": global_step,
                "epoch_step": epoch_step,
                "rows_seen_in_epoch": rows_seen,
                "interval_avg_loss": f"{interval_avg_loss:.8f}",
                "epoch_avg_loss": f"{epoch_avg_loss:.8f}",
                "epoch_accuracy": f"{epoch_accuracy:.8f}",
            }
        )
        self.handle.flush()
        self.logs_since_plot += 1
        if self.logs_since_plot >= self.plot_every_logs:
            save_loss_plot(
                self.metrics_path,
                self.plot_path,
                self.smoothing_window,
            )
            self.logs_since_plot = 0

    def close(self) -> None:
        if self.closed:
            return
        self.handle.flush()
        self.handle.close()
        if self.logs_since_plot:
            save_loss_plot(
                self.metrics_path,
                self.plot_path,
                self.smoothing_window,
            )
        self.closed = True


def train(
    args: argparse.Namespace, context: DistributedContext
) -> None:
    train_root = args.train_dir.expanduser().resolve()
    vocab_path = args.vocab_path.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    if args.best_model_path is None:
        best_model_path = checkpoint_dir / "best_model.pt"
    else:
        best_model_path = (
            args.best_model_path.expanduser().resolve()
        )

    if args.loss_metrics_path is None:
        loss_metrics_path = checkpoint_dir / "training_metrics.csv"
    else:
        loss_metrics_path = (
            args.loss_metrics_path.expanduser().resolve()
        )
    if args.loss_plot_path is None:
        loss_plot_path = checkpoint_dir / "training_loss.png"
    else:
        loss_plot_path = args.loss_plot_path.expanduser().resolve()

    files = discover_files(train_root, args.pattern)
    if context.is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    distributed_barrier(context)

    file_row_counts: dict[Path, int] | None = None
    if context.enabled:
        manifest_path = checkpoint_dir / "train_file_row_counts.json"
        file_row_counts, count_source = distributed_file_row_counts(
            files,
            manifest_path,
            args.num_workers,
            context,
        )
        expected_rows = sum(file_row_counts.values())
        if (
            args.expected_train_rows is not None
            and args.expected_train_rows != expected_rows
        ):
            raise ValueError(
                f"--expected-train-rows={args.expected_train_rows:,} but "
                f"the exact file total is {expected_rows:,}"
            )
        summary_rows = infer_rows_from_summary(train_root)
        if summary_rows is not None and summary_rows != expected_rows:
            raise ValueError(
                f"split_summary.json reports {summary_rows:,} train rows "
                f"but the exact file total is {expected_rows:,}"
            )
    else:
        expected_rows, count_source = determine_expected_rows(
            train_root,
            files,
            args.expected_train_rows,
            args.num_workers,
        )

    rows_per_rank = math.ceil(expected_rows / context.world_size)
    effective_rows = rows_per_rank * context.world_size
    expected_steps = math.ceil(rows_per_rank / args.batch_size)
    global_batch_size = args.batch_size * context.world_size
    skill2idx = load_skill_vocabulary(vocab_path)
    device = context.device
    pin_memory = bool(
        args.pin_memory and device.type == "cuda"
    )
    amp_enabled = bool(
        args.mixed_precision != "no" and device.type == "cuda"
    )
    amp_dtype = (
        torch.bfloat16
        if args.mixed_precision == "bf16"
        else torch.float16
    )
    scaler_enabled = bool(
        amp_enabled and args.mixed_precision == "fp16"
    )
    if (
        args.mixed_precision == "bf16"
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("BF16 requested but this CUDA device lacks support")

    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tokenizer = BertTokenizerFast.from_pretrained(
        args.model_name
    )
    if tokenizer.mask_token_id is None:
        raise ValueError(
            f"tokenizer for {args.model_name!r} has no mask token"
        )

    if context.is_main:
        print(f"train root:          {train_root}")
        print(f"train files:         {len(files):,}")
        print(
            f"unique rows/epoch:   {expected_rows:,} "
            f"({count_source})"
        )
        print(f"rows per rank:       {rows_per_rank:,}")
        if effective_rows != expected_rows:
            print(
                f"DDP padding rows:    "
                f"{effective_rows - expected_rows:,} per epoch"
            )
        print(f"steps per epoch:     {expected_steps:,}")
        print(
            f"batch per GPU/global:{args.batch_size:,} / "
            f"{global_batch_size:,}"
        )
        print(
            f"filtered vocabulary: {len(skill2idx):,} "
            f"({vocab_path})"
        )
        print(f"model:               {args.model_name}")
        print(f"learning rate:       {args.learning_rate}")
        print(f"weight decay:        {args.weight_decay}")
        print(
            f"world size/workers:  {context.world_size} / "
            f"{args.num_workers} per rank"
        )
        print(f"rank-0 device:       {device}")
        print(f"mixed precision:     {args.mixed_precision}")
        print(f"fused optimizer:     {args.fused_optimizer}")
        print(f"TF32:                {args.tf32}")
        print(
            f"attention impl:      {args.attention_implementation}"
        )

    model = BERTForSkillPrediction(
        args.model_name,
        num_skills=len(skill2idx),
        attn_implementation=args.attention_implementation,
    ).to(device)
    # The classifier uses last_hidden_state only; BERT's pooler is never in
    # the forward graph. Freezing it avoids DDP unused-parameter failures.
    pooler = getattr(model.bert, "pooler", None)
    if pooler is not None:
        pooler.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=bool(args.fused_optimizer and device.type == "cuda"),
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=scaler_enabled,
    )

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32

    start_epoch = 0
    best_loss = float("inf")
    global_step_offset = 0
    if args.resume_from is not None:
        resume_path = args.resume_from.expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(
                f"resume checkpoint not found: {resume_path}"
            )
        checkpoint = torch.load(
            resume_path, map_location="cpu"
        )
        model.load_state_dict(
            checkpoint["model_state_dict"], strict=True
        )
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(
                checkpoint["scaler_state_dict"]
            )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(
            checkpoint.get("best_loss", best_loss)
        )
        global_step_offset = int(
            checkpoint.get(
                "global_optimizer_steps",
                start_epoch * int(checkpoint["optimizer_steps"]),
            )
        )
        if context.is_main:
            print(
                f"resumed from {resume_path}; next epoch="
                f"{start_epoch + 1}, best_loss={best_loss:.6f}"
            )

    if context.enabled:
        ddp_device_ids = (
            [context.local_rank] if device.type == "cuda" else None
        )
        model = DistributedDataParallel(
            model,
            device_ids=ddp_device_ids,
            output_device=(
                context.local_rank if device.type == "cuda" else None
            ),
            broadcast_buffers=False,
        )

    loss_history: LossHistory | None = None
    if context.is_main:
        loss_history = LossHistory(
            metrics_path=loss_metrics_path,
            plot_path=loss_plot_path,
            smoothing_window=args.loss_smoothing_window,
            plot_every_logs=args.loss_plot_every_logs,
            resume=args.resume_from is not None,
            completed_steps=global_step_offset,
        )
        print(f"loss metrics:        {loss_metrics_path}")
        print(f"loss plot:           {loss_plot_path}")

    collator = TokenizeBatch(tokenizer, args.max_length)
    for epoch in range(start_epoch, args.epochs):
        dataset = MaskedSkillIterableDataset(
            files=files,
            skill2idx=skill2idx,
            expected_rows=expected_rows,
            shuffle_buffer_size=args.shuffle_buffer_size,
            seed=args.seed,
            epoch=epoch,
            rank=context.rank,
            world_size=context.world_size,
            file_row_counts=file_row_counts,
        )
        loader = make_loader(
            dataset, collator, args, pin_memory
        )
        model.train()
        total_loss = 0.0
        correct = 0
        samples_seen = 0
        steps_seen = 0
        interval_loss = 0.0
        interval_rows = 0

        progress = tqdm(
            total=effective_rows,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            unit="row",
            dynamic_ncols=True,
            disable=not context.is_main,
        )
        try:
            batches = iter_globally_batched(
                loader, args.batch_size
            )
            for batch in batches:
                (
                    input_ids,
                    attention_mask,
                    mask_indices,
                    labels,
                ) = batch
                input_ids = input_ids.to(
                    device, non_blocking=pin_memory
                )
                attention_mask = attention_mask.to(
                    device, non_blocking=pin_memory
                )
                mask_indices = mask_indices.to(
                    device, non_blocking=pin_memory
                )
                labels = labels.to(
                    device, non_blocking=pin_memory
                )

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    logits = model(
                        input_ids,
                        attention_mask,
                        mask_indices,
                    )
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                batch_rows = int(labels.shape[0])
                batch_loss = float(loss.detach().item())
                weighted_loss = batch_loss * batch_rows
                samples_seen += batch_rows
                steps_seen += 1
                total_loss += weighted_loss
                interval_loss += weighted_loss
                interval_rows += batch_rows
                correct += int(
                    (
                        logits.argmax(dim=-1) == labels
                    ).sum().item()
                )
                if (
                    steps_seen % args.loss_log_steps == 0
                    or steps_seen == expected_steps
                ):
                    (
                        global_interval_loss,
                        global_interval_rows,
                        global_total_loss,
                        global_samples_seen,
                        global_correct,
                    ) = reduce_training_totals(
                        (
                            interval_loss,
                            interval_rows,
                            total_loss,
                            samples_seen,
                            correct,
                        ),
                        context,
                    )
                    if context.is_main:
                        assert loss_history is not None
                        loss_history.record(
                            epoch=epoch + 1,
                            global_step=(
                                global_step_offset + steps_seen
                            ),
                            epoch_step=steps_seen,
                            rows_seen=int(global_samples_seen),
                            interval_avg_loss=(
                                global_interval_loss
                                / global_interval_rows
                            ),
                            epoch_avg_loss=(
                                global_total_loss
                                / global_samples_seen
                            ),
                            epoch_accuracy=(
                                global_correct / global_samples_seen
                            ),
                        )
                    interval_loss = 0.0
                    interval_rows = 0
                if context.is_main:
                    progress.update(batch_rows * context.world_size)
                    progress.set_postfix(
                        loss=f"{total_loss / samples_seen:.4f}",
                        acc=f"{correct / samples_seen:.4f}",
                        refresh=False,
                    )
        finally:
            progress.close()

        if samples_seen != rows_per_rank:
            raise RuntimeError(
                f"rank {context.rank}, epoch {epoch + 1} yielded "
                f"{samples_seen:,} rows; expected "
                f"{rows_per_rank:,}. No checkpoint was saved."
            )
        if steps_seen != expected_steps:
            raise RuntimeError(
                f"rank {context.rank}, epoch {epoch + 1} used "
                f"{steps_seen:,} steps; expected "
                f"{expected_steps:,}"
            )

        (
            global_total_loss,
            global_samples_seen,
            global_correct,
        ) = reduce_training_totals(
            (total_loss, samples_seen, correct),
            context,
        )
        average_loss = global_total_loss / global_samples_seen
        accuracy = global_correct / global_samples_seen
        is_best = average_loss < best_loss
        if is_best:
            best_loss = average_loss

        if context.is_main:
            model_to_save = (
                model.module
                if isinstance(model, DistributedDataParallel)
                else model
            )
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "avg_loss": average_loss,
                "accuracy": accuracy,
                "best_loss": best_loss,
                "samples": expected_rows,
                "effective_samples": int(global_samples_seen),
                "optimizer_steps": steps_seen,
                "global_optimizer_steps": (
                    global_step_offset + steps_seen
                ),
                "loss_log_steps": args.loss_log_steps,
                "num_skills": len(skill2idx),
                "max_length": args.max_length,
                "batch_size": args.batch_size,
                "per_device_batch_size": args.batch_size,
                "global_batch_size": global_batch_size,
                "world_size": context.world_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "mixed_precision": args.mixed_precision,
                "fused_optimizer": args.fused_optimizer,
                "tf32": args.tf32,
                "attention_implementation": args.attention_implementation,
                "num_workers_per_rank": args.num_workers,
                "prefetch_factor": args.prefetch_factor,
                "shuffle_buffer_size": args.shuffle_buffer_size,
                "seed": args.seed,
            }
            epoch_path = (
                checkpoint_dir / f"epoch_{epoch + 1}.pt"
            )
            atomic_torch_save(checkpoint, epoch_path)
            if is_best:
                # Inference scripts expect a plain model state_dict.
                atomic_torch_save(
                    model_to_save.state_dict(), best_model_path
                )

            print(
                f"epoch {epoch + 1}: loss={average_loss:.6f}, "
                f"acc@1={accuracy:.6f}, unique rows="
                f"{expected_rows:,}, effective rows="
                f"{int(global_samples_seen):,}, steps={steps_seen:,}"
            )
            print(f"checkpoint: {epoch_path}")
            if is_best:
                print(f"best model: {best_model_path}")
        distributed_barrier(context)
        global_step_offset += steps_seen

    if context.is_main:
        assert loss_history is not None
        loss_history.close()
        print(
            f"training complete; best training loss="
            f"{best_loss:.6f}"
        )


def main() -> int:
    args = parse_args()
    context = initialize_distributed(args.device)
    try:
        train(args, context)
        return 0
    finally:
        cleanup_distributed(context)


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed(requested_device: str) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        return DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            device=resolve_device(requested_device),
        )

    requested = requested_device.strip().lower()
    if requested == "cpu":
        backend = "gloo"
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "torchrun requested multiple processes but CUDA is unavailable"
            )
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only "
                f"{torch.cuda.device_count()} visible CUDA devices exist"
            )
        backend = "nccl"
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def distributed_barrier(context: DistributedContext) -> None:
    if context.enabled:
        dist.barrier()


def cleanup_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()

def count_file_rows_with_signature(path: Path) -> tuple[str, int, int, int]:
    stat = path.stat()
    return (
        str(path),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        count_csv_rows(path),
    )


def load_file_row_manifest(
    manifest_path: Path,
    files: Sequence[Path],
) -> dict[Path, int] | None:
    if not manifest_path.is_file():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("version") != FILE_ROW_MANIFEST_VERSION:
            return None
        records = payload.get("files")
        if not isinstance(records, list) or len(records) != len(files):
            return None

        counts: dict[Path, int] = {}
        for path, record in zip(files, records):
            if not isinstance(record, dict) or record.get("path") != str(path):
                return None
            stat = path.stat()
            if (
                record.get("size") != int(stat.st_size)
                or record.get("mtime_ns") != int(stat.st_mtime_ns)
            ):
                return None
            rows = record.get("rows")
            if not isinstance(rows, int) or rows < 0:
                return None
            counts[path] = rows
        return counts
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def atomic_write_file_row_manifest(
    manifest_path: Path,
    records: Sequence[tuple[str, int, int, int]],
) -> None:
    payload = {
        "version": FILE_ROW_MANIFEST_VERSION,
        "files": [
            {
                "path": path,
                "size": size,
                "mtime_ns": mtime_ns,
                "rows": rows,
            }
            for path, size, mtime_ns, rows in records
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{manifest_path.name}.",
        suffix=".tmp",
        dir=manifest_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, manifest_path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    finally:
        temporary_path.unlink(missing_ok=True)


def build_file_row_counts(
    files: Sequence[Path],
    num_workers: int,
) -> tuple[dict[Path, int], list[tuple[str, int, int, int]]]:
    worker_count = min(max(1, num_workers), len(files))
    if worker_count <= 1:
        records = [count_file_rows_with_signature(path) for path in files]
    else:
        with Pool(processes=worker_count) as pool:
            records = list(
                pool.imap(
                    count_file_rows_with_signature,
                    files,
                    chunksize=16,
                )
            )
    counts = {Path(path): rows for path, _, _, rows in records}
    return counts, records


def distributed_file_row_counts(
    files: Sequence[Path],
    manifest_path: Path,
    num_workers: int,
    context: DistributedContext,
) -> tuple[dict[Path, int], str]:
    message: list[object] = [None]
    if context.is_main:
        try:
            counts = load_file_row_manifest(manifest_path, files)
            if counts is None:
                print(
                    "building exact per-file row-count manifest for DDP ...",
                    flush=True,
                )
                counts, records = build_file_row_counts(files, num_workers)
                atomic_write_file_row_manifest(manifest_path, records)
                source = f"CSV scan cached at {manifest_path}"
            else:
                source = str(manifest_path)
            message[0] = {
                "counts": {str(path): rows for path, rows in counts.items()},
                "source": source,
            }
        except BaseException:
            message[0] = {"error": traceback.format_exc()}

    if context.enabled:
        dist.broadcast_object_list(message, src=0)
    payload = message[0]
    if not isinstance(payload, dict):
        raise RuntimeError("rank 0 did not broadcast the file row counts")
    if "error" in payload:
        raise RuntimeError(
            "rank 0 failed while preparing DDP row counts:\n"
            f"{payload['error']}"
        )
    raw_counts = payload.get("counts")
    if not isinstance(raw_counts, dict):
        raise RuntimeError("invalid file row-count payload from rank 0")
    return (
        {Path(path): int(rows) for path, rows in raw_counts.items()},
        str(payload.get("source", manifest_path)),
    )



def reduce_training_totals(
    values: Sequence[float | int],
    context: DistributedContext,
) -> tuple[float, ...]:
    totals = torch.tensor(
        list(values),
        dtype=torch.float64,
        device=context.device,
    )
    if context.enabled:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return tuple(float(value) for value in totals.cpu().tolist())
if __name__ == "__main__":
    raise SystemExit(main())
