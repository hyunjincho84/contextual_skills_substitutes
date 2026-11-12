#!/usr/bin/env python3
"""Stream BERT evaluation over independent masked-sentence test rows.

All source files for one month are atomically combined into the legacy
OUT/pred/YYYY/predictions_YYYY-MM.csv.gz layout. Completion markers make a
rerun skip only outputs whose inputs and model configuration still match.
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
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence, TextIO

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm
from transformers import BertTokenizerFast

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR.parent / "model_trains"))
from model import BERTForSkillPrediction  # noqa: E402

BASE_DATA_DIR = Path(
    os.environ.get("BASE_DATA_DIR", "/home/jovyan/LEM_data3/hyunjincho/LEM")
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
DEFAULT_CHECKPOINT_PATH = Path(
    os.environ.get(
        "BEST_MODEL_PT",
        str(BASE_DATA_DIR / "checkpoints(www)_new" / "best_model.pt"),
    )
)
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("BERT_OUT_DIR", str(BASE_DATA_DIR / "bert_pred_new"))
)
DEFAULT_PATTERN = "preprocessed_*.csv.gz"
MAX_WORKERS = 8
OUTPUT_SCHEMA_VERSION = 3
REQUIRED_COLUMNS = (
    "masked_sentence",
    "true_skill",
    "row_idx",
    "soc_2_name",
    "file_path",
)
PREDICTION_COLUMNS = (
    "year",
    "file",
    "preprocessed_file_path",
    "truth",
    "substitution",
    "pred_top1",
    "pred_top5",
    "pred_top5_probs",
    "pred_top10",
    "pred_top10_probs",
    "pred_topk",
    "pred_topk_probs",
    "target_probability",
    "target_rank",
)
MONTH_RE = re.compile(r"^(20[0-9]{2})-(0[1-9]|1[0-2])$")


def parse_bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate masked_sentence -> true_skill rows from the recursive "
            "posting-safe test split."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--test-dir", type=Path, default=None,
        help="Default: <data-root>/test",
    )
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of truth-inclusive ranked predictions to retain (minimum 10).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(MAX_WORKERS, os.cpu_count() or 1),
        help=f"Parallel CSV/tokenization workers, 0-{MAX_WORKERS}.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=1,
        help="gzip output compression level, 0-9 (default: 1 for speed).",
    )
    parser.add_argument(
        "--use-amp", type=parse_bool, default=True,
        help="Use CUDA float16 autocast; automatically disabled off CUDA.",
    )
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--strict", type=parse_bool, default=True,
        help="Fail instead of skipping malformed/out-of-vocabulary rows.",
    )
    parser.add_argument(
        "--overwrite", type=parse_bool, default=False,
        help="Recompute months with valid completion markers.",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Torch device such as auto, cuda, cuda:0, or cpu.",
    )
    # Kept so the existing pred_all.sh CLI remains accepted. csv.reader streams.
    parser.add_argument(
        "--chunk-size", type=int, default=200_000, help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.test_dir is None:
        args.test_dir = args.data_root / "test"
    if args.max_length < 3:
        parser.error("--max-length must be >= 3")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.top_k < 10:
        parser.error("--top-k must be >= 10 for compatible Top-10 output")
    if not 0 <= args.num_workers <= MAX_WORKERS:
        parser.error(f"--num-workers must be between 0 and {MAX_WORKERS}")
    if not 0 <= args.compression_level <= 9:
        parser.error("--compression-level must be between 0 and 9")
    if args.prefetch_factor < 1 or args.chunk_size < 1:
        parser.error("--prefetch-factor and --chunk-size must be >= 1")
    return args


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    return device


def discover_files(root: Path, pattern: str) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"test directory not found: {root}")
    files = sorted(path for path in root.rglob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(
            f"no files matching {pattern!r} recursively under {root}"
        )
    return files


def month_key_for(path: Path, test_root: Path) -> str:
    relative = path.relative_to(test_root)
    for part in reversed(relative.parts[:-1]):
        if MONTH_RE.fullmatch(part):
            return part
    match = re.search(r"(20[0-9]{2})-(0[1-9]|1[0-2])", relative.name)
    if match:
        return match.group(0)
    raise ValueError(f"cannot infer YYYY-MM from test file: {path}")


def group_files_by_month(
    files: Sequence[Path], test_root: Path
) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in files:
        groups.setdefault(month_key_for(path, test_root), []).append(path)
    return {month: sorted(paths) for month, paths in sorted(groups.items())}


def monthly_output_path(output_dir: Path, month: str) -> Path:
    return (
        output_dir / "pred" / month[:4] / f"predictions_{month}.csv.gz"
    )


def marker_path_for(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".complete.json")


def open_csv(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="\n")
    return path.open("r", encoding="utf-8", newline="\n")


def normalized_csv_lines(handle: TextIO) -> Iterator[str]:
    for line in handle:
        yield line.replace("\r\n", "\n").replace("\r", " ")


def read_header(path: Path) -> list[str]:
    with open_csv(path) as handle:
        reader = csv.reader(normalized_csv_lines(handle))
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path} has no CSV header") from exc
    missing = [column for column in REQUIRED_COLUMNS if column not in header]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    return header


def load_skill_vocabulary(
    path: Path,
) -> tuple[dict[str, int], dict[int, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"skill vocabulary not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path} must contain a non-empty JSON object")
    skill2idx: dict[str, int] = {}
    for skill, index in raw.items():
        normalized = str(skill).strip().lower()
        if not normalized or normalized in skill2idx:
            raise ValueError(f"invalid/duplicate normalized skill: {skill!r}")
        if not isinstance(index, int):
            raise ValueError(f"non-integer index for {skill!r}: {index!r}")
        skill2idx[normalized] = index
    if set(skill2idx.values()) != set(range(len(skill2idx))):
        raise ValueError(f"{path} indices must be contiguous from 0")
    return skill2idx, {index: skill for skill, index in skill2idx.items()}


def stat_fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def run_signature(args: argparse.Namespace) -> str:
    payload = {
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "vocab": stat_fingerprint(args.vocab_path),
        "checkpoint": stat_fingerprint(args.checkpoint_path),
        "model_name": str(args.model_name),
        "max_length": args.max_length,
        "top_k": args.top_k,
        "strict": args.strict,
        "compression_level": args.compression_level,
        "use_amp": args.amp_enabled,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def month_signature(
    month: str,
    paths: Sequence[Path],
    test_root: Path,
    run_id: str,
) -> str:
    sources = [
        {
            "relative_path": path.relative_to(test_root).as_posix(),
            **stat_fingerprint(path),
        }
        for path in paths
    ]
    payload = {"month": month, "run_signature": run_id, "sources": sources}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_valid_marker(
    output_path: Path, expected_signature: str, run_id: str
) -> dict[str, Any] | None:
    marker_path = marker_path_for(output_path)
    if not output_path.is_file() or not marker_path.is_file():
        return None
    try:
        with marker_path.open("r", encoding="utf-8") as handle:
            marker = json.load(handle)
    except (OSError, ValueError):
        return None
    output_stat = stat_fingerprint(output_path)
    valid = (
        marker.get("complete") is True
        and marker.get("run_signature") == run_id
        and marker.get("month_signature") == expected_signature
        and marker.get("output_rows", -1) >= 0
        and marker.get("output_size") == output_stat["size"]
        and marker.get("output_mtime_ns") == output_stat["mtime_ns"]
    )
    return marker if valid else None


# kind, month, relative source, row/header, sentence/stats, label/file count
Record = tuple[str, str, Any, Any, Any, Any]


class PredictionIterableDataset(IterableDataset):
    """Give each worker whole months, preserving atomic month boundaries."""

    def __init__(
        self,
        month_groups: Sequence[tuple[str, tuple[Path, ...]]],
        test_root: Path,
        skill2idx: dict[str, int],
        expected_header: Sequence[str],
        strict: bool,
    ) -> None:
        super().__init__()
        self.month_groups = tuple(month_groups)
        self.test_root = test_root
        self.skill2idx = skill2idx
        self.expected_header = tuple(expected_header)
        self.strict = strict

    def _problem(
        self, message: str, stats: dict[str, int], reason: str
    ) -> None:
        if self.strict:
            raise ValueError(message)
        stats[reason] += 1

    def _iter_month(
        self, month: str, paths: Sequence[Path]
    ) -> Iterator[Record]:
        stats = {
            "input_rows": 0,
            "eligible_rows": 0,
            "malformed_rows": 0,
            "missing_mask": 0,
            "empty_sentence": 0,
            "empty_skill": 0,
            "out_of_vocabulary": 0,
        }
        for path in paths:
            relative = path.relative_to(self.test_root).as_posix()
            with open_csv(path) as handle:
                reader = csv.reader(normalized_csv_lines(handle))
                try:
                    header = next(reader)
                except StopIteration as exc:
                    raise ValueError(f"{path} has no CSV header") from exc
                if tuple(header) != self.expected_header:
                    raise ValueError(f"{path} has a different schema: {header}")
                for csv_row_number, values in enumerate(reader, start=2):
                    stats["input_rows"] += 1
                    origin = f"{path}:{csv_row_number}"
                    if len(values) != len(header):
                        self._problem(
                            f"{origin} has {len(values)} fields; "
                            f"expected {len(header)}",
                            stats,
                            "malformed_rows",
                        )
                        continue
                    row = dict(zip(header, values))
                    sentence = row["masked_sentence"]
                    if not sentence:
                        self._problem(
                            f"{origin} has an empty masked_sentence",
                            stats,
                            "empty_sentence",
                        )
                        continue
                    if "[MASK]" not in sentence:
                        self._problem(
                            f"{origin} masked_sentence has no [MASK]",
                            stats,
                            "missing_mask",
                        )
                        continue
                    skill = row["true_skill"].strip().lower()
                    if not skill:
                        self._problem(
                            f"{origin} has an empty true_skill",
                            stats,
                            "empty_skill",
                        )
                        continue
                    label = self.skill2idx.get(skill)
                    if label is None:
                        self._problem(
                            f"{origin} skill {skill!r} is outside vocabulary",
                            stats,
                            "out_of_vocabulary",
                        )
                        continue
                    stats["eligible_rows"] += 1
                    yield (
                        "sample",
                        month,
                        relative,
                        row,
                        sentence,
                        label,
                    )
        yield (
            "end",
            month,
            "",
            list(self.expected_header),
            stats,
            len(paths),
        )

    def __iter__(self) -> Iterator[Record]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        for month, paths in self.month_groups[worker_id::worker_count]:
            yield from self._iter_month(month, paths)


class TokenizeBatch:
    """Lazily create one fast tokenizer in each persistent loader worker."""

    def __init__(self, model_name: str, max_length: int) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer: BertTokenizerFast | None = None

    def __call__(self, records: Sequence[Record]) -> dict[str, Any]:
        samples = [record for record in records if record[0] == "sample"]
        ends = [record for record in records if record[0] == "end"]
        if not samples:
            return {"tensors": None, "metadata": [], "ends": ends}
        if self.tokenizer is None:
            self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)
        encoded = self.tokenizer(
            [record[4] for record in samples],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        mask_locations = input_ids.eq(self.tokenizer.mask_token_id)
        missing = ~mask_locations.any(dim=1)
        if missing.any():
            index = int(missing.nonzero(as_tuple=False)[0].item())
            relative, row = samples[index][2], samples[index][3]
            raise ValueError(
                f"{relative} row_idx={row.get('row_idx', '')!r} lost [MASK] "
                "after tokenization; --max-length differs from preprocessing"
            )
        mask_indices = mask_locations.to(torch.int64).argmax(dim=1)
        labels = torch.tensor(
            [record[5] for record in samples], dtype=torch.long
        )
        metadata = [
            (record[1], record[2], record[3]) for record in samples
        ]
        return {
            "tensors": (input_ids, attention_mask, mask_indices, labels),
            "metadata": metadata,
            "ends": ends,
        }


def rank_predictions_and_substitution(
    logits: torch.Tensor, labels: torch.Tensor, top_k: int
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Rank with truth included and select the best truth-excluded substitute."""
    probabilities = F.softmax(logits.float(), dim=-1)
    target_probabilities = probabilities.gather(
        1, labels[:, None]
    ).squeeze(1)
    target_logits = logits.gather(1, labels[:, None])
    target_ranks = logits.gt(target_logits).sum(dim=1) + 1

    top = torch.topk(probabilities, k=top_k, dim=-1)
    substitution_probabilities = probabilities.clone()
    substitution_probabilities.scatter_(1, labels[:, None], float("-inf"))
    substitution_indices = substitution_probabilities.argmax(dim=1)
    return (
        top.indices,
        top.values,
        substitution_indices,
        target_probabilities,
        target_ranks,
    )


def prediction_columns(header: Sequence[str]) -> list[str]:
    return list(header) + [
        column for column in PREDICTION_COLUMNS if column not in header
    ]


@dataclass
class AtomicCsvWriter:
    destination: Path
    header: list[str]
    compression_level: int = 1

    def __post_init__(self) -> None:
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=self.destination.name + ".",
            suffix=".tmp",
            dir=self.destination.parent,
            delete=False,
        )
        self.temporary_path = Path(handle.name)
        handle.close()
        self._gzip = gzip.open(
            self.temporary_path,
            "wt",
            encoding="utf-8",
            newline="",
            compresslevel=self.compression_level,
        )
        self._writer = csv.DictWriter(
            self._gzip,
            fieldnames=prediction_columns(self.header),
            extrasaction="ignore",
        )
        self._writer.writeheader()
        self.rows = 0

    def write(self, row: dict[str, Any]) -> None:
        self._writer.writerow(row)
        self.rows += 1

    def commit(self) -> None:
        self._gzip.close()
        os.replace(self.temporary_path, self.destination)

    def abort(self) -> None:
        try:
            self._gzip.close()
        finally:
            self.temporary_path.unlink(missing_ok=True)


def atomic_json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_path, path)


def atomic_text_dump(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(value)
    os.replace(temporary_path, path)


def load_model(
    model_name: str,
    checkpoint_path: Path,
    num_skills: int,
    device: torch.device,
) -> BERTForSkillPrediction:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    model = BERTForSkillPrediction(model_name, num_skills=num_skills)
    try:
        state = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"unsupported checkpoint format: {checkpoint_path}")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def empty_counts() -> dict[str, int]:
    return {
        "samples": 0,
        "input_rows": 0,
        "skipped_rows": 0,
        "top1_correct": 0,
        "top3_correct": 0,
        "top5_correct": 0,
    }


def add_marker(counts: dict[str, int], marker: dict[str, Any]) -> None:
    counts["samples"] += int(marker["output_rows"])
    counts["input_rows"] += int(marker["input_rows"])
    counts["skipped_rows"] += int(marker["skipped_rows"])
    for key in ("top1_correct", "top3_correct", "top5_correct"):
        counts[key] += int(marker[key])


def evaluate(args: argparse.Namespace) -> None:
    args.test_dir = args.test_dir.expanduser().resolve()
    args.vocab_path = args.vocab_path.expanduser().resolve()
    args.checkpoint_path = args.checkpoint_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    device = resolve_device(args.device)
    args.amp_enabled = args.use_amp and device.type == "cuda"
    skill2idx, idx2skill = load_skill_vocabulary(args.vocab_path)
    if args.top_k > len(skill2idx):
        raise ValueError(
            f"--top-k ({args.top_k}) exceeds vocabulary size ({len(skill2idx)})"
        )

    files = discover_files(args.test_dir, args.pattern)
    groups = group_files_by_month(files, args.test_dir)
    header = read_header(files[0])
    run_id = run_signature(args)
    aggregate = empty_counts()
    pending: list[tuple[str, tuple[Path, ...]]] = []
    month_signatures: dict[str, str] = {}
    resumed_months = 0
    for month, paths in groups.items():
        signature = month_signature(month, paths, args.test_dir, run_id)
        month_signatures[month] = signature
        output_path = monthly_output_path(args.output_dir, month)
        marker = None if args.overwrite else load_valid_marker(
            output_path, signature, run_id
        )
        if marker is None:
            pending.append((month, tuple(paths)))
        else:
            resumed_months += 1
            add_marker(aggregate, marker)

    print(f"test files: {len(files):,}")
    print(f"test months: {len(groups):,}")
    print(f"already complete: {resumed_months:,} months")
    print(f"pending: {len(pending):,} months")

    if pending:
        model = load_model(
            args.model_name, args.checkpoint_path, len(skill2idx), device
        )
        worker_count = min(args.num_workers, len(pending))
        dataset = PredictionIterableDataset(
            pending, args.test_dir, skill2idx, header, args.strict
        )
        loader_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": args.batch_size,
            "shuffle": False,
            "num_workers": worker_count,
            "collate_fn": TokenizeBatch(args.model_name, args.max_length),
            "pin_memory": device.type == "cuda",
        }
        if worker_count:
            loader_kwargs.update(
                prefetch_factor=args.prefetch_factor,
                persistent_workers=True,
            )
        loader = DataLoader(**loader_kwargs)
        writers: dict[str, AtomicCsvWriter] = {}
        month_counts: dict[str, dict[str, int]] = {}
        progress = tqdm(total=len(pending), desc="BERT test", unit="month")
        try:
            with torch.inference_mode():
                for payload in loader:
                    if payload["tensors"] is not None:
                        input_ids, attention_mask, mask_indices, labels = (
                            payload["tensors"]
                        )
                        with torch.autocast(
                            device_type=device.type,
                            dtype=torch.float16,
                            enabled=args.amp_enabled,
                        ):
                            logits = model(
                                input_ids.to(device, non_blocking=True),
                                attention_mask.to(device, non_blocking=True),
                                mask_indices.to(device, non_blocking=True),
                            )
                        labels_device = labels.to(device, non_blocking=True)
                        (
                            candidate_indices,
                            candidate_probs,
                            substitution_indices,
                            target_probs,
                            target_ranks,
                        ) = rank_predictions_and_substitution(
                            logits, labels_device, args.top_k
                        )
                        candidate_indices = candidate_indices.cpu()
                        candidate_probs = candidate_probs.cpu()
                        substitution_indices = substitution_indices.cpu()
                        target_probs = target_probs.cpu()
                        target_ranks = target_ranks.cpu()
                        for index, (
                            month,
                            relative,
                            source_row,
                        ) in enumerate(payload["metadata"]):
                            if month not in writers:
                                writers[month] = AtomicCsvWriter(
                                    monthly_output_path(
                                        args.output_dir, month
                                    ),
                                    header,
                                    args.compression_level,
                                )
                                month_counts[month] = empty_counts()
                            truth_index = int(labels[index].item())
                            truth = idx2skill[truth_index]
                            candidates = [
                                idx2skill[int(value)]
                                for value in candidate_indices[index].tolist()
                            ]
                            probabilities = [
                                float(value)
                                for value in candidate_probs[index].tolist()
                            ]
                            substitution = idx2skill[
                                int(substitution_indices[index].item())
                            ]
                            rank = int(target_ranks[index].item())
                            output_row = dict(source_row)
                            output_row.update(
                                {
                                    "year": month[:4],
                                    "file": Path(relative).name,
                                    "preprocessed_file_path": relative,
                                    "truth": truth,
                                    "substitution": substitution,
                                    "pred_top1": candidates[0],
                                    "pred_top5": "|".join(candidates[:5]),
                                    "pred_top5_probs": "|".join(
                                        f"{value:.9g}"
                                        for value in probabilities[:5]
                                    ),
                                    "pred_top10": "|".join(candidates[:10]),
                                    "pred_top10_probs": "|".join(
                                        f"{value:.9g}"
                                        for value in probabilities[:10]
                                    ),
                                    "pred_topk": "|".join(candidates),
                                    "pred_topk_probs": "|".join(
                                        f"{value:.9g}"
                                        for value in probabilities
                                    ),
                                    "target_probability": (
                                        f"{float(target_probs[index]):.9g}"
                                    ),
                                    "target_rank": rank,
                                }
                            )
                            writers[month].write(output_row)
                            counts = month_counts[month]
                            counts["top1_correct"] += int(rank <= 1)
                            counts["top3_correct"] += int(rank <= 3)
                            counts["top5_correct"] += int(rank <= 5)

                    for (
                        _,
                        month,
                        _,
                        end_header,
                        stats,
                        source_file_count,
                    ) in payload["ends"]:
                        if list(end_header) != header:
                            raise RuntimeError(f"{month}: schema changed")
                        if month not in writers:
                            writers[month] = AtomicCsvWriter(
                                monthly_output_path(args.output_dir, month),
                                header,
                                args.compression_level,
                            )
                            month_counts[month] = empty_counts()
                        writer = writers[month]
                        if writer.rows != stats["eligible_rows"]:
                            raise RuntimeError(
                                f"{month}: wrote {writer.rows} rows but "
                                f"streamed {stats['eligible_rows']}"
                            )
                        counts = month_counts[month]
                        writer.commit()
                        output_stat = stat_fingerprint(writer.destination)
                        skipped = (
                            stats["input_rows"] - stats["eligible_rows"]
                        )
                        marker = {
                            "complete": True,
                            "month": month,
                            "source_files": source_file_count,
                            "output_path": str(writer.destination),
                            "output_rows": writer.rows,
                            "output_size": output_stat["size"],
                            "output_mtime_ns": output_stat["mtime_ns"],
                            "input_rows": stats["input_rows"],
                            "skipped_rows": skipped,
                            "top1_correct": counts["top1_correct"],
                            "top3_correct": counts["top3_correct"],
                            "top5_correct": counts["top5_correct"],
                            "skip_reasons": {
                                key: value
                                for key, value in stats.items()
                                if key not in {"input_rows", "eligible_rows"}
                            },
                            "run_signature": run_id,
                            "month_signature": month_signatures[month],
                        }
                        atomic_json_dump(
                            marker, marker_path_for(writer.destination)
                        )
                        add_marker(aggregate, marker)
                        del writers[month]
                        del month_counts[month]
                        progress.update(1)
        except BaseException:
            for writer in writers.values():
                writer.abort()
            raise
        finally:
            progress.close()

    total = aggregate["samples"]
    summary = {
        "test_root": str(args.test_dir),
        "output_root": str(args.output_dir / "pred"),
        "test_files": len(files),
        "test_months": len(groups),
        "resumed_months": resumed_months,
        "processed_months": len(pending),
        **aggregate,
        "top1_accuracy": (
            aggregate["top1_correct"] / total if total else 0.0
        ),
        "top3_accuracy": (
            aggregate["top3_correct"] / total if total else 0.0
        ),
        "top5_accuracy": (
            aggregate["top5_correct"] / total if total else 0.0
        ),
        "ranked_predictions_include_truth": True,
        "substitution_excludes_truth": True,
        "top_k": args.top_k,
        "max_length": args.max_length,
        "compression_level": args.compression_level,
        "use_amp": args.amp_enabled,
        "run_signature": run_id,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(summary, args.output_dir / "test_results.json")
    result_text = (
        "=== Test Evaluation ===\n"
        f"Samples: {total:,}\n"
        f"Top-1 Acc: {summary['top1_accuracy']:.4f}\n"
        f"Top-3 Acc: {summary['top3_accuracy']:.4f}\n"
        f"Top-5 Acc: {summary['top5_accuracy']:.4f}\n"
    )
    atomic_text_dump(result_text, args.output_dir / "test_results.txt")
    print(result_text, end="")
    print(f"predictions: {args.output_dir / 'pred'}")
    print(f"summary: {args.output_dir / 'test_results.json'}")


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
