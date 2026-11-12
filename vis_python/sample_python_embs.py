#!/usr/bin/env python3
"""Sample Python masked-skill instances and export fine-tuned BERT MLP embeddings.

The output layout intentionally matches the input expected by ``vis_python.py``:

    OUTPUT_DIR/2010_emb_mlp.npy
    OUTPUT_DIR/2010.csv
    ...
    OUTPUT_DIR/2025_emb_mlp.npy
    OUTPUT_DIR/2025.csv

Each CSV row is aligned with the corresponding row of the NumPy array.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import random
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TextIO

import numpy as np
import torch
from transformers import BertTokenizerFast

BASE_DATA_DIR = Path("/home/jovyan/LEM_data3/hyunjincho/LEM")
DEFAULT_REPO_ROOT = Path(
    "/home/jovyan/hyunjincho/LEM_journal/contextual_skills_substitutes"
)
DEFAULT_TEST_ROOT = BASE_DATA_DIR / "all_train_test" / "test"
DEFAULT_MODEL_NAME = BASE_DATA_DIR / "bert_pretrained"
DEFAULT_CHECKPOINT = BASE_DATA_DIR / "models" / "bert_fine_tuned_model" / "epoch_3.pt"
DEFAULT_VOCAB = BASE_DATA_DIR / "all_filterd" / "skill2idx.json"
DEFAULT_OUTPUT = BASE_DATA_DIR / "vis_python_samlpes"
DEFAULT_PATTERN = "preprocessed_*.csv.gz"
DEFAULT_YEARS = tuple(range(2010, 2026))
REQUIRED_COLUMNS = (
    "masked_sentence",
    "true_skill",
    "row_idx",
    "file_path",
    "soc_2_name",
)
OUTPUT_COLUMNS = (
    "year",
    "onet_major_group",
    "soc_2_name",
    "true_skill",
    "masked_sentence",
    "row_idx",
    "file_path",
    "preprocessed_file_path",
)
_PLANG_TAIL = re.compile(r"\s*\(programming language\)\s*$", re.IGNORECASE)
_YEAR = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


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
            "Reservoir-sample target-skill rows by year and export the "
            "fine-tuned BERT MLP representation at [MASK]."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--model-name", type=Path, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-skill", default="python")
    parser.add_argument("--years", nargs="+", type=int, default=list(DEFAULT_YEARS))
    parser.add_argument(
        "--samples-per-year",
        type=int,
        default=0,
        help="Rows retained per year; 0 keeps every eligible target-skill row.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-amp", type=parse_bool, default=True)
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    args = parser.parse_args()

    for name in (
        "repo_root",
        "test_root",
        "model_name",
        "checkpoint_path",
        "vocab_path",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.samples_per_year < 0 or args.batch_size < 1 or args.max_length < 3:
        parser.error(
            "--samples-per-year must be >= 0, --batch-size >= 1, "
            "and --max-length >= 3"
        )
    if not args.pattern:
        parser.error("--pattern must not be empty")
    if len(set(args.years)) != len(args.years):
        parser.error("--years contains duplicates")
    return args


def normalize_skill(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return _PLANG_TAIL.sub("", value.strip()).strip().lower()


def open_csv(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def infer_year(path: Path, root: Path) -> int:
    relative = path.relative_to(root)
    for part in relative.parts:
        if re.fullmatch(r"20\d{2}", part):
            return int(part)
    matches = _YEAR.findall(relative.as_posix())
    if matches:
        return int(matches[0])
    raise ValueError(f"cannot infer year from {path}")


def discover_files(root: Path, pattern: str, years: set[int]) -> list[tuple[int, Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"test root not found: {root}")
    found: list[tuple[int, Path]] = []
    for path in sorted(root.rglob(pattern)):
        if path.is_file():
            year = infer_year(path, root)
            if year in years:
                found.append((year, path))
    if not found:
        raise FileNotFoundError(f"no files matching {pattern!r} for requested years")
    return found


def validate_header(path: Path, fieldnames: list[str] | None) -> None:
    if not fieldnames:
        raise ValueError(f"CSV has no header: {path}")
    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")


def reservoir_sample(
    files: Iterable[tuple[int, Path]],
    root: Path,
    years: set[int],
    target_skill: str,
    samples_per_year: int,
    seed: int,
) -> tuple[dict[int, list[dict[str, str]]], dict[str, Any]]:
    rng = random.Random(seed)
    reservoirs = {year: [] for year in sorted(years)}
    eligible_seen = Counter()
    input_rows = Counter()
    skip_reasons = Counter()

    for file_index, (year, path) in enumerate(files, start=1):
        relative = path.relative_to(root).as_posix()
        with open_csv(path) as handle:
            reader = csv.DictReader(handle)
            validate_header(path, reader.fieldnames)
            for row in reader:
                input_rows[year] += 1
                if normalize_skill(row.get("true_skill")) != target_skill:
                    continue
                sentence = row.get("masked_sentence", "")
                if not sentence:
                    skip_reasons["empty_sentence"] += 1
                    continue
                if "[MASK]" not in sentence:
                    skip_reasons["missing_mask"] += 1
                    continue
                soc = (row.get("soc_2_name") or "").strip()
                if not soc:
                    skip_reasons["empty_soc_2_name"] += 1
                sample = {
                    "year": str(year),
                    "onet_major_group": soc,
                    "soc_2_name": soc,
                    "true_skill": (row.get("true_skill") or "").strip(),
                    "masked_sentence": sentence,
                    "row_idx": row.get("row_idx", ""),
                    "file_path": row.get("file_path", ""),
                    "preprocessed_file_path": relative,
                }
                eligible_seen[year] += 1
                seen = eligible_seen[year]
                reservoir = reservoirs[year]
                if samples_per_year == 0:
                    reservoir.append(sample)
                elif len(reservoir) < samples_per_year:
                    reservoir.append(sample)
                else:
                    replacement = rng.randrange(seen)
                    if replacement < samples_per_year:
                        reservoir[replacement] = sample
        if file_index % 100 == 0:
            print(f"[sample] scanned {file_index:,} files", flush=True)

    for year in reservoirs:
        reservoirs[year].sort(
            key=lambda row: (
                row["preprocessed_file_path"],
                row["row_idx"],
                row["file_path"],
            )
        )
    summary = {
        "input_rows_by_year": {str(k): input_rows[k] for k in sorted(years)},
        "eligible_python_rows_by_year": {
            str(k): eligible_seen[k] for k in sorted(years)
        },
        "retained_rows_by_year": {
            str(k): len(reservoirs[k]) for k in sorted(years)
        },
        "sampling_mode": (
            "all_eligible_rows"
            if samples_per_year == 0
            else "reservoir_sample_per_year"
        ),
        "skip_reasons": dict(skip_reasons),
    }
    return reservoirs, summary


def load_vocab_size(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"vocabulary not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        vocab = json.load(handle)
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError(f"expected a non-empty JSON object: {path}")
    indices = set(vocab.values())
    if indices != set(range(len(vocab))):
        raise ValueError("skill2idx indices must be contiguous from zero")
    return len(vocab)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    return device


def import_model(repo_root: Path) -> type[torch.nn.Module]:
    model_dir = repo_root / "model_trains"
    if not (model_dir / "model.py").is_file():
        raise FileNotFoundError(f"model.py not found under {model_dir}")
    sys.path.insert(0, str(model_dir))
    from model import BERTForSkillPrediction  # type: ignore

    return BERTForSkillPrediction


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model_class = import_model(args.repo_root)
    model = model_class(str(args.model_name), num_skills=load_vocab_size(args.vocab_path))
    try:
        checkpoint = torch.load(
            args.checkpoint_path, map_location="cpu", weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError(f"unsupported checkpoint format: {args.checkpoint_path}")
    model.load_state_dict(checkpoint, strict=True)
    model.eval().to(device)
    return model


def extract_mlp_embeddings(
    rows: list[dict[str, str]],
    tokenizer: BertTokenizerFast,
    model: torch.nn.Module,
    device: torch.device,
    max_length: int,
    batch_size: int,
    amp_enabled: bool,
) -> np.ndarray:
    batches: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded = tokenizer(
            [row["masked_sentence"] for row in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        mask_locations = input_ids.eq(tokenizer.mask_token_id)
        missing = ~mask_locations.any(dim=1)
        if missing.any():
            index = int(missing.nonzero(as_tuple=False)[0].item())
            row = batch[index]
            raise ValueError(
                "[MASK] was lost during tokenization for "
                f"{row['preprocessed_file_path']} row_idx={row['row_idx']}; "
                "use the same --max-length as preprocessing"
            )
        mask_idx = mask_locations.to(torch.int64).argmax(dim=1).to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            mask_vec = outputs.last_hidden_state[
                torch.arange(input_ids.size(0), device=device), mask_idx
            ]
            mlp = model.layer_norm(model.transform_act_fn(model.dense(mask_vec)))
        batches.append(mlp.float().cpu().numpy())
    hidden_size = int(model.bert.config.hidden_size)
    return (
        np.concatenate(batches, axis=0)
        if batches
        else np.empty((0, hidden_size), dtype=np.float32)
    )


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def atomic_save_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    years = set(args.years)
    target_skill = normalize_skill(args.target_skill)
    if not target_skill:
        raise ValueError("--target-skill normalizes to an empty value")
    for required in (args.model_name, args.checkpoint_path, args.vocab_path):
        if not required.exists():
            raise FileNotFoundError(required)
    existing = [
        args.output_dir / f"{year}_emb_mlp.npy"
        for year in years
        if (args.output_dir / f"{year}_emb_mlp.npy").exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{len(existing)} output arrays already exist; pass --overwrite true"
        )

    files = discover_files(args.test_root, args.pattern, years)
    print(f"[sample] discovered {len(files):,} source files")
    reservoirs, sampling_summary = reservoir_sample(
        files,
        args.test_root,
        years,
        target_skill,
        args.samples_per_year,
        args.seed,
    )

    device = resolve_device(args.device)
    amp_enabled = bool(args.use_amp and device.type == "cuda")
    tokenizer = BertTokenizerFast.from_pretrained(str(args.model_name))
    model = load_model(args, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dimensions: dict[str, list[int]] = {}
    for year in sorted(years):
        rows = reservoirs[year]
        print(f"[embed] {year}: {len(rows):,} rows", flush=True)
        embeddings = extract_mlp_embeddings(
            rows,
            tokenizer,
            model,
            device,
            args.max_length,
            args.batch_size,
            amp_enabled,
        )
        atomic_save_npy(args.output_dir / f"{year}_emb_mlp.npy", embeddings)
        atomic_save_csv(args.output_dir / f"{year}.csv", rows)
        dimensions[str(year)] = list(embeddings.shape)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_skill": target_skill,
        "years": sorted(years),
        "samples_per_year": args.samples_per_year,
        "seed": args.seed,
        "test_root": str(args.test_root),
        "source_pattern": args.pattern,
        "model_name": str(args.model_name),
        "checkpoint_path": str(args.checkpoint_path),
        "checkpoint_sha256": sha256(args.checkpoint_path),
        "vocab_path": str(args.vocab_path),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "device": str(device),
        "amp_enabled": amp_enabled,
        "representation": "mask_hidden_dense_activation_layernorm",
        "array_shapes": dimensions,
        "sampling": sampling_summary,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=args.output_dir, delete=False
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, manifest_path)
    print(f"[done] outputs: {args.output_dir}")
    print(f"[done] manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
