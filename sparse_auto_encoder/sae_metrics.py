#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate layer-wise sparse autoencoders on a fixed test set.

The script performs inference only; it does not update BERT or SAE weights.
It prints a layer-wise report and writes the same report to a text file.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


BASE_DATA_DIR = Path("/home/jovyan/LEM_data3/hyunjincho/LEM")
DEFAULT_TEST_PATTERN = str(
    BASE_DATA_DIR
    / "all_train_test"
    / "test_clean"
    / "20*"
    / "20*"
    / "preprocessed_*.csv.gz"
)
DEFAULT_MODEL_NAME = BASE_DATA_DIR / "bert_pretrained"
DEFAULT_BERT_CHECKPOINT = (
    BASE_DATA_DIR / "models" / "bert_fine_tuned_model" / "best_model.pt"
)
DEFAULT_SAE_ROOT = BASE_DATA_DIR / "models" / "sae_layerwise_out"
DEFAULT_OUTPUT_DIR = BASE_DATA_DIR / "sparse_auto_encoder" / "metrics"


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got: {value}")


def parse_layers(value: str) -> list[int]:
    if "-" in value:
        start_text, end_text = value.split("-", maxsplit=1)
        layers = list(range(int(start_text), int(end_text) + 1))
    else:
        layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not layers or any(layer < 1 or layer > 12 for layer in layers):
        raise argparse.ArgumentTypeError("layers must be within 1..12")
    if len(layers) != len(set(layers)):
        raise argparse.ArgumentTypeError("layers must not contain duplicates")
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute activation, dead-feature, and reconstruction metrics for "
            "layer-wise SAE checkpoints."
        )
    )
    parser.add_argument("--test-pattern", default=DEFAULT_TEST_PATTERN)
    parser.add_argument("--text-col", default="masked_sentence")
    parser.add_argument("--model-name", type=Path, default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--best-model-pt", type=Path, default=DEFAULT_BERT_CHECKPOINT
    )
    parser.add_argument("--sae-root", type=Path, default=DEFAULT_SAE_ROOT)
    parser.add_argument("--checkpoint-name", default="sae_best.pt")
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("1-12"))
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--pin-memory", type=parse_bool, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-amp", type=parse_bool, default=True)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--allow-tf32", type=parse_bool, default=True)
    parser.add_argument(
        "--activation-threshold",
        type=float,
        default=0.0,
        help="An SAE feature is active when f > threshold (default: 0).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for a quick run; default evaluates the full matched test set.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="sae_metrics.txt")
    return parser.parse_args()


class StreamingTextDataset(IterableDataset):
    """Stream one text column from a deterministic, worker-sharded file list."""

    def __init__(self, files: Iterable[str], text_col: str):
        self.files = list(files)
        self.text_col = text_col

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        for csv_path in self.files[worker_id::num_workers]:
            frame = pd.read_csv(
                csv_path,
                compression="gzip",
                usecols=[self.text_col],
            )
            sentences = frame[self.text_col].dropna().astype(str)
            yield from sentences.tolist()


def make_collate_fn(tokenizer: BertTokenizer, max_len: int):
    def collate(sentences: list[str]):
        encoded = tokenizer(
            sentences,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        mask_positions = input_ids.eq(tokenizer.mask_token_id)
        has_mask = mask_positions.any(dim=1)
        dropped = int((~has_mask).sum().item())

        if not bool(has_mask.all()):
            input_ids = input_ids[has_mask]
            attention_mask = attention_mask[has_mask]
            mask_positions = mask_positions[has_mask]

        if input_ids.size(0) == 0:
            mask_index = torch.empty(0, dtype=torch.long)
        else:
            # Use the first [MASK], matching train_sae.py.
            mask_index = mask_positions.to(torch.int64).argmax(dim=1)
        return input_ids, attention_mask, mask_index, dropped

    return collate


class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.W_e = nn.Linear(d_in, d_hidden, bias=True)
        self.W_d = nn.Linear(d_hidden, d_in, bias=False)
        self.b_d = nn.Parameter(torch.zeros(d_in))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = F.relu(self.W_e(x - self.b_d))
        reconstruction = self.W_d(features) + self.b_d
        return features, reconstruction


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    # Accept checkpoints saved from DistributedDataParallel as well.
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def load_finetuned_bert(
    model_name: Path, checkpoint_path: Path, device: torch.device
) -> BertModel:
    model = BertModel.from_pretrained(model_name)
    full_state = load_state_dict(checkpoint_path)
    bert_state = {
        key.removeprefix("bert."): value
        for key, value in full_state.items()
        if key.startswith("bert.")
    }
    if not bert_state:
        # Also support a checkpoint containing a BertModel state_dict directly.
        bert_state = full_state
    model.load_state_dict(bert_state, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_sae(
    checkpoint_path: Path, expected_d_in: int, device: torch.device
) -> SparseAutoencoder:
    state = load_state_dict(checkpoint_path)
    if "W_e.weight" not in state:
        raise KeyError(f"W_e.weight not found in SAE checkpoint: {checkpoint_path}")
    d_hidden, d_in = state["W_e.weight"].shape
    if d_in != expected_d_in:
        raise ValueError(
            f"SAE input dimension mismatch in {checkpoint_path}: "
            f"checkpoint={d_in}, BERT={expected_d_in}"
        )
    sae = SparseAutoencoder(d_in=d_in, d_hidden=d_hidden)
    sae.load_state_dict(state, strict=True)
    sae.to(device).eval()
    for parameter in sae.parameters():
        parameter.requires_grad_(False)
    return sae


@dataclass
class LayerAccumulator:
    d_in: int
    d_hidden: int

    def __post_init__(self):
        self.n = 0
        self.input_sum = torch.zeros(self.d_in, dtype=torch.float64)
        self.input_squared_sum = 0.0
        self.residual_squared_sum = 0.0
        self.feature_sum = 0.0
        self.positive_feature_sum = 0.0
        self.active_entries = 0
        self.feature_fire_counts = torch.zeros(self.d_hidden, dtype=torch.int64)

    def update(
        self,
        hidden: torch.Tensor,
        features: torch.Tensor,
        reconstruction: torch.Tensor,
        threshold: float,
    ) -> None:
        # Float64 accumulation makes the global-mean NMSE denominator stable.
        hidden_cpu = hidden.detach().to(device="cpu", dtype=torch.float64)
        residual_cpu = (hidden - reconstruction).detach().to(
            device="cpu", dtype=torch.float64
        )
        features_cpu = features.detach().to(device="cpu", dtype=torch.float64)
        active = features_cpu.gt(threshold)

        self.n += hidden_cpu.size(0)
        self.input_sum += hidden_cpu.sum(dim=0)
        self.input_squared_sum += float(hidden_cpu.square().sum().item())
        self.residual_squared_sum += float(residual_cpu.square().sum().item())
        self.feature_sum += float(features_cpu.sum().item())
        self.positive_feature_sum += float(features_cpu.masked_select(active).sum().item())
        self.active_entries += int(active.sum().item())
        self.feature_fire_counts += active.sum(dim=0, dtype=torch.int64)

    def finalize(self) -> dict[str, float | int]:
        if self.n == 0:
            raise RuntimeError("No valid [MASK] samples were evaluated.")

        # sum_i ||h_i - mean(h)||^2 = sum_i ||h_i||^2 - ||sum_i h_i||^2 / N
        centered_input_ss = self.input_squared_sum - float(
            self.input_sum.square().sum().item()
        ) / self.n
        # Protect only against tiny negative round-off, not a truly constant input.
        if centered_input_ss < 0.0 and abs(centered_input_ss) < 1e-8:
            centered_input_ss = 0.0
        nmse = (
            self.residual_squared_sum / centered_input_ss
            if centered_input_ss > 0.0
            else math.nan
        )
        total_feature_entries = self.n * self.d_hidden
        dead_features = int(self.feature_fire_counts.eq(0).sum().item())

        return {
            "n": self.n,
            "d_hidden": self.d_hidden,
            "mean_activation": self.feature_sum / total_feature_entries,
            "mean_positive_activation": (
                self.positive_feature_sum / self.active_entries
                if self.active_entries > 0
                else 0.0
            ),
            "mean_l0": self.active_entries / self.n,
            "activation_density": self.active_entries / total_feature_entries,
            "dead_features": dead_features,
            "dead_feature_ratio": dead_features / self.d_hidden,
            "reconstruction_mse": (
                self.residual_squared_sum / (self.n * self.d_in)
            ),
            "nmse": nmse,
            "explained_variance": 1.0 - nmse,
        }


def validate_args(args: argparse.Namespace) -> None:
    if args.max_len < 1 or args.batch_size < 1:
        raise ValueError("--max-len and --batch-size must be >= 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be >= 1")
    if args.activation_threshold < 0:
        raise ValueError("--activation-threshold must be >= 0")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be >= 1")
    if Path(args.output_name).name != args.output_name:
        raise ValueError("--output-name must be a file name, not a path")
    if not args.output_name.endswith(".txt"):
        raise ValueError("--output-name must end with .txt")

    required_paths = [args.model_name, args.best_model_pt, args.sae_root]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")


def format_report(
    args: argparse.Namespace,
    files: list[str],
    metrics: dict[int, dict[str, float | int]],
    dropped_no_mask: int,
) -> str:
    lines = [
        "Layer-wise SAE evaluation metrics",
        f"created_utc: {datetime.now(timezone.utc).isoformat()}",
        f"test_pattern: {args.test_pattern}",
        f"matched_files: {len(files)}",
        f"model_name: {args.model_name}",
        f"bert_checkpoint: {args.best_model_pt}",
        f"sae_root: {args.sae_root}",
        f"sae_checkpoint_name: {args.checkpoint_name}",
        f"layers: {','.join(str(layer) for layer in args.layers)}",
        f"activation_definition: f > {args.activation_threshold:g}",
        f"max_samples: {args.max_samples if args.max_samples is not None else 'none (full test set)'}",
        f"dropped_without_mask_after_tokenization: {dropped_no_mask}",
        "nmse_definition: sum(||h-hat_h||^2) / sum(||h-mean_test(h)||^2)",
        "",
        (
            "layer  n          d_sae  mean_activation  mean_positive_activation  "
            "mean_L0    density      dead       dead_ratio   recon_MSE    NMSE         1-NMSE"
        ),
        "-" * 154,
    ]

    for layer in args.layers:
        row = metrics[layer]
        lines.append(
            f"{layer:>5d}  "
            f"{int(row['n']):>10d}  "
            f"{int(row['d_hidden']):>5d}  "
            f"{float(row['mean_activation']):>15.8g}  "
            f"{float(row['mean_positive_activation']):>24.8g}  "
            f"{float(row['mean_l0']):>10.4f}  "
            f"{float(row['activation_density']):>11.6f}  "
            f"{int(row['dead_features']):>5d}  "
            f"{float(row['dead_feature_ratio']):>11.6f}  "
            f"{float(row['reconstruction_mse']):>11.6g}  "
            f"{float(row['nmse']):>11.6g}  "
            f"{float(row['explained_variance']):>11.6g}"
        )

    lines.extend(
        [
            "",
            "Definitions:",
            "  mean_activation: mean SAE code value over all samples and features.",
            "  mean_positive_activation: mean code value among active entries only.",
            "  mean_L0: mean number of active SAE features per sample.",
            "  density: fraction of all sample-feature entries that are active.",
            "  dead: features that never activate anywhere in this evaluation set.",
            "  recon_MSE: reconstruction SSE divided by N * BERT hidden dimension.",
            "  1-NMSE: reconstruction explained variance relative to the test mean.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    validate_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda was requested, but CUDA is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32

    files = sorted(glob.glob(args.test_pattern))
    if not files:
        raise FileNotFoundError(f"No test files matched: {args.test_pattern}")
    print(f"[INFO] Matched test files: {len(files)}")

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    print(f"[INFO] Loading fine-tuned BERT: {args.best_model_pt}")
    bert = load_finetuned_bert(args.model_name, args.best_model_pt, device)
    d_in = int(bert.config.hidden_size)

    saes: dict[int, SparseAutoencoder] = {}
    accumulators: dict[int, LayerAccumulator] = {}
    for layer in args.layers:
        checkpoint_path = (
            args.sae_root / f"layer_{layer:02d}" / args.checkpoint_name
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"SAE checkpoint not found: {checkpoint_path}")
        sae = load_sae(checkpoint_path, expected_d_in=d_in, device=device)
        saes[layer] = sae
        accumulators[layer] = LayerAccumulator(
            d_in=d_in,
            d_hidden=sae.W_e.out_features,
        )
        print(
            f"[INFO] Loaded layer {layer:02d} SAE "
            f"(d_in={d_in}, d_hidden={sae.W_e.out_features})"
        )

    dataset = StreamingTextDataset(files=files, text_col=args.text_col)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "collate_fn": make_collate_fn(tokenizer, args.max_len),
        "drop_last": False,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = False
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)

    amp_enabled = args.use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    evaluated = 0
    dropped_no_mask = 0

    progress = tqdm(loader, desc="Evaluating SAE metrics", unit="batch", dynamic_ncols=True)
    with torch.inference_mode():
        for input_ids, attention_mask, mask_index, dropped in progress:
            dropped_no_mask += int(dropped)
            if input_ids.size(0) == 0:
                continue

            if args.max_samples is not None:
                remaining = args.max_samples - evaluated
                if remaining <= 0:
                    break
                if input_ids.size(0) > remaining:
                    input_ids = input_ids[:remaining]
                    attention_mask = attention_mask[:remaining]
                    mask_index = mask_index[:remaining]

            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            mask_index = mask_index.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                hidden_states = bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                ).hidden_states

            batch_rows = input_ids.size(0)
            row_index = torch.arange(batch_rows, device=device)
            for layer in args.layers:
                # Evaluate SAE arithmetic and metric reductions in float32/float64,
                # even when BERT inference uses bf16/fp16 autocast.
                hidden = hidden_states[layer][row_index, mask_index].float()
                features, reconstruction = saes[layer](hidden)
                accumulators[layer].update(
                    hidden=hidden,
                    features=features,
                    reconstruction=reconstruction,
                    threshold=args.activation_threshold,
                )

            evaluated += batch_rows
            progress.set_postfix(samples=f"{evaluated:,}", refresh=False)
            del hidden_states

            if args.max_samples is not None and evaluated >= args.max_samples:
                break

    metrics = {layer: accumulators[layer].finalize() for layer in args.layers}
    report = format_report(args, files, metrics, dropped_no_mask)

    print("\n" + report, end="")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    output_path.write_text(report, encoding="utf-8")
    print(f"[OK] Metrics saved: {output_path}")


if __name__ == "__main__":
    main()
