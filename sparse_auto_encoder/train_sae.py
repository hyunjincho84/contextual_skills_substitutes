# -*- coding: utf-8 -*-
import argparse
import os, sys, json, random, glob, math
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
from transformers import BertTokenizer
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "model_trains"))
sys.path.insert(0, MODEL_DIR)

from model import BERTForSkillPrediction


# ─── Utils ────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def iter_train_files(train_pattern: str) -> List[str]:
    return sorted(glob.glob(train_pattern))

def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got: {value}")

def parse_layers(value: str) -> List[int]:
    if "-" in value:
        start, end = value.split("-", maxsplit=1)
        layers = list(range(int(start), int(end) + 1))
    else:
        layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not layers or any(layer < 1 or layer > 12 for layer in layers):
        raise argparse.ArgumentTypeError("layers must be within 1..12")
    return layers

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one sparse autoencoder per selected BERT layer."
    )
    parser.add_argument("--train-pattern", required=True)
    parser.add_argument("--total-samples", required=True, type=int)
    parser.add_argument("--vocab-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--best-model-pt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--text-col", required=True)
    parser.add_argument("--layers", required=True, type=parse_layers)
    parser.add_argument("--max-len", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--num-workers", required=True, type=int)
    parser.add_argument("--pin-memory", required=True, type=parse_bool)
    parser.add_argument("--persistent-workers", required=True, type=parse_bool)
    parser.add_argument("--prefetch-factor", required=True, type=int)
    parser.add_argument("--shuffle-files", required=True, type=parse_bool)
    parser.add_argument("--shuffle-rows", required=True, type=parse_bool)
    parser.add_argument("--drop-last", required=True, type=parse_bool)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", required=True)
    parser.add_argument("--sae-hidden", required=True, type=int)
    parser.add_argument("--sae-epochs", required=True, type=int)
    parser.add_argument("--sae-lr", required=True, type=float)
    parser.add_argument("--l1", required=True, type=float)
    parser.add_argument("--l2", required=True, type=float)
    parser.add_argument("--use-amp", required=True, type=parse_bool)
    parser.add_argument("--amp-dtype", required=True, choices=["bf16", "fp16"])
    parser.add_argument("--allow-tf32", required=True, type=parse_bool)
    parser.add_argument("--fused-optimizer", required=True, type=parse_bool)
    return parser.parse_args()


# ─── Dataset (masked_sentence 전체 사용) ───────────────────────────────────
class MaskedSentenceDataset(Dataset):
    """
    - iterrows 제거
    - pandas vectorized loading
    - [MASK] 포함 여부로 필터링하지 않고, masked_sentence 컬럼 전체 사용
      (전제: 이 컬럼은 이미 마스킹된 문장들로 구성)
    """
    def __init__(self, csv_path: str, text_col: str):
        df = pd.read_csv(csv_path, compression="gzip", usecols=[text_col])
        self.sentences = df[text_col].dropna().astype(str).tolist()

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


class StreamingMaskedSentenceDataset(IterableDataset):
    """Stream many CSV files through one DataLoader instead of one loader per file."""

    def __init__(self, files, text_col, shuffle_files, shuffle_rows, seed):
        self.files = list(files)
        self.text_col = text_col
        self.shuffle_files = shuffle_files
        self.shuffle_rows = shuffle_rows
        self.seed = seed

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        rng = random.Random(self.seed + worker_id)

        files = self.files[worker_id::num_workers]
        if self.shuffle_files:
            rng.shuffle(files)

        for csv_path in files:
            frame = pd.read_csv(csv_path, compression="gzip", usecols=[self.text_col])
            sentences = frame[self.text_col].dropna().astype(str).tolist()
            if self.shuffle_rows:
                rng.shuffle(sentences)
            yield from sentences


# ─── Collate: batch-tokenize ──────────────────────────────────────────────
def make_collate_fn(tokenizer: BertTokenizer, max_len: int):
    def collate(batch):
        # batch는 sentence(str) 리스트
        enc = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]           # (B, T)
        attention_mask = enc["attention_mask"] # (B, T)

        # 첫 번째 [MASK] 위치. (masked_sentence라면 항상 존재한다고 가정)
        mask_pos = (input_ids == tokenizer.mask_token_id)
        has_mask = mask_pos.any(dim=1)

        # 방어: 혹시 마스크가 없는 샘플이 섞여 있으면 제거
        if not bool(has_mask.all()):
            keep = has_mask.nonzero(as_tuple=True)[0]
            input_ids = input_ids[keep]
            attention_mask = attention_mask[keep]
            mask_pos = mask_pos[keep]

        mask_idx = mask_pos.float().argmax(dim=1)  # (B,)
        return input_ids, attention_mask, mask_idx
    return collate


# ─── SAE ──────────────────────────────────────────────────────────────────
class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.W_e = nn.Linear(d_in, d_hidden, bias=True)
        self.W_d = nn.Linear(d_hidden, d_in, bias=False)
        self.b_d = nn.Parameter(torch.zeros(d_in))

        nn.init.kaiming_uniform_(self.W_e.weight, a=math.sqrt(5))
        nn.init.zeros_(self.W_e.bias)
        nn.init.zeros_(self.W_d.weight)

    def encode(self, x):
        return F.relu(self.W_e(x - self.b_d))

    def forward(self, x):
        f = self.encode(x)
        xhat = self.W_d(f) + self.b_d
        return f, xhat


def sae_loss(x, xhat, f, l1: float, l2: float):
    return (
        F.mse_loss(xhat, x)
        + l1 * f.abs().mean()
        + l2 * f.pow(2).mean()
    )


# ─── Efficient BERT forward (layer까지만) ─────────────────────────────────
@torch.no_grad()
def forward_until_layer(bert, input_ids, attention_mask, layer: int):
    """
    layer: 1..12
    BERT embeddings + encoder.layer[0:layer]까지만 통과.
    """
    device = input_ids.device
    x = bert.embeddings(input_ids)

    # ⚠️ device 인자 없이 호출 (FutureWarning 방지)
    ext_mask = bert.get_extended_attention_mask(
        attention_mask, attention_mask.shape
    )

    # attention mask를 현재 device로 보장
    ext_mask = ext_mask.to(device)

    for i in range(layer):
        x = bert.encoder.layer[i](x, ext_mask)[0]

    return x  # (B, T, H)


# ─── Train SAE per layer ──────────────────────────────────────────────────
def train_sae_layers(bert, tokenizer, files: List[str], args):
    """Train all requested layer SAEs from one shared full-BERT forward pass."""
    saes = {}
    optimizers = {}
    best_losses = {layer: float("inf") for layer in args.layers}
    for layer in args.layers:
        layer_dir = os.path.join(args.out_dir, f"layer_{layer:02d}")
        ensure_dir(layer_dir)
        sae = SparseAutoencoder(
            d_in=bert.config.hidden_size,
            d_hidden=args.sae_hidden,
        ).to(args.device)
        saes[layer] = sae
        optimizers[layer] = torch.optim.AdamW(
            sae.parameters(), lr=args.sae_lr, fused=args.fused_optimizer
        )

    collate_fn = make_collate_fn(tokenizer, args.max_len)

    for ep in range(args.sae_epochs):
        for sae in saes.values():
            sae.train()
        total_losses = {layer: 0.0 for layer in args.layers}
        total_n = 0

        print(
            f"\n[SAE] Layers {args.layers[0]:02d}-{args.layers[-1]:02d} | "
            f"Epoch {ep+1}/{args.sae_epochs}"
        )
        ds = StreamingMaskedSentenceDataset(
            files=files,
            text_col=args.text_col,
            shuffle_files=args.shuffle_files,
            shuffle_rows=args.shuffle_rows,
            seed=args.seed + ep * 100_000,
        )
        loader_kwargs = {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory,
            "collate_fn": collate_fn,
            "drop_last": args.drop_last,
        }
        if args.num_workers > 0:
            # The loader lives for one epoch; workers are shut down cleanly afterward.
            loader_kwargs["persistent_workers"] = False
            loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader = DataLoader(ds, **loader_kwargs)

        if args.drop_last:
            total_batches = args.total_samples // args.batch_size
        else:
            total_batches = math.ceil(args.total_samples / args.batch_size)
        progress = tqdm(
            loader,
            total=total_batches,
            desc=f"Epoch {ep+1}/{args.sae_epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for input_ids, attn_mask, mask_idx in progress:
            if input_ids.numel() == 0:
                continue

            input_ids = input_ids.to(args.device, non_blocking=True)
            attn_mask = attn_mask.to(args.device, non_blocking=True)
            mask_idx = mask_idx.to(args.device, non_blocking=True)
            amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
            device_type = torch.device(args.device).type

            with torch.no_grad(), torch.autocast(
                device_type, dtype=amp_dtype, enabled=args.use_amp
            ):
                hidden_states = bert(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    output_hidden_states=True,
                    return_dict=True,
                ).hidden_states

            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)

            losses = {}
            with torch.autocast(device_type, dtype=amp_dtype, enabled=args.use_amp):
                for layer in args.layers:
                    h = hidden_states[layer]
                    x = h[
                        torch.arange(h.size(0), device=args.device), mask_idx
                    ]
                    f, xhat = saes[layer](x)
                    losses[layer] = sae_loss(x, xhat, f, args.l1, args.l2)

            sum(losses.values()).backward()
            for opt in optimizers.values():
                opt.step()

            bs = input_ids.size(0)
            total_n += bs
            for layer, loss in losses.items():
                total_losses[layer] += float(loss.item()) * bs

            progress.set_postfix(samples=f"{total_n:,}", refresh=False)

            del hidden_states, losses

        for layer in args.layers:
            avg_loss = total_losses[layer] / max(1, total_n)
            print(f"Layer {layer:02d} | avg_loss={avg_loss:.6f} | samples={total_n:,}")
            if avg_loss < best_losses[layer]:
                best_losses[layer] = avg_loss
                layer_dir = os.path.join(args.out_dir, f"layer_{layer:02d}")
                torch.save(saes[layer].state_dict(), os.path.join(layer_dir, "sae_best.pt"))
                print(f"Layer {layer:02d} | saved best (loss={avg_loss:.6f})")

    for layer in args.layers:
        layer_dir = os.path.join(args.out_dir, f"layer_{layer:02d}")
        torch.save(saes[layer].state_dict(), os.path.join(layer_dir, "sae_last.pt"))
        print(f"Layer {layer:02d} | saved last")


# ─── Main ────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    if args.max_len < 1:
        raise ValueError("--max-len must be >= 1")
    if args.total_samples < 1:
        raise ValueError("--total-samples must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be >= 1")
    if args.sae_hidden < 1 or args.sae_epochs < 1:
        raise ValueError("--sae-hidden and --sae-epochs must be >= 1")
    if args.sae_lr <= 0:
        raise ValueError("--sae-lr must be > 0")
    if args.l1 < 0 or args.l2 < 0:
        raise ValueError("--l1 and --l2 must be >= 0")
    if args.fused_optimizer and torch.device(args.device).type != "cuda":
        raise ValueError("--fused-optimizer=true requires a CUDA device")

    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32

    ensure_dir(args.out_dir)
    set_seed(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    with open(args.vocab_path) as f:
        skill2idx = json.load(f)

    model = BERTForSkillPrediction(
        args.model_name, num_skills=len(skill2idx)
    ).to(args.device)
    model.load_state_dict(torch.load(args.best_model_pt, map_location="cpu"))
    model.eval()

    bert = model.bert.to(args.device)
    bert.eval()
    for p in bert.parameters():
        p.requires_grad_(False)

    files = iter_train_files(args.train_pattern)
    if not files:
        raise FileNotFoundError(f"No train files matched: {args.train_pattern}")

    train_sae_layers(bert, tokenizer, files, args)

    print("🎉 DONE")


if __name__ == "__main__":
    main()
