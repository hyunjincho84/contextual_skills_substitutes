# -*- coding: utf-8 -*-
import os, json, random, glob, math
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

from model import BERTForSkillPrediction


# ─── Config ───────────────────────────────────────────────────────────────
DATA_ROOT      = "/home/jovyan/LEM_data2/hyunjincho/preprocessed_www_new"
TRAIN_DIR      = os.path.join(DATA_ROOT, "train")
VOCAB_PATH     = os.path.join(DATA_ROOT, "skill2idx.json")

MODEL_NAME     = "/home/jovyan/LEM_data2/hyunjincho/bert_pretrained/checkpoint-165687"
CKPT_DIR       = "/home/jovyan/LEM_data2/hyunjincho/checkpoints(www)_new"
BEST_MODEL_PT  = os.path.join(CKPT_DIR, "best_model.pt")

OUT_DIR        = "/home/jovyan/LEM_data2/hyunjincho/sae_layerwise_out_8192"

MAX_LEN        = 512
BATCH_SIZE     = 256
NUM_WORKERS    = 8
PIN_MEMORY     = True
SEED           = 42
DEVICE         = "cuda"

# SAE hyperparams
SAE_HIDDEN     = 4096
SAE_EPOCHS     = 3
SAE_LR         = 1e-4
L1             = 1e-3
L2             = 1e-4

USE_AMP        = True
AMP_DTYPE      = torch.bfloat16   # A100 최적

TEXT_COL       = "masked_sentence"


# ─── Global CUDA tuning ───────────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ─── Utils ────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def iter_train_files(train_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(train_dir, "[0-9]"*4, "preprocessed_*.csv.gz")))


# ─── Dataset (masked_sentence 전체 사용) ───────────────────────────────────
class MaskedSentenceDataset(Dataset):
    """
    - iterrows 제거
    - pandas vectorized loading
    - [MASK] 포함 여부로 필터링하지 않고, masked_sentence 컬럼 전체 사용
      (전제: 이 컬럼은 이미 마스킹된 문장들로 구성)
    """
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path, compression="gzip", usecols=[TEXT_COL])
        self.sentences = df[TEXT_COL].dropna().astype(str).tolist()

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


# ─── Collate: batch-tokenize ──────────────────────────────────────────────
def make_collate_fn(tokenizer: BertTokenizer):
    def collate(batch):
        # batch는 sentence(str) 리스트
        enc = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
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


def sae_loss(x, xhat, f):
    return (
        F.mse_loss(xhat, x)
        + L1 * f.abs().mean()
        + L2 * f.pow(2).mean()
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
def train_sae_one_layer(layer: int, bert, tokenizer, files: List[str]):

    layer_dir = os.path.join(OUT_DIR, f"layer_{layer:02d}")
    ensure_dir(layer_dir)

    sae = SparseAutoencoder(
        d_in=bert.config.hidden_size,
        d_hidden=SAE_HIDDEN
    ).to(DEVICE)

    opt = torch.optim.AdamW(sae.parameters(), lr=SAE_LR, fused=True)

    best_loss = float("inf")
    collate_fn = make_collate_fn(tokenizer)

    for ep in range(SAE_EPOCHS):
        sae.train()
        random.shuffle(files)
        total_loss, total_n = 0.0, 0

        print(f"\n[SAE] Layer {layer:02d} | Epoch {ep+1}/{SAE_EPOCHS}")

        for fp in tqdm(files, desc="Files", unit="file"):
            ds = MaskedSentenceDataset(fp)
            if len(ds) == 0:
                continue

            loader = DataLoader(
                ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                persistent_workers=True,
                prefetch_factor=4,
                collate_fn=collate_fn,
                drop_last=False,
            )

            for input_ids, attn_mask, mask_idx in loader:
                if input_ids.numel() == 0:
                    continue

                input_ids = input_ids.to(DEVICE, non_blocking=True)
                attn_mask = attn_mask.to(DEVICE, non_blocking=True)
                mask_idx  = mask_idx.to(DEVICE, non_blocking=True)

                with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
                    h = forward_until_layer(bert, input_ids, attn_mask, layer)
                    x = h[torch.arange(h.size(0), device=DEVICE), mask_idx]  # (B, H)
                    f, xhat = sae(x)
                    loss = sae_loss(x, xhat, f)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                bs = x.size(0)
                total_loss += float(loss.item()) * bs
                total_n += bs

        avg_loss = total_loss / max(1, total_n)
        print(f"avg_loss={avg_loss:.6f} | samples={total_n:,}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(sae.state_dict(), os.path.join(layer_dir, "sae_best.pt"))
            print(f"🏆 saved best (loss={best_loss:.6f})")

    torch.save(sae.state_dict(), os.path.join(layer_dir, "sae_last.pt"))
    print("💾 saved last")


# ─── Main ────────────────────────────────────────────────────────────────
def main():
    ensure_dir(OUT_DIR)
    set_seed(SEED)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    with open(VOCAB_PATH) as f:
        skill2idx = json.load(f)

    model = BERTForSkillPrediction(MODEL_NAME, num_skills=len(skill2idx)).to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PT, map_location="cpu"))
    model.eval()

    bert = model.bert.to(DEVICE)
    bert.eval()
    for p in bert.parameters():
        p.requires_grad_(False)

    files = iter_train_files(TRAIN_DIR)
    if not files:
        raise FileNotFoundError(f"No train files found under: {TRAIN_DIR}")

    for layer in range(1, 13):
        train_sae_one_layer(layer, bert, tokenizer, files)

    print("🎉 DONE")


if __name__ == "__main__":
    main()