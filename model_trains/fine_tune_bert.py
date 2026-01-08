# -*- coding: utf-8 -*-
import os, json, random, glob
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

from model import BERTForSkillPrediction

# ─── Config ───────────────────────────────────────────────────────────────
# Training target: preprocessed_www/train/**/preprocessed_YYYY-MM.csv.gz
DATA_ROOT      = "/home/jovyan/LEM_data2/hyunjincho/preprocessed_www_new"
TRAIN_DIR      = os.path.join(DATA_ROOT, "train")
VOCAB_PATH     = os.path.join(DATA_ROOT, "skill2idx.json")

# model/ckpt
MODEL_NAME     = "/home/jovyan/LEM_data2/hyunjincho/bert_pretrained/checkpoint-165687"
CKPT_DIR       = "/home/jovyan/LEM_data2/hyunjincho/checkpoints(www)_new"
BEST_MODEL_PT  = os.path.join(CKPT_DIR, "best_model.pt")
RESUME_FROM    = None  # ex: os.path.join(CKPT_DIR, "epoch_3.pt")

# hyperparameter
MAX_LEN        = 512
BATCH_SIZE     = 64
EPOCHS         = 3
LR             = 2e-5
NUM_WORKERS    = 2
PIN_MEMORY     = True
USE_AMP        = True
SEED           = 42
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Utils ────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def iter_train_files(train_dir: str) -> List[str]:
    # train/{year}/preprocessed_*.csv.gz
    files = sorted(glob.glob(os.path.join(train_dir, "[0-9]"*4, "preprocessed_*.csv.gz")))
    return files

# ─── Dataset ──────────────────────────────────────────────────────────────
class MaskedSkillDataset(Dataset):
    """
    Loads a single preprocessed_YYYY-MM.csv.gz file and uses only samples that:
    - contain a [MASK] token in masked_sentence, and
    - have true_skill present in the vocabulary.
    """
    def __init__(self, csv_path: str, tokenizer: BertTokenizer, skill2idx: dict, max_len: int):
        self.tokenizer = tokenizer
        self.skill2idx = skill2idx
        self.max_len = max_len
        self.data: List[Tuple[str, int]] = []

        df = pd.read_csv(csv_path, compression="gzip").dropna(subset=["masked_sentence", "true_skill"])
        for _, row in df.iterrows():
            sent = str(row["masked_sentence"])
            if "[MASK]" not in sent:
                continue
            skill = str(row["true_skill"]).lower().strip()
            if skill in self.skill2idx:
                self.data.append((sent, self.skill2idx[skill]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, label = self.data[idx]
        enc = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

         # Use the first [MASK] position as the representative label position
        mask_positions = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        if mask_positions[0].numel() == 0:
            # Defensive handling for rare cases where [MASK] is truncated during tokenization
            # (filtered out later in collate_fn)
            return None
        mask_idx = mask_positions[0][0].item()

        return input_ids, attention_mask, mask_idx, label

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    input_ids, attention_mask, mask_idx, labels = zip(*batch)
    return (
        torch.stack(input_ids),
        torch.stack(attention_mask),
        torch.tensor(mask_idx, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long)
    )

# ─── Train ────────────────────────────────────────────────────────────────
def train():
    os.makedirs(CKPT_DIR, exist_ok=True)
    set_seed(SEED)

    # tokenizer & vocab
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        skill2idx = json.load(f)
    num_skills = len(skill2idx)
    print(f"✅ Vocab loaded: {num_skills} skills")

    # Model / optimizer / scaler
    model = BERTForSkillPrediction(MODEL_NAME, num_skills=num_skills).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    start_epoch = 0
    best_loss = float("inf")

    # Resume training if checkpoint is provided
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        print(f"🔄 Resuming from checkpoint: {RESUME_FROM}")
        ckpt = torch.load(RESUME_FROM, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_loss = float(ckpt.get("best_loss", best_loss))
        print(f"▶ Restart at epoch {start_epoch+1}, best_loss={best_loss:.4f}")

    # List of training files
    files = iter_train_files(TRAIN_DIR)
    if not files:
        raise FileNotFoundError(f"No train files found under: {TRAIN_DIR}")

    for epoch in range(start_epoch, EPOCHS):
        print(f"\n🔥 Epoch {epoch+1}/{EPOCHS}")
        total_loss, correct, total = 0.0, 0, 0

        # Shuffle file order
        random.shuffle(files)

        for file_path in tqdm(files, desc="Files", unit="file"):
            dataset = MaskedSkillDataset(file_path, tokenizer, skill2idx, MAX_LEN)
            if len(dataset) == 0:
                continue
            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                collate_fn=collate_fn,
                drop_last=False,
            )

            for batch in tqdm(loader, desc=os.path.basename(file_path), leave=False):
                if batch is None:
                    continue
                input_ids, attn_mask, mask_idx, labels = batch
                input_ids = input_ids.to(DEVICE, non_blocking=True)
                attn_mask = attn_mask.to(DEVICE, non_blocking=True)
                mask_idx  = mask_idx.to(DEVICE, non_blocking=True)
                labels    = labels.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    logits = model(input_ids, attn_mask, mask_idx)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                bs = labels.size(0)
                total_loss += loss.item() * bs
                correct += (logits.argmax(dim=-1) == labels).sum().item()
                total += bs

        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)
        print(f"✅ Epoch {epoch+1} | Loss={avg_loss:.4f} | Acc@1={acc:.4f} | Samples={total:,}")

        # Save checkpoint
        ckpt_path = os.path.join(CKPT_DIR, f"epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "avg_loss": avg_loss,
            "best_loss": best_loss,
        }, ckpt_path)
        print(f"💾 Saved checkpoint: {ckpt_path}")

        # Update best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), BEST_MODEL_PT)
            print(f"🏆 New best model saved → {BEST_MODEL_PT}")

    print(f"\n🎉 Training complete. Best Loss={best_loss:.4f}")

if __name__ == "__main__":
    train()