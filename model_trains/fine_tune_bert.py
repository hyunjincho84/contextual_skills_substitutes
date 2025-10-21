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
# 학습 대상: preprocessed_www/train/**/preprocessed_YYYY-MM.csv.gz
DATA_ROOT      = "/home/jovyan/LEM_data2/hyunjincho/preprocessed_www"
TRAIN_DIR      = os.path.join(DATA_ROOT, "train")
VOCAB_PATH     = os.path.join(DATA_ROOT, "skill2idx.json")

# 모델/체크포인트
MODEL_NAME     = "/home/jovyan/LEM_data2/hyunjincho/bert_pretrained/checkpoint-165687"
CKPT_DIR       = "/home/jovyan/LEM_data2/hyunjincho/checkpoints(www)"
BEST_MODEL_PT  = os.path.join(CKPT_DIR, "best_model.pt")
RESUME_FROM    = None  # 예: os.path.join(CKPT_DIR, "epoch_3.pt")

# 하이퍼파라미터
MAX_LEN        = 512
BATCH_SIZE     = 64
EPOCHS         = 3
LR             = 2e-5
NUM_WORKERS    = 2
PIN_MEMORY     = True
USE_AMP        = True  # fp16 혼합정밀도 사용(가능할 때)
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
    한 개의 preprocessed_YYYY-MM.csv.gz 파일을 로드하여:
    - masked_sentence에 [MASK]가 있고
    - true_skill이 vocab에 존재하는 샘플만 사용
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

        # 첫 번째 [MASK] 위치를 라벨 토큰의 대표 위치로 사용
        mask_positions = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        if mask_positions[0].numel() == 0:
            # 드물게 토크나이저 전처리 중 [MASK]가 잘리는 경우 방어적으로 처리
            # (collate_fn에서 None 제거)
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

    # 토크나이저 & vocab
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        skill2idx = json.load(f)
    num_skills = len(skill2idx)
    print(f"✅ Vocab loaded: {num_skills} skills")

    # 모델/옵티마/스케일러
    model = BERTForSkillPrediction(MODEL_NAME, num_skills=num_skills).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    start_epoch = 0
    best_loss = float("inf")

    # 이어하기
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        print(f"🔄 Resuming from checkpoint: {RESUME_FROM}")
        ckpt = torch.load(RESUME_FROM, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_loss = float(ckpt.get("best_loss", best_loss))
        print(f"▶ Restart at epoch {start_epoch+1}, best_loss={best_loss:.4f}")

    # 학습 대상 파일 목록
    files = iter_train_files(TRAIN_DIR)
    if not files:
        raise FileNotFoundError(f"No train files found under: {TRAIN_DIR}")

    for epoch in range(start_epoch, EPOCHS):
        print(f"\n🔥 Epoch {epoch+1}/{EPOCHS}")
        total_loss, correct, total = 0.0, 0, 0

        # 파일 순서 셔플
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

        # 체크포인트
        ckpt_path = os.path.join(CKPT_DIR, f"epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "avg_loss": avg_loss,
            "best_loss": best_loss,
        }, ckpt_path)
        print(f"💾 Saved checkpoint: {ckpt_path}")

        # 베스트 모델 갱신
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), BEST_MODEL_PT)
            print(f"🏆 New best model saved → {BEST_MODEL_PT}")

    print(f"\n🎉 Training complete. Best Loss={best_loss:.4f}")

if __name__ == "__main__":
    train()