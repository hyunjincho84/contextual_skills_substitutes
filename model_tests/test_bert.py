# -*- coding: utf-8 -*-
import os, json, glob, re
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
import torch.nn.functional as F

from model import BERTForSkillPrediction

# ─── Config ───────────────────────────────────────────────
DATA_ROOT       = "/home/jovyan/LEM_data2/hyunjincho/preprocessed_www"
TEST_DIR        = os.path.join(DATA_ROOT, "test")
VOCAB_PATH      = os.path.join(DATA_ROOT, "skill2idx.json")

MODEL_NAME    = "/home/jovyan/LEM_data2/hyunjincho/bert_pretrained/checkpoint-165687"
BEST_MODEL_PT = "/home/jovyan/LEM_data2/hyunjincho/checkpoints(www)/best_model.pt"

MAX_LEN    = 512
BATCH_SIZE = 64
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR          = "/home/jovyan/LEM_data2/hyunjincho/bert_pred"
PRED_FLUSH_EVERY = 100_000

MONTH_RE = re.compile(r"preprocessed_(20\d{2})-(\d{2})\.csv\.gz$")

def extract_year_month(path: str):
    m = MONTH_RE.search(os.path.basename(path))
    if not m:
        return ("unknown", None)
    yyyy, mm = m.groups()
    return (yyyy, int(mm))

def month_dir_path(year: str):
    d = os.path.join(OUT_DIR, "pred", year)
    os.makedirs(d, exist_ok=True)
    return d

def write_preds_month(year: str, month: int, rows: list):
    out_path = os.path.join(month_dir_path(year), f"predictions_{year}-{month:02d}.csv.gz")
    df = pd.DataFrame(rows)
    # append 모드에서는 압축이 어렵기 때문에 월별로 한 번에 저장
    if os.path.exists(out_path):
        df.to_csv(out_path, mode="a", header=False, index=False, compression="gzip")
    else:
        df.to_csv(out_path, mode="w", header=True, index=False, compression="gzip")

# ─── Dataset ─────────────────────────────────────────────
class MaskedSkillDataset(Dataset):
    def __init__(self, csv_path, skill2idx, chunksize=200_000):
        self.sentences = []
        self.labels = []
        self.raw_sentences = []

        req_cols = ["masked_sentence", "true_skill"]
        for chunk in pd.read_csv(csv_path, usecols=req_cols, chunksize=chunksize, compression="gzip"):
            if len(chunk) == 0:
                continue
            sub = chunk.copy()
            sub = sub[sub["masked_sentence"].astype(str).str.contains(r"\[MASK\]")]
            sub["true_skill"] = sub["true_skill"].astype(str).str.lower().str.strip()
            sub = sub[sub["true_skill"].isin(skill2idx.keys())]
            if len(sub) == 0:
                continue
            self.sentences.extend(sub["masked_sentence"].tolist())
            self.labels.extend([skill2idx[s] for s in sub["true_skill"].tolist()])
            self.raw_sentences.extend(sub["masked_sentence"].tolist())

    def __len__(self): 
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.sentences[idx], self.labels[idx], self.raw_sentences[idx])

def make_collate_fn(tokenizer):
    def collate_fn(batch):
        if not batch:
            return None
        sentences, labels, raw_sentences = zip(*batch)
        enc = tokenizer(
            list(sentences), padding=True, truncation=True,
            max_length=MAX_LEN, return_tensors="pt"
        )
        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]
        mask_id = tokenizer.mask_token_id
        mask_positions = (input_ids == mask_id)
        mask_idx = torch.argmax(mask_positions.int(), dim=1)
        labels = torch.tensor(labels, dtype=torch.long)
        return (input_ids, attn_mask, mask_idx, labels, list(raw_sentences))
    return collate_fn

# ─── Evaluate ─────────────────────────────────────────────
def evaluate():
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        skill2idx = json.load(f)
    num_skills = len(skill2idx)
    idx2skill = {int(v): k for k, v in skill2idx.items()}

    model = BERTForSkillPrediction(MODEL_NAME, num_skills=num_skills).to(DEVICE)
    state = torch.load(BEST_MODEL_PT, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    test_files = sorted(glob.glob(os.path.join(TEST_DIR, "[0-9]"*4, "preprocessed_*.csv.gz")))
    if not test_files:
        raise FileNotFoundError(f"No test files found under {TEST_DIR}")

    collate_fn = make_collate_fn(tokenizer)
    total = 0
    top1_correct = 0

    with torch.inference_mode():
        for fp in tqdm(test_files, desc="Test files", unit="file"):
            year, month = extract_year_month(fp)
            if month is None:
                continue

            dataset = MaskedSkillDataset(fp, skill2idx=skill2idx, chunksize=200_000)
            if len(dataset) == 0:
                continue

            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=min(8, os.cpu_count() or 4),
                pin_memory=(DEVICE == "cuda"),
            )

            pred_buffer = []
            for batch in loader:
                if batch is None:
                    continue
                input_ids, attn_mask, mask_idx, labels, raw_sentences = batch
                input_ids = input_ids.to(DEVICE, non_blocking=True)
                attn_mask = attn_mask.to(DEVICE, non_blocking=True)
                mask_idx  = mask_idx.to(DEVICE, non_blocking=True)
                labels    = labels.to(DEVICE, non_blocking=True)

                logits = model(input_ids, attn_mask, mask_idx)  # (B, num_skills)
                probs  = F.softmax(logits, dim=-1)
                top5 = torch.topk(probs, k=5, dim=-1)
                top5_idx = top5.indices

                # 정확도 집계
                eq_matrix = (top5_idx == labels.unsqueeze(1))
                top1_correct += eq_matrix[:, :1].any(dim=1).sum().item()
                total += labels.size(0)

                # CPU로 변환
                top5_idx_cpu = top5_idx.detach().cpu()
                labels_cpu   = labels.detach().cpu()

                for i in range(labels.size(0)):
                    truth_idx = int(labels_cpu[i].item())
                    truth = idx2skill[truth_idx]
                    preds_idx_i = [int(x) for x in top5_idx_cpu[i].tolist()]
                    preds_names = [idx2skill[j] for j in preds_idx_i]

                    pred_buffer.append({
                        "year": year,
                        "file": os.path.basename(fp),
                        "truth": truth,
                        "pred_top1": preds_names[0],
                        "pred_top5": "|".join(preds_names),
                        "masked_sentence": raw_sentences[i],
                    })

                    if len(pred_buffer) >= PRED_FLUSH_EVERY:
                        write_preds_month(year, month, pred_buffer)
                        pred_buffer = []

            if pred_buffer:
                write_preds_month(year, month, pred_buffer)

    acc1 = top1_correct/total if total else 0.0
    print("\n=== Test Evaluation ===")
    print(f"Samples: {total:,}")
    print(f"Top-1 Acc: {acc1:.4f}")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    evaluate()