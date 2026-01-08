# -*- coding: utf-8 -*-
import os, json, glob, re, sys
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "model_trains"))
sys.path.insert(0, MODEL_DIR)

from model import BERTForSkillPrediction

# ─── Config ───────────────────────────────────────────────
DATA_ROOT       = "/home/jovyan/LEM_data2/hyunjincho/preprocessed_www_new"
TEST_DIR        = os.path.join(DATA_ROOT, "test")
VOCAB_PATH      = os.path.join(DATA_ROOT, "skill2idx.json")

MODEL_NAME    = "/home/jovyan/LEM_data2/hyunjincho/bert_pretrained/checkpoint-165687"
BEST_MODEL_PT = "/home/jovyan/LEM_data2/hyunjincho/checkpoints(www)_new/best_model.pt"

MAX_LEN    = 512
BATCH_SIZE = 64
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR          = "/home/jovyan/LEM_data2/hyunjincho/bert_pred_new"
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
    if os.path.exists(out_path):
        df.to_csv(out_path, mode="a", header=False, index=False, compression="gzip")
    else:
        df.to_csv(out_path, mode="w", header=True, index=False, compression="gzip")

# ─── Dataset ─────────────────────────────────────────────
class MaskedSkillDataset(Dataset):
    def __init__(self, csv_path, skill2idx, chunksize=200_000):
        self.samples = []  # (masked_sentence, label, raw_sentence, row_idx, soc_2_name)

        req_cols = ["masked_sentence", "true_skill", "row_idx", "soc_2_name"]

        for chunk in pd.read_csv(csv_path, usecols=req_cols, chunksize=chunksize, compression="gzip"):
            if len(chunk) == 0:
                continue

            sub = chunk.copy()
            sub = sub[sub["masked_sentence"].astype(str).str.contains(r"\[MASK\]")]
            sub["true_skill"] = sub["true_skill"].astype(str).str.lower().str.strip()

            # keep only in-vocab skills
            sub = sub[sub["true_skill"].isin(skill2idx)]
            if len(sub) == 0:
                continue

            # make row_idx int-like (optional but nice)
            # (row_idx가 NaN일 수도 있으면, 아래 줄을 쓰지 말고 그냥 그대로 저장하세요)
            # sub["row_idx"] = sub["row_idx"].astype(int)

            for ms, ts, ridx, soc2 in zip(
                sub["masked_sentence"].tolist(),
                sub["true_skill"].tolist(),
                sub["row_idx"].tolist(),
                sub["soc_2_name"].tolist(),
            ):
                self.samples.append((ms, skill2idx[ts], ms, ridx, soc2))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
def make_collate_fn(tokenizer):
    def collate_fn(batch):
        if not batch:
            return None
        sentences, labels, raw_sentences, row_idxs, soc_2_names = zip(*batch)
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
        return (input_ids, attn_mask, mask_idx, labels, list(raw_sentences), list(row_idxs),list(soc_2_names))
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
    top3_correct = 0
    top5_correct = 0

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

                input_ids, attn_mask, mask_idx, labels, raw_sentences, row_idx_list, soc_2_name_list = batch
                input_ids = input_ids.to(DEVICE, non_blocking=True)
                attn_mask = attn_mask.to(DEVICE, non_blocking=True)
                mask_idx  = mask_idx.to(DEVICE, non_blocking=True)
                labels    = labels.to(DEVICE, non_blocking=True)

                logits = model(input_ids, attn_mask, mask_idx)
                probs  = F.softmax(logits, dim=-1)
                top5 = torch.topk(probs, k=5, dim=-1)
                top5_idx = top5.indices
                top5_p   = top5.values

                eq = (top5_idx == labels.unsqueeze(1))  # (B, 5)

                top1_correct += eq[:, :1].any(dim=1).sum().item()
                top3_correct += eq[:, :3].any(dim=1).sum().item()
                top5_correct += eq[:, :5].any(dim=1).sum().item()
                total += labels.size(0)


                top5_idx_cpu = top5_idx.detach().cpu()
                top5_p_cpu   = top5_p.detach().cpu()
                labels_cpu   = labels.detach().cpu()

                for i in range(labels.size(0)):
                    truth_idx = int(labels_cpu[i].item())
                    truth = idx2skill[truth_idx]

                    preds_idx_i = [int(x) for x in top5_idx_cpu[i].tolist()]
                    preds_names = [idx2skill[j] for j in preds_idx_i]

                    probs_i = [float(x) for x in top5_p_cpu[i].tolist()]

                    pred_buffer.append({
                        "year": year,
                        "file": os.path.basename(fp),

                        "row_idx": row_idx_list[i],
                        "soc_2_name": soc_2_name_list[i],

                        "truth": truth,
                        "pred_top1": preds_names[0],
                        "pred_top5": "|".join(preds_names),
                        "pred_top5_probs": "|".join([f"{p:.6f}" for p in probs_i]),

                        "masked_sentence": raw_sentences[i],
                    })

            if pred_buffer:
                write_preds_month(year, month, pred_buffer)

    acc1 = top1_correct / total if total else 0.0
    acc3 = top3_correct / total if total else 0.0
    acc5 = top5_correct / total if total else 0.0

    result_text = (
        "=== Test Evaluation ===\n"
        f"Samples: {total:,}\n"
        f"Top-1 Acc: {acc1:.4f}\n"
        f"Top-3 Acc: {acc3:.4f}\n"
        f"Top-5 Acc: {acc5:.4f}\n"
    )

    print(result_text)

    out_path = os.path.join(OUT_DIR, "test_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result_text)

    print(f"[OK] Test results saved to: {out_path}")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    evaluate()