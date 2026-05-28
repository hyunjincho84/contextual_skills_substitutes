# -*- coding: utf-8 -*-
import os, json, glob, re, sys
import argparse
import multiprocessing as mp

import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "model_trains"))
sys.path.insert(0, MODEL_DIR)

from model import BERTForSkillPrediction

# ─── Config ───────────────────────────────────────────────
BASE_DATA_DIR   = os.environ.get("BASE_DATA_DIR", "/home/jovyan/LEM_data2/data")
DATA_ROOT       = os.environ.get("DATA_ROOT", os.path.join(BASE_DATA_DIR, "preprocessed_www_new"))
VOCAB_PATH      = os.path.join(DATA_ROOT, "skill2idx.json")
INPUT_GLOB      = os.environ.get("TOP2_INPUT_GLOB", os.path.join(BASE_DATA_DIR, "bert_pred_new", "pred", "*", "sv_summary_llama_full_bert_*.csv.gz"))

MODEL_NAME    = os.environ.get("BERT_MODEL_NAME", os.path.join(BASE_DATA_DIR, "bert_pretrained"))
BEST_MODEL_PT = os.environ.get("BEST_MODEL_PT", os.path.join(BASE_DATA_DIR, "checkpoints(www)_new", "best_model.pt"))

MAX_LEN    = 512
BATCH_SIZE = int(os.environ.get("BERT_BATCH_SIZE", "64"))

OUT_DIR          = os.environ.get("TOP10_OUT_DIR", os.path.join(BASE_DATA_DIR, "bert_pred_new_top10_from_summary"))
PRED_FLUSH_EVERY = 100_000

MONTH_RE = re.compile(r"sv_summary_llama_full_bert_(20\d{2})-(\d{2})\.csv\.gz$")
INPUT_COLS = [
    "year",
    "file",
    "truth",
    "masked_sentence",
]

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

def output_path_for(year: str, month: int):
    return os.path.join(month_dir_path(year), f"sv_summary_llama_full_bert_top10_{year}-{month:02d}.csv.gz")

def write_preds_month(year: str, month: int, rows: list):
    out_path = output_path_for(year, month)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, mode="w", header=True, index=False, compression="gzip")
    return out_path

# ─── Dataset ─────────────────────────────────────────────
class MaskedSkillDataset(Dataset):
    def __init__(self, csv_path, skill2idx, chunksize=200_000):
        self.samples = []  # (masked_sentence, label, original row metadata)

        for chunk in pd.read_csv(csv_path, usecols=INPUT_COLS, chunksize=chunksize, compression="gzip"):
            if len(chunk) == 0:
                continue

            sub = chunk.copy()
            sub = sub[sub["masked_sentence"].astype(str).str.contains(r"\[MASK\]")]
            sub["truth"] = sub["truth"].astype(str).str.lower().str.strip()

            # keep only in-vocab skills
            sub = sub[sub["truth"].isin(skill2idx)]
            if len(sub) == 0:
                continue

            for row in sub.to_dict("records"):
                masked_sentence = row["masked_sentence"]
                truth = row["truth"]
                self.samples.append((masked_sentence, skill2idx[truth], row))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class BertTop10Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        if not batch:
            return None
        sentences, labels, rows = zip(*batch)
        enc = self.tokenizer(
            list(sentences), padding=True, truncation=True,
            max_length=MAX_LEN, return_tensors="pt"
        )
        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]
        mask_id = self.tokenizer.mask_token_id
        mask_positions = (input_ids == mask_id)
        mask_idx = torch.argmax(mask_positions.int(), dim=1)
        labels = torch.tensor(labels, dtype=torch.long)
        return (input_ids, attn_mask, mask_idx, labels, list(rows))

# ─── GPU helpers ─────────────────────────────────────────
def parse_gpus(gpus: str):
    gpus = str(gpus).strip().lower()
    if gpus == "cpu" or not torch.cuda.is_available():
        return ["cpu"]
    if gpus == "auto":
        return [str(i) for i in range(torch.cuda.device_count())]
    parsed = [x.strip() for x in gpus.split(",") if x.strip() != ""]
    return parsed or ["0"]

def resolve_device(gpu_id: str):
    if gpu_id == "cpu" or not torch.cuda.is_available():
        return "cpu"
    return f"cuda:{gpu_id}"

def split_round_robin(items, n: int):
    chunks = [[] for _ in range(n)]
    for i, item in enumerate(items):
        chunks[i % n].append(item)
    return chunks

def stats_zero():
    return {"total": 0, "top1": 0, "top3": 0, "top5": 0, "files": 0, "skipped": 0}

def stats_add(dst, src):
    for key in dst:
        dst[key] += src.get(key, 0)
    return dst

# ─── Evaluate ─────────────────────────────────────────────
def load_worker_state(device: str, worker_label: str):
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        skill2idx = json.load(f)
    num_skills = len(skill2idx)
    idx2skill = {int(v): k for k, v in skill2idx.items()}

    model = BERTForSkillPrediction(MODEL_NAME, num_skills=num_skills).to(device)
    state = torch.load(BEST_MODEL_PT, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[{worker_label}] model loaded on {device}")
    return tokenizer, skill2idx, idx2skill, model

def process_one_file(fp, args, tokenizer, skill2idx, idx2skill, model, device: str, worker_label: str):
    year, month = extract_year_month(fp)
    if month is None:
        return stats_zero()

    out_path = output_path_for(year, month)
    if args.resume and os.path.exists(out_path):
        print(f"[{worker_label}][SKIP][resume] already exists: {out_path}")
        out = stats_zero()
        out["skipped"] = 1
        return out

    dataset = MaskedSkillDataset(fp, skill2idx=skill2idx, chunksize=args.chunksize)
    if len(dataset) == 0:
        return stats_zero()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=BertTop10Collator(tokenizer),
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=False,
    )

    pred_buffer = []
    out = stats_zero()
    with torch.inference_mode():
        for batch in loader:
            if batch is None:
                continue

            input_ids, attn_mask, mask_idx, labels, row_meta_list = batch
            input_ids = input_ids.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)
            mask_idx  = mask_idx.to(device, non_blocking=True)
            labels    = labels.to(device, non_blocking=True)

            logits = model(input_ids, attn_mask, mask_idx)
            probs  = F.softmax(logits, dim=-1)

            top10 = torch.topk(probs, k=10, dim=-1)
            top10_idx = top10.indices
            top10_p   = top10.values

            eq = (top10_idx == labels.unsqueeze(1))
            out["top1"] += eq[:, :1].any(dim=1).sum().item()
            out["top3"] += eq[:, :3].any(dim=1).sum().item()
            out["top5"] += eq[:, :5].any(dim=1).sum().item()
            out["total"] += labels.size(0)

            top10_idx_cpu = top10_idx.detach().cpu()
            top10_p_cpu   = top10_p.detach().cpu()
            labels_cpu    = labels.detach().cpu()

            for i in range(labels.size(0)):
                truth_idx = int(labels_cpu[i].item())
                truth = idx2skill[truth_idx]

                preds_idx_i = [int(x) for x in top10_idx_cpu[i].tolist()]
                preds_names = [idx2skill[j] for j in preds_idx_i]
                probs_i = [float(x) for x in top10_p_cpu[i].tolist()]

                row_meta = row_meta_list[i]
                pred_buffer.append({
                    "year": row_meta["year"],
                    "file": row_meta["file"],
                    "truth": truth,
                    "masked_sentence": row_meta["masked_sentence"],
                    "bert_top1": preds_names[0],
                    "bert_top10": "|".join(preds_names),
                    "bert_top10_probs": "|".join([f"{p:.6f}" for p in probs_i]),
                })

    if pred_buffer:
        write_preds_month(year, month, pred_buffer)
        out["files"] = 1
        print(f"[{worker_label}][{year}-{month:02d}] saved {len(pred_buffer):,} rows -> {out_path}")
    return out

def process_files(file_items, args, gpu_id: str, worker_idx: int):
    device = resolve_device(gpu_id)
    worker_label = f"worker{worker_idx}:gpu{gpu_id}"
    if device.startswith("cuda"):
        torch.cuda.set_device(device)

    tokenizer, skill2idx, idx2skill, model = load_worker_state(device, worker_label)
    out = stats_zero()
    for fp in tqdm(file_items, desc=f"[{worker_label}] BERT top10", unit="file", position=worker_idx):
        stats_add(out, process_one_file(fp, args, tokenizer, skill2idx, idx2skill, model, device, worker_label))
    return out

def process_files_entry(queue, file_items, args, gpu_id: str, worker_idx: int):
    try:
        queue.put((worker_idx, process_files(file_items, args, gpu_id, worker_idx), None))
    except Exception as exc:
        queue.put((worker_idx, None, repr(exc)))
        raise

def evaluate(args):
    test_files = sorted(glob.glob(args.input_glob))
    if not test_files:
        raise FileNotFoundError(f"No input files matched: {args.input_glob}")

    gpus = parse_gpus(args.gpus)
    use_parallel = (not args.no_parallel and len(gpus) > 1 and torch.cuda.is_available())
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Requested GPUs: {args.gpus}")
    print(f"Using devices: {gpus}")
    print(f"Batch size per worker: {args.batch_size}")
    print(f"DataLoader workers per GPU: {args.num_workers}")
    print(f"Input files: {len(test_files)}")

    if not use_parallel:
        final = process_files(test_files, args, gpus[0], 0)
    else:
        chunks = split_round_robin(test_files, len(gpus))
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        procs = []
        for worker_idx, (gpu_id, chunk) in enumerate(zip(gpus, chunks)):
            if not chunk:
                continue
            p = ctx.Process(target=process_files_entry, args=(queue, chunk, args, gpu_id, worker_idx))
            p.start()
            procs.append(p)

        final = stats_zero()
        errors = []
        for _ in procs:
            worker_idx, worker_stats, err = queue.get()
            if err is not None:
                errors.append((worker_idx, err))
            else:
                stats_add(final, worker_stats)

        for p in procs:
            p.join()
            if p.exitcode != 0:
                errors.append((p.pid, f"exitcode={p.exitcode}"))

        if errors:
            raise RuntimeError(f"BERT prediction workers failed: {errors}")

    acc1 = final["top1"] / final["total"] if final["total"] else 0.0
    acc3 = final["top3"] / final["total"] if final["total"] else 0.0
    acc5 = final["top5"] / final["total"] if final["total"] else 0.0

    result_text = (
        "=== Test Evaluation ===\n"
        f"Samples: {final['total']:,}\n"
        f"Files saved: {final['files']:,}\n"
        f"Files skipped: {final['skipped']:,}\n"
        f"Top-1 Acc: {acc1:.4f}\n"
        f"Top-3 Acc: {acc3:.4f}\n"
        f"Top-5 Acc: {acc5:.4f}\n"
    )

    print(result_text)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "test_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result_text)

    print(f"[OK] Test results saved to: {out_path}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-glob", default=INPUT_GLOB)
    ap.add_argument("--gpus", default=os.environ.get("BERT_GPUS", os.environ.get("GPU_IDS", "auto")))
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--num-workers", type=int, default=int(os.environ.get("BERT_NUM_WORKERS", str(min(4, os.cpu_count() or 4)))))
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--no-parallel", action="store_true")
    ap.add_argument("--resume", action="store_true", default=os.environ.get("BERT_RESUME", "0") == "1")
    return ap.parse_args()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    evaluate(parse_args())
