import os
import re
import gzip
import random
import pandas as pd
from pathlib import Path
from typing import List, Dict

from torch.utils.data import IterableDataset
from transformers import (
    BertTokenizer, BertForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments, Trainer
)
from tqdm import tqdm

# ─── 설정 ─────────────────────────────────────────────
BASE_DIR   = "/home/jovyan/LEM_data/us/csv/fortnightly/all/20250607"  # ✅ Root containing all year/month snapshots
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "/home/jovyan/LEM_data2/hyunjincho/bert_pretrained"
MAX_LEN    = 512
MLM_PROB   = 0.15
FILES_PER_MONTH = 1
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)

# ─── Initialize tokenizer ───────────────────────────────
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# ─── Sample files per month from snapshot directories ─────────────────
SNAPSHOT_DIR_PATTERN = re.compile(r"all_for_(\d{4})-(\d{2})-01$")

def collect_monthly_files(base_dir: str, files_per_month: int, seed: int = 42) -> pd.DataFrame:
    """
    base_dir 하위의 .../<YEAR>/all_for_YYYY-MM-01/ 디렉토리들을 찾아
    각 월별로 .csv.gz 파일을 3개(부족하면 있는 만큼) 샘플링하여 DataFrame으로 반환.
    """
    rows: List[Dict] = []

    for root, dirs, files in os.walk(base_dir):
        # Select only monthly snapshot directories
        m = SNAPSHOT_DIR_PATTERN.search(os.path.basename(root))
        if not m:
            continue

        year, month = m.group(1), m.group(2)
        csv_gz_files = sorted([
            os.path.join(root, f)
            for f in files
            if f.endswith(".csv.gz")
        ])

        if len(csv_gz_files) == 0:
            continue

        # Reproducible sampling
        local_rng = random.Random((hash(root) ^ seed) & 0xffffffff)
        if len(csv_gz_files) > files_per_month:
            sampled = local_rng.sample(csv_gz_files, files_per_month)
        else:
            sampled = csv_gz_files

        for fpath in sampled:
            rows.append({
                "year": year,
                "month": month,
                "snapshot_dir": root,
                "file_path": fpath
            })

    df = pd.DataFrame(rows).sort_values(by=["year", "month", "file_path"]).reset_index(drop=True)
    return df

used_files_df = collect_monthly_files(BASE_DIR, FILES_PER_MONTH, SEED)
if used_files_df.empty:
    raise RuntimeError(f"No CSV.GZ files found under monthly snapshots in: {BASE_DIR}")

# Save the list of used files
used_files_csv = os.path.join(OUTPUT_DIR, "used_files.csv")
used_files_df.to_csv(used_files_csv, index=False)
print(f"✅ Using {len(used_files_df)} files across months. Saved list to: {used_files_csv}")

# ─── IterableDataset definition ───────────────────────────
class JobPostingIterableDataset(IterableDataset):
    def __init__(self, file_list_df: pd.DataFrame, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        # DataFrame → list of file paths
        self.files = file_list_df["file_path"].tolist()

        
        self.total_samples = 0
        for path in tqdm(self.files, desc="Counting rows", unit="file"):
            try:
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    df = pd.read_csv(f, usecols=["body"])
                    self.total_samples += df["body"].dropna().shape[0]
            except Exception as e:
                print(f"❗ Error counting rows in {path}: {e}")

    def __len__(self):
        return self.total_samples  # Used by Trainer to estimate steps

    def parse_file(self, path):
        print(f"📄 Processing: {os.path.basename(path)}")
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                df = pd.read_csv(f)
                if "body" not in df.columns:
                    return
                bodies = df["body"].dropna().astype(str).tolist()
                for text in tqdm(bodies, desc=f"📚 {os.path.basename(path)}", unit="body", leave=False):
                    encoded = self.tokenizer(
                        text,
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_len,
                        return_tensors="pt"
                    )
                    yield {k: v.squeeze(0) for k, v in encoded.items()}
        except Exception as e:
            print(f"❗ Error reading {path}: {e}")

    def __iter__(self):
        for path in tqdm(self.files, desc="📂 File Progress", unit="file"):
            yield from self.parse_file(path)

# ─── Dataset and data collator ────────────────────────────
dataset = JobPostingIterableDataset(used_files_df, tokenizer, MAX_LEN)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=MLM_PROB
)

# ─── Initialize model ─────────────────────────────────────
model = BertForMaskedLM.from_pretrained(MODEL_NAME)

# ─── Training arguments ──────────────────────────────────
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=64,
    save_steps=1000,
    save_total_limit=2,
    logging_steps=1000,
    logging_dir=f"{OUTPUT_DIR}/logs",
    report_to=["none"],
)

# ─── Trainer setup and training ────────────────────────────
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=data_collator,
)

print("🚀 Starting MLM pretraining...")
trainer.train()
print(f"✅ Model saved to: {OUTPUT_DIR}")
print(f"📝 List of used files: {used_files_csv}")