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
from tqdm import tqdm  # 진행률 표시용

# ─── 설정 ─────────────────────────────────────────────
BASE_DIR   = "/home/jovyan/LEM_data/us/csv/fortnightly/all/20250607"  # ✅ 연/월 전체가 들어있는 루트
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "/home/jovyan/LEM_data2/hyunjincho/bert_pretrained"
MAX_LEN    = 512
MLM_PROB   = 0.15
FILES_PER_MONTH = 1
SEED = 42  # ✅ 재현성

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)

# ─── 토크나이저 초기화 ───────────────────────────────
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# ─── 스냅샷 디렉토리에서 월별로 파일 3개 샘플링 ─────────────────
SNAPSHOT_DIR_PATTERN = re.compile(r"all_for_(\d{4})-(\d{2})-01$")

def collect_monthly_files(base_dir: str, files_per_month: int, seed: int = 42) -> pd.DataFrame:
    """
    base_dir 하위의 .../<YEAR>/all_for_YYYY-MM-01/ 디렉토리들을 찾아
    각 월별로 .csv.gz 파일을 3개(부족하면 있는 만큼) 샘플링하여 DataFrame으로 반환.
    """
    rows: List[Dict] = []

    for root, dirs, files in os.walk(base_dir):
        # 월 스냅샷 디렉토리만 선별
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

        # 재현 가능한 샘플링
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

# 사용 파일 목록 저장
used_files_csv = os.path.join(OUTPUT_DIR, "used_files.csv")
used_files_df.to_csv(used_files_csv, index=False)
print(f"✅ Using {len(used_files_df)} files across months. Saved list to: {used_files_csv}")

# ─── IterableDataset 정의 ─────────────────────────────
class JobPostingIterableDataset(IterableDataset):
    def __init__(self, file_list_df: pd.DataFrame, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        # DataFrame → 파일 경로 리스트
        self.files = file_list_df["file_path"].tolist()

        # ✅ 전체 샘플 수 미리 계산(선택이지만 Trainer에 도움)
        self.total_samples = 0
        for path in tqdm(self.files, desc="Counting rows", unit="file"):
            try:
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    # body만 미리 읽어 fast count
                    df = pd.read_csv(f, usecols=["body"])
                    self.total_samples += df["body"].dropna().shape[0]
            except Exception as e:
                print(f"❗ Error counting rows in {path}: {e}")

    def __len__(self):
        return self.total_samples  # Trainer가 스텝 추정 시 사용

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

# ─── 데이터셋 및 콜레이터 ─────────────────────────────
dataset = JobPostingIterableDataset(used_files_df, tokenizer, MAX_LEN)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=MLM_PROB
)

# ─── 모델 초기화 ─────────────────────────────────────
model = BertForMaskedLM.from_pretrained(MODEL_NAME)

# ─── 학습 인자 설정 ──────────────────────────────────
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=64,  # ⚠️ MAX_LEN=512이면 GPU 메모리 크게 필요
    save_steps=1000,
    save_total_limit=2,
    logging_steps=1000,
    logging_dir=f"{OUTPUT_DIR}/logs",
    report_to=["none"],
)

# ─── Trainer 구성 및 학습 ────────────────────────────
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=data_collator,
)

print("🚀 MLM Pretraining 시작...")
trainer.train()
print(f"✅ 모델 저장 완료: {OUTPUT_DIR}")
print(f"📝 사용한 파일 목록: {used_files_csv}")