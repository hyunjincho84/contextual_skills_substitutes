# -*- coding: utf-8 -*-
"""
Evaluate Skill2Vec by cosine similarity (truth excluded) on the FULL test set (no sampling).
- Dataset-normalization keeps spaces; W2V lookup uses spaces->underscores.
- Streams each monthly CSV in the test set and evaluates all rows.
- SAVES monthly outputs to: {OUT_DIR}/pred/{YYYY}/predictions_{YYYY-MM}.csv.gz
"""

import os
import re
import gc
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from gensim.models import Word2Vec

# ========== Config ==========
MODEL_PATH = "/home/jovyan/LEM_data2/hyunjincho/skill2vec/skill2vec_norm_sg1_d300_win50_neg10_ep5.model"
TEST_ROOT  = "/home/jovyan/LEM_data2/hyunjincho/preprocessed_www/test"
OUT_DIR    = "/home/jovyan/LEM_data2/hyunjincho/skill2vec"
CHUNKSIZE  = 200_000
MAX_PRINT_PER_REASON = 10
TOPK = 5
# ============================

# ----- dataset normalization (spaces kept) -----
_plang_tail = re.compile(r"\s*\(programming language\)\s*$", flags=re.IGNORECASE)
def normalize_for_dataset(s: str) -> str:
    s = str(s or "").strip()
    s = _plang_tail.sub("", s).strip()
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ----- W2V key helpers (spaces <-> underscores) -----
def to_w2v_key(skill_with_spaces: str) -> str:
    return skill_with_spaces.replace(" ", "_")

def from_w2v_key(key: str) -> str:
    return key.replace("_", " ")

def iter_test_files(root: str):
    # preprocessed_YYYY-MM.csv or .csv.gz
    pattern1 = glob.glob(os.path.join(root, "[0-9]"*4, "preprocessed_*.csv.gz"))
    pattern2 = glob.glob(os.path.join(root, "[0-9]"*4, "preprocessed_*.csv"))
    files = sorted(list(set(pattern1 + pattern2)))
    for fp in files:
        yield fp

def extract_year_month(path: str):
    # filenames like: preprocessed_2010-01.csv or .csv.gz
    m = re.search(r"preprocessed_(20\d{2})-(\d{2})\.csv(?:\.gz)?$", os.path.basename(path))
    if not m:
        y = re.search(r"/(20\d{2})/", path)
        return (y.group(1) if y else "unknown", None)
    yyyy, mm = m.groups()
    return (yyyy, int(mm))

def month_out_path(year: str, month: int) -> str:
    year_dir = os.path.join(OUT_DIR, "pred", year)
    os.makedirs(year_dir, exist_ok=True)
    return os.path.join(year_dir, f"predictions_{year}-{month:02d}.csv.gz")

def write_preds_month(year: str, month: int, rows: list):
    """Save ONE monthly file as gzip (no append)."""
    out_path = month_out_path(year, month)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, compression="gzip")

def main():
    # load model
    model = Word2Vec.load(MODEL_PATH)
    wv = model.wv
    cand_keys = wv.index_to_key
    cand_mat  = wv.get_normed_vectors()  # L2-normalized

    total = 0
    per_year_counts = {}

    # skip stats
    skipped_not_string = 0
    skipped_not_in_vocab = 0
    skipped_no_candidates = 0
    skipped_examples = {"not_string": [], "not_in_vocab": [], "no_candidates": []}

    files = list(iter_test_files(TEST_ROOT))
    print(f"[INFO] Found {len(files)} test files")

    for fp in tqdm(files, desc="Evaluating files"):
        year, month = extract_year_month(fp)
        if month is None:
            continue

        pred_buffer = []  # collect for THIS month file; write once at the end

        read_kwargs = dict(chunksize=CHUNKSIZE)
        if fp.endswith(".gz"):
            read_kwargs["compression"] = "gzip"

        for df in pd.read_csv(fp, **read_kwargs):
            n = len(df)
            if n == 0:
                break

            # 필요 컬럼만
            has_masked = "masked_sentence" in df.columns
            cols = ["true_skill"] + (["masked_sentence"] if has_masked else [])
            sub = df[cols].dropna(subset=["true_skill"])

            if sub.empty:
                continue

            for _, row in sub.iterrows():
                truth_raw = row["true_skill"]
                masked_sentence = row["masked_sentence"] if has_masked else ""

                if not isinstance(truth_raw, str):
                    skipped_not_string += 1
                    if len(skipped_examples["not_string"]) < MAX_PRINT_PER_REASON:
                        skipped_examples["not_string"].append(str(truth_raw))
                    continue

                truth_space = normalize_for_dataset(truth_raw)
                truth_key = to_w2v_key(truth_space)

                if truth_key not in wv:
                    skipped_not_in_vocab += 1
                    if len(skipped_examples["not_in_vocab"]) < MAX_PRINT_PER_REASON:
                        skipped_examples["not_in_vocab"].append(f"{truth_space} -> {truth_key}")
                    continue

                v_truth = wv.get_vector(truth_key, norm=True)

                # exclude the truth from candidates
                try:
                    truth_idx = wv.key_to_index[truth_key]
                except KeyError:
                    skipped_not_in_vocab += 1
                    if len(skipped_examples["not_in_vocab"]) < MAX_PRINT_PER_REASON:
                        skipped_examples["not_in_vocab"].append(f"{truth_space} -> {truth_key}")
                    continue

                if len(cand_keys) <= 1:
                    skipped_no_candidates += 1
                    if len(skipped_examples["no_candidates"]) < MAX_PRINT_PER_REASON:
                        skipped_examples["no_candidates"].append(truth_space)
                    continue

                # cosine similarities against all, then mask out truth index
                sims = cand_mat @ v_truth
                sims[truth_idx] = -np.inf  # exclude truth

                k = min(TOPK, len(cand_keys) - 1)
                top_idx = np.argpartition(-sims, k-1)[:k]
                top_idx = top_idx[np.argsort(-sims[top_idx])]
                preds_keys = [cand_keys[i] for i in top_idx]
                preds_space = [from_w2v_key(k) for k in preds_keys]

                total += 1
                per_year_counts[year] = per_year_counts.get(year, 0) + 1

                pred_buffer.append({
                    "year": year,
                    "file": os.path.basename(fp),
                    "truth": truth_space,
                    "pred_top1": preds_space[0],
                    "pred_top5": "|".join(preds_space),
                    "masked_sentence": masked_sentence
                })

        # 월 파일 하나 끝났을 때 한 번만 저장 (gzip)
        if pred_buffer:
            write_preds_month(year, month, pred_buffer)
        gc.collect()

    print("\n========== W2V Prediction Summary (FULL TEST SET) ==========")
    print(f"Saved predictions (rows): {total:,}")

    print("\n========== Skipped Summary ==========")
    print(f"Not string:    {skipped_not_string}")
    print(f"Not in vocab:  {skipped_not_in_vocab}")
    print(f"No candidates: {skipped_no_candidates}")
    print(f"Total skipped: {skipped_not_string + skipped_not_in_vocab + skipped_no_candidates}")

    # Year summary
    year_summary = pd.DataFrame(
        sorted(per_year_counts.items()),
        columns=["year", "evaluated_samples"]
    )
    year_summary_path = os.path.join(OUT_DIR, "pred", "year_counts.csv")
    os.makedirs(os.path.dirname(year_summary_path), exist_ok=True)
    year_summary.to_csv(year_summary_path, index=False)
    print(f"[OK] Wrote year counts → {year_summary_path}")

if __name__ == "__main__":
    main()