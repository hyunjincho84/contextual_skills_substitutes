# -*- coding: utf-8 -*-
"""
Evaluate conditional-probability baseline on the FULL test set (no sampling).
- For each row's true_skill = s1, rank candidates by P(s | s1) from train co-occurrence
- Exclude s == s1 from candidates
- Save per-month predictions to {OUT_DIR}/pred/{YYYY}/predictions_{YYYY-MM}.csv.gz
- Print skip summary with examples

Normalization strictly matches the preprocessing script:
  - remove "(programming language)" suffix
  - lowercase
  - trim & collapse spaces
  - KEEP SPACES (no underscores in saved outputs)
"""

import os
import re
import gc
import glob
import pandas as pd
from tqdm import tqdm

# ===== Paths =====
CONDPROB_DIR = "/home/jovyan/LEM_data2/hyunjincho/condprob_new"
COOC_PROBS   = os.path.join(CONDPROB_DIR, "cooc_probs.csv.gz")  # expects columns: s1, s2, p_cond
TEST_ROOT    = "/home/jovyan/LEM_data2/hyunjincho/preprocessed_www_new/test"
OUT_DIR      = "/home/jovyan/LEM_data2/hyunjincho/condprob_pred_new"
CHUNKSIZE    = 200_000
TOPK         = 5
MAX_PRINT_PER_REASON = 10

# ================= Normalization (exactly as your preprocessing) =================
_plang_tail = re.compile(r"\s*\(programming language\)\s*$", flags=re.IGNORECASE)
def normalize_for_dataset(s: str) -> str:
    """Keep spaces. Remove '(programming language)'. Lowercase. Collapse spaces."""
    s = str(s or "").strip()
    s = _plang_tail.sub("", s).strip()
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ================= Utils =================
def iter_test_files(root: str):
    # preprocessed_YYYY-MM.csv(.gz)
    files = []
    files += glob.glob(os.path.join(root, "[0-9]"*4, "preprocessed_*.csv.gz"))
    files += glob.glob(os.path.join(root, "[0-9]"*4, "preprocessed_*.csv"))
    for fp in sorted(set(files)):
        yield fp

def extract_year_month(path: str):
    # filenames like: preprocessed_2010-01.csv[.gz]
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
    if len(df) == 0:
        return
    df.to_csv(out_path, index=False, compression="gzip")

# ================= Load P(s2|s1) (normalize both s1/s2 to DATASET RULE) =================
def load_conditional_table(path: str):
    """
    Returns: dict[str (space-form) -> list[(s2_space, p_cond)]], sorted desc by p_cond.
    - s1/s2 are normalized to the dataset rule (spaces kept).
    - If duplicates exist after normalization, keep the max p_cond.
    """
    print(f"[INFO] Loading co-occurrence probabilities from {path}")
    df = pd.read_csv(path)
    if not {"s1","s2","p_cond"}.issubset(df.columns):
        raise ValueError("cooc_probs.csv.gz must have columns: s1, s2, p_cond")

    # Normalize columns to match dataset rule
    df["s1_norm"] = df["s1"].astype(str).map(normalize_for_dataset)
    df["s2_norm"] = df["s2"].astype(str).map(normalize_for_dataset)

    # Aggregate duplicates by max p_cond and sort within each s1
    agg = (
        df.groupby(["s1_norm","s2_norm"], as_index=False)["p_cond"]
          .max()
          .sort_values(["s1_norm", "p_cond"], ascending=[True, False])
    )

    groups = {}
    for s1, sub in agg.groupby("s1_norm", sort=False):
        lst = list(zip(sub["s2_norm"].tolist(), sub["p_cond"].astype(float).tolist()))
        groups[s1] = lst

    print(f"[INFO] Loaded conditional table for {len(groups):,} normalized source skills (space-form)")
    return groups

# ================= Evaluation (FULL TEST SET, monthly save) =================
def main():
    # 1) load table
    cond = load_conditional_table(COOC_PROBS)

    total = 0
    per_year_counts = {}

    skipped_counts = {
        "no_truth_column": 0,
        "non_string_truth": 0,
        "s1_not_in_cond": 0,
        "no_candidates": 0,
    }
    skipped_examples = {
        "no_truth_column": [],
        "non_string_truth": [],
        "s1_not_in_cond": [],
        "no_candidates": [],
    }

    files = list(iter_test_files(TEST_ROOT))
    print(f"[INFO] Found {len(files)} test files")

    for fp in tqdm(files, desc="Evaluating files"):
        year, month = extract_year_month(fp)
        if month is None:
            continue

        pred_buffer = []

        read_kwargs = dict(chunksize=CHUNKSIZE)
        if fp.endswith(".gz"):
            read_kwargs["compression"] = "gzip"

        for df in pd.read_csv(fp, **read_kwargs):
            n = len(df)
            if n == 0:
                break

            true_col = next((c for c in ["true_skill", "target_skill", "label"] if c in df.columns), None)
            if true_col is None:
                skipped_counts["no_truth_column"] += 1
                if len(skipped_examples["no_truth_column"]) < MAX_PRINT_PER_REASON:
                    skipped_examples["no_truth_column"].append(os.path.basename(fp))
                continue

            has_masked   = "masked_sentence" in df.columns
            has_sentence = "sentence" in df.columns

            # ✅ 추가: row_idx / soc_2_name 있으면 같이 저장
            has_row_idx  = "row_idx" in df.columns
            has_soc2     = "soc_2_name" in df.columns

            keep_cols = [true_col]
            if has_masked:
                keep_cols.append("masked_sentence")
            elif has_sentence:
                keep_cols.append("sentence")

            if has_row_idx:
                keep_cols.append("row_idx")
            if has_soc2:
                keep_cols.append("soc_2_name")

            sub = df[keep_cols].dropna(subset=[true_col])
            if sub.empty:
                continue

            for _, row in sub.iterrows():
                truth_raw = row[true_col]
                if not isinstance(truth_raw, str):
                    skipped_counts["non_string_truth"] += 1
                    if len(skipped_examples["non_string_truth"]) < MAX_PRINT_PER_REASON:
                        skipped_examples["non_string_truth"].append(str(truth_raw))
                    continue

                s1 = normalize_for_dataset(truth_raw)

                if s1 not in cond:
                    skipped_counts["s1_not_in_cond"] += 1
                    if len(skipped_examples["s1_not_in_cond"]) < MAX_PRINT_PER_REASON:
                        skipped_examples["s1_not_in_cond"].append(f"{truth_raw} -> {s1}")
                    continue

                # ✅ candidates excluding self
                pairs = [(s2, p) for (s2, p) in cond[s1] if s2 != s1]
                if not pairs:
                    skipped_counts["no_candidates"] += 1
                    if len(skipped_examples["no_candidates"]) < MAX_PRINT_PER_REASON:
                        skipped_examples["no_candidates"].append(f"{truth_raw} -> {s1}")
                    continue

                top_pairs = pairs[:TOPK]
                preds  = [s for (s, _) in top_pairs]
                probs  = [float(p) for (_, p) in top_pairs]   # ✅ pred_top5_probs

                if not preds:
                    continue

                total += 1
                per_year_counts[year] = per_year_counts.get(year, 0) + 1

                masked_sentence = ""
                if has_masked:
                    masked_sentence = row.get("masked_sentence", "")
                elif has_sentence:
                    masked_sentence = row.get("sentence", "")

                # ✅ row_idx / soc_2_name (없으면 빈 값)
                row_idx_val = row.get("row_idx", "")
                soc2_val    = row.get("soc_2_name", "")

                pred_buffer.append({
                    "year": year,
                    "file": os.path.basename(fp),
                    "row_idx": row_idx_val,
                    "soc_2_name": soc2_val,
                    "truth": s1,                          # space-form
                    "pred_top1": preds[0],                # space-form
                    "pred_top5": "|".join(preds),         # space-form
                    "pred_top5_probs": "|".join([f"{p:.6f}" for p in probs]),
                    "masked_sentence": masked_sentence
                })

        # === save this month ===
        if pred_buffer:
            write_preds_month(year, month, pred_buffer)
        gc.collect()

    print("\n========== Conditional-Prob Prediction Summary (FULL TEST SET) ==========")
    print(f"Saved predictions (rows): {total:,}")

    print("\n========== Skipped Summary ==========")
    print(f"No truth column (files/chunks): {skipped_counts['no_truth_column']}")
    print(f"Non-string truth:               {skipped_counts['non_string_truth']}")
    print(f"Not in vocab (s1 not in cond):  {skipped_counts['s1_not_in_cond']}")
    print(f"No candidates:                  {skipped_counts['no_candidates']}")
    total_skipped = sum(skipped_counts.values())
    print(f"Total skipped:                  {total_skipped}")

    year_summary = pd.DataFrame(
        sorted(per_year_counts.items()),
        columns=["year", "evaluated_samples"]
    )
    year_summary_path = os.path.join(OUT_DIR, "pred", "year_counts_conditional.csv")
    os.makedirs(os.path.dirname(year_summary_path), exist_ok=True)
    year_summary.to_csv(year_summary_path, index=False)
    print(f"[OK] Wrote year counts → {year_summary_path}")

if __name__ == "__main__":
    main()