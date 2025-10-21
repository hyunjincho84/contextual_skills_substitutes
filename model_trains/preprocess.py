# -*- coding: utf-8 -*-
"""
Preprocess job postings (ONLY IT skills), producing train/test/findings CSVs
that include the original source row index (row_idx) for each extracted sample.

What it does
------------
- Scans monthly raw CSV(.gz) under INPUT_ROOT
- Uses a curated IT skill list (target_skills.csv) with the same normalization rule
- From each posting:
    * finds sentences containing matched IT skills
    * for each matched skill in the sentence, replaces that skill with [MASK] (one row per skill)
    * expands left/right context under MAX_TOKENS
    * emits one row per (sentence, matched skill) including:
        - row_idx (original row index in the raw CSV)
        - true_skill, masked_sentence
        - lot_v7_career_area_name, salary
        - onet_name, lot_v7_occupation_name, lot_v7_specialized_occupation_name
        - file_path (absolute path to the raw CSV file)

Outputs
-------
/home/jovyan/LEM_data2/hyunjincho/preprocessed_www/
  ├─ train/{year}/preprocessed_{yyyy-mm}.csv.gz
  ├─ test/{year}/preprocessed_{yyyy-mm}.csv.gz
  └─ findings/{year}/preprocessed_{yyyy-mm}.csv.gz
Plus:
  - sampled_files_train.csv / sampled_files_test.csv / sampled_files_findings.csv
  - preprocess_global_log.txt
  - skill2idx.json  (vocab of all IT skills that actually appeared)

Notes
-----
- Set DO_RESAMPLE_TRAIN_TEST=True to re-sample train/test from scratch.
- FINDINGS are sampled from (all files - used_files - train - test).
"""

import os, re, json, glob, random
from pathlib import Path
from typing import Dict, Tuple, List, Set

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer

# ─── Config ─────────────────────────────
INPUT_ROOT     = "/home/jovyan/LEM_data/us/csv/fortnightly/all/20250607"   # year/month root
OUTPUT_ROOT    = "/home/jovyan/LEM_data2/hyunjincho/preprocessed_www/"
GLOBAL_LOG     = os.path.join(OUTPUT_ROOT, "preprocess_global_log.txt")
VOCAB_OUTPUT   = os.path.join(OUTPUT_ROOT, "skill2idx.json")

IT_SKILLS_FILE = "./target_skills.csv"   # column 'skill' (or first column)
USED_FILES_CSV = "/home/jovyan/LEM_data2/hyunjincho/bert_pretrained/used_files.csv"

TRAIN_SAMPLE_FRAC = 0.10   # 10% of unused per month → train
TEST_SAMPLE_FRAC  = 0.10   # 10% of (unused - train) per month → test
FINDINGS_FRAC     = 0.30   # 30% of (unused - train - test - used) per month → findings
RANDOM_SEED       = 42

# If True, re-sample train/test (ignore existing sampled_files_train/test.csv)
DO_RESAMPLE_TRAIN_TEST = False

MODEL_NAME = "bert-base-uncased"
MAX_TOKENS = 512

# ─── Initialize ─────────────────────────
os.makedirs(OUTPUT_ROOT, exist_ok=True)
for d in ["train", "test", "findings"]:
    os.makedirs(os.path.join(OUTPUT_ROOT, d), exist_ok=True)

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
all_skills_set: Set[str] = set()
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─── Normalization helpers ─────────────────────────
_plang_tail = re.compile(r"\s*\(programming language\)\s*$", flags=re.IGNORECASE)

def normalize_plang(skill: str) -> str:
    s = (skill or "").strip()
    s = _plang_tail.sub("", s).strip()
    return s

def load_it_skills(path: str) -> set:
    df = pd.read_csv(path)
    col = "skill" if "skill" in df.columns else df.columns[0]
    skills = (
        df[col]
        .astype(str)
        .map(normalize_plang)
        .str.lower()
        .str.strip()
    )
    return set([s for s in skills if s])

IT_SKILL_SET = load_it_skills(IT_SKILLS_FILE)
print(f"✅ Loaded {len(IT_SKILL_SET):,} IT skills from {IT_SKILLS_FILE}")

# ─── used_files.csv ─────────────────────
def load_used_files(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    required = ["year", "month", "snapshot_dir", "file_path"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"used_files.csv missing column: '{c}'")
    df["year"] = df["year"].str.zfill(4)
    df["month"] = df["month"].str.zfill(2)
    df["yyyy_mm"] = df["year"] + "-" + df["month"]
    return df

USED_DF = load_used_files(USED_FILES_CSV)
USED_PATH_SET: Set[str] = set(USED_DF["file_path"].tolist())

# ─── enumerate monthly files ────────────
MONTH_DIR_RE = re.compile(r"all_for_(\d{4})-(\d{2})-\d{2}$")
def iter_month_files(root: str):
    for ydir in sorted(glob.glob(os.path.join(root, "20[0-9][0-9]"))):
        year = os.path.basename(ydir)
        for mdir in sorted(glob.glob(os.path.join(ydir, "all_for_*"))):
            m = MONTH_DIR_RE.search(mdir)
            if not m:
                continue
            yyyy, mm = m.group(1), m.group(2)
            yyyy_mm = f"{yyyy}-{mm}"
            for fp in sorted(glob.glob(os.path.join(mdir, "*.csv*"))):
                if fp.endswith(".csv") or fp.endswith(".csv.gz"):
                    yield (year, yyyy_mm, fp)

# ─── token budget helpers ───────────────
def add_tokens_side(context_list, sentences, budget, reverse=False):
    if reverse:
        sentences = list(reversed(sentences))
    used = 0
    for sent in sentences:
        sent = sent.replace("\n", " ")
        words = sent.split()
        if reverse:
            words = list(reversed(words))
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if used + len(word_tokens) > budget:
                return context_list, used
            if reverse:
                context_list.insert(0, word)
            else:
                context_list.append(word)
            used += len(word_tokens)
    return context_list, used

# ─── core preprocessing (ALL matched skills per sentence) ──────────
def preprocess_file(file_path: str) -> List[dict]:
    want_cols = [
        "skills_name",
        "body",
        "lot_v7_career_area_name",
        "salary",
        "onet_name",
        "lot_v7_occupation_name",
        "lot_v7_specialized_occupation_name",
    ]
    try:
        df = pd.read_csv(
            file_path,
            compression="gzip" if file_path.endswith(".gz") else None,
            low_memory=False,
            usecols=want_cols,
        )
    except ValueError:
        df = pd.read_csv(
            file_path,
            compression="gzip" if file_path.endswith(".gz") else None,
            low_memory=False,
        )
        for c in want_cols:
            if c not in df.columns:
                df[c] = np.nan
        df = df[want_cols]

    # keep only rows with skills & body
    df = df.dropna(subset=["skills_name", "body"])

    results: List[dict] = []

    for orig_idx, row in tqdm(
        df.iterrows(), total=len(df),
        desc=f"Processing {os.path.basename(file_path)}", leave=False
    ):
        # IT-skill filtering with consistent normalization
        raw_tokens   = [s.strip() for s in str(row["skills_name"]).split("|") if s.strip()]
        skills_norm  = [normalize_plang(s) for s in raw_tokens]
        skills_lower = [s.lower() for s in skills_norm]
        skills_it    = [s for s in skills_lower if s in IT_SKILL_SET]
        if not skills_it:
            continue

        body = str(row["body"]).replace("\n", " ")
        all_skills_set.update(skills_it)

        # Use text after "job description:"
        desc_start = body.lower().find("job description:")
        if desc_start == -1:
            continue
        desc_text = body[desc_start + len("job description:"):].strip()

        # sentence split (simple '.' split)
        sentences = [s.strip() for s in desc_text.split(".") if s.strip()]
        for i, sentence in enumerate(sentences):
            sentence = sentence.replace("\n", " ")

            # collect ALL matched IT skills in this sentence
            matched_skills: List[str] = []
            for skill_lower in skills_it:
                boundary_pat = re.compile(r'(?<!\w)' + re.escape(skill_lower) + r'(?!\w)', flags=re.IGNORECASE)
                if boundary_pat.search(sentence):
                    matched_skills.append(skill_lower)

            if not matched_skills:
                continue

            # create one row PER matched skill
            for matched_skill in matched_skills:
                pattern = re.compile(r'(?<!\w)' + re.escape(matched_skill) + r'(?!\w)', flags=re.IGNORECASE)
                masked_sentence = pattern.sub('[MASK]', sentence, count=1)

                # check token budget for center
                center_tokens = tokenizer.tokenize(masked_sentence)
                if len(center_tokens) >= MAX_TOKENS - 2:
                    continue

                # add left/right context around the center under MAX_TOKENS
                left_sentences  = sentences[:i]
                right_sentences = sentences[i+1:]

                budget = MAX_TOKENS - len(center_tokens) - 2
                left_budget  = budget // 2
                right_budget = budget - left_budget

                left_context,  used_left  = add_tokens_side([], left_sentences,  left_budget,  reverse=True)
                right_context, used_right = add_tokens_side([], right_sentences, right_budget, reverse=False)

                remaining = budget - (used_left + used_right)
                if remaining > 0:
                    left_context, extra_used_left = add_tokens_side(left_context, left_sentences, remaining // 2, reverse=True)
                    remaining -= extra_used_left
                if remaining > 0:
                    right_context, extra_used_right = add_tokens_side(right_context, right_sentences, remaining, reverse=False)

                full_text = " ".join(left_context + [masked_sentence] + right_context)
                if "[MASK]" not in full_text:
                    continue

                # include original row index
                results.append({
                    "row_idx": int(orig_idx),
                    "true_skill": matched_skill,
                    "masked_sentence": full_text,
                    "lot_v7_career_area_name": row.get("lot_v7_career_area_name", np.nan),
                    "salary": row.get("salary", np.nan),
                    "onet_name": row.get("onet_name", np.nan),
                    "lot_v7_occupation_name": row.get("lot_v7_occupation_name", np.nan),
                    "lot_v7_specialized_occupation_name": row.get("lot_v7_specialized_occupation_name", np.nan),
                    "file_path": file_path,
                })

    return results

# ─── sample train/test (or reuse) ───────
def load_or_sample_train_test_paths(
    monthly_files: Dict[Tuple[str, str], List[str]],
    used_set: Set[str],
    train_frac: float,
    test_frac: float,
    do_resample: bool
) -> Tuple[Set[str], Set[str]]:
    train_list_csv = os.path.join(OUTPUT_ROOT, "sampled_files_train.csv")
    test_list_csv  = os.path.join(OUTPUT_ROOT, "sampled_files_test.csv")

    sampled_train_paths: Set[str] = set()
    sampled_test_paths: Set[str]  = set()

    if (not do_resample) and os.path.exists(train_list_csv) and os.path.exists(test_list_csv):
        tr = pd.read_csv(train_list_csv)
        te = pd.read_csv(test_list_csv)
        if "file_path" in tr.columns: sampled_train_paths = set(tr["file_path"].tolist())
        if "file_path" in te.columns: sampled_test_paths  = set(te["file_path"].tolist())
        print("♻️  Reusing existing sampled_files_train.csv / sampled_files_test.csv")
        return sampled_train_paths, sampled_test_paths

    sampled_train: List[Tuple[str, str, str]] = []
    sampled_test:  List[Tuple[str, str, str]] = []
    group_counts_train, group_counts_test = {}, {}

    for (year, yyyy_mm), files in tqdm(monthly_files.items(), desc="Sampling & processing (train+test outside used)"):
        unused = [f for f in files if f not in used_set]
        if not unused:
            group_counts_train[(year, yyyy_mm)] = 0
            group_counts_test[(year, yyyy_mm)]  = 0
            continue

        k_train = max(1, int(round(len(unused) * train_frac)))
        k_train = min(k_train, len(unused))
        train_files = random.sample(unused, k=k_train)

        remaining = list(set(unused) - set(train_files))
        test_files = []
        if remaining:
            k_test = max(1, int(round(len(unused) * test_frac)))
            k_test = min(k_test, len(remaining))
            if k_test > 0:
                test_files = random.sample(remaining, k=k_test)

        # train
        train_results = []
        for fpath in train_files:
            train_results.extend(preprocess_file(fpath))
            sampled_train.append((year, yyyy_mm, fpath))
            sampled_train_paths.add(fpath)
        out_dir = os.path.join(OUTPUT_ROOT, "train", year)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(out_dir, f"preprocessed_{yyyy_mm}.csv.gz")
        if train_results:
            pd.DataFrame(train_results).to_csv(out_path, index=False, compression="gzip")
            print(f"✅ [train] {year}-{yyyy_mm} rows: {len(train_results):,} → {out_path}")
            group_counts_train[(year, yyyy_mm)] = len(train_results)
        else:
            print(f"⚠ [train] No valid rows for {year}-{yyyy_mm}")
            group_counts_train[(year, yyyy_mm)] = 0

        # test
        test_results = []
        for fpath in test_files:
            test_results.extend(preprocess_file(fpath))
            sampled_test.append((year, yyyy_mm, fpath))
            sampled_test_paths.add(fpath)
        out_dir = os.path.join(OUTPUT_ROOT, "test", year)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(out_dir, f"preprocessed_{yyyy_mm}.csv.gz")
        if test_results:
            pd.DataFrame(test_results).to_csv(out_path, index=False, compression="gzip")
            print(f"✅ [test ] {year}-{yyyy_mm} rows: {len(test_results):,} → {out_path}")
            group_counts_test[(year, yyyy_mm)] = len(test_results)
        else:
            print(f"⚠ [test ] No valid rows for {year}-{yyyy_mm}")
            group_counts_test[(year, yyyy_mm)] = 0

    if sampled_train:
        pd.DataFrame(sampled_train, columns=["year", "yyyy_mm", "file_path"]).to_csv(
            train_list_csv, index=False
        )
        print(f"🗂  [train] sampled file list saved → {train_list_csv}")
    if sampled_test:
        pd.DataFrame(sampled_test, columns=["year", "yyyy_mm", "file_path"]).to_csv(
            test_list_csv, index=False
        )
        print(f"🗂  [test ] sampled file list saved → {test_list_csv}")

    with open(GLOBAL_LOG, "a") as g:
        g.write("\n=== SPLIT: TRAIN (outside used) ===\n")
        for (year, yyyy_mm), cnt in sorted(group_counts_train.items()):
            g.write(f"[{year} | {yyyy_mm}] total_rows: {cnt}\n")
        g.write("\n=== SPLIT: TEST (outside used) ===\n")
        for (year, yyyy_mm), cnt in sorted(group_counts_test.items()):
            g.write(f"[{year} | {yyyy_mm}] total_rows: {cnt}\n")

    return sampled_train_paths, sampled_test_paths

# ─── findings from remaining files ──────
def process_findings(
    monthly_files: Dict[Tuple[str, str], List[str]],
    used_set: Set[str],
    exclude_paths: Set[str],
    findings_frac: float
):
    sampled_findings: List[Tuple[str, str, str]] = []
    group_counts = {}

    for (year, yyyy_mm), files in tqdm(monthly_files.items(), desc="Sampling & processing (findings excluding used/train/test)"):
        candidates = [f for f in files if (f not in used_set) and (f not in exclude_paths)]
        if not candidates:
            group_counts[(year, yyyy_mm)] = 0
            continue

        k_findings = max(1, int(round(len(candidates) * findings_frac)))
        k_findings = min(k_findings, len(candidates))
        findings_files = random.sample(candidates, k=k_findings)

        findings_results = []
        for fpath in findings_files:
            findings_results.extend(preprocess_file(fpath))
            sampled_findings.append((year, yyyy_mm, fpath))

        out_dir = os.path.join(OUTPUT_ROOT, "findings", year)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(out_dir, f"preprocessed_{yyyy_mm}.csv.gz")
        if findings_results:
            pd.DataFrame(findings_results).to_csv(out_path, index=False, compression="gzip")
            print(f"✅ [find] {year}-{yyyy_mm} rows: {len(findings_results):,} → {out_path}")
            group_counts[(year, yyyy_mm)] = len(findings_results)
        else:
            print(f"⚠ [find] No valid rows for {year}-{yyyy_mm}")
            group_counts[(year, yyyy_mm)] = 0

    if sampled_findings:
        out_csv = os.path.join(OUTPUT_ROOT, "sampled_files_findings.csv")
        pd.DataFrame(sampled_findings, columns=["year", "yyyy_mm", "file_path"]).to_csv(out_csv, index=False)
        print(f"🗂  [find] sampled file list saved → {out_csv}")

    with open(GLOBAL_LOG, "a") as g:
        g.write("\n=== SPLIT: FINDINGS (outside used, train, test) ===\n")
        for (year, yyyy_mm), cnt in sorted(group_counts.items()):
            g.write(f"[{year} | {yyyy_mm}] total_rows: {cnt}\n")

# ─── Main ───────────────────────────────
def main():
    with open(GLOBAL_LOG, "w") as g:
        g.write("Preprocessing Log (ONLY IT skills)\n")
        g.write("==================================\n")
        g.write(f"TRAIN_SAMPLE_FRAC={TRAIN_SAMPLE_FRAC}, TEST_SAMPLE_FRAC={TEST_SAMPLE_FRAC}, FINDINGS_FRAC={FINDINGS_FRAC}, SEED={RANDOM_SEED}\n")
        g.write(f"USED_FILES_CSV={USED_FILES_CSV}\n")

    monthly_files: Dict[Tuple[str, str], List[str]] = {}
    for year, yyyy_mm, fpath in iter_month_files(INPUT_ROOT):
        monthly_files.setdefault((year, yyyy_mm), []).append(fpath)

    # 1) train/test (reuse or resample)
    sampled_train_paths, sampled_test_paths = load_or_sample_train_test_paths(
        monthly_files, USED_PATH_SET, TRAIN_SAMPLE_FRAC, TEST_SAMPLE_FRAC, DO_RESAMPLE_TRAIN_TEST
    )
    exclude_paths = set(sampled_train_paths) | set(sampled_test_paths)

    # 2) findings on remaining
    process_findings(
        monthly_files=monthly_files,
        used_set=USED_PATH_SET,
        exclude_paths=exclude_paths,
        findings_frac=FINDINGS_FRAC
    )

    # 3) export vocab (skills actually observed)
    skill2idx = {s: i for i, s in enumerate(sorted(all_skills_set))}
    with open(VOCAB_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(skill2idx, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved skill2idx.json with {len(skill2idx)} skills → {VOCAB_OUTPUT}")

if __name__ == "__main__":
    main()