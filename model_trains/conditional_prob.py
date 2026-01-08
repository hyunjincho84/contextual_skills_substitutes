# -*- coding: utf-8 -*-
"""
Train conditional-probability skill substitution from co-occurrence:
    P(s2 | s1) = count(s1, s2) / count(s1)

Normalization (consistent with BERT-side preprocessing, but no underscore join):
  1) Remove "(programming language)" suffix
  2) Lowercase
  3) Keep only [a-z0-9 + # . / & -]
  4) Collapse multiple spaces into one
  5) Keep spaces (do not join with "_")
"""

import os, re
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations
from typing import Optional, Set, Dict
from tqdm import tqdm

# =================== Config ===================
TRAIN_INDEX_CSV   = "/home/jovyan/LEM_data2/hyunjincho/preprocessed_www_new/sampled_files_train.csv"
SKILL_COL         = "skills_name"
TARGET_SKILLS_CSV = "./target_skills.csv"
OUT_DIR           = "/home/jovyan/LEM_data2/hyunjincho/condprob_new"
CHUNKSIZE         = 200_000

LOWERCASE         = True
_ALLOWED_PATTERN  = r"[^a-z0-9\+\#\./&\-\s]"
_PLANG_TAIL_RE    = re.compile(r"\s*\(programming language\)\s*$", flags=re.IGNORECASE)

MIN_EDGE_COUNT_TO_SAVE = 1
TOPN_PER_S1_TO_SAVE    = None
SMOOTH_ALPHA           = 0.0
# ==============================================

os.makedirs(OUT_DIR, exist_ok=True)

# ============== Normalization ==============
def _normalize_plang(skill: str) -> str:
    return _PLANG_TAIL_RE.sub("", (skill or "").strip())

def normalize_skill(s: str) -> Optional[str]:
    if not isinstance(s, str):
        return None
    x = _normalize_plang(s)
    if LOWERCASE:
        x = x.lower()
    x = re.sub(_ALLOWED_PATTERN, " ", x)   # keep allowed chars
    x = re.sub(r"\s+", " ", x).strip()     # collapse spaces
    return x or None   # space 유지 (언더스코어 join 안 함)

# =================== Vocab loader ===================
def load_target_vocab(path: Optional[str]) -> Optional[Set[str]]:
    if not path or not os.path.exists(path):
        print("[INFO] Using ALL skills (no target vocab filter).")
        return None
    df = pd.read_csv(path)
    col = "NAME" if "NAME" in df.columns else df.columns[0]
    vocab = set()
    for s in df[col].dropna().astype(str):
        ns = normalize_skill(s)
        if ns:
            vocab.add(ns)
    print(f"[INFO] Loaded {len(vocab)} normalized target skills from {path}")
    return vocab

# =================== Posting iterator ===================
def iter_postings(train_index_csv: str):
    idx = pd.read_csv(train_index_csv)
    if "file_path" not in idx.columns:
        raise ValueError("sampled_files_train.csv must include 'file_path'")

    file_paths = idx["file_path"].astype(str).tolist()

    for fp in tqdm(file_paths, desc="Reading job files"):
        try:
            read_kwargs = dict(usecols=[SKILL_COL], chunksize=CHUNKSIZE)
            if fp.endswith(".gz"):
                read_kwargs["compression"] = "gzip"
            it = pd.read_csv(fp, **read_kwargs)
        except Exception as e:
            print(f"[WARN] Skip {fp}: {e}")
            continue

        for chunk in it if CHUNKSIZE else [it]:
            col = chunk[SKILL_COL].fillna("")
            for raw in col:
                if not isinstance(raw, str) or not raw:
                    continue
                toks = [normalize_skill(p) for p in raw.split("|") if p]
                toks = [t for t in dict.fromkeys(toks) if t]
                if len(toks) >= 2:
                    yield toks

# =================== Training counts with progress ===================
def train_conditional_counts(vocab: Optional[Set[str]] = None):
    count_s: Counter = Counter()
    cooc: Dict[str, Counter] = defaultdict(Counter)

    n_postings = 0
    for toks in tqdm(iter_postings(TRAIN_INDEX_CSV), desc="Building co-occurrence"):
        if vocab:
            toks = [t for t in toks if t in vocab]
        if len(toks) < 2:
            continue

        n_postings += 1
        count_s.update(toks)

        # all unordered pairs; update both directions
        for a, b in combinations(toks, 2):
            if a == b:
                continue
            cooc[a][b] += 1
            cooc[b][a] += 1

    print(f"[INFO] total postings used: {n_postings:,}")
    print(f"[INFO] unique skills kept: {len(count_s):,}")
    return count_s, cooc

# =================== Save outputs ===================
def save_outputs(count_s: Counter, cooc: Dict[str, Counter]):
    df_counts = pd.DataFrame([(s, c) for s, c in count_s.items()],
                             columns=["skill", "count"]).sort_values("count", ascending=False)
    df_counts.to_csv(os.path.join(OUT_DIR, "skill_counts.csv"), index=False)
    print(f"[OK] saved skill_counts.csv ({len(df_counts):,} skills)")

    rows = []
    for s1, neigh in tqdm(cooc.items(), desc="Saving cooc_probs"):
        total = count_s.get(s1, 0)
        if total <= 0:
            continue
        items = neigh.items()
        if TOPN_PER_S1_TO_SAVE:
            items = sorted(items, key=lambda x: x[1], reverse=True)[:TOPN_PER_S1_TO_SAVE]
        for s2, c12 in items:
            if c12 < MIN_EDGE_COUNT_TO_SAVE:
                continue
            if SMOOTH_ALPHA > 0:
                p = (c12 + SMOOTH_ALPHA) / (total + SMOOTH_ALPHA * max(1, len(neigh)))
            else:
                p = c12 / total
            rows.append((s1, s2, int(c12), float(p)))

    df_edges = pd.DataFrame(rows, columns=["s1", "s2", "count12", "p_cond"])
    df_edges.sort_values(["s1", "p_cond"], ascending=[True, False], inplace=True)
    out_path = os.path.join(OUT_DIR, "cooc_probs.csv.gz")
    df_edges.to_csv(out_path, index=False, compression="gzip")
    print(f"[OK] saved {out_path} (rows={len(df_edges):,})")

# =================== Query helper ===================
def topk_substitutes(s1: str,
                     count_s: Counter,
                     cooc: Dict[str, Counter],
                     k: int = 10,
                     min_count: int = 1):
    s1n = normalize_skill(s1)
    total = count_s.get(s1n, 0)
    if total < min_count or s1n not in cooc:
        return []
    cand = []
    for s2, c12 in cooc[s1n].items():
        if c12 < min_count:
            continue
        p = (c12 + SMOOTH_ALPHA) / (total + SMOOTH_ALPHA * max(1, len(cooc[s1n]))) if SMOOTH_ALPHA > 0 else c12 / total
        cand.append((s2, p, c12))
    cand.sort(key=lambda x: x[1], reverse=True)
    return cand[:k]

# =================== Main ===================
def main():
    vocab = load_target_vocab(TARGET_SKILLS_CSV)
    count_s, cooc = train_conditional_counts(vocab)
    save_outputs(count_s, cooc)

    demo = ["Python", "Java", "C", "UNIX", "application services"]
    for q in demo:
        res = topk_substitutes(q, count_s, cooc, k=5, min_count=5)
        if res:
            print(f"\n[Top substitutes for '{q}'] (s2, P(s2|{normalize_skill(q)}), count)")
            for s2, p, c in res:
                print(f"  {s2:30s}  {p:.4f}  (c12={c})")
        else:
            print(f"\n[No substitutes for '{q}']")

if __name__ == "__main__":
    main()