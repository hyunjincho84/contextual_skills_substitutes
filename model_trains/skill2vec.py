# -*- coding: utf-8 -*-
"""
Skill2Vec training with consistent normalization
- Both job posting skills_name and target_skills.csv are normalized
  using the same rules as BERT preprocessing:
    1) Remove "(programming language)" suffix
    2) Lowercase
    3) Keep only [a-z0-9 + # . / & -]
    4) Collapse spaces
    5) Join multi-words with "_"

Epoch-wise shuffling ENABLED.
"""

import os
import re
import time
import random
import pandas as pd
from typing import Iterator, List, Optional, Set
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

# ========================= Config =========================
TRAIN_INDEX_CSV   = "/home/jovyan/LEM_data2/hyunjincho/preprocessed_www/sampled_files_train.csv"
SKILL_COL         = "skills_name"
TARGET_SKILLS_CSV = "./target_skills.csv"  # must contain column 'NAME'
CHUNKSIZE         = 200_000

# Token normalization
LOWERCASE    = True
STRIP_SPACES = True
JOINER       = "_"

# Word2Vec hyperparams
VECTOR_SIZE  = 300
WINDOW       = 50
MIN_COUNT    = 1
SG           = 1
NEGATIVE     = 10
EPOCHS       = 5
WORKERS      = max(1, os.cpu_count() - 1)

# Output
OUT_DIR    = "/home/jovyan/LEM_data2/hyunjincho/skill2vec_unshuffle"
MODEL_NAME = f"skill2vec_norm_sg{SG}_d{VECTOR_SIZE}_win{WINDOW}_neg{NEGATIVE}_ep{EPOCHS}"
MODEL_PATH = os.path.join(OUT_DIR, f"{MODEL_NAME}.model")
KV_PATH    = os.path.join(OUT_DIR, f"{MODEL_NAME}.kv")
os.makedirs(OUT_DIR, exist_ok=True)
# ==========================================================

# ─── Normalization (same as BERT preprocessing) ─────────────────────────
_plang_tail = re.compile(r"\s*\(programming language\)\s*$", flags=re.IGNORECASE)

def normalize_plang(skill: str) -> str:
    """Remove (programming language) suffix."""
    return _plang_tail.sub("", (skill or "").strip())

def normalize_skill(s: str) -> Optional[str]:
    """Normalize skill string for consistent vocab."""
    if not isinstance(s, str):
        return None
    s = normalize_plang(s)
    if LOWERCASE:
        s = s.lower()
    s = re.sub(r"[^a-z0-9\+\#\./&\-\s]", " ", s)  # keep only allowed chars
    s = re.sub(r"\s+", " ", s).strip()
    if STRIP_SPACES and s:
        s = JOINER.join(s.split())
    return s or None

# ─── Load target vocab ─────────────────────────
def load_target_vocab(path: str) -> Set[str]:
    df = pd.read_csv(path)
    col = "NAME" if "NAME" in df.columns else df.columns[0]
    skills = set()
    for s in df[col].dropna().astype(str).tolist():
        norm = normalize_skill(s)
        if norm:
            skills.add(norm)
    print(f"[INFO] Loaded {len(skills)} normalized target skills from {path}")
    return skills

# ─── Sentence builder from job postings ─────────────────────────
def skills_row_to_sentence(raw: str, vocab: Set[str]) -> List[str]:
    if not isinstance(raw, str) or not raw:
        return []
    toks, seen = [], set()
    for p in str(raw).split("|"):
        tok = normalize_skill(p)
        if tok and tok in vocab and tok not in seen:
            seen.add(tok)
            toks.append(tok)
    return toks

def iter_skill_sentences(train_index_csv: str, vocab: Set[str]) -> Iterator[List[str]]:
    idx = pd.read_csv(train_index_csv)
    if "file_path" not in idx.columns:
        raise ValueError("sampled_files_train.csv must have a 'file_path' column.")
    file_paths = idx["file_path"].astype(str).tolist()

    for fp in tqdm(file_paths, desc="Reading job files"):
        read_kwargs = {"chunksize": CHUNKSIZE} if CHUNKSIZE else {}
        try:
            if fp.endswith(".gz"):
                df_iter = pd.read_csv(fp, compression="gzip", usecols=[SKILL_COL], **read_kwargs)
            else:
                df_iter = pd.read_csv(fp, usecols=[SKILL_COL], **read_kwargs)
        except Exception as e:
            print(f"[WARN] Skip {fp}: {e}")
            continue

        if CHUNKSIZE:
            for chunk in df_iter:
                for raw in chunk[SKILL_COL].fillna(""):
                    sent = skills_row_to_sentence(raw, vocab)
                    if len(sent) > 1:
                        yield sent
        else:
            for raw in df_iter[SKILL_COL].fillna(""):
                sent = skills_row_to_sentence(raw, vocab)
                if len(sent) > 1:
                    yield sent

# ─── Training (with epoch-wise shuffling) ─────────────────────────
def train_skill2vec(sentences, resume: bool = True) -> Word2Vec:
    cached = [s for s in sentences]
    print(f"[INFO] Number of postings (sentences): {len(cached)}")

    if resume and os.path.exists(MODEL_PATH):
        print(f"[INFO] Resuming from {MODEL_PATH}")
        model = Word2Vec.load(MODEL_PATH)
    else:
        model = Word2Vec(
            vector_size=VECTOR_SIZE,
            window=WINDOW,
            min_count=MIN_COUNT,
            sg=SG,
            negative=NEGATIVE,
            workers=WORKERS,
            compute_loss=True,
        )
        model.build_vocab(cached)
        print(f"[INFO] Vocab size: {len(model.wv)}")

    prev_loss = model.get_latest_training_loss()
    for epoch in range(EPOCHS):
        print(f"[INFO] Epoch {epoch+1}/{EPOCHS} starting...")

        # 🔀 Shuffle sentences each epoch for better SGD mixing
        # random.shuffle(cached)

        start = time.time()
        model.train(cached, total_examples=len(cached), epochs=1, report_delay=30.0)
        end = time.time()
        current_loss = model.get_latest_training_loss()
        print(f"[INFO] Epoch {epoch+1} finished in {end-start:.1f}s, "
              f"loss={current_loss - prev_loss:.2f}")
        prev_loss = current_loss

        # checkpoint save
        model.save(MODEL_PATH)
        model.wv.save(KV_PATH)
        print(f"[OK] Saved checkpoint → {MODEL_PATH}")

    return model

# ─── Export ─────────────────────────
def export_tsv(wv: KeyedVectors, out_prefix: str, topn: Optional[int] = None):
    keys = wv.index_to_key if topn is None else wv.index_to_key[:topn]
    vec_tsv  = os.path.join(OUT_DIR, f"{out_prefix}.tsv")
    meta_tsv = os.path.join(OUT_DIR, f"{out_prefix}.meta.tsv")
    with open(vec_tsv, "w", encoding="utf-8") as fv, open(meta_tsv, "w", encoding="utf-8") as fm:
        for k in keys:
            fv.write("\t".join(map(str, wv[k].tolist())) + "\n")
            fm.write(k + "\n")
    print(f"[OK] Exported TSV: {vec_tsv}, {meta_tsv}")

# ─── Main ─────────────────────────
def main():
    vocab = load_target_vocab(TARGET_SKILLS_CSV)
    sentences = iter_skill_sentences(TRAIN_INDEX_CSV, vocab)
    model = train_skill2vec(sentences, resume=True)
    export_tsv(model.wv, out_prefix=f"{MODEL_NAME}_topall")

if __name__ == "__main__":
    main()