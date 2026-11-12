# -*- coding: utf-8 -*-
"""
Preprocess all job postings (ONLY target IT skills), without splitting or sampling.
Each output row includes the original source row index (row_idx).

What it does
------------
- Scans every monthly raw CSV(.gz) under INPUT_ROOT
- Before masking, globally keeps one posting per SHA-256 fingerprint of the
  NFKC/casefold/whitespace-normalized original body. Only rows containing
  "Job Description:" and at least one target skill are eligible.
- Resumes by default, skipping completed source-file outputs
- Uses a curated IT skill list (target_skills.csv) with the same normalization rule
- From each posting:
    * finds sentences containing matched IT skills
    * for each matched skill in the sentence, replaces that skill with [MASK] (one row per skill)
    * expands left/right context under MAX_TOKENS
    * emits one row per (sentence, matched skill) including:
        - row_idx (original row index in the raw CSV)
        - posting_fingerprint (normalized original-body SHA-256)
        - true_skill, masked_sentence
        - lot_v7_career_area_name, salary
        - onet_name, lot_v7_occupation_name, lot_v7_specialized_occupation_name
        - file_path (absolute path to the raw CSV file)

Outputs
-------
${BASE_DATA_DIR}/preprocessed_www_new/ (default: /home/jovyan/LEM_data2/data/preprocessed_www_new/)
  └─ all/{year}/{yyyy-mm}/preprocessed_{source-file}.csv.gz
Plus:
  - processed_files_all.csv
  - preprocessing_stats_by_year.csv
  - temporary per-output *.resume.json metadata (removed after each successful year)
  - preprocess_global_log.txt
  - skill2idx.json  (vocab of all IT skills that actually appeared)

Debug mode
----------
	•	Uses only one input CSV (or .csv.gz) file specified by --debug_file (required).
	•	Samples at most --debug_n rows from the file (default: 10).
	•	Writes the complete preprocessing results for the sampled rows to
OUTPUT_ROOT/debug_sample.csv.gz.
"""

import os, re, json, glob, argparse, hashlib, unicodedata
import sqlite3, tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple, List, Set, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer

# ─── Config ─────────────────────────────
BASE_DATA_DIR  = os.environ.get("BASE_DATA_DIR", "/home/jovyan/LEM_data2/data")
INPUT_ROOT     = os.environ.get("RAW_INPUT_ROOT", "/home/jovyan/LEM_data/us/csv/fortnightly/all/20250607")   # year/month root
OUTPUT_ROOT    = os.environ.get("PREPROCESSED_ROOT", os.path.join(BASE_DATA_DIR, "preprocessed_www_new"))
GLOBAL_LOG     = os.path.join(OUTPUT_ROOT, "preprocess_global_log.txt")
VOCAB_OUTPUT   = os.path.join(OUTPUT_ROOT, "skill2idx.json")
STATS_OUTPUT   = os.path.join(OUTPUT_ROOT, "preprocessing_stats_by_year.csv")
DEFAULT_BODY_DEDUP_SEED = 42

REPO_ROOT = Path(__file__).resolve().parent.parent
IT_SKILLS_FILE = os.environ.get(
    "TARGET_SKILLS_CSV",
    str(REPO_ROOT / "target_skills.csv"),
)

MODEL_NAME = "bert-base-uncased"
MAX_TOKENS = 256
DEFAULT_WORKERS = int(os.environ.get("PREPROCESS_WORKERS", "8"))

INPUT_COLUMNS = [
    "skills_name",
    "body",
    "lot_v7_career_area_name",
    "salary",
    "onet_name",
    "lot_v7_occupation_name",
    "lot_v7_specialized_occupation_name",
    "soc_2_name",
    "state_name",
    "education_levels_name",
]
REQUIRED_INPUT_COLUMNS = {"skills_name", "body"}
OUTPUT_COLUMNS = [
    "row_idx",
    "posting_fingerprint",
    "true_skill",
    "masked_sentence",
    "lot_v7_career_area_name",
    "salary",
    "onet_name",
    "lot_v7_occupation_name",
    "lot_v7_specialized_occupation_name",
    "soc_2_name",
    "state_name",
    "education_levels_name",
    "file_path",
]

# ─── Initialize ─────────────────────────
ALL_OUTPUT_ROOT = os.path.join(OUTPUT_ROOT, "all")
tokenizer = None  # loaded in main for debug, or once per worker
selection_connection = None


def configure_output_paths(output_path: str):
    global OUTPUT_ROOT, ALL_OUTPUT_ROOT, GLOBAL_LOG, VOCAB_OUTPUT, STATS_OUTPUT

    OUTPUT_ROOT = os.path.abspath(os.path.expanduser(output_path))
    ALL_OUTPUT_ROOT = os.path.join(OUTPUT_ROOT, "all")
    GLOBAL_LOG = os.path.join(OUTPUT_ROOT, "preprocess_global_log.txt")
    VOCAB_OUTPUT = os.path.join(OUTPUT_ROOT, "skill2idx.json")
    STATS_OUTPUT = os.path.join(
        OUTPUT_ROOT,
        "preprocessing_stats_by_year.csv",
    )

    os.makedirs(ALL_OUTPUT_ROOT, exist_ok=True)

# ─── Normalization helpers ─────────────────────────
_plang_tail = re.compile(r"\s*\(programming language\)\s*$", flags=re.IGNORECASE)
_body_whitespace = re.compile(r"\s+")


def normalize_body(body: object) -> str:
    """Return the canonical body text used only for duplicate detection."""
    normalized = unicodedata.normalize("NFKC", str(body)).casefold()
    return _body_whitespace.sub(" ", normalized).strip()


def body_fingerprint(body: object) -> str:
    """Return a stable SHA-256 fingerprint of the normalized original body."""
    return hashlib.sha256(normalize_body(body).encode("utf-8")).hexdigest()


def body_representative_priority(seed: int, file_path: str, row_idx: int) -> bytes:
    """Return a deterministic pseudo-random priority for one raw posting."""
    digest = hashlib.blake2b(digest_size=16, person=b"LEMbodypick-v1")
    digest.update(str(seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(file_path.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(row_idx).encode("ascii"))
    return digest.digest()


def scan_body_file(args: Tuple[str, int]) -> dict:
    """Scan one raw file and retain one local representative per body hash."""
    file_path, seed = args
    compression = "gzip" if file_path.endswith(".gz") else None
    records: Dict[str, Tuple[int, bytes, int]] = {}
    input_rows = 0
    eligible_postings = 0
    valid_job_description_postings = 0

    chunks = pd.read_csv(
        file_path,
        compression=compression,
        low_memory=False,
        usecols=["skills_name", "body"],
        chunksize=100_000,
    )
    for chunk in chunks:
        input_rows += len(chunk)
        valid = chunk.dropna(subset=["skills_name", "body"])
        for row_idx, skills_value, body in valid[
            ["skills_name", "body"]
        ].itertuples(index=True, name=None):
            original_body = str(body)
            normalized_body = normalize_body(original_body)
            if "job description:" not in normalized_body:
                continue

            raw_skills = [
                normalize_plang(skill).lower()
                for skill in str(skills_value).split("|")
                if skill.strip()
            ]
            if not raw_skills:
                continue
            valid_job_description_postings += 1

            target_skill_count = len({
                skill for skill in raw_skills if skill in IT_SKILL_SET
            })
            if target_skill_count == 0:
                continue

            eligible_postings += 1
            fingerprint = hashlib.sha256(
                normalized_body.encode("utf-8")
            ).hexdigest()
            priority = body_representative_priority(
                seed,
                file_path,
                int(row_idx),
            )
            candidate = (target_skill_count, priority, int(row_idx))
            previous = records.get(fingerprint)
            if (
                previous is None
                or target_skill_count > previous[0]
                or (
                    target_skill_count == previous[0]
                    and priority < previous[1]
                )
            ):
                records[fingerprint] = candidate

    return {
        "file_path": file_path,
        "input_rows": input_rows,
        "eligible_postings": eligible_postings,
        "valid_job_description_postings": (
            valid_job_description_postings
        ),
        "records": [
            (fingerprint, target_skill_count, priority, row_idx)
            for fingerprint, (
                target_skill_count,
                priority,
                row_idx,
            ) in records.items()
        ],
    }


def initialize_body_index(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    connection.execute("PRAGMA temp_store = FILE")
    connection.execute(
        """
        CREATE TABLE posting_choice (
            posting_fingerprint TEXT PRIMARY KEY,
            target_skill_count INTEGER NOT NULL,
            priority BLOB NOT NULL,
            file_path TEXT NOT NULL,
            row_idx INTEGER NOT NULL
        ) WITHOUT ROWID
        """
    )
    return connection


def build_body_dedup_index(
    db_path: Path,
    monthly_files: Dict[Tuple[str, str], List[str]],
    workers: int,
    seed: int,
) -> dict:
    """Choose one raw posting globally for every normalized-body fingerprint."""
    file_paths = sorted(
        file_path
        for files in monthly_files.values()
        for file_path in files
    )
    stats = {
        "input_files": len(file_paths),
        "input_rows": 0,
        "eligible_job_description_postings": 0,
        "valid_job_description_postings": 0,
        "unique_body_postings": 0,
        "duplicate_body_postings_dropped": 0,
        "representative_replacements": 0,
        "representative_policy": (
            "maximum_target_skill_count_then_minimum_seeded_priority"
        ),
        "normalization": "NFKC + casefold + collapse_whitespace + strip",
        "hash": "sha256",
        "seed": seed,
    }
    connection = initialize_body_index(db_path)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(scan_body_file, (file_path, seed)): file_path
            for file_path in file_paths
        }
        try:
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Indexing unique raw bodies",
                unit="file",
            ):
                result = future.result()
                stats["input_rows"] += int(result["input_rows"])
                stats["eligible_job_description_postings"] += int(
                    result["eligible_postings"]
                )
                stats["valid_job_description_postings"] += int(
                    result["valid_job_description_postings"]
                )
                for (
                    fingerprint,
                    target_skill_count,
                    priority,
                    row_idx,
                ) in result["records"]:
                    cursor = connection.execute(
                        """
                        INSERT OR IGNORE INTO posting_choice(
                            posting_fingerprint, target_skill_count,
                            priority, file_path, row_idx
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            fingerprint,
                            target_skill_count,
                            priority,
                            result["file_path"],
                            row_idx,
                        ),
                    )
                    if cursor.rowcount == 1:
                        stats["unique_body_postings"] += 1
                        continue

                    previous = connection.execute(
                        """
                        SELECT target_skill_count, priority
                        FROM posting_choice
                        WHERE posting_fingerprint = ?
                        """,
                        (fingerprint,),
                    ).fetchone()
                    if previous is None:
                        raise RuntimeError("body fingerprint has no representative")
                    if (
                        target_skill_count > int(previous[0])
                        or (
                            target_skill_count == int(previous[0])
                            and priority < previous[1]
                        )
                    ):
                        connection.execute(
                            """
                            UPDATE posting_choice
                            SET target_skill_count = ?, priority = ?,
                                file_path = ?, row_idx = ?
                            WHERE posting_fingerprint = ?
                            """,
                            (
                                target_skill_count,
                                priority,
                                result["file_path"],
                                row_idx,
                                fingerprint,
                            ),
                        )
                        stats["representative_replacements"] += 1
                connection.commit()
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    stats["duplicate_body_postings_dropped"] = (
        stats["eligible_job_description_postings"]
        - stats["unique_body_postings"]
    )
    connection.execute(
        "CREATE INDEX selected_postings_by_file "
        "ON posting_choice(file_path, row_idx)"
    )
    connection.commit()
    connection.close()
    return stats


def preprocessing_input_signature(
    monthly_files: Dict[Tuple[str, str], List[str]],
    seed: int,
) -> str:
    """Hash the complete raw input inventory and body-dedup configuration."""
    digest = hashlib.sha256()
    digest.update(b"LEM-preprocess-body-dedup-v1\0")
    digest.update(Path(__file__).read_bytes())
    digest.update(b"\0")
    target_signature = file_signature(IT_SKILLS_FILE)
    digest.update(str(Path(IT_SKILLS_FILE).resolve()).encode("utf-8"))
    digest.update(str(target_signature["size"]).encode("ascii"))
    digest.update(str(target_signature["mtime_ns"]).encode("ascii"))
    digest.update(str(seed).encode("ascii"))
    for file_path in sorted(
        file_path
        for files in monthly_files.values()
        for file_path in files
    ):
        signature = file_signature(file_path)
        digest.update(file_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(signature["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(signature["mtime_ns"]).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()

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

# ─── tokenizer loader ───────────────────
def load_tokenizer(model_name_or_path: str) -> BertTokenizer:
    tok = BertTokenizer.from_pretrained(model_name_or_path)
    print(f"✅ Loaded tokenizer from: {model_name_or_path}")
    return tok


def init_worker(model_name_or_path: str, selection_db_path: str):
    # Each process owns its tokenizer; tokenizers are not shared between workers.
    global tokenizer, selection_connection
    tokenizer = load_tokenizer(model_name_or_path)
    db_uri = f"file:{Path(selection_db_path).as_posix()}?mode=ro&immutable=1"
    selection_connection = sqlite3.connect(db_uri, uri=True)


def selected_row_indices(file_path: str) -> Set[int]:
    if selection_connection is None:
        raise RuntimeError("body deduplication index is not initialized")
    return {
        int(row[0]) for row in selection_connection.execute(
            "SELECT row_idx FROM posting_choice WHERE file_path = ?", (file_path,)
        )

    }
# ─── token budget helpers ───────────────
def add_tokens_side(context_list, sentences, budget, reverse=False):
    """
    Fills the left and right context of a sentence within a given token budget.
    Since calling this function multiple times on the same sentence may introduce duplicate tokens,
    it should be invoked at most once per side (left/right) for each sentence.
    """
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

# ─── Common preprocessing logic: DataFrame subset ─────
def preprocess_rows(
    df_subset: pd.DataFrame,
    file_path: str,
    show_progress: bool = True,
) -> Tuple[List[dict], Set[str]]:
    """
    Preprocess a DataFrame and return output rows plus observed target skills.
    """
    results: List[dict] = []
    observed_skills: Set[str] = set()

    rows = df_subset.iterrows()
    if show_progress:
        rows = tqdm(
            rows,
            total=len(df_subset),
            desc=f"Processing {os.path.basename(file_path)} subset",
            leave=False,
        )

    for orig_idx, row in rows:
        # IT-skill filtering with consistent normalization
        raw_tokens   = [s.strip() for s in str(row["skills_name"]).split("|") if s.strip()]
        skills_norm  = [normalize_plang(s) for s in raw_tokens]
        skills_lower = [s.lower() for s in skills_norm]
        skills_it    = [s for s in skills_lower if s in IT_SKILL_SET]
        if not skills_it:
            continue

        original_body = str(row["body"])
        posting_fingerprint = body_fingerprint(original_body)
        body = original_body.replace("\n", " ")
        observed_skills.update(skills_it)

        # Use text after "job description:"
        desc_start = body.lower().find("job description:")
        if desc_start == -1:
            continue
        desc_text = body[desc_start + len("job description:"):].strip()

        # sentence split (simple '.' split)
        sentences = [s.strip() for s in desc_text.split(".") if s.strip()]
        if not sentences:
            continue

        for i, sentence in enumerate(sentences):
            sentence = sentence.replace("\n", " ")

            # collect ALL matched IT skills in this sentence
            matched_skills: List[str] = []
            for skill_lower in skills_it:
                boundary_pat = re.compile(
                    r'(?<!\w)' + re.escape(skill_lower) + r'(?!\w)',
                    flags=re.IGNORECASE
                )
                if boundary_pat.search(sentence):
                    matched_skills.append(skill_lower)

            if not matched_skills:
                continue

            # create one row PER matched skill
            for matched_skill in matched_skills:
                pattern = re.compile(
                    r'(?<!\w)' + re.escape(matched_skill) + r'(?!\w)',
                    flags=re.IGNORECASE
                )
                masked_sentence = pattern.sub('[MASK]', sentence, count=1)

                # check token budget for center
                center_tokens = tokenizer.tokenize(masked_sentence)
                if len(center_tokens) >= MAX_TOKENS - 2:
                    continue

                # add left/right context around the center under MAX_TOKENS
                left_sentences  = sentences[:i]
                right_sentences = sentences[i+1:]

                budget = MAX_TOKENS - len(center_tokens) - 2
                if budget <= 0:
                    full_text = masked_sentence
                else:
                    left_budget  = budget // 2
                    right_budget = budget - left_budget

                    left_context,  _ = add_tokens_side([], left_sentences,  left_budget,  reverse=True)
                    right_context, _ = add_tokens_side([], right_sentences, right_budget, reverse=False)

                    full_text = " ".join(left_context + [masked_sentence] + right_context)

                if "[MASK]" not in full_text:
                    continue

                results.append({
                    "row_idx": int(orig_idx),
                    "posting_fingerprint": posting_fingerprint,
                    "true_skill": matched_skill,
                    "masked_sentence": full_text,
                    "lot_v7_career_area_name": row.get("lot_v7_career_area_name", np.nan),
                    "salary": row.get("salary", np.nan),
                    "onet_name": row.get("onet_name", np.nan),
                    "lot_v7_occupation_name": row.get("lot_v7_occupation_name", np.nan),
                    "lot_v7_specialized_occupation_name": row.get("lot_v7_specialized_occupation_name", np.nan),
                    "soc_2_name": row.get("soc_2_name", np.nan),
                    "state_name": row.get("state_name", np.nan),
                    "education_levels_name": row.get("education_levels_name", np.nan),
                    "file_path": file_path,
                })

    return results, observed_skills

# ─── Process one complete source file ─────
def preprocess_file(file_path: str) -> Tuple[List[dict], Set[str]]:
    compression = "gzip" if file_path.endswith(".gz") else None
    available_columns = set(
        pd.read_csv(file_path, compression=compression, nrows=0).columns
    )
    missing_required = REQUIRED_INPUT_COLUMNS - available_columns
    if missing_required:
        missing_text = ", ".join(sorted(missing_required))
        raise ValueError(f"{file_path} is missing required columns: {missing_text}")

    present_columns = [c for c in INPUT_COLUMNS if c in available_columns]
    df = pd.read_csv(
        file_path,
        compression=compression,
        low_memory=False,
        usecols=present_columns,
    )
    for column in INPUT_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    df = df[INPUT_COLUMNS]

    selected_indices = selected_row_indices(file_path)
    if not selected_indices:
        return [], set()
    df = df.loc[df.index.isin(selected_indices)]
    df = df.dropna(subset=["skills_name", "body"])
    if df.empty:
        return [], set()

    return preprocess_rows(df, file_path, show_progress=False)


# ─── Write one source file per worker ─────
def source_output_path(year: str, yyyy_mm: str, file_path: str) -> str:
    source_name = Path(file_path).name
    if source_name.endswith(".gz"):
        source_name = source_name[:-3]
    if source_name.endswith(".csv"):
        source_name = source_name[:-4]
    snapshot_name = Path(file_path).parent.name
    out_dir = os.path.join(ALL_OUTPUT_ROOT, year, yyyy_mm)
    return os.path.join(
        out_dir,
        f"preprocessed_{snapshot_name}__{source_name}.csv.gz",
    )


def file_signature(path: str) -> Dict[str, int]:
    stat = os.stat(path)
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def resume_metadata_path(output_path: str) -> str:
    return f"{output_path}.resume.json"


def atomic_write_json(path: str, payload: dict):
    temp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def write_resume_metadata(
    file_path: str,
    output_path: str,
    postings_with_masked_sentence: int,
    masked_sentences: int,
    observed_skills: Set[str],
    input_signature: str,
):
    metadata = {
        "version": 3,
        "source": file_signature(file_path),
        "output": file_signature(output_path),
        "input_signature": input_signature,
        "postings_with_masked_sentence": int(postings_with_masked_sentence),
        "masked_sentences": int(masked_sentences),
        "observed_skills": sorted(observed_skills),
    }
    atomic_write_json(resume_metadata_path(output_path), metadata)


def load_resume_result(
    file_path: str,
    output_path: str,
    input_signature: str,
) -> Optional[Tuple[int, int, Set[str]]]:
    if not os.path.isfile(output_path):
        return None

    metadata_path = resume_metadata_path(output_path)
    if not os.path.isfile(metadata_path):
        return None

    try:
        source_sig = file_signature(file_path)
        output_sig = file_signature(output_path)
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        if (
            metadata.get("version") != 3
            or metadata.get("source") != source_sig
            or metadata.get("output") != output_sig
            or metadata.get("input_signature") != input_signature
        ):
            return None
        return (
            int(metadata["postings_with_masked_sentence"]),
            int(metadata["masked_sentences"]),
            set(metadata.get("observed_skills", [])),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None
def preprocess_file_to_output(
    file_path: str,
    output_path: str,
    input_signature: str,
) -> Tuple[int, int, Set[str]]:
    results, observed_skills = preprocess_file(file_path)
    postings_with_masked_sentence = len({
        result["row_idx"] for result in results
    })
    masked_sentences = len(results)
    temp_path = f"{output_path}.tmp.{os.getpid()}"
    try:
        pd.DataFrame(results, columns=OUTPUT_COLUMNS).to_csv(
            temp_path,
            index=False,
            compression="gzip",
        )
        os.replace(temp_path, output_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    write_resume_metadata(
        file_path=file_path,
        output_path=output_path,
        postings_with_masked_sentence=postings_with_masked_sentence,
        masked_sentences=masked_sentences,
        observed_skills=observed_skills,
        input_signature=input_signature,
    )
    return (
        postings_with_masked_sentence,
        masked_sentences,
        observed_skills,
    )
def empty_year_stats() -> Dict[str, int]:
    return {
        "postings_with_masked_sentence": 0,
        "masked_sentences": 0,
    }


def save_preprocessing_stats(
    yearly_stats: Dict[str, Dict[str, int]],
):
    rows = [
        {"year": year, **yearly_stats[year]}
        for year in sorted(yearly_stats)
    ]
    rows.append({
        "year": "ALL",
        "postings_with_masked_sentence": sum(
            stats["postings_with_masked_sentence"]
            for stats in yearly_stats.values()
        ),
        "masked_sentences": sum(
            stats["masked_sentences"]
            for stats in yearly_stats.values()
        ),
    })
    columns = [
        "year",
        "postings_with_masked_sentence",
        "masked_sentences",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(STATS_OUTPUT, index=False)
    print(f"📊 Yearly/all mask stats saved → {STATS_OUTPUT}")


# ─── Process every file without split or sampling ─────
def process_all_data(
    monthly_files: Dict[Tuple[str, str], List[str]],
    workers: int,
    bert_path: str,
    resume: bool,
    selection_db_path: Path,
    input_signature: str,
) -> Tuple[Set[str], List[Tuple[str, str]]]:
    all_skills: Set[str] = set()
    processed_files: List[dict] = []
    group_counts: Dict[Tuple[str, str], int] = {}
    yearly_stats: Dict[str, Dict[str, int]] = {}
    failures: List[Tuple[str, str]] = []
    remaining_months_by_year: Dict[str, int] = {}
    for year, _ in monthly_files:
        remaining_months_by_year[year] = (
            remaining_months_by_year.get(year, 0) + 1
        )

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(bert_path, str(selection_db_path)),
    ) as executor:
        for (year, yyyy_mm), files in tqdm(
            monthly_files.items(),
            desc="Processing all monthly data",
        ):
            year_stats = yearly_stats.setdefault(year, empty_year_stats())
            out_dir = os.path.join(ALL_OUTPUT_ROOT, year, yyyy_mm)
            Path(out_dir).mkdir(parents=True, exist_ok=True)

            future_info = {}
            month_rows = 0
            resumed_files = 0

            for file_path in files:
                output_path = source_output_path(year, yyyy_mm, file_path)
                resume_result = None
                if resume:
                    resume_result = load_resume_result(
                        file_path,
                        output_path,
                        input_signature,
                    )

                if resume_result is not None:
                    (
                        postings_with_masked_sentence,
                        output_rows,
                        observed_skills,
                    ) = resume_result
                    resumed_files += 1
                    month_rows += output_rows
                    all_skills.update(observed_skills)
                    year_stats["postings_with_masked_sentence"] += (
                        postings_with_masked_sentence
                    )
                    year_stats["masked_sentences"] += output_rows
                    processed_files.append({
                        "year": year,
                        "yyyy_mm": yyyy_mm,
                        "file_path": file_path,
                        "preprocessed_path": output_path,
                        "output_rows": output_rows,
                        "status": "resumed",
                        "error": "",
                    })
                    continue

                future = executor.submit(
                    preprocess_file_to_output,
                    file_path,
                    output_path,
                    input_signature,
                )
                future_info[future] = (file_path, output_path)

            for future in tqdm(
                as_completed(future_info),
                total=len(future_info),
                desc=f"{yyyy_mm} files",
                leave=False,
            ):
                file_path, output_path = future_info[future]
                try:
                    (
                        postings_with_masked_sentence,
                        output_rows,
                        observed_skills,
                    ) = future.result()
                except Exception as exc:
                    error_text = f"{type(exc).__name__}: {exc}"
                    failures.append((file_path, error_text))
                    processed_files.append({
                        "year": year,
                        "yyyy_mm": yyyy_mm,
                        "file_path": file_path,
                        "preprocessed_path": output_path,
                        "output_rows": 0,
                        "status": "failed",
                        "error": error_text,
                    })
                    print(f"\n❌ Failed: {file_path}\n   {error_text}")
                    continue

                month_rows += output_rows
                all_skills.update(observed_skills)
                year_stats["postings_with_masked_sentence"] += (
                    postings_with_masked_sentence
                )
                year_stats["masked_sentences"] += output_rows
                processed_files.append({
                    "year": year,
                    "yyyy_mm": yyyy_mm,
                    "file_path": file_path,
                    "preprocessed_path": output_path,
                    "output_rows": output_rows,
                    "status": "ok",
                    "error": "",
                })

            group_counts[(year, yyyy_mm)] = month_rows
            print(
                f"✅ [all] {yyyy_mm}: {len(files):,} files, "
                f"{month_rows:,} rows, {resumed_files:,} resumed → {out_dir}"
            )

            remaining_months_by_year[year] -= 1
            if remaining_months_by_year[year] == 0:
                print(
                    f"♻️  Retained resume metadata for {year} "
                    "to validate body-dedup selection on future runs."
                )

    index_path = os.path.join(OUTPUT_ROOT, "processed_files_all.csv")
    index_columns = [
        "year",
        "yyyy_mm",
        "file_path",
        "preprocessed_path",
        "output_rows",
        "status",
        "error",
    ]
    index_df = pd.DataFrame(processed_files, columns=index_columns)
    if not index_df.empty:
        index_df = index_df.sort_values(["year", "yyyy_mm", "file_path"])
    index_df.to_csv(index_path, index=False)
    print(f"🗂  Complete file list saved → {index_path}")

    save_preprocessing_stats(yearly_stats)

    with open(GLOBAL_LOG, "a") as g:
        g.write("\n=== ALL DATA (no split, no sampling) ===\n")
        for (year, yyyy_mm), count in sorted(group_counts.items()):
            g.write(f"[{year} | {yyyy_mm}] total_rows: {count}\n")
        g.write(f"\nfailed_files: {len(failures)}\n")
        for file_path, error_text in failures:
            g.write(f"[FAILED] {file_path}: {error_text}\n")

    return all_skills, failures


# ─── Debug mode: one file, 10 original rows ──────
def debug_mode(debug_file: str, n_rows: int = 10):
    """
    Debug mode:
	•	Uses only the single file specified by debug_file.
	•	Samples up to n_rows original rows from that file.
	•	Saves the complete preprocessing outputs for these rows to debug_sample.csv.gz.
    """
    if not os.path.exists(debug_file):
        raise FileNotFoundError(f"debug_file not found: {debug_file}")

    print(f"[DEBUG] Using file: {debug_file}")

    # read original file
    df = pd.read_csv(
        debug_file,
        compression="gzip" if debug_file.endswith(".gz") else None,
        low_memory=False
    )

    if "skills_name" not in df.columns or "body" not in df.columns:
        raise RuntimeError("Input file does not contain 'skills_name' and 'body' columns.")

    df = df.dropna(subset=["skills_name", "body"])
    if df.empty:
        print("[DEBUG] No rows with both skills_name and body.")
        return

    # Randomly select up to n_rows original rows
    if len(df) > n_rows:
        df_subset = df.sample(n=n_rows, random_state=42)
    else:
        df_subset = df

    print(f"[DEBUG] Selected {len(df_subset)} original rows")

   
    results, _ = preprocess_rows(df_subset, debug_file)
    print(f"[DEBUG] Generated {len(results)} processed rows from {len(df_subset)} original rows")

    if not results:
        print("[DEBUG] No processed rows generated from selected originals.")
        return

    out_path = os.path.join(OUTPUT_ROOT, "debug_sample.csv.gz")
    pd.DataFrame(results).to_csv(out_path, index=False, compression="gzip")
    print(f"[DEBUG] Saved processed rows → {out_path}")

    print("\n[DEBUG] Example rows:")
    print(pd.DataFrame(results)[["row_idx", "true_skill", "masked_sentence"]].head(5))

# ─── Main ───────────────────────────────
def main():
    global tokenizer

    parser = argparse.ArgumentParser(description="Preprocess job postings (ONLY IT skills)")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, run in debug mode on a single file and up to N original rows."
    )
    parser.add_argument(
        "--debug_file",
        type=str,
        default=None,
        help="CSV/CSV.GZ file path for debug mode (required when --debug is set)."
    )
    parser.add_argument(
        "--debug_n",
        type=int,
        default=10,
        help="Number of original rows to use in debug mode (default: 10)."
    )
    parser.add_argument(
        "--bert_path",
        type=str,
        default=MODEL_NAME,
        help="Path or HF model name for BertTokenizer (default: bert-base-uncased)."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=OUTPUT_ROOT,
        help="Directory where all preprocessing outputs will be saved.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of file-level worker processes (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--body-dedup-seed",
        type=int,
        default=DEFAULT_BODY_DEDUP_SEED,
        help="Seed for deterministic normalized-body representative selection.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from completed per-source outputs (default: enabled).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore completed outputs and preprocess every source file again.",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.body_dedup_seed < 0:
        parser.error("--body-dedup-seed must be >= 0")

    configure_output_paths(args.output_path)
    print(f"📁 Output root: {OUTPUT_ROOT}")
    print(f"♻️  Resume mode: {args.resume}")

    # debug mode
    if args.debug:
        if args.debug_file is None:
            raise ValueError("--debug_file must be provided when using --debug mode.")
        tokenizer = load_tokenizer(args.bert_path)
        debug_mode(debug_file=args.debug_file, n_rows=args.debug_n)
        return

    with open(GLOBAL_LOG, "w") as g:
        g.write("Preprocessing Log (all data, ONLY target IT skills)\n")
        g.write("===================================================\n")
        g.write(f"INPUT_ROOT={INPUT_ROOT}\n")
        g.write(f"TARGET_SKILLS_CSV={IT_SKILLS_FILE}\n")
        g.write(f"WORKERS={args.workers}\n")
        g.write(f"BODY_DEDUP_SEED={args.body_dedup_seed}\n")

    monthly_files: Dict[Tuple[str, str], List[str]] = {}
    for year, yyyy_mm, fpath in iter_month_files(INPUT_ROOT):
        monthly_files.setdefault((year, yyyy_mm), []).append(fpath)

    if not monthly_files:
        raise FileNotFoundError(f"No monthly CSV/CSV.GZ files found under: {INPUT_ROOT}")

    total_files = sum(len(files) for files in monthly_files.values())
    print(
        f"📂 Found {total_files:,} files across {len(monthly_files):,} months; "
        f"processing all files with {args.workers} workers."
    )

    input_signature = preprocessing_input_signature(
        monthly_files,
        args.body_dedup_seed,
    )
    with tempfile.TemporaryDirectory(
        prefix=".body_dedup_work_",
        dir=OUTPUT_ROOT,
    ) as work_dir:
        selection_db_path = Path(work_dir) / "body_posting_choice.sqlite"
        body_stats = build_body_dedup_index(
            db_path=selection_db_path,
            monthly_files=monthly_files,
            workers=args.workers,
            seed=args.body_dedup_seed,
        )
        body_stats["input_signature"] = input_signature
        body_summary_path = Path(OUTPUT_ROOT) / "body_dedup_summary.json"
        atomic_write_json(body_summary_path, body_stats)
        print(
            "🧹 Body dedup: "
            f"{body_stats['eligible_job_description_postings']:,} eligible -> "
            f"{body_stats['unique_body_postings']:,} unique "
            f"({body_stats['duplicate_body_postings_dropped']:,} dropped)"
        )

        all_skills_set, failures = process_all_data(
            monthly_files=monthly_files,
            workers=args.workers,
            bert_path=args.bert_path,
            resume=args.resume,
            selection_db_path=selection_db_path,
            input_signature=input_signature,
        )

    skill2idx = {s: i for i, s in enumerate(sorted(all_skills_set))}
    with open(VOCAB_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(skill2idx, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved skill2idx.json with {len(skill2idx)} skills → {VOCAB_OUTPUT}")

    if failures:
        raise RuntimeError(
            f"Preprocessing finished with {len(failures)} failed source file(s). "
            f"See {GLOBAL_LOG} and processed_files_all.csv for details."
        )

if __name__ == "__main__":
    main()


"""
python3 /home/jovyan/hyunjincho/LEM_journal/contextual_skills_substitutes/model_trains/preprocess.py \
  --output_path /home/jovyan/LEM_data3/hyunjincho/LEM \
  --workers 8
  """