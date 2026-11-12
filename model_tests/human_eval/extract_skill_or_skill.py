#!/usr/bin/env python3
"""Extract strict ``skill A or skill B`` rows and shuffle them.

Only the ``[MASK] or <skill B>`` direction is accepted. ``true_skill`` is
the masked answer, while the visible skill B must be an exact member of the
skill inventory after conservative text normalization.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

try:
    from .human_eval_predictions import OUTPUT_COLUMNS, attach_all_predictions
except ImportError:  # Direct script execution
    from human_eval_predictions import OUTPUT_COLUMNS, attach_all_predictions


# ---------------------------------------------------------------------------
# Editable defaults
# ---------------------------------------------------------------------------
BASE_DATA_DIR = Path("/home/jovyan/LEM_data3/hyunjincho/LEM")
TEST_ROOT = BASE_DATA_DIR / "all_train_test/test"
INPUT_GLOB = str(TEST_ROOT / "20*/20*/*.csv.gz")
SKILL_JSON = BASE_DATA_DIR / "all_filterd/skill2idx.json"
OUTPUT_CSV = BASE_DATA_DIR / "human_eval/skill_or_skill_shuffled.csv.gz"
BERT_PREDICTION_GLOB = str(
    BASE_DATA_DIR / "predictions/bert_pred/pred/20*/predictions_*.csv.gz"
)
SKILL2VEC_MODEL = (
    BASE_DATA_DIR
    / "models/skill2vec_model/skill2vec_norm_sg1_d300_win50_neg10_ep5.kv"
)
SKILL2VEC_TOKEN_MAP = (
    BASE_DATA_DIR
    / "models/skill2vec_model/skill2vec_norm_sg1_d300_win50_neg10_ep5.skill_tokens.json"
)
CONDITIONAL_TABLE = BASE_DATA_DIR / "models/condprob_model/cooc_probs.csv.gz"

CHUNK_SIZE = 200_000
RANDOM_SEED = 42
REQUIRED_COLUMNS = ("row_idx", "true_skill", "masked_sentence")

MASK_OR_RE = re.compile(r"\[MASK\]\s+or\s+", flags=re.IGNORECASE)
NON_DIRECT_MODIFIERS = {"similar", "related", "equivalent", "alternative"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find strict '[MASK] or <whitelist skill>' rows, add skill-pair "
            "metadata, shuffle rows reproducibly, and save one CSV."
        )
    )
    parser.add_argument("--input-glob", default=INPUT_GLOB)
    parser.add_argument("--skill-json", type=Path, default=SKILL_JSON)
    parser.add_argument("--test-root", type=Path, default=TEST_ROOT)
    parser.add_argument("--bert-prediction-glob", default=BERT_PREDICTION_GLOB)
    parser.add_argument("--skill2vec-model", type=Path, default=SKILL2VEC_MODEL)
    parser.add_argument(
        "--skill2vec-token-map", type=Path, default=SKILL2VEC_TOKEN_MAP
    )
    parser.add_argument("--conditional-table", type=Path, default=CONDITIONAL_TABLE)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--years", nargs="*", default=None, help="Example: --years 2020 2021")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Process only the first N sorted files (for a small smoke test).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace --output if it already exists.",
    )
    return parser.parse_args()


def normalize_skill_text(value: Any) -> str:
    """Normalize text in the same spirit as the reference implementation."""
    if value is None or pd.isna(value):
        return ""

    text = str(value).lower().strip()
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9+#./\-\s\[\]]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_skill_inventory(skill_json: Path) -> tuple[dict[str, str], int]:
    if not skill_json.is_file():
        raise FileNotFoundError(f"Skill inventory not found: {skill_json}")

    with skill_json.open("r", encoding="utf-8") as handle:
        skill2idx = json.load(handle)

    if not isinstance(skill2idx, dict) or not skill2idx:
        raise ValueError(f"Expected a non-empty JSON object: {skill_json}")

    normalized_to_canonical: dict[str, str] = {}
    for skill in skill2idx:
        normalized = normalize_skill_text(skill)
        if normalized:
            normalized_to_canonical.setdefault(normalized, str(skill))

    if not normalized_to_canonical:
        raise ValueError(f"No valid skills found in: {skill_json}")

    max_words = max(len(skill.split()) for skill in normalized_to_canonical)
    return normalized_to_canonical, max_words


def longest_skill_from_prefix(
    text: str,
    normalized_to_canonical: dict[str, str],
    max_words: int,
) -> str | None:
    """Return the longest whitelist skill beginning immediately at ``text``."""
    normalized = normalize_skill_text(text)
    tokens = normalized.split()
    if not tokens or tokens[0] in NON_DIRECT_MODIFIERS:
        return None

    for width in range(min(max_words, len(tokens)), 0, -1):
        # Strip only sentence-ending periods; keep internal dots in skills
        # such as ``node.js`` and leading dots in skills such as ``.net``.
        candidate = " ".join(tokens[:width]).rstrip(".")
        canonical = normalized_to_canonical.get(candidate)
        if canonical is not None:
            return canonical
    return None


def extract_skill_b(
    masked_sentence: Any,
    normalized_to_canonical: dict[str, str],
    max_words: int,
) -> str | None:
    if masked_sentence is None or pd.isna(masked_sentence):
        return None

    sentence = str(masked_sentence)
    match = MASK_OR_RE.search(sentence)
    if match is None:
        return None

    return longest_skill_from_prefix(
        sentence[match.end() :],
        normalized_to_canonical,
        max_words,
    )


def extract_year(path: str) -> str | None:
    for part in Path(path).parts:
        if re.fullmatch(r"20\d{2}", part):
            return part
    return None


def list_input_files(
    input_glob: str,
    years: list[str] | None,
    max_files: int | None,
) -> list[str]:
    files = sorted(glob.glob(input_glob, recursive=True))

    if years:
        requested = {str(year) for year in years}
        invalid = sorted(year for year in requested if not re.fullmatch(r"20\d{2}", year))
        if invalid:
            raise ValueError(f"Invalid years: {invalid}")
        files = [path for path in files if extract_year(path) in requested]

    if max_files is not None:
        if max_files <= 0:
            raise ValueError("--max-files must be greater than zero")
        files = files[:max_files]

    return files


def validate_header(path: str) -> list[str]:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")

    generated = {
        "shuffle_row_id",
        "source_csv_path",
        "source_csv_row",
        "template_name",
        "skill_a",
        "skill_b",
        "skill_pair",
        "mask_side",
    }
    collisions = sorted(generated.intersection(columns))
    if collisions:
        raise ValueError(f"Generated output columns already exist in {path}: {collisions}")
    return columns


def process_file(
    path: str,
    normalized_to_canonical: dict[str, str],
    max_words: int,
    chunk_size: int,
) -> tuple[list[dict[str, Any]], dict[str, int], list[str]]:
    input_columns = validate_header(path)
    matched_rows: list[dict[str, Any]] = []
    stats = {"rows_read": 0, "template_candidate_rows": 0, "matched_rows": 0}
    source_row_offset = 0

    for chunk in pd.read_csv(path, chunksize=chunk_size):
        chunk = chunk.copy()
        chunk["source_csv_row"] = range(source_row_offset, source_row_offset + len(chunk))
        source_row_offset += len(chunk)
        stats["rows_read"] += len(chunk)

        candidate_mask = chunk["masked_sentence"].astype("string").str.contains(
            MASK_OR_RE,
            na=False,
        )
        candidates = chunk.loc[candidate_mask]
        stats["template_candidate_rows"] += len(candidates)

        for row in candidates.to_dict(orient="records"):
            normalized_true_skill = normalize_skill_text(row["true_skill"])
            true_skill = normalized_to_canonical.get(normalized_true_skill)
            if true_skill is None:
                continue

            visible_skill_b = extract_skill_b(
                row["masked_sentence"],
                normalized_to_canonical,
                max_words,
            )
            if visible_skill_b is None:
                continue

            skill_a = true_skill
            skill_b = visible_skill_b
            mask_side = "left"

            output_row = {
                "source_csv_path": path,
                "source_csv_row": int(row.pop("source_csv_row")),
                "template_name": "skill_a_or_skill_b",
                "skill_a": skill_a,
                "skill_b": skill_b,
                "skill_pair": f"{skill_a} or {skill_b}",
                "mask_side": mask_side,
            }
            output_row.update(row)
            matched_rows.append(output_row)

    stats["matched_rows"] = len(matched_rows)
    return matched_rows, stats, input_columns


def save_output(rows: list[dict[str, Any]], output: Path, seed: int) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if not frame.empty:
        frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if output.name.endswith(".gz") else None
    frame.to_csv(output, index=False, compression=compression)
    return frame


def main() -> None:
    args = parse_args()

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be greater than zero")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite: {args.output}")

    skill_inventory, max_words = load_skill_inventory(args.skill_json)
    files = list_input_files(args.input_glob, args.years, args.max_files)
    if not files:
        raise FileNotFoundError(f"No input files matched: {args.input_glob}")

    print(f"[INFO] input files : {len(files):,}")
    print(f"[INFO] skills      : {len(skill_inventory):,}")
    print(f"[INFO] seed        : {args.seed}")
    print(f"[INFO] output      : {args.output}")

    all_rows: list[dict[str, Any]] = []
    totals = {"rows_read": 0, "template_candidate_rows": 0, "matched_rows": 0}

    for path in tqdm(files, desc="Extracting", unit="file"):
        rows, stats, _ = process_file(path, skill_inventory, max_words, args.chunk_size)
        all_rows.extend(rows)
        for name in totals:
            totals[name] += stats[name]

    prediction_rows, prediction_stats = attach_all_predictions(
        all_rows,
        args.test_root,
        args.bert_prediction_glob,
        args.skill2vec_model,
        args.skill2vec_token_map,
        args.conditional_table,
        args.chunk_size,
    )
    shuffled = save_output(prediction_rows, args.output, args.seed)

    print("\n[SUMMARY]")
    print(f"  rows read        : {totals['rows_read']:,}")
    print(f"  template rows    : {totals['template_candidate_rows']:,}")
    print(f"  matched rows     : {totals['matched_rows']:,}")
    print(
        "  W2V missing      : "
        f"{prediction_stats['skill2vec_missing_queries']:,}"
    )
    print(
        "  CondProb missing : "
        f"{prediction_stats['condprob_missing_queries']:,}"
    )
    print(f"  saved rows       : {len(shuffled):,}")
    print(f"  saved to         : {args.output}")




if __name__ == "__main__":
    main()
