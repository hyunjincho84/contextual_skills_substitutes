#!/usr/bin/env python3
"""Add validated GPT Top-5 predictions to the human-evaluation CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_tests.test_gpt import (  # noqa: E402
    build_prompt_cache_key,
    call_gpt,
    load_all_skills,
    print_usage,
)

BASE_DATA_DIR = Path(
    os.environ.get("BASE_DATA_DIR", "/home/jovyan/LEM_data3/hyunjincho/LEM")
)
DEFAULT_INPUT_CSV = BASE_DATA_DIR / "human_eval" / "skill_or_skill_shuffled.csv"
DEFAULT_SKILL2IDX = BASE_DATA_DIR / "all_filterd" / "skill2idx.json"
DEFAULT_MODEL = os.environ.get("GPT_MODEL", "gpt-5.4-2026-03-05")
PREDICTION_COLUMN = "gpt_top5_pred"
IDENTITY_COLUMNS = (
    "row_idx",
    "file_path",
    "preprocessed_file_path",
    "source_csv_row",
    "mask_answer",
    "or_skill",
    "masked_sentence",
)
CHECKPOINT_COLUMNS = (
    "row_position",
    "identity_hash",
    PREDICTION_COLUMN,
    "gpt_model",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Infer GPT Top-5 skills for each human-evaluation row and append "
            f"the {PREDICTION_COLUMN!r} column."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"Source human-evaluation CSV (default: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Destination CSV; defaults to updating --input-csv in place.",
    )
    parser.add_argument(
        "--skill2idx",
        type=Path,
        default=DEFAULT_SKILL2IDX,
        help=f"Allowed-skill JSON (default: {DEFAULT_SKILL2IDX})",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=3.0)
    parser.add_argument(
        "--format-retries",
        type=int,
        default=3,
        help="Number of GPT calls allowed when an answer violates Top-5 format.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Per-row checkpoint CSV (default: <output-csv>.gpt_checkpoint.csv).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse validated predictions in the output/checkpoint and fill missing rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard existing GPT predictions/checkpoint and recompute every row.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many missing rows (useful for a smoke test).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not make a timestamped backup before replacing an existing output.",
    )
    args = parser.parse_args()

    if args.max_retries < 1:
        parser.error("--max-retries must be at least 1")
    if args.format_retries < 1:
        parser.error("--format-retries must be at least 1")
    if args.retry_delay < 0:
        parser.error("--retry-delay must be nonnegative")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite cannot be used together")
    return args


def normalize_skill(value: Any) -> str:
    return "" if value is None else str(value).strip().casefold()


def build_skill_lookup(all_skills: list[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for skill in all_skills:
        canonical = str(skill).strip()
        key = normalize_skill(canonical)
        if not key:
            raise ValueError("The skill whitelist contains an empty skill")
        previous = lookup.get(key)
        if previous is not None and previous != canonical:
            raise ValueError(
                "Case-insensitive skill collision in whitelist: "
                f"{previous!r} vs {canonical!r}"
            )
        lookup[key] = canonical
    return lookup


def validate_top5(value: Any, skill_lookup: dict[str, str]) -> str:
    """Return a pipe-delimited Top-5, preserving out-of-whitelist skills."""
    raw_items = str(value).strip().split("|")
    items = [item.strip() for item in raw_items]
    if len(items) != 5 or any(not item for item in items):
        raise ValueError(f"expected exactly five non-empty pipe-delimited skills: {value!r}")

    normalized = [normalize_skill(item) for item in items]
    if len(set(normalized)) != 5:
        raise ValueError(f"expected five distinct skills: {value!r}")

    return "|".join(
        skill_lookup.get(key, item) for item, key in zip(items, normalized)
    )


def find_unknown_skills(
    prediction: str, skill_lookup: dict[str, str]
) -> list[str]:
    return [
        item.strip()
        for item in prediction.split("|")
        if normalize_skill(item) not in skill_lookup
    ]


def row_identity_hash(row: pd.Series) -> str:
    payload = [str(row.get(column, "")) for column in IDENTITY_COLUMNS]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def validate_schema(frame: pd.DataFrame, path: Path) -> None:
    required = set(IDENTITY_COLUMNS)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"No data rows found: {path}")
    empty_masks = frame["masked_sentence"].astype(str).str.strip().eq("")
    if empty_masks.any():
        positions = frame.index[empty_masks].tolist()[:10]
        raise ValueError(f"Empty masked_sentence at row positions: {positions}")


def assert_same_rows(source: pd.DataFrame, output: pd.DataFrame) -> None:
    if len(source) != len(output):
        raise ValueError(
            f"Resume output row count differs from input: {len(output):,} != {len(source):,}"
        )
    mismatches = [
        position
        for position in range(len(source))
        if row_identity_hash(source.iloc[position])
        != row_identity_hash(output.iloc[position])
    ]
    if mismatches:
        raise ValueError(
            "Resume output rows do not match the input; first positions="
            f"{mismatches[:10]}"
        )


def load_working_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, bool]:
    input_path = args.input_csv.expanduser().resolve()
    output_path = args.output_csv.expanduser().resolve()
    source = read_csv(input_path)
    validate_schema(source, input_path)

    if output_path != input_path and output_path.exists():
        if not (args.resume or args.overwrite):
            raise FileExistsError(
                f"Output exists; use --resume or --overwrite: {output_path}"
            )
        if args.resume:
            frame = read_csv(output_path)
            validate_schema(frame, output_path)
            assert_same_rows(source, frame)
        else:
            frame = source.copy()
    else:
        frame = source.copy()

    had_prediction_column = PREDICTION_COLUMN in frame.columns
    if not had_prediction_column:
        frame[PREDICTION_COLUMN] = ""
    elif args.overwrite:
        frame[PREDICTION_COLUMN] = ""
    elif not args.resume and frame[PREDICTION_COLUMN].astype(str).str.strip().ne("").any():
        raise FileExistsError(
            f"{PREDICTION_COLUMN} already contains values; use --resume or --overwrite"
        )

    return frame, not had_prediction_column or args.overwrite


def checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint.expanduser().resolve()
    output = args.output_csv.expanduser().resolve()
    return Path(f"{output}.gpt_checkpoint.csv")


def load_checkpoint(
    path: Path,
    frame: pd.DataFrame,
    model: str,
    skill_lookup: dict[str, str],
) -> dict[int, str]:
    predictions: dict[int, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CHECKPOINT_COLUMNS:
            raise ValueError(f"Checkpoint schema mismatch: {path}")
        for checkpoint_row in reader:
            position = int(checkpoint_row["row_position"])
            if not 0 <= position < len(frame):
                raise ValueError(f"Checkpoint row position is out of range: {position}")
            expected_hash = row_identity_hash(frame.iloc[position])
            if checkpoint_row["identity_hash"] != expected_hash:
                raise ValueError(f"Checkpoint identity mismatch at row position {position}")
            if checkpoint_row["gpt_model"] != model:
                raise ValueError(
                    "Checkpoint model mismatch at row position "
                    f"{position}: {checkpoint_row['gpt_model']!r} != {model!r}"
                )
            prediction = validate_top5(
                checkpoint_row[PREDICTION_COLUMN], skill_lookup
            )
            previous = predictions.get(position)
            if previous is not None and previous != prediction:
                raise ValueError(f"Conflicting checkpoint entries at row position {position}")
            predictions[position] = prediction
    return predictions


def collect_completed(
    frame: pd.DataFrame,
    saved: dict[int, str],
    skill_lookup: dict[str, str],
) -> tuple[dict[int, str], bool]:
    completed: dict[int, str] = {}
    changed = False
    for position, value in enumerate(frame[PREDICTION_COLUMN].tolist()):
        if str(value).strip():
            completed[position] = validate_top5(value, skill_lookup)
            if completed[position] != value:
                frame.at[position, PREDICTION_COLUMN] = completed[position]
                changed = True

    for position, prediction in saved.items():
        existing = completed.get(position)
        if existing is not None and existing != prediction:
            raise ValueError(f"Output/checkpoint conflict at row position {position}")
        if existing is None:
            completed[position] = prediction
            frame.at[position, PREDICTION_COLUMN] = prediction
            changed = True
    return completed, changed


def call_validated_gpt(
    client: Any,
    model: str,
    masked_sentence: str,
    all_skills: list[str],
    skill_lookup: dict[str, str],
    max_retries: int,
    retry_delay: float,
    format_retries: int,
    prompt_cache_key: str,
    usage_totals: dict[str, int],
) -> str:
    last_error: ValueError | None = None
    for attempt in range(1, format_retries + 1):
        raw_prediction = call_gpt(
            client,
            model,
            masked_sentence,
            all_skills,
            max_retries,
            retry_delay,
            prompt_cache_key,
            usage_totals,
        )
        try:
            prediction = validate_top5(raw_prediction, skill_lookup)
            unknown = find_unknown_skills(prediction, skill_lookup)
            if unknown:
                tqdm.write(
                    "[WHITELIST WARNING] saving out-of-whitelist skills "
                    f"without stopping: {unknown}"
                )
            return prediction
        except ValueError as exc:
            last_error = exc
            tqdm.write(f"[FORMAT ERROR] attempt {attempt}/{format_retries}: {exc}")
    raise RuntimeError(
        f"GPT output remained invalid after {format_retries} calls: {last_error}"
    )


def open_checkpoint_writer(path: Path) -> tuple[Any, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    handle = path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=CHECKPOINT_COLUMNS)
    if write_header:
        writer.writeheader()
        handle.flush()
        os.fsync(handle.fileno())
    return handle, writer


def append_checkpoint(
    handle: Any,
    writer: csv.DictWriter,
    position: int,
    frame: pd.DataFrame,
    prediction: str,
    model: str,
) -> None:
    writer.writerow(
        {
            "row_position": position,
            "identity_hash": row_identity_hash(frame.iloc[position]),
            PREDICTION_COLUMN: prediction,
            "gpt_model": model,
        }
    )
    handle.flush()
    os.fsync(handle.fileno())


def write_output_atomic(
    frame: pd.DataFrame,
    output_path: Path,
    make_backup: bool,
) -> Path | None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path: Path | None = None
    if output_path.exists() and make_backup:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = output_path.with_name(
            f"{output_path.name}.before_gpt_top5.{timestamp}.bak"
        )
        shutil.copy2(output_path, backup_path)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.to_csv(temporary_path, index=False)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return backup_path


def main() -> None:
    args = parse_args()
    input_path = args.input_csv.expanduser().resolve()
    output_path = args.output_csv.expanduser().resolve()
    checkpoint = checkpoint_path(args)

    frame, changed = load_working_frame(args)
    all_skills = load_all_skills(str(args.skill2idx.expanduser().resolve()))
    skill_lookup = build_skill_lookup(all_skills)

    if checkpoint.exists():
        if args.overwrite:
            checkpoint.unlink()
        elif not args.resume:
            raise FileExistsError(
                f"Checkpoint exists; use --resume or --overwrite: {checkpoint}"
            )

    saved = (
        load_checkpoint(checkpoint, frame, args.model, skill_lookup)
        if checkpoint.exists()
        else {}
    )
    completed, completed_changed = collect_completed(frame, saved, skill_lookup)
    changed = changed or completed_changed

    pending = [position for position in range(len(frame)) if position not in completed]
    missing_before_limit = len(pending)
    if args.limit is not None:
        pending = pending[: args.limit]

    usage_totals = {
        "requests": 0,
        "input_tokens": 0,
        "cached_tokens": 0,
        "output_tokens": 0,
    }
    newly_called = 0
    if pending:
        from openai import OpenAI

        client = OpenAI()
        prompt_cache_key = build_prompt_cache_key(args.model, all_skills)
        print(f"[INFO] Input: {input_path}")
        print(f"[INFO] Output: {output_path}")
        print(f"[INFO] Model: {args.model}")
        print(f"[INFO] Prompt cache key: {prompt_cache_key}")
        print(
            f"[INFO] Existing={len(completed):,}, missing={missing_before_limit:,}, "
            f"this run={len(pending):,}"
        )

        handle, writer = open_checkpoint_writer(checkpoint)
        try:
            for position in tqdm(pending, desc="GPT human eval", unit="row"):
                prediction = call_validated_gpt(
                    client=client,
                    model=args.model,
                    masked_sentence=str(frame.at[position, "masked_sentence"]),
                    all_skills=all_skills,
                    skill_lookup=skill_lookup,
                    max_retries=args.max_retries,
                    retry_delay=args.retry_delay,
                    format_retries=args.format_retries,
                    prompt_cache_key=prompt_cache_key,
                    usage_totals=usage_totals,
                )
                frame.at[position, PREDICTION_COLUMN] = prediction
                append_checkpoint(
                    handle, writer, position, frame, prediction, args.model
                )
                completed[position] = prediction
                newly_called += 1
                changed = True
        finally:
            handle.close()
            print_usage(usage_totals)
    else:
        if missing_before_limit:
            print("[INFO] --limit selected zero missing rows; no GPT call was made.")
        else:
            print("[INFO] No missing GPT predictions to process.")

    if changed or output_path != input_path or not output_path.exists():
        backup = write_output_atomic(
            frame, output_path, make_backup=not args.no_backup
        )
        print(f"[INFO] Saved: {output_path}")
        if backup is not None:
            print(f"[INFO] Backup: {backup}")

    available = frame[PREDICTION_COLUMN].astype(str).str.strip().ne("").sum()
    print(
        f"[DONE] {PREDICTION_COLUMN}: {available:,}/{len(frame):,}; "
        f"GPT-called this run={newly_called:,}; checkpoint={checkpoint}"
    )
    if available < len(frame):
        print("[INFO] Run again with --resume to fill the remaining rows.")


if __name__ == "__main__":
    main()
