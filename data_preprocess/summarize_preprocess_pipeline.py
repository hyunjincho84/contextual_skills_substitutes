#!/usr/bin/env python3
"""Write and print end-to-end preprocessing pipeline stage counts."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"required summary not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"summary must be a JSON object: {path}")
    return payload


def required_int(payload: dict, key: str, path: Path) -> int:
    if key not in payload:
        raise KeyError(f"{path} is missing required key: {key}")
    return int(payload[key])


def read_all_preprocess_stats(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"required preprocessing stats not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("year", "")).strip().upper() == "ALL":
                return {
                    "postings_with_masked_sentence": int(
                        row["postings_with_masked_sentence"]
                    ),
                    "masked_sentences": int(row["masked_sentences"]),
                }
    raise ValueError(f"{path} does not contain an ALL row")


def format_count(value: int) -> str:
    return f"{value:,}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print and save preprocessing pipeline stage counts."
    )
    parser.add_argument("--preprocess-root", type=Path, required=True)
    parser.add_argument("--filter-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-txt", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    preprocess_root = args.preprocess_root.expanduser().resolve()
    filter_root = args.filter_root.expanduser().resolve()
    split_root = args.split_root.expanduser().resolve()
    output_txt = args.output_txt.expanduser().resolve()

    body_path = preprocess_root / "body_dedup_summary.json"
    preprocess_stats_path = preprocess_root / "preprocessing_stats_by_year.csv"
    filter_path = filter_root / "filter_summary.json"
    split_path = split_root / "split_summary.json"

    body = read_json(body_path)
    preprocess_stats = read_all_preprocess_stats(preprocess_stats_path)
    filtered = read_json(filter_path)
    split = read_json(split_path)

    raw_rows = required_int(body, "input_rows", body_path)
    valid_job_description = required_int(
        body,
        "valid_job_description_postings",
        body_path,
    )
    target_eligible = required_int(
        body,
        "eligible_job_description_postings",
        body_path,
    )
    unique_body_postings = required_int(body, "unique_body_postings", body_path)
    duplicate_body_postings = required_int(
        body,
        "duplicate_body_postings_dropped",
        body_path,
    )
    masked_postings = preprocess_stats["postings_with_masked_sentence"]
    masked_rows = preprocess_stats["masked_sentences"]

    filter_input_rows = required_int(filtered, "input_rows", filter_path)
    valid_pair_rows = required_int(filtered, "valid_pair_rows", filter_path)
    duplicate_pair_rows = required_int(
        filtered,
        "duplicate_pair_rows",
        filter_path,
    )
    unique_pairs = required_int(filtered, "unique_pairs", filter_path)
    unique_pair_postings = required_int(
        filtered,
        "unique_pair_postings",
        filter_path,
    )
    final_pairs = required_int(filtered, "output_rows", filter_path)
    final_postings = required_int(filtered, "output_postings", filter_path)

    split_input_rows = required_int(split, "input_rows", split_path)
    split_postings = required_int(split, "global_postings", split_path)
    train_pairs = required_int(split, "train_rows", split_path)
    train_postings = required_int(split, "train_postings", split_path)
    test_pairs = required_int(split, "test_rows", split_path)
    test_postings = required_int(split, "test_postings", split_path)

    if filter_input_rows != masked_rows:
        raise RuntimeError(
            "preprocess/filter row mismatch: "
            f"{masked_rows:,} != {filter_input_rows:,}"
        )
    if final_pairs != split_input_rows:
        raise RuntimeError(
            "filter/split row mismatch: "
            f"{final_pairs:,} != {split_input_rows:,}"
        )
    if final_postings != split_postings:
        raise RuntimeError(
            "filter/split posting mismatch: "
            f"{final_postings:,} != {split_postings:,}"
        )
    if train_pairs + test_pairs != split_input_rows:
        raise RuntimeError("train/test pair accounting does not balance")
    if train_postings + test_postings != split_postings:
        raise RuntimeError("train/test posting accounting does not balance")

    lines = [
        "=" * 72,
        "PREPROCESSING PIPELINE STAGE COUNTS",
        "=" * 72,
        "",
        "[1] Raw posting scan",
        f"  Raw rows scanned: {format_count(raw_rows)}",
        (
            "  Valid skills_name/body + Job Description postings: "
            f"{format_count(valid_job_description)}"
        ),
        (
            "  Of those, postings with >=1 target skill: "
            f"{format_count(target_eligible)}"
        ),
        "",
        "[2] Normalized-body posting deduplication + masking",
        (
            "  Duplicate body postings removed: "
            f"{format_count(duplicate_body_postings)}"
        ),
        (
            "  Unique normalized-body postings: "
            f"{format_count(unique_body_postings)}"
        ),
        (
            "  Unique postings producing >=1 masked_sentence: "
            f"{format_count(masked_postings)}"
        ),
        (
            "  Generated masked_sentence rows: "
            f"{format_count(masked_rows)}"
        ),
        "",
        "[3] Exact (masked_sentence, true_skill) deduplication",
        f"  Valid preprocessed pair rows: {format_count(valid_pair_rows)}",
        (
            "  Duplicate pair rows removed: "
            f"{format_count(duplicate_pair_rows)}"
        ),
        (
            "  Unique (masked_sentence, true_skill) pairs: "
            f"{format_count(unique_pairs)}"
        ),
        (
            "  Postings represented after exact pair dedup: "
            f"{format_count(unique_pair_postings)}"
        ),
        "  After frequency/exclusion/cooccurrence vocabulary filters:",
        f"    Final unique pair rows: {format_count(final_pairs)}",
        f"    Final postings: {format_count(final_postings)}",
        "",
        "[4] Train/test split by posting_fingerprint",
        (
            f"  Train: postings={format_count(train_postings)}, "
            f"unique pairs={format_count(train_pairs)}"
        ),
        (
            f"  Test:  postings={format_count(test_postings)}, "
            f"unique pairs={format_count(test_pairs)}"
        ),
        (
            f"  Total: postings={format_count(split_postings)}, "
            f"unique pairs={format_count(split_input_rows)}"
        ),
        "",
        "Source summaries:",
        f"  {body_path}",
        f"  {preprocess_stats_path}",
        f"  {filter_path}",
        f"  {split_path}",
        "=" * 72,
    ]
    report = "\n".join(lines) + "\n"

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_txt.with_name(f"{output_txt.name}.tmp.{os.getpid()}")
    try:
        temp_path.write_text(report, encoding="utf-8")
        os.replace(temp_path, output_txt)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    print(report, end="")
    print(f"Saved stage-count report: {output_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

