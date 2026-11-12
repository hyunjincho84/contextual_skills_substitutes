#!/usr/bin/env python3
"""Attach BERT, Skill2Vec, and conditional Top-5 predictions to OR rows."""

from __future__ import annotations

import glob
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd


OUTPUT_COLUMNS = [
    "row_idx",
    "file_path",
    "preprocessed_file_path",
    "source_csv_row",
    "mask_answer",
    "or_skill",
    "masked_sentence",
    "bert_pred_top5",
    "skill2vec_pred_top5",
    "condprob_pred_top5",
]

_BERT_MONTH_RE = re.compile(r"predictions_(20\d{2}-(?:0[1-9]|1[0-2]))\.csv\.gz$")


def prepare_requested_columns(
    extracted_rows: list[dict[str, Any]],
    test_root: Path,
) -> list[dict[str, Any]]:
    """Reduce extracted records to the requested fields plus empty predictions."""
    resolved_root = test_root.expanduser().resolve()
    prepared: list[dict[str, Any]] = []

    for row in extracted_rows:
        source_path = Path(str(row["source_csv_path"])).expanduser().resolve()
        try:
            relative = source_path.relative_to(resolved_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Preprocessed file is outside --test-root: {source_path}"
            ) from exc

        mask_side = row["mask_side"]
        if mask_side != "left":
            raise ValueError(
                f"Only '[MASK] or skill B' rows are supported: {mask_side!r}"
            )
        mask_answer = row["skill_a"]
        or_skill = row["skill_b"]

        prepared.append(
            {
                "row_idx": row["row_idx"],
                "file_path": row["file_path"],
                "preprocessed_file_path": relative,
                "source_csv_row": int(row["source_csv_row"]),
                "mask_answer": mask_answer,
                "or_skill": or_skill,
                "masked_sentence": row["masked_sentence"],
                "bert_pred_top5": "",
                "skill2vec_pred_top5": "",
                "condprob_pred_top5": "",
            }
        )

    return prepared


def _month_from_relative_path(relative_path: str) -> str:
    parts = Path(relative_path).parts
    matches = [part for part in parts if re.fullmatch(r"20\d{2}-(?:0[1-9]|1[0-2])", part)]
    if len(matches) != 1:
        raise ValueError(f"Cannot identify one YYYY-MM component: {relative_path}")
    return matches[0]


def _bert_files_by_month(bert_prediction_glob: str) -> dict[str, Path]:
    by_month: dict[str, Path] = {}
    for value in sorted(glob.glob(bert_prediction_glob)):
        path = Path(value)
        match = _BERT_MONTH_RE.search(path.name)
        if match is None:
            continue
        month = match.group(1)
        if month in by_month:
            raise ValueError(f"Multiple BERT prediction files for {month}")
        by_month[month] = path
    return by_month


def _same_scalar(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return str(left) == str(right)


def attach_bert_top5(
    rows: list[dict[str, Any]],
    bert_prediction_glob: str,
    chunk_size: int,
) -> None:
    """Read BERT Top-5 and join by preprocessed file plus its row position.

    BERT output preserves source-file order. Metadata is checked after the
    positional join so a shifted or mismatched row fails loudly.
    """
    if not rows:
        return

    target_by_location: dict[tuple[str, int], int] = {}
    for index, row in enumerate(rows):
        key = (row["preprocessed_file_path"], int(row["source_csv_row"]))
        if key in target_by_location:
            raise ValueError(f"Duplicate extracted source location: {key}")
        target_by_location[key] = index

    required_months = {
        _month_from_relative_path(row["preprocessed_file_path"]) for row in rows
    }
    files_by_month = _bert_files_by_month(bert_prediction_glob)
    missing_months = sorted(required_months - files_by_month.keys())
    if missing_months:
        raise FileNotFoundError(f"Missing BERT prediction months: {missing_months}")

    target_paths = {row["preprocessed_file_path"] for row in rows}
    path_offsets: defaultdict[str, int] = defaultdict(int)
    assigned: set[int] = set()
    usecols = [
        "preprocessed_file_path",
        "row_idx",
        "true_skill",
        "masked_sentence",
        "file_path",
        "pred_top5",
    ]

    for month in sorted(required_months):
        prediction_path = files_by_month[month]
        for chunk in pd.read_csv(prediction_path, usecols=usecols, chunksize=chunk_size):
            paths = chunk["preprocessed_file_path"].astype(str)
            within_chunk = chunk.groupby("preprocessed_file_path", sort=False).cumcount()
            starts = paths.map(dict(path_offsets)).fillna(0).astype("int64")
            chunk = chunk.copy()
            chunk["source_csv_row_position"] = within_chunk + starts

            counts = paths.value_counts(sort=False)
            for relative_path, count in counts.items():
                path_offsets[str(relative_path)] += int(count)

            relevant = chunk.loc[paths.isin(target_paths)]
            for record in relevant.itertuples(index=False):
                relative = str(record.preprocessed_file_path)
                location = (relative, int(record.source_csv_row_position))
                target_index = target_by_location.get(location)
                if target_index is None:
                    continue

                target = rows[target_index]
                checks = {
                    "row_idx": (record.row_idx, target["row_idx"]),
                    "true_skill": (record.true_skill, target["mask_answer"]),
                    "masked_sentence": (record.masked_sentence, target["masked_sentence"]),
                    "file_path": (record.file_path, target["file_path"]),
                }
                mismatched = [
                    name for name, (left, right) in checks.items() if not _same_scalar(left, right)
                ]
                if mismatched:
                    raise ValueError(
                        f"BERT row mismatch at {location}; fields={mismatched}"
                    )
                if target_index in assigned:
                    raise ValueError(f"BERT prediction assigned twice at {location}")

                target["bert_pred_top5"] = str(record.pred_top5)
                assigned.add(target_index)

    missing_indices = sorted(set(range(len(rows))) - assigned)
    if missing_indices:
        examples = [
            (
                rows[index]["preprocessed_file_path"],
                rows[index]["source_csv_row"],
            )
            for index in missing_indices[:10]
        ]
        raise ValueError(
            f"Missing BERT predictions for {len(missing_indices):,} rows; "
            f"examples={examples}"
        )


def attach_skill2vec_top5(
    rows: list[dict[str, Any]],
    model_path: Path,
    token_map_path: Path,
    top_k: int,
) -> int:
    """Predict cosine-neighbor Top-5 from the visible OR-side skill."""
    from model_tests.test_skill2vec import (
        load_keyed_vectors,
        normalize_dataset_skill,
        precompute_neighbors,
        resolve_token_mapping,
    )

    model_path = model_path.expanduser().resolve()
    token_map_path = token_map_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Skill2Vec model not found: {model_path}")
    if not token_map_path.is_file():
        raise FileNotFoundError(f"Skill2Vec token map not found: {token_map_path}")

    vectors = load_keyed_vectors(model_path)
    mapping = resolve_token_mapping(model_path, token_map_path, vectors, "_")
    lookup, _, _ = precompute_neighbors(vectors, mapping, top_k)

    missing = 0
    for row in rows:
        query = normalize_dataset_skill(row["or_skill"])
        prediction = lookup.get(query) if query is not None else None
        if prediction is None:
            missing += 1
            row["skill2vec_pred_top5"] = ""
        else:
            row["skill2vec_pred_top5"] = prediction[1]
    return missing


def attach_condprob_top5(
    rows: list[dict[str, Any]],
    conditional_table: Path,
    top_k: int,
) -> int:
    """Predict conditional-probability Top-5 from the visible OR-side skill."""
    from model_tests.test_conditional_prob import load_conditional_table, normalize_skill

    conditional_table = conditional_table.expanduser().resolve()
    if not conditional_table.is_file():
        raise FileNotFoundError(f"Conditional table not found: {conditional_table}")

    lookup, _, _ = load_conditional_table(conditional_table, top_k)
    missing = 0
    for row in rows:
        query = normalize_skill(row["or_skill"])
        prediction = lookup.get(query) if query is not None else None
        if prediction is None:
            missing += 1
            row["condprob_pred_top5"] = ""
        else:
            row["condprob_pred_top5"] = prediction[1]
    return missing


def attach_all_predictions(
    extracted_rows: list[dict[str, Any]],
    test_root: Path,
    bert_prediction_glob: str,
    skill2vec_model: Path,
    skill2vec_token_map: Path,
    conditional_table: Path,
    chunk_size: int,
    top_k: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = prepare_requested_columns(extracted_rows, test_root)
    attach_bert_top5(rows, bert_prediction_glob, chunk_size)
    w2v_missing = attach_skill2vec_top5(
        rows,
        skill2vec_model,
        skill2vec_token_map,
        top_k,
    )
    cond_missing = attach_condprob_top5(rows, conditional_table, top_k)
    return rows, {
        "skill2vec_missing_queries": w2v_missing,
        "condprob_missing_queries": cond_missing,
    }
