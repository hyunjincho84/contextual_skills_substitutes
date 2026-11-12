#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Attention-weighted semantic validation with a BERT encoder.

The default inputs are the monthly BERT, Skill2Vec, and conditional-probability
predictions below ``predictions_clean``.  For each prediction, this script:

1. fills the masked sentence with the true and substituted skills;
2. builds the same left/right context for both sentences;
3. aligns identical left and right context token IDs, while mean-pooling each
   possibly multi-token skill span into one comparison unit; and
4. weights the final-four-layer cosine similarities with incoming attention to
   the first token of the original skill span.

The alignment in steps 2-3 intentionally matches ``scoring_other_models.py``.
For Hugging Face attentions shaped ``[batch, head, query, key]``, the weight for
token i and anchor k is ``attention[:, :, i, k]``, averaged over all layers and
heads.  The final score is a weighted sum, without re-normalizing the attention
weights, matching the definition used by ``scoring_gpt.py``.

Outputs are written to::

    <output-root>/<baseline>/<year>/
        sv_summary_bert_full_<baseline>_<YYYY-MM>.csv.gz

Every output contains exactly ``row_idx``, ``masked_sentence``, ``pred_top1``,
``pred_top5``, ``pred_top5_probs``, ``truth``, ``substitution``, and the judge's
actual first-skill-token ``anchor_text``, and score column (``sv_bert`` by
default), in that order.
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

CACHE_DIR = os.environ.get("HF_CACHE_DIR", "/home/jovyan/LEM_data2/hf_cache")
os.environ.setdefault("HF_HOME", CACHE_DIR)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(CACHE_DIR, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(CACHE_DIR, "transformers"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402


BASE_DATA_DIR = os.environ.get(
    "BASE_DATA_DIR", "/home/jovyan/LEM_data3/hyunjincho/LEM"
)
DEFAULT_INPUT_ROOT = os.path.join(BASE_DATA_DIR, "predictions_clean")
DEFAULT_MODEL_NAME = os.environ.get("BERT_BASE_MODEL", "bert-base-uncased")
DEFAULT_JUDGE_NAME = os.environ.get("BERT_JUDGE_NAME", "bert")
DEFAULT_OUTPUT_ROOT = os.path.join(BASE_DATA_DIR, "evaluation_sv", "bert")
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAMES = ("bert", "w2v", "conditional")
OUTPUT_BASE_COLUMNS = (
    "row_idx",
    "masked_sentence",
    "pred_top1",
    "pred_top5",
    "pred_top5_probs",
    "truth",
    "substitution",
)
PREDICTION_SUBDIRS = {
    "bert": os.path.join("bert_pred", "pred"),
    "w2v": os.path.join("skill2vec_pred", "pred"),
    "conditional": os.path.join("condprob_pred", "pred"),
}
torch.set_float32_matmul_precision("high")


def output_columns(judge_name: str) -> List[str]:
    return [*OUTPUT_BASE_COLUMNS, "anchor_text", f"sv_{judge_name}"]


def selected_models(value: str) -> Set[str]:
    """Parse ``all`` or a comma-separated baseline list."""
    if value.strip().lower() == "all":
        return set(MODEL_NAMES)
    selected = {part.strip().lower() for part in value.split(",") if part.strip()}
    unknown = sorted(selected.difference(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown baseline names: {unknown}")
    if not selected:
        raise ValueError("No baselines selected")
    return selected


def list_monthly_files(base_dir: str, years: Sequence[str]) -> Dict[str, str]:
    """Return one prediction file per YYYY-MM key."""
    if not os.path.isdir(base_dir):
        return {}
    year_filter = {str(year) for year in years} if years else None
    year_dirs = sorted(
        entry
        for entry in os.listdir(base_dir)
        if entry.isdigit() and (year_filter is None or entry in year_filter)
    )
    result: Dict[str, str] = {}
    for year in year_dirs:
        year_dir = os.path.join(base_dir, year)
        paths = sorted(
            glob.glob(os.path.join(year_dir, "predictions_*.csv.gz"))
            + glob.glob(os.path.join(year_dir, "predictions_*.csv"))
        )
        for path in paths:
            match = re.search(r"(\d{4}-\d{2})", os.path.basename(path))
            if match is None:
                continue
            month = match.group(1)
            if month in result:
                raise ValueError(
                    f"Duplicate files for {month}: {result[month]} and {path}"
                )
            result[month] = path
    return result


def output_path(
    root: str,
    judge_name: str,
    model_name: str,
    source_path: str,
    debug: bool,
) -> str:
    """Build the standard judge/baseline output path."""
    matches = re.findall(r"(\d{4}-\d{2})", os.path.basename(source_path))
    month = matches[0] if matches else "debug"
    if debug:
        directory = os.path.join(root, model_name, "debug")
        filename = f"sv_summary_{judge_name}_debug_{model_name}.csv"
    else:
        year = month[:4] if re.fullmatch(r"\d{4}-\d{2}", month) else "unknown"
        directory = os.path.join(root, model_name, year)
        filename = f"sv_summary_{judge_name}_full_{model_name}_{month}.csv.gz"
    return os.path.join(directory, filename)


@dataclass(frozen=True)
class FilledWindow:
    """A model input plus token references for its shared context."""

    text: str
    left_reference_ids: Tuple[int, ...]
    right_reference_ids: Tuple[int, ...]


def _token_ids(text: str, tokenizer, add_special_tokens: bool) -> List[int]:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    ids = encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _encoded_length(text: str, tokenizer) -> int:
    return len(_token_ids(text, tokenizer, add_special_tokens=True))


def build_aligned_windows(
    masked_sentence: str,
    truth: str,
    substitute: str,
    tokenizer,
    window_size: int,
    max_len: int,
) -> Tuple[FilledWindow, FilledWindow]:
    """Build two inputs with exactly the same left and right text context.

    This is kept identical to the LLaMA/Qwen/Gemma alignment implementation in
    ``scoring_other_models.py``.
    """
    masked_sentence = str(masked_sentence)
    if masked_sentence.count("[MASK]") != 1:
        raise ValueError("masked_sentence must contain exactly one [MASK]")
    prefix, _, suffix = masked_sentence.partition("[MASK]")
    truth = str(truth).strip()
    substitute = str(substitute).strip()
    if not truth or not substitute:
        raise ValueError("truth or substitute is empty")

    prefix_ids = _token_ids(prefix, tokenizer, add_special_tokens=False)
    suffix_ids = _token_ids(suffix, tokenizer, add_special_tokens=False)
    context_limit = max(0, min(window_size, max_len))

    while True:
        if context_limit <= 0:
            left_text = ""
            right_text = ""
        else:
            left_text = (
                prefix
                if context_limit >= len(prefix_ids)
                else tokenizer.decode(
                    prefix_ids[-context_limit:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
            right_text = (
                suffix
                if context_limit >= len(suffix_ids)
                else tokenizer.decode(
                    suffix_ids[:context_limit],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )

        original_text = left_text + truth + right_text
        changed_text = left_text + substitute + right_text
        longest = max(
            _encoded_length(original_text, tokenizer),
            _encoded_length(changed_text, tokenizer),
        )
        if longest <= max_len:
            left_reference_ids = tuple(
                _token_ids(left_text, tokenizer, add_special_tokens=True)
            )
            right_reference_ids = tuple(
                _token_ids(right_text, tokenizer, add_special_tokens=True)
            )
            return (
                FilledWindow(
                    text=original_text,
                    left_reference_ids=left_reference_ids,
                    right_reference_ids=right_reference_ids,
                ),
                FilledWindow(
                    text=changed_text,
                    left_reference_ids=left_reference_ids,
                    right_reference_ids=right_reference_ids,
                ),
            )
        if context_limit == 0:
            raise ValueError(
                "inserted skill exceeds max_len even without surrounding context"
            )
        overflow = longest - max_len
        context_limit = max(0, context_limit - max(1, (overflow + 1) // 2))


def _span_and_context_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    left_reference_ids: Sequence[int],
    right_reference_ids: Sequence[int],
) -> Tuple[List[int], List[int], List[int]]:
    """Locate the changed span by matching the shared context from both ends."""
    active = [
        index
        for index, value in enumerate(attention_mask.tolist())
        if int(value) != 0
    ]
    active_ids = [int(input_ids[index]) for index in active]

    left_count = 0
    for token_id, reference_id in zip(active_ids, left_reference_ids):
        if token_id != reference_id:
            break
        left_count += 1

    right_count = 0
    available = len(active) - left_count
    for token_id, reference_id in zip(
        reversed(active_ids), reversed(right_reference_ids)
    ):
        if right_count >= available or token_id != reference_id:
            break
        right_count += 1

    span_end = len(active) - right_count if right_count else len(active)
    span = active[left_count:span_end]
    if not span:
        raise ValueError("tokenizer produced no changed tokens for the inserted skill")
    return active[:left_count], span, active[span_end:]


def _matching_context_pairs(
    original_ids: torch.Tensor,
    substitute_ids: torch.Tensor,
    original_left: Sequence[int],
    substitute_left: Sequence[int],
    original_right: Sequence[int],
    substitute_right: Sequence[int],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Align identical context token IDs without aligning unequal skill spans."""
    left_pairs: List[Tuple[int, int]] = []
    for original_index, substitute_index in zip(original_left, substitute_left):
        if int(original_ids[original_index]) != int(
            substitute_ids[substitute_index]
        ):
            break
        left_pairs.append((original_index, substitute_index))

    reversed_right: List[Tuple[int, int]] = []
    for original_index, substitute_index in zip(
        reversed(original_right), reversed(substitute_right)
    ):
        if int(original_ids[original_index]) != int(
            substitute_ids[substitute_index]
        ):
            break
        reversed_right.append((original_index, substitute_index))
    return left_pairs, list(reversed(reversed_right))


def _attention_weighted_row_score(
    original_layers: Sequence[torch.Tensor],
    substitute_layers: Sequence[torch.Tensor],
    original_attentions: Sequence[torch.Tensor],
    batch_index: int,
    context_pairs: Sequence[Tuple[int, int]],
    original_span: Sequence[int],
    substitute_span: Sequence[int],
) -> float:
    """Score one aligned pair using incoming attention to the original anchor.

    Context tokens are compared one-to-one.  The original and substitute skill
    spans are separately mean-pooled into one cosine unit.  That skill unit is
    weighted by the mean w_{p,k} over all original-skill query tokens p, while
    key k remains fixed to the first original-skill token.
    """
    if not original_span or not substitute_span:
        raise ValueError("empty original or substitute skill span")
    if min(len(original_layers), len(substitute_layers)) < 4:
        raise RuntimeError("judge model has fewer than four hidden layers")
    if not original_attentions:
        raise RuntimeError("judge model did not return attentions")

    device = original_layers[0].device
    unit_count = len(context_pairs) + 1
    dot = torch.zeros(unit_count, device=device, dtype=torch.float32)
    norm_original = torch.zeros_like(dot)
    norm_substitute = torch.zeros_like(dot)

    if context_pairs:
        original_positions = torch.tensor(
            [pair[0] for pair in context_pairs], device=device, dtype=torch.long
        )
        substitute_positions = torch.tensor(
            [pair[1] for pair in context_pairs], device=device, dtype=torch.long
        )
    original_span_positions = torch.tensor(
        original_span, device=device, dtype=torch.long
    )
    substitute_span_positions = torch.tensor(
        substitute_span, device=device, dtype=torch.long
    )

    # Accumulating dot products and squared norms is equivalent to cosine over
    # the concatenation of the final four hidden layers, but uses less memory.
    for original_layer, substitute_layer in zip(
        original_layers[-4:], substitute_layers[-4:]
    ):
        if context_pairs:
            original_context = original_layer[
                batch_index, original_positions, :
            ].float()
            substitute_context = substitute_layer[
                batch_index, substitute_positions, :
            ].float()
            dot[:-1] += (original_context * substitute_context).sum(dim=-1)
            norm_original[:-1] += original_context.pow(2).sum(dim=-1)
            norm_substitute[:-1] += substitute_context.pow(2).sum(dim=-1)

        original_pooled = original_layer[
            batch_index, original_span_positions, :
        ].float().mean(dim=0)
        substitute_pooled = substitute_layer[
            batch_index, substitute_span_positions, :
        ].float().mean(dim=0)
        dot[-1] += (original_pooled * substitute_pooled).sum()
        norm_original[-1] += original_pooled.pow(2).sum()
        norm_substitute[-1] += substitute_pooled.pow(2).sum()

    cosine = dot / (
        norm_original.sqrt() * norm_substitute.sqrt()
    ).clamp_min(1e-12)

    anchor = int(original_span[0])
    weights = torch.zeros(unit_count, device=device, dtype=torch.float32)
    for attention in original_attentions:
        # HF attention dimensions are [batch, head, query, key].  Therefore
        # attention[..., i, anchor] is w_{i,k}, not attention[..., anchor, i].
        if context_pairs:
            weights[:-1] += attention[
                batch_index, :, original_positions, anchor
            ].float().mean(dim=0)
        # Match the mean-pooled skill representation with the mean attention
        # from every original-skill query token p to the fixed first-token key k.
        weights[-1] += attention[
            batch_index, :, original_span_positions, anchor
        ].float().mean()
    weights /= float(len(original_attentions))
    weights.clamp_min_(0.0)

    # Do not normalize by weights.sum(): scoring_gpt.py and the paper use a sum.
    return float((weights * cosine).sum().item())


@torch.no_grad()
def compute_sv_batch(
    originals: Sequence[FilledWindow],
    substitutes: Sequence[FilledWindow],
    tokenizer,
    model,
    max_len: int,
    sent_batch: int,
    device: str,
) -> Tuple[List[float], List[str]]:
    """Return scores and the exact original-skill anchor WordPiece tokens."""
    if len(originals) != len(substitutes):
        raise ValueError("original/substitute batch lengths differ")

    scores: List[float] = []
    anchor_texts: List[str] = []
    batch_size = max(1, sent_batch)
    row_offset = 0
    device_type = torch.device(device).type
    use_amp = device_type == "cuda"

    while row_offset < len(originals):
        current = min(batch_size, len(originals) - row_offset)
        original_batch = originals[row_offset : row_offset + current]
        substitute_batch = substitutes[row_offset : row_offset + current]
        try:
            original_encoding = tokenizer(
                [item.text for item in original_batch],
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            substitute_encoding = tokenizer(
                [item.text for item in substitute_batch],
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            original_ids = original_encoding["input_ids"].detach().cpu()
            substitute_ids = substitute_encoding["input_ids"].detach().cpu()
            original_masks = original_encoding["attention_mask"].detach().cpu()
            substitute_masks = substitute_encoding["attention_mask"].detach().cpu()
            if (
                original_ids.shape[1] > max_len
                or substitute_ids.shape[1] > max_len
            ):
                raise ValueError("aligned context window exceeds max_len")

            original_encoding = original_encoding.to(device)
            substitute_encoding = substitute_encoding.to(device)
            with torch.autocast(
                device_type=device_type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                original_output = model(
                    **original_encoding,
                    output_hidden_states=True,
                    output_attentions=True,
                    return_dict=True,
                )
                substitute_output = model(
                    **substitute_encoding,
                    output_hidden_states=True,
                    output_attentions=False,
                    return_dict=True,
                )

            original_layers = original_output.hidden_states
            substitute_layers = substitute_output.hidden_states
            original_attentions = original_output.attentions
            if original_layers is None or substitute_layers is None:
                raise RuntimeError("judge model did not return hidden states")
            if original_attentions is None:
                raise RuntimeError(
                    "judge model did not return attentions; eager attention is required"
                )

            batch_scores: List[float] = []
            batch_anchor_texts: List[str] = []
            for batch_index, (original_item, substitute_item) in enumerate(
                zip(original_batch, substitute_batch)
            ):
                original_left, original_span, original_right = (
                    _span_and_context_positions(
                        original_ids[batch_index],
                        original_masks[batch_index],
                        original_item.left_reference_ids,
                        original_item.right_reference_ids,
                    )
                )
                substitute_left, substitute_span, substitute_right = (
                    _span_and_context_positions(
                        substitute_ids[batch_index],
                        substitute_masks[batch_index],
                        substitute_item.left_reference_ids,
                        substitute_item.right_reference_ids,
                    )
                )
                left_pairs, right_pairs = _matching_context_pairs(
                    original_ids[batch_index],
                    substitute_ids[batch_index],
                    original_left,
                    substitute_left,
                    original_right,
                    substitute_right,
                )
                # k is fixed to the first token of the original skill span.
                anchor_token_id = int(
                    original_ids[batch_index, int(original_span[0])]
                )
                anchor_token = tokenizer.convert_ids_to_tokens([anchor_token_id])[0]
                batch_anchor_texts.append(str(anchor_token))
                batch_scores.append(
                    _attention_weighted_row_score(
                        original_layers,
                        substitute_layers,
                        original_attentions,
                        batch_index,
                        left_pairs + right_pairs,
                        original_span,
                        substitute_span,
                    )
                )
            scores.extend(batch_scores)
            anchor_texts.extend(batch_anchor_texts)

            del original_encoding, substitute_encoding
            del original_output, substitute_output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            row_offset += current
        except torch.cuda.OutOfMemoryError:
            # Drop every reference that may retain the failed large batch before
            # retrying.  empty_cache alone cannot release live tensors.
            original_encoding = None
            substitute_encoding = None
            original_output = None
            substitute_output = None
            original_layers = None
            substitute_layers = None
            original_attentions = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if current == 1:
                raise
            batch_size = max(1, current // 2)
            print(
                f"[OOM] sent-batch {current} -> {batch_size}; "
                f"retry row offset {row_offset}"
            )
    return scores, anchor_texts


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _pipe_candidates(value: object) -> List[str]:
    text = _clean_text(value)
    return [part.strip() for part in text.split("|") if part.strip()]


def choose_substitution(row: pd.Series, model_name: str) -> str:
    """Select the first candidate after explicitly excluding the true skill."""
    truth = _clean_text(row.get("truth"))
    candidates: List[str] = []
    if model_name == "bert":
        candidates.append(_clean_text(row.get("substitution")))
    candidates.append(_clean_text(row.get("pred_top1")))
    candidates.extend(_pipe_candidates(row.get("pred_top5")))

    seen: Set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if candidate != truth:
            return candidate
    return ""


def score_frame(
    model_name: str,
    frame: pd.DataFrame,
    tokenizer,
    model,
    judge_name: str,
    window_size: int,
    max_len: int,
    sent_batch: int,
    device: str,
    strict: bool,
) -> pd.DataFrame:
    """Score one dataframe and return the fixed output schema."""
    required = set(OUTPUT_BASE_COLUMNS).difference({"substitution"})
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"[{model_name}] missing columns: {missing}")
    if not {"substitution", "pred_top1", "pred_top5"}.intersection(frame.columns):
        raise ValueError(
            f"[{model_name}] needs substitution, pred_top1, or pred_top5"
        )

    frame = frame.copy()
    originals: List[FilledWindow] = []
    substitutes: List[FilledWindow] = []
    valid_indices: List[int] = []
    selected_substitutes: List[str] = []
    errors: List[str] = []

    for position, (_, row) in enumerate(frame.iterrows()):
        truth = _clean_text(row.get("truth"))
        masked_sentence = _clean_text(row.get("masked_sentence"))
        substitute = choose_substitution(row, model_name)
        selected_substitutes.append(substitute)
        try:
            if not truth:
                raise ValueError("truth is empty")
            if not substitute:
                raise ValueError("no truth-excluded substitute candidate")
            original, changed = build_aligned_windows(
                masked_sentence,
                truth,
                substitute,
                tokenizer,
                window_size,
                max_len,
            )
            originals.append(original)
            substitutes.append(changed)
            valid_indices.append(position)
            errors.append("")
        except (TypeError, ValueError) as exc:
            if strict:
                raise ValueError(f"[{model_name}] row {position}: {exc}") from exc
            errors.append(str(exc))

    score_values = [float("nan")] * len(frame)
    anchor_values = [""] * len(frame)
    if originals:
        valid_scores, valid_anchor_texts = compute_sv_batch(
            originals,
            substitutes,
            tokenizer,
            model,
            max_len,
            sent_batch,
            device,
        )
        for position, score, anchor_text in zip(
            valid_indices, valid_scores, valid_anchor_texts
        ):
            score_values[position] = score
            anchor_values[position] = anchor_text

    score_column = f"sv_{judge_name}"
    frame["substitution"] = selected_substitutes
    frame["anchor_text"] = anchor_values
    frame[score_column] = score_values
    result = frame.loc[:, output_columns(judge_name)].copy()
    result.attrs["failure_count"] = sum(bool(error) for error in errors)
    return result


def _count_rows(path: str, column: str) -> int:
    total = 0
    for chunk in pd.read_csv(path, usecols=[column], chunksize=100_000):
        total += len(chunk)
    return total


def validate_resume_output(
    source: str,
    destination: str,
    judge_name: str,
) -> Tuple[bool, str]:
    """Validate a completed atomic output before skipping it."""
    if not os.path.isfile(destination):
        return False, "output does not exist"
    score_column = f"sv_{judge_name}"
    try:
        source_header = pd.read_csv(source, nrows=0)
        output_header = pd.read_csv(destination, nrows=0)
        expected = output_columns(judge_name)
        if list(output_header.columns) != expected:
            return False, (
                f"schema/order mismatch: expected={expected}, "
                f"actual={list(output_header.columns)}"
            )

        identity_column = str(source_header.columns[0])
        source_rows = _count_rows(source, identity_column)
        output_rows = _count_rows(destination, score_column)
        if source_rows != output_rows:
            return False, f"row-count mismatch: input={source_rows}, output={output_rows}"

    except Exception as exc:
        return False, f"unreadable output: {exc}"
    return True, f"validated {output_rows:,} rows"


def write_frame_atomically(frame: pd.DataFrame, path: str) -> None:
    """Atomically write a debug dataframe."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    try:
        frame.to_csv(
            temporary,
            index=False,
            compression="gzip" if path.endswith(".gz") else None,
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def score_file_atomically(
    model_name: str,
    source: str,
    destination: str,
    tokenizer,
    model,
    judge_name: str,
    window_size: int,
    max_len: int,
    sent_batch: int,
    chunk_size: int,
    device: str,
    strict: bool,
) -> Tuple[int, int]:
    """Stream one source file into a temporary gzip file, then replace."""
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    temporary = f"{destination}.tmp.{os.getpid()}"
    total_rows = 0
    failed_rows = 0
    wrote_header = False
    try:
        for frame in pd.read_csv(source, chunksize=chunk_size):
            scored = score_frame(
                model_name,
                frame,
                tokenizer,
                model,
                judge_name,
                window_size,
                max_len,
                sent_batch,
                device,
                strict,
            )
            total_rows += len(scored)
            failed_rows += int(scored.attrs.get("failure_count", 0))
            scored.to_csv(
                temporary,
                index=False,
                compression="gzip",
                mode="at" if wrote_header else "wt",
                header=not wrote_header,
            )
            wrote_header = True

        if not wrote_header:
            empty = pd.read_csv(source, nrows=0)
            scored = score_frame(
                model_name,
                empty,
                tokenizer,
                model,
                judge_name,
                window_size,
                max_len,
                sent_batch,
                device,
                strict,
            )
            scored.to_csv(temporary, index=False, compression="gzip")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return total_rows, failed_rows


def infer_debug_model(path: str) -> Optional[str]:
    normalized = path.replace(os.sep, "/").lower()
    if "bert_pred" in normalized:
        return "bert"
    if "skill2vec" in normalized or "w2v" in normalized:
        return "w2v"
    if "condprob" in normalized or "conditional" in normalized:
        return "conditional"
    return None


def build_tasks(
    files_by_model: Dict[str, Dict[str, str]],
    months: Sequence[str],
    root: str,
    judge_name: str,
) -> List[Tuple[str, str, str, str]]:
    tasks: List[Tuple[str, str, str, str]] = []
    names = [name for name in MODEL_NAMES if name in files_by_model]
    for month_index, month in enumerate(months):
        shift = month_index % len(names)
        for model_name in names[shift:] + names[:shift]:
            source = files_by_model[model_name][month]
            destination = output_path(
                root, judge_name, model_name, source, debug=False
            )
            tasks.append((model_name, month, source, destination))
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score predictions_clean with attention-weighted BERT SV."
    )
    parser.add_argument(
        "--start-from",
        default="all",
        help="all or comma-separated: bert,w2v,conditional",
    )
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--bert-pred-dir", default=None)
    parser.add_argument("--w2v-pred-dir", default=None)
    parser.add_argument("--cond-pred-dir", default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--judge-name", default=DEFAULT_JUDGE_NAME)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-dir", default=CACHE_DIR)
    parser.add_argument("--years", nargs="*", default=[])
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--sent-batch", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=2_000)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--task-rank", type=int, default=0)
    parser.add_argument("--task-world-size", type=int, default=1)
    parser.add_argument("--debug-file", default=None)
    parser.add_argument("--debug-model", choices=MODEL_NAMES, default=None)
    parser.add_argument("--debug-n", type=int, default=10)
    parser.add_argument("--debug-output", default=None)
    args = parser.parse_args()

    if args.window_size < 0 or args.max_len < 1:
        parser.error("window-size and max-len must be valid")
    if args.sent_batch < 1 or args.chunk_size < 1 or args.debug_n < 1:
        parser.error("sent-batch, chunk-size, and debug-n must be positive")
    if args.task_world_size < 1:
        parser.error("--task-world-size must be positive")
    if not 0 <= args.task_rank < args.task_world_size:
        parser.error("--task-rank must be within task-world-size")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.judge_name):
        parser.error("--judge-name may contain only letters, digits, _, ., and -")
    return args


def load_model(args: argparse.Namespace):
    """Load BERT with eager attention so output_attentions is reliable."""
    os.makedirs(args.cache_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        use_fast=True,
    )
    tokenizer.padding_side = "right"

    load_kwargs = {
        "cache_dir": args.cache_dir,
        "local_files_only": args.local_files_only,
    }
    try:
        model = AutoModel.from_pretrained(
            args.model_name,
            attn_implementation="eager",
            **load_kwargs,
        )
    except TypeError:
        # Compatibility with older transformers versions.
        model = AutoModel.from_pretrained(args.model_name, **load_kwargs)
        if hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation("eager")
        elif hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"

    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions is not None and args.max_len > int(max_positions):
        raise ValueError(
            f"--max-len {args.max_len} exceeds model limit {max_positions}"
        )
    model = model.to(args.device).eval()
    return tokenizer, model


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        args.device = DEFAULT_DEVICE
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    baselines = selected_models(args.start_from)
    root = os.path.abspath(args.output_root)
    print(f"Judge: {args.judge_name} ({args.model_name})")
    print(f"Device: {args.device}")
    print(f"Baselines: {sorted(baselines)}")
    print(f"Input root: {os.path.abspath(args.input_root)}")
    print(f"Output root: {root}")
    print(
        "Alignment: identical LLaMA-style left/right context token IDs plus "
        "one mean-pooled skill-span cosine unit weighted by mean_p(w_p,k)"
    )
    print(
        "Attention: mean over BERT layers/heads of A[query=i, key=anchor], "
        "weighted sum without renormalization"
    )
    print(
        "Precision: model forward uses AMP on CUDA; pooling, dot/norm, cosine, "
        "attention aggregation, and final scoring use FP32"
    )
    tokenizer, model = load_model(args)

    if args.debug_file:
        model_name = args.debug_model or infer_debug_model(args.debug_file)
        if model_name is None:
            raise ValueError("Could not infer --debug-model")
        frame = pd.read_csv(args.debug_file, nrows=args.debug_n)
        if frame.empty:
            raise ValueError(f"Empty debug input: {args.debug_file}")
        scored = score_frame(
            model_name,
            frame,
            tokenizer,
            model,
            args.judge_name,
            args.window_size,
            args.max_len,
            args.sent_batch,
            args.device,
            args.strict,
        )
        destination = args.debug_output or output_path(
            root, args.judge_name, model_name, args.debug_file, debug=True
        )
        write_frame_atomically(scored, destination)
        failures = int(scored.attrs.get("failure_count", 0))
        print(
            f"[{model_name}] saved: {destination} "
            f"(rows={len(scored):,}, failed={failures:,})"
        )
        return

    directories = {
        "bert": args.bert_pred_dir
        or os.path.join(args.input_root, PREDICTION_SUBDIRS["bert"]),
        "w2v": args.w2v_pred_dir
        or os.path.join(args.input_root, PREDICTION_SUBDIRS["w2v"]),
        "conditional": args.cond_pred_dir
        or os.path.join(args.input_root, PREDICTION_SUBDIRS["conditional"]),
    }
    files_by_model: Dict[str, Dict[str, str]] = {}
    for model_name in MODEL_NAMES:
        if model_name not in baselines:
            continue
        monthly = list_monthly_files(directories[model_name], args.years)
        if not monthly:
            raise FileNotFoundError(
                f"[{model_name}] no prediction files in {directories[model_name]}"
            )
        files_by_model[model_name] = monthly

    common_months = sorted(
        set.intersection(*(set(files) for files in files_by_model.values()))
    )
    if not common_months:
        raise FileNotFoundError("No common months across selected baselines")
    all_tasks = build_tasks(
        files_by_model, common_months, root, args.judge_name
    )
    tasks = [
        task
        for index, task in enumerate(all_tasks)
        if index % args.task_world_size == args.task_rank
    ]
    print(
        f"Worker {args.task_rank}/{args.task_world_size}: "
        f"{len(tasks)} of {len(all_tasks)} tasks"
    )

    grand_rows = 0
    grand_failures = 0
    for model_name, month, source, destination in tqdm(
        tasks, desc=f"SV({args.judge_name})", unit="task"
    ):
        if args.resume:
            complete, reason = validate_resume_output(
                source, destination, args.judge_name
            )
            if complete:
                print(f"[{model_name}][{month}][RESUME] skip: {reason}")
                continue
            if os.path.exists(destination):
                print(f"[{model_name}][{month}][RESUME] recompute: {reason}")

        rows, failures = score_file_atomically(
            model_name,
            source,
            destination,
            tokenizer,
            model,
            args.judge_name,
            args.window_size,
            args.max_len,
            args.sent_batch,
            args.chunk_size,
            args.device,
            args.strict,
        )
        grand_rows += rows
        grand_failures += failures
        print(
            f"[{model_name}][{month}] saved: {destination} "
            f"(rows={rows:,}, failed={failures:,})"
        )

    print(
        f"Done: processed={grand_rows:,}, failed={grand_failures:,}, "
        f"worker={args.task_rank}/{args.task_world_size}"
    )


if __name__ == "__main__":
    main()
