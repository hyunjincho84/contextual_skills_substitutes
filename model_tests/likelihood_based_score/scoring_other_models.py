# -*- coding: utf-8 -*-
"""Unified span-aligned SV scorer for LLaMA, Qwen, and Gemma."""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

CACHE_DIR = os.environ.get("HF_CACHE_DIR", "/home/jovyan/LEM_data2/hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
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
DEFAULT_PREDICTION_DIRS = {
    "bert": os.path.join(BASE_DATA_DIR, "predictions", "bert_pred", "pred"),
    "w2v": os.path.join(BASE_DATA_DIR, "predictions", "skill2vec_pred", "pred"),
    "conditional": os.path.join(
        BASE_DATA_DIR, "predictions", "condprob_pred", "pred"
    ),
}
DEFAULT_CHECKPOINT = "meta-llama/Llama-3.2-3B"
DEFAULT_EVALUATION_ROOT = os.path.join(BASE_DATA_DIR, "evaluation_sv")
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
torch.set_float32_matmul_precision("high")


def output_columns(judge_name: str) -> List[str]:
    return [*OUTPUT_BASE_COLUMNS, f"sv_{judge_name}"]


def candidate_column(model_name: str) -> str:
    return "substitution" if model_name == "bert" else "pred_top1"


def infer_judge_name(checkpoint: str) -> str:
    name = checkpoint.lower()
    if "llama" in name:
        return "llama"
    if "qwen" in name:
        return "qwen25"
    if "gemma" in name:
        return "gemma"
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def selected_models(value: str) -> Set[str]:
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
        files = sorted(
            glob.glob(os.path.join(year_dir, "predictions_*.csv.gz"))
            + glob.glob(os.path.join(year_dir, "predictions_*.csv"))
        )
        for path in files:
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
    matches = re.findall(r"(\d{4}-\d{2})", os.path.basename(source_path))
    month = matches[0] if matches else "debug"
    if debug:
        directory = os.path.join(root, model_name, "debug")
        filename = f"sv_summary_{judge_name}_debug_{model_name}.csv"
    else:
        year = month[:4] if re.fullmatch(r"\d{4}-\d{2}", month) else "unknown"
        directory = os.path.join(root, model_name, year)
        filename = f"sv_summary_{judge_name}_full_{model_name}_{month}.csv.gz"
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, filename)


def write_atomically(frame: pd.DataFrame, path: str) -> None:
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


def validate_resume_output(
    path: str,
    input_frame: pd.DataFrame,
    judge_name: str,
) -> Tuple[bool, str]:
    if not os.path.isfile(path):
        return False, "output does not exist"
    score_column = f"sv_{judge_name}"
    expected_columns = output_columns(judge_name)
    try:
        header = pd.read_csv(path, nrows=0)
        actual_columns = list(header.columns)
        if actual_columns != expected_columns:
            return False, (
                f"schema/order mismatch: expected={expected_columns}, "
                f"actual={actual_columns}"
            )
        scores = pd.read_csv(path, usecols=[score_column])
    except Exception as exc:
        return False, f"unreadable output: {exc}"
    if len(scores) != len(input_frame):
        return False, (
            f"row-count mismatch: output={len(scores):,}, "
            f"input={len(input_frame):,}"
        )
    if pd.to_numeric(scores[score_column], errors="coerce").isna().any():
        return False, f"{score_column} contains invalid values"
    return True, f"validated {len(scores):,} rows"


@dataclass(frozen=True)
class FilledWindow:
    """A model input with transient token references for its shared context."""

    text: str
    left_reference_ids: Tuple[int, ...]
    right_reference_ids: Tuple[int, ...]


def _token_ids(
    text: str, tokenizer, add_special_tokens: bool
) -> List[int]:
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
    """Build two inputs with exactly the same left and right text context."""
    prefix, separator, suffix = str(masked_sentence).partition("[MASK]")
    if not separator:
        raise ValueError("masked_sentence does not contain [MASK]")
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
        raise ValueError(
            "tokenizer produced no changed tokens for the inserted skill"
        )
    return (
        active[:left_count],
        span,
        active[span_end:],
    )


def _matching_context_pairs(
    original_ids: torch.Tensor,
    substitute_ids: torch.Tensor,
    original_left: Sequence[int],
    substitute_left: Sequence[int],
    original_right: Sequence[int],
    substitute_right: Sequence[int],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
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


def _aligned_row_score(
    original_layers: Sequence[torch.Tensor],
    substitute_layers: Sequence[torch.Tensor],
    batch_index: int,
    context_pairs: Sequence[Tuple[int, int]],
    original_span: Sequence[int],
    substitute_span: Sequence[int],
) -> float:
    """Compute span-aligned cosine similarity with FP32 reductions.

    The model forward pass may return BF16/FP16 hidden states. Casting the
    selected vectors before pooling and reduction cannot recover precision
    already lost in the forward pass, but it avoids further low-precision
    rounding in means, dot products, norms, and the final cosine average.
    """
    device = original_layers[0].device
    unit_count = len(context_pairs) + 1
    dot = torch.zeros(unit_count, device=device, dtype=torch.float32)
    norm_original = torch.zeros_like(dot)
    norm_substitute = torch.zeros_like(dot)

    if context_pairs:
        original_positions = torch.tensor(
            [pair[0] for pair in context_pairs],
            device=device,
            dtype=torch.long,
        )
        substitute_positions = torch.tensor(
            [pair[1] for pair in context_pairs],
            device=device,
            dtype=torch.long,
        )
    original_span_positions = torch.tensor(
        original_span, device=device, dtype=torch.long
    )
    substitute_span_positions = torch.tensor(
        substitute_span, device=device, dtype=torch.long
    )

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
    return float(cosine.mean().item())


@torch.no_grad()
def compute_sv_batch(
    originals: Sequence[FilledWindow],
    substitutes: Sequence[FilledWindow],
    tokenizer,
    model,
    max_len: int,
    sent_batch: int,
    device: str,
) -> List[float]:
    scores: List[float] = []
    batch_size = max(1, sent_batch)
    row_offset = 0
    while row_offset < len(originals):
        current = min(batch_size, len(originals) - row_offset)
        original_batch = originals[row_offset:row_offset + current]
        substitute_batch = substitutes[row_offset:row_offset + current]
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
            original_output = model(
                **original_encoding,
                output_hidden_states=True,
                output_attentions=False,
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
            if original_layers is None or substitute_layers is None:
                raise RuntimeError("judge model did not return hidden states")
            if min(len(original_layers), len(substitute_layers)) < 4:
                raise RuntimeError("judge model has fewer than four hidden layers")

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
                scores.append(
                    _aligned_row_score(
                        original_layers,
                        substitute_layers,
                        batch_index,
                        left_pairs + right_pairs,
                        original_span,
                        substitute_span,
                    )
                )

            del original_encoding, substitute_encoding
            del original_output, substitute_output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            row_offset += current
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if current == 1:
                raise
            batch_size = max(1, current // 2)
            print(
                f"[OOM] sent-batch {current} -> {batch_size}; "
                f"retry row offset {row_offset}"
            )
    return scores


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
) -> pd.DataFrame:
    prediction_column = candidate_column(model_name)
    required = set(OUTPUT_BASE_COLUMNS).difference({"substitution"})
    required.add(prediction_column)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"[{model_name}] missing columns: {missing}")

    frame = frame.copy()
    frame["truth"] = frame["truth"].astype(str)
    frame["masked_sentence"] = frame["masked_sentence"].astype(str)
    frame[prediction_column] = frame[prediction_column].fillna("").astype(str)
    originals: List[FilledWindow] = []
    substitutes: List[FilledWindow] = []
    selected_substitutes: List[str] = []
    for row_index, row in frame.iterrows():
        truth = row["truth"].strip()
        substitute = row[prediction_column].strip()
        if not substitute or substitute == truth:
            raise ValueError(
                f"[{model_name}] row {row_index} has no truth-excluded substitute"
            )
        try:
            original, changed = build_aligned_windows(
                row["masked_sentence"],
                truth,
                substitute,
                tokenizer,
                window_size,
                max_len,
            )
            originals.append(original)
            substitutes.append(changed)
        except ValueError as exc:
            raise ValueError(f"[{model_name}] row {row_index}: {exc}") from exc
        selected_substitutes.append(substitute)

    scores = compute_sv_batch(
        originals,
        substitutes,
        tokenizer,
        model,
        max_len,
        sent_batch,
        device,
    )
    frame["substitution"] = selected_substitutes
    frame[f"sv_{judge_name}"] = scores
    return frame.loc[:, output_columns(judge_name)].copy()


def infer_debug_model(path: str) -> Optional[str]:
    normalized = path.replace(os.sep, "/").lower()
    filename = os.path.basename(path).lower()
    for name in MODEL_NAMES:
        if f"_{name}_" in filename:
            return name
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-from", default="all")
    parser.add_argument(
        "--bert_pred_dir", default=DEFAULT_PREDICTION_DIRS["bert"]
    )
    parser.add_argument("--w2v_pred_dir", default=DEFAULT_PREDICTION_DIRS["w2v"])
    parser.add_argument(
        "--cond_pred_dir", default=DEFAULT_PREDICTION_DIRS["conditional"]
    )
    parser.add_argument(
        "--judge-ckpt",
        "--judge_ckpt",
        dest="judge_ckpt",
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--judge-name", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--cache-dir", default=CACHE_DIR)
    parser.add_argument("--years", nargs="*", default=[])
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--sent-batch", type=int, default=4)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--task-rank", type=int, default=0)
    parser.add_argument("--task-world-size", type=int, default=1)
    parser.add_argument("--debug-file", default=None)
    parser.add_argument("--debug-model", choices=MODEL_NAMES, default=None)
    parser.add_argument("--debug-n", type=int, default=10)
    parser.add_argument("--debug-output", default=None)
    auth = parser.add_mutually_exclusive_group()
    auth.add_argument(
        "--use-auth-token", dest="use_auth_token", action="store_true"
    )
    auth.add_argument(
        "--no-auth-token", dest="use_auth_token", action="store_false"
    )
    parser.set_defaults(use_auth_token=True)
    args = parser.parse_args()
    if args.window_size < 0 or args.max_len < 1 or args.sent_batch < 1:
        parser.error("window-size, max-len, and sent-batch must be valid")
    if args.task_world_size < 1:
        parser.error("--task-world-size must be positive")
    if not 0 <= args.task_rank < args.task_world_size:
        parser.error("--task-rank must be within task-world-size")
    return args


def load_model(args: argparse.Namespace):
    environment_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    token = (environment_token or True) if args.use_auth_token else False
    tokenizer = AutoTokenizer.from_pretrained(
        args.judge_ckpt,
        token=token,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer has no pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModel.from_pretrained(
        args.judge_ckpt,
        torch_dtype="auto",
        token=token,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    ).to(args.device).eval()
    return tokenizer, model


def main() -> None:
    args = parse_args()
    baselines = selected_models(args.start_from)
    judge_name = args.judge_name or infer_judge_name(args.judge_ckpt)
    root = os.path.abspath(
        args.output_root
        or os.path.join(DEFAULT_EVALUATION_ROOT, judge_name)
    )
    os.makedirs(root, exist_ok=True)
    print(f"Judge: {judge_name} ({args.judge_ckpt})")
    print(f"Device: {args.device}")
    print(f"Baselines: {sorted(baselines)}")
    print(f"Output root: {root}")
    print(
        "SV aggregation: uniform mean over aligned context tokens "
        "and one mean-pooled skill-span unit"
    )
    print(
        "Precision: model forward uses checkpoint dtype; pooling, dot/norm, "
        "cosine, and final averaging use FP32"
    )
    print(f"Saved scoring columns: substitution, sv_{judge_name}")
    tokenizer, model = load_model(args)

    if args.debug_file:
        model_name = args.debug_model or infer_debug_model(args.debug_file)
        if model_name is None:
            raise ValueError("Could not infer --debug-model")
        frame = pd.read_csv(args.debug_file)
        if len(frame) > args.debug_n:
            frame = frame.sample(
                n=args.debug_n, random_state=args.random_seed
            ).reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"Empty debug input: {args.debug_file}")
        scored = score_frame(
            model_name,
            frame,
            tokenizer,
            model,
            judge_name,
            args.window_size,
            args.max_len,
            args.sent_batch,
            args.device,
        )
        destination = args.debug_output or output_path(
            root, judge_name, model_name, args.debug_file, debug=True
        )
        os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
        write_atomically(scored, destination)
        print(f"[{model_name}] saved: {destination} (rows={len(scored)})")
        return

    directories = {
        "bert": args.bert_pred_dir,
        "w2v": args.w2v_pred_dir,
        "conditional": args.cond_pred_dir,
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
    all_tasks = build_tasks(files_by_model, common_months, root, judge_name)
    tasks = [
        task
        for index, task in enumerate(all_tasks)
        if index % args.task_world_size == args.task_rank
    ]
    print(
        f"Worker {args.task_rank}/{args.task_world_size}: "
        f"{len(tasks)} of {len(all_tasks)} tasks"
    )
    for model_name, month, source, destination in tqdm(
        tasks, desc=f"SV({judge_name})", unit="task"
    ):
        frame = pd.read_csv(source)
        if frame.empty:
            print(f"[{model_name}][{month}] empty input; skip")
            continue
        if args.resume:
            complete, reason = validate_resume_output(
                destination, frame, judge_name
            )
            if complete:
                print(
                    f"[{model_name}][{month}][RESUME] skip: {reason}"
                )
                continue
            if os.path.exists(destination):
                print(
                    f"[{model_name}][{month}][RESUME] "
                    f"recompute: {reason}"
                )
        scored = score_frame(
            model_name,
            frame,
            tokenizer,
            model,
            judge_name,
            args.window_size,
            args.max_len,
            args.sent_batch,
            args.device,
        )
        write_atomically(scored, destination)
        print(f"[{model_name}] saved: {destination} (rows={len(scored)})")


if __name__ == "__main__":
    main()
