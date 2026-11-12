#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Score matched GPT and BERT substitutes with LLaMA, Gemma, and Qwen.

The semantic-validation implementation is shared with
``scoring_other_models.py``. Original and substituted sentences use identical
left/right token context, context tokens are aligned one-to-one, and each
possibly multi-token skill span is mean-pooled into one comparison unit. The
score is the uniform mean of final-four-layer cosine similarities over those
aligned units.

``gpt.csv`` is evaluated only with GPT candidates and ``bert.csv`` only with
BERT candidates. In both cases, the true skill is explicitly excluded before
the highest-ranked remaining candidate is selected.
"""

from __future__ import annotations

import argparse
import gc
import os
import random
from typing import Dict, Iterable, List, Sequence, Set, Tuple

try:
    from .scoring_other_models import (
        CACHE_DIR,
        build_aligned_windows,
        compute_sv_batch as compute_other_sv_batch,
    )
except ImportError:  # Direct script execution.
    from scoring_other_models import (
        CACHE_DIR,
        build_aligned_windows,
        compute_sv_batch as compute_other_sv_batch,
    )

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


BASE_DATA_DIR = os.environ.get(
    "BASE_DATA_DIR", "/home/jovyan/LEM_data3/hyunjincho/LEM"
)
SAMPLE_ROOT = os.environ.get(
    "GPT_SAMPLINGS_DIR",
    os.path.join(BASE_DATA_DIR, "predictions_clean", "gpt_samplings"),
)
DEFAULT_GPT_INPUT = os.path.join(SAMPLE_ROOT, "gpt.csv")
DEFAULT_BERT_INPUT = os.path.join(SAMPLE_ROOT, "bert.csv")
DEFAULT_OUTPUT_ROOT = os.path.join(
    BASE_DATA_DIR, "evaluation_sv", "gpt_other_models"
)
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CHECKPOINTS = {
    "llama": "meta-llama/Llama-3.2-3B",
    "gemma": "google/gemma-2-2b-it",
    "qwen25": "Qwen/Qwen2.5-3B-Instruct",
}
JUDGE_ORDER = ("llama", "gemma", "qwen25")
REQUIRED_COLUMNS = ("row_idx", "truth", "masked_sentence")
GENERATED_COLUMNS = (
    "prediction_source",
    "judge_name",
    "judge_checkpoint",
    "pred_used",
    "substitution",
)
torch.set_float32_matmul_precision("high")


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _pipe_candidates(value: object) -> List[str]:
    return [
        part.strip()
        for part in _clean_text(value).split("|")
        if part.strip()
    ]


def choose_substitution(row: pd.Series, prediction_source: str) -> str:
    """Select the first ranked candidate after case-insensitive truth removal."""
    truth = _clean_text(row.get("truth")).casefold()
    if prediction_source == "bert":
        candidates = [
            _clean_text(row.get("bert_substitution")),
            *_pipe_candidates(row.get("bert_pred_top5")),
        ]
    elif prediction_source == "gpt":
        candidates = [
            _clean_text(row.get("gpt_substitution")),
            _clean_text(row.get("pred_used")),
            _clean_text(row.get("pred_top1")),
            *_pipe_candidates(row.get("gpt_pred_top5")),
            *_pipe_candidates(row.get("pred_top5")),
        ]
    else:
        raise ValueError(f"Unknown prediction source: {prediction_source}")

    seen: Set[str] = set()
    for candidate in candidates:
        normalized = candidate.casefold()
        if not candidate or normalized in seen:
            continue
        seen.add(normalized)
        if normalized != truth:
            return candidate
    return ""


def score_column(judge_name: str) -> str:
    return f"sv_{judge_name}"


def output_columns(input_columns: Sequence[str], judge_name: str) -> List[str]:
    columns = list(input_columns)
    for column in (*GENERATED_COLUMNS, score_column(judge_name)):
        if column not in columns:
            columns.append(column)
    return columns


def score_chunk(
    frame: pd.DataFrame,
    prediction_source: str,
    judge_name: str,
    judge_checkpoint: str,
    tokenizer,
    model,
    window_size: int,
    max_len: int,
    sent_batch: int,
    device: str,
    strict: bool,
) -> pd.DataFrame:
    """Score valid rows and retain invalid rows with NaN scores."""
    input_columns = list(frame.columns)
    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    candidate_columns = (
        {"bert_substitution", "bert_pred_top5"}
        if prediction_source == "bert"
        else {
            "gpt_substitution",
            "gpt_pred_top5",
            "pred_used",
            "pred_top1",
            "pred_top5",
        }
    )
    if not candidate_columns.intersection(frame.columns):
        raise ValueError(
            f"Missing {prediction_source.upper()} substitution/candidate columns"
        )

    frame = frame.copy()
    frame["pred_used"] = frame.apply(
        choose_substitution,
        axis=1,
        prediction_source=prediction_source,
    ).astype(str)
    originals = []
    substitutes = []
    valid_positions: List[int] = []
    failure_count = 0

    for position, (_, row) in enumerate(frame.iterrows()):
        truth = _clean_text(row.get("truth"))
        masked_sentence = _clean_text(row.get("masked_sentence"))
        substitute = _clean_text(row.get("pred_used"))
        try:
            if not truth:
                raise ValueError("truth is empty")
            if not substitute:
                raise ValueError(
                    f"no truth-excluded {prediction_source.upper()} candidate"
                )
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
            valid_positions.append(position)
        except (TypeError, ValueError) as exc:
            failure_count += 1
            if strict:
                raise ValueError(f"row {position}: {exc}") from exc

    scores = [float("nan")] * len(frame)
    if originals:
        valid_scores = compute_other_sv_batch(
            originals,
            substitutes,
            tokenizer,
            model,
            max_len,
            sent_batch,
            device,
        )
        for position, score in zip(valid_positions, valid_scores):
            scores[position] = score

    frame["prediction_source"] = prediction_source
    frame["judge_name"] = judge_name
    frame["judge_checkpoint"] = judge_checkpoint
    frame["substitution"] = frame["pred_used"]
    frame[score_column(judge_name)] = scores
    result = frame.loc[:, output_columns(input_columns, judge_name)].copy()
    result.attrs["failure_count"] = failure_count
    return result


def _year_mask(frame: pd.DataFrame, years: Sequence[str]) -> pd.Series:
    wanted = {str(year) for year in years}
    numeric = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    return numeric.astype(str).isin(wanted)


def filter_years(
    frame: pd.DataFrame,
    years: Sequence[str],
    source_path: str,
) -> pd.DataFrame:
    if not years:
        return frame
    if "year" not in frame.columns:
        raise ValueError(f"--years requires a year column: {source_path}")
    return frame.loc[_year_mask(frame, years)].copy()


def iter_filtered_chunks(
    source_path: str,
    chunk_size: int,
    years: Sequence[str],
) -> Iterable[pd.DataFrame]:
    for frame in pd.read_csv(source_path, chunksize=chunk_size):
        yield filter_years(frame, years, source_path)


def count_filtered_rows(
    source_path: str,
    chunk_size: int,
    years: Sequence[str],
) -> int:
    return sum(
        len(frame)
        for frame in iter_filtered_chunks(source_path, chunk_size, years)
    )


def output_path(
    output_root: str,
    judge_name: str,
    prediction_source: str,
    input_path: str,
) -> str:
    basename = os.path.basename(input_path)
    if basename.endswith(".csv.gz"):
        stem = basename[: -len(".csv.gz")]
    elif basename.endswith(".csv"):
        stem = basename[: -len(".csv")]
    else:
        stem = os.path.splitext(basename)[0]
    directory = os.path.join(output_root, judge_name)
    os.makedirs(directory, exist_ok=True)
    label = (
        stem
        if stem.casefold() == prediction_source
        else f"{stem}_{prediction_source}"
    )
    return os.path.join(directory, f"{label}_with_sv_{judge_name}.csv.gz")


def append_gzip(path: str, frame: pd.DataFrame, write_header: bool) -> None:
    frame.to_csv(
        path,
        index=False,
        compression="gzip",
        mode="wt" if write_header else "at",
        header=write_header,
    )


def validate_resume_output(
    input_path: str,
    destination: str,
    prediction_source: str,
    judge_name: str,
    judge_checkpoint: str,
    chunk_size: int,
    years: Sequence[str],
) -> Tuple[bool, str]:
    """Accept only a complete output for the current source and checkpoint."""
    if not os.path.isfile(destination):
        return False, "output does not exist"
    try:
        input_header = pd.read_csv(input_path, nrows=0)
        output_header = pd.read_csv(destination, nrows=0)
        expected = output_columns(input_header.columns, judge_name)
        if list(output_header.columns) != expected:
            return False, "schema/order mismatch"

        metadata = pd.read_csv(
            destination,
            usecols=[
                "prediction_source",
                "judge_name",
                "judge_checkpoint",
                score_column(judge_name),
            ],
        )
        input_rows = count_filtered_rows(input_path, chunk_size, years)
        if len(metadata) != input_rows:
            return False, (
                f"row-count mismatch: input={input_rows}, output={len(metadata)}"
            )
        if not metadata["prediction_source"].eq(prediction_source).all():
            return False, "prediction_source mismatch"
        if not metadata["judge_name"].eq(judge_name).all():
            return False, "judge_name mismatch"
        if not metadata["judge_checkpoint"].eq(judge_checkpoint).all():
            return False, "judge_checkpoint mismatch"
    except Exception as exc:
        return False, f"unreadable output: {exc}"
    return True, f"validated {len(metadata):,} rows"


def score_file_atomically(
    input_path: str,
    destination: str,
    prediction_source: str,
    judge_name: str,
    judge_checkpoint: str,
    tokenizer,
    model,
    window_size: int,
    max_len: int,
    sent_batch: int,
    chunk_size: int,
    device: str,
    years: Sequence[str],
    strict: bool,
) -> Tuple[int, int, int, float]:
    """Stream one input to a temporary gzip and replace only after success."""
    temporary = f"{destination}.tmp.{os.getpid()}"
    total_rows = 0
    failed_rows = 0
    valid_scores = 0
    score_sum = 0.0
    wrote_header = False
    try:
        for frame in iter_filtered_chunks(input_path, chunk_size, years):
            if frame.empty:
                continue
            scored = score_chunk(
                frame,
                prediction_source,
                judge_name,
                judge_checkpoint,
                tokenizer,
                model,
                window_size,
                max_len,
                sent_batch,
                device,
                strict,
            )
            total_rows += len(scored)
            failed_rows += int(scored.attrs.get("failure_count", 0))
            numeric_scores = pd.to_numeric(
                scored[score_column(judge_name)], errors="coerce"
            ).dropna()
            valid_scores += len(numeric_scores)
            score_sum += float(numeric_scores.sum())
            append_gzip(temporary, scored, write_header=not wrote_header)
            wrote_header = True

        if not wrote_header:
            input_header = pd.read_csv(input_path, nrows=0)
            empty = input_header.copy()
            for column in output_columns(input_header.columns, judge_name):
                if column not in empty.columns:
                    empty[column] = pd.Series(dtype="object")
            append_gzip(temporary, empty, write_header=True)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return total_rows, failed_rows, valid_scores, score_sum


def parse_judges(value: str) -> Tuple[str, ...]:
    normalized = value.strip().lower()
    if normalized == "all":
        return JUDGE_ORDER
    aliases = {"qwen": "qwen25", "qwen2.5": "qwen25"}
    requested = {
        aliases.get(part.strip(), part.strip())
        for part in normalized.split(",")
        if part.strip()
    }
    unknown = sorted(requested.difference(JUDGE_ORDER))
    if unknown:
        raise ValueError(f"Unknown judges: {unknown}")
    if not requested:
        raise ValueError("No judges selected")
    return tuple(name for name in JUDGE_ORDER if name in requested)


def checkpoint_map(args: argparse.Namespace) -> Dict[str, str]:
    return {
        "llama": args.llama_ckpt,
        "gemma": args.gemma_ckpt,
        "qwen25": args.qwen_ckpt,
    }


def load_judge(
    checkpoint: str,
    cache_dir: str,
    device: str,
    use_auth_token: bool,
    local_files_only: bool,
    max_len: int,
):
    environment_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    token = (environment_token or True) if use_auth_token else False
    common_kwargs = {
        "token": token,
        "cache_dir": cache_dir,
        "trust_remote_code": True,
        "local_files_only": local_files_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, **common_kwargs)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModel.from_pretrained(
        checkpoint,
        torch_dtype="auto",
        **common_kwargs,
    ).to(device).eval()
    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions is not None and max_len > int(max_positions):
        raise ValueError(
            f"--max-len={max_len} exceeds model max_position_embeddings="
            f"{max_positions} for {checkpoint}"
        )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return tokenizer, model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score GPT and matched BERT substitutes with LLaMA/Gemma/Qwen."
    )
    parser.add_argument("--gpt-input", default=DEFAULT_GPT_INPUT)
    parser.add_argument("--bert-input", default=DEFAULT_BERT_INPUT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--judges",
        "--judge",
        dest="judges",
        default="all",
        help="all or comma-separated llama,gemma,qwen",
    )
    parser.add_argument("--llama-ckpt", default=DEFAULT_CHECKPOINTS["llama"])
    parser.add_argument("--gemma-ckpt", default=DEFAULT_CHECKPOINTS["gemma"])
    parser.add_argument("--qwen-ckpt", default=DEFAULT_CHECKPOINTS["qwen25"])
    parser.add_argument("--cache-dir", default=CACHE_DIR)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--years", nargs="*", default=[])
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=320)
    parser.add_argument("--sent-batch", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=2_000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-gpt", action="store_true")
    parser.add_argument("--skip-bert", action="store_true")
    auth = parser.add_mutually_exclusive_group()
    auth.add_argument(
        "--use-auth-token", dest="use_auth_token", action="store_true"
    )
    auth.add_argument(
        "--no-auth-token", dest="use_auth_token", action="store_false"
    )
    parser.set_defaults(use_auth_token=True)
    args = parser.parse_args()

    try:
        args.selected_judges = parse_judges(args.judges)
    except ValueError as exc:
        parser.error(str(exc))
    if args.skip_gpt and args.skip_bert:
        parser.error("--skip-gpt and --skip-bert cannot both be set")
    if args.window_size < 0 or args.max_len < 1:
        parser.error("window-size and max-len must be valid")
    if args.sent_batch < 1 or args.chunk_size < 1:
        parser.error("sent-batch and chunk-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        args.device = DEFAULT_DEVICE
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    sources: List[Tuple[str, str]] = []
    if not args.skip_gpt:
        sources.append(("gpt", args.gpt_input))
    if not args.skip_bert:
        sources.append(("bert", args.bert_input))
    missing = [path for _, path in sources if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")

    checkpoints = checkpoint_map(args)
    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)
    print(f"Judges: {list(args.selected_judges)}")
    print(f"Sources: {sources}")
    print(f"Device: {args.device}")
    print(f"Output root: {output_root}")
    print(
        f"Scoring config: window_size={args.window_size}, max_len={args.max_len}, "
        f"sent_batch={args.sent_batch}, chunk_size={args.chunk_size}"
    )
    print(
        "SV aggregation: uniform mean over aligned context tokens and one "
        "mean-pooled skill-span unit"
    )

    summaries = []
    for judge_name in args.selected_judges:
        checkpoint = checkpoints[judge_name]
        print(f"\n[JUDGE] {judge_name}: {checkpoint}")
        tokenizer, model = load_judge(
            checkpoint,
            args.cache_dir,
            args.device,
            args.use_auth_token,
            args.local_files_only,
            args.max_len,
        )
        try:
            for prediction_source, input_path in sources:
                destination = output_path(
                    output_root,
                    judge_name,
                    prediction_source,
                    input_path,
                )
                print(
                    f"[{judge_name}][{prediction_source}] {input_path} "
                    f"-> {destination}"
                )
                if args.resume:
                    complete, reason = validate_resume_output(
                        input_path,
                        destination,
                        prediction_source,
                        judge_name,
                        checkpoint,
                        args.chunk_size,
                        args.years,
                    )
                    if complete:
                        print(f"  [RESUME] skip: {reason}")
                        continue
                    if os.path.exists(destination):
                        print(f"  [RESUME] recompute: {reason}")

                rows, failures, valid, score_sum = score_file_atomically(
                    input_path,
                    destination,
                    prediction_source,
                    judge_name,
                    checkpoint,
                    tokenizer,
                    model,
                    args.window_size,
                    args.max_len,
                    args.sent_batch,
                    args.chunk_size,
                    args.device,
                    args.years,
                    args.strict,
                )
                mean_score = score_sum / valid if valid else float("nan")
                summaries.append(
                    {
                        "judge": judge_name,
                        "prediction_source": prediction_source,
                        "rows": rows,
                        "failed": failures,
                        "valid_scores": valid,
                        "mean_score": mean_score,
                        "output": destination,
                    }
                )
                print(
                    f"  [DONE] rows={rows:,}, failed={failures:,}, "
                    f"valid_scores={valid:,}, mean={mean_score:.8f}"
                )
        finally:
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n===== SUMMARY =====")
    if not summaries:
        print("No new tasks were scored (all selected outputs resumed).")
    for summary in summaries:
        print(
            f"{summary['judge']:7s} {summary['prediction_source']:4s} "
            f"rows={summary['rows']:,} failed={summary['failed']:,} "
            f"mean={summary['mean_score']:.8f} -> {summary['output']}"
        )


if __name__ == "__main__":
    main()
