#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run GPT baseline (full whitelist).

- Default (no --num-rows): read the combined bert.csv and write one gpt.csv.
- Python 2010 preset (--python-2010): process the 12 monthly python_2010 files.
- Test mode (--num-rows N): run ONLY ONE file (the first matched file),
  process first N rows, and save row-by-row to gpt_samplings/gpt_test.csv.
"""

import os
import glob
import hashlib
import json
import time
import argparse
import pandas as pd
from tqdm import tqdm

# ---------- DEFAULTS ----------
BASE_DATA_DIR = os.environ.get("BASE_DATA_DIR", "/home/jovyan/LEM_data3/hyunjincho/LEM")
GPT_SAMPLINGS_DIR = os.path.join(BASE_DATA_DIR, "predictions_clean", "gpt_samplings")
DEFAULT_IN_PATTERN = os.environ.get("GPT_SAMPLE_PATH", os.path.join(GPT_SAMPLINGS_DIR, "bert.csv"))
DEFAULT_OUT_PATH = os.environ.get("GPT_OUTPUT_PATH", os.path.join(GPT_SAMPLINGS_DIR, "gpt.csv"))
DEFAULT_TEST_OUT = os.environ.get("GPT_TEST_OUT", os.path.join(GPT_SAMPLINGS_DIR, "gpt_test.csv"))
DEFAULT_OUT_SUFFIX = "_with_gpt_pred"  # per-file output suffix (non-test mode)
PYTHON_2010_DIR = os.path.join(BASE_DATA_DIR, "predictions_clean", "python_2010")
PYTHON_2010_OUT_ROOT = os.path.join(BASE_DATA_DIR, "predictions_clean", "gpt_pred")
PYTHON_2010_TEST_OUT = os.path.join(
    PYTHON_2010_OUT_ROOT,
    "python_2010",
    "gpt_test.csv",
)

MAX_RETRIES = 3
RETRY_DELAY = 3.0


# ---------- utils ----------
def load_all_skills(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    skills = list(data.keys())
    print(f"[INFO] Loaded {len(skills):,} skills")
    return skills


def build_prompt(masked_sentence: str, all_skills):
    allowed_list = " | ".join(all_skills)

    system_msg = (
        "You are an expert in labor economics and job skill analysis.\n"
        "You are given a job posting sentence that contains a [MASK] token.\n"
        "Infer the FIVE most appropriate skills to replace [MASK].\n\n"
        "HARD CONSTRAINT:\n"
        "• Choose ONLY from the allowed skill list.\n"
        "• Do NOT invent new skills.\n"
        "• Use EXACT spelling from the list.\n\n"
        f"Allowed skills ({len(all_skills)}):\n"
        f"{allowed_list}\n\n"
        "Output ONLY this format:\n"
        "skill1|skill2|skill3|skill4|skill5\n"
        "No extra text."
    )

    user_msg = f"job posting:\n{masked_sentence}\n"

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def build_prompt_cache_key(model: str, all_skills) -> str:
    """Return a stable routing key for the shared system-prompt prefix."""
    system_msg = build_prompt("", all_skills)[0]["content"]
    digest = hashlib.sha256(f"{model}\0{system_msg}".encode("utf-8")).hexdigest()[:24]
    return f"skill-whitelist-{digest}"


def record_usage(usage_totals: dict, resp) -> None:
    """Accumulate Responses API token usage, including prompt-cache hits."""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return

    details = getattr(usage, "input_tokens_details", None)
    usage_totals["requests"] += 1
    usage_totals["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
    usage_totals["cached_tokens"] += getattr(details, "cached_tokens", 0) or 0
    usage_totals["output_tokens"] += getattr(usage, "output_tokens", 0) or 0


def print_usage(usage_totals: dict) -> None:
    input_tokens = usage_totals["input_tokens"]
    cached_tokens = usage_totals["cached_tokens"]
    uncached_tokens = max(input_tokens - cached_tokens, 0)
    cache_rate = 100.0 * cached_tokens / input_tokens if input_tokens else 0.0
    print(
        "[USAGE] "
        f"requests={usage_totals['requests']:,}, "
        f"input={input_tokens:,}, "
        f"cached={cached_tokens:,}, "
        f"uncached={uncached_tokens:,}, "
        f"cache_rate={cache_rate:.1f}%, "
        f"output={usage_totals['output_tokens']:,}"
    )


def select_substitution(pred_top5: str, truth: str) -> str:
    """Return the highest-ranked prediction after explicitly excluding truth."""
    truth_normalized = str(truth).strip().lower()
    candidates = [skill.strip() for skill in pred_top5.split("|") if skill.strip()]
    return next(
        (skill for skill in candidates if skill.lower() != truth_normalized),
        "",
    )


def resolve_input_files(in_pattern: str):
    """Resolve either a glob pattern, one file, or a flat CSV directory."""
    if os.path.isdir(in_pattern):
        files = glob.glob(os.path.join(in_pattern, "*.csv"))
        files.extend(glob.glob(os.path.join(in_pattern, "*.csv.gz")))
        return sorted(files)
    return sorted(glob.glob(in_pattern))


def resolve_truth_column(columns) -> str:
    """Support the sampled-data `truth` and clean-data `true_skill` schemas."""
    for column in ("truth", "true_skill"):
        if column in columns:
            return column
    raise ValueError("Input is missing a truth column; expected 'truth' or 'true_skill'")


def call_gpt(
    client,
    model: str,
    masked_sentence: str,
    all_skills,
    max_retries: int,
    retry_delay: float,
    prompt_cache_key: str,
    usage_totals: dict,
):
    messages = build_prompt(masked_sentence, all_skills)

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=messages,
                prompt_cache_key=prompt_cache_key,
            )
            record_usage(usage_totals, resp)
            out = resp.output[0].content[0]
            text = getattr(out, "text", str(out)).strip()

            # first line only
            if "\n" in text:
                text = text.split("\n")[0].strip()

            return text.lower()

        except Exception as e:
            print(f"[GPT ERROR] attempt {attempt}/{max_retries}: {e}")
            if attempt == max_retries:
                raise
            time.sleep(retry_delay)


def infer_out_path_per_file(in_path: str, out_suffix: str, out_root: str = None):
    """Build a full-mode output path, optionally under out_root/{year}."""
    base = os.path.basename(in_path)
    if base.endswith(".csv.gz"):
        stem = base[:-len(".csv.gz")]
        ext = ".csv.gz"
    elif base.endswith(".csv"):
        stem = base[:-len(".csv")]
        ext = ".csv"
    else:
        stem = base
        ext = ".csv"

    out_name = f"{stem}{out_suffix}{ext}"
    if out_root is None:
        out_dir = os.path.dirname(in_path)
    else:
        year = os.path.basename(os.path.dirname(in_path))
        out_dir = os.path.join(out_root, year)
        os.makedirs(out_dir, exist_ok=True)

    return os.path.join(out_dir, out_name)


def append_row_to_csv(out_path: str, row_dict: dict, write_header: bool):
    pd.DataFrame([row_dict]).to_csv(
        out_path,
        mode="a",
        header=write_header,
        index=False,
    )


def validate_resume_prefix(
    input_df: pd.DataFrame,
    out_path: str,
    model: str,
    truth_column: str,
) -> int:
    """Validate an existing output against the input prefix and return saved rows."""
    existing = pd.read_csv(out_path)
    saved_rows = len(existing)
    expected_columns = list(input_df.columns) + [
        "gpt_pred_top5",
        "gpt_substitution",
        "gpt_model",
    ]
    if list(existing.columns) != expected_columns:
        raise ValueError("Resume output schema does not match the current input/output schema")

    if saved_rows > len(input_df):
        raise ValueError(
            f"Resume output has more rows than input: {saved_rows:,} > {len(input_df):,}"
        )

    for column in (truth_column, "masked_sentence"):
        if column not in existing.columns:
            raise ValueError(f"Resume output is missing required column: {column}")
        input_values = input_df.iloc[:saved_rows][column].fillna("").astype(str).reset_index(drop=True)
        output_values = existing[column].fillna("").astype(str).reset_index(drop=True)
        if not input_values.equals(output_values):
            raise ValueError(
                f"Resume validation failed: existing {column} does not match input prefix"
            )

    saved_models = set(existing["gpt_model"].dropna().astype(str).unique())
    if saved_models and saved_models != {model}:
        raise ValueError(
            f"Resume model mismatch: output={sorted(saved_models)}, requested={model}"
        )

    print(f"[RESUME] Validated {saved_rows:,} saved rows; continuing at row {saved_rows:,}")
    return saved_rows

# ---------- args ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-pattern", type=str, default=DEFAULT_IN_PATTERN)
    ap.add_argument(
        "--python-2010",
        action="store_true",
        help=(
            "Process all monthly CSV files in predictions_clean/python_2010 and "
            "write per-file outputs under predictions_clean/gpt_pred/python_2010."
        ),
    )
    ap.add_argument("--skill2idx", type=str, default=os.path.join(BASE_DATA_DIR, "all_filterd", "skill2idx.json"))
    ap.add_argument("--model", type=str, default="gpt-5.1")
    ap.add_argument(
        "--num-rows",
        type=int,
        default=None,
        help="TEST MODE: if set, run ONLY ONE file and process first N rows; save to --test-out",
    )
    ap.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    ap.add_argument("--retry-delay", type=float, default=RETRY_DELAY)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a validated existing output file, one completed row at a time.",
    )

    ap.add_argument("--out-suffix", type=str, default=DEFAULT_OUT_SUFFIX)
    ap.add_argument(
        "--out-path",
        type=str,
        default=DEFAULT_OUT_PATH,
        help="FULL MODE: output path when processing the single combined sample file.",
    )
    ap.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="FULL MODE: optional output root; preserves the input year directory.",
    )
    ap.add_argument(
        "--test-out",
        type=str,
        default=DEFAULT_TEST_OUT,
        help="Output CSV used only with --num-rows.",
    )
    return ap.parse_args()


def apply_python_2010_preset(args) -> None:
    """Apply input and output defaults for the monthly Python 2010 dataset."""
    if not args.python_2010:
        return

    args.in_pattern = PYTHON_2010_DIR
    if args.out_path == DEFAULT_OUT_PATH:
        args.out_path = None
    if args.out_root is None:
        args.out_root = PYTHON_2010_OUT_ROOT
    if args.test_out == DEFAULT_TEST_OUT:
        args.test_out = PYTHON_2010_TEST_OUT


# ---------- main ----------
def main():
    args = parse_args()
    apply_python_2010_preset(args)

    from openai import OpenAI

    files = resolve_input_files(args.in_pattern)
    if not files:
        raise FileNotFoundError(f"No files matched: {args.in_pattern}")

    print(f"[INFO] Matched {len(files)} files")

    # whitelist load (once)
    all_skills = load_all_skills(args.skill2idx)
    prompt_cache_key = build_prompt_cache_key(args.model, all_skills)
    usage_totals = {
        "requests": 0,
        "input_tokens": 0,
        "cached_tokens": 0,
        "output_tokens": 0,
    }
    print(f"[INFO] Prompt cache key: {prompt_cache_key}")

    # client
    client = OpenAI()

    # -------------------
    # TEST MODE
    # -------------------
    if args.num_rows is not None:
        fp = files[0]
        out_path = args.test_out

        print(f"\n[TEST MODE] Using ONLY ONE file:\n  - {fp}")
        print(f"[TEST MODE] Saving to:\n  - {out_path}")
        print(f"[TEST MODE] Rows to process: first {args.num_rows}")

        df = pd.read_csv(fp)
        if "masked_sentence" not in df.columns:
            raise ValueError(f"'masked_sentence' missing in {fp}")
        truth_column = resolve_truth_column(df.columns)

        df = df.iloc[: args.num_rows].copy()

        out_dir = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(out_dir, exist_ok=True)
        if args.resume and os.path.exists(out_path):
            start_row = validate_resume_prefix(
                df,
                out_path,
                args.model,
                truth_column,
            )
        else:
            start_row = 0
        if not args.resume and os.path.exists(out_path):
            os.remove(out_path)

        remaining_df = df.iloc[start_row:]
        write_header = not os.path.exists(out_path)
        for _, row in tqdm(
            remaining_df.iterrows(), total=len(remaining_df), desc="GPT(TEST)"
        ):
            masked_sentence = str(row["masked_sentence"])
            pred_top5 = call_gpt(
                client,
                args.model,
                masked_sentence,
                all_skills,
                args.max_retries,
                args.retry_delay,
                prompt_cache_key,
                usage_totals,
            )

            gpt_substitution = select_substitution(pred_top5, row[truth_column])

            out_row = row.to_dict()
            out_row["gpt_pred_top5"] = pred_top5
            out_row["gpt_substitution"] = gpt_substitution
            out_row["gpt_model"] = args.model

            append_row_to_csv(out_path, out_row, write_header)
            write_header = False

        print(
            f"\n[TEST MODE DONE] Saved → {out_path} "
            f"(total={len(df):,}, resumed={start_row:,}, new={len(remaining_df):,})"
        )
        print_usage(usage_totals)
        return

    # -------------------
    # FULL MODE (all files)
    # -------------------
    if args.out_path:
        if len(files) != 1:
            raise ValueError("--out-path requires exactly one input file")
        out_dir = os.path.dirname(os.path.abspath(args.out_path))
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(args.out_path) and not args.resume:
            raise FileExistsError(
                f"Refusing to append to existing combined output: {args.out_path}"
            )
        print(f"[INFO] Combined GPT output: {args.out_path}")

    total_rows = 0
    for fp in files:
        print(f"\n[INFO] Processing: {fp}")
        df = pd.read_csv(fp)

        if "masked_sentence" not in df.columns:
            print("[SKIP] masked_sentence missing")
            continue
        truth_column = resolve_truth_column(df.columns)

        out_path = args.out_path or infer_out_path_per_file(fp, args.out_suffix, args.out_root)
        if args.resume and os.path.exists(out_path):
            start_row = validate_resume_prefix(
                df,
                out_path,
                args.model,
                truth_column,
            )
        else:
            start_row = 0
        remaining_df = df.iloc[start_row:]
        write_header = not os.path.exists(out_path)

        for _, row in tqdm(remaining_df.iterrows(), total=len(remaining_df), desc="GPT"):
            masked_sentence = str(row["masked_sentence"])
            pred_top5 = call_gpt(
                client,
                args.model,
                masked_sentence,
                all_skills,
                args.max_retries,
                args.retry_delay,
                prompt_cache_key,
                usage_totals,
            )

            gpt_substitution = select_substitution(pred_top5, row[truth_column])

            out_row = row.to_dict()
            out_row["gpt_pred_top5"] = pred_top5
            out_row["gpt_substitution"] = gpt_substitution

            out_row["gpt_model"] = args.model
            append_row_to_csv(out_path, out_row, write_header)
            write_header = False

        total_rows += len(remaining_df)
        print(
            f"[INFO] Saved → {out_path} "
            f"(total={len(df):,}, resumed={start_row:,}, new={len(remaining_df):,})"
        )

    print(f"\n[DONE] Total GPT-called rows: {total_rows:,}")
    print_usage(usage_totals)


if __name__ == "__main__":
    main()
