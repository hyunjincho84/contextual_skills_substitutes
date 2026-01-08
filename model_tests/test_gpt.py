#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run GPT baseline (full whitelist).

- Default (no --num-rows): run ALL matched files, save per-file outputs next to each input.
- Test mode (--num-rows N): run ONLY ONE file (the first matched file),
  process first N rows, and save row-by-row to ./gpt_test.csv (crash-safe).
"""

import os
import glob
import json
import time
import argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ---------- DEFAULTS ----------
DEFAULT_IN_PATTERN = (
    "/home/jovyan/LEM_data2/hyunjincho/gpt_samples/"
    "20*/gpt_unique_samples_*_global*0.csv.gz"
)
DEFAULT_OUT_SUFFIX = "_with_gpt_pred"  # per-file output suffix (non-test mode)

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


def call_gpt(client: OpenAI, model: str, masked_sentence: str, all_skills):
    messages = build_prompt(masked_sentence, all_skills)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.responses.create(model=model, input=messages)
            out = resp.output[0].content[0]
            text = getattr(out, "text", str(out)).strip()

            # first line only
            if "\n" in text:
                text = text.split("\n")[0].strip()

            return text.lower()

        except Exception as e:
            print(f"[GPT ERROR] attempt {attempt}/{MAX_RETRIES}: {e}")
            if attempt == MAX_RETRIES:
                return ""
            time.sleep(RETRY_DELAY)


def infer_out_path_per_file(in_path: str):
    """Non-test mode: save next to input file with suffix."""
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

    out_name = f"{stem}{DEFAULT_OUT_SUFFIX}{ext}"
    return os.path.join(os.path.dirname(in_path), out_name)


def append_row_to_csv(out_path: str, row_dict: dict, write_header: bool):
    pd.DataFrame([row_dict]).to_csv(
        out_path,
        mode="a",
        header=write_header,
        index=False,
    )


# ---------- args ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-pattern", type=str, default=DEFAULT_IN_PATTERN)
    ap.add_argument("--skill2idx", type=str, default="./skill2idx_org.json")
    ap.add_argument("--model", type=str, default="gpt-5.1")
    ap.add_argument(
        "--num-rows",
        type=int,
        default=None,
        help="TEST MODE: if set, run ONLY ONE file and process first N rows; save to ./gpt_test.csv",
    )
    return ap.parse_args()


# ---------- main ----------
def main():
    args = parse_args()

    files = sorted(glob.glob(args.in_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.in_pattern}")

    print(f"[INFO] Matched {len(files)} files")

    # whitelist load (once)
    all_skills = load_all_skills(args.skill2idx)

    # client
    client = OpenAI()

    # -------------------
    # TEST MODE
    # -------------------
    if args.num_rows is not None:
        fp = files[0]
        out_path = "./gpt_test.csv"

        print(f"\n[TEST MODE] Using ONLY ONE file:\n  - {fp}")
        print(f"[TEST MODE] Saving to:\n  - {out_path}")
        print(f"[TEST MODE] Rows to process: first {args.num_rows}")

        df = pd.read_csv(fp)
        if "masked_sentence" not in df.columns:
            raise ValueError(f"'masked_sentence' missing in {fp}")

        df = df.iloc[: args.num_rows].copy()

        # overwrite test output each run (원하면 append 유지로 바꿀 수 있음)
        if os.path.exists(out_path):
            os.remove(out_path)

        write_header = True
        for _, row in tqdm(df.iterrows(), total=len(df), desc="GPT(TEST)"):
            masked_sentence = str(row["masked_sentence"])
            pred_top5 = call_gpt(client, args.model, masked_sentence, all_skills)

            parts = [s.strip() for s in pred_top5.split("|") if s.strip()]
            pred_top1 = parts[0] if parts else ""

            out_row = row.to_dict()
            out_row["pred_top5"] = pred_top5
            out_row["pred_top1"] = pred_top1

            append_row_to_csv(out_path, out_row, write_header)
            write_header = False

        print(f"\n[TEST MODE DONE] Saved → {out_path} (rows={len(df):,})")
        return

    # -------------------
    # FULL MODE (all files)
    # -------------------
    total_rows = 0
    for fp in files:
        print(f"\n[INFO] Processing: {fp}")
        df = pd.read_csv(fp)

        if "masked_sentence" not in df.columns:
            print("[SKIP] masked_sentence missing")
            continue

        out_path = infer_out_path_per_file(fp)
        write_header = not os.path.exists(out_path)

        for _, row in tqdm(df.iterrows(), total=len(df), desc="GPT"):
            masked_sentence = str(row["masked_sentence"])
            pred_top5 = call_gpt(client, args.model, masked_sentence, all_skills)

            parts = [s.strip() for s in pred_top5.split("|") if s.strip()]
            pred_top1 = parts[0] if parts else ""

            out_row = row.to_dict()
            out_row["pred_top5"] = pred_top5
            out_row["pred_top1"] = pred_top1

            append_row_to_csv(out_path, out_row, write_header)
            write_header = False

        total_rows += len(df)
        print(f"[INFO] Saved → {out_path}")

    print(f"\n[DONE] Total GPT-called rows: {total_rows:,}")


if __name__ == "__main__":
    main()