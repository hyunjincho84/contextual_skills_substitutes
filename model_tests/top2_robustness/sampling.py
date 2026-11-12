# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm

BASE_DATA_DIR = os.environ.get("BASE_DATA_DIR", "/home/jovyan/LEM_data2/data")
INPUT_GLOB = os.environ.get("TOP10_INPUT_GLOB", os.path.join(BASE_DATA_DIR, "bert_pred_new_top10_from_summary", "pred", "*", "sv_summary_llama_full_bert_top10_*.csv.gz"))
OUTPUT_ROOT = os.environ.get("SAMPLING_OUTPUT_ROOT", os.path.join(BASE_DATA_DIR, "bert_sampling_variants_from_summary"))
TOP10_COL = "bert_top10"
TOP10_PROBS_COL = "bert_top10_probs"


def parse_list_field(x, sep="|"):
    if pd.isna(x):
        return []
    return [s for s in str(x).split(sep) if s != ""]


def parse_prob_field(x, sep="|"):
    if pd.isna(x):
        return []
    vals = []
    for s in str(x).split(sep):
        s = s.strip()
        if not s:
            continue
        try:
            vals.append(float(s))
        except Exception:
            vals.append(0.0)
    return vals


def normalize_probs(probs):
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 1e-12, None)
    s = probs.sum()
    if s <= 0:
        return np.ones_like(probs) / len(probs)
    return probs / s


def filter_truth(cands, probs, truth):
    kept = [(c, p) for c, p in zip(cands, probs) if c != truth]
    if not kept:
        return [], []
    kept_cands = [x[0] for x in kept]
    kept_probs = [x[1] for x in kept]
    return kept_cands, kept_probs


def pick_top2_like(cands):
    if len(cands) == 0:
        return ""
    return cands[0]


def sample_topk(cands, probs, rng):
    if len(cands) == 0:
        return ""
    p = normalize_probs(probs)
    idx = rng.choice(len(cands), p=p)
    return cands[idx]


def threshold_sampling(cands, probs, rng, threshold=0.005):
    if len(cands) == 0:
        return ""
    kept = [(c, p) for c, p in zip(cands, probs) if p >= threshold]
    if len(kept) == 0:
        return ""
    kept_cands = [x[0] for x in kept]
    kept_probs = [x[1] for x in kept]
    p = normalize_probs(kept_probs)
    idx = rng.choice(len(kept_cands), p=p)
    return kept_cands[idx]


def temperature_sampling(cands, probs, rng, temperature=2.0):
    if len(cands) == 0:
        return ""
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 1e-12, None)

    logits = np.log(probs)
    scaled = logits / temperature
    scaled = scaled - scaled.max()
    p = np.exp(scaled)
    p = p / p.sum()

    idx = rng.choice(len(cands), p=p)
    return cands[idx]


def process_file(fp, args):
    rng = np.random.default_rng(args.random_seed)
    df = pd.read_csv(fp, compression="gzip")

    if args.top10_col not in df.columns or args.top10_probs_col not in df.columns or "truth" not in df.columns:
        raise ValueError(f"Required columns missing in {fp}")

    out_rows = []

    for _, row in df.iterrows():
        truth = str(row["truth"]).strip()
        cands = parse_list_field(row[args.top10_col], sep="|")
        probs = parse_prob_field(row[args.top10_probs_col], sep="|")

        n = min(len(cands), len(probs))
        cands = cands[:n]
        probs = probs[:n]

        cands, probs = filter_truth(cands, probs, truth)

        row_out = row.to_dict()
        row_out["sample_top2_like"] = pick_top2_like(cands)
        row_out["sample_top10"] = sample_topk(cands, probs, rng)
        row_out["sample_thresh_0005_sampling"] = threshold_sampling(cands, probs, rng, threshold=args.threshold_0005)
        row_out["sample_thresh_0001_sampling"] = threshold_sampling(cands, probs, rng, threshold=args.threshold_0001)
        row_out["sample_temp_15"] = temperature_sampling(cands, probs, rng, temperature=args.temperature_15)
        row_out["sample_temp_20"] = temperature_sampling(cands, probs, rng, temperature=args.temperature_20)

        out_rows.append(row_out)

    out_df = pd.DataFrame(out_rows)

    year = os.path.basename(os.path.dirname(fp))
    out_dir = os.path.join(args.output_root, year)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, os.path.basename(fp).replace(".csv.gz", "_sampling.csv.gz"))
    out_df.to_csv(out_path, index=False, compression="gzip")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", default=INPUT_GLOB)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--top10-col", default=TOP10_COL)
    parser.add_argument("--top10-probs-col", default=TOP10_PROBS_COL)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--threshold-0005", type=float, default=0.005)
    parser.add_argument("--threshold-0001", type=float, default=0.001)
    parser.add_argument("--temperature-15", type=float, default=1.5)
    parser.add_argument("--temperature-20", type=float, default=2.0)
    args = parser.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.input_glob}")

    print(f"Found {len(files)} files")
    for fp in tqdm(files, desc="Processing files", unit="file"):
        out_path = process_file(fp, args)
        print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    main()