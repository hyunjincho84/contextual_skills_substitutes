#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
For a TARGET_SKILL, compute YEAR-wise Top-K substitutes using posting-level
"most-sub" weights (post-normalized within each job posting).

Core idea
- A posting is identified by (source-file identifier, row_idx).
- For each target occurrence, use the explicit truth-excluded substitution column.
- For a posting, we count how many times each candidate was chosen (cnt),
  then distribute weight by cnt / n_occ (post-normalization within posting).
- Aggregate these posting-level weights by YEAR to obtain the final ranking.
- Ranking criterion: weight_sum per year (sum of posting-level weights).
  For readability, we also provide year-wise normalized weight (weight_norm).

Expected input format (BERT monthly Top-10 predictions):
- Files under:
    pred_root/YYYY/predictions_YYYY-MM.csv.gz   (or .csv)
- Each predictions file must contain columns:
    row_idx, truth, substitution, and a source identifier such as
    preprocessed_file_path.

Because row_idx restarts for each source file, a posting is identified by
(__src, row_idx), where __src is derived from --posting-source-col.

Output CSV columns:
    truth, year, rank, substitute, weight_sum, weight_norm
"""

"""
Usage
python3 yearwise_substitutes.py \
  --pred-root /home/jovyan/LEM_data2/data/bert_pred_new/pred \
  --target-skill python \
  --out-topk 5 \
  --out ./python_yearwise_top5.csv
"""
import os
import re
import glob
import math
import argparse
from typing import List, Optional

import pandas as pd
from tqdm import tqdm


# ------------------------
# Helpers
# ------------------------
def safe_lower(x) -> str:
    return str(x).lower().strip() if pd.notna(x) else ""


def find_prediction_files(pred_root: str) -> List[str]:
    """
    pred_root can be:
      - directory like .../bert_pred_new/pred
      - or a single file path (then we just use it)

    We search for:
      pred_root/YYYY/predictions_*.csv.gz  (preferred)
      pred_root/YYYY/predictions_*.csv
    """
    if os.path.isfile(pred_root):
        return [pred_root]

    files = sorted(glob.glob(os.path.join(pred_root, "[0-9]" * 4, "predictions_*.csv.gz")))
    if not files:
        files = sorted(glob.glob(os.path.join(pred_root, "[0-9]" * 4, "predictions_*.csv")))
    if not files:
        # fallback: flat
        files = sorted(glob.glob(os.path.join(pred_root, "predictions_*.csv.gz")))
        if not files:
            files = sorted(glob.glob(os.path.join(pred_root, "predictions_*.csv")))

    if not files:
        raise FileNotFoundError(f"No predictions_*.csv[.gz] files found under: {pred_root}")
    return files


def infer_year_from_path(fp: str) -> Optional[str]:
    """
    Try to infer year from path (directory or filename).
    """
    m = re.search(r"/(20\d{2})/", fp)
    if m:
        return m.group(1)
    m = re.search(r"(20\d{2})", os.path.basename(fp))
    return m.group(1) if m else None


def pick_most_sub_for_row(pred_top10: str, pred_probs: str, truth_lower: str) -> Optional[str]:
    """
    Pick exactly ONE "most-sub" candidate from a single occurrence row.
    - default: Top-1 from pred_top10
    - if Top-1 == truth, use Top-2
    - if both are truth or no candidate, return None

    pred_probs is parsed defensively only for integrity checks (not used in scoring).
    """
    cands = [c.strip() for c in str(pred_top10 or "").split("|") if c.strip()]

    # Defensive parsing of probs (not used for scoring)
    probs = []
    for p in str(pred_probs or "").split("|"):
        try:
            v = float(p)
            if math.isfinite(v):
                probs.append(v)
        except Exception:
            pass

    # If lengths mismatch, truncate to the shorter length (integrity cleanup)
    if probs and len(probs) != len(cands):
        m = min(len(cands), len(probs))
        cands, probs = cands[:m], probs[:m]

    if not cands:
        return None

    top1 = safe_lower(cands[0])
    if top1 != truth_lower:
        return cands[0]

    if len(cands) >= 2 and safe_lower(cands[1]) != truth_lower:
        return cands[1]

    return None


# ------------------------
# Load
# ------------------------
def load_predictions(
    pred_root: str,
    usecols_only: bool = True,
    candidate_col: str = "substitution",
    posting_source_col: str = "preprocessed_file_path",
    years: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load monthly predictions files and attach year (from file path if needed).
    """
    files = find_prediction_files(pred_root)
    if years:
        wanted = set(map(str, years))
        files = [fp for fp in files if infer_year_from_path(fp) in wanted]
        if not files:
            raise FileNotFoundError("After --years filtering, no files remain.")
    dfs = []

    need_cols = [
        "row_idx",
        "truth",
        candidate_col,
        posting_source_col,
        "file",
    ]

    for fp in tqdm(files, desc="Loading predictions files", unit="file"):
        try:
            if usecols_only:
                header = pd.read_csv(fp, nrows=0)
                usecols = [c for c in need_cols if c in header.columns]
                df = pd.read_csv(fp, usecols=usecols)
            else:
                df = pd.read_csv(fp)

            # attach year
            if "year" not in df.columns:
                y = infer_year_from_path(fp)
                df["year"] = y
            if "file" not in df.columns:
                df["file"] = os.path.basename(fp)
            if candidate_col not in df.columns:
                raise ValueError(f"missing candidate column: {candidate_col}")

            fallback_source = fp + "::" + df["file"].fillna("").astype(str)
            if posting_source_col in df.columns:
                posting_source = (
                    df[posting_source_col].fillna("").astype(str).str.strip()
                )
                posting_source = posting_source.where(
                    posting_source.ne(""),
                    fallback_source,
                )
            else:
                posting_source = fallback_source
            df["__src"] = posting_source
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] skip {fp}: {e}")

    if not dfs:
        raise RuntimeError("No valid prediction files loaded.")
    return pd.concat(dfs, ignore_index=True)


# ------------------------
# Core
# ------------------------
def yearwise_topk_mostsubs_postnorm(
    df: pd.DataFrame,
    target_skill: str,
    candidate_col: str = "substitution",
    out_topk: int = 5,
) -> pd.DataFrame:
    """
    For a target truth skill, pick 1 "most-sub" per occurrence, apply posting-level
    post-normalization (cnt / n_occ), then aggregate by year.

    Returns columns:
      truth, year, rank, substitute, weight_sum, weight_norm
    """
    required = {
        "__src",
        "row_idx",
        "truth",
        candidate_col,
        "year",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    t_low = safe_lower(target_skill)

    # Filter rows where truth == target
    df_t = df[df["truth"].astype(str).str.lower().str.strip() == t_low].copy()
    if df_t.empty:
        raise ValueError(f"No rows for TARGET_SKILL='{target_skill}'")

    # 1) Use the explicit truth-excluded substitute for each occurrence.
    picks = []
    for _, r in df_t.iterrows():
        year = r.get("year", None)
        raw_sub = r.get(candidate_col, "")
        sub = str(raw_sub).strip() if pd.notna(raw_sub) else ""
        if not sub or safe_lower(sub) == t_low:
            continue

        picks.append({
            "__src": r["__src"],
            "row_idx": r["row_idx"],
            "year": year,
            "truth": r["truth"],
            "candidate": sub,
        })

    if not picks:
        return pd.DataFrame(columns=["truth", "year", "rank", "substitute", "weight_sum", "weight_norm"])

    pick_df = pd.DataFrame(picks)

    # 2) n_occ per posting: (__src, row_idx, year, truth)
    occ_count = (
        df_t.groupby(["__src", "row_idx", "year", "truth"])
            .size().rename("n_occ").reset_index()
    )

    # 3) cnt per posting: (__src, row_idx, year, truth, candidate)
    post_counts = (
        pick_df.groupby(["__src", "row_idx", "year", "truth", "candidate"])
               .size().rename("cnt").reset_index()
    )

    # 4) posting-level weight = cnt / n_occ
    post_weights = (
        post_counts.merge(
            occ_count,
            on=["__src", "row_idx", "year", "truth"],
            how="left",
        ).assign(weight=lambda x: x["cnt"] / x["n_occ"].clip(lower=1))
    )

    # 5) aggregate by year: sum weights
    year_agg = (
        post_weights.groupby(["year", "truth", "candidate"])["weight"]
                    .sum().rename("weight_sum").reset_index()
    )

    # 6) per-year normalize + topk + rank
    results = []
    for year, g in year_agg.groupby("year"):
        g = g.sort_values("weight_sum", ascending=False)
        denom = g["weight_sum"].sum()
        g = g.assign(weight_norm=(g["weight_sum"] / denom) if denom > 0 else 0.0)
        g = g.head(out_topk).reset_index(drop=True)
        g["rank"] = range(1, len(g) + 1)

        for _, row in g.iterrows():
            results.append({
                "truth": target_skill,
                "year": year,
                "rank": int(row["rank"]),
                "substitute": row["candidate"],
                "weight_sum": float(row["weight_sum"]),
                "weight_norm": float(row["weight_norm"]),
            })

    if not results:
        return pd.DataFrame(columns=["truth", "year", "rank", "substitute", "weight_sum", "weight_norm"])

    return pd.DataFrame(results).sort_values(["year", "rank"])


# ------------------------
# CLI
# ------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True,
                    help="Prediction root dir containing YYYY/predictions_*.csv[.gz]")
    ap.add_argument("--target-skill", required=True,
                    help="Target truth skill to evaluate (case-insensitive)")
    ap.add_argument("--out", required=True,
                    help="Output CSV path")
    ap.add_argument("--candidate-col", default="substitution",
                    help="Truth-excluded substitute column in the prediction files")
    ap.add_argument("--posting-source-col", default="preprocessed_file_path",
                    help="Column identifying the source file in which row_idx is defined")
    ap.add_argument("--years", nargs="*", default=None,
                    help="Optional years to include, e.g. --years 2018 2019")
    ap.add_argument("--out-topk", type=int, default=5,
                    help="Top-K substitutes to keep per year")
    ap.add_argument("--no-usecols", action="store_true",
                    help="If set, read full CSV instead of only required columns.")
    return ap.parse_args()


def main():
    args = parse_args()

    df = load_predictions(
        args.pred_root,
        usecols_only=(not args.no_usecols),
        candidate_col=args.candidate_col,
        posting_source_col=args.posting_source_col,
        years=args.years,
    )
    print(f"[INFO] Loaded {len(df):,} rows from predictions files")

    out_df = yearwise_topk_mostsubs_postnorm(
        df,
        target_skill=args.target_skill,
        candidate_col=args.candidate_col,
        out_topk=args.out_topk,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"[OK] Saved -> {args.out}")
    print(out_df.head(15))


if __name__ == "__main__":
    main()