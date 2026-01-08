#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Find substitutes for a target skill, grouped by soc_2_name, with post-normalization
inside each job posting (row_idx).

Input:
- BERT prediction outputs produced by your evaluation script:
  OUT_DIR/pred/{YYYY}/predictions_{YYYY-MM}.csv.gz

Each input file must contain at least:
- row_idx
- soc_2_name
- truth
- pred_top5
- (optional) year

Core idea (count-based, post-normalized "most-sub"):
1) Filter rows where truth == target_skill.
2) For each occurrence row, pick exactly ONE "most-sub":
   - use top1 from pred_top5
   - if top1 == truth and top2 exists, use top2
   - if no valid substitute remains, skip that row
3) For each posting: n_occ = count of truth occurrences within (row_idx, soc_2_name, truth)
4) For each posting: cnt = count of chosen substitute within (row_idx, soc_2_name, truth, substitute)
5) Post weight = cnt / n_occ
6) Aggregate by soc_2_name: total_weight = sum(post weight)
7) Return Top-K substitutes per soc_2_name.
"""

"""
Usage
python3 areawise_substitutes.py \
  --pred-root /home/jovyan/LEM_data2/hyunjincho/bert_pred_new/pred \
  --target-skill python \
  --topk-per-soc2 5 \
  --out ./subs_python_by_soc2.csv \
  --usecols
"""

import os
import re
import glob
import argparse
from typing import List, Optional

import pandas as pd
from tqdm import tqdm


# ------------------------
# Utilities
# ------------------------
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def find_prediction_files(pred_root: str) -> List[str]:
    """
    pred_root can be:
      - a directory like /.../bert_pred_new/pred
      - a specific file path
    We will search recursively for predictions_*.csv.gz (or .csv).
    """
    if os.path.isfile(pred_root):
        return [pred_root]

    # Typical: pred_root/YYYY/predictions_YYYY-MM.csv.gz
    files = sorted(glob.glob(os.path.join(pred_root, "[0-9]" * 4, "predictions_*.csv.gz")))
    if not files:
        files = sorted(glob.glob(os.path.join(pred_root, "[0-9]" * 4, "predictions_*.csv")))
    if not files:
        # fallback: maybe flat directory
        files = sorted(glob.glob(os.path.join(pred_root, "predictions_*.csv.gz")))
        if not files:
            files = sorted(glob.glob(os.path.join(pred_root, "predictions_*.csv")))

    if not files:
        raise FileNotFoundError(f"No predictions_*.csv[.gz] files found under: {pred_root}")
    return files


def pick_most_sub(pred_top5: str, truth_lower: str) -> Optional[str]:
    """
    Pick exactly one substitute for a row:
      - default: top1
      - if top1 == truth and top2 exists and top2 != truth: use top2
      - otherwise: None
    """
    cands = [c.strip() for c in str(pred_top5 or "").split("|") if c.strip()]
    if not cands:
        return None
    top1 = cands[0].lower()
    if top1 != truth_lower:
        return cands[0]
    if len(cands) >= 2 and cands[1].lower() != truth_lower:
        return cands[1]
    return None


# ------------------------
# Core computation
# ------------------------
def topk_substitutes_by_soc2_postnormalized(
    df: pd.DataFrame,
    target_skill: str,
    topk_per_soc2: int = 5,
) -> pd.DataFrame:
    """
    Post-normalized most-sub counting, aggregated by soc_2_name.
    """
    required = {"row_idx", "soc_2_name", "truth", "pred_top5"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    t_low = normalize_text(target_skill)

    df = df.copy()
    df["truth_norm"] = df["truth"].astype(str).map(normalize_text)
    df["soc_2_name"] = df["soc_2_name"].astype(str)

    # 1) Filter rows for target truth
    df_t = df[df["truth_norm"] == t_low].copy()
    if df_t.empty:
        raise ValueError(f"No rows found for target skill: {target_skill}")

    # 2) Pick one "most-sub" per row
    picks = []
    for _, r in df_t.iterrows():
        sub = pick_most_sub(r.get("pred_top5"), t_low)
        if sub is None:
            continue
        if normalize_text(sub) == t_low:
            continue
        picks.append({
            "row_idx": r["row_idx"],
            "soc_2_name": r["soc_2_name"],
            "truth": r["truth"],          # keep original casing if any
            "substitute": sub,
            "year": r.get("year", None),
            "file": r.get("file", None),
        })

    if not picks:
        return pd.DataFrame(columns=[
            "truth", "soc_2_name", "rank", "substitute",
            "total_weight", "weight_norm", "count_posts", "count_norm"
        ])

    pick_df = pd.DataFrame(picks)

    # 3) n_occ per posting: count of truth occurrences within (row_idx, soc_2_name, truth)
    occ_count = (
        df_t.groupby(["row_idx", "soc_2_name", "truth"], dropna=False)
            .size().rename("n_occ").reset_index()
    )

    # 4) cnt per posting: count of picked substitute within (row_idx, soc_2_name, truth, substitute)
    post_counts = (
        pick_df.groupby(["row_idx", "soc_2_name", "truth", "substitute"], dropna=False)
               .size().rename("cnt").reset_index()
    )

    # 5) post weight = cnt / n_occ
    post_weights = (
        post_counts.merge(occ_count, on=["row_idx", "soc_2_name", "truth"], how="left")
                   .assign(weight=lambda x: x["cnt"] / x["n_occ"].clip(lower=1))
    )

    # 6) aggregate by soc_2_name
    agg = (
        post_weights.groupby(["soc_2_name", "truth", "substitute"], dropna=False)["weight"]
                    .agg(total_weight="sum", count_posts="count")
                    .reset_index()
    )

    # 7) normalize within each soc_2_name (for readability)
    agg["weight_norm"] = agg["total_weight"] / (
        agg.groupby("soc_2_name")["total_weight"].transform("sum").replace(0, float("nan"))
    )
    agg["count_norm"] = agg["count_posts"] / (
        agg.groupby("soc_2_name")["count_posts"].transform("sum").replace(0, float("nan"))
    )

    # rank + topk
    agg = agg.sort_values(["soc_2_name", "total_weight"], ascending=[True, False])
    agg["rank"] = agg.groupby("soc_2_name")["total_weight"].rank(method="first", ascending=False).astype(int)

    out = (
        agg[agg["rank"] <= topk_per_soc2]
        .sort_values(["soc_2_name", "rank"])
        [["truth", "soc_2_name", "rank", "substitute", "total_weight", "weight_norm", "count_posts", "count_norm"]]
    )
    return out


# ------------------------
# CLI
# ------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True,
                    help="Root directory of predictions_*.csv[.gz] (e.g., .../bert_pred_new/pred)")
    ap.add_argument("--target-skill", required=True,
                    help="Target truth skill to analyze (e.g., python)")
    ap.add_argument("--out", required=True,
                    help="Output CSV path")
    ap.add_argument("--topk-per-soc2", type=int, default=5,
                    help="Top-K substitutes per soc_2_name")
    ap.add_argument("--years", nargs="*", default=None,
                    help="Optional: only include files whose path contains these years (e.g., 2018 2019)")
    ap.add_argument("--usecols", action="store_true",
                    help="Read only required columns (faster, recommended).")
    return ap.parse_args()


def main():
    args = parse_args()

    files = find_prediction_files(args.pred_root)

    # optional year filter by path substring
    if args.years:
        want = set(map(str, args.years))
        files = [fp for fp in files if any(y in fp for y in want)]
        if not files:
            raise FileNotFoundError("After --years filtering, no files remain.")

    req_cols = ["row_idx", "soc_2_name", "truth", "pred_top5", "year", "file"]

    dfs = []
    for fp in tqdm(files, desc="Loading prediction files", unit="file"):
        try:
            if args.usecols:
                header = pd.read_csv(fp, nrows=0)
                use = [c for c in req_cols if c in header.columns]
                df = pd.read_csv(fp, usecols=use)
            else:
                df = pd.read_csv(fp)

            # If year/file columns are missing, create lightweight versions
            if "year" not in df.columns:
                m = re.search(r"/(20\d{2})/", fp)
                df["year"] = m.group(1) if m else None
            if "file" not in df.columns:
                df["file"] = os.path.basename(fp)

            dfs.append(df)
        except Exception as e:
            print(f"[WARN] skip {fp}: {e}")

    if not dfs:
        raise FileNotFoundError("No valid prediction files loaded.")

    big = pd.concat(dfs, ignore_index=True)

    out = topk_substitutes_by_soc2_postnormalized(
        big,
        target_skill=args.target_skill,
        topk_per_soc2=args.topk_per_soc2,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[OK] Saved -> {args.out}")
    print(out.head(20))


if __name__ == "__main__":
    main()