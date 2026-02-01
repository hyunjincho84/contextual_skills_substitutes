#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute posting-level positive-margin ratio (2025)
AND plot posting-level max margin_ratio distribution.

Pipeline
--------
1) Read prediction files (predictions_2025-*.csv.gz) that contain:
   ['year', 'file', 'truth', 'pred_top1', 'pred_top5', 'masked_sentence'].

2) For each prediction row, define `subs` as:
   - parse pred_top5 into a list
   - pick the first candidate != truth  (top-ranked WRONG)

3) Use preprocessed files referenced by pred['file'] under:
   --preprocessed-root
   to attach posting id columns:
   - row_idx
   - file_path
   Posting id = (row_idx, file_path)

4) Read trends table (--margin-csv) with columns:
   truth, subs, year_month, truth_trend, subs_trend, trend_diff
   and compute:
   margin_ratio = (subs_trend - truth_trend) / truth_trend * 100
   (or trend_diff / truth_trend * 100 if trend_diff exists)

5) Merge margin_ratio onto (truth, subs, year_month)

6) Optional: drop bidirectional swaps within the same posting
   (A->B and B->A both exist in same posting)

7) Posting-level aggregation:
   - has_positive_margin := any(margin_ratio > 0)
   - max_margin_ratio := max(margin_ratio)

8) Print posting-level positive ratio
9) Plot histogram of posting-level max_margin_ratio for positive-only, log-binned x-axis.
"""

import os
import re
import glob
import json
import ast
import argparse
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)


def parse_year_month_from_filename(path: str) -> str:
    """
    Expect filenames like predictions_2025-01.csv.gz, predictions_2025-12.csv.gz
    """
    base = os.path.basename(path)
    m = re.search(r"(\d{4})-(\d{1,2})", base)
    if not m:
        # fallback: any 6-digit
        m2 = re.search(r"(\d{6})", base)
        if not m2:
            raise ValueError(f"Cannot parse year_month from {path}")
        return m2.group(1)

    y = m.group(1)
    mm = str(int(m.group(2))).zfill(2)
    return f"{y}{mm}"


def safe_parse_topk_list(x) -> List[str]:
    """
    Parse pred_top5 which may be:
      - python list string: "['a','b']"
      - json list string: '["a","b"]'
      - delimited string: "a|b|c" or "a, b, c"
      - already a list
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]

    s = str(x).strip()
    if not s:
        return []

    # try JSON / python literal
    for fn in (json.loads, ast.literal_eval):
        try:
            obj = fn(s)
            if isinstance(obj, list):
                return [str(v).strip() for v in obj if str(v).strip()]
        except Exception:
            pass

    # try delimiters
    if "|" in s:
        return [p.strip() for p in s.split("|") if p.strip()]
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]

    return [s]


def pick_top_wrong(truth: str, top_list: List[str]) -> Optional[str]:
    """
    Return first candidate != truth.
    """
    t = (truth or "").strip()
    for c in top_list:
        c = (c or "").strip()
        if c and c != t:
            return c
    return None


def attach_posting_ids(pred_block: pd.DataFrame, preprocessed_path: str) -> pd.DataFrame:
    """
    Attach row_idx/file_path by assuming row-order alignment:
      pred_block row i corresponds to preprocessed row i.
    """
    pre = pd.read_csv(
        preprocessed_path,
        compression="gzip" if preprocessed_path.endswith(".gz") else None,
        low_memory=False,
    )

    need = {"row_idx", "file_path"}
    miss = need - set(pre.columns)
    if miss:
        raise ValueError(f"[preprocessed] Missing {miss} in {preprocessed_path}. Columns={list(pre.columns)}")

    pre = pre.reset_index(drop=True)
    out = pred_block.reset_index(drop=True).copy()

    if len(out) > len(pre):
        raise ValueError(
            f"[align] pred rows ({len(out)}) > pre rows ({len(pre)}) for {preprocessed_path}"
        )

    out["row_idx"] = pre.loc[:len(out)-1, "row_idx"].to_numpy()
    out["file_path"] = pre.loc[:len(out)-1, "file_path"].astype(str).to_numpy()
    return out


def drop_bidirectional(df_long: pd.DataFrame, posting_cols: List[str]) -> pd.DataFrame:
    """
    Drop rows where within the same posting, both A->B and B->A exist.
    df_long must contain: posting_cols + truth + subs
    """
    work = df_long.copy()

    a = work["truth"].astype(str)
    b = work["subs"].astype(str)
    work["_a"] = a.where(a <= b, b)
    work["_b"] = b.where(a <= b, a)
    work["_dir"] = (work["truth"] != work["_a"]).astype(int)

    grp = work.groupby(posting_cols + ["_a", "_b"])["_dir"].nunique()
    bidir = grp[grp == 2].reset_index()[posting_cols + ["_a", "_b"]]

    marked = work.merge(
        bidir,
        on=posting_cols + ["_a", "_b"],
        how="left",
        indicator=True
    )

    to_drop = (marked["_merge"] == "both")
    removed = int(to_drop.sum())
    out = marked.loc[~to_drop].drop(columns=["_a", "_b", "_dir", "_merge"]).copy()

    print("==============================================")
    print("[Bidirectional-swap removal]")
    print(f"Posting id columns: {posting_cols}")
    print(f"Rows removed: {removed:,}")
    print(f"Remaining long rows: {len(out):,}")
    print("==============================================")
    return out


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True,
                    help="Directory containing prediction files, e.g. .../bert_pred_new/pred/2025")
    ap.add_argument("--pred-glob", default="predictions_2025-*.csv.gz",
                    help="Glob inside pred-dir (default: predictions_2025-*.csv.gz)")
    ap.add_argument("--preprocessed-root", required=True,
                    help="Root directory for preprocessed files, e.g. .../preprocessed_www_new/test/2025")
    ap.add_argument("--margin-csv", required=True,
                    help="CSV with columns: truth, subs, year_month, truth_trend, subs_trend, trend_diff")
    ap.add_argument("--out-fig", default="./posting_margin_ratio_hist_2025.png",
                    help="Output figure path")
    ap.add_argument("--drop-bidirectional", action="store_true",
                    help="Drop bidirectional swaps within the same posting")
    ap.add_argument("--encoding", default="utf-8")
    args = ap.parse_args()

    # ---- load trends table & compute margin_ratio ----
    tr = pd.read_csv(args.margin_csv, encoding=args.encoding, low_memory=False)

    need_tr = {"truth", "subs", "year_month", "truth_trend", "subs_trend"}
    miss_tr = need_tr - set(tr.columns)
    if miss_tr:
        raise ValueError(f"[margin-csv] Missing columns: {miss_tr}. Columns={list(tr.columns)}")

    # normalize keys
    for c in ["truth", "subs", "year_month"]:
        tr[c] = tr[c].astype(str).str.strip()

    # numeric
    tr["truth_trend"] = pd.to_numeric(tr["truth_trend"], errors="coerce")
    tr["subs_trend"] = pd.to_numeric(tr["subs_trend"], errors="coerce")

    if "trend_diff" in tr.columns:
        tr["trend_diff"] = pd.to_numeric(tr["trend_diff"], errors="coerce")
        diff = tr["trend_diff"]
    else:
        diff = tr["subs_trend"] - tr["truth_trend"]

    # avoid division by zero
    tr = tr.dropna(subset=["truth_trend"]).copy()
    tr = tr[tr["truth_trend"] != 0].copy()

    tr["margin_ratio"] = (diff / tr["truth_trend"]) * 100.0
    tr = tr[["truth", "subs", "year_month", "margin_ratio"]].drop_duplicates(
        subset=["truth", "subs", "year_month"], keep="last"
    )

    # ---- iterate prediction files ----
    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, args.pred_glob)))
    if not pred_files:
        raise FileNotFoundError(f"No pred files matched: {os.path.join(args.pred_dir, args.pred_glob)}")

    all_rows = []

    for pf in pred_files:
        ym = parse_year_month_from_filename(pf)
        print(f"[LOAD] {pf} -> year_month={ym}")

        pred = pd.read_csv(
            pf,
            compression="gzip" if pf.endswith(".gz") else None,
            encoding=args.encoding,
            low_memory=False
        )

        # required columns
        for col in ["file", "truth", "pred_top5"]:
            if col not in pred.columns:
                raise ValueError(f"[pred] Missing '{col}' in {pf}. Columns={list(pred.columns)}")

        pred["year_month"] = ym
        pred["truth"] = pred["truth"].astype(str).str.strip()

        # group by referenced preprocessed file
        for fname, block in pred.groupby("file", sort=False):
            pre_path = os.path.join(args.preprocessed_root, str(fname))
            if not os.path.exists(pre_path):
                raise FileNotFoundError(f"[preprocessed] Not found: {pre_path} (from file={fname})")

            block2 = attach_posting_ids(block, pre_path)

            # subs = top-ranked WRONG from pred_top5
            subs = []
            for t, x in zip(block2["truth"].tolist(), block2["pred_top5"].tolist()):
                top_list = safe_parse_topk_list(x)
                subs.append(pick_top_wrong(t, top_list))

            out = pd.DataFrame({
                "row_idx": block2["row_idx"].to_numpy(),
                "file_path": block2["file_path"].astype(str).to_numpy(),
                "year_month": block2["year_month"].astype(str).to_numpy(),
                "truth": block2["truth"].astype(str).str.strip().to_numpy(),
                "subs": pd.Series(subs, dtype="object"),
            })

            # drop rows with no valid wrong candidate
            out = out[out["subs"].notna()].copy()
            out["subs"] = out["subs"].astype(str).str.strip()
            out = out[out["subs"] != ""].copy()

            # merge margin_ratio
            out = out.merge(
                tr,
                on=["truth", "subs", "year_month"],
                how="left",
                validate="many_to_one"
            )

            out["margin_ratio"] = pd.to_numeric(out["margin_ratio"], errors="coerce")
            all_rows.append(out)

    df = pd.concat(all_rows, ignore_index=True)

    posting_cols = ["row_idx", "file_path"]

    # ---- optional: drop bidirectional swaps ----
    if args.drop_bidirectional:
        df = drop_bidirectional(df, posting_cols)

    # ---- posting-level aggregation ----
    df["_pos"] = (df["margin_ratio"] > 0).fillna(False)

    posting = (
        df.groupby(posting_cols, as_index=False)
          .agg(
              has_positive_margin=("_pos", "any"),
              max_margin_ratio=("margin_ratio", "max")
          )
    )

    total = len(posting)
    pos = int(posting["has_positive_margin"].sum())
    pct = (pos / total * 100.0) if total > 0 else 0.0

    print("==============================================")
    print("Posting-level positive-margin substitution rate (2025)")
    print(f"Posting id = {posting_cols}")
    print(f"Total postings: {total:,}")
    print(f"Postings with >=1 positive-margin substitution: {pos:,} ({pct:.2f}%)")
    print("==============================================")

    # ---- plot histogram (posting-level max, positive only) ----
    plot_vals = posting.loc[posting["max_margin_ratio"] > 0, "max_margin_ratio"].dropna()
    if plot_vals.empty:
        print("[WARN] No positive max_margin_ratio values to plot.")
        return

    # log bins
    min_val = 1e-1
    max_val = float(plot_vals.max())
    if max_val <= min_val:
        bins = 20
        use_log = False
    else:
        bins = np.logspace(np.log10(min_val), np.log10(max_val), 40)
        use_log = True

    sns.set(style="ticks", font_scale=1.4)
    plt.figure(figsize=(8, 5))

    sns.histplot(
        plot_vals,
        bins=bins,
        color="#55C0C2",
        edgecolor="#333333",
        linewidth=0.6,
        alpha=0.85,
    )

    if use_log:
        plt.xscale("log")
    plt.margins(x=0)
    plt.grid(False)

    plt.xlabel("Exposure Gain (%)", fontsize=22)
    plt.ylabel("Posting Count", fontsize=22)

    plt.tight_layout()
    ensure_dir(os.path.dirname(os.path.abspath(args.out_fig)))
    plt.savefig(args.out_fig, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved histogram -> {args.out_fig}")


if __name__ == "__main__":
    main()



    
    
"""
python3 tmp.py \
  --pred-dir /home/jovyan/LEM_data2/hyunjincho/bert_pred_new/pred/2025 \
  --preprocessed-root /home/jovyan/LEM_data2/hyunjincho/preprocessed_www_new/test/2025 \
  --margin-csv ./counts_by_pair_with_trends_monthly.csv \
  --drop-bidirectional \
  --out-fig ./posting_margin_ratio_hist_2025.png
"""