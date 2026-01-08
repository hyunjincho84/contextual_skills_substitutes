#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot pairwise overlap vs layer (10 pairs from 5 industries)
+ legend outside
+ trend-friendly view: spaghetti (thin) + mean line (thick) + IQR band

Input:
  - overlap_matrix_layer_XX.csv files in --in-dir

Output (in --out-dir):
  - pairwise_overlap_by_layer.csv
  - pairwise_overlap_lines_trend.png
"""

import os, re, glob, argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

SHORT_ORDER = ["Computer", "Financial", "Management", "Sales", "Educational"]

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def parse_layer_from_path(p: str) -> int:
    m = re.search(r"layer_(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else -1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=str, required=True,
                    help="Directory containing overlap_matrix_layer_*.csv")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--font-size", type=int, default=14)
    ap.add_argument("--normalize-by-topk", type=int, default=0,
                    help="If >0, plot overlap/topk instead of raw counts (e.g., 128)")
    ap.add_argument("--alpha-lines", type=float, default=0.35,
                    help="Opacity for individual pair lines")
    ap.add_argument("--lw-lines", type=float, default=1.5,
                    help="Line width for individual pair lines")
    ap.add_argument("--lw-mean", type=float, default=3.0,
                    help="Line width for mean trend line")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    mpl.rcParams.update({
        "font.size": args.font_size,
        "axes.titlesize": args.font_size + 2,
        "axes.labelsize": args.font_size,
        "xtick.labelsize": args.font_size - 2,
        "ytick.labelsize": args.font_size - 2,
        "legend.fontsize": args.font_size - 3,
    })

    files = sorted(glob.glob(os.path.join(args.in_dir, "overlap_matrix_layer_*.csv")))
    if not files:
        raise FileNotFoundError("No overlap_matrix_layer_*.csv found in --in-dir")

    # collect long-form
    rows = []
    for fp in files:
        layer = parse_layer_from_path(fp)
        df = pd.read_csv(fp, index_col=0)
        df = df.reindex(index=SHORT_ORDER, columns=SHORT_ORDER)
        M = df.to_numpy(dtype=float)

        K = len(SHORT_ORDER)
        for i in range(K):
            for j in range(i + 1, K):
                v = M[i, j]
                if args.normalize_by_topk and args.normalize_by_topk > 0:
                    v = v / float(args.normalize_by_topk)
                rows.append({
                    "layer": layer,
                    "pair": f"{SHORT_ORDER[i]}–{SHORT_ORDER[j]}",
                    "i": SHORT_ORDER[i],
                    "j": SHORT_ORDER[j],
                    "overlap": float(v),
                })

    long_df = pd.DataFrame(rows).sort_values(["pair", "layer"]).reset_index(drop=True)
    out_csv = os.path.join(args.out_dir, "pairwise_overlap_by_layer.csv")
    long_df.to_csv(out_csv, index=False)

    # ---------- trend-friendly plot ----------
    layers_sorted = sorted(long_df["layer"].unique().tolist())

    # pivot: rows=layer, cols=pair
    piv = long_df.pivot_table(index="layer", columns="pair", values="overlap", aggfunc="mean").reindex(layers_sorted)

    # summary across pairs at each layer
    mean = piv.mean(axis=1)
    q25 = piv.quantile(0.25, axis=1)
    q75 = piv.quantile(0.75, axis=1)

    fig = plt.figure(figsize=(10.8, 5.8))
    ax = fig.add_subplot(1, 1, 1)

    # IQR band first (so lines sit on top)
    ax.fill_between(layers_sorted, q25.values, q75.values, alpha=0.18, label="IQR (25–75%)")

    # spaghetti lines (thin)
    for pair in piv.columns:
        ax.plot(layers_sorted, piv[pair].values, marker="o",
                linewidth=args.lw_lines, alpha=args.alpha_lines, label=pair)

    # mean trend (thick)
    ax.plot(layers_sorted, mean.values, marker="o", linewidth=args.lw_mean,
            label="Mean across 10 pairs")

    ax.set_xticks(layers_sorted)
    ax.set_xlabel("Layer", fontsize = 26)
    ylabel = "Overlap" if not (args.normalize_by_topk and args.normalize_by_topk > 0) else f"Overlap / {args.normalize_by_topk}"
    ax.set_ylabel(ylabel, fontsize = 26)

    title = ""
    if args.normalize_by_topk and args.normalize_by_topk > 0:
        title += f" (normalized)"
    ax.set_title(title)

    ax.grid(False)

    # legend OUTSIDE (right)
    # (IQR + Mean + 10 pairs = 12 entries; 2 columns helps)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels,
              loc="center left",
              bbox_to_anchor=(1.02, 0.5),
              frameon=True,
              ncol=1)

    plt.tight_layout(rect=[0.0, 0.0, 0.80, 1.0])  # leave space for legend
    out_png = os.path.join(args.out_dir, "pairwise_overlap_lines_trend.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print("[DONE]")
    print(f"  saved csv : {out_csv}")
    print(f"  saved plot: {out_png}")

if __name__ == "__main__":
    main()

    """
    python3 overlap_by_industry_graph.py   --in-dir ./python_industry/overlaps   --out-dir ./python_industry/overlaps     --alpha-lines 0.25

    """