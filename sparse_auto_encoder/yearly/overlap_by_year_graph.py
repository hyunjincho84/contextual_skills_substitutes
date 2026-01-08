#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot pairwise overlap vs layer for YEAR-based overlap matrices (RAW COUNTS ONLY).

Input:
  - overlap_matrix_layer_XX.csv files in --in-dir
    (rows/cols are years, e.g., 2010,2011,...)

Output (in --out-dir):
  - pairwise_overlap_by_layer.csv
  - pairwise_overlap_lines_trend.png
"""

import os, re, glob, argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def parse_layer_from_path(p: str) -> int:
    m = re.search(r"layer_(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else -1

def natural_year_sort(xs):
    try:
        return sorted(xs, key=lambda z: int(str(z)))
    except Exception:
        return sorted(xs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=str, required=True,
                    help="Directory containing overlap_matrix_layer_*.csv")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--font-size", type=int, default=14)
    ap.add_argument("--alpha-lines", type=float, default=0.35,
                    help="Opacity for individual pair lines")
    ap.add_argument("--lw-lines", type=float, default=1.5,
                    help="Line width for individual pair lines")
    ap.add_argument("--lw-mean", type=float, default=3.0,
                    help="Line width for mean trend line")
    ap.add_argument("--order", type=str, default="",
                    help="Optional explicit order of year labels, e.g. '2010,2011,2012' (otherwise inferred)")
    ap.add_argument("--legend-max-pairs", type=int, default=12,
                    help="If too many year-pairs, only show up to this many pair labels in legend (plus IQR/Mean).")
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

    # --- determine YEAR labels / order ---
    if args.order.strip():
        ORDER = [x.strip() for x in args.order.split(",") if x.strip()]
    else:
        df0 = pd.read_csv(files[0], index_col=0)
        ORDER = [str(x) for x in df0.index.tolist()]
        ORDER = natural_year_sort(ORDER)

    # collect long-form
    rows = []
    for fp in files:
        layer = parse_layer_from_path(fp)
        df = pd.read_csv(fp, index_col=0)

        # force labels to string for consistent matching
        df.index = df.index.map(lambda x: str(x))
        df.columns = df.columns.map(lambda x: str(x))

        # align to ORDER (only keep intersection)
        keep = [y for y in ORDER if (y in df.index and y in df.columns)]
        if len(keep) < 2:
            print(f"[WARN] layer={layer}: <2 years after alignment. skip.")
            continue

        df = df.reindex(index=keep, columns=keep)
        M = df.to_numpy(dtype=float)

        K = len(keep)
        for i in range(K):
            for j in range(i + 1, K):
                v = M[i, j]  # RAW COUNT
                rows.append({
                    "layer": int(layer),
                    "pair": f"{keep[i]}–{keep[j]}",
                    "i": keep[i],
                    "j": keep[j],
                    "overlap": float(v),
                })

    if not rows:
        raise RuntimeError("No pairwise rows collected. Check your input matrices / ordering.")

    long_df = pd.DataFrame(rows).sort_values(["pair", "layer"]).reset_index(drop=True)
    out_csv = os.path.join(args.out_dir, "pairwise_overlap_by_layer.csv")
    long_df.to_csv(out_csv, index=False)

    # ---------- trend-friendly plot ----------
    layers_sorted = sorted(long_df["layer"].unique().tolist())

    piv = (long_df
           .pivot_table(index="layer", columns="pair", values="overlap", aggfunc="mean")
           .reindex(layers_sorted))

    mean = piv.mean(axis=1)
    q25 = piv.quantile(0.25, axis=1)
    q75 = piv.quantile(0.75, axis=1)

    fig = plt.figure(figsize=(10.8, 5.8))
    ax = fig.add_subplot(1, 1, 1)

    ax.fill_between(layers_sorted, q25.values, q75.values, alpha=0.18, label="IQR (25–75%)")

    pair_cols = list(piv.columns)
    show_in_legend = set(pair_cols[:max(0, int(args.legend_max_pairs))])

    for pair in pair_cols:
        ax.plot(
            layers_sorted,
            piv[pair].values,
            marker="o",
            linewidth=args.lw_lines,
            alpha=args.alpha_lines,
            label=(pair if pair in show_in_legend else "_nolegend_"),
        )

    ax.plot(
        layers_sorted,
        mean.values,
        marker="o",
        linewidth=args.lw_mean,
        label=f"Mean across {len(pair_cols)} pairs",
    )

    ax.set_xticks(layers_sorted)
    ax.set_xlabel("Layer", fontsize=26)
    ax.set_ylabel("Overlap", fontsize=26)

    ax.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        ncol=1,
    )

    plt.tight_layout(rect=[0.0, 0.0, 0.80, 1.0])
    out_png = os.path.join(args.out_dir, "pairwise_overlap_lines_trend.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print("[DONE]")
    print(f"  saved csv : {out_csv}")
    print(f"  saved plot: {out_png}")

if __name__ == "__main__":
    main()

"""
Example (RAW COUNTS):
python3 overlap_by_year_graph.py \
  --in-dir  ./python_yearly/overlaps \
  --out-dir ./python_yearly/overlaps \
  --alpha-lines 0.20
"""