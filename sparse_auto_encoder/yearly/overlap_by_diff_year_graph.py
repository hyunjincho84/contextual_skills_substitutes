#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_layer_from_path(p: str) -> int:
    m = re.search(r"layer_(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else -1

def to_int_year(x):
    return int(str(x).strip().replace(".0", ""))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="dir with overlap_matrix_layer_*.csv")
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--layers", default="1,4,8,12", help="comma layers to plot")
    ap.add_argument("--agg", default="mean", choices=["mean","median"], help="per-layer center line")
    ap.add_argument("--alpha-lines", type=float, default=0.30, help="opacity for layer lines")
    ap.add_argument("--lw-lines", type=float, default=1.8, help="linewidth for layer lines")
    ap.add_argument("--lw-global", type=float, default=3.2, help="linewidth for global mean")
    ap.add_argument("--iqr-alpha", type=float, default=0.18, help="alpha for IQR band")
    ap.add_argument("--min-diff-count", type=int, default=1,
                    help="require at least this many pairs for a given Δyear bucket")
    args = ap.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    if not layers:
        raise ValueError("--layers is empty")

    # layer -> dict(diff -> list overlaps)
    per_layer_bucket = {}

    for layer in layers:
        fp = os.path.join(args.in_dir, f"overlap_matrix_layer_{layer:02d}.csv")
        if not os.path.isfile(fp):
            fp2 = os.path.join(args.in_dir, f"overlap_matrix_layer_{layer}.csv")
            if os.path.isfile(fp2):
                fp = fp2
            else:
                raise FileNotFoundError(f"Missing: {fp}")

        df = pd.read_csv(fp, index_col=0)
        df.index = [to_int_year(x) for x in df.index]
        df.columns = [to_int_year(x) for x in df.columns]

        years = sorted(set(df.index).intersection(set(df.columns)))
        if len(years) < 2:
            raise RuntimeError(f"Layer {layer}: <2 years found in matrix")

        df = df.reindex(index=years, columns=years)
        M = df.to_numpy(float)

        bucket = {}  # diff -> list overlaps
        n = len(years)
        for i in range(n):
            for j in range(i + 1, n):
                d = abs(years[i] - years[j])
                bucket.setdefault(d, []).append(float(M[i, j]))

        # optional filter: drop diffs with too few samples
        bucket = {d: vs for d, vs in bucket.items() if len(vs) >= args.min_diff_count}
        per_layer_bucket[layer] = bucket

    # ---- common Δyear axis across layers (intersection so curves align) ----
    common_diffs = None
    for layer in layers:
        diffs = set(per_layer_bucket[layer].keys())
        common_diffs = diffs if common_diffs is None else (common_diffs & diffs)

    if not common_diffs:
        raise RuntimeError("No common Δyear buckets across layers. "
                           "Try lowering --min-diff-count or check your year coverage.")

    diffs = sorted(common_diffs)

    # ---- build per-layer center curves + global distribution per Δyear ----
    layer_center = {}   # layer -> np.array(len(diffs))
    all_values_by_diff = {d: [] for d in diffs}  # aggregate across all layers + all pairs

    for layer in layers:
        ys = []
        for d in diffs:
            v = np.array(per_layer_bucket[layer][d], dtype=float)
            # collect for global stats
            all_values_by_diff[d].extend(v.tolist())

            if args.agg == "mean":
                ys.append(float(v.mean()))
            else:
                ys.append(float(np.median(v)))
        layer_center[layer] = np.array(ys, dtype=float)

    # global mean + IQR across (all layers, all year-pairs) per diff
    global_mean = np.array([np.mean(all_values_by_diff[d]) for d in diffs], dtype=float)
    global_q25  = np.array([np.quantile(all_values_by_diff[d], 0.25) for d in diffs], dtype=float)
    global_q75  = np.array([np.quantile(all_values_by_diff[d], 0.75) for d in diffs], dtype=float)

    # ---- plot ----
    fig = plt.figure(figsize=(12.0, 6.0))
    ax = fig.add_subplot(1, 1, 1)

    # IQR band first
    ax.fill_between(diffs, global_q25, global_q75, alpha=args.iqr_alpha, label="IQR (25–75%)")

    # layer lines (thin)
    for layer in layers:
        ax.plot(
            diffs,
            layer_center[layer],
            marker="o",
            linewidth=args.lw_lines,
            alpha=args.alpha_lines,
            label=f"Layer {layer}",
        )

    # global mean (thick)
    ax.plot(
        diffs,
        global_mean,
        marker="o",
        linewidth=args.lw_global,
        label=f"Global mean",
    )

    ax.set_xlabel("Year difference |Δyear|", fontsize=26)
    ax.set_ylabel(f"Overlap", fontsize=26)
    ax.grid(False)
    ax.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[DONE]", args.out_png)


if __name__ == "__main__":
    main()


"""
python3 overlap_by_year_diff_graph.py \
  --in-dir  ./python_yearly/overlaps \
  --out-png ./python_yearly/overlaps/overlap_diff.png \ 
  --layers 1,2,3,4,5,6,7,8,9,10,11,12 \
  --agg mean \
  --alpha-lines 0.25
"""