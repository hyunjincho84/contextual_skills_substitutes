#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt


# -------------------------
# CONFIG
# -------------------------
FIELD_LABEL_MAP = {
    "Computer and Mathematical Occupations": "Computer",
    "Business and Financial Operations Occupations": "Financial",
    "Management Occupations": "Management",
    "Sales and Related Occupations": "Sales",
    "Educational Instruction and Library Occupations": "Educational",
}

SHORT_ORDER = ["Computer", "Financial", "Management", "Sales", "Educational"]


# -------------------------
# Utils
# -------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def get_feat_cols(df):
    cols = [c for c in df.columns if re.fullmatch(r"f\d+", c)]
    return sorted(cols, key=lambda x: int(x[1:]))

def topk_feature_ids(X, topk, exclude=None):
    """
    X: (n, d)
    exclude: set of feature indices to skip
    """
    mu = X.mean(axis=0)
    d = mu.shape[0]

    if exclude:
        mu = mu.copy()
        mu[list(exclude)] = -np.inf

    topk = min(topk, d - (len(exclude) if exclude else 0))
    idx = np.argpartition(-mu, topk - 1)[:topk]
    idx = idx[np.argsort(-mu[idx])]
    return idx.astype(np.int64)

def overlap_count(a, b):
    return len(np.intersect1d(a, b, assume_unique=False))

def plot_overlap_grid(mats, layer_ids, labels, out_png, title_prefix="Layer", ncols=4, dpi=300):
    """
    mats: List[np.ndarray], each (K,K)
    layer_ids: List[int]
    labels: List[str]
    Saves ONE grid image with a dedicated colorbar axis (no overlap).
    """
    assert len(mats) == len(layer_ids)

    n = len(mats)
    if n == 0:
        print("[WARN] No matrices to plot (empty).")
        return

    K = len(labels)
    nrows = int(np.ceil(n / ncols))

    # shared vmax for consistent color scale
    vmax = int(max([m.max() for m in mats])) if mats else 1
    vmax = max(vmax, 1)

    # ✅ Make extra column for colorbar (GridSpec)
    width_ratios = [1.0] * ncols + [0.06]  # last column reserved for colorbar
    fig = plt.figure(figsize=(4.2 * ncols + 1.2, 3.8 * nrows))

    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols + 1,
        width_ratios=width_ratios,
        wspace=0.35,   # horizontal spacing
        hspace=0.45,   # vertical spacing
    )

    axes = []
    for r in range(nrows):
        for c in range(ncols):
            axes.append(fig.add_subplot(gs[r, c]))

    # colorbar axis occupies the whole last column
    cax = fig.add_subplot(gs[:, -1])

    im_for_cbar = None

    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue

        M = mats[i]
        layer = layer_ids[i]

        im = ax.imshow(M, cmap="RdBu_r", vmin=0, vmax=vmax, aspect="auto")
        if im_for_cbar is None:
            im_for_cbar = im

        ax.set_title(f"{title_prefix} {layer:02d}", fontsize=18, pad=8)
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
        ax.set_yticklabels(labels, fontsize=11)

        for rr in range(K):
            for cc in range(K):
                ax.text(cc, rr, int(M[rr, cc]), ha="center", va="center", fontsize=9)

    if im_for_cbar is not None:
        cbar = fig.colorbar(im_for_cbar, cax=cax)
        cbar.set_label("Overlap", fontsize=12)

    # ✅ IMPORTANT: don't use tight_layout / bbox_inches="tight" with GridSpec+cax
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--topk", type=int, default=128)
    ap.add_argument("--layers", default="1-12")
    ap.add_argument("--samples-per-industry", type=int, default=250)
    ap.add_argument("--font-size", type=int, default=16)
    ap.add_argument("--max-iters", type=int, default=20,
                    help="safety cap for iterative common-feature removal")

    ap.add_argument("--grid-out-name", type=str, default="heatmap_overlap_all_layers.png",
                    help="Output filename for the combined grid image.")
    ap.add_argument("--grid-cols", type=int, default=4)

    # ✅ always save per-layer CSVs into overlaps/csvs
    ap.add_argument("--csv-subdir", type=str, default="csvs",
                    help="Subdirectory under --out-dir to store per-layer CSVs. (default: csvs)")

    args = ap.parse_args()

    ensure_dir(args.out_dir)
    csv_dir = os.path.join(args.out_dir, args.csv_subdir)
    ensure_dir(csv_dir)

    mpl.rcParams.update({
        "font.size": args.font_size,
        "axes.titlesize": args.font_size + 2,
        "axes.labelsize": args.font_size,
    })

    # parse layers
    if "-" in args.layers:
        a, b = args.layers.split("-")
        layers = list(range(int(a), int(b) + 1))
    else:
        layers = [int(x) for x in args.layers.split(",")]

    print(f"[INFO] loading {args.features_parquet}")
    df = pd.read_parquet(args.features_parquet)

    df["field_short"] = df["field"].map(FIELD_LABEL_MAP)
    df = df.dropna(subset=["field_short"])
    df = df[df["field_short"].isin(SHORT_ORDER)]

    feat_cols = get_feat_cols(df)
    d = len(feat_cols)
    print(f"[INFO] feature dim = {d}")

    all_overlap_rows = []
    all_counts = []

    mats = []
    layer_ids = []

    for layer in tqdm(layers, desc="Layers"):
        subL = df[df["layer"] == layer].copy()
        if len(subL) == 0:
            continue

        subL = (
            subL.groupby("field_short", group_keys=False)
                .apply(lambda g: g.sample(
                    n=min(len(g), args.samples_per_industry),
                    random_state=42))
                .reset_index(drop=True)
        )

        cnt = subL["field_short"].value_counts().reindex(SHORT_ORDER).fillna(0).astype(int)
        for ind, n in cnt.items():
            all_counts.append({"layer": layer, "industry": ind, "n_samples": int(n)})

        X_by_ind = {
            ind: subL[subL["field_short"] == ind][feat_cols].to_numpy(np.float32)
            for ind in SHORT_ORDER
        }

        blacklist = set()
        iteration = 0
        top_sets = {}

        while True:
            iteration += 1
            if iteration > args.max_iters:
                print(f"[WARN] layer {layer}: reached max_iter={args.max_iters}")
                break

            for ind in SHORT_ORDER:
                X = X_by_ind[ind]
                if len(X) == 0:
                    top_sets[ind] = np.array([], dtype=np.int64)
                else:
                    top_sets[ind] = topk_feature_ids(X, args.topk, exclude=blacklist)

            common = set(top_sets[SHORT_ORDER[0]])
            for ind in SHORT_ORDER[1:]:
                common &= set(top_sets[ind])

            print(
                f"[Layer {layer:02d}] iter={iteration} | "
                f"new_common={len(common)} | total_excluded={len(blacklist)}"
            )

            if len(common) == 0:
                break

            blacklist |= common

        print(
            f"[Layer {layer:02d}] DONE | "
            f"iterations={iteration}, "
            f"excluded={len(blacklist)}/{d}"
        )

        K = len(SHORT_ORDER)
        M = np.zeros((K, K), dtype=np.int32)

        for i, a in enumerate(SHORT_ORDER):
            for j, b in enumerate(SHORT_ORDER):
                M[i, j] = overlap_count(top_sets[a], top_sets[b])
                all_overlap_rows.append({
                    "layer": layer,
                    "industry_i": a,
                    "industry_j": b,
                    "overlap": int(M[i, j]),
                    "topk": args.topk,
                    "excluded_features": len(blacklist),
                })

        # ✅ ALWAYS save per-layer CSV into overlaps/csvs/
        out_csv = os.path.join(csv_dir, f"overlap_matrix_layer_{layer:02d}.csv")
        pd.DataFrame(M, index=SHORT_ORDER, columns=SHORT_ORDER).to_csv(out_csv)

        mats.append(M)
        layer_ids.append(layer)

    # ✅ ONE combined grid image
    out_grid = os.path.join(args.out_dir, args.grid_out_name)
    plot_overlap_grid(
        mats=mats,
        layer_ids=layer_ids,
        labels=SHORT_ORDER,
        out_png=out_grid,
        title_prefix="Layer",
        ncols=args.grid_cols,
        dpi=300,
    )
    print(f"[OK] Saved combined heatmap grid -> {out_grid}")
    print(f"[OK] Saved per-layer CSVs -> {csv_dir}")

    pd.DataFrame(all_overlap_rows).to_csv(
        os.path.join(args.out_dir, "overlap_longform_all_layers.csv"),
        index=False,
    )
    pd.DataFrame(all_counts).to_csv(
        os.path.join(args.out_dir, "sample_counts_per_layer.csv"),
        index=False,
    )

    print("[DONE]")
    print(f"Saved results to: {args.out_dir}")


if __name__ == "__main__":
    main()
    
    
"""
python3 overlap_by_industry.py \
  --features-parquet ./python_industry/features.parquet \
  --out-dir ./python_industry/overlaps \
  --topk 128 \
  --layers 1-12 \
  --samples-per-industry 250 \
  --font-size 16 \
  --grid-cols 4 \
  --grid-out-name heatmap_overlap_all_layers.png
"""