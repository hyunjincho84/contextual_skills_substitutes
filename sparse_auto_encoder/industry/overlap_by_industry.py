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


def plot_overlap_heatmap(mat, labels, out_png, title):
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(1, 1, 1)

    im = ax.imshow(mat, cmap="RdBu_r", vmin=0, vmax=int(mat.max()) if mat.max() > 0 else 1)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize = 15)
    ax.set_yticklabels(labels, fontsize = 15)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title, pad=12, fontsize=26)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


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
    args = ap.parse_args()

    ensure_dir(args.out_dir)

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

    # -------------------------
    # Layer loop
    # -------------------------
    for layer in tqdm(layers, desc="Layers"):
        subL = df[df["layer"] == layer].copy()
        if len(subL) == 0:
            continue

        # sample per industry
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

        # mean activation cache
        X_by_ind = {
            ind: subL[subL["field_short"] == ind][feat_cols].to_numpy(np.float32)
            for ind in SHORT_ORDER
        }

        # -------------------------
        # ITERATIVE COMMON REMOVAL
        # -------------------------
        blacklist = set()
        iteration = 0

        while True:
            iteration += 1
            if iteration > args.max_iters:
                print(f"[WARN] layer {layer}: reached max_iter={args.max_iters}")
                break

            top_sets = {}
            for ind in SHORT_ORDER:
                X = X_by_ind[ind]
                if len(X) == 0:
                    top_sets[ind] = np.array([], dtype=np.int64)
                else:
                    top_sets[ind] = topk_feature_ids(
                        X, args.topk, exclude=blacklist
                    )

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

        # -------------------------
        # Final overlap matrix
        # -------------------------
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

        # save csv + heatmap
        pd.DataFrame(M, index=SHORT_ORDER, columns=SHORT_ORDER).to_csv(
            os.path.join(args.out_dir, f"overlap_matrix_layer_{layer:02d}.csv")
        )

        plot_overlap_heatmap(
            M,
            SHORT_ORDER,
            os.path.join(args.out_dir, f"heatmap_overlap_layer_{layer:02d}.png"),
            title=f"Layer {layer:02d}",
        )

    # -------------------------
    # Save logs
    # -------------------------
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
python3 overlap_by_industry.py   --features-parquet ./python_industry/features.parquet   --out-dir ./python_industry/overlaps   --topk 128   --layers 1-12   --samples-per-industry 250   --font-size 16


"""

