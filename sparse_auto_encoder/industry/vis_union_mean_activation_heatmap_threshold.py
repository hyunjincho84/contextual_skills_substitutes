#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Union(Top-K per industry) heatmap per layer
+ iterative removal of ALL-industry-common features
+ NEW: layer-wise top-q% binary visualization (2-color)

Binary threshold:
  - For each layer, threshold = q-quantile of union activation values
  - Activation >= threshold -> 1
  - Else -> 0
"""

import os, re, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from matplotlib.colors import ListedColormap, BoundaryNorm

from matplotlib.colors import ListedColormap

INDUSTRY_COLORS = {
    "Computer":     "#4CAF50",
    "Financial":    "#2196F3",
    "Management":   "#FF9800",
    "Sales":        "#9C27B0",
    "Educational":  "#795548",
}
OFF_COLOR = "#D3D3D3"

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


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def get_feat_cols(df):
    cols = [c for c in df.columns if re.fullmatch(r"f\d+", c)]
    return sorted(cols, key=lambda x: int(x[1:]))


def topk_ranked_feature_ids_from_mean(mu, k):
    k = min(k, mu.shape[0])
    idx = np.argpartition(-mu, k - 1)[:k]
    idx = idx[np.argsort(-mu[idx])]
    return idx.astype(np.int64)


def hex_to_rgb01(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0,2,4)]) / 255.0

def binary_to_rgb(bin_mat, row_labels):
    K, M = bin_mat.shape
    img = np.zeros((K, M, 3), dtype=np.float32)

    off = hex_to_rgb01(OFF_COLOR)

    for i, ind in enumerate(row_labels):
        on = hex_to_rgb01(INDUSTRY_COLORS[ind])
        for j in range(M):
            img[i, j] = on if bin_mat[i, j] == 1 else off

    return img

def reorder_columns_by_feature_clustering(mat, union_ids, n_clusters=12):
    Xf = normalize(mat.T.astype(np.float32), axis=1)
    try:
        clu = AgglomerativeClustering(
            n_clusters=n_clusters, metric="cosine", linkage="average"
        )
    except TypeError:
        clu = AgglomerativeClustering(
            n_clusters=n_clusters, affinity="cosine", linkage="average"
        )
    labels = clu.fit_predict(Xf)
    dominant = np.nanargmax(mat, axis=0)
    order = np.lexsort((union_ids, dominant, labels))
    return mat[:, order], union_ids[order], labels[order]


def iterative_common_removed_topk(ranked_lists, topk, max_rounds=10000):
    inds = list(ranked_lists.keys())
    selected = {ind: list(ranked_lists[ind][:topk]) for ind in inds}
    removed_all = set()
    rounds = 0

    while rounds < max_rounds:
        rounds += 1
        common = set.intersection(*[set(selected[i]) for i in inds])
        if not common:
            break
        removed_all |= common
        for ind in inds:
            new_sel = []
            for fid in ranked_lists[ind]:
                if fid in removed_all or fid in new_sel:
                    continue
                new_sel.append(int(fid))
                if len(new_sel) >= topk:
                    break
            selected[ind] = new_sel
    return selected, removed_all, rounds


# ---------- NEW: quantile threshold ----------
def compute_layer_quantile_threshold(mat, q):
    vals = mat[np.isfinite(mat)].ravel()
    return float(np.quantile(vals, q))


def binarize_by_threshold(mat, thr):
    return (mat >= thr).astype(np.int8)


def plot_binary_union_heatmap(
    bin_mat,
    row_labels,
    out_png,
    title,
):
    rgb_img = binary_to_rgb(bin_mat, row_labels)

    fig = plt.figure(figsize=(18, 3.2))
    ax = fig.add_subplot(1, 1, 1)

    ax.imshow(rgb_img, aspect="auto")

    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=15)
    ax.set_xticks([])
    ax.set_ylabel("")
    ax.set_title(title, fontsize=26, pad=10)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_layers(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--layers", default="1-12")
    ap.add_argument("--samples-per-industry", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--remove-all-common", action="store_true")
    ap.add_argument("--candidate-mult", type=int, default=10)

    ap.add_argument("--col-order", default="id", choices=["id", "dominant", "cluster"])
    ap.add_argument("--n-clusters", type=int, default=12)

    # NEW
    ap.add_argument("--quantile", type=float, default=0.95)

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    layers = parse_layers(args.layers)

    df = pd.read_parquet(args.features_parquet)
    df["field_short"] = df["field"].map(FIELD_LABEL_MAP)
    df = df[df["field_short"].isin(SHORT_ORDER)].copy()

    feat_cols = get_feat_cols(df)
    d = len(feat_cols)
    print(f"[INFO] feature dim = {d}")

    for layer in tqdm(layers, desc="Layers"):
        sub = df[df["layer"] == layer].copy()
        if sub.empty:
            continue

        sub = (
            sub.groupby("field_short", group_keys=False)
               .apply(lambda g: g.sample(
                   n=min(len(g), args.samples_per_industry),
                   random_state=args.seed))
               .reset_index(drop=True)
        )

        mu_by_ind = {}
        ranked_by_ind = {}
        cand_k = min(d, args.topk * args.candidate_mult)

        for ind in SHORT_ORDER:
            g = sub[sub["field_short"] == ind]
            X = g[feat_cols].to_numpy(np.float32)
            mu = X.mean(axis=0)
            mu_by_ind[ind] = mu
            ranked_by_ind[ind] = topk_ranked_feature_ids_from_mean(mu, cand_k)

        if args.remove_all_common:
            selected, removed, rounds = iterative_common_removed_topk(
                ranked_by_ind, args.topk
            )
            print(f"[L{layer:02d}] removed common = {len(removed)}")
            top_by_ind = selected
        else:
            top_by_ind = {i: ranked_by_ind[i][:args.topk] for i in SHORT_ORDER}

        union_ids = np.unique(np.concatenate(list(top_by_ind.values())))
        union_ids.sort()

        mat = np.vstack([mu_by_ind[ind][union_ids] for ind in SHORT_ORDER])

        if args.col_order == "cluster":
            mat, union_ids, _ = reorder_columns_by_feature_clustering(
                mat, union_ids, args.n_clusters
            )

        # ---------- binary visualization ----------
        thr = compute_layer_quantile_threshold(mat, args.quantile)
        bin_mat = binarize_by_threshold(mat, thr)

        out_png = os.path.join(
            args.out_dir,
            f"union_binary_q{int(args.quantile*100)}_layer_{layer:02d}.png"
        )

        plot_binary_union_heatmap(
            bin_mat,
            SHORT_ORDER,
            out_png,
            title=f"Layer {layer:02d}"
        )

    print("[DONE]")


if __name__ == "__main__":
    main()

"""
python3 vis_union_mean_activation_heatmap_threshold.py \
  --features-parquet ./python_industry/features.parquet \
  --out-dir ./python_industry/feature_id_vis \
  --layers 1-12 \
  --topk 64 \
  --samples-per-industry 250 \
  --remove-all-common \
  --candidate-mult 10 \
  --col-order cluster \
  --n-clusters 12 \
  --quantile 0.75    
"""