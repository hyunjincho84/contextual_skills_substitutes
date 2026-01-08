#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Union(Top-K per industry) heatmap per layer
+ (NEW) iterative removal of features that are in top-K for ALL industries.

Procedure (per layer):
  1) For each industry, compute mean activation vector mu_ind
  2) Get a candidate ranked list of features per industry (top-K * expand)
  3) Initialize each industry's selection S_ind as first topK from its list
  4) Repeat:
       common = intersection over industries of S_ind
       if common empty: stop
       remove common from every industry's selection
       fill back to size topK from that industry's ranked list skipping removed
  5) Union across industries => U
  6) Build mean-activation matrix (industries x |U|)
  7) Optional column order: id | dominant | cluster

Outputs:
  - union_mean_heatmap_layer_XX_<order>.png
  - union_mean_layer_XX_<order>.csv
  - union_meta_layer_XX_<order>.csv
  - sample_counts_per_layer.csv
  - (NEW) removed_common_per_layer.csv  (how many features were globally removed)

Dependencies:
  - pandas, numpy, matplotlib, tqdm
  - scikit-learn (for --col-order cluster)
"""

import os, re, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize


# -------------------------
# CONFIG: label map (원본 → 짧은 이름)
# -------------------------
FIELD_LABEL_MAP = {
    "Computer and Mathematical Occupations": "Computer",
    "Business and Financial Operations Occupations": "Financial",
    "Management Occupations": "Management",
    "Sales and Related Occupations": "Sales",
    "Educational Instruction and Library Occupations": "Educational",
}
SHORT_ORDER = ["Computer", "Financial", "Management", "Sales", "Educational"]


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def get_feat_cols(df: pd.DataFrame):
    feat_cols = [c for c in df.columns if re.fullmatch(r"f\d+", c)]
    feat_cols = sorted(feat_cols, key=lambda x: int(x[1:]))
    return feat_cols


def topk_ranked_feature_ids_from_mean(mu: np.ndarray, k: int) -> np.ndarray:
    """
    mu: (d,)
    return: top-k feature indices by mu (descending), sorted by score
    """
    k = min(k, mu.shape[0])
    idx = np.argpartition(-mu, k - 1)[:k]
    idx = idx[np.argsort(-mu[idx])]
    return idx.astype(np.int64)


def reorder_columns_by_feature_clustering(mat: np.ndarray, union_ids: np.ndarray, n_clusters: int = 12):
    """
    Method 1: feature activation profile clustering
      - feature vector = activation across industries (K-dim)
      - L2 normalize -> cosine
      - AgglomerativeClustering(metric=cosine, linkage=average)
      - sort by (cluster, dominant_industry, feature_id)
    """
    Xf = mat.T.astype(np.float32)     # (M, K)
    Xf = normalize(Xf, axis=1)

    try:
        clu = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
    except TypeError:
        clu = AgglomerativeClustering(n_clusters=n_clusters, affinity="cosine", linkage="average")

    labels = clu.fit_predict(Xf)     # (M,)
    dominant = np.nanargmax(mat, axis=0)

    order = np.lexsort((union_ids, dominant, labels))
    return mat[:, order], union_ids[order], labels[order]


def plot_union_heatmap(mat, row_labels, out_png, title, cmap="winter", vmin=None, vmax=None):
    fig = plt.figure(figsize=(18, 3.2))
    ax = fig.add_subplot(1, 1, 1)

    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels,fontsize = 15)
    ax.set_xticks([])
    ax.set_ylabel("")
    ax.set_title(title, pad=10,fontsize=26)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("", rotation=90)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def iterative_common_removed_topk(
    ranked_lists: dict,
    topk: int,
    max_rounds: int = 10_000,
) -> tuple[dict, set, int]:
    """
    ranked_lists[ind] = np.ndarray of feature_ids sorted by descending score (long enough)

    Returns:
      selected: dict[ind] -> list of length topk (after removing ALL-common iteratively)
      removed_all_common: set of feature_ids removed because they were common across all industries
      rounds: number of iterations performed
    """
    inds = list(ranked_lists.keys())

    # helper: fill selection to topk skipping removed + already selected
    def fill(ind: str, selected_set: set, removed: set) -> list:
        out = []
        for fid in ranked_lists[ind]:
            if fid in removed:        # removed by "global common"
                continue
            if fid in selected_set:   # avoid duplicates within industry selection
                continue
            out.append(int(fid))
            if len(out) >= topk:
                break
        return out

    # initialize selections
    selected = {}
    for ind in inds:
        init = []
        seen = set()
        for fid in ranked_lists[ind]:
            if fid in seen:
                continue
            init.append(int(fid))
            seen.add(int(fid))
            if len(init) >= topk:
                break
        selected[ind] = init

    removed_all_common = set()
    rounds = 0

    while rounds < max_rounds:
        rounds += 1
        sets = [set(selected[ind]) for ind in inds]
        common = set.intersection(*sets)

        if len(common) == 0:
            break

        # remove these from everyone and remember
        removed_all_common.update(common)

        for ind in inds:
            kept = [fid for fid in selected[ind] if fid not in common]
            # refill
            kept_set = set(kept)
            refill = fill(ind, kept_set, removed_all_common)
            selected[ind] = refill  # refill already includes kept_set? no -> we need prepend kept
            # Actually refill() builds from scratch skipping removed/dups, so do:
            # Make sure kept stays first, then add new until topk.
            new_sel = kept[:]
            new_set = set(new_sel)
            for fid in ranked_lists[ind]:
                fid = int(fid)
                if fid in removed_all_common or fid in new_set:
                    continue
                new_sel.append(fid)
                new_set.add(fid)
                if len(new_sel) >= topk:
                    break
            selected[ind] = new_sel

        # safety: if any industry cannot fill to topk, stop (should be rare if ranked lists are long)
        if any(len(selected[ind]) < topk for ind in inds):
            break

    return selected, removed_all_common, rounds


def parse_layers(s: str):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-parquet", type=str, required=True,
                    help="Path to features.parquet (must include columns: field, layer, f0..fD)")
    ap.add_argument("--out-dir", type=str, required=True)

    ap.add_argument("--topk", type=int, default=64, help="Top-K per industry (for union)")
    ap.add_argument("--layers", type=str, default="1-12")
    ap.add_argument("--samples-per-industry", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)

    # NEW: iterative common removal
    ap.add_argument("--remove-all-common", action="store_true",
                    help="Iteratively remove features that appear in topK for ALL industries, then refill.")
    ap.add_argument("--candidate-mult", type=int, default=10,
                    help="How many candidates to keep per industry: topk * candidate_mult. (need enough to refill)")
    ap.add_argument("--max-rounds", type=int, default=10000)

    # column ordering
    ap.add_argument("--col-order", type=str, default="id", choices=["id", "dominant", "cluster"])
    ap.add_argument("--n-clusters", type=int, default=12)

    # plot
    ap.add_argument("--font-size", type=int, default=16)
    ap.add_argument("--cmap", type=str, default="winter")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    mpl.rcParams.update({
        "font.size": args.font_size,
        "axes.titlesize": args.font_size + 2,
        "axes.labelsize": args.font_size,
        "xtick.labelsize": args.font_size - 2,
        "ytick.labelsize": args.font_size - 2,
        "legend.fontsize": args.font_size - 2,
    })

    layers = parse_layers(args.layers)

    print(f"[INFO] loading parquet: {args.features_parquet}")
    df = pd.read_parquet(args.features_parquet)

    df["field_short"] = df["field"].map(FIELD_LABEL_MAP)
    df = df.dropna(subset=["field_short"]).copy()
    df = df[df["field_short"].isin(SHORT_ORDER)].copy()

    feat_cols = get_feat_cols(df)
    if len(feat_cols) == 0:
        raise RuntimeError("No feature columns like f0.. found in parquet.")
    d = len(feat_cols)
    print(f"[INFO] feature dim = {d}")

    all_sample_counts = []
    removed_log = []

    for layer in tqdm(layers, desc="Layers"):
        subL = df[df["layer"] == layer].copy()
        if len(subL) == 0:
            print(f"[WARN] layer={layer}: empty")
            continue

        # sample cap per industry
        subL = (
            subL.groupby("field_short", group_keys=False)
                .apply(lambda g: g.sample(n=min(len(g), args.samples_per_industry), random_state=args.seed))
                .reset_index(drop=True)
        )

        # sample counts
        cnt = subL["field_short"].value_counts().reindex(SHORT_ORDER).fillna(0).astype(int)
        for ind, n in cnt.items():
            all_sample_counts.append({"layer": int(layer), "industry": ind, "n_samples": int(n)})

        # mean vectors and ranked candidates
        mu_by_ind = {}
        ranked_by_ind = {}
        cand_k = min(d, args.topk * args.candidate_mult)

        for ind in SHORT_ORDER:
            g = subL[subL["field_short"] == ind]
            if len(g) == 0:
                mu_by_ind[ind] = None
                ranked_by_ind[ind] = np.array([], dtype=np.int64)
                continue

            X = g[feat_cols].to_numpy(np.float32)   # (n,d)
            mu = X.mean(axis=0)                     # (d,)
            mu_by_ind[ind] = mu
            ranked_by_ind[ind] = topk_ranked_feature_ids_from_mean(mu, cand_k)

        # choose topK per industry (optionally with iterative common removal)
        if args.remove_all_common:
            selected, removed_common, rounds = iterative_common_removed_topk(
                ranked_lists=ranked_by_ind,
                topk=args.topk,
                max_rounds=args.max_rounds,
            )
            n_removed = len(removed_common)
            print(f"[L{layer:02d}] removed common features = {n_removed} (iter rounds={rounds})")
            removed_log.append({
                "layer": int(layer),
                "topk": int(args.topk),
                "candidate_k": int(cand_k),
                "removed_common_count": int(n_removed),
                "iter_rounds": int(rounds),
            })
            top_by_ind = {ind: np.array(selected[ind], dtype=np.int64) for ind in SHORT_ORDER}
        else:
            top_by_ind = {}
            for ind in SHORT_ORDER:
                top_by_ind[ind] = ranked_by_ind[ind][:args.topk].astype(np.int64)
            removed_common = set()
            rounds = 0

        # union
        union_ids = np.unique(np.concatenate([top_by_ind[ind] for ind in SHORT_ORDER if top_by_ind[ind].size > 0]))
        union_ids = union_ids.astype(np.int64)
        union_ids.sort()

        if union_ids.size == 0:
            print(f"[WARN] layer={layer}: union_ids empty")
            continue

        # mean matrix (K x M)
        K = len(SHORT_ORDER)
        M = union_ids.size
        mat = np.zeros((K, M), dtype=np.float32)
        for i, ind in enumerate(SHORT_ORDER):
            mu = mu_by_ind[ind]
            mat[i, :] = np.nan if (mu is None) else mu[union_ids]

        # ----- column ordering -----
        col_cluster_labels = None
        if args.col_order == "id":
            pass
        elif args.col_order == "dominant":
            dominant = np.nanargmax(mat, axis=0)
            order = np.lexsort((union_ids, dominant))
            mat = mat[:, order]
            union_ids = union_ids[order]
        elif args.col_order == "cluster":
            mat, union_ids, col_cluster_labels = reorder_columns_by_feature_clustering(
                mat=mat, union_ids=union_ids, n_clusters=args.n_clusters
            )

        # save matrix csv
        out_csv = os.path.join(args.out_dir, f"union_mean_layer_{layer:02d}_{args.col_order}.csv")
        mat_df = pd.DataFrame(mat, index=SHORT_ORDER, columns=[f"f{fid}" for fid in union_ids])
        mat_df.to_csv(out_csv)

        # meta
        meta = pd.DataFrame({
            "layer": int(layer),
            "col_index": np.arange(len(union_ids), dtype=int),
            "feature_id": union_ids.astype(int),
        })
        if col_cluster_labels is not None:
            meta["cluster"] = col_cluster_labels.astype(int)
        out_meta = os.path.join(args.out_dir, f"union_meta_layer_{layer:02d}_{args.col_order}.csv")
        meta.to_csv(out_meta, index=False)

        # heatmap
        n_min = int(cnt.min())
        extra = "common_removed" if args.remove_all_common else "plain"
        title = f"Layer {layer:02d}"
        out_png = os.path.join(args.out_dir, f"union_mean_heatmap_layer_{layer:02d}_{args.col_order}.png")
        plot_union_heatmap(mat, SHORT_ORDER, out_png, title, cmap=args.cmap)

    # save sample counts
    out_counts = os.path.join(args.out_dir, "sample_counts_per_layer.csv")
    pd.DataFrame(all_sample_counts).to_csv(out_counts, index=False)

    # save removed log
    if args.remove_all_common:
        out_removed = os.path.join(args.out_dir, "removed_common_per_layer.csv")
        pd.DataFrame(removed_log).to_csv(out_removed, index=False)
        print(f"[INFO] saved removed log: {out_removed}")

    print("[DONE]")
    print(f"  out_dir: {args.out_dir}")
    print(f"  sample counts: {out_counts}")
    print("  per-layer outputs:")
    print("    - union_mean_heatmap_layer_XX_<order>.png")
    print("    - union_mean_layer_XX_<order>.csv")
    print("    - union_meta_layer_XX_<order>.csv")
    if args.remove_all_common:
        print("    - removed_common_per_layer.csv")


if __name__ == "__main__":
    main()

    """
    python3 vis_union_mean_activation_heatmap.py \
  --features-parquet ./python_industry/features.parquet \
  --out-dir ./python_industry/feature_id_vis \
  --layers 1-12 \
  --topk 64 \
  --samples-per-industry 250 \
  --remove-all-common \
  --candidate-mult 10 \
  --col-order cluster \
  --n-clusters 12 \
  --font-size 16 \
  --cmap winter
    """