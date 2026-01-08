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
FIELD_ORDER = ["Computer", "Financial", "Management", "Sales", "Educational"]


# -------------------------
# Utils
# -------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def get_feat_cols(df):
    cols = [c for c in df.columns if re.fullmatch(r"f\d+", c)]
    return sorted(cols, key=lambda x: int(x[1:]))

def topk_feature_ids(X, topk, exclude=None):
    mu = X.mean(axis=0)
    d = mu.shape[0]

    if exclude:
        mu = mu.copy()
        mu[list(exclude)] = -np.inf

    topk = min(topk, d - (len(exclude) if exclude else 0))
    if topk <= 0:
        return np.array([], dtype=np.int64)

    idx = np.argpartition(-mu, topk - 1)[:topk]
    idx = idx[np.argsort(-mu[idx])]
    return idx.astype(np.int64)

def overlap_count(a, b):
    return len(np.intersect1d(a, b, assume_unique=False))

def parse_layers(s: str):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip()]

def parse_years_arg(years_str: str):
    if years_str is None:
        return None
    years_str = years_str.strip().lower()
    if years_str == "all":
        return "all"
    if "-" in years_str:
        a, b = years_str.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in years_str.split(",") if x.strip()]

def plot_overlap_grid(mats, layer_ids, labels, out_png, title_prefix="Layer", ncols=4, dpi=300):
    """
    mats: list of (K,K)
    layer_ids: list of layer ints
    labels: tick labels for rows/cols (strings)
    """
    assert len(mats) == len(layer_ids)

    n = len(mats)
    if n == 0:
        print("[WARN] No matrices to plot (empty).")
        return

    nrows = int(np.ceil(n / ncols))
    vmax = int(max([m.max() for m in mats]))
    vmax = max(vmax, 1)

    K = len(labels)

    # ✅ GridSpec으로 컬러바 전용 column 확보 (마지막 컬럼)
    # width_ratios: [1,1,1,1,0.06] (ncols=4일 때)
    width_ratios = [1.0] * ncols + [0.06]
    fig = plt.figure(figsize=(4.2 * ncols + 1.0, 3.8 * nrows))

    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols + 1,
        width_ratios=width_ratios,
        wspace=0.35,   # subplot 간 간격
        hspace=0.45,   # row 간 간격
    )

    axes = []
    for r in range(nrows):
        for c in range(ncols):
            axes.append(fig.add_subplot(gs[r, c]))

    # 컬러바 전용 축: 오른쪽 마지막 column 전체 사용
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
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)

        # annotate numbers
        for rr in range(K):
            for cc in range(K):
                ax.text(cc, rr, int(M[rr, cc]), ha="center", va="center", fontsize=8)

    # ✅ 컬러바는 전용 축(cax)에만 그린다 -> 절대 안 겹침
    if im_for_cbar is not None:
        cbar = fig.colorbar(im_for_cbar, cax=cax)
        cbar.set_label("Overlap", fontsize=12)

    # ✅ tight_layout/bbox_inches="tight" 제거 (GridSpec이 이미 레이아웃 관리)
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
    ap.add_argument("--samples-per-group", type=int, default=250)

    ap.add_argument("--group-by", choices=["field", "year"], default="field")
    ap.add_argument("--years", default=None,
                    help="Only used when --group-by year. e.g., all | 2010-2023 | 2010,2012,2015")
    ap.add_argument("--max-groups", type=int, default=None,
                    help="Optional cap on number of groups (useful for years). Keeps earliest groups.")

    ap.add_argument("--font-size", type=int, default=16)
    ap.add_argument("--max-iters", type=int, default=100,
                    help="safety cap for iterative common-feature removal")

    # ✅ single-grid output controls
    ap.add_argument("--grid-out-name", type=str, default="heatmap_overlap_all_layers.png")
    ap.add_argument("--grid-cols", type=int, default=4)

    # ✅ per-layer CSV location: out_dir/csvs/
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

    layers = parse_layers(args.layers)

    print(f"[INFO] loading {args.features_parquet}")
    df = pd.read_parquet(args.features_parquet)

    feat_cols = get_feat_cols(df)
    d = len(feat_cols)
    print(f"[INFO] feature dim = {d}")

    # -------------------------
    # Choose grouping
    # -------------------------
    if args.group_by == "field":
        if "field" not in df.columns:
            raise ValueError("[ERROR] --group-by field requires column: field")

        df["group"] = df["field"].map(FIELD_LABEL_MAP)
        df = df.dropna(subset=["group"])
        df = df[df["group"].isin(FIELD_ORDER)]
        group_order = FIELD_ORDER[:]  # fixed order

    else:  # year
        if "year" not in df.columns:
            raise ValueError("[ERROR] --group-by year requires column: year")

        df["group"] = df["year"].astype(int)

        yrs = parse_years_arg(args.years)
        if yrs != "all" and yrs is not None:
            df = df[df["group"].isin(yrs)]

        group_order = sorted(df["group"].unique().tolist())
        if args.max_groups is not None:
            group_order = group_order[:args.max_groups]
            df = df[df["group"].isin(group_order)]

        group_order = [int(y) for y in group_order]

    labels = [str(x) for x in group_order]

    # -------------------------
    # Logs
    # -------------------------
    all_overlap_rows = []
    all_counts = []

    mats = []
    layer_ids = []

    # -------------------------
    # Layer loop
    # -------------------------
    for layer in tqdm(layers, desc="Layers"):
        subL = df[df["layer"] == layer].copy()
        if len(subL) == 0:
            continue

        # sample per group
        subL = (
            subL.groupby("group", group_keys=False)
                .apply(lambda g: g.sample(
                    n=min(len(g), args.samples_per_group),
                    random_state=42))
                .reset_index(drop=True)
        )

        cnt = subL["group"].value_counts().reindex(group_order).fillna(0).astype(int)
        for g, n in cnt.items():
            all_counts.append({
                "layer": int(layer),
                "group_by": args.group_by,
                "group": str(g),
                "n_samples": int(n)
            })

        X_by_group = {
            g: subL[subL["group"] == g][feat_cols].to_numpy(np.float32)
            for g in group_order
        }

        # -------------------------
        # ITERATIVE COMMON REMOVAL
        # -------------------------
        blacklist = set()
        iteration = 0
        top_sets = {g: np.array([], dtype=np.int64) for g in group_order}

        while True:
            iteration += 1
            if iteration > args.max_iters:
                print(f"[WARN] layer {layer}: reached max_iter={args.max_iters}")
                break

            for g in group_order:
                X = X_by_group[g]
                if len(X) == 0:
                    top_sets[g] = np.array([], dtype=np.int64)
                else:
                    top_sets[g] = topk_feature_ids(X, args.topk, exclude=blacklist)

            common = set(top_sets[group_order[0]]) if len(group_order) > 0 else set()
            for g in group_order[1:]:
                common &= set(top_sets[g])

            print(
                f"[Layer {int(layer):02d}] iter={iteration} | "
                f"new_common={len(common)} | total_excluded={len(blacklist)}"
            )

            if len(common) == 0:
                break

            blacklist |= common

        print(
            f"[Layer {int(layer):02d}] DONE | "
            f"iterations={iteration}, excluded={len(blacklist)}/{d}"
        )

        # -------------------------
        # Final overlap matrix
        # -------------------------
        K = len(group_order)
        M = np.zeros((K, K), dtype=np.int32)

        for i, a in enumerate(group_order):
            for j, b in enumerate(group_order):
                M[i, j] = overlap_count(top_sets[a], top_sets[b])
                all_overlap_rows.append({
                    "layer": int(layer),
                    "group_by": args.group_by,
                    "group_i": str(a),
                    "group_j": str(b),
                    "overlap": int(M[i, j]),
                    "topk": int(args.topk),
                    "excluded_features": int(len(blacklist)),
                })

        # ✅ ALWAYS save per-layer CSV into out_dir/csvs/
        out_csv = os.path.join(csv_dir, f"overlap_matrix_layer_{int(layer):02d}.csv")
        pd.DataFrame(M, index=labels, columns=labels).to_csv(out_csv)

        mats.append(M)
        layer_ids.append(int(layer))

    # ✅ single combined grid image
    out_grid = os.path.join(args.out_dir, args.grid_out_name)
    plot_overlap_grid(
        mats=mats,
        layer_ids=layer_ids,
        labels=labels,
        out_png=out_grid,
        title_prefix="Layer",
        ncols=args.grid_cols,
        dpi=300,
    )
    print(f"[OK] Saved combined heatmap grid -> {out_grid}")
    print(f"[OK] Saved per-layer CSVs -> {csv_dir}")

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
python3 overlap_by_year.py \
  --features-parquet ./python_yearly/features.parquet \
  --out-dir ./python_yearly/overlaps \
  --group-by year \
  --years 2010-2025 \
  --topk 128 \
  --layers 1-12 \
  --samples-per-group 1000 \
  --font-size 16 \
  --grid-cols 4 \
  --grid-out-name heatmap_overlap_all_layers.png
"""