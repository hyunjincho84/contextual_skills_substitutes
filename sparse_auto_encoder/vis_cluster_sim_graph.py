#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot element-centric cluster similarity curve (k-means vs random-null) from CSVs.

Inputs (inside --in-dir):
  - element_centric_similarity_by_layer.csv
  - element_centric_similarity_random_null_by_layer.csv
Each must contain columns: layer, similarity

Outputs:
  - element_centric_similarity_kmeans_vs_random_bigfont.png
  - (optional) element_centric_similarity_legend.png  [if --legend-out-png is set]

Legend PNG (when saved separately) is MARKER-ONLY (no line segments) and VERTICAL (ncol=1).
"""

import os
import argparse
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Optional

# ---------------------------
# Global style
# ---------------------------
def set_global_fonts(font_size: int):
    mpl.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size + 6,
        "axes.labelsize": font_size + 10,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,   # default; legend() will override explicitly
        "figure.titlesize": font_size + 8,
    })

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# ---------------------------
# Marker-only legend handles
# ---------------------------
def make_marker_only_handles(markersize: float = 14):
    """
    Create marker-only legend handles (no lines).
    """
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", color="blue",
               markersize=markersize, label="k-means"),
        Line2D([0], [0], marker="o", linestyle="None", color="red",
               markersize=markersize, label="random-null"),
    ]
    labels = ["k-means", "random-null"]
    return handles, labels

# ---------------------------
# Legend-only saver
# ---------------------------
def save_legend_only(
    handles,
    labels,
    out_png: str,
    fontsize: int = 24,
    frameon: bool = True,
    # vertical layout controls
    ncol: int = 1,  # ✅ 세로 방향
    columnspacing: float = 1.2,
    handletextpad: float = 0.6,
    borderpad: float = 0.8,
    labelspacing: float = 0.6,
):
    """
    Save legend as a standalone PNG (vertical).
    """
    ensure_dir(os.path.dirname(out_png) or ".")

    # Heuristic sizing for vertical legend
    fig_w = 4.4
    fig_h = max(1.8, 1.2 + 0.9 * len(labels))

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111)
    ax.axis("off")

    leg = ax.legend(
        handles,
        labels,
        loc="center",
        frameon=frameon,
        fontsize=fontsize,
        ncol=ncol,                  # ✅ 세로
        columnspacing=columnspacing,
        handlelength=0.0,           # ✅ 선 공간 제거
        handletextpad=handletextpad,
        borderpad=borderpad,
        labelspacing=labelspacing,
    )

    if frameon:
        leg.get_frame().set_linewidth(1.5)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_alpha(0.95)

    # for t in leg.get_texts():
    #     t.set_fontweight("bold")

    fig.savefig(out_png, dpi=240, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

# ---------------------------
# Plot
# ---------------------------
def plot_similarity_curve_two_from_csv(
    sim_csv: str,
    rand_csv: str,
    out_png: str,
    xlabel: str = "",
    ylabel: str = "Cluster Similarity",
    # legend controls
    legend_out_png: Optional[str] = None,
    legend_fontsize: int = 24,
    legend_frameon: bool = True,
    legend_markersize: float = 14,
):
    df_sim = pd.read_csv(sim_csv).sort_values("layer").reset_index(drop=True)
    df_rand = pd.read_csv(rand_csv).sort_values("layer").reset_index(drop=True)

    if not {"layer", "similarity"}.issubset(df_sim.columns):
        raise ValueError(f"[SIM] {sim_csv} must contain columns: layer, similarity")
    if not {"layer", "similarity"}.issubset(df_rand.columns):
        raise ValueError(f"[RAND] {rand_csv} must contain columns: layer, similarity")

    fig = plt.figure(figsize=(7.6, 4.8))
    ax = fig.add_subplot(1, 1, 1)

    # Main plot stays the same (line + markers)
    line1, = ax.plot(
        df_sim["layer"].values,
        df_sim["similarity"].values,
        marker="o",
        linewidth=2.5,
        color="blue",
        label="k-means",
    )
    line2, = ax.plot(
        df_rand["layer"].values,
        df_rand["similarity"].values,
        marker="o",
        linewidth=2.5,
        color="red",
        label="random-null",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(df_sim["layer"].values.tolist())
    ax.grid(False)

    # If legend_out_png is given, DO NOT draw legend on the main plot
    if legend_out_png is None:
        leg = ax.legend(frameon=True, fontsize=legend_fontsize)
        leg.get_frame().set_linewidth(1.5)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_alpha(0.95)
        # for t in leg.get_texts():
        #     t.set_fontweight("bold")
    else:
        # Save marker-only legend separately (vertical)
        handles, labels = make_marker_only_handles(markersize=legend_markersize)
        save_legend_only(
            handles=handles,
            labels=labels,
            out_png=legend_out_png,
            fontsize=legend_fontsize,
            frameon=legend_frameon,
            ncol=1,  # ✅ 세로
        )

    plt.tight_layout()
    plt.savefig(out_png, dpi=240, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=str, required=True,
                    help="Directory containing similarity CSVs")
    ap.add_argument("--font-size", type=int, default=16)

    # Legend controls (optional)
    ap.add_argument("--legend-out-png", type=str, default=None,
                    help="If set, save legend as a separate PNG and omit legend from the main plot.")
    ap.add_argument("--legend-fontsize", type=int, default=24)
    ap.add_argument("--legend-frameon", action="store_true",
                    help="If set, draw a frame around the standalone legend.")
    ap.add_argument("--legend-markersize", type=float, default=14,
                    help="Marker size for the standalone legend (marker-only).")

    args = ap.parse_args()

    sim_csv = os.path.join(args.in_dir, "element_centric_similarity_by_layer.csv")
    rand_csv = os.path.join(args.in_dir, "element_centric_similarity_random_null_by_layer.csv")

    for p in [sim_csv, rand_csv]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    set_global_fonts(args.font_size)

    out_png = os.path.join(
        args.in_dir,
        "element_centric_similarity_kmeans_vs_random_bigfont.png"
    )

    plot_similarity_curve_two_from_csv(
        sim_csv=sim_csv,
        rand_csv=rand_csv,
        out_png=out_png,
        legend_out_png=args.legend_out_png,
        legend_fontsize=args.legend_fontsize,
        legend_frameon=args.legend_frameon if args.legend_out_png else True,
        legend_markersize=args.legend_markersize,
    )

    print(f"[OK] Saved: {out_png}")
    if args.legend_out_png is not None:
        print(f"[OK] Saved legend: {args.legend_out_png}")

if __name__ == "__main__":
    main()


"""
# Default: legend embedded (but enlarged)
python3 element_graph_fontsize.py \
    --in-dir /home/jovyan/hyunjincho/LEM2/sparse_encoder/yearly/conti/umap_and_sim_javascript_sae_years_500 \
    --font-size 26 \
    --legend-out-png /home/jovyan/hyunjincho/LEM2/sparse_encoder/yearly/conti/umap_and_sim_javascript_sae_years_500/element_centric_similarity_legend.png \

# Separate legend PNG (recommended for paper figures)
python3 element_graph_fontsize.py \
  --in-dir /home/jovyan/hyunjincho/LEM2/sparse_encoder/vis_results/umap_and_sim_javascript_sae_soc \
  --font-size 18 \
  --legend-out-png /home/jovyan/hyunjincho/LEM2/sparse_encoder/vis_results/umap_and_sim_javascript_sae_soc/element_centric_similarity_legend.png \
  --legend-fontsize 28 \
  --legend-markersize 16 \
  --legend-frameon
"""