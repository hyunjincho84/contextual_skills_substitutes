#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Re-plot UMAP layer grid from an existing umap_2d.csv.

Input CSV must have columns:
  sample_id, group, year, layer, u0, u1

Modes:
  --group-by field : categorical coloring + legend (like your original)
      - By default: legend is NOT drawn on the grid.
      - If --legend-out-png is provided: legend will be saved as a separate PNG.
  --group-by year  : continuous coloring by year + ONE shared colorbar placed OUTSIDE (no overlap)

Output:
  --out-png : umap_layer_grid.png
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Optional

# ------------------------------- plotting config -------------------------------
mpl.rcParams.update({
    "font.size": 24,
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 22,
    "figure.titlesize": 18,
})

# Optional: only used when group_by="field"
FIELD_LABEL_MAP = {
    "Computer and Mathematical Occupations": "Computer",
    "Business and Financial Operations Occupations": "Financial",
    "Management Occupations": "Management",
    "Sales and Related Occupations": "Sales",
    "Educational Instruction and Library Occupations": "Educational",
}

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def filter_umap_percentile(df_umap: pd.DataFrame, q=0.99) -> pd.DataFrame:
    x = df_umap["u0"].values
    y = df_umap["u1"].values
    x_lo, x_hi = np.quantile(x, [1 - q, q])
    y_lo, y_hi = np.quantile(y, [1 - q, q])
    mask = (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
    return df_umap[mask]

# def save_legend_only(
#     legend_handles,
#     legend_labels,
#     out_png: str,
#     legend_fontsize: int = 22,
#     markerscale: float = 8.0,
#     ncol: int = 1,
# ):
#     """
#     Save a legend as a standalone image.
#     """
#     ensure_dir(os.path.dirname(out_png) or ".")

#     # Heuristic figure size: taller for more labels
#     fig_h = max(2.0, 0.55 * len(legend_labels))
#     fig_w = 4.8 if ncol == 1 else 8.0
#     fig_leg = plt.figure(figsize=(fig_w, fig_h))
#     ax_leg = fig_leg.add_subplot(111)
#     ax_leg.axis("off")

#     leg = ax_leg.legend(
#         legend_handles,
#         legend_labels,
#         loc="center",
#         frameon=False,
#         fontsize=legend_fontsize,
#         markerscale=markerscale,
#         ncol=ncol,
#     )

#     # Bold legend text (optional)
#     for txt in leg.get_texts():
#         txt.set_fontweight("bold")

#     fig_leg.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.02)
#     plt.close(fig_leg)

def save_legend_only(
    legend_handles,
    legend_labels,
    out_png: str,
    legend_fontsize: int = 22,
    markerscale: float = 6.0,
):
    """
    Save a legend as a standalone image:
    - SINGLE ROW
    - NO bold text
    """
    ensure_dir(os.path.dirname(out_png) or ".")

    N = len(legend_labels)
    ncol = N   # ✅ 한 줄로 강제

    # 가로로 길고 얇게
    fig_w = 1.8 * N
    fig_h = 1.2

    fig_leg = plt.figure(figsize=(fig_w, fig_h))
    ax_leg = fig_leg.add_subplot(111)
    ax_leg.axis("off")

    ax_leg.legend(
        legend_handles,
        legend_labels,
        loc="center",
        frameon=False,
        fontsize=legend_fontsize,
        markerscale=markerscale,
        ncol=ncol,
        columnspacing=1.4,
        handletextpad=0.5,
    )

    fig_leg.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig_leg)

# def save_colorbar_only(
#     cmap,
#     norm,
#     out_png: str,
#     label: str = "Year",
#     fontsize: int = 18,
# ):
#     """
#     Save a standalone vertical colorbar.
#     """
#     ensure_dir(os.path.dirname(out_png) or ".")

#     fig = plt.figure(figsize=(1.6, 5.0))
#     ax = fig.add_axes([0.35, 0.05, 0.30, 0.90])

#     sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])

#     cbar = fig.colorbar(sm, cax=ax)
#     cbar.set_label(label, fontsize=fontsize)

#     fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.02)
#     plt.close(fig)
def save_colorbar_only(
    cmap,
    norm,
    out_png: str,
    label: str = "Year",
    fontsize: int = 18,
):
    """
    Save a long, thin HORIZONTAL colorbar
    - integer year ticks only
    - 0-degree tick labels (centered)
    - label below
    - keep everything within the canvas (no overflow)
    """
    ensure_dir(os.path.dirname(out_png) or ".")

    # 더 길고 얇게 + 아래에 tick/label 공간 확보
    fig = plt.figure(figsize=(10.5, 1.35))

    # [left, bottom, width, height]
    # bar를 조금 위로 올려서 tick/label 공간 확보
    ax = fig.add_axes([0.06, 0.62, 0.88, 0.14])

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=ax, orientation="horizontal")

    # ---- integer ticks only (필요하면 step 조절) ----
    vmin = int(norm.vmin)
    vmax = int(norm.vmax)

    step = 2  # 1로 하면 너무 빽빽하면 2 유지
    ticks = list(range(vmin, vmax + 1, step))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(t) for t in ticks])

    # ---- 0-degree tick labels, centered, and kept close to the bar ----
    cbar.ax.tick_params(axis="x", labelsize=fontsize - 2, pad=2)
    for t in cbar.ax.get_xticklabels():
        t.set_rotation(0)
        t.set_ha("center")
        t.set_va("top")

    # ---- label below (centered) ----
    cbar.set_label(label, fontsize=fontsize, labelpad=10)

    # bbox_inches="tight" 쓰면 또 바깥 여백을 재단하니,
    # 여기서는 캔버스 안에서 해결하고 tight는 유지해도 괜찮게 구성.
    fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def plot_umap_by_layer(
    df_umap: pd.DataFrame,
    out_png: str,
    group_by: str,
    legend_fontsize: int = 22,
    point_size: float = 0.5,
    point_alpha: float = 0.75,
    crop_q: float = 0.98,
    field_label_map=None,
    cmap_name: str = "viridis",
    cbar_x0: float = 0.92,     # push colorbar right by increasing this
    cbar_w: float = 0.02,
    cbar_y0: float = 0.18,
    cbar_h: float = 0.64,
    legend_out_png: Optional[str] = None,  # ✅ save legend separately in field mode
    cbar_out_png: Optional[str] = None,
):
    """
    group_by:
      - "field": discrete categories (legend saved separately if legend_out_png is set)
      - "year" : continuous colormap by numeric year + shared colorbar
    """
    assert group_by in ("field", "year")

    # layers
    layers = sorted(df_umap["layer"].astype(int).unique().tolist())

    # fixed 4x3 grid (12 panels)
    ncols, nrows = 4, 3
    max_panels = ncols * nrows
    if len(layers) > max_panels:
        print(f"[WARN] layers={len(layers)} > {max_panels}. Only first {max_panels} layers will be plotted.")
        layers = layers[:max_panels]

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.8 * nrows))
    axes = np.array(axes).reshape(-1)

    # ----------------------- field mode: categorical -----------------------
    if group_by == "field":
        groups = sorted(df_umap["group"].astype(str).unique().tolist())
        legend_handles = None
        legend_labels = None

        for k in range(max_panels):
            ax = axes[k]
            if k >= len(layers):
                ax.axis("off")
                continue

            layer = layers[k]
            sub = df_umap[df_umap["layer"].astype(int) == layer].copy()
            sub = filter_umap_percentile(sub, q=crop_q)

            for g in groups:
                sg = sub[sub["group"].astype(str) == g]
                if len(sg) == 0:
                    continue

                label = field_label_map.get(g, g) if field_label_map is not None else g
                ax.scatter(
                    sg["u0"].values,
                    sg["u1"].values,
                    s=point_size,
                    alpha=point_alpha,
                    label=label,
                    linewidths=0.0,
                )

            ax.set_title(f"")
            ax.set_xticks([])
            ax.set_yticks([])

            # collect legend entries once
            if k == 0:
                h, l = ax.get_legend_handles_labels()
                uniq = {}
                for hh, ll in zip(h, l):
                    uniq[ll] = hh
                legend_labels = list(uniq.keys())
                legend_handles = [uniq[ll] for ll in legend_labels]

        # ✅ Save legend separately (if requested)
        if legend_out_png is not None and legend_handles is not None and legend_labels is not None:
            save_legend_only(
                legend_handles=legend_handles,
                legend_labels=legend_labels,
                out_png=legend_out_png,
                legend_fontsize=legend_fontsize,
                markerscale=8.0,
            )

        # No legend on the grid. Tight spacing.
        fig.subplots_adjust(
            left=0.01, right=0.99,
            top=0.99, bottom=0.01,
            wspace=0.005, hspace=0.005
        )

        plt.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)
        return

    # ----------------------- year mode: continuous colormap -----------------------
    year_vals = df_umap["year"].astype(int).to_numpy()
    vmin = int(np.min(year_vals))
    vmax = int(np.max(year_vals))

    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    mappable_for_cbar = None

    for k in range(max_panels):
        ax = axes[k]
        if k >= len(layers):
            ax.axis("off")
            continue

        layer = layers[k]
        sub = df_umap[df_umap["layer"].astype(int) == layer].copy()
        sub = filter_umap_percentile(sub, q=crop_q)

        sc = ax.scatter(
            sub["u0"].values,
            sub["u1"].values,
            c=sub["year"].astype(int).values,
            s=point_size,
            alpha=point_alpha,
            cmap=cmap,
            norm=norm,
            linewidths=0.0,
        )
        if mappable_for_cbar is None:
            mappable_for_cbar = sc

        ax.set_title(f"")
        ax.set_xticks([])
        ax.set_yticks([])

    # ✅ reserve right margin and place colorbar OUTSIDE
    fig.subplots_adjust(
        left=0.01, right=0.99,   # right: reserve space for colorbar
        top=0.99, bottom=0.01,
        wspace=0.005, hspace=0.005
    )

    # Save colorbar separately (if requested)
    if cbar_out_png is not None:
        save_colorbar_only(
            cmap=cmap,
            norm=norm,
            out_png=cbar_out_png,
            label="Year",
            fontsize=18,
        )

    plt.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umap-csv", type=str, required=True, help="Path to umap_2d.csv")
    ap.add_argument("--out-png", type=str, required=True, help="Output PNG path")
    ap.add_argument("--group-by", type=str, required=True, choices=["field", "year"])
    ap.add_argument("--legend-fontsize", type=int, default=22)
    ap.add_argument("--legend-out-png", type=str, default=None,
                    help="(field mode only) Save legend as a separate PNG (no legend drawn on grid).")
    ap.add_argument("--point-size", type=float, default=0.5)
    ap.add_argument("--point-alpha", type=float, default=0.75)
    ap.add_argument("--crop-q", type=float, default=0.98, help="Percentile crop for each layer panel")
    ap.add_argument("--cmap", type=str, default="viridis", help="Colormap name for year mode")
    ap.add_argument("--cbar-x0", type=float, default=0.92, help="Colorbar x0 (increase to push right)")
    ap.add_argument("--cbar-w", type=float, default=0.02, help="Colorbar width")
    ap.add_argument("--cbar-y0", type=float, default=0.18, help="Colorbar y0")
    ap.add_argument("--cbar-h", type=float, default=0.64, help="Colorbar height")
    ap.add_argument(
    "--cbar-out-png",
    type=str,
    default=None,
    help="(year mode only) Save colorbar as a separate PNG"
    )
    args = ap.parse_args()

    ensure_dir(os.path.dirname(args.out_png) or ".")

    df = pd.read_csv(args.umap_csv)

    # --- backward-compatible column fix ---
    if "group" not in df.columns:
        if "field" in df.columns:
            df["group"] = df["field"]
        elif "gt_group" in df.columns:
            df["group"] = df["gt_group"]
        else:
            raise RuntimeError("CSV has no 'group' column, and no fallback ('field' or 'gt_group') found.")

    needed = {"sample_id", "group", "year", "layer", "u0", "u1"}
    missing = sorted(list(needed - set(df.columns)))
    if missing:
        raise RuntimeError(f"CSV missing columns: {missing}. Expected at least {sorted(list(needed))}")

    plot_umap_by_layer(
        df_umap=df,
        out_png=args.out_png,
        group_by=args.group_by,
        legend_fontsize=args.legend_fontsize,
        point_size=args.point_size,
        point_alpha=args.point_alpha,
        crop_q=args.crop_q,
        field_label_map=FIELD_LABEL_MAP if args.group_by == "field" else None,
        cmap_name=args.cmap,
        cbar_x0=args.cbar_x0,
        cbar_w=args.cbar_w,
        cbar_y0=args.cbar_y0,
        cbar_h=args.cbar_h,
        legend_out_png=args.legend_out_png,
        cbar_out_png=args.cbar_out_png,   # ✅ 추가
    )

    print("[DONE]")
    print(f"  read : {args.umap_csv}")
    print(f"  saved: {args.out_png}")
    if args.legend_out_png is not None and args.group_by == "field":
        print(f"  legend: {args.legend_out_png}")

if __name__ == "__main__":
    main()


"""
# field mode: grid only + separate legend
python3 vis_umap_conti.py \
  --umap-csv /home/jovyan/hyunjincho/LEM2/sparse_encoder/vis_results/umap_and_sim_python_sae_soc/umap_2d.csv \
  --out-png  /home/jovyan/hyunjincho/LEM2/sparse_encoder/vis_results/umap_and_sim_python_sae_soc/umap_layer_grid_replot.png \
  --legend-out-png /home/jovyan/hyunjincho/LEM2/sparse_encoder/vis_results/umap_and_sim_python_sae_soc/umap_legend.png \
  --group-by field \
  --point-size 3 \
  --legend-fontsize 26

# year mode: continuous colormap + shared colorbar
python3 vis_umap_conti.py \
  --umap-csv /home/jovyan/hyunjincho/LEM2/sparse_encoder/yearly/umap_and_sim_python_sae_years_all_continuous/umap_2d.csv \
  --out-png  /home/jovyan/hyunjincho/LEM2/sparse_encoder/yearly/umap_and_sim_python_sae_years_all_continuous/umap_layer_grid_year_conti.png \
  --cbar-out-png /home/jovyan/hyunjincho/LEM2/sparse_encoder/yearly/umap_and_sim_python_sae_years_all_continuous/umap_year_colorbar.png \
  --group-by year \
  --point-size 3.0 \
  --cmap viridis
"""