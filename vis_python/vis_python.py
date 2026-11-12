#!/usr/bin/env python3
"""Export separate year and SOC panels for Python contextual embeddings."""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from matplotlib.lines import Line2D

DEFAULT_INPUT = Path(
    "/home/jovyan/LEM_data3/hyunjincho/LEM/vis_python_samlpes"
)
DEFAULT_YEARS = tuple(range(2010, 2026))
ORIGINAL_PANEL_SIZE = (7.0, 6.0)
SOC_TOP_K = 8
COMPUTING_SOC = "Computer and Mathematical Occupations"
UNCLASSIFIED_SOC = "Unclassified Occupation"
EXCLUDED_SOC_CATEGORIES = (
    COMPUTING_SOC,
    UNCLASSIFIED_SOC,
)
ABBREVIATIONS = {
    "Computer and Mathematical Occupations": "Computing",
    "Business and Financial Operations Occupations": "Financial",
    "Architecture and Engineering Occupations": "Engineering",
    "Management Occupations": "Management",
    "Life, Physical, and Social Science Occupations": "Science",
    "Sales and Related Occupations": "Sales",
    "Arts, Design, Entertainment, Sports, and Media Occupations": "Arts",
    "Office and Administrative Support Occupations": "Office Administration",
    "Educational Instruction and Library Occupations": "Education",
    "Other": "Other",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export separate Python UMAP panels as PNG files."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--years", nargs="+", type=int, default=list(DEFAULT_YEARS))
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--samples-per-year",
        type=int,
        default=500,
        help="Rows sampled per year before UMAP; 0 uses all stored rows.",
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=42,
        help="Deterministic per-year sampling seed.",
    )
    parser.add_argument("--kde-bw-adjust", type=float, default=0.6)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--limit-quantile",
        type=float,
        default=0.01,
        help=(
            "Tail fraction omitted only when deriving automatic panel limits; "
            "use 0 for the full coordinate range."
        ),
    )
    parser.add_argument(
        "--limit-padding",
        type=float,
        default=0.30,
        help="Fractional padding around automatically derived panel limits.",
    )
    parser.add_argument("--xlim", nargs=2, type=float, default=None)
    parser.add_argument("--ylim", nargs=2, type=float, default=None)
    parser.add_argument("--save-coordinates", action="store_true")
    args = parser.parse_args()
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else args.input_dir / "figures"
    )
    if args.n_neighbors < 2:
        parser.error("--n-neighbors must be >= 2")
    if args.min_dist < 0 or args.dpi < 1 or args.samples_per_year < 0:
        parser.error(
            "--min-dist and --samples-per-year must be >= 0; --dpi >= 1"
        )
    if not 0 <= args.limit_quantile < 0.5:
        parser.error("--limit-quantile must be in [0, 0.5)")
    if args.limit_padding < 0:
        parser.error("--limit-padding must be >= 0")
    return args


def load_inputs(input_dir: Path, years: list[int]) -> tuple[np.ndarray, pd.DataFrame]:
    arrays: list[np.ndarray] = []
    frames: list[pd.DataFrame] = []
    dimension: int | None = None
    for year in years:
        array_path = input_dir / f"{year}_emb_mlp.npy"
        csv_path = input_dir / f"{year}.csv"
        if not array_path.is_file() or not csv_path.is_file():
            raise FileNotFoundError(
                f"missing aligned input pair for {year}: {array_path}, {csv_path}"
            )
        array = np.load(array_path, allow_pickle=False)
        frame = pd.read_csv(csv_path)
        if array.ndim != 2:
            raise ValueError(f"{array_path} must be a 2-D array")
        if len(frame) != len(array):
            raise ValueError(
                f"{year}: array has {len(array)} rows but CSV has {len(frame)}"
            )
        if dimension is None:
            dimension = array.shape[1]
        elif array.shape[1] != dimension:
            raise ValueError(f"{year}: inconsistent embedding dimension")
        if "year" not in frame:
            frame["year"] = year
        if "onet_major_group" not in frame:
            if "soc_2_name" not in frame:
                raise ValueError(
                    f"{csv_path} needs onet_major_group or soc_2_name"
                )
            frame["onet_major_group"] = frame["soc_2_name"]
        frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
        arrays.append(array.astype(np.float32, copy=False))
        frames.append(frame)
    if not arrays:
        raise ValueError("no input rows")
    return np.vstack(arrays), pd.concat(frames, ignore_index=True)


def fit_umap(
    embeddings: np.ndarray, n_neighbors: int, min_dist: float, seed: int
) -> np.ndarray:
    if len(embeddings) <= n_neighbors:
        raise ValueError(
            f"need more than {n_neighbors} samples, found {len(embeddings)}"
        )
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(embeddings)


def sample_by_year(
    embeddings: np.ndarray,
    frame: pd.DataFrame,
    samples_per_year: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, int]]:
    """Select aligned embedding/metadata rows deterministically within year."""
    input_counts = {
        str(int(year)): int(count)
        for year, count in frame.groupby("year", sort=True).size().items()
    }
    if samples_per_year == 0:
        return embeddings, frame.reset_index(drop=True), input_counts

    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for _, indices in frame.groupby("year", sort=True).groups.items():
        positions = np.asarray(list(indices), dtype=np.int64)
        if len(positions) > samples_per_year:
            # Match Untitled2.ipynb: preserve rng.choice order before UMAP.
            positions = rng.choice(
                positions, size=samples_per_year, replace=False
            )
        selected.append(positions)
    selected_indices = np.concatenate(selected)
    sampled_embeddings = embeddings[selected_indices]
    sampled_frame = frame.iloc[selected_indices].reset_index(drop=True)
    return sampled_embeddings, sampled_frame, input_counts


def clean_soc(value: Any) -> str:
    if not isinstance(value, str):
        return "Other"
    value = value.strip()
    return value if value else "Other"


def short_soc_label(category: str) -> str:
    if category in ABBREVIATIONS:
        return ABBREVIATIONS[category]
    return category.removesuffix(" Occupations").strip()


def slugify_label(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in value)
    return "_".join(part for part in slug.split("_") if part)


def category_colors(categories: list[str]) -> dict[str, Any]:
    # Untitled2.ipynb indexes tab20 after the omitted Computing category.
    cmap = plt.get_cmap("tab20")
    colors = {
        category: cmap(index)
        for index, category in enumerate(categories, start=1)
    }
    colors["Other"] = cmap(len(categories) + 1)
    return colors


def set_embedding_axes(
    ax: mpl.axes.Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


def kde_or_note(
    ax: mpl.axes.Axes,
    data: pd.DataFrame,
    color: Any,
    bandwidth: float,
) -> None:
    if len(data) >= 3 and data[["x", "y"]].nunique().min() >= 2:
        sns.kdeplot(
            data=data,
            x="x",
            y="y",
            ax=ax,
            bw_adjust=bandwidth,
            color=color,
        )
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)


def save_panel(fig: mpl.figure.Figure, path: Path, dpi: int) -> None:
    """Match the notebook normal Matplotlib canvas and save behavior."""
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def save_auxiliary(fig: mpl.figure.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)


def automatic_limits(
    frame: pd.DataFrame,
    tail_fraction: float,
    padding_fraction: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Derive centered, padded limits without clipping KDE contour tails."""
    if frame.empty:
        raise ValueError("cannot derive plot limits from an empty frame")

    lower, upper = tail_fraction, 1.0 - tail_fraction
    x_low, x_high = np.quantile(frame.x, [lower, upper])
    y_low, y_high = np.quantile(frame.y, [lower, upper])

    def pad(low: float, high: float) -> tuple[float, float]:
        span = float(high - low)
        if not np.isfinite(span) or span <= 0:
            span = 1.0
        margin = span * padding_fraction
        return float(low - margin), float(high + margin)

    return pad(x_low, x_high), pad(y_low, y_high)


def save_year_scatter(
    frame: pd.DataFrame,
    output_dir: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    dpi: int,
) -> None:
    year_min, year_max = int(frame.year.min()), int(frame.year.max())
    norm = mpl.colors.Normalize(year_min, year_max)
    fig, ax = plt.subplots(figsize=ORIGINAL_PANEL_SIZE)
    scatter = ax.scatter(
        frame.x,
        frame.y,
        c=frame.year,
        cmap="viridis",
        norm=norm,
        s=6,
        alpha=0.7,
        linewidths=0,
    )
    set_embedding_axes(ax, xlim, ylim)
    save_panel(fig, output_dir / "year_scatter.png", dpi)

    colorbar_fig, colorbar_ax = plt.subplots(figsize=(1.4, 5.5))
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap="viridis")
    colorbar = colorbar_fig.colorbar(scalar, cax=colorbar_ax)
    colorbar.set_label("Year", fontsize=20)
    colorbar.ax.tick_params(labelsize=15)
    colorbar.set_ticks(np.arange(year_min, year_max + 1, 3))
    save_auxiliary(colorbar_fig, output_dir / "year_colorbar.png", dpi)

    combined_fig, combined_ax = plt.subplots(figsize=ORIGINAL_PANEL_SIZE)
    combined_scatter = combined_ax.scatter(
        frame.x,
        frame.y,
        c=frame.year,
        cmap="viridis",
        norm=norm,
        s=6,
        alpha=0.7,
        linewidths=0,
    )
    combined_colorbar = combined_fig.colorbar(
        combined_scatter,
        ax=combined_ax,
        pad=0.01,
    )
    combined_colorbar.set_label("Year", fontsize=20)
    combined_colorbar.ax.tick_params(labelsize=15)
    combined_colorbar.set_ticks(np.arange(year_min, year_max + 1, 3))
    set_embedding_axes(combined_ax, xlim, ylim)
    combined_fig.tight_layout()
    save_panel(
        combined_fig,
        output_dir / "year_scatter_with_legend.png",
        dpi,
    )


def save_year_panels(
    frame: pd.DataFrame,
    output_dir: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    bandwidth: float,
    dpi: int,
) -> list[str]:
    norm = mpl.colors.Normalize(frame.year.min(), frame.year.max())
    cmap = plt.get_cmap("viridis")
    paths: list[str] = []
    for year in (int(frame.year.min()), int(frame.year.max())):
        fig, ax = plt.subplots(figsize=ORIGINAL_PANEL_SIZE)
        panel_data = frame.loc[frame.year == year]
        panel_data = panel_data.loc[
            panel_data["x"].between(*xlim)
            & panel_data["y"].between(*ylim)
        ]
        kde_or_note(ax, panel_data, cmap(norm(year)), bandwidth)
        set_embedding_axes(ax, xlim, ylim)
        path = output_dir / f"year_kde_{year}.png"
        save_panel(fig, path, dpi)
        paths.append(path.name)
    return paths


def save_soc_scatter(
    frame: pd.DataFrame,
    top_soc: list[str],
    grouped_counts: dict[str, int],
    output_dir: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    dpi: int,
) -> None:
    colors = category_colors(top_soc)
    order = [*top_soc, "Other"]
    fig, ax = plt.subplots(figsize=ORIGINAL_PANEL_SIZE)
    ax.scatter(
        frame.x,
        frame.y,
        c=[colors[value] for value in frame.soc_group],
        s=6,
        alpha=0.7,
        linewidths=0,
    )
    set_embedding_axes(ax, xlim, ylim)
    save_panel(fig, output_dir / "soc_scatter.png", dpi)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            label=f"{short_soc_label(category)} ({grouped_counts[category]})",
            markerfacecolor=colors[category],
            markeredgecolor="none",
            markersize=10,
        )
        for category in order
    ]
    legend_fig, legend_ax = plt.subplots(figsize=(6.5, 4.5))
    legend_ax.axis("off")
    legend_ax.legend(
        handles=handles,
        title="SOC Major Group",
        loc="center left",
        frameon=False,
        fontsize=13,
        title_fontsize=17,
    )
    save_auxiliary(legend_fig, output_dir / "soc_legend.png", dpi)

    # Additional notebook-style export with scatter and legend in one canvas.
    combined_fig, combined_ax = plt.subplots(figsize=(9.5, 7.0))
    combined_ax.scatter(
        frame.x,
        frame.y,
        c=[colors[value] for value in frame.soc_group],
        s=6,
        alpha=0.7,
        linewidths=0,
    )
    set_embedding_axes(combined_ax, xlim, ylim)
    combined_legend = combined_ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.9, 1.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=15,
        title="SOC Major Group",
    )
    combined_legend.get_title().set_fontsize(20)
    combined_fig.tight_layout()
    save_panel(
        combined_fig,
        output_dir / "soc_scatter_with_legend.png",
        dpi,
    )


def save_soc_panels(
    frame: pd.DataFrame,
    top_soc: list[str],
    output_dir: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    bandwidth: float,
    dpi: int,
) -> list[str]:
    colors = category_colors(top_soc)
    paths: list[str] = []
    for rank, category in enumerate(top_soc, start=1):
        label = short_soc_label(category)
        fig, ax = plt.subplots(figsize=ORIGINAL_PANEL_SIZE)
        panel_data = frame.loc[frame.soc_group == category]
        panel_data = panel_data.loc[
            panel_data["x"].between(*xlim)
            & panel_data["y"].between(*ylim)
        ]
        kde_or_note(
            ax,
            panel_data,
            colors[category],
            bandwidth,
        )
        set_embedding_axes(ax, xlim, ylim)
        path = output_dir / f"soc_{rank:02d}_{slugify_label(label)}.png"
        save_panel(fig, path, dpi)
        paths.append(path.name)
    return paths


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    embeddings, frame = load_inputs(args.input_dir, args.years)
    stored_rows = len(frame)
    full_soc_counts = frame["onet_major_group"].map(clean_soc).value_counts()
    embeddings, frame, input_counts = sample_by_year(
        embeddings,
        frame,
        args.samples_per_year,
        args.sampling_seed,
    )
    sampled_soc_counts = (
        frame["onet_major_group"].map(clean_soc).value_counts()
    )
    eligible_soc_counts = sampled_soc_counts.loc[
        ~sampled_soc_counts.index.isin(EXCLUDED_SOC_CATEGORIES)
    ]
    top_soc = (
        eligible_soc_counts.rename_axis("soc_major_group")
        .reset_index(name="count")
        .sort_values(
            ["count", "soc_major_group"],
            ascending=[False, True],
            kind="mergesort",
        )
        .head(SOC_TOP_K)["soc_major_group"]
        .tolist()
    )
    if len(top_soc) < SOC_TOP_K:
        raise ValueError(
            f"need at least {SOC_TOP_K} eligible SOC groups; "
            f"found {len(top_soc)}"
        )
    full_grouped_counts = {
        category: int(full_soc_counts.get(category, 0))
        for category in top_soc
    }
    full_grouped_counts["Other"] = int(
        full_soc_counts.loc[
            ~full_soc_counts.index.isin(
                [*EXCLUDED_SOC_CATEGORIES, *top_soc]
            )
        ].sum()
    )
    print(
        f"[load] stored={stored_rows:,}; selected={len(frame):,}; "
        f"embedding shape={embeddings.shape}"
    )
    coordinates = fit_umap(
        embeddings, args.n_neighbors, args.min_dist, args.random_state
    )
    frame = frame.copy()
    frame["x"] = coordinates[:, 0]
    frame["y"] = coordinates[:, 1]
    frame["soc_raw"] = frame["onet_major_group"].map(clean_soc)
    frame["soc_group"] = np.where(
        frame["soc_raw"].isin(top_soc),
        frame["soc_raw"],
        np.where(
            frame["soc_raw"].isin(EXCLUDED_SOC_CATEGORIES),
            "Excluded",
            "Other",
        ),
    )
    soc_frame = frame.loc[
        ~frame["soc_raw"].isin(EXCLUDED_SOC_CATEGORIES)
    ].copy()
    grouped_counts = {
        category: int((soc_frame["soc_group"] == category).sum())
        for category in (*top_soc, "Other")
    }
    year_default_xlim, year_default_ylim = automatic_limits(
        frame, args.limit_quantile, args.limit_padding
    )
    soc_scatter_default_xlim, soc_scatter_default_ylim = automatic_limits(
        soc_frame, args.limit_quantile, args.limit_padding
    )
    # The eight requested KDE panels share limits derived from those eight
    # categories. Computing and Unclassified are excluded from SOC plots
    # and from Other.
    soc_panel_reference = soc_frame.loc[soc_frame["soc_raw"].isin(top_soc)]
    soc_panel_default_xlim, soc_panel_default_ylim = automatic_limits(
        soc_panel_reference, args.limit_quantile, args.limit_padding
    )
    year_xlim = (
        tuple(args.xlim) if args.xlim is not None else year_default_xlim
    )
    year_ylim = (
        tuple(args.ylim) if args.ylim is not None else year_default_ylim
    )
    soc_scatter_xlim = (
        tuple(args.xlim)
        if args.xlim is not None
        else soc_scatter_default_xlim
    )
    soc_scatter_ylim = (
        tuple(args.ylim)
        if args.ylim is not None
        else soc_scatter_default_ylim
    )
    soc_panel_xlim = (
        tuple(args.xlim)
        if args.xlim is not None
        else soc_panel_default_xlim
    )
    soc_panel_ylim = (
        tuple(args.ylim)
        if args.ylim is not None
        else soc_panel_default_ylim
    )
    save_year_scatter(
        frame, args.output_dir, year_xlim, year_ylim, args.dpi
    )
    year_panel_files = save_year_panels(
        frame,
        args.output_dir,
        year_xlim,
        year_ylim,
        args.kde_bw_adjust,
        args.dpi,
    )
    save_soc_scatter(
        soc_frame,
        top_soc,
        grouped_counts,
        args.output_dir,
        soc_scatter_xlim,
        soc_scatter_ylim,
        args.dpi,
    )
    soc_panel_files = save_soc_panels(
        soc_frame,
        top_soc,
        args.output_dir,
        soc_panel_xlim,
        soc_panel_ylim,
        args.kde_bw_adjust,
        args.dpi,
    )
    pd.DataFrame(
        {
            "rank": [*range(1, len(top_soc) + 1), len(top_soc) + 1],
            "soc_major_group": [*top_soc, "Other"],
            "full_python_count": [
                *[full_grouped_counts[value] for value in top_soc],
                full_grouped_counts["Other"],
            ],
            "plotted_sample_count": [
                *[grouped_counts[value] for value in top_soc],
                grouped_counts["Other"],
            ],
        }
    ).to_csv(args.output_dir / "soc_top8_counts.csv", index=False)
    if args.save_coordinates:
        frame.to_csv(args.output_dir / "umap_coordinates.csv.gz", index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "years": args.years,
        "rows": len(frame),
        "stored_rows": stored_rows,
        "stored_rows_by_year": input_counts,
        "samples_per_year": args.samples_per_year,
        "sampling_seed": args.sampling_seed,
        "embedding_shape": list(embeddings.shape),
        "n_neighbors": args.n_neighbors,
        "min_dist": args.min_dist,
        "metric": "cosine",
        "random_state": args.random_state,
        "kde_bw_adjust": args.kde_bw_adjust,
        "limit_quantile": args.limit_quantile,
        "limit_padding": args.limit_padding,
        "kde_outliers_outside_auto_limits_omitted": True,
        "soc_top8": [
            {
                "rank": rank,
                "name": name,
                "full_python_count": full_grouped_counts[name],
                "plotted_sample_count": grouped_counts[name],
            }
            for rank, name in enumerate(top_soc, start=1)
        ],
        "other_full_python_count": full_grouped_counts["Other"],
        "other_plotted_sample_count": grouped_counts["Other"],
        "soc_plotted_rows": len(soc_frame),
        "year_panel_files": year_panel_files,
        "soc_panel_files": soc_panel_files,
        "combined_year_soc_figure_created": False,
        "data_panels_have_titles": False,
        "soc_top_k": SOC_TOP_K,
        "soc_top8_selected_after_year_sampling": True,
        "soc_excluded_categories": list(EXCLUDED_SOC_CATEGORIES),
        "soc_computing_and_unclassified_excluded": True,
        "soc_scatter_with_legend_file": "soc_scatter_with_legend.png",
        "year_scatter_with_legend_file": "year_scatter_with_legend.png",
        "computing_full_python_count": int(
            full_soc_counts.get(COMPUTING_SOC, 0)
        ),
        "computing_plotted_sample_count": int(
            (frame["soc_raw"] == COMPUTING_SOC).sum()
        ),
        "unclassified_full_python_count": int(
            full_soc_counts.get(UNCLASSIFIED_SOC, 0)
        ),
        "unclassified_plotted_sample_count": int(
            (frame["soc_raw"] == UNCLASSIFIED_SOC).sum()
        ),
        "soc_top8_order": top_soc,
        "year_colorbar_file": "year_colorbar.png",
        "year_xlim": list(year_xlim),
        "year_ylim": list(year_ylim),
        "soc_scatter_xlim": list(soc_scatter_xlim),
        "soc_scatter_ylim": list(soc_scatter_ylim),
        "soc_panel_xlim": list(soc_panel_xlim),
        "soc_panel_ylim": list(soc_panel_ylim),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": mpl.__version__,
            "seaborn": sns.__version__,
            "umap": umap.__version__,
        },
    }
    destination = args.output_dir / "figure_manifest.json"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=args.output_dir, delete=False
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)
    print(f"[done] figures: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
