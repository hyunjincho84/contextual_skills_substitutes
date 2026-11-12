#!/usr/bin/env python3
"""Compute industry-level SAE mean-L0 for ground-truth skill ``python``.

A feature is active for an industry when its mean activation over Python
samples from that industry is at least the layer-specific quantile threshold.
The reported mean_L0 is the (optionally sample-weighted) mean number of active
features across industries.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_FEATURES = Path(
    "/home/jovyan/LEM_data2/data/sparse_auto_encoder/"
    "python_industry/features.parquet"
)


def parse_layers(value: str) -> list[int]:
    if "-" in value:
        start, end = value.split("-", maxsplit=1)
        layers = list(range(int(start), int(end) + 1))
    else:
        layers = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not layers:
        raise argparse.ArgumentTypeError("--layers must not be empty")
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-parquet", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--industry-col", default="field")
    parser.add_argument("--layer-col", default="layer")
    parser.add_argument(
        "--truth-col",
        default="auto",
        help=(
            "Ground-truth skill column. 'auto' checks true_skill, truth, and "
            "target_skill. If none exists, the input is assumed Python-only."
        ),
    )
    parser.add_argument("--target-skill", default="python")
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("1-12"))
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.75,
        help=(
            "Layer-wise quantile of industry-feature mean activations used as "
            "the active threshold (default: 0.75, matching the existing heatmap run)."
        ),
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Weight industries by their Python sample counts when computing mean_L0.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: mean_l0_python_qXX.csv beside the input).",
    )
    parser.add_argument(
        "--industry-output",
        type=Path,
        default=None,
        help="Optional CSV containing each industry's L0.",
    )
    return parser.parse_args()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [c for c in frame.columns if re.fullmatch(r"f\d+", str(c))]
    return sorted(columns, key=lambda c: int(c[1:]))


def resolve_truth_column(frame: pd.DataFrame, requested: str) -> str | None:
    if requested != "auto":
        if requested not in frame.columns:
            raise KeyError(f"Truth column not found: {requested}")
        return requested
    for candidate in ("true_skill", "truth", "target_skill"):
        if candidate in frame.columns:
            return candidate
    return None


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.quantile <= 1.0:
        raise ValueError("--quantile must be within [0, 1]")
    if not args.features_parquet.is_file():
        raise FileNotFoundError(args.features_parquet)

    frame = pd.read_parquet(args.features_parquet)
    required = {args.industry_col, args.layer_col}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    truth_col = resolve_truth_column(frame, args.truth_col)
    if truth_col is None:
        print("[INFO] No truth column found; assuming the parquet is Python-only.")
    else:
        normalized = frame[truth_col].astype("string").str.strip().str.casefold()
        target = args.target_skill.strip().casefold()
        frame = frame.loc[normalized.eq(target)].copy()
        print(f"[INFO] Filtered {truth_col} == {args.target_skill!r}: {len(frame):,} rows")

    frame = frame.loc[frame[args.layer_col].isin(args.layers)].copy()
    if frame.empty:
        raise RuntimeError("No Python rows remain for the requested layers.")

    feat_cols = feature_columns(frame)
    if not feat_cols:
        raise RuntimeError("No SAE feature columns matching f0, f1, ... were found.")

    group_cols = [args.layer_col, args.industry_col]
    means = frame.groupby(group_cols, observed=True, sort=True)[feat_cols].mean()
    counts = frame.groupby(group_cols, observed=True, sort=True).size().rename("n_samples")

    summary_rows: list[dict[str, float | int | str]] = []
    industry_rows: list[dict[str, float | int | str]] = []
    for layer in args.layers:
        try:
            layer_means = means.xs(layer, level=args.layer_col)
            layer_counts = counts.xs(layer, level=args.layer_col)
        except KeyError:
            print(f"[WARN] Layer {layer} has no rows; skipping.")
            continue

        values = layer_means.to_numpy(dtype=np.float64)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            print(f"[WARN] Layer {layer} has no finite activations; skipping.")
            continue

        threshold = float(np.quantile(finite_values, args.quantile))
        # Match vis_union_mean_activation_heatmap_threshold.py: activation >= threshold.
        active = np.greater_equal(values, threshold) & np.isfinite(values)
        industry_l0 = active.sum(axis=1).astype(np.int64)
        sample_counts = layer_counts.reindex(layer_means.index).to_numpy(np.int64)

        if args.weighted:
            mean_l0 = float(np.average(industry_l0, weights=sample_counts))
            aggregation = "sample_weighted_industry_mean"
        else:
            mean_l0 = float(industry_l0.mean())
            aggregation = "unweighted_industry_mean"

        summary_rows.append(
            {
                "truth_skill": args.target_skill,
                "layer": int(layer),
                "quantile": args.quantile,
                "active_threshold": threshold,
                "n_industries": len(industry_l0),
                "n_samples": int(sample_counts.sum()),
                "n_features": len(feat_cols),
                "mean_L0": mean_l0,
                "aggregation": aggregation,
            }
        )
        for industry, n_samples, l0 in zip(
            layer_means.index, sample_counts, industry_l0, strict=True
        ):
            industry_rows.append(
                {
                    "truth_skill": args.target_skill,
                    "layer": int(layer),
                    "industry": industry,
                    "n_samples": int(n_samples),
                    "quantile": args.quantile,
                    "active_threshold": threshold,
                    "L0": int(l0),
                    "activation_density": float(l0 / len(feat_cols)),
                }
            )

    if not summary_rows:
        raise RuntimeError("No requested layer could be evaluated.")

    summary = pd.DataFrame(summary_rows)
    industry_detail = pd.DataFrame(industry_rows)
    output = args.output or args.features_parquet.with_name(
        f"mean_l0_python_q{args.quantile:g}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(summary.to_string(index=False))
    print(f"[OK] Layer summary saved: {output}")

    if args.industry_output is not None:
        args.industry_output.parent.mkdir(parents=True, exist_ok=True)
        industry_detail.to_csv(args.industry_output, index=False)
        print(f"[OK] Industry detail saved: {args.industry_output}")


if __name__ == "__main__":
    main()
