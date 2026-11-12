#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-pass yearly SAE feature extraction and overlap heatmap pipeline."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import shutil
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_DATA_DIR = os.environ.get(
    "BASE_DATA_DIR", "/home/jovyan/LEM_data3/hyunjincho/LEM"
)
DEFAULT_PREDICTIONS_ROOT = os.path.join(
    BASE_DATA_DIR, "predictions_clean", "bert_pred", "pred"
)
DEFAULT_MODEL_NAME = os.path.join(BASE_DATA_DIR, "bert_pretrained")
DEFAULT_VOCAB_PATH = os.path.join(BASE_DATA_DIR, "all_filterd", "skill2idx.json")
DEFAULT_MODEL_PT = os.path.join(
    BASE_DATA_DIR, "models", "bert_fine_tuned_model", "best_model.pt"
)
DEFAULT_SAE_ROOT = os.path.join(BASE_DATA_DIR, "models", "sae_layerwise_out")
FORMAT_VERSION = 1


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_layers(value: str) -> List[int]:
    if "-" in value:
        start, end = value.split("-", 1)
        layers = list(range(int(start), int(end) + 1))
    else:
        layers = [int(item) for item in value.split(",") if item.strip()]
    if not layers or any(layer <= 0 for layer in layers):
        raise ValueError("--layers must contain positive layer numbers")
    return layers


def parse_years(value: str) -> Optional[List[int]]:
    if value.strip().lower() == "all":
        return None
    if "-" in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return sorted({int(item) for item in value.split(",") if item.strip()})


def normalize_skill(value: str) -> str:
    return str(value).strip().casefold()


def skill_tag(value: str) -> str:
    value = normalize_skill(value).replace(" ", "_").replace("/", "_")
    return re.sub(r"[^a-z0-9_.-]+", "_", value)


def atomic_json(payload: Dict[str, object], path: str) -> None:
    temp = f"{path}.tmp.{os.getpid()}"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temp, path)


def discover_files(
    root: str, requested_years: Optional[Sequence[int]]
) -> Tuple[List[str], List[int]]:
    requested = set(requested_years) if requested_years is not None else None
    files, years = [], set()
    for path in sorted(glob.glob(os.path.join(root, "*", "predictions_*.csv.gz"))):
        parent = os.path.basename(os.path.dirname(path))
        if not parent.isdigit():
            continue
        year = int(parent)
        if requested is not None and year not in requested:
            continue
        files.append(path)
        years.add(year)
    if not files:
        raise FileNotFoundError(f"No predictions found under: {root}")
    if requested is not None and requested != years:
        raise FileNotFoundError(
            f"Missing requested years: {sorted(requested - years)}"
        )
    return files, sorted(years)


def run_signature(
    args: argparse.Namespace, layers: Sequence[int], years: Sequence[int]
) -> Dict[str, object]:
    return {
        "format_version": FORMAT_VERSION,
        "target_skill": normalize_skill(args.target_skill),
        "predictions_root": os.path.abspath(args.predictions_root),
        "years": list(years),
        "layers": list(layers),
        "data_mode": args.data_mode,
        "samples_per_year": (
            args.samples_per_year if args.data_mode == "sample" else None
        ),
        "seed": args.seed,
        "model_name": os.path.abspath(args.model_name),
        "vocab_path": os.path.abspath(args.vocab_path),
        "best_model_pt": os.path.abspath(args.best_model_pt),
        "sae_root": os.path.abspath(args.sae_root),
        "d_hidden": args.d_hidden,
        "max_len": args.max_len,
        "truth_col": args.truth_col,
        "text_col": args.text_col,
        "row_id_col": args.row_id_col,
    }


def load_manifest(features_dir: str) -> Optional[Dict[str, object]]:
    path = os.path.join(features_dir, "manifest.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    return manifest if manifest.get("complete") else None


def matching_chunks(
    files: Sequence[str], args: argparse.Namespace
) -> Iterable[pd.DataFrame]:
    target = normalize_skill(args.target_skill)
    usecols = [args.truth_col, args.text_col, args.row_id_col]
    for path in tqdm(files, desc="Filtering predictions_clean", unit="file"):
        year = int(os.path.basename(os.path.dirname(path)))
        for chunk in pd.read_csv(
            path,
            usecols=usecols,
            chunksize=args.read_chunksize,
            low_memory=False,
        ):
            truth = chunk[args.truth_col].astype("string").str.strip().str.casefold()
            selected = chunk.loc[
                truth.eq(target) & chunk[args.text_col].notna(),
                [args.row_id_col, args.text_col],
            ].copy()
            if selected.empty:
                continue
            selected.columns = ["source_row_idx", "sentence"]
            selected["source_row_idx"] = (
                selected["source_row_idx"].astype("string").fillna("")
            )
            selected["sentence"] = selected["sentence"].astype(str)
            selected["source_file"] = os.path.basename(path)
            selected["year"] = year
            yield selected[
                ["source_row_idx", "source_file", "year", "sentence"]
            ].reset_index(drop=True)


def select_rows(
    files: Sequence[str], years: Sequence[int], args: argparse.Namespace
) -> Tuple[Iterable[pd.DataFrame], Dict[int, int], Dict[int, int]]:
    chunks = matching_chunks(files, args)
    if args.data_mode == "all":
        eligible = {year: 0 for year in years}

        def iterator() -> Iterable[pd.DataFrame]:
            for chunk in chunks:
                for year, count in chunk["year"].value_counts().items():
                    eligible[int(year)] += int(count)
                yield chunk

        return iterator(), eligible, eligible

    limit = args.samples_per_year
    pools: Dict[int, List[Tuple[str, str, int, str]]] = {
        year: [] for year in years
    }
    seen = {year: 0 for year in years}
    rng = {year: random.Random(args.seed + year) for year in years}
    for chunk in chunks:
        for row in chunk.itertuples(index=False):
            year = int(row.year)
            seen[year] += 1
            record = (
                str(row.source_row_idx),
                str(row.source_file),
                year,
                str(row.sentence),
            )
            if len(pools[year]) < limit:
                pools[year].append(record)
            else:
                index = rng[year].randrange(seen[year])
                if index < limit:
                    pools[year][index] = record

    selected = {year: len(pools[year]) for year in years}
    missing = [year for year, count in selected.items() if count == 0]
    if missing:
        raise RuntimeError(f"No matching rows for years: {missing}")

    def iterator() -> Iterable[pd.DataFrame]:
        rows = [row for year in years for row in pools[year]]
        random.Random(args.seed).shuffle(rows)
        for start in range(0, len(rows), args.read_chunksize):
            yield pd.DataFrame(
                rows[start : start + args.read_chunksize],
                columns=[
                    "source_row_idx",
                    "source_file",
                    "year",
                    "sentence",
                ],
            )

    return iterator(), seen, selected


def extract_features(
    args: argparse.Namespace,
    files: Sequence[str],
    years: Sequence[int],
    layers: Sequence[int],
) -> Dict[str, object]:
    signature = run_signature(args, layers, years)
    old = load_manifest(args.features_dir)
    if old is not None and old.get("signature") == signature:
        print(f"[RESUME] Reusing completed features: {args.features_dir}")
        return old
    if os.path.exists(args.features_dir):
        if not args.overwrite_features:
            raise FileExistsError(
                f"Existing feature directory does not match: {args.features_dir}. "
                "Use --overwrite-features or a different --out-dir."
            )
        shutil.rmtree(args.features_dir)

    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from transformers import BertTokenizer

    model_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "model_trains")
    )
    sys.path.insert(0, model_dir)
    from model import BERTForSkillPrediction

    class SparseAutoencoder(nn.Module):
        def __init__(self, d_in: int, d_hidden: int):
            super().__init__()
            self.W_e = nn.Linear(d_in, d_hidden, bias=True)
            self.W_d = nn.Linear(d_hidden, d_in, bias=False)
            self.b_d = nn.Parameter(torch.zeros(d_in))

        def encode(self, values):
            return functional.relu(self.W_e(values - self.b_d))

    selected_batches, eligible, selected = select_rows(files, years, args)
    with open(args.vocab_path, encoding="utf-8") as handle:
        skill2idx = json.load(handle)
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BERTForSkillPrediction(
        args.model_name, num_skills=len(skill2idx)
    ).to(args.device)
    model.load_state_dict(torch.load(args.best_model_pt, map_location="cpu"))
    model.eval()
    bert = model.bert
    for parameter in bert.parameters():
        parameter.requires_grad_(False)

    saes = {}
    for layer in layers:
        checkpoint = os.path.join(
            args.sae_root, f"layer_{layer:02d}", "sae_best.pt"
        )
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Missing SAE checkpoint: {checkpoint}")
        sae = SparseAutoencoder(
            bert.config.hidden_size, args.d_hidden
        ).to(args.device)
        sae.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        sae.eval()
        saes[layer] = sae

    temp_dir = f"{args.features_dir}.tmp.{os.getpid()}"
    ensure_dir(temp_dir)
    feature_schema = pa.schema(
        [
            ("sample_id", pa.int64()),
            ("year", pa.int16()),
            ("layer", pa.int16()),
            ("features", pa.list_(pa.float32(), args.d_hidden)),
        ]
    )
    index_schema = pa.schema(
        [
            ("sample_id", pa.int64()),
            ("source_row_idx", pa.string()),
            ("source_file", pa.string()),
            ("year", pa.int16()),
        ]
    )
    compression = None if args.parquet_compression == "none" else args.parquet_compression
    index_writer = pq.ParquetWriter(
        os.path.join(temp_dir, "data_index.parquet"),
        index_schema,
        compression=compression,
    )
    writers = {}
    encoded_counts = {year: 0 for year in years}
    next_id, skipped = 0, 0
    device_type = torch.device(args.device).type
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    amp_enabled = args.use_amp and device_type == "cuda"

    try:
        with torch.no_grad():
            for selected_frame in selected_batches:
                for start in range(0, len(selected_frame), args.batch_size):
                    meta = selected_frame.iloc[
                        start : start + args.batch_size
                    ].reset_index(drop=True)
                    ids = np.arange(next_id, next_id + len(meta), dtype=np.int64)
                    next_id += len(meta)
                    tokens = tokenizer(
                        meta["sentence"].tolist(),
                        padding=True,
                        truncation=True,
                        max_length=args.max_len,
                        return_tensors="pt",
                    )
                    input_ids = tokens["input_ids"]
                    attention_mask = tokens["attention_mask"]
                    mask = input_ids.eq(tokenizer.mask_token_id)
                    keep = mask.any(dim=1).nonzero(as_tuple=True)[0]
                    skipped += len(meta) - len(keep)
                    if len(keep) == 0:
                        continue
                    positions = mask[keep].float().argmax(dim=1)
                    keep_np = keep.numpy()
                    meta = meta.iloc[keep_np].reset_index(drop=True)
                    ids = ids[keep_np]
                    input_ids = input_ids[keep].to(args.device)
                    attention_mask = attention_mask[keep].to(args.device)
                    positions = positions.to(args.device)
                    years_np = meta["year"].to_numpy(np.int16)

                    index_writer.write_table(
                        pa.Table.from_arrays(
                            [
                                pa.array(ids, type=pa.int64()),
                                pa.array(meta["source_row_idx"], type=pa.string()),
                                pa.array(meta["source_file"], type=pa.string()),
                                pa.array(years_np, type=pa.int16()),
                            ],
                            schema=index_schema,
                        )
                    )

                    with torch.autocast(
                        device_type=device_type,
                        dtype=amp_dtype,
                        enabled=amp_enabled,
                    ):
                        output = bert(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                        hidden = output.hidden_states
                        batch_index = torch.arange(len(ids), device=args.device)
                        for layer in layers:
                            if layer >= len(hidden):
                                raise ValueError(f"Model has no layer {layer}")
                            masked = hidden[layer][batch_index, positions]
                            features = (
                                saes[layer].encode(masked)
                                .float()
                                .cpu()
                                .numpy()
                                .astype(np.float32, copy=False)
                            )
                            for year in np.unique(years_np):
                                selected_positions = np.flatnonzero(years_np == year)
                                values = np.ascontiguousarray(
                                    features[selected_positions]
                                )
                                vector_array = pa.FixedSizeListArray.from_arrays(
                                    pa.array(
                                        values.reshape(-1), type=pa.float32()
                                    ),
                                    args.d_hidden,
                                )
                                table = pa.Table.from_arrays(
                                    [
                                        pa.array(
                                            ids[selected_positions], type=pa.int64()
                                        ),
                                        pa.array(
                                            np.full(
                                                len(selected_positions),
                                                year,
                                                dtype=np.int16,
                                            )
                                        ),
                                        pa.array(
                                            np.full(
                                                len(selected_positions),
                                                layer,
                                                dtype=np.int16,
                                            )
                                        ),
                                        vector_array,
                                    ],
                                    schema=feature_schema,
                                )
                                key = (layer, int(year))
                                if key not in writers:
                                    layer_dir = os.path.join(
                                        temp_dir, f"layer_{layer:02d}"
                                    )
                                    ensure_dir(layer_dir)
                                    writers[key] = pq.ParquetWriter(
                                        os.path.join(
                                            layer_dir,
                                            f"year_{int(year)}.parquet",
                                        ),
                                        feature_schema,
                                        compression=compression,
                                    )
                                writers[key].write_table(table)
                    for year, count in meta["year"].value_counts().items():
                        encoded_counts[int(year)] += int(count)

        index_writer.close()
        for writer in writers.values():
            writer.close()
        missing = [year for year, count in encoded_counts.items() if count == 0]
        if missing:
            raise RuntimeError(f"No valid [MASK] rows encoded for years: {missing}")
        manifest = {
            "complete": True,
            "signature": signature,
            "feature_dim": args.d_hidden,
            "eligible_counts_by_year": {str(k): int(v) for k, v in eligible.items()},
            "selected_counts_by_year": {str(k): int(v) for k, v in selected.items()},
            "encoded_counts_by_year": {
                str(k): int(v) for k, v in encoded_counts.items()
            },
            "skipped_no_mask": skipped,
            "layout": "layer_XX/year_YYYY.parquet",
        }
        atomic_json(manifest, os.path.join(temp_dir, "manifest.json"))
        os.replace(temp_dir, args.features_dir)
        print(f"[OK] Saved SAE features: {args.features_dir}")
        return manifest
    except Exception:
        try:
            index_writer.close()
        except Exception:
            pass
        for writer in writers.values():
            try:
                writer.close()
            except Exception:
                pass
        raise
    finally:
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()


def topk_ids(mean: np.ndarray, topk: int, excluded: set) -> np.ndarray:
    values = np.asarray(mean, dtype=np.float64).copy()
    if excluded:
        values[list(excluded)] = -np.inf
    count = min(topk, len(values) - len(excluded))
    if count <= 0:
        return np.array([], dtype=np.int64)
    indices = np.argpartition(-values, count - 1)[:count]
    return indices[np.argsort(-values[indices])].astype(np.int64)


def remove_universal_features(
    means: Dict[int, np.ndarray],
    years: Sequence[int],
    topk: int,
    max_iters: int,
    layer: int,
) -> Tuple[Dict[int, np.ndarray], set, int]:
    excluded: set = set()
    for iteration in range(1, max_iters + 1):
        top_sets = {
            year: topk_ids(means[year], topk, excluded) for year in years
        }
        common = set(top_sets[years[0]])
        for year in years[1:]:
            common.intersection_update(top_sets[year])
        print(
            f"[Layer {layer:02d}] iter={iteration}, "
            f"new_common={len(common)}, excluded={len(excluded)}"
        )
        if not common:
            return top_sets, excluded, iteration
        excluded.update(common)
    raise RuntimeError(
        f"Layer {layer} did not converge within --max-iters={max_iters}"
    )


def partition_mean(
    path: str, feature_dim: int, batch_size: int
) -> Tuple[np.ndarray, int]:
    import pyarrow.parquet as pq

    total = np.zeros(feature_dim, dtype=np.float64)
    count = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        columns=["features"], batch_size=batch_size
    ):
        array = batch.column(0)
        values = np.asarray(
            array.values.to_numpy(zero_copy_only=False), dtype=np.float32
        ).reshape(len(array), feature_dim)
        total += values.sum(axis=0, dtype=np.float64)
        count += len(values)
    if count == 0:
        raise ValueError(f"Empty feature partition: {path}")
    return total / count, count


def overlap_stage(
    args: argparse.Namespace,
    layers: Sequence[int],
    years: Sequence[int],
) -> None:
    manifest = load_manifest(args.features_dir)
    if manifest is None:
        raise FileNotFoundError(
            f"No complete feature manifest: {args.features_dir}"
        )
    if manifest.get("signature") != run_signature(args, layers, years):
        raise ValueError(
            "Feature manifest does not match the current skill/options."
        )
    feature_dim = int(manifest["feature_dim"])
    csv_dir = os.path.join(args.out_dir, args.csv_subdir)
    ensure_dir(csv_dir)
    overlap_rows, count_rows, top_rows = [], [], []

    for layer in tqdm(layers, desc="Computing overlap", unit="layer"):
        means, counts = {}, {}
        for year in years:
            path = os.path.join(
                args.features_dir,
                f"layer_{layer:02d}",
                f"year_{year}.parquet",
            )
            means[year], counts[year] = partition_mean(
                path, feature_dim, args.aggregate_batch_size
            )
        top_sets, excluded, iterations = remove_universal_features(
            means, years, args.topk, args.max_iters, layer
        )
        matrix = np.zeros((len(years), len(years)), dtype=np.int32)
        for i, left in enumerate(years):
            count_rows.append(
                {"layer": layer, "year": left, "n_samples": counts[left]}
            )
            for rank, feature_id in enumerate(top_sets[left], 1):
                top_rows.append(
                    {
                        "layer": layer,
                        "year": left,
                        "rank": rank,
                        "feature_id": int(feature_id),
                    }
                )
            for j, right in enumerate(years):
                overlap = len(np.intersect1d(top_sets[left], top_sets[right]))
                matrix[i, j] = overlap
                overlap_rows.append(
                    {
                        "layer": layer,
                        "year_i": left,
                        "year_j": right,
                        "overlap": overlap,
                        "topk": args.topk,
                        "excluded_features": len(excluded),
                        "iterations": iterations,
                    }
                )
        pd.DataFrame(matrix, index=years, columns=years).to_csv(
            os.path.join(csv_dir, f"overlap_matrix_layer_{layer:02d}.csv")
        )

    pd.DataFrame(overlap_rows).to_csv(
        os.path.join(args.out_dir, "overlap_longform_all_layers.csv"),
        index=False,
    )
    pd.DataFrame(count_rows).to_csv(
        os.path.join(args.out_dir, "sample_counts_per_layer.csv"),
        index=False,
    )
    pd.DataFrame(top_rows).to_csv(
        os.path.join(args.out_dir, "top_feature_ids_by_layer_year.csv"),
        index=False,
    )
    print(f"[OK] Saved overlap CSVs: {csv_dir}")


def legacy_overlap_stage(
    args: argparse.Namespace, layers: Sequence[int]
) -> None:
    """Keep the former precomputed features.parquet workflow working."""
    frame = pd.read_parquet(args.features_parquet)
    feature_columns = sorted(
        [column for column in frame.columns if re.fullmatch(r"f\d+", column)],
        key=lambda column: int(column[1:]),
    )
    if not feature_columns or "year" not in frame.columns:
        raise ValueError(
            "--features-parquet requires year and f0, f1, ... columns"
        )
    frame["year"] = frame["year"].astype(int)
    requested = parse_years(args.years)
    if requested is not None:
        frame = frame[frame["year"].isin(requested)]
    years = sorted(frame["year"].unique().tolist())
    if args.max_groups is not None:
        years = years[: args.max_groups]
        frame = frame[frame["year"].isin(years)]
    if len(years) < 2:
        raise ValueError("At least two years are required for overlap")

    csv_dir = os.path.join(args.out_dir, args.csv_subdir)
    ensure_dir(csv_dir)
    overlap_rows, count_rows, top_rows = [], [], []
    for layer in tqdm(layers, desc="Computing legacy overlap", unit="layer"):
        subset = frame[frame["layer"] == layer].copy()
        if subset.empty:
            continue
        if args.samples_per_group is not None:
            subset = (
                subset.groupby("year", group_keys=False, observed=True)
                .apply(
                    lambda group: group.sample(
                        n=min(len(group), args.samples_per_group),
                        random_state=args.seed,
                    )
                )
                .reset_index(drop=True)
            )
        means, counts = {}, {}
        for year in years:
            values = subset.loc[
                subset["year"] == year, feature_columns
            ].to_numpy(np.float32)
            if len(values) == 0:
                raise ValueError(f"Layer {layer} has no rows for year {year}")
            means[year] = values.mean(axis=0)
            counts[year] = len(values)
        top_sets, excluded, iterations = remove_universal_features(
            means, years, args.topk, args.max_iters, layer
        )
        matrix = np.zeros((len(years), len(years)), dtype=np.int32)
        for i, left in enumerate(years):
            count_rows.append(
                {"layer": layer, "year": left, "n_samples": counts[left]}
            )
            for rank, feature_id in enumerate(top_sets[left], 1):
                top_rows.append(
                    {
                        "layer": layer,
                        "year": left,
                        "rank": rank,
                        "feature_id": int(feature_id),
                    }
                )
            for j, right in enumerate(years):
                overlap = len(np.intersect1d(top_sets[left], top_sets[right]))
                matrix[i, j] = overlap
                overlap_rows.append(
                    {
                        "layer": layer,
                        "year_i": left,
                        "year_j": right,
                        "overlap": overlap,
                        "topk": args.topk,
                        "excluded_features": len(excluded),
                        "iterations": iterations,
                    }
                )
        pd.DataFrame(matrix, index=years, columns=years).to_csv(
            os.path.join(csv_dir, f"overlap_matrix_layer_{layer:02d}.csv")
        )
    pd.DataFrame(overlap_rows).to_csv(
        os.path.join(args.out_dir, "overlap_longform_all_layers.csv"),
        index=False,
    )
    pd.DataFrame(count_rows).to_csv(
        os.path.join(args.out_dir, "sample_counts_per_layer.csv"), index=False
    )
    pd.DataFrame(top_rows).to_csv(
        os.path.join(args.out_dir, "top_feature_ids_by_layer_year.csv"),
        index=False,
    )
    print(f"[OK] Saved legacy overlap CSVs: {csv_dir}")


def plot_grid(
    matrices: Sequence[np.ndarray],
    layers: Sequence[int],
    labels: Sequence[str],
    output_path: str,
    columns: int,
    dpi: int,
    panel_width: float,
    panel_height: float,
    annotation_font_size: float,
    tick_font_size: float,
    title_font_size: float,
    x_label_rotation: float,
) -> None:
    # Avoid an empty panel when fewer layers than --grid-cols are plotted.
    columns = min(columns, len(matrices))
    rows = math.ceil(len(matrices) / columns)
    vmax = max(1, max(int(matrix.max()) for matrix in matrices))
    figure = plt.figure(
        figsize=(panel_width * columns + 1.0, panel_height * rows)
    )
    grid = figure.add_gridspec(
        rows,
        columns + 1,
        width_ratios=[1.0] * columns + [0.06],
        wspace=0.25,
        hspace=0.45,
    )
    axes = [
        figure.add_subplot(grid[row, column])
        for row in range(rows)
        for column in range(columns)
    ]
    color_axis = figure.add_subplot(grid[:, -1])
    image = None
    for index, axis in enumerate(axes):
        if index >= len(matrices):
            axis.axis("off")
            continue
        matrix = matrices[index]
        image = axis.imshow(
            matrix, cmap="RdBu_r", vmin=0, vmax=vmax, aspect="equal"
        )
        axis.set_title(f"Layer {layers[index]:02d}", fontsize=title_font_size)
        axis.set_xticks(range(len(labels)))
        axis.set_yticks(range(len(labels)))
        axis.set_xticklabels(
            labels, rotation=x_label_rotation, ha="right",
            fontsize=tick_font_size
        )
        axis.set_yticklabels(labels, fontsize=tick_font_size)
        for row in range(len(labels)):
            for column in range(len(labels)):
                axis.text(
                    column,
                    row,
                    int(matrix[row, column]),
                    ha="center",
                    va="center",
                    fontsize=annotation_font_size,
                )
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label("Top-K feature overlap")
    colorbar.ax.tick_params(labelsize=tick_font_size)
    figure.savefig(
        output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1
    )
    plt.close(figure)


def visualize_stage(args: argparse.Namespace, layers: Sequence[int]) -> None:
    csv_dir = os.path.join(args.out_dir, args.csv_subdir)
    matrices, plotted_layers = [], []
    labels = None
    for layer in layers:
        path = os.path.join(
            csv_dir, f"overlap_matrix_layer_{layer:02d}.csv"
        )
        if not os.path.isfile(path):
            continue
        frame = pd.read_csv(path, index_col=0)
        current = [str(column) for column in frame.columns]
        if labels is None:
            labels = current
        elif labels != current:
            raise ValueError("Matrix labels differ across layers")
        matrices.append(frame.to_numpy(np.int32))
        plotted_layers.append(layer)
    if not matrices:
        raise FileNotFoundError(f"No overlap matrices found in {csv_dir}")
    output = os.path.join(args.out_dir, args.grid_out_name)
    plot_grid(
        matrices,
        plotted_layers,
        labels,
        output,
        args.grid_cols,
        args.dpi,
        args.panel_width,
        args.panel_height,
        args.annotation_font_size,
        args.tick_font_size,
        args.title_font_size,
        args.x_label_rotation,
    )
    print(f"[OK] Saved heatmap: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "predictions_clean -> yearly per-data/per-layer SAE features -> "
            "yearly overlap -> overlap heatmap"
        )
    )
    parser.add_argument(
        "--stage",
        choices=["features", "overlap", "visualize", "all"],
        default="all",
    )
    parser.add_argument("--target-skill", default=None)
    parser.add_argument(
        "--predictions-root", default=DEFAULT_PREDICTIONS_ROOT
    )
    parser.add_argument("--truth-col", default="true_skill")
    parser.add_argument("--text-col", default="masked_sentence")
    parser.add_argument("--row-id-col", default="row_idx")
    parser.add_argument("--years", default="all")
    parser.add_argument(
        "--data-mode", choices=["all", "sample"], default="all"
    )
    parser.add_argument("--samples-per-year", type=int, default=1000)
    parser.add_argument("--read-chunksize", type=int, default=100000)

    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--vocab-path", default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--best-model-pt", default=DEFAULT_MODEL_PT)
    parser.add_argument("--sae-root", default=DEFAULT_SAE_ROOT)
    parser.add_argument("--d-hidden", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument(
        "--amp-dtype", choices=["bf16", "fp16"], default="bf16"
    )
    parser.add_argument(
        "--parquet-compression",
        choices=["zstd", "snappy", "gzip", "none"],
        default="zstd",
    )
    parser.add_argument("--overwrite-features", action="store_true")

    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--features-parquet", default=None)
    parser.add_argument("--group-by", choices=["year"], default="year")
    parser.add_argument("--samples-per-group", type=int, default=None)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--features-dir", default=None)
    parser.add_argument("--layers", default="1-12")
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--font-size", type=int, default=16)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument("--panel-width", type=float, default=5.8)
    parser.add_argument("--panel-height", type=float, default=5.4)
    parser.add_argument(
        "--annotation-font-size", type=float, default=8.0
    )
    parser.add_argument("--tick-font-size", type=float, default=11.0)
    parser.add_argument("--title-font-size", type=float, default=20.0)
    parser.add_argument(
        "--x-label-rotation", type=float, default=45.0
    )
    parser.add_argument(
        "--grid-out-name", default="overlap_heatmap.png"
    )
    parser.add_argument("--csv-subdir", default="csvs")
    parser.add_argument(
        "--aggregate-batch-size", type=int, default=4096
    )
    args = parser.parse_args()

    if args.samples_per_year <= 0:
        parser.error("--samples-per-year must be positive")
    if args.batch_size <= 0 or args.read_chunksize <= 0:
        parser.error("--batch-size and --read-chunksize must be positive")
    if args.topk <= 0 or args.max_iters <= 0:
        parser.error("--topk and --max-iters must be positive")
    if args.grid_cols <= 0:
        parser.error("--grid-cols must be positive")
    if args.panel_width <= 0 or args.panel_height <= 0:
        parser.error("--panel-width and --panel-height must be positive")
    if (
        args.annotation_font_size <= 0
        or args.tick_font_size <= 0
        or args.title_font_size <= 0
    ):
        parser.error("plot font sizes must be positive")
    if not 0 <= args.x_label_rotation <= 90:
        parser.error("--x-label-rotation must be between 0 and 90")

    if args.stage != "visualize" and not args.target_skill and not args.features_parquet:
        parser.error("provide --target-skill or legacy --features-parquet")
    if args.stage == "features" and not args.target_skill:
        parser.error("--stage features requires --target-skill")
    if args.out_dir is None and args.target_skill:
        args.out_dir = os.path.join(
            BASE_DATA_DIR,
            "sparse_auto_encoder",
            f"{skill_tag(args.target_skill)}_yearly",
        )
    elif args.out_dir is None:
        parser.error("--out-dir is required")
    args.out_dir = os.path.abspath(args.out_dir)
    if args.features_dir is None:
        args.features_dir = os.path.join(
            args.out_dir, "features_by_layer"
        )
    args.features_dir = os.path.abspath(args.features_dir)
    return args


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    layers = parse_layers(args.layers)
    mpl.rcParams.update(
        {
            "font.size": args.font_size,
            "axes.titlesize": args.font_size + 2,
            "axes.labelsize": args.font_size,
        }
    )
    # Visualization reuses existing overlap CSVs and needs no source data/model.
    if args.stage == "visualize":
        visualize_stage(args, layers)
        print("[DONE]")
        return
    if args.target_skill:
        files, years = discover_files(
            os.path.abspath(args.predictions_root), parse_years(args.years)
        )
        if len(years) < 2 and args.stage in {"overlap", "all"}:
            raise ValueError("At least two years are required for overlap")
        print(
            f"[INFO] skill={args.target_skill!r}, files={len(files)}, "
            f"years={years}, data_mode={args.data_mode}"
        )
        if args.stage in {"features", "all"}:
            extract_features(args, files, years, layers)
        if args.stage in {"overlap", "all"}:
            overlap_stage(args, layers, years)
    elif args.stage in {"overlap", "all"}:
        legacy_overlap_stage(args, layers)
    if args.stage in {"visualize", "all"}:
        visualize_stage(args, layers)
    print("[DONE]")


if __name__ == "__main__":
    main()

