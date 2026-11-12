#!/usr/bin/env python3
"""Posting-weighted Python SAE year-overlap heatmaps within O*NET occupations.

Reuse the existing complete, unsampled layer-12 cache. Reconstruct its input
order and validate its index before attaching audited raw posting metadata.
The Top-K selection, iterative universal-feature removal, and overlap counts
are imported unchanged from sparse_auto_encoder/yearly/overlap_by_year.py.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "sparse_auto_encoder/yearly/overlap_by_year.py"
spec = importlib.util.spec_from_file_location("existing_yearly_overlap", BASELINE_PATH)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def save_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def parse_years(value):
    years = baseline.parse_years(value)
    require(years is not None and len(years) >= 2, "Specify at least two explicit years")
    return years


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data-dir", type=Path, default=Path(os.environ.get(
        "BASE_DATA_DIR", "/home/jovyan/LEM_data3/hyunjincho/LEM")))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--predictions-root", type=Path)
    parser.add_argument("--posting-metadata", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--years", default="2010-2025")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--min-postings-per-year", type=int, default=10)
    parser.add_argument("--expected-occupations", type=int, default=8,
                        help="Fail if the selected count differs; 0 disables this check")
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--read-chunksize", type=int, default=100000)
    parser.add_argument("--feature-batch-size", type=int, default=2048)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--figure-size", type=float, default=10.0)
    parser.add_argument("--annotation-font-size", type=float, default=9.0)
    parser.add_argument("--tick-font-size", type=float, default=11.0)
    parser.add_argument("--title-font-size", type=float, default=15.0)
    parser.add_argument("--stage", choices=["all", "plot"], default="all",
                        help="plot reuses saved group means without rereading predictions")
    args = parser.parse_args()
    base = args.base_data_dir.expanduser().resolve()
    defaults = {
        "cache_dir": base / "sparse_auto_encoder/python_yearly/overlaps/features_by_layer",
        "predictions_root": base / "predictions_clean/bert_pred/pred",
        "posting_metadata": base / "tmp/python_onet_scale/python_posting_metadata_references.csv.gz",
        "output_dir": base / "tmp/python_sae_onet_min10",
    }
    args.base_data_dir = base
    for name, default in defaults.items():
        setattr(args, name, (getattr(args, name) or default).expanduser().resolve())
    args.years = parse_years(args.years)
    for name in ["min_postings_per_year", "topk", "max_iters", "read_chunksize",
                 "feature_batch_size", "dpi", "figure_size", "annotation_font_size",
                 "tick_font_size", "title_font_size", "layer"]:
        require(getattr(args, name) > 0, f"--{name.replace('_', '-')} must be positive")
    require(args.expected_occupations >= 0, "--expected-occupations must be nonnegative")
    return args


def read_metadata(path):
    fields = ["id", "posted", "posted_year", "onet", "onet_name", "source_file",
              "row_idx", "n_python_instances", "n_valid_mask_instances"]
    meta = pd.read_csv(path, usecols=fields, dtype=str, keep_default_na=False)
    meta = meta.rename(columns={"id": "posting_id", "onet": "onet_code"})
    require(not meta.empty, "Posting metadata is empty")
    for field in ["posting_id", "source_file", "onet_code", "onet_name"]:
        require(meta[field].str.strip().ne("").all(), f"Empty {field} in metadata")
    for field in ["row_idx", "posted_year", "n_python_instances", "n_valid_mask_instances"]:
        meta[field] = pd.to_numeric(meta[field], errors="raise").astype(np.int64)
    require(meta["row_idx"].ge(0).all(), "Negative original row index")
    require(meta["n_python_instances"].gt(0).all(), "Nonpositive Python-instance count")
    require(meta["n_python_instances"].eq(meta["n_valid_mask_instances"]).all(),
            "Audit contains unencoded/missing-mask instances; reextract a fully indexed cache")
    require(not meta.duplicated(["source_file", "row_idx"]).any(),
            "Duplicate (raw source_file, row_idx) in metadata")
    dates = pd.to_datetime(meta["posted"], errors="raise")
    require(dates.dt.year.eq(meta["posted_year"]).all(), "posted_year disagrees with posted")
    for field in ["posted_year", "onet_code", "onet_name"]:
        require(meta.groupby("posting_id")[field].nunique().le(1).all(),
                f"One posting ID has conflicting {field} values")
    return meta


def select_occupations(meta, years, minimum, expected):
    postings = meta.drop_duplicates("posting_id")
    postings = postings[postings["posted_year"].isin(years)]
    postings = postings[postings["onet_code"].ne("99-9999.00")]
    counts = pd.crosstab(postings["onet_code"], postings["posted_year"])
    counts = counts.reindex(columns=years, fill_value=0)
    counts = counts.loc[counts.min(axis=1).ge(minimum)]
    require(len(counts) > 0, "No occupations satisfy the annual posting threshold")
    if expected:
        require(len(counts) == expected,
                f"Expected {expected} occupations, selected {len(counts)}. Inspect input/years.")
    names = postings.groupby("onet_code")["onet_name"].agg(lambda x: " | ".join(sorted(set(x))))
    result = counts.copy()
    result.columns = [str(y) for y in years]
    result.insert(0, "onet_name", names.reindex(result.index))
    result["n_postings"] = counts.sum(axis=1)
    result["min_postings_per_year"] = counts.min(axis=1)
    return result.reset_index().sort_values(["n_postings", "onet_code"], ascending=[False, True])


def validate_cache(args):
    manifest_path = args.cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sig = manifest["signature"]
    require(manifest.get("complete"), "SAE cache is incomplete")
    require(sig.get("format_version") == 1, "Unsupported legacy cache format")
    require(sig.get("target_skill", "").casefold() == "python", "Cache is not Python-only")
    require(sig.get("data_mode") == "all", "A sampled cache cannot support this posting audit")
    require(manifest.get("skipped_no_mask") == 0,
            "Legacy cache skipped instances; it cannot be aligned without explicit instance IDs")
    require(args.layer in sig["layers"], "Requested layer is absent from cache")
    require(set(args.years).issubset(sig["years"]), "Requested years are absent from cache")
    require(Path(sig["predictions_root"]).resolve() == args.predictions_root,
            "Prediction root differs from the cache signature")
    model_config = json.loads((Path(sig["model_name"]) / "config.json").read_text())
    require(model_config["num_hidden_layers"] == args.layer,
            "This analysis requires the final BERT layer")
    for key in ["eligible_counts_by_year", "selected_counts_by_year"]:
        require(manifest[key] == manifest["encoded_counts_by_year"],
                "Cache eligibility/selection/encoding counts differ")
    require(args.topk <= manifest["feature_dim"], "Top-K exceeds the SAE dimension")
    index = pd.read_parquet(args.cache_dir / "data_index.parquet")
    require(np.array_equal(index["sample_id"].to_numpy(), np.arange(len(index))),
            "Legacy cache sample IDs must be contiguous and in extraction order")
    observed = index["year"].value_counts().to_dict()
    require({str(k): int(v) for k, v in observed.items()} == manifest["encoded_counts_by_year"],
            "Cache index counts disagree with its manifest")
    files, _ = baseline.discover_files(str(args.predictions_root), sig["years"])
    cutoff = manifest_path.stat().st_mtime_ns
    require(all(Path(p).stat().st_mtime_ns <= cutoff for p in files),
            "A prediction file is newer than the cache; regenerate the cache before reuse")
    return manifest, index, [Path(p) for p in files]


def align_instances(args, cache_manifest, cache_index, files, meta):
    """Reproduce matching_chunks order; row_idx alone is never a posting key."""
    sig = cache_manifest["signature"]
    truth_col, text_col, row_col = sig["truth_col"], sig["text_col"], sig["row_id_col"]
    usecols = list(dict.fromkeys([truth_col, text_col, row_col, "file_path", "onet_name"]))
    pieces, cursor = [], 0
    input_signatures = []
    for path in tqdm(files, desc="Validate cached input order", unit="file"):
        before = file_signature(path)
        year = int(path.parent.name)
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=args.read_chunksize,
                                 low_memory=False):
            truth = chunk[truth_col].astype("string").str.strip().str.casefold()
            selected = chunk.loc[truth.eq("python") & chunk[text_col].notna()].copy()
            if selected.empty:
                continue
            n = len(selected)
            cached = cache_index.iloc[cursor:cursor+n]
            require(len(cached) == n, "Predictions contain more Python rows than cache")
            row_strings = selected[row_col].astype("string").fillna("").to_numpy()
            require(np.array_equal(cached["source_row_idx"].to_numpy(), row_strings),
                    f"Cache row order mismatch: {path}, Python offset {cursor}")
            require(cached["source_file"].eq(path.name).all() and cached["year"].eq(year).all(),
                    f"Cache file/year order mismatch: {path}")
            require(selected[text_col].astype(str).str.contains("[MASK]", regex=False).all(),
                    f"Missing [MASK] in a supposedly encoded input: {path}")
            piece = pd.DataFrame({
                "sample_id": cached["sample_id"].to_numpy(),
                "source_file": selected["file_path"].astype(str).to_numpy(),
                "row_idx": pd.to_numeric(selected[row_col], errors="raise").to_numpy(np.int64),
                "prediction_file": str(path),
                "prediction_row_number": selected.index.to_numpy(np.int64),
                "cache_year": year,
                "prediction_onet_name": selected["onet_name"].fillna("").astype(str).to_numpy(),
                "masked_text_sha256": [hashlib.sha256(s.encode()).hexdigest()
                                       for s in selected[text_col].astype(str)],
            })
            pieces.append(piece)
            cursor += n
        require(file_signature(path) == before, f"Prediction file changed while reading: {path}")
        input_signatures.append(before)
    require(cursor == len(cache_index), "Predictions contain fewer Python rows than cache")
    instances = pd.concat(pieces, ignore_index=True)
    joined = instances.merge(meta, on=["source_file", "row_idx"], how="left",
                             validate="many_to_one", indicator=True, sort=False)
    require(joined["_merge"].eq("both").all(), "A cached Python instance lacks audited metadata")
    require(joined["posted_year"].eq(joined["cache_year"]).all(),
            "Raw posting year disagrees with the cache year")
    require(joined["onet_name"].eq(joined["prediction_onet_name"]).all(),
            "Prediction and raw O*NET names disagree")
    observed = joined.groupby(["source_file", "row_idx"]).size().sort_index()
    expected = meta.set_index(["source_file", "row_idx"])["n_python_instances"].sort_index()
    require(observed.index.equals(expected.index) and np.array_equal(observed.values, expected.values),
            "Python instance counts do not reproduce the complete test_clean audit")
    joined = joined.drop(columns=["_merge", "prediction_onet_name"])
    require(np.array_equal(joined["sample_id"], np.arange(len(cache_index))),
            "Metadata merge changed instance order")
    return joined, input_signatures


def save_posting_vectors(path, posting_frame, vectors, counts, layer):
    dim = vectors.shape[1]
    data = pa.table({
        "posting_id": posting_frame["posting_id"].tolist(),
        "posted_year": pa.array(posting_frame["posted_year"], type=pa.int16()),
        "onet_code": posting_frame["onet_code"].tolist(),
        "onet_name": posting_frame["onet_name"].tolist(),
        "n_python_instances": pa.array(counts, type=pa.int32()),
        "layer": pa.array(np.full(len(vectors), layer), type=pa.int16()),
        "features": pa.FixedSizeListArray.from_arrays(
            pa.array(vectors.reshape(-1), type=pa.float32()), dim),
    })
    pq.write_table(data, path, compression="zstd")


def aggregate(args, manifest, instances, selected):
    codes = selected["onet_code"].tolist()
    dimension = int(manifest["feature_dim"])
    means = np.empty((len(codes), len(args.years), dimension), dtype=np.float64)
    code_lookup = {code: i for i, code in enumerate(codes)}
    eligible = instances["onet_code"].isin(codes) & instances["posted_year"].isin(args.years)
    instances["selected_for_analysis"] = eligible
    instances.to_parquet(args.output_dir / "validated_instance_index.parquet", index=False)
    out = args.output_dir / "posting_representations"
    out.mkdir()
    counts_rows, feature_signatures = [], []
    for yi, year in enumerate(tqdm(args.years, desc="Posting means", unit="year")):
        year_all = instances.loc[instances["cache_year"].eq(year)]
        year_selected = year_all.loc[year_all["selected_for_analysis"]]
        posting_frame = (year_selected[["posting_id", "posted_year", "onet_code", "onet_name"]]
                         .drop_duplicates("posting_id").sort_values("posting_id").reset_index(drop=True))
        position_by_id = pd.Index(posting_frame["posting_id"])
        # -1 denotes an instance outside the eight selected occupations.
        instance_positions = np.full(len(instances), -1, dtype=np.int64)
        instance_positions[year_selected["sample_id"].to_numpy()] = position_by_id.get_indexer(
            year_selected["posting_id"])
        sums = np.zeros((len(posting_frame), dimension), dtype=np.float64)
        counts = np.zeros(len(posting_frame), dtype=np.int64)
        cache_file = args.cache_dir / f"layer_{args.layer:02d}" / f"year_{year}.parquet"
        before = file_signature(cache_file)
        parquet = pq.ParquetFile(cache_file)
        expected_ids = year_all["sample_id"].to_numpy()
        cursor = 0
        for batch in parquet.iter_batches(batch_size=args.feature_batch_size,
                                          columns=["sample_id", "year", "layer", "features"]):
            ids = batch.column(0).to_numpy()
            require(np.array_equal(ids, expected_ids[cursor:cursor+len(ids)]),
                    f"Feature sample IDs/order disagree with index: {cache_file}")
            require(np.all(batch.column(1).to_numpy() == year) and
                    np.all(batch.column(2).to_numpy() == args.layer),
                    f"Wrong feature year/layer: {cache_file}")
            array = batch.column(3)
            require(pa.types.is_fixed_size_list(array.type) and array.type.list_size == dimension,
                    "Wrong SAE feature dimension")
            values = array.flatten().to_numpy(zero_copy_only=False).reshape(len(array), dimension)
            require(np.isfinite(values).all() and (values >= 0).all(),
                    f"Nonfinite or negative ReLU SAE activation: {cache_file}")
            positions = instance_positions[ids]
            keep = positions >= 0
            np.add.at(sums, positions[keep], values[keep])
            np.add.at(counts, positions[keep], 1)
            cursor += len(ids)
        require(cursor == len(expected_ids), f"Feature rows are missing: {cache_file}")
        require(file_signature(cache_file) == before, f"Feature file changed while reading: {cache_file}")
        feature_signatures.append(before)
        expected_counts = year_selected.groupby("posting_id").size().reindex(posting_frame["posting_id"])
        require(np.array_equal(counts, expected_counts.to_numpy()) and (counts > 0).all(),
                f"Posting occurrence counts disagree for {year}")
        # Sum occurrences in float64, average within posting, then retain float32
        # posting vectors. Year means use those saved vectors with float64 sums.
        sums /= counts[:, None]
        posting_means = sums.astype(np.float32)
        del sums
        save_posting_vectors(out / f"year_{year}.parquet", posting_frame, posting_means, counts, args.layer)
        for code, ci in code_lookup.items():
            keep = posting_frame["onet_code"].eq(code).to_numpy()
            n_postings = int(keep.sum())
            require(n_postings == int(selected.loc[selected["onet_code"].eq(code), str(year)].iloc[0]),
                    f"Posting count differs from audited selection: {code}, {year}")
            require(n_postings >= args.min_postings_per_year, "Annual posting threshold no longer holds")
            means[ci, yi] = posting_means[keep].mean(axis=0, dtype=np.float64)
            counts_rows.append({"onet_code": code, "onet_name": posting_frame.loc[keep, "onet_name"].iloc[0],
                                "posted_year": year, "n_postings": n_postings,
                                "n_python_instances": int(counts[keep].sum()),
                                "layer": args.layer, "failed_instances": 0, "dropped_instances": 0})
    np.savez_compressed(args.output_dir / "group_means.npz", means=means,
                        onet_codes=np.asarray(codes), years=np.asarray(args.years), layer=args.layer)
    count_frame = pd.DataFrame(counts_rows)
    count_frame.to_csv(args.output_dir / "posting_counts_by_onet_year.csv", index=False)
    return count_frame, feature_signatures


def draw_heatmap(matrix, years, counts, code, name, excluded_count, args, output):
    figure, axis = plt.subplots(figsize=(args.figure_size+1.5, args.figure_size), layout="constrained")
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=0, vmax=args.topk, aspect="equal")
    axis.set_xticks(range(len(years)), [str(y) for y in years], rotation=45, ha="right",
                    fontsize=args.tick_font_size)
    axis.set_yticks(range(len(years)), [f"{y} (n={int(counts[y])})" for y in years],
                    fontsize=args.tick_font_size)
    for i in range(len(years)):
        for j in range(len(years)):
            axis.text(j, i, str(int(matrix[i,j])), ha="center", va="center",
                      fontsize=args.annotation_font_size, color="black")
    axis.set_title(f"{code} | {textwrap.fill(name, 58)}\nPython | Layer {args.layer:02d}",
                   fontsize=args.title_font_size, pad=14)
    axis.set_xlabel("Posting year", fontsize=args.tick_font_size)
    axis.set_ylabel("Posting year (n = unique postings)", fontsize=args.tick_font_size)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.03)
    colorbar.set_label("Top-K feature overlap", fontsize=args.tick_font_size)
    colorbar.ax.tick_params(labelsize=args.tick_font_size)
    figure.suptitle(f"Top-{args.topk}; iterative common-feature removal ({excluded_count} removed)",
                    fontsize=args.tick_font_size)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(figure)


def plot_outputs(args, run_manifest):
    with np.load(args.output_dir / "group_means.npz", allow_pickle=False) as saved:
        means, codes, years, layer = saved["means"], saved["onet_codes"].tolist(), saved["years"].tolist(), int(saved["layer"])
    require(years == args.years and layer == args.layer, "Plot arguments differ from saved years/layer")
    selected = pd.read_csv(args.output_dir / "selected_occupations.csv", dtype={"onet_code":str})
    counts = pd.read_csv(args.output_dir / "posting_counts_by_onet_year.csv", dtype={"onet_code":str})
    require(codes == selected["onet_code"].tolist(), "Saved occupation order differs from saved means")
    require(args.topk <= means.shape[-1], "Top-K exceeds SAE dimension")
    require(run_manifest["parameters"]["min_postings_per_year"] == args.min_postings_per_year,
            "Plot annual threshold differs from saved selection")
    all_overlap, all_top, occupation_rows = [], [], []
    for ci, code in enumerate(codes):
        name = selected.loc[selected["onet_code"].eq(code), "onet_name"].iloc[0]
        year_means = {year:means[ci, yi] for yi, year in enumerate(years)}
        top_sets, excluded, iterations = baseline.remove_universal_features(
            year_means, years, args.topk, args.max_iters, args.layer)
        require(all(len(top_sets[y]) == args.topk for y in years),
                f"Too few features remain after common-feature removal: {code}")
        matrix = np.asarray([[len(np.intersect1d(top_sets[a], top_sets[b])) for b in years]
                             for a in years], dtype=np.int32)
        require(np.array_equal(matrix, matrix.T) and np.all(np.diag(matrix) == args.topk),
                f"Invalid overlap matrix: {code}")
        directory = args.output_dir / code
        directory.mkdir(exist_ok=True)
        pd.DataFrame(matrix, index=years, columns=years).to_csv(
            directory / f"overlap_matrix_layer_{args.layer:02d}.csv")
        save_json(directory / "excluded_feature_ids.json", {
            "onet_code":code, "layer":args.layer, "topk":args.topk, "iterations":iterations,
            "excluded_feature_ids":sorted(int(x) for x in excluded),
            "exclusion_scope":"within this O*NET occupation across the requested years"})
        zero_counts = {}
        for yi, year in enumerate(years):
            zero_counts[year] = int(np.count_nonzero(year_means[year][top_sets[year]] == 0))
            for rank, feature_id in enumerate(top_sets[year], 1):
                all_top.append({"onet_code":code, "posted_year":year, "layer":args.layer,
                                "rank":rank, "feature_id":int(feature_id),
                                "mean_activation":float(year_means[year][feature_id])})
            for yj, other in enumerate(years):
                all_overlap.append({"onet_code":code, "year_i":year, "year_j":other,
                                    "overlap":int(matrix[yi,yj]), "topk":args.topk,
                                    "excluded_features":len(excluded), "iterations":iterations})
        annual_counts = counts.loc[counts["onet_code"].eq(code)].set_index("posted_year")["n_postings"].to_dict()
        draw_heatmap(matrix, years, annual_counts, code, name, len(excluded), args,
                     directory / "overlap_heatmap_large.png")
        occupation_rows.append({"onet_code":code, "onet_name":name, "excluded_features":len(excluded),
                                "iterations":iterations, "max_zero_activation_top_features":max(zero_counts.values())})
        print(f"[OK] {code}: {directory / 'overlap_heatmap_large.png'}", flush=True)
    pd.DataFrame(all_overlap).to_csv(args.output_dir / "overlap_longform.csv", index=False)
    pd.DataFrame(all_top).to_csv(args.output_dir / "top_feature_ids_by_onet_year.csv", index=False)
    pd.DataFrame(occupation_rows).to_csv(args.output_dir / "feature_removal_summary.csv", index=False)
    return occupation_rows


def main():
    args = parse_args()
    output = args.output_dir
    if args.stage == "all":
        require(not output.exists() or not any(output.iterdir()),
                "Output directory is not empty. Use a new --output-dir, or --stage plot for saved means.")
        output.mkdir(parents=True, exist_ok=True)
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                               text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = None
        run = {"status":"running", "started_utc":datetime.now(timezone.utc).isoformat(),
               "parameters":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
               "script_sha256":sha256(__file__), "baseline_sha256":sha256(BASELINE_PATH),
               "git_revision":revision,
               "aggregation":"mean SAE activation within posting, then equal-weight mean of postings within O*NET/year",
               "overlap":"unchanged baseline Top-K, iterative universal-feature removal within occupation, intersection count",
               "software":{"python":sys.version, "numpy":np.__version__, "pandas":pd.__version__,
                           "pyarrow":pa.__version__, "matplotlib":matplotlib.__version__},
               "limitations":["Legacy cache has no text hashes. Alignment reproduces original extraction order and checks row/file/year sequences, full occurrence counts, and source timestamps.",
                              "Universal features are removed separately within each occupation; excluded feature sets can differ across occupations."]}
    else:
        run = json.loads((output / "run_manifest.json").read_text())
        require(run.get("aggregation_complete"), "No completed group means to plot")
    try:
        if args.stage == "all":
            save_json(output / "run_manifest.json", run)
            metadata_signature = file_signature(args.posting_metadata)
            metadata = read_metadata(args.posting_metadata)
            run["posting_metadata"] = {**metadata_signature, "sha256":sha256(args.posting_metadata)}
            selected = select_occupations(metadata, args.years, args.min_postings_per_year,
                                          args.expected_occupations)
            selected.to_csv(output / "selected_occupations.csv", index=False)
            print("[SELECTED]\n" + selected[["onet_code","onet_name","n_postings","min_postings_per_year"]].to_string(index=False), flush=True)
            manifest, index, files = validate_cache(args)
            run["cache_manifest"] = manifest
            run["cache_manifest_sha256"] = sha256(args.cache_dir / "manifest.json")
            run["cache_index"] = {**file_signature(args.cache_dir / "data_index.parquet"),
                                  "sha256":sha256(args.cache_dir / "data_index.parquet")}
            instances, predictions = align_instances(args, manifest, index, files, metadata)
            require(file_signature(args.posting_metadata) == metadata_signature, "Metadata changed while reading")
            counts, features = aggregate(args, manifest, instances, selected)
            run.update({"aggregation_complete":True, "n_occupations":len(selected),
                        "n_audited_python_instances":len(instances),
                        "n_audited_python_postings":int(instances["posting_id"].nunique()),
                        "n_selected_instances":int(counts["n_python_instances"].sum()),
                        "n_selected_postings":int(counts["n_postings"].sum()),
                        "out_of_scope_instances":int(len(instances)-counts["n_python_instances"].sum()),
                        "failed_instances":0, "dropped_instances":0,
                        "prediction_files":predictions, "feature_files":features,
                        "group_means_sha256":sha256(output / "group_means.npz")})
            save_json(output / "run_manifest.json", run)
        require(sha256(output / "group_means.npz") == run["group_means_sha256"], "Saved means checksum mismatch")
        run["occupation_feature_removal"] = plot_outputs(args, run)
        run.update({"status":"complete", "completed_utc":datetime.now(timezone.utc).isoformat(),
                    "plot_parameters":{"topk":args.topk, "max_iters":args.max_iters,
                                       "dpi":args.dpi, "figure_size":args.figure_size},
                    "plot_script_sha256":sha256(__file__)})
        save_json(output / "run_manifest.json", run)
        print(f"[COMPLETE] {output}", flush=True)
    except Exception as error:
        run.update({"status":"failed", "error":f"{type(error).__name__}: {error}"})
        save_json(output / "run_manifest.json", run)
        raise


if __name__ == "__main__":
    main()
