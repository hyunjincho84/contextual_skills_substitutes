#!/usr/bin/env python3
"""Exact annual mean pairwise cosine similarity of posting-averaged Python SAE vectors.

This measures within-year concentration/dispersion, not maximum pairwise distance
(diameter). All 4096 features are used; the overlap feature-removal rule is not
applied. All Python postings and the previously selected O*NET groups are reported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
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

HELPER_PATH = Path(__file__).with_name("python_onet_yearly_overlap.py")
spec = importlib.util.spec_from_file_location("posting_overlap_helpers", HELPER_PATH)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
require = helper.require


def cosine_summary(vectors):
    """Exact off-diagonal pair mean, with O(nd) work and no n-by-n matrix.

    u_i = z_i / ||z_i||; sum_{i != j} u_i dot u_j = ||sum_i u_i||^2 - n.
    Zero vectors have undefined cosine and are counted separately.
    """
    values = np.asarray(vectors, dtype=np.float64)
    require(values.ndim == 2 and np.isfinite(values).all(), "Invalid posting vectors")
    norms = np.linalg.norm(values, axis=1)
    valid = norms > 0
    unit = values[valid] / norms[valid, None]
    n = len(unit)
    numerator = float(np.square(unit.sum(axis=0)).sum() - n) if n else np.nan
    denominator = n * (n - 1)
    mean = float(np.clip(numerator / denominator, -1, 1)) if denominator else np.nan
    return {"n_postings":len(values), "n_valid_postings":n,
            "n_zero_norm_postings":int((~valid).sum()), "n_unordered_pairs":denominator // 2,
            "pair_cosine_numerator_ordered":numerator,
            "pair_cosine_denominator_ordered":denominator,
            "mean_pairwise_cosine":mean, "mean_cosine_distance":1 - mean}


def validate_source(source_dir, years):
    run = json.loads((source_dir / "run_manifest.json").read_text())
    require(run.get("status") == "complete", "Source overlap analysis is incomplete")
    require(set(years).issubset(run["parameters"]["years"]), "Requested years absent from source run")
    for signature in run["prediction_files"] + run["feature_files"]:
        require(helper.file_signature(signature["path"]) == signature,
                f"Source file changed since validation: {signature['path']}")
    meta_sig = run["posting_metadata"]
    require(helper.sha256(meta_sig["path"]) == meta_sig["sha256"], "Audited metadata changed")
    cache = Path(run["parameters"]["cache_dir"])
    require(helper.sha256(cache / "manifest.json") == run["cache_manifest_sha256"],
            "SAE cache manifest changed")
    require(helper.sha256(cache / "data_index.parquet") == run["cache_index"]["sha256"],
            "SAE cache index changed")
    instances = pd.read_parquet(source_dir / "validated_instance_index.parquet")
    index = pd.read_parquet(cache / "data_index.parquet")
    require(np.array_equal(instances["sample_id"], np.arange(len(index))), "Instance IDs/order changed")
    require(np.array_equal(index["source_row_idx"], instances["row_idx"].astype(str)),
            "Original row indices disagree")
    require(np.array_equal(index["year"], instances["posted_year"]), "Posting years disagree")
    names = instances["prediction_file"].str.rsplit("/", n=1).str[-1]
    require(np.array_equal(index["source_file"], names), "Prediction-file identities disagree")
    meta = helper.read_metadata(Path(meta_sig["path"]))
    fields = ["source_file", "row_idx", "posting_id", "posted_year", "onet_code", "onet_name"]
    check = instances[fields].merge(meta[fields], on=["source_file", "row_idx"],
                                   how="left", validate="many_to_one", suffixes=("", "_audit"))
    for field in fields[2:]:
        require(check[field].eq(check[field + "_audit"]).all(), f"Metadata mismatch: {field}")
    actual = instances.groupby(["source_file", "row_idx"]).size().sort_index()
    expected = meta.set_index(["source_file", "row_idx"])["n_python_instances"].sort_index()
    require(actual.index.equals(expected.index) and np.array_equal(actual, expected),
            "Instance counts do not reproduce the complete posting audit")
    selected = helper.select_occupations(meta, run["parameters"]["years"],
                                         run["parameters"]["min_postings_per_year"], run["n_occupations"])
    return run, cache, instances, selected


def posting_vectors(instances, cache, year, layer, dimension, batch_size):
    rows = instances.loc[instances["posted_year"].eq(year)]
    postings = (rows[["posting_id", "posted_year", "onet_code", "onet_name"]]
                .drop_duplicates("posting_id").sort_values("posting_id").reset_index(drop=True))
    lookup = np.full(len(instances), -1, dtype=np.int64)
    lookup[rows["sample_id"].to_numpy()] = pd.Index(postings["posting_id"]).get_indexer(rows["posting_id"])
    sums = np.zeros((len(postings), dimension), dtype=np.float64)
    counts = np.zeros(len(postings), dtype=np.int64)
    expected_ids = rows["sample_id"].to_numpy()
    path = cache / f"layer_{layer:02d}" / f"year_{year}.parquet"
    signature = helper.file_signature(path)
    cursor = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        ids = batch.column(batch.schema.get_field_index("sample_id")).to_numpy()
        require(np.array_equal(ids, expected_ids[cursor:cursor+len(ids)]), "Feature instance order mismatch")
        require(np.all(batch.column(batch.schema.get_field_index("year")).to_numpy() == year), "Wrong year")
        require(np.all(batch.column(batch.schema.get_field_index("layer")).to_numpy() == layer), "Wrong layer")
        array = batch.column(batch.schema.get_field_index("features"))
        require(array.type.list_size == dimension, "Wrong feature dimension")
        values = array.flatten().to_numpy(zero_copy_only=False).reshape(len(array), dimension)
        require(np.isfinite(values).all() and (values >= 0).all(), "Invalid ReLU activations")
        positions = lookup[ids]
        require((positions >= 0).all(), "A feature instance is missing its posting ID")
        np.add.at(sums, positions, values)
        np.add.at(counts, positions, 1)
        cursor += len(ids)
    require(cursor == len(expected_ids), "Missing feature rows")
    require(helper.file_signature(path) == signature, "Feature file changed during read")
    expected_counts = rows.groupby("posting_id").size().reindex(postings["posting_id"])
    require(np.array_equal(counts, expected_counts) and (counts > 0).all(), "Posting counts disagree")
    sums /= counts[:, None]
    return postings, sums.astype(np.float32), counts


def draw_overall_only(frame, output, dpi):
    overall = frame.loc[frame["scope"].eq("all_python")].sort_values("posted_year")
    figure, axis = plt.subplots(figsize=(10, 4), layout="constrained")
    axis.plot(overall["posted_year"], overall["mean_cosine_distance"], "o-", color="#2166ac")
    axis.set_ylabel("Mean pairwise cosine distance")
    axis.set_xlabel("Posting year")
    axis.set_title("Python | Layer 12 | Within-year representation dispersion")
    axis.set_xticks(overall["posted_year"], labels=overall["posted_year"].astype(str), rotation=45)
    axis.grid(alpha=.2)
    figure.savefig(output / "overall_dispersion_only.png", dpi=dpi)
    plt.close(figure)


def draw_results(frame, selected, output, dpi):
    overall = frame.loc[frame["scope"].eq("all_python")]
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    axes[0].plot(overall["posted_year"], overall["mean_cosine_distance"], "o-", color="#2166ac")
    axes[0].set_ylabel("Mean pairwise cosine distance")
    axes[0].set_title("Python | Layer 12 | Within-year representation dispersion")
    axes[0].grid(alpha=.2)
    axes[1].plot(overall["posted_year"], overall["n_valid_postings"], "o-", color="#666666")
    axes[1].set_ylabel("Valid unique postings")
    axes[1].set_xlabel("Posting year")
    axes[1].set_xticks(overall["posted_year"], labels=overall["posted_year"].astype(str), rotation=45)
    axes[1].grid(alpha=.2)
    figure.savefig(output / "overall_dispersion.png",dpi=dpi)
    plt.close(figure)
    draw_overall_only(frame, output, dpi)
    columns = min(4,len(selected))
    figure, axes = plt.subplots((len(selected)+columns-1)//columns,columns,
                                figsize=(4.8*columns,4.2*((len(selected)+columns-1)//columns)),
                                squeeze=False,sharex=True,sharey=True,layout="constrained")
    for axis, row in zip(axes.flat,selected.itertuples(index=False)):
        data = frame.loc[frame["scope"].eq(row.onet_code)]
        axis.plot(data["posted_year"],data["mean_cosine_distance"],"o-",color="#2166ac",markersize=4)
        axis.set_title(f"{row.onet_code}\n{textwrap.fill(row.onet_name,36)}",fontsize=10)
        axis.set_xlabel(f"Posting year | n={data.n_valid_postings.min()}–{data.n_valid_postings.max()}")
        axis.set_ylabel("Mean pairwise cosine distance")
        ticks = data["posted_year"].iloc[::3]
        axis.set_xticks(ticks,labels=ticks.astype(str),rotation=45)
        axis.grid(alpha=.2)
    for axis in axes.flat[len(selected):]: axis.axis("off")
    figure.suptitle("Python | Layer 12 | Within-occupation annual dispersion (full SAE vectors)")
    figure.savefig(output / "onet_dispersion.png",dpi=dpi)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data-dir",type=Path,default=Path(os.environ.get("BASE_DATA_DIR","/home/jovyan/LEM_data3/hyunjincho/LEM")))
    parser.add_argument("--source-run",type=Path)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--years",default="2010-2025")
    parser.add_argument("--batch-size",type=int,default=2048)
    parser.add_argument("--dpi",type=int,default=200)
    parser.add_argument("--stage",choices=["all","plot"],default="all")
    args = parser.parse_args()
    source = (args.source_run or args.base_data_dir / "tmp/python_sae_onet_min10").resolve()
    output = (args.output_dir or args.base_data_dir / "tmp/python_sae_yearly_dispersion").resolve()
    years = helper.parse_years(args.years)
    require(args.batch_size>0 and args.dpi>0,"Batch size and DPI must be positive")
    if args.stage == "all":
        require(not output.exists() or not any(output.iterdir()),"Choose an empty --output-dir")
    else:
        require((output / "yearly_dispersion.csv").is_file(),"No saved annual metrics to plot")
    output.mkdir(parents=True,exist_ok=True)
    manifest = {"status":"running","started_utc":datetime.now(timezone.utc).isoformat(),
                "metric":"within-year mean off-diagonal pairwise cosine; dispersion = 1 - mean cosine",
                "is_maximum_pairwise_diameter":False,
                "aggregation":"mean SAE vectors within posting, L2 normalize each posting vector, equal-weight distinct posting pairs",
                "features":"all SAE dimensions, without Top-K or common-feature removal",
                "parameters":{"source_run":str(source),"output_dir":str(output),"years":years,
                              "batch_size":args.batch_size,"dpi":args.dpi,"stage":args.stage},
                "code_sha256":helper.sha256(__file__),"helper_sha256":helper.sha256(HELPER_PATH),
                "software":{"python":sys.version,"numpy":np.__version__,"pandas":pd.__version__,
                            "pyarrow":pa.__version__,"matplotlib":matplotlib.__version__},
                "uncertainty":"Not estimated; pair counts are not independent sample sizes"}
    if args.stage == "plot":
        previous = json.loads((output / "run_manifest.json").read_text())
        require(previous["parameters"]["years"] == years and
                previous["parameters"]["source_run"] == str(source),"Plot source/years differ from saved metrics")
        manifest["metric_code_sha256"] = previous.get("metric_code_sha256",previous["code_sha256"])
    else:
        manifest["metric_code_sha256"] = manifest["code_sha256"]
    helper.save_json(output / "run_manifest.json",manifest)
    try:
        run, cache, instances, selected = validate_source(source,years)
        selected.to_csv(output / "selected_occupations.csv",index=False)
        layer = int(run["parameters"]["layer"])
        require(layer==12,"This plot is configured for the final layer 12")
        dimension = int(run["cache_manifest"]["feature_dim"])
        metrics, zero_rows, yearly_audits = [], [], []
        for year in tqdm(years if args.stage == "all" else [],desc="Annual cosine dispersion",unit="year"):
            postings, vectors, occurrence_counts = posting_vectors(instances,cache,year,layer,dimension,args.batch_size)
            scopes = [("all_python","All Python postings",np.ones(len(postings),dtype=bool))]
            scopes.extend((r.onet_code,r.onet_name,postings["onet_code"].eq(r.onet_code).to_numpy())
                          for r in selected.itertuples(index=False))
            for code, name, keep in scopes:
                stats = cosine_summary(vectors[keep])
                metrics.append({"scope":code,"scope_name":name,"posted_year":year,"layer":layer,
                                "n_python_instances":int(occurrence_counts[keep].sum()),**stats})
            zero = ~np.any(vectors != 0,axis=1)
            zero_rows.extend(postings.loc[zero].to_dict("records"))
            yearly_audits.append({"posted_year":year,"n_postings":len(postings),
                                  "n_python_instances":int(occurrence_counts.sum()),
                                  "zero_norm_postings":int(zero.sum()),"failed_instances":0})
            pd.DataFrame(metrics).to_csv(output / "yearly_dispersion.csv",index=False)
        if args.stage == "all":
            result = pd.DataFrame(metrics)
            pd.DataFrame(zero_rows,columns=["posting_id","posted_year","onet_code","onet_name"]).to_csv(
                output / "zero_norm_postings.csv",index=False)
        else:
            result = pd.read_csv(output / "yearly_dispersion.csv",dtype={"scope":str})
            zero_rows = pd.read_csv(output / "zero_norm_postings.csv").to_dict("records")
            expected_scopes = set(selected["onet_code"]) | {"all_python"}
            require(set(result["scope"]) == expected_scopes and len(result)==len(years)*len(expected_scopes),
                    "Saved metric scopes/counts disagree")
            require(not result.duplicated(["scope","posted_year"]).any(),"Duplicate annual metric rows")
            for row in result.itertuples(index=False):
                eligible = instances[instances["posted_year"].eq(row.posted_year)]
                if row.scope != "all_python": eligible=eligible[eligible["onet_code"].eq(row.scope)]
                require(row.n_postings==eligible["posting_id"].nunique() and row.n_python_instances==len(eligible),
                        "Saved sample sizes disagree with validated input")
            require(np.allclose(result["mean_cosine_distance"],1-result["mean_pairwise_cosine"]),
                    "Saved metric definitions disagree")
            for row in result[result.scope.eq("all_python")].itertuples(index=False):
                yearly_audits.append({"posted_year":row.posted_year,"n_postings":row.n_postings,
                                      "n_python_instances":row.n_python_instances,
                                      "zero_norm_postings":row.n_zero_norm_postings,"failed_instances":0})
        draw_results(result,selected,output,args.dpi)
        manifest.update({"status":"complete","completed_utc":datetime.now(timezone.utc).isoformat(),
                         "source_manifest_sha256":helper.sha256(source / "run_manifest.json"),
                         "source_index_sha256":helper.sha256(source / "validated_instance_index.parquet"),
                         "cache_manifest":run["cache_manifest"],"yearly_audits":yearly_audits,
                         "n_total_postings":int(sum(r["n_postings"] for r in yearly_audits)),
                         "n_total_python_instances":int(sum(r["n_python_instances"] for r in yearly_audits)),
                         "n_zero_norm_postings":len(zero_rows),"failed_instances":0})
        helper.save_json(output / "run_manifest.json",manifest)
        print(result.loc[result.scope.eq("all_python"),["posted_year","n_postings","mean_pairwise_cosine","mean_cosine_distance"]].to_string(index=False))
        print(f"[COMPLETE] {output}")
    except Exception as error:
        manifest.update({"status":"failed","error":f"{type(error).__name__}: {error}"})
        helper.save_json(output / "run_manifest.json",manifest)
        raise


if __name__ == "__main__":
    main()
