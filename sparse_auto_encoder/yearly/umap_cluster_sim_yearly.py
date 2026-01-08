#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UMAP + Element-centric cluster similarity (CluSim) across BERT layers
Supports grouping by:
  - field (industry/occupation)
  - year  (continuous colormap in UMAP)

NEW:
  --group-by year
  --years (optional) if omitted -> infer all years from matched files (fast path)

Minimal Output (only what downstream scripts need):
  out_dir/
    element_centric_similarity_by_layer.csv
    element_centric_similarity_random_null_by_layer.csv
    umap_2d.csv
"""

import os, re, glob, json, random, argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt  # kept (harmless), but we won't save plots here

from model import BERTForSkillPrediction

from clusim.clustering import Clustering
import clusim.sim as sim

mpl.rcParams.update({
    "font.size": 24,
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 18,
    "figure.titlesize": 18,
})

FIELD_LABEL_MAP = {
    "Computer and Mathematical Occupations": "Computer",
    "Business and Financial Operations Occupations": "Financial",
    "Management Occupations": "Management",
    "Sales and Related Occupations": "Sales",
    "Educational Instruction and Library Occupations": "Educational",
}

# ------------------------------- utils -------------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def extract_year_from_path(path: str) -> int:
    m = re.search(r"/(?:train|test)/(20\d{2})/", path)
    if m:
        return int(m.group(1))
    m2 = re.search(r"preprocessed_(20\d{2})-\d{2}\.csv", os.path.basename(path))
    return int(m2.group(1)) if m2 else -1

def infer_years_from_files(files: List[str]) -> List[int]:
    ys = []
    for fp in files:
        y = extract_year_from_path(fp)
        if y != -1:
            ys.append(y)
    return sorted(list(set(ys)))

@dataclass
class Reservoir:
    k: int
    data: List[Tuple[str, int]]  # (sentence, year)
    seen: int = 0

    def add(self, item: Tuple[str, int]):
        self.seen += 1
        if len(self.data) < self.k:
            self.data.append(item)
        else:
            j = random.randint(1, self.seen)
            if j <= self.k:
                self.data[j - 1] = item

# ------------------------------- SAE -------------------------------

class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.W_e = nn.Linear(d_in, d_hidden, bias=True)
        self.W_d = nn.Linear(d_hidden, d_in, bias=False)
        self.b_d = nn.Parameter(torch.zeros(d_in))

    def encode(self, x):
        return F.relu(self.W_e(x - self.b_d))

@torch.no_grad()
def forward_until_layer(bert, input_ids, attention_mask, layer: int):
    x = bert.embeddings(input_ids)
    ext_mask = bert.get_extended_attention_mask(attention_mask, attention_mask.shape).to(x.device)
    for i in range(layer):
        x = bert.encoder.layer[i](x, ext_mask)[0]
    return x

def load_sae_for_layer(sae_root: str, layer: int, d_in: int, d_hidden: int, device: str):
    ckpt = os.path.join(sae_root, f"layer_{layer:02d}", "sae_best.pt")
    sae = SparseAutoencoder(d_in=d_in, d_hidden=d_hidden).to(device)
    sae.load_state_dict(torch.load(ckpt, map_location="cpu"))
    sae.eval()
    return sae

# ------------------------------- tokenize -------------------------------

def batch_tokenize(tokenizer, sentences: List[str], max_len: int):
    enc = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attn_mask = enc["attention_mask"]

    mask_pos = (input_ids == tokenizer.mask_token_id)
    has_mask = mask_pos.any(dim=1)
    if not bool(has_mask.all()):
        keep = has_mask.nonzero(as_tuple=True)[0]
        input_ids = input_ids[keep]
        attn_mask = attn_mask[keep]
        mask_pos = mask_pos[keep]

    if input_ids.numel() == 0:
        return input_ids, attn_mask, None, None

    mask_idx = mask_pos.float().argmax(dim=1)
    keep_idx = torch.arange(input_ids.size(0))
    return input_ids, attn_mask, mask_idx, keep_idx

# ------------------------------- sampling -------------------------------

def sample_per_group(
    files: List[str],
    truth_col: str,
    text_col: str,
    field_col: Optional[str],
    target_skill: str,
    group_by: str,                      # "field" or "year"
    field_values: Optional[List[str]],
    years: Optional[List[int]],
    per_group: int,
    seed: int,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    random.seed(seed)

    if group_by == "field":
        assert field_col is not None
        assert field_values is not None and len(field_values) > 0
        groups = [str(v) for v in field_values]
    else:
        assert years is not None and len(years) > 0
        groups = [str(y) for y in years]  # group label as string

    res_k = max(per_group * 3, per_group)
    reservoirs: Dict[str, Reservoir] = {g: Reservoir(k=res_k, data=[]) for g in groups}

    usecols = [text_col, truth_col]
    if group_by == "field":
        usecols.append(field_col)

    for fp in tqdm(files, desc=f"Sampling (stream, group_by={group_by})"):
        file_year = extract_year_from_path(fp)

        if group_by == "year" and years is not None and file_year not in years:
            continue

        for chunk in pd.read_csv(fp, compression="gzip", usecols=usecols, chunksize=chunksize):
            chunk = chunk.dropna(subset=[text_col, truth_col])

            sub = chunk[chunk[truth_col].astype(str) == target_skill]
            if len(sub) == 0:
                continue

            if group_by == "field":
                sub = sub.dropna(subset=[field_col])
                sub = sub[sub[field_col].astype(str).isin(field_values)]
                if len(sub) == 0:
                    continue
                for s, fv in zip(sub[text_col].astype(str).tolist(),
                                 sub[field_col].astype(str).tolist()):
                    reservoirs[str(fv)].add((s, file_year))
            else:
                g = str(file_year)
                if g not in reservoirs:
                    continue
                for s in sub[text_col].astype(str).tolist():
                    reservoirs[g].add((s, file_year))

    rows = []
    for g in groups:
        pool = reservoirs[g].data
        if len(pool) == 0:
            print(f"[WARN] No samples collected for group='{g}'.")
            continue
        take = min(per_group, len(pool))
        picked = random.sample(pool, take)
        for s, y in picked:
            rows.append({"group": g, "sentence": s, "year": int(y)})
        if take < per_group:
            print(f"[WARN] group='{g}' only has {take} samples (<{per_group}).")

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

# ------------------------------- feature extraction -------------------------------

@torch.no_grad()
def extract_features_all_layers(
    df_samples: pd.DataFrame,
    bert,
    tokenizer,
    layers: List[int],
    device: str,
    batch_size: int,
    max_len: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
    repr_mode: str,
    sae_root: Optional[str] = None,
    d_hidden: int = 4096,
) -> pd.DataFrame:
    assert repr_mode in ("sae", "bert")

    sentences = df_samples["sentence"].tolist()
    groups = df_samples["group"].astype(str).tolist()
    years = df_samples["year"].tolist()
    n = len(sentences)

    all_rows = []
    d_in = bert.config.hidden_size
    out_dim = d_hidden if repr_mode == "sae" else d_in

    for layer in tqdm(layers, desc=f"Layers (extract repr={repr_mode})"):
        sae = None
        if repr_mode == "sae":
            if sae_root is None:
                raise ValueError("--sae-root is required when --repr sae")
            sae = load_sae_for_layer(sae_root, layer, d_in=d_in, d_hidden=d_hidden, device=device)

        for i in tqdm(range(0, n, batch_size), desc=f"  layer={layer:02d}", leave=False):
            batch_s = sentences[i:i+batch_size]
            batch_g = groups[i:i+batch_size]
            batch_y = years[i:i+batch_size]

            input_ids, attn_mask, mask_idx, keep_idx = batch_tokenize(tokenizer, batch_s, max_len=max_len)
            if input_ids.numel() == 0 or mask_idx is None:
                continue

            kept_groups = [batch_g[j] for j in keep_idx.tolist()]
            kept_years  = [batch_y[j] for j in keep_idx.tolist()]
            kept_sent_i = [i + j for j in keep_idx.tolist()]

            input_ids = input_ids.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)
            mask_idx  = mask_idx.to(device, non_blocking=True)

            use_autocast = (use_amp and device.startswith("cuda") and torch.cuda.is_available())
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_autocast):
                h = forward_until_layer(bert, input_ids, attn_mask, layer)
                x = h[torch.arange(h.size(0), device=device), mask_idx]
                f = sae.encode(x) if repr_mode == "sae" else x

            f = f.float().cpu().numpy()

            for b in range(f.shape[0]):
                row = {
                    "sample_id": int(kept_sent_i[b]),
                    "group": str(kept_groups[b]),
                    "year": int(kept_years[b]),
                    "layer": int(layer),
                }
                row.update({f"f{j}": float(f[b, j]) for j in range(out_dim)})
                all_rows.append(row)

        if sae is not None:
            del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(all_rows)

# ------------------------------- CluSim -------------------------------

def element_centric_similarity_partition(gt_labels: np.ndarray, pred_labels: np.ndarray, alpha: float = 0.90) -> float:
    c_gt = Clustering().from_membership_list(gt_labels.tolist())
    c_pr = Clustering().from_membership_list(pred_labels.tolist())
    return float(sim.element_sim(c_gt, c_pr, alpha=alpha, r=1.0))

# ------------------------------- UMAP -------------------------------

def run_umap(X: np.ndarray, n_neighbors: int, min_dist: float, metric: str, seed: int) -> np.ndarray:
    import umap
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
        n_components=2,
    )
    return reducer.fit_transform(X)

# ------------------------------- k-means + similarity -------------------------------

def kmeans_and_similarity_per_layer(
    df_feat: pd.DataFrame,
    feat_cols: List[str],
    group_values: List[str],
    alpha: float,
    seed: int,
    k: int,
):
    from sklearn.cluster import KMeans

    group_to_gt = {g: i for i, g in enumerate(group_values)}

    layers = sorted(df_feat["layer"].unique().tolist())
    sim_rows, rand_rows = [], []

    for layer in tqdm(layers, desc="Layer-wise k-means + similarity"):
        sub = df_feat[df_feat["layer"] == layer].copy()
        sub = sub.sort_values("sample_id").reset_index(drop=True)

        X = sub[feat_cols].to_numpy(np.float32)
        gt_group = sub["group"].astype(str).tolist()
        gt = np.array([group_to_gt[g] for g in gt_group], dtype=np.int64)

        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        pred = km.fit_predict(X).astype(np.int64)
        sim_km = element_centric_similarity_partition(gt, pred, alpha=alpha)
        sim_rows.append({"layer": int(layer), "similarity": float(sim_km), "n": int(len(sub))})

        rng = np.random.default_rng(seed + int(layer))
        pred_rand = rng.integers(low=0, high=k, size=len(gt), dtype=np.int64)
        sim_rand = element_centric_similarity_partition(gt, pred_rand, alpha=alpha)
        rand_rows.append({"layer": int(layer), "similarity": float(sim_rand), "n": int(len(sub))})

    return pd.DataFrame(sim_rows), pd.DataFrame(rand_rows)

# ------------------------------- main -------------------------------

def parse_layers(s: str) -> List[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-pattern", type=str,
                    default="/home/jovyan/LEM_data2/hyunjincho/preprocessed_www/test/20*/preprocessed_*csv.gz")
    ap.add_argument("--truth-col", type=str, default="true_skill")
    ap.add_argument("--text-col", type=str, default="masked_sentence")
    ap.add_argument("--target-skill", type=str, required=True)

    ap.add_argument("--group-by", type=str, default="field", choices=["field", "year"])
    ap.add_argument("--field-col", type=str, default=None)
    ap.add_argument("--field-values", nargs="+", default=None)

    ap.add_argument("--years", nargs="+", type=int, default=None,
                    help="If --group-by year and omitted, infer all years from files.")

    ap.add_argument("--per-group", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--model-name", type=str, required=True)
    ap.add_argument("--vocab-path", type=str, required=True)
    ap.add_argument("--best-model-pt", type=str, required=True)

    ap.add_argument("--repr", type=str, default="sae", choices=["sae", "bert"])
    ap.add_argument("--sae-root", type=str, default=None)
    ap.add_argument("--d-hidden", type=int, default=4096)

    ap.add_argument("--layers", type=str, default="1-12")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=512)

    ap.add_argument("--use-amp", action="store_true")
    ap.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    ap.add_argument("--umap-n-neighbors", type=int, default=30)
    ap.add_argument("--umap-min-dist", type=float, default=0.10)
    ap.add_argument("--umap-metric", type=str, default="cosine")

    ap.add_argument("--kmeans-k", type=int, default=None)
    ap.add_argument("--ec-alpha", type=float, default=0.90)

    ap.add_argument("--out-dir", type=str, default=None)

    args = ap.parse_args()

    if args.group_by == "field":
        if not args.field_col:
            raise ValueError("--field-col is required when --group-by field")
        if not args.field_values:
            raise ValueError("--field-values is required when --group-by field")

    if args.repr == "sae" and not args.sae_root:
        raise ValueError("--sae-root is required when --repr sae")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.out_dir is None:
        tag = args.target_skill.replace(" ", "_")
        args.out_dir = f"viz_{args.group_by}_{args.repr}_{tag}"
    ensure_dir(args.out_dir)

    layers = parse_layers(args.layers)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    files = sorted(glob.glob(args.test_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.test_pattern}")
    print(f"[INFO] Found {len(files)} files")

    # Infer years if needed
    if args.group_by == "year":
        if args.years is None or len(args.years) == 0:
            args.years = infer_years_from_files(files)
            if not args.years:
                raise RuntimeError("Failed to infer years from file paths.")
            print(f"[INFO] Inferred years: {args.years[:5]} ... {args.years[-5:]} (n={len(args.years)})")
        else:
            args.years = sorted(list({int(y) for y in args.years}))

    # 1) sampling (in-memory only; no samples.csv saved)
    df_samples = sample_per_group(
        files=files,
        truth_col=args.truth_col,
        text_col=args.text_col,
        field_col=args.field_col if args.group_by == "field" else None,
        target_skill=args.target_skill,
        group_by=args.group_by,
        field_values=args.field_values if args.group_by == "field" else None,
        years=args.years if args.group_by == "year" else None,
        per_group=args.per_group,
        seed=args.seed,
    )

    if df_samples.empty:
        raise RuntimeError("No samples collected. Check target skill / columns / filters.")

    if args.group_by == "field":
        group_values = [str(v) for v in args.field_values]
    else:
        group_values = [str(y) for y in args.years]

    # 2) load model
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    with open(args.vocab_path) as f:
        skill2idx = json.load(f)

    model = BERTForSkillPrediction(args.model_name, num_skills=len(skill2idx)).to(args.device)
    model.load_state_dict(torch.load(args.best_model_pt, map_location="cpu"))
    model.eval()

    bert = model.bert.to(args.device)
    bert.eval()
    for p in bert.parameters():
        p.requires_grad_(False)

    # 3) features (NO parquet saved)
    df_feat = extract_features_all_layers(
        df_samples=df_samples,
        bert=bert,
        tokenizer=tokenizer,
        layers=layers,
        device=args.device,
        batch_size=args.batch_size,
        max_len=args.max_len,
        use_amp=args.use_amp,
        amp_dtype=amp_dtype,
        repr_mode=args.repr,
        sae_root=args.sae_root,
        d_hidden=args.d_hidden,
    )

    if df_feat.empty:
        raise RuntimeError("Feature extraction produced empty dataframe.")

    feat_cols = [c for c in df_feat.columns if re.fullmatch(r"f\d+", c)]
    print(f"[INFO] feature dim = {len(feat_cols)} (repr={args.repr})")

    # 4) k-means + similarity (save ONLY 2 CSVs)
    k = args.kmeans_k if args.kmeans_k is not None else len(group_values)
    df_sim, df_rand = kmeans_and_similarity_per_layer(
        df_feat=df_feat,
        feat_cols=feat_cols,
        group_values=group_values,
        alpha=args.ec_alpha,
        seed=args.seed,
        k=k,
    )

    out_sim_csv = os.path.join(args.out_dir, "element_centric_similarity_by_layer.csv")
    out_rand_csv = os.path.join(args.out_dir, "element_centric_similarity_random_null_by_layer.csv")
    df_sim.to_csv(out_sim_csv, index=False)
    df_rand.to_csv(out_rand_csv, index=False)
    print(f"[INFO] Saved: {out_sim_csv}")
    print(f"[INFO] Saved: {out_rand_csv}")

    # 5) UMAP (save ONLY umap_2d.csv)
    X_all = df_feat[feat_cols].to_numpy(np.float32)
    Y = run_umap(
        X_all,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        seed=args.seed,
    )

    df_umap = df_feat[["sample_id", "group", "year", "layer"]].copy()
    df_umap["u0"] = Y[:, 0]
    df_umap["u1"] = Y[:, 1]

    out_umap = os.path.join(args.out_dir, "umap_2d.csv")
    df_umap.to_csv(out_umap, index=False)
    print(f"[DONE] Saved: {out_umap}")

if __name__ == "__main__":
    main()
    
    
    """
python3 vis_umap_cluster_sim_yearly.py \
  --group-by year \
  --test-pattern "/home/jovyan/LEM_data2/hyunjincho/preprocessed_www_new/test/20*/preprocessed_*soc.csv.gz" \
  --target-skill python \
  --per-group 500 \
  --repr sae \
  --sae-root /home/jovyan/LEM_data2/hyunjincho/sae_layerwise_out \
  --model-name /home/jovyan/LEM_data2/hyunjincho/bert_pretrained/checkpoint-165687 \
  --vocab-path /home/jovyan/LEM_data2/hyunjincho/preprocessed_www_new/skill2idx.json \
  --best-model-pt "/home/jovyan/LEM_data2/hyunjincho/checkpoints(www)_new/best_model.pt" \
  --use-amp --amp-dtype bf16 \
  --out-dir ./python_yearly \
  --year-cmap viridis
    """