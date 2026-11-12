#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob, json, random, argparse, sys, gc
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "model_trains"))
sys.path.insert(0, MODEL_DIR)

from model import BERTForSkillPrediction

from clusim.clustering import Clustering
import clusim.sim as sim


# ------------------------------- utils -------------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def extract_year_from_path(path: str) -> int:
    """
    Extract year from a preprocessed file path.
    Examples:
      .../preprocessed_www/train/2010/preprocessed_2010-01.csv.gz
      .../preprocessed_www/test/2010/preprocessed_2010-01.csv.gz
    """
    m = re.search(r"/(?:train|test(?:_clean)?)/(20\d{2})/", path)
    if m:
        return int(m.group(1))
    m2 = re.search(r"preprocessed_(20\d{2})-\d{2}\.csv", os.path.basename(path))
    return int(m2.group(1)) if m2 else -1


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
    """
    Return hidden states after applying encoder.layer[0..layer-1].
    So layer=1 returns the output of the 1st transformer block.
    """
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

    keep = has_mask.nonzero(as_tuple=True)[0]
    if not bool(has_mask.all()):
        input_ids = input_ids[keep]
        attn_mask = attn_mask[keep]
        mask_pos = mask_pos[keep]

    if input_ids.numel() == 0:
        return input_ids, attn_mask, None, None

    mask_idx = mask_pos.float().argmax(dim=1)
    return input_ids, attn_mask, mask_idx, keep


# ------------------------------- sampling -------------------------------

def sample_per_field(
    files: List[str],
    truth_col: str,
    text_col: str,
    field_col: str,
    target_skill: str,
    field_values: List[str],
    per_field: int,
    seed: int,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """
    Sample per_field examples per field value using reservoir sampling.
    Stores (sentence, year).
    """
    random.seed(seed)

    # Oversample reservoir to reduce variance; we later subsample to per_field.
    res_k = max(per_field * 3, per_field)
    reservoirs: Dict[str, Reservoir] = {v: Reservoir(k=res_k, data=[]) for v in field_values}

    usecols = [text_col, truth_col, field_col]

    for fp in tqdm(files, desc="Sampling (stream)"):
        year = extract_year_from_path(fp)
        for chunk in pd.read_csv(fp, compression="gzip", usecols=usecols, chunksize=chunksize):
            chunk = chunk.dropna(subset=[text_col, truth_col, field_col])

            sub = chunk[chunk[truth_col].astype(str) == target_skill]
            if len(sub) == 0:
                continue

            sub = sub[sub[field_col].astype(str).isin(field_values)]
            if len(sub) == 0:
                continue

            for s, fv in zip(sub[text_col].astype(str).tolist(), sub[field_col].astype(str).tolist()):
                reservoirs[fv].add((s, year))

    rows = []
    for fv in field_values:
        pool = reservoirs[fv].data
        if len(pool) == 0:
            print(f"[WARN] No samples collected for field='{fv}'.")
            continue

        take = min(per_field, len(pool))
        picked = random.sample(pool, take)
        for s, y in picked:
            rows.append({"field": fv, "sentence": s, "year": y})

        if take < per_field:
            print(f"[WARN] field='{fv}' only has {take} samples (<{per_field}).")

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def collect_all_per_field(
    files: List[str], truth_col: str, text_col: str, field_col: str,
    target_skill: str, field_values: List[str], chunksize: int = 200_000,
) -> pd.DataFrame:
    """Collect every eligible target-skill row for cluster similarity."""
    parts = []
    usecols = [text_col, truth_col, field_col]
    for fp in tqdm(files, desc="Collect all rows for similarity"):
        year = extract_year_from_path(fp)
        for chunk in pd.read_csv(fp, compression="gzip", usecols=usecols, chunksize=chunksize):
            chunk = chunk.dropna(subset=[text_col, truth_col, field_col])
            sub = chunk[
                chunk[truth_col].astype(str).eq(target_skill)
                & chunk[field_col].astype(str).isin(field_values)
            ]
            if not sub.empty:
                parts.append(pd.DataFrame({
                    "field": sub[field_col].astype(str).to_numpy(),
                    "sentence": sub[text_col].astype(str).to_numpy(),
                    "year": year,
                }))
    if not parts:
        return pd.DataFrame(columns=["field", "sentence", "year"])
    return pd.concat(parts, ignore_index=True)


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
    repr_mode: str,                 # "sae" or "bert"
    sae_root: Optional[str] = None, # required if repr_mode=="sae"
    d_hidden: int = 4096,           # SAE hidden dim
) -> pd.DataFrame:
    """
    df_samples: columns [field, sentence, year]
    returns long-form df with columns [sample_id, field, year, layer, f0..f(D-1)]
      - repr_mode="bert": D=768
      - repr_mode="sae" : D=d_hidden (e.g. 4096)
    """
    assert repr_mode in ("sae", "bert")

    sentences = df_samples["sentence"].tolist()
    fields = df_samples["field"].tolist()
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
            batch_s = sentences[i:i + batch_size]
            batch_f = fields[i:i + batch_size]
            batch_y = years[i:i + batch_size]

            input_ids, attn_mask, mask_idx, keep_idx = batch_tokenize(tokenizer, batch_s, max_len=max_len)
            if input_ids.numel() == 0 or mask_idx is None:
                continue

            kept_fields = [batch_f[j] for j in keep_idx.tolist()]
            kept_years  = [batch_y[j] for j in keep_idx.tolist()]
            kept_sent_i = [i + j for j in keep_idx.tolist()]  # global sample id

            input_ids = input_ids.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)
            mask_idx  = mask_idx.to(device, non_blocking=True)

            use_autocast = (use_amp and device.startswith("cuda") and torch.cuda.is_available())
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_autocast):
                h = forward_until_layer(bert, input_ids, attn_mask, layer)
                x = h[torch.arange(h.size(0), device=device), mask_idx]  # (B, 768)

                if repr_mode == "sae":
                    f = sae.encode(x)  # (B, d_hidden)
                else:
                    f = x              # (B, 768)

            f = f.float().cpu().numpy()  # (B, out_dim)

            for b in range(f.shape[0]):
                row = {
                    "sample_id": int(kept_sent_i[b]),
                    "field": kept_fields[b],
                    "year": int(kept_years[b]),
                    "layer": int(layer),
                }
                row.update({f"f{j}": float(f[b, j]) for j in range(out_dim)})
                all_rows.append(row)

        if sae is not None:
            del sae
        torch.cuda.empty_cache()

    return pd.DataFrame(all_rows)


# ------------------------------- element-centric similarity (CluSim) -------------------------------

def element_centric_similarity_partition(gt_labels: np.ndarray, pred_labels: np.ndarray, alpha: float = 0.90) -> float:
    """Exact CluSim element similarity for hard partitions without NxN matrices."""
    if gt_labels.shape != pred_labels.shape:
        raise ValueError("Partition label arrays must have the same shape.")
    if len(gt_labels) == 0:
        return float("nan")
    _, gt = np.unique(gt_labels, return_inverse=True)
    _, pred = np.unique(pred_labels, return_inverse=True)
    table = np.zeros((gt.max() + 1, pred.max() + 1), dtype=np.int64)
    np.add.at(table, (gt, pred), 1)
    denom = np.maximum(table.sum(axis=1)[:, None], table.sum(axis=0)[None, :])
    return float(np.sum(table * table / denom) / len(gt))


# ------------------------------- k-means per layer + similarity (+ random null) -------------------------------

def kmeans_and_similarity_per_layer(
    df_feat: pd.DataFrame,
    feat_cols: List[str],
    field_values: List[str],
    alpha: float,
    seed: int,
    k: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - df_sim  : k-means element-centric similarity by layer
      - df_rand : random-null element-centric similarity by layer
    (Assignments are NOT stored to reduce outputs.)
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise RuntimeError("scikit-learn is required for KMeans. Install with: pip install scikit-learn") from e

    field_to_gt = {f: i for i, f in enumerate(field_values)}

    layers = sorted(df_feat["layer"].unique().tolist())
    sim_rows = []
    rand_rows = []

    for layer in tqdm(layers, desc="Layer-wise k-means + similarity"):
        sub = df_feat[df_feat["layer"] == layer].copy()
        sub = sub.sort_values("sample_id").reset_index(drop=True)

        X = sub[feat_cols].to_numpy(np.float32)
        gt_field = sub["field"].astype(str).tolist()
        gt = np.array([field_to_gt[f] for f in gt_field], dtype=np.int64)

        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        pred = km.fit_predict(X).astype(np.int64)
        sim_km = element_centric_similarity_partition(gt, pred, alpha=alpha)

        sim_rows.append({"layer": int(layer), "similarity": float(sim_km), "n": int(len(sub))})

        rng = np.random.default_rng(seed + int(layer))
        pred_rand = rng.integers(low=0, high=k, size=len(gt), dtype=np.int64)
        sim_rand = element_centric_similarity_partition(gt, pred_rand, alpha=alpha)

        rand_rows.append({"layer": int(layer), "similarity": float(sim_rand), "n": int(len(sub))})

    return pd.DataFrame(sim_rows), pd.DataFrame(rand_rows)


@torch.no_grad()
def full_data_similarity_per_layer(
    df_all, bert, tokenizer, layers, device, batch_size, max_len,
    use_amp, amp_dtype, repr_mode, sae_root, d_hidden,
    field_values, alpha, seed, k, checkpoint_dir,
):
    """Cluster each sampled/full layer with KMeans and checkpoint its similarity."""
    from sklearn.cluster import KMeans

    sentences = df_all["sentence"].astype(str).tolist()
    labels = df_all["field"].astype(str).tolist()
    label_to_gt = {v: i for i, v in enumerate(field_values)}
    n = len(sentences)
    out_dim = d_hidden if repr_mode == "sae" else bert.config.hidden_size
    sim_path = os.path.join(checkpoint_dir, "element_centric_similarity_by_layer.csv")
    rand_path = os.path.join(checkpoint_dir, "element_centric_similarity_random_null_by_layer.csv")
    cache_dir = os.path.join(checkpoint_dir, ".similarity_cache")
    ensure_dir(cache_dir)

    def load_results(csv_path):
        if not os.path.isfile(csv_path):
            return []
        old = pd.read_csv(csv_path)
        if "data_scope" not in old.columns:
            return []
        return old[old["data_scope"].eq("full")].to_dict("records")

    sim_rows = [
        row for row in load_results(sim_path)
        if row.get("clustering") == "KMeans"
    ]
    kmeans_layers = {int(row["layer"]) for row in sim_rows}
    rand_rows = [
        row for row in load_results(rand_path)
        if int(row["layer"]) in kmeans_layers
    ]
    completed = {int(r["layer"]) for r in sim_rows} & {int(r["layer"]) for r in rand_rows}

    for layer in tqdm(layers, desc="KMeans similarity by layer"):
        if layer in completed:
            print(f"[RESUME] Layer {layer:02d} similarity already saved.")
            continue
        sae = load_sae_for_layer(sae_root, layer, bert.config.hidden_size, d_hidden, device) if repr_mode == "sae" else None
        feature_path = os.path.join(cache_dir, f"layer_{layer:02d}_features.f32")
        label_path = os.path.join(cache_dir, f"layer_{layer:02d}_labels.i32")
        meta_path = os.path.join(cache_dir, f"layer_{layer:02d}.json")
        cache_ready = all(os.path.isfile(p) for p in (feature_path, label_path, meta_path))

        if cache_ready:
            with open(meta_path) as handle:
                offset = int(json.load(handle)["rows"])
            print(f"[RESUME] Reusing layer {layer:02d} cache ({offset:,} rows).")
        else:
            Xw = np.memmap(feature_path, mode="w+", dtype=np.float32, shape=(n, out_dim))
            yw = np.memmap(label_path, mode="w+", dtype=np.int32, shape=(n,))
            offset = 0
            for start in tqdm(range(0, n, batch_size), desc=f"  layer={layer:02d}", leave=False):
                ids, mask, mask_idx, keep_idx = batch_tokenize(tokenizer, sentences[start:start + batch_size], max_len=max_len)
                if ids.numel() == 0 or mask_idx is None:
                    continue
                ids, mask, mask_idx = ids.to(device), mask.to(device), mask_idx.to(device)
                enabled = use_amp and device.startswith("cuda") and torch.cuda.is_available()
                with torch.autocast("cuda", dtype=amp_dtype, enabled=enabled):
                    h = forward_until_layer(bert, ids, mask, layer)
                    x = h[torch.arange(h.size(0), device=device), mask_idx]
                    features = sae.encode(x) if repr_mode == "sae" else x
                features = features.float().cpu().numpy()
                kept = keep_idx.tolist()
                count = len(kept)
                Xw[offset:offset + count] = features
                yw[offset:offset + count] = [label_to_gt[labels[start + j]] for j in kept]
                offset += count
            Xw.flush(); yw.flush()
            del Xw, yw
            with open(meta_path, "w") as handle:
                json.dump({"rows": offset, "dim": out_dim}, handle)
            print(f"[CHECKPOINT] Layer {layer:02d} feature cache saved ({offset:,} rows).")

        X = np.memmap(feature_path, mode="r", dtype=np.float32, shape=(n, out_dim))
        ymap = np.memmap(label_path, mode="r", dtype=np.int32, shape=(n,))
        y = np.asarray(ymap[:offset])
        X_fit = np.asarray(X[:offset])
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        pred = km.fit_predict(X_fit).astype(np.int32)

        score = element_centric_similarity_partition(y, pred, alpha)
        sim_rows.append({"layer": layer, "similarity": score, "n": offset, "data_scope": "full", "clustering": "KMeans"})
        rng = np.random.default_rng(seed + layer)
        random_pred = rng.integers(0, k, size=offset, dtype=np.int32)
        random_score = element_centric_similarity_partition(y, random_pred, alpha)
        rand_rows.append({"layer": layer, "similarity": random_score, "n": offset, "data_scope": "full", "clustering": "random"})
        pd.DataFrame(sim_rows).sort_values("layer").to_csv(sim_path, index=False)
        pd.DataFrame(rand_rows).sort_values("layer").to_csv(rand_path, index=False)
        print(f"[CHECKPOINT] Layer {layer:02d} similarity saved.")

        del X, X_fit, ymap, y, pred, random_pred, km, sae
        for p in (feature_path, label_path, meta_path):
            if os.path.exists(p): os.remove(p)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    return pd.DataFrame(sim_rows).sort_values("layer"), pd.DataFrame(rand_rows).sort_values("layer")


# ------------------------------- main -------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--test-pattern", type=str,
                    default=os.path.join(os.environ.get("BASE_DATA_DIR", "/home/jovyan/LEM_data2/data"), "preprocessed_www_new", "test", "20*", "preprocessed_*csv.gz"))
    ap.add_argument("--truth-col", type=str, default="true_skill")
    ap.add_argument("--text-col", type=str, default="masked_sentence")

    ap.add_argument("--target-skill", type=str, required=True)

    ap.add_argument("--field-col", type=str, required=True, help="e.g., lot_v7_career_area_name")
    ap.add_argument("--field-values", nargs="+", required=True,
                    help='e.g., "Information Technology and Computer Science" Engineering Finance Healthcare')

    ap.add_argument("--per-field", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--model-name", type=str, required=True)
    ap.add_argument("--vocab-path", type=str, required=True)
    ap.add_argument("--best-model-pt", type=str, required=True)

    # representation mode
    ap.add_argument("--repr", type=str, default="sae", choices=["sae", "bert"],
                    help="Feature representation: 'sae' uses SAE codes; 'bert' uses raw BERT layer output at [MASK].")

    ap.add_argument("--sae-root", type=str, default=None,
                    help="Required only when --repr sae. Root dir containing layer_{XX}/sae_best.pt")
    ap.add_argument("--d-hidden", type=int, default=4096)

    ap.add_argument("--layers", type=str, default="1-12", help="e.g., 1-12 or 1,3,5")
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=512)

    ap.add_argument("--use-amp", action="store_true")
    ap.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    ap.add_argument("--umap-n-neighbors", type=int, default=30)
    ap.add_argument("--umap-min-dist", type=float, default=0.10)
    ap.add_argument("--umap-metric", type=str, default="cosine")

    ap.add_argument("--kmeans-k", type=int, default=None, help="default: len(field_values)")
    ap.add_argument("--ec-alpha", type=float, default=0.90)
    ap.add_argument(
        "--similarity-per-field", type=int, default=None,
        help="Balanced cluster-similarity sample per SOC field; default uses all rows.",
    )

    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    # validate
    if args.repr == "sae" and not args.sae_root:
        raise ValueError("--sae-root is required when --repr sae")
    if args.repr == "bert" and args.sae_root:
        print("[INFO] --repr bert: ignoring --sae-root (not needed).")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.out_dir is None:
        tag = args.target_skill.replace(" ", "_")
        args.out_dir = f"viz_{args.repr}_{tag}"
    ensure_dir(args.out_dir)

    # layers parse
    if "-" in args.layers:
        a, b = args.layers.split("-")
        layers = list(range(int(a), int(b) + 1))
    else:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    files = sorted(glob.glob(args.test_pattern))

    # If field_col is LOT7 career area, exclude *_soc.csv.gz (those are the SOC-labeled split)
    if args.field_col == "lot_v7_career_area_name":
        files = [f for f in files if "_soc.csv.gz" not in os.path.basename(f)]
        print("[INFO] Excluding *_soc.csv.gz files (LOT7 field)")

    if not files:
        raise FileNotFoundError(f"No files matched after filtering: {args.test_pattern}")

    print(f"[INFO] Found {len(files)} files")

    out_feat = os.path.join(args.out_dir, "features.parquet")
    if os.path.isfile(out_feat):
        df_feat = pd.read_parquet(out_feat)
        df_samples = None
        print(f"[RESUME] Reusing sampled UMAP features: {out_feat} (rows={len(df_feat)})")
    else:
        # 1) balanced sampling for UMAP only
        df_samples = sample_per_field(
            files=files,
            truth_col=args.truth_col,
            text_col=args.text_col,
            field_col=args.field_col,
            target_skill=args.target_skill,
            field_values=args.field_values,
            per_field=args.per_field,
            seed=args.seed,
        )
        print(f"[INFO] samples collected: n={len(df_samples)}")

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

    # 3) sampled features for UMAP only
    if df_samples is not None:
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
        print(f"[INFO] features extracted (rows={len(df_feat)})")
        df_feat.to_parquet(out_feat, index=False)
        print(f"[OK] Saved: {out_feat}")
    
    feat_cols = [c for c in df_feat.columns if re.fullmatch(r"f\d+", c)]
    print(f"[INFO] feature dim = {len(feat_cols)} (repr={args.repr})")

    # 4) Cluster similarity optionally uses a balanced sample per SOC field.
    full_rows_cache = os.path.join(args.out_dir, "full_similarity_rows.parquet")
    if os.path.isfile(full_rows_cache):
        df_all = pd.read_parquet(full_rows_cache)
        print(f"[RESUME] Reusing full similarity rows: {full_rows_cache}")
    else:
        df_all = collect_all_per_field(
            files=files,
            truth_col=args.truth_col,
            text_col=args.text_col,
            field_col=args.field_col,
            target_skill=args.target_skill,
            field_values=args.field_values,
        )
        if df_all.empty:
            raise RuntimeError("No full-data rows collected for cluster similarity.")
        df_all.to_parquet(full_rows_cache, index=False)
        print(f"[CHECKPOINT] Full similarity rows saved: {full_rows_cache}")
    print(f"[INFO] eligible similarity rows: n={len(df_all)}")
    similarity_checkpoint_dir = args.out_dir
    if args.similarity_per_field is not None:
        if args.similarity_per_field <= 0:
            raise ValueError("--similarity-per-field must be positive.")
        similarity_checkpoint_dir = os.path.join(
            args.out_dir, f"similarity_per_field_{args.similarity_per_field}"
        )
        ensure_dir(similarity_checkpoint_dir)
        sampled_rows_cache = os.path.join(similarity_checkpoint_dir, "similarity_rows.parquet")
        if os.path.isfile(sampled_rows_cache):
            df_all = pd.read_parquet(sampled_rows_cache)
            print(f"[RESUME] Reusing balanced similarity sample: {sampled_rows_cache}")
        else:
            counts = df_all.groupby("field", observed=True).size()
            missing = [v for v in args.field_values if counts.get(v, 0) < args.similarity_per_field]
            if missing:
                details = ", ".join(f"{v}={int(counts.get(v, 0))}" for v in missing)
                raise RuntimeError(
                    f"Not enough rows for --similarity-per-field={args.similarity_per_field}: {details}"
                )
            df_all = (
                df_all.groupby("field", group_keys=False, observed=True)
                .sample(n=args.similarity_per_field, replace=False, random_state=args.seed)
                .sample(frac=1.0, random_state=args.seed)
                .reset_index(drop=True)
            )
            df_all.to_parquet(sampled_rows_cache, index=False)
            print(f"[CHECKPOINT] Balanced similarity sample saved: {sampled_rows_cache}")
        print(
            f"[INFO] balanced SOC similarity rows: n={len(df_all)} "
            f"({args.similarity_per_field} per field)"
        )
    k = args.kmeans_k if args.kmeans_k is not None else len(args.field_values)
    df_sim, df_rand = full_data_similarity_per_layer(
        df_all, bert, tokenizer, layers, args.device, args.batch_size,
        args.max_len, args.use_amp, amp_dtype, args.repr, args.sae_root,
        args.d_hidden, args.field_values, args.ec_alpha, args.seed, k,
        similarity_checkpoint_dir,
    )
    del df_all

    out_sim_csv = os.path.join(args.out_dir, "element_centric_similarity_by_layer.csv")
    df_sim.to_csv(out_sim_csv, index=False)
    print(f"[OK] Saved: {out_sim_csv}")

    out_rand_csv = os.path.join(args.out_dir, "element_centric_similarity_random_null_by_layer.csv")
    df_rand.to_csv(out_rand_csv, index=False)
    print(f"[OK] Saved: {out_rand_csv}")

    merged = df_sim.merge(df_rand, on=["layer", "n"], suffixes=("_kmeans", "_rand"))
    print("\n===== Element-centric similarity: k-means vs Random-null =====")
    for _, r in merged.sort_values("layer").iterrows():
        print(f"Layer {int(r['layer']):02d} | kmeans={r['similarity_kmeans']:.4f} | rand={r['similarity_rand']:.4f}")
    print("-------------------------------------------------------------")
    print(f"MEAN  | kmeans={merged['similarity_kmeans'].mean():.4f} | rand={merged['similarity_rand'].mean():.4f}\n")

    # 5) UMAP coordinates (save CSV ONLY; replot later with vis_umap_conti.py)
    try:
        import umap
    except ImportError as e:
        raise RuntimeError("umap-learn is not installed. Install with: pip install umap-learn") from e

    X_all = df_feat[feat_cols].to_numpy(np.float32)
    reducer = umap.UMAP(
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.seed,
        n_components=2,
    )
    Y = reducer.fit_transform(X_all)

    df_umap = df_feat[["sample_id", "field", "year", "layer"]].copy()
    df_umap["u0"] = Y[:, 0]
    df_umap["u1"] = Y[:, 1]

    out_umap = os.path.join(args.out_dir, "umap_2d.csv")
    df_umap.to_csv(out_umap, index=False)
    print(f"[OK] Saved: {out_umap}")


if __name__ == "__main__":
    main()

"""
python3 vis_umap_cluster_sim_industry.py \
  --repr sae \
  --test-pattern "/home/jovyan/LEM_data2/data/preprocessed_www_new/t*/20*/preprocessed_*.csv.gz" \
  --target-skill "python" \
  --field-col soc_2_name \
  --field-values "Computer and Mathematical Occupations" "Business and Financial Operations Occupations" "Management Occupations" "Sales and Related Occupations" "Educational Instruction and Library Occupations" \
  --per-field 250 \
  --model-name /home/jovyan/LEM_data2/data/bert_pretrained \
  --vocab-path /home/jovyan/LEM_data2/data/preprocessed_www_new/skill2idx.json \
  --best-model-pt "/home/jovyan/LEM_data2/data/checkpoints(www)_new/best_model.pt" \
  --sae-root /home/jovyan/LEM_data2/data/sae_layerwise_out_8192 \
  --use-amp --amp-dtype bf16 \
  --out-dir ./python_industry
"""
