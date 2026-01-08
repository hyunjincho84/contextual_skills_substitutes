#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Union Mean Activation Heatmap Pipeline
#  1) Raw union-mean heatmap (top-K, cluster-ordered)
#  2) Thresholded union-mean heatmap (quantile-based)
# ============================================================

# ---------- helpers ----------
BLUE="\033[1;34m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
RESET="\033[0m"

section () {
  echo -e "\n${BLUE}══════════════════════════════════════════════════════════════${RESET}"
  echo -e "${BLUE}▶ $1${RESET}"
  echo -e "${BLUE}══════════════════════════════════════════════════════════════${RESET}\n"
}

step () {
  echo -e "${GREEN}➤ $1${RESET}"
}

ts () {
  echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${RESET} $1"
}

trap 'echo -e "${RED}✗ Failed at line $LINENO${RESET}"' ERR

PYTHON_BIN="python3"

# ---------- paths ----------
FEATURES_PARQUET="../sparse_auto_encoder/python_industry/features.parquet"
OUT_DIR="../sparse_auto_encoder/python_industry/feature_id_vis"

# ---------- common params ----------
LAYERS="1-12"
TOPK=64
SAMPLES_PER_INDUSTRY=250
CANDIDATE_MULT=10
N_CLUSTERS=12
FONT_SIZE=16
COL_ORDER="cluster"

# ============================================================
# [1/2] Union mean activation heatmap (raw)
# ============================================================
section "[1/2] Union mean activation heatmap (raw)"
ts "Running vis_union_mean_activation_heatmap.py"

$PYTHON_BIN ../sparse_auto_encoder/industry/vis_union_mean_activation_heatmap.py \
  --features-parquet "${FEATURES_PARQUET}" \
  --out-dir "${OUT_DIR}" \
  --layers ${LAYERS} \
  --topk ${TOPK} \
  --samples-per-industry ${SAMPLES_PER_INDUSTRY} \
  --remove-all-common \
  --candidate-mult ${CANDIDATE_MULT} \
  --col-order ${COL_ORDER} \
  --n-clusters ${N_CLUSTERS} \
  --font-size ${FONT_SIZE} \
  --cmap winter

# ============================================================
# [2/2] Union mean activation heatmap (quantile-thresholded)
# ============================================================
section "[2/2] Union mean activation heatmap (quantile threshold)"
ts "Running vis_union_mean_activation_heatmap_threshold.py"

$PYTHON_BIN ../sparse_auto_encoder/industry/vis_union_mean_activation_heatmap_threshold.py \
  --features-parquet "${FEATURES_PARQUET}" \
  --out-dir "${OUT_DIR}" \
  --layers ${LAYERS} \
  --topk ${TOPK} \
  --samples-per-industry ${SAMPLES_PER_INDUSTRY} \
  --remove-all-common \
  --candidate-mult ${CANDIDATE_MULT} \
  --col-order ${COL_ORDER} \
  --n-clusters ${N_CLUSTERS} \
  --quantile 0.75

# ============================================================
# DONE
# ============================================================
section "DONE"
ts "Union mean activation heatmap pipeline completed successfully"