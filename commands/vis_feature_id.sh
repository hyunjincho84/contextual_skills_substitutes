#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ============================================================
# UNION MEAN ACTIVATION HEATMAP PIPELINE
#
# Purpose:
#  - Visualize important SAE feature IDs across industries using union-mean activation heatmaps.
#
# Runs:
#  1) sparse_auto_encoder/industry/vis_union_mean_activation_heatmap.py
#     - Builds a raw union-mean activation heatmap.
#  2) sparse_auto_encoder/industry/vis_union_mean_activation_heatmap_threshold.py
#     - Builds a quantile-thresholded union-mean activation heatmap.
#
# Input:
#  - FEATURES_PARQUET: ${BASE_DATA_DIR}/sparse_auto_encoder/python_industry/features.parquet
#  - LAYERS: 1-12
#  - TOPK: 64
#  - SAMPLES_PER_INDUSTRY: 250
#
# Output:
#  - OUT_DIR: ${BASE_DATA_DIR}/sparse_auto_encoder/python_industry/feature_id_vis
#  - Raw and thresholded heatmap PNG/CSV outputs from the visualization scripts.
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

PYTHON_BIN="${PYTHON_BIN:-python3}"

# ---------- paths ----------
BASE_DATA_DIR="${BASE_DATA_DIR:-/home/jovyan/LEM_data2/data}"
FEATURES_PARQUET="${FEATURES_PARQUET:-${BASE_DATA_DIR}/sparse_auto_encoder/python_industry/features.parquet}"
OUT_DIR="${OUT_DIR:-${BASE_DATA_DIR}/sparse_auto_encoder/python_industry/feature_id_vis}"

# ============================================================
# EDIT PARAMETERS HERE
# ============================================================
LAYERS="1-12"
TOPK=64
SAMPLES_PER_INDUSTRY=250
SEED=42
REMOVE_ALL_COMMON=true
CANDIDATE_MULT=10
MAX_ROUNDS=10000
COL_ORDER="cluster"
N_CLUSTERS=12

RAW_FONT_SIZE=16
RAW_CMAP="winter"
THRESHOLD_QUANTILE="0.75"
THRESHOLD_FONT_SIZE=16
FIGURE_WIDTH="18.0"
FIGURE_HEIGHT="3.2"
PLOT_DPI=300
# ============================================================

if [[ "${REMOVE_ALL_COMMON}" != "true" && "${REMOVE_ALL_COMMON}" != "false" ]]; then
  echo "REMOVE_ALL_COMMON must be true or false." >&2
  exit 2
fi

COMMON_FLAG_ARGS=()
if [[ "${REMOVE_ALL_COMMON}" == "true" ]]; then
  COMMON_FLAG_ARGS+=(--remove-all-common)
fi

# ============================================================
# [1/2] Union mean activation heatmap (raw)
# ============================================================
section "[1/2] Union mean activation heatmap (raw)"
ts "Running vis_union_mean_activation_heatmap.py"

"${PYTHON_BIN}" "${REPO_ROOT}/sparse_auto_encoder/industry/vis_union_mean_activation_heatmap.py" \
  --features-parquet "${FEATURES_PARQUET}" \
  --out-dir "${OUT_DIR}" \
  --layers "${LAYERS}" \
  --topk "${TOPK}" \
  --samples-per-industry "${SAMPLES_PER_INDUSTRY}" \
  --seed "${SEED}" \
  "${COMMON_FLAG_ARGS[@]}" \
  --candidate-mult "${CANDIDATE_MULT}" \
  --max-rounds "${MAX_ROUNDS}" \
  --col-order "${COL_ORDER}" \
  --n-clusters "${N_CLUSTERS}" \
  --font-size "${RAW_FONT_SIZE}" \
  --cmap "${RAW_CMAP}" \
  --figure-width "${FIGURE_WIDTH}" \
  --figure-height "${FIGURE_HEIGHT}" \
  --dpi "${PLOT_DPI}"

# ============================================================
# [2/2] Union mean activation heatmap (quantile-thresholded)
# ============================================================
section "[2/2] Union mean activation heatmap (quantile threshold)"
ts "Running vis_union_mean_activation_heatmap_threshold.py"

"${PYTHON_BIN}" "${REPO_ROOT}/sparse_auto_encoder/industry/vis_union_mean_activation_heatmap_threshold.py" \
  --features-parquet "${FEATURES_PARQUET}" \
  --out-dir "${OUT_DIR}" \
  --layers "${LAYERS}" \
  --topk "${TOPK}" \
  --samples-per-industry "${SAMPLES_PER_INDUSTRY}" \
  --seed "${SEED}" \
  "${COMMON_FLAG_ARGS[@]}" \
  --candidate-mult "${CANDIDATE_MULT}" \
  --max-rounds "${MAX_ROUNDS}" \
  --col-order "${COL_ORDER}" \
  --n-clusters "${N_CLUSTERS}" \
  --quantile "${THRESHOLD_QUANTILE}" \
  --font-size "${THRESHOLD_FONT_SIZE}" \
  --figure-width "${FIGURE_WIDTH}" \
  --figure-height "${FIGURE_HEIGHT}" \
  --dpi "${PLOT_DPI}"

# ============================================================
# DONE
# ============================================================
section "DONE"
ts "Union mean activation heatmap pipeline completed successfully"