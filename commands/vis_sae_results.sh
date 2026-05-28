#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ============================================================
# SPARSE AUTOENCODER VISUALIZATION PIPELINE
#
# Purpose:
#  - Build SAE-based UMAP and similarity visualizations by industry and year.
#
# Runs:
#  1) sparse_auto_encoder/industry/umap_cluster_sim_industry.py
#     - Extracts SAE representations and computes industry UMAP/similarity data.
#  2) sparse_auto_encoder/vis_umap.py
#     - Replots industry UMAP and legend.
#  3) sparse_auto_encoder/vis_cluster_sim_graph.py
#     - Plots industry similarity curves.
#  4) sparse_auto_encoder/yearly/umap_cluster_sim_yearly.py
#     - Extracts SAE representations and computes yearly UMAP/similarity data.
#  5) sparse_auto_encoder/vis_umap.py
#     - Replots yearly UMAP and colorbar.
#  6) sparse_auto_encoder/vis_cluster_sim_graph.py
#     - Plots yearly similarity curves.
#
# Input:
#  - TEST_PATTERN:      ${BASE_DATA_DIR}/preprocessed_www_new/test/20*/preprocessed_*.csv.gz
#  - YEAR_TEST_PATTERN: ${BASE_DATA_DIR}/preprocessed_www_new/test/20*/preprocessed_*.csv.gz
#  - MODEL_NAME:        ${BASE_DATA_DIR}/bert_pretrained
#  - VOCAB_PATH:        ${BASE_DATA_DIR}/preprocessed_www_new/skill2idx.json
#  - BEST_MODEL_PT:     ${BASE_DATA_DIR}/checkpoints(www)_new/best_model.pt
#  - SAE_ROOT:          ${BASE_DATA_DIR}/sae_layerwise_out_8192
#
# Output:
#  - Industry outputs: ${BASE_DATA_DIR}/sparse_auto_encoder/python_industry
#  - Yearly outputs:   ${BASE_DATA_DIR}/sparse_auto_encoder/python_yearly
#  - UMAP CSV/PNGs, legends/colorbars, feature parquet files, and similarity plots.
# ============================================================

BLUE="\033[1;34m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
RESET="\033[0m"

section () {
  echo -e "\n${BLUE}══════════════════════════════════════════════════════════════${RESET}"
  echo -e "${BLUE}▶ $1${RESET}"
  echo -e "${BLUE}══════════════════════════════════════════════════════════════${RESET}\n"
}

step () { echo -e "${GREEN}➤ $1${RESET}"; }

ts () { echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${RESET} $1"; }

PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_DATA_DIR="${BASE_DATA_DIR:-/home/jovyan/LEM_data2/data}"
export BASE_DATA_DIR
cd "${REPO_ROOT}"

OUT_DIR="${OUT_DIR:-${BASE_DATA_DIR}/sparse_auto_encoder/python_industry}"
TEST_PATTERN="${TEST_PATTERN:-${BASE_DATA_DIR}/preprocessed_www_new/test/20*/preprocessed_*.csv.gz}"
MODEL_NAME="${MODEL_NAME:-${BASE_DATA_DIR}/bert_pretrained}"
VOCAB_PATH="${VOCAB_PATH:-${BASE_DATA_DIR}/preprocessed_www_new/skill2idx.json}"
BEST_MODEL_PT="${BEST_MODEL_PT:-${BASE_DATA_DIR}/checkpoints(www)_new/best_model.pt}"
SAE_ROOT="${SAE_ROOT:-${BASE_DATA_DIR}/sae_layerwise_out_8192}"
TARGET_SKILL="${TARGET_SKILL:-python}"
FIELD_COL="soc_2_name"
FIELD_VALUES=(
  "Computer and Mathematical Occupations"
  "Business and Financial Operations Occupations"
  "Management Occupations"
  "Sales and Related Occupations"
  "Educational Instruction and Library Occupations"
)

YEAR_OUT_DIR="${YEAR_OUT_DIR:-${BASE_DATA_DIR}/sparse_auto_encoder/python_yearly}"
YEAR_TEST_PATTERN="${YEAR_TEST_PATTERN:-${BASE_DATA_DIR}/preprocessed_www_new/test/20*/preprocessed_*.csv.gz}"
YEAR_PER_GROUP=500
YEAR_CMAP="viridis"

section "SPARSE AUTOENCODER PIPELINE"
ts "Target skill = ${TARGET_SKILL}"

section "1/2 Industry SOC grouping"
step "Run umap_cluster_sim_industry.py"
"${PYTHON_BIN}" "${REPO_ROOT}/sparse_auto_encoder/industry/umap_cluster_sim_industry.py" \
  --repr sae \
  --test-pattern "${TEST_PATTERN}" \
  --target-skill "${TARGET_SKILL}" \
  --field-col "${FIELD_COL}" \
  --field-values "${FIELD_VALUES[@]}" \
  --per-field 250 \
  --model-name "${MODEL_NAME}" \
  --vocab-path "${VOCAB_PATH}" \
  --best-model-pt "${BEST_MODEL_PT}" \
  --sae-root "${SAE_ROOT}" \
  --use-amp --amp-dtype bf16 \
  --out-dir "${OUT_DIR}"

step "Replot UMAP by field"
"${PYTHON_BIN}" "${REPO_ROOT}/sparse_auto_encoder/vis_umap.py" \
  --umap-csv "${OUT_DIR}/umap_2d.csv" \
  --out-png  "${OUT_DIR}/umap_layer_grid_replot.png" \
  --legend-out-png "${OUT_DIR}/umap_legend.png" \
  --group-by field \
  --point-size 3 \
  --legend-fontsize 26

step "Plot similarity curve"
"${PYTHON_BIN}" "${REPO_ROOT}/sparse_auto_encoder/vis_cluster_sim_graph.py" \
  --in-dir "${OUT_DIR}" \
  --font-size 18 \
  --legend-out-png "${OUT_DIR}/element_centric_similarity_legend.png" \
  --legend-fontsize 28 \
  --legend-markersize 16 \
  --legend-frameon

section "2/2 Yearly grouping"
step "Run umap_cluster_sim_yearly.py"
"${PYTHON_BIN}" "${REPO_ROOT}/sparse_auto_encoder/yearly/umap_cluster_sim_yearly.py" \
  --group-by year \
  --test-pattern "${YEAR_TEST_PATTERN}" \
  --target-skill "${TARGET_SKILL}" \
  --per-group "${YEAR_PER_GROUP}" \
  --repr sae \
  --sae-root "${SAE_ROOT}" \
  --model-name "${MODEL_NAME}" \
  --vocab-path "${VOCAB_PATH}" \
  --best-model-pt "${BEST_MODEL_PT}" \
  --use-amp --amp-dtype bf16 \
  --out-dir "${YEAR_OUT_DIR}"

step "Replot UMAP by year + colorbar"
"${PYTHON_BIN}" "${REPO_ROOT}/sparse_auto_encoder/vis_umap.py" \
  --umap-csv "${YEAR_OUT_DIR}/umap_2d.csv" \
  --out-png  "${YEAR_OUT_DIR}/umap_layer_grid_year_conti.png" \
  --cbar-out-png "${YEAR_OUT_DIR}/umap_year_colorbar.png" \
  --group-by year \
  --point-size 3.0 \
  --cmap "${YEAR_CMAP}"

step "Plot similarity curve yearly"
"${PYTHON_BIN}" "${REPO_ROOT}/sparse_auto_encoder/vis_cluster_sim_graph.py" \
  --in-dir "${YEAR_OUT_DIR}" \
  --font-size 18 \
  --legend-out-png "${YEAR_OUT_DIR}/element_centric_similarity_legend.png" \
  --legend-fontsize 28 \
  --legend-markersize 16 \
  --legend-frameon

section "DONE"
ts "All steps completed successfully"