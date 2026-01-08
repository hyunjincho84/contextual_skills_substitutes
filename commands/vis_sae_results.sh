#!/usr/bin/env bash
set -euo pipefail

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

PYTHON_BIN="python3"

OUT_DIR="../sparse_auto_encoder/python_industry"
TEST_PATTERN="/home/jovyan/LEM_data2/hyunjincho/preprocessed_www_new/test/20*/preprocessed_*soc.csv.gz"
MODEL_NAME="/home/jovyan/LEM_data2/hyunjincho/bert_pretrained/checkpoint-165687"
VOCAB_PATH="/home/jovyan/LEM_data2/hyunjincho/preprocessed_www_new/skill2idx.json"
BEST_MODEL_PT="/home/jovyan/LEM_data2/hyunjincho/checkpoints(www)_new/best_model.pt"
SAE_ROOT="/home/jovyan/LEM_data2/hyunjincho/sae_layerwise_out"
TARGET_SKILL="python"
FIELD_COL="soc_2_name"
FIELD_VALUES=(
  "Computer and Mathematical Occupations"
  "Business and Financial Operations Occupations"
  "Management Occupations"
  "Sales and Related Occupations"
  "Educational Instruction and Library Occupations"
)

YEAR_OUT_DIR="../sparse_auto_encoder/python_yearly"
YEAR_TEST_PATTERN="/home/jovyan/LEM_data2/hyunjincho/preprocessed_www_new/test/20*/preprocessed_*soc.csv.gz"
YEAR_PER_GROUP=500
YEAR_CMAP="viridis"

section "SPARSE AUTOENCODER PIPELINE"
ts "Target skill = ${TARGET_SKILL}"

section "1/2 Industry SOC grouping"
step "Run umap_cluster_sim_industry.py"
$PYTHON_BIN ../sparse_auto_encoder/industry/umap_cluster_sim_industry.py \
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
$PYTHON_BIN ../sparse_auto_encoder/vis_umap.py \
  --umap-csv "${OUT_DIR}/umap_2d.csv" \
  --out-png  "${OUT_DIR}/umap_layer_grid_replot.png" \
  --legend-out-png "${OUT_DIR}/umap_legend.png" \
  --group-by field \
  --point-size 3 \
  --legend-fontsize 26

step "Plot similarity curve"
$PYTHON_BIN ../sparse_auto_encoder/vis_cluster_sim_graph.py \
  --in-dir "${OUT_DIR}" \
  --font-size 18 \
  --legend-out-png "${OUT_DIR}/element_centric_similarity_legend.png" \
  --legend-fontsize 28 \
  --legend-markersize 16 \
  --legend-frameon

section "2/2 Yearly grouping"
step "Run umap_cluster_sim_yearly.py"
$PYTHON_BIN ../sparse_auto_encoder/yearly/umap_cluster_sim_yearly.py \
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
$PYTHON_BIN ../sparse_auto_encoder/vis_umap.py \
  --umap-csv "${YEAR_OUT_DIR}/umap_2d.csv" \
  --out-png  "${YEAR_OUT_DIR}/umap_layer_grid_year_conti.png" \
  --cbar-out-png "${YEAR_OUT_DIR}/umap_year_colorbar.png" \
  --group-by year \
  --point-size 3.0 \
  --cmap "${YEAR_CMAP}"

step "Plot similarity curve yearly"
$PYTHON_BIN ../sparse_auto_encoder/vis_cluster_sim_graph.py \
  --in-dir "${YEAR_OUT_DIR}" \
  --font-size 18 \
  --legend-out-png "${YEAR_OUT_DIR}/element_centric_similarity_legend.png" \
  --legend-fontsize 28 \
  --legend-markersize 16 \
  --legend-frameon

section "DONE"
ts "All steps completed successfully"