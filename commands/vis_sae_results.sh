#!/usr/bin/env bash
set -e

# ============================================================
# Sparse Autoencoder + UMAP + Clustering Similarity Pipeline
#  1) Industry/SOC grouping (categorical UMAP + similarity curve)
#  2) Yearly grouping (continuous-year UMAP + similarity curve)
# ============================================================

# ---------- helpers ----------
BLUE="\033[1;34m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
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

# ---------- config ----------
PYTHON_BIN="python3"

# -----------------------------
# Config (FIELD/SOC grouping)
# -----------------------------
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

# -----------------------------
# Config (YEAR grouping)
# -----------------------------
YEAR_OUT_DIR="../sparse_auto_encoder/python_yearly"
YEAR_TEST_PATTERN="/home/jovyan/LEM_data2/hyunjincho/preprocessed_www_new/test/20*/preprocessed_*soc.csv.gz"
YEAR_PER_GROUP=500
YEAR_CMAP="viridis"

# ---------- run ----------
section "SPARSE AUTOENCODER PIPELINE"
ts "Target skill = ${TARGET_SKILL}"

# ============================================================
# [1/2] FIELD/SOC pipeline
# ============================================================
section "[1/2] Industry/SOC grouping (categorical UMAP + similarity)"

step "Running umap_cluster_sim_industry.py (extract UMAP + similarity CSVs)"
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
ts "Saved outputs -> ${OUT_DIR}"

step "Re-plot UMAP by field (grid + separate legend)"
$PYTHON_BIN ../sparse_auto_encoder/vis_umap.py \
  --umap-csv "${OUT_DIR}/umap_2d.csv" \
  --out-png  "${OUT_DIR}/umap_layer_grid_replot.png" \
  --legend-out-png "${OUT_DIR}/umap_legend.png" \
  --group-by field \
  --point-size 3 \
  --legend-fontsize 26
ts "Saved UMAP grid + legend -> ${OUT_DIR}"

step "Plot element-centric similarity curve (k-means vs random-null)"
$PYTHON_BIN ../sparse_auto_encoder/vis_cluster_sim_graph.py \
  --in-dir "${OUT_DIR}" \
  --font-size 18 \
  --legend-out-png "${OUT_DIR}/element_centric_similarity_legend.png" \
  --legend-fontsize 28 \
  --legend-markersize 16 \
  --legend-frameon
ts "Saved similarity plot + legend -> ${OUT_DIR}"

section "FIELD/SOC pipeline DONE"
ts "Completed Industry/SOC grouping"

# ============================================================
# [2/2] YEARLY pipeline
# ============================================================
section "[2/2] Yearly grouping (continuous-year UMAP + similarity)"

step "Running vis_umap_cluster_sim_yearly.py (extract UMAP + similarity CSVs by year)"
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
ts "Saved outputs -> ${YEAR_OUT_DIR}"

step "Re-plot UMAP by year (continuous colormap) + separate colorbar"
$PYTHON_BIN ../sparse_auto_encoder/vis_umap.py \
  --umap-csv "${YEAR_OUT_DIR}/umap_2d.csv" \
  --out-png  "${YEAR_OUT_DIR}/umap_layer_grid_year_conti.png" \
  --cbar-out-png "${YEAR_OUT_DIR}/umap_year_colorbar.png" \
  --group-by year \
  --point-size 3.0 \
  --cmap "${YEAR_CMAP}"
ts "Saved UMAP grid + colorbar -> ${YEAR_OUT_DIR}"

step "Plot element-centric similarity curve (yearly) + separate legend"
$PYTHON_BIN ../sparse_auto_encoder/vis_cluster_sim_graph.py \
  --in-dir "${YEAR_OUT_DIR}" \
  --font-size 18 \
  --legend-out-png "${YEAR_OUT_DIR}/element_centric_similarity_legend.png" \
  --legend-fontsize 28 \
  --legend-markersize 16 \
  --legend-frameon
ts "Saved similarity plot + legend -> ${YEAR_OUT_DIR}"

section "DONE"
ts "All steps (industry/SOC + yearly) completed successfully"