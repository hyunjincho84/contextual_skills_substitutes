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
#  7) sparse_auto_encoder/yearly/overlap_by_year.py
#     - Extracts/reuses yearly SAE features, computes Top-K overlap, and plots
#       the overlap heatmap.
#
# Input:
#  - TEST_PATTERN:      ${BASE_DATA_DIR}/all_train_test/test/20*/20*/preprocessed_*.csv.gz
#  - YEAR_TEST_PATTERN: ${BASE_DATA_DIR}/all_train_test/test/20*/20*/preprocessed_*.csv.gz
#  - MODEL_NAME:        ${BASE_DATA_DIR}/bert_pretrained
#  - VOCAB_PATH:        ${BASE_DATA_DIR}/all_filterd/skill2idx.json
#  - BEST_MODEL_PT:     ${BASE_DATA_DIR}/models/bert_fine_tuned_model/best_model.pt
#  - SAE_ROOT:          ${BASE_DATA_DIR}/models/sae_layerwise_out
#
# Output:
#  - Industry outputs: ${BASE_DATA_DIR}/sparse_auto_encoder/python_industry
#  - Yearly outputs:   ${BASE_DATA_DIR}/sparse_auto_encoder/python_yearly
#  - UMAP CSV/PNGs, legends/colorbars, feature parquet files, similarity plots,
#    and yearly overlap heatmaps.
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

# ============================================================
# EDIT PARAMETERS HERE
# ============================================================
PYTHON_BIN="/home/jovyan/LEM_data2/hyunjincho/env/lem/bin/python"
BASE_DATA_DIR="/home/jovyan/LEM_data3/hyunjincho/LEM"
RUN_MODE="soc"  # all | soc | year; CLI flags below override this value.

MODEL_NAME="${BASE_DATA_DIR}/bert_pretrained"
VOCAB_PATH="${BASE_DATA_DIR}/all_filterd/skill2idx.json"
BEST_MODEL_PT="${BASE_DATA_DIR}/models/bert_fine_tuned_model/best_model.pt"
SAE_ROOT="${BASE_DATA_DIR}/models/sae_layerwise_out"
TARGET_SKILLS=(
  "python"
  "c"
  "javascript"
  "java"
)
TRUTH_COL="true_skill"
TEXT_COL="masked_sentence"

REPRESENTATION="sae"
D_HIDDEN=4096
LAYERS="1-12"
DEVICE="cuda"
BATCH_SIZE=256
MAX_LEN=256
USE_AMP=true
AMP_DTYPE="bf16"
SEED=42
EC_ALPHA="0.90"
UMAP_N_NEIGHBORS=30
UMAP_MIN_DIST="0.10"
UMAP_METRIC="cosine"

# Industry/SOC extraction
OUTPUT_ROOT="${BASE_DATA_DIR}/sparse_auto_encoder"
TEST_PATTERN="${BASE_DATA_DIR}/all_train_test/test_clean/20*/20*/preprocessed_*.csv.gz"
FIELD_COL="soc_2_name"
FIELD_VALUES=(
  "Computer and Mathematical Occupations"
  "Business and Financial Operations Occupations"
  "Management Occupations"
  "Sales and Related Occupations"
  "Educational Instruction and Library Occupations"
)
# SOC sampling sizes by target skill.
# *_PER_FIELD controls the UMAP sample cap per SOC group.
# *_SOC_SIM_PER_GROUP is the required/sample size per SOC group for similarity.
PYTHON_PER_FIELD=200
PYTHON_SOC_SIM_PER_GROUP=500000
C_PER_FIELD=200
C_SOC_SIM_PER_GROUP=200
JAVASCRIPT_PER_FIELD=200
JAVASCRIPT_SOC_SIM_PER_GROUP=500
JAVA_PER_FIELD=200
JAVA_SOC_SIM_PER_GROUP=400
INDUSTRY_KMEANS_K=5

# Industry UMAP figure
INDUSTRY_UMAP_GROUP_BY="field"
INDUSTRY_POINT_SIZE=3
INDUSTRY_POINT_ALPHA="0.75"
INDUSTRY_CROP_Q="0.98"
INDUSTRY_LEGEND_FONTSIZE=26

# Industry similarity figure
INDUSTRY_SIM_FONT_SIZE=18
INDUSTRY_SIM_LEGEND_FONTSIZE=28
INDUSTRY_SIM_LEGEND_MARKERSIZE=16
INDUSTRY_SIM_LEGEND_FRAMEON=true

# Year extraction; leave YEAR_VALUES empty to infer years from file paths.
YEAR_GROUP_BY="year"
YEAR_TEST_PATTERN="${BASE_DATA_DIR}/all_train_test/test_clean/20*/20*/preprocessed_*.csv.gz"
YEAR_VALUES=()
YEAR_PER_GROUP=500
YEAR_SIM_PER_GROUP=800
YEAR_KMEANS_K=""

# Year UMAP figure
YEAR_UMAP_GROUP_BY="year"
YEAR_POINT_SIZE="3.0"
YEAR_POINT_ALPHA="0.75"
YEAR_CROP_Q="0.98"
YEAR_CMAP="viridis"
YEAR_CBAR_X0="0.92"
YEAR_CBAR_W="0.02"
YEAR_CBAR_Y0="0.18"
YEAR_CBAR_H="0.64"
YEAR_LEGEND_FONTSIZE=22

# Year similarity figure
YEAR_SIM_FONT_SIZE=18
YEAR_SIM_LEGEND_FONTSIZE=28
YEAR_SIM_LEGEND_MARKERSIZE=16
YEAR_SIM_LEGEND_FRAMEON=true

# Yearly Top-K SAE feature overlap
YEAR_OVERLAP_PREDICTIONS_ROOT="${BASE_DATA_DIR}/predictions_clean/bert_pred/pred"
YEAR_OVERLAP_STAGE="all"  # features | overlap | visualize | all
YEAR_OVERLAP_YEARS="2010-2025"
YEAR_OVERLAP_DATA_MODE="all"  # all | sample
YEAR_OVERLAP_SAMPLES_PER_YEAR=1000
YEAR_OVERLAP_READ_CHUNKSIZE=100000
YEAR_OVERLAP_MAX_LEN=512
YEAR_OVERLAP_LAYERS="1,6,12"
YEAR_OVERLAP_TOPK=128
YEAR_OVERLAP_MAX_ITERS=100
YEAR_OVERLAP_AGGREGATE_BATCH_SIZE=4096
YEAR_OVERLAP_PARQUET_COMPRESSION="zstd"

# Yearly overlap heatmap
YEAR_OVERLAP_DPI=300
YEAR_OVERLAP_GRID_COLS=3
YEAR_OVERLAP_PANEL_WIDTH="6.5"
YEAR_OVERLAP_PANEL_HEIGHT="6.0"
YEAR_OVERLAP_ANNOTATION_FONT_SIZE="8"
YEAR_OVERLAP_TICK_FONT_SIZE="11"
YEAR_OVERLAP_TITLE_FONT_SIZE="20"
YEAR_OVERLAP_X_LABEL_ROTATION="45"
YEAR_OVERLAP_GRID_OUT_NAME="overlap_heatmap_large.png"
# ============================================================

cd "${REPO_ROOT}"

while (( $# > 0 )); do
  case "$1" in
    --soc-only)
      RUN_MODE="soc"
      ;;
    --year-only)
      RUN_MODE="year"
      ;;
    --all)
      RUN_MODE="all"
      ;;
    -h|--help)
      echo "Usage: $0 [--soc-only | --year-only | --all]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--soc-only | --year-only | --all]" >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "${RUN_MODE}" != "all" && "${RUN_MODE}" != "soc" && "${RUN_MODE}" != "year" ]]; then
  echo "RUN_MODE must be one of: all, soc, year." >&2
  exit 2
fi

if [[ "${YEAR_OVERLAP_DATA_MODE}" != "all" && "${YEAR_OVERLAP_DATA_MODE}" != "sample" ]]; then
  echo "YEAR_OVERLAP_DATA_MODE must be one of: all, sample." >&2
  exit 2
fi

if [[ "${YEAR_OVERLAP_STAGE}" != "features" && \
      "${YEAR_OVERLAP_STAGE}" != "overlap" && \
      "${YEAR_OVERLAP_STAGE}" != "visualize" && \
      "${YEAR_OVERLAP_STAGE}" != "all" ]]; then
  echo "YEAR_OVERLAP_STAGE must be one of: features, overlap, visualize, all." >&2
  exit 2
fi

for bool_name in USE_AMP INDUSTRY_SIM_LEGEND_FRAMEON YEAR_SIM_LEGEND_FRAMEON; do
  bool_value="${!bool_name}"
  if [[ "${bool_value}" != "true" && "${bool_value}" != "false" ]]; then
    echo "${bool_name} must be true or false." >&2
    exit 2
  fi
done

AMP_ARGS=()
if [[ "${USE_AMP}" == "true" ]]; then
  AMP_ARGS+=(--use-amp)
fi

INDUSTRY_KMEANS_ARGS=()
if [[ -n "${INDUSTRY_KMEANS_K}" ]]; then
  INDUSTRY_KMEANS_ARGS+=(--kmeans-k "${INDUSTRY_KMEANS_K}")
fi

INDUSTRY_SIM_FLAG_ARGS=()
if [[ "${INDUSTRY_SIM_LEGEND_FRAMEON}" == "true" ]]; then
  INDUSTRY_SIM_FLAG_ARGS+=(--legend-frameon)
fi

YEAR_VALUES_ARGS=()
if (( ${#YEAR_VALUES[@]} > 0 )); then
  YEAR_VALUES_ARGS+=(--years "${YEAR_VALUES[@]}")
fi

YEAR_KMEANS_ARGS=()
if [[ -n "${YEAR_KMEANS_K}" ]]; then
  YEAR_KMEANS_ARGS+=(--kmeans-k "${YEAR_KMEANS_K}")
fi

YEAR_SIM_FLAG_ARGS=()
if [[ "${YEAR_SIM_LEGEND_FRAMEON}" == "true" ]]; then
  YEAR_SIM_FLAG_ARGS+=(--legend-frameon)
fi

if (( ${#TARGET_SKILLS[@]} == 0 )); then
  echo "TARGET_SKILLS must contain at least one skill." >&2
  exit 2
fi

section "SPARSE AUTOENCODER PIPELINE"
ts "Target skills = ${TARGET_SKILLS[*]}"
ts "Run mode = ${RUN_MODE}"

for TARGET_SKILL in "${TARGET_SKILLS[@]}"; do
case "${TARGET_SKILL}" in
  python)
    PER_FIELD="${PYTHON_PER_FIELD}"
    SOC_SIM_PER_GROUP="${PYTHON_SOC_SIM_PER_GROUP}"
    ;;
  c)
    PER_FIELD="${C_PER_FIELD}"
    SOC_SIM_PER_GROUP="${C_SOC_SIM_PER_GROUP}"
    ;;
  javascript)
    PER_FIELD="${JAVASCRIPT_PER_FIELD}"
    SOC_SIM_PER_GROUP="${JAVASCRIPT_SOC_SIM_PER_GROUP}"
    ;;
  java)
    PER_FIELD="${JAVA_PER_FIELD}"
    SOC_SIM_PER_GROUP="${JAVA_SOC_SIM_PER_GROUP}"
    ;;
  *)
    echo "No SOC sampling configuration for TARGET_SKILL=${TARGET_SKILL}" >&2
    exit 2
    ;;
esac

for soc_sample_name in PER_FIELD SOC_SIM_PER_GROUP; do
  soc_sample_value="${!soc_sample_name}"
  if ! [[ "${soc_sample_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${soc_sample_name} for ${TARGET_SKILL} must be a positive integer." >&2
    exit 2
  fi
done

SKILL_TAG="${TARGET_SKILL// /_}"
OUT_DIR="${OUTPUT_ROOT}/${SKILL_TAG}_industry"
INDUSTRY_UMAP_CSV="${OUT_DIR}/umap_2d.csv"
INDUSTRY_UMAP_PNG="${OUT_DIR}/umap_layer_grid_replot.png"
INDUSTRY_LEGEND_PNG="${OUT_DIR}/umap_legend.png"
INDUSTRY_SIM_LEGEND_PNG="${OUT_DIR}/element_centric_similarity_legend.png"

YEAR_OUT_DIR="${OUTPUT_ROOT}/${SKILL_TAG}_yearly"
YEAR_UMAP_CSV="${YEAR_OUT_DIR}/umap_2d.csv"
YEAR_UMAP_PNG="${YEAR_OUT_DIR}/umap_layer_grid_year_conti.png"
YEAR_COLORBAR_PNG="${YEAR_OUT_DIR}/umap_year_colorbar.png"
YEAR_SIM_LEGEND_PNG="${YEAR_OUT_DIR}/element_centric_similarity_legend.png"
YEAR_OVERLAP_OUT_DIR="${YEAR_OUT_DIR}/overlaps"

section "TARGET SKILL: ${TARGET_SKILL}"
ts "SOC UMAP per field = ${PER_FIELD}; SOC similarity per field = ${SOC_SIM_PER_GROUP}"

if [[ "${RUN_MODE}" == "all" || "${RUN_MODE}" == "soc" ]]; then
section "1/2 Industry SOC grouping"
step "Run umap_cluster_sim_industry.py"
"${PYTHON_BIN}" "${REPO_ROOT}/sparse_auto_encoder/industry/umap_cluster_sim_industry.py" \
  --test-pattern "${TEST_PATTERN}" \
  --truth-col "${TRUTH_COL}" \
  --text-col "${TEXT_COL}" \
  --target-skill "${TARGET_SKILL}" \
  --field-col "${FIELD_COL}" \
  --field-values "${FIELD_VALUES[@]}" \
  --per-field "${PER_FIELD}" \
  --similarity-per-field "${SOC_SIM_PER_GROUP}" \
  --seed "${SEED}" \
  --model-name "${MODEL_NAME}" \
  --vocab-path "${VOCAB_PATH}" \
  --best-model-pt "${BEST_MODEL_PT}" \
  --repr "${REPRESENTATION}" \
  --sae-root "${SAE_ROOT}" \
  --d-hidden "${D_HIDDEN}" \
  --layers "${LAYERS}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --max-len "${MAX_LEN}" \
  "${AMP_ARGS[@]}" \
  --amp-dtype "${AMP_DTYPE}" \
  "${INDUSTRY_KMEANS_ARGS[@]}" \
  --ec-alpha "${EC_ALPHA}" \
  --umap-n-neighbors "${UMAP_N_NEIGHBORS}" \
  --umap-min-dist "${UMAP_MIN_DIST}" \
  --umap-metric "${UMAP_METRIC}" \
  --out-dir "${OUT_DIR}"

step "Replot UMAP by field"
"${PYTHON_BIN}" "${REPO_ROOT}/sparse_auto_encoder/vis_umap.py" \
  --umap-csv "${INDUSTRY_UMAP_CSV}" \
  --out-png "${INDUSTRY_UMAP_PNG}" \
  --legend-out-png "${INDUSTRY_LEGEND_PNG}" \
  --group-by "${INDUSTRY_UMAP_GROUP_BY}" \
  --legend-fontsize "${INDUSTRY_LEGEND_FONTSIZE}" \
  --point-size "${INDUSTRY_POINT_SIZE}" \
  --point-alpha "${INDUSTRY_POINT_ALPHA}" \
  --crop-q "${INDUSTRY_CROP_Q}"

step "Plot similarity curve"
"${PYTHON_BIN}" "${REPO_ROOT}/sparse_auto_encoder/vis_cluster_sim_graph.py" \
  --in-dir "${OUT_DIR}" \
  --font-size "${INDUSTRY_SIM_FONT_SIZE}" \
  --legend-out-png "${INDUSTRY_SIM_LEGEND_PNG}" \
  --legend-fontsize "${INDUSTRY_SIM_LEGEND_FONTSIZE}" \
  --legend-markersize "${INDUSTRY_SIM_LEGEND_MARKERSIZE}" \
  "${INDUSTRY_SIM_FLAG_ARGS[@]}"
fi

if [[ "${RUN_MODE}" == "all" || "${RUN_MODE}" == "year" ]]; then
section "2/2 Yearly grouping"
step "Run umap_cluster_sim_yearly.py"
"${PYTHON_BIN}" "${REPO_ROOT}/sparse_auto_encoder/yearly/umap_cluster_sim_yearly.py" \
  --test-pattern "${YEAR_TEST_PATTERN}" \
  --truth-col "${TRUTH_COL}" \
  --text-col "${TEXT_COL}" \
  --target-skill "${TARGET_SKILL}" \
  --group-by "${YEAR_GROUP_BY}" \
  "${YEAR_VALUES_ARGS[@]}" \
  --per-group "${YEAR_PER_GROUP}" \
  --similarity-per-group "${YEAR_SIM_PER_GROUP}" \
  --seed "${SEED}" \
  --model-name "${MODEL_NAME}" \
  --vocab-path "${VOCAB_PATH}" \
  --best-model-pt "${BEST_MODEL_PT}" \
  --repr "${REPRESENTATION}" \
  --sae-root "${SAE_ROOT}" \
  --d-hidden "${D_HIDDEN}" \
  --layers "${LAYERS}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --max-len "${MAX_LEN}" \
  "${AMP_ARGS[@]}" \
  --amp-dtype "${AMP_DTYPE}" \
  --umap-n-neighbors "${UMAP_N_NEIGHBORS}" \
  --umap-min-dist "${UMAP_MIN_DIST}" \
  --umap-metric "${UMAP_METRIC}" \
  "${YEAR_KMEANS_ARGS[@]}" \
  --ec-alpha "${EC_ALPHA}" \
  --out-dir "${YEAR_OUT_DIR}"

step "Replot UMAP by year + colorbar"
"${PYTHON_BIN}" "${REPO_ROOT}/sparse_auto_encoder/vis_umap.py" \
  --umap-csv "${YEAR_UMAP_CSV}" \
  --out-png "${YEAR_UMAP_PNG}" \
  --cbar-out-png "${YEAR_COLORBAR_PNG}" \
  --group-by "${YEAR_UMAP_GROUP_BY}" \
  --legend-fontsize "${YEAR_LEGEND_FONTSIZE}" \
  --point-size "${YEAR_POINT_SIZE}" \
  --point-alpha "${YEAR_POINT_ALPHA}" \
  --crop-q "${YEAR_CROP_Q}" \
  --cmap "${YEAR_CMAP}" \
  --cbar-x0 "${YEAR_CBAR_X0}" \
  --cbar-w "${YEAR_CBAR_W}" \
  --cbar-y0 "${YEAR_CBAR_Y0}" \
  --cbar-h "${YEAR_CBAR_H}"

step "Plot similarity curve yearly"
"${PYTHON_BIN}" "${REPO_ROOT}/sparse_auto_encoder/vis_cluster_sim_graph.py" \
  --in-dir "${YEAR_OUT_DIR}" \
  --font-size "${YEAR_SIM_FONT_SIZE}" \
  --legend-out-png "${YEAR_SIM_LEGEND_PNG}" \
  --legend-fontsize "${YEAR_SIM_LEGEND_FONTSIZE}" \
  --legend-markersize "${YEAR_SIM_LEGEND_MARKERSIZE}" \
  "${YEAR_SIM_FLAG_ARGS[@]}"

step "Extract/reuse yearly SAE features, compute Top-K overlap, and plot heatmap"
"${PYTHON_BIN}" "${REPO_ROOT}/sparse_auto_encoder/yearly/overlap_by_year.py" \
  --stage "${YEAR_OVERLAP_STAGE}" \
  --target-skill "${TARGET_SKILL}" \
  --predictions-root "${YEAR_OVERLAP_PREDICTIONS_ROOT}" \
  --truth-col "${TRUTH_COL}" \
  --text-col "${TEXT_COL}" \
  --row-id-col "row_idx" \
  --years "${YEAR_OVERLAP_YEARS}" \
  --data-mode "${YEAR_OVERLAP_DATA_MODE}" \
  --samples-per-year "${YEAR_OVERLAP_SAMPLES_PER_YEAR}" \
  --read-chunksize "${YEAR_OVERLAP_READ_CHUNKSIZE}" \
  --model-name "${MODEL_NAME}" \
  --vocab-path "${VOCAB_PATH}" \
  --best-model-pt "${BEST_MODEL_PT}" \
  --sae-root "${SAE_ROOT}" \
  --d-hidden "${D_HIDDEN}" \
  --layers "${YEAR_OVERLAP_LAYERS}" \
  --topk "${YEAR_OVERLAP_TOPK}" \
  --max-iters "${YEAR_OVERLAP_MAX_ITERS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --max-len "${YEAR_OVERLAP_MAX_LEN}" \
  "${AMP_ARGS[@]}" \
  --amp-dtype "${AMP_DTYPE}" \
  --parquet-compression "${YEAR_OVERLAP_PARQUET_COMPRESSION}" \
  --aggregate-batch-size "${YEAR_OVERLAP_AGGREGATE_BATCH_SIZE}" \
  --out-dir "${YEAR_OVERLAP_OUT_DIR}" \
  --dpi "${YEAR_OVERLAP_DPI}" \
  --grid-cols "${YEAR_OVERLAP_GRID_COLS}" \
  --panel-width "${YEAR_OVERLAP_PANEL_WIDTH}" \
  --panel-height "${YEAR_OVERLAP_PANEL_HEIGHT}" \
  --annotation-font-size "${YEAR_OVERLAP_ANNOTATION_FONT_SIZE}" \
  --tick-font-size "${YEAR_OVERLAP_TICK_FONT_SIZE}" \
  --title-font-size "${YEAR_OVERLAP_TITLE_FONT_SIZE}" \
  --x-label-rotation "${YEAR_OVERLAP_X_LABEL_ROTATION}" \
  --grid-out-name "${YEAR_OVERLAP_GRID_OUT_NAME}"
fi

section "DONE"
ts "Completed target skill: ${TARGET_SKILL}"
done

section "ALL DONE"
ts "All target skills completed successfully"
