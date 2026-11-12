#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ============================================================
# OVERLAP ANALYSIS PIPELINE
#
# Purpose:
#  - Measure and visualize feature overlap across industries and years.
#
# Runs:
#  1) sparse_auto_encoder/industry/overlap_by_industry.py
#     - Builds industry overlap matrices, per-layer CSVs, and grid heatmap.
#  2) sparse_auto_encoder/industry/overlap_by_industry_graph.py
#     - Draws industry overlap graph from per-layer CSVs.
#  3) sparse_auto_encoder/yearly/overlap_by_year.py
#     - Builds yearly overlap matrices, per-layer CSVs, and grid heatmap.
#  4) sparse_auto_encoder/yearly/overlap_by_year_graph.py
#     - Draws year overlap graph from per-layer CSVs.
#  5) sparse_auto_encoder/yearly/overlap_by_diff_year_graph.py
#     - Draws year-over-year overlap difference graph.
#
# Input (generated from all_train_test/test_clean by vis_sae_results.sh):
#  - ${CLEAN_FEATURE_ROOT}/${skill}_industry/features.parquet
#  - ${CLEAN_FEATURE_ROOT}/${skill}_yearly/features.parquet
#  - TARGET_SKILLS and YEARS are edited as literal text in the config below.
#
# Output:
#  - Industry outputs: ${IND_DIR}/overlaps
#  - Yearly outputs:   ${YEAR_DIR}/overlaps
#  - Per-layer CSVs:   overlaps/csvs
#  - Heatmaps/graphs:  PNG files under each overlaps directory.
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

step () { echo -e "${GREEN}➤ $1${RESET}"; }
ts () { echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${RESET} $1"; }
fail () { echo -e "${RED}ERROR: $1${RESET}" >&2; exit 1; }

trap 'echo -e "${RED}✗ Failed at line $LINENO${RESET}"' ERR

# ============================================================
# INPUT PARAMETERS — edit these literal text values here
# ============================================================
PYTHON_BIN="/home/jovyan/LEM_data2/hyunjincho/env/lem/bin/python"
CLEAN_FEATURE_ROOT="/home/jovyan/LEM_data3/hyunjincho/LEM/sparse_auto_encoder"
TARGET_SKILLS=("python")
YEARS="2010-2025"

# ============================================================
# ANALYSIS PARAMETERS — edit values here if needed
# ============================================================

# Industry overlap matrix
IND_TOPK="128"
IND_LAYERS="1-12"
IND_SAMPLES_PER_INDUSTRY="250"
IND_FONT_SIZE="16"
IND_MAX_ITERS="20"
IND_SEED="42"
IND_DPI="300"
IND_GRID_COLS="4"
IND_GRID_OUT_NAME="heatmap_overlap_all_layers.png"
IND_CSV_SUBDIR="csvs"

# Industry overlap graph
IND_GRAPH_FONT_SIZE="14"
IND_NORMALIZE_BY_TOPK="0"
IND_GRAPH_ALPHA_LINES="0.25"
IND_GRAPH_LW_LINES="1.5"
IND_GRAPH_LW_MEAN="3.0"
IND_GRAPH_IQR_ALPHA="0.18"
IND_GRAPH_DPI="300"

# Year overlap matrix
YEAR_GROUP_BY="year"
YEAR_TOPK="128"
YEAR_LAYERS="1-12"
YEAR_SAMPLES_PER_GROUP="1000"
YEAR_MAX_GROUPS=""  # empty = no limit
YEAR_FONT_SIZE="16"
YEAR_MAX_ITERS="100"
YEAR_SEED="42"
YEAR_DPI="300"
YEAR_GRID_COLS="4"
YEAR_GRID_OUT_NAME="heatmap_overlap_all_layers.png"
YEAR_CSV_SUBDIR="csvs"

# Year overlap graph
YEAR_GRAPH_FONT_SIZE="14"
YEAR_GRAPH_ALPHA_LINES="0.20"
YEAR_GRAPH_LW_LINES="1.5"
YEAR_GRAPH_LW_MEAN="3.0"
YEAR_GRAPH_ORDER=""  # empty = infer from CSV
YEAR_GRAPH_LEGEND_MAX_PAIRS="12"
YEAR_GRAPH_IQR_ALPHA="0.18"
YEAR_GRAPH_DPI="300"

# Year-difference graph
DIFF_LAYERS="1,2,3,4,5,6,7,8,9,10,11,12"
DIFF_AGG="mean"
DIFF_ALPHA_LINES="0.25"
DIFF_LW_LINES="1.8"
DIFF_LW_GLOBAL="3.2"
DIFF_IQR_ALPHA="0.18"
DIFF_MIN_COUNT="1"
DIFF_DPI="300"

[[ -x "${PYTHON_BIN}" ]] || fail "Python executable not found: ${PYTHON_BIN}"
[[ -d "${CLEAN_FEATURE_ROOT}" ]] || fail "Clean feature root not found: ${CLEAN_FEATURE_ROOT}"
(( ${#TARGET_SKILLS[@]} > 0 )) || fail "TARGET_SKILLS must contain at least one skill."

section "CONFIG"
echo "PYTHON_BIN        = ${PYTHON_BIN}"
echo "CLEAN_FEATURE_ROOT = ${CLEAN_FEATURE_ROOT}"
echo "TARGET_SKILLS      = ${TARGET_SKILLS[*]}"
echo "YEARS              = ${YEARS}"

for TARGET_SKILL in "${TARGET_SKILLS[@]}"; do
  SKILL_TAG="${TARGET_SKILL,,}"
  SKILL_TAG="${SKILL_TAG// /_}"
  SKILL_TAG="${SKILL_TAG//\//_}"

  IND_DIR="${CLEAN_FEATURE_ROOT}/${SKILL_TAG}_industry"
  YEAR_DIR="${CLEAN_FEATURE_ROOT}/${SKILL_TAG}_yearly"
  IND_FEATURES_PARQUET="${IND_DIR}/features.parquet"
  YEAR_FEATURES_PARQUET="${YEAR_DIR}/features.parquet"

  IND_OVERLAP_DIR="${IND_DIR}/overlaps"
  YEAR_OVERLAP_DIR="${YEAR_DIR}/overlaps"
  IND_CSV_DIR="${IND_OVERLAP_DIR}/${IND_CSV_SUBDIR}"
  YEAR_CSV_DIR="${YEAR_OVERLAP_DIR}/${YEAR_CSV_SUBDIR}"
  DIFF_OUT_PNG="${YEAR_OVERLAP_DIR}/yearly_diff.png"

  [[ -f "${IND_FEATURES_PARQUET}" ]] || fail "Missing clean industry features: ${IND_FEATURES_PARQUET}"
  [[ -f "${YEAR_FEATURES_PARQUET}" ]] || fail "Missing clean yearly features: ${YEAR_FEATURES_PARQUET}"

  section "TARGET SKILL: ${TARGET_SKILL}"
  echo "IND_FEATURES_PARQUET  = ${IND_FEATURES_PARQUET}"
  echo "YEAR_FEATURES_PARQUET = ${YEAR_FEATURES_PARQUET}"
# ============================================================
# [1/5] Industry-wise overlap (matrix + single grid heatmap)
# ============================================================
section "[1/5] Industry-wise overlap"
ts "Running overlap_by_industry.py"

"${PYTHON_BIN}" "${REPO_ROOT}/sparse_auto_encoder/industry/overlap_by_industry.py" \
  --features-parquet "${IND_FEATURES_PARQUET}" \
  --out-dir "${IND_OVERLAP_DIR}" \
  --topk "${IND_TOPK}" \
  --layers "${IND_LAYERS}" \
  --samples-per-industry "${IND_SAMPLES_PER_INDUSTRY}" \
  --font-size "${IND_FONT_SIZE}" \
  --max-iters "${IND_MAX_ITERS}" \
  --seed "${IND_SEED}" \
  --dpi "${IND_DPI}" \
  --grid-cols "${IND_GRID_COLS}" \
  --grid-out-name "${IND_GRID_OUT_NAME}" \
  --csv-subdir "${IND_CSV_SUBDIR}"

ts "Saved industry overlap outputs -> ${IND_OVERLAP_DIR}"
ts "Per-layer CSVs -> ${IND_CSV_DIR}"

# ============================================================
# [2/5] Industry overlap graph
#   IMPORTANT: read per-layer CSVs from overlaps/csvs
# ============================================================
section "[2/5] Industry overlap graph"
ts "Running overlap_by_industry_graph.py"

"${PYTHON_BIN}" "${REPO_ROOT}/sparse_auto_encoder/industry/overlap_by_industry_graph.py" \
  --in-dir "${IND_CSV_DIR}" \
  --out-dir "${IND_OVERLAP_DIR}" \
  --font-size "${IND_GRAPH_FONT_SIZE}" \
  --normalize-by-topk "${IND_NORMALIZE_BY_TOPK}" \
  --alpha-lines "${IND_GRAPH_ALPHA_LINES}" \
  --lw-lines "${IND_GRAPH_LW_LINES}" \
  --lw-mean "${IND_GRAPH_LW_MEAN}" \
  --iqr-alpha "${IND_GRAPH_IQR_ALPHA}" \
  --dpi "${IND_GRAPH_DPI}"

ts "Saved industry overlap graph -> ${IND_OVERLAP_DIR}"

# ============================================================
# [3/5] Year-wise overlap (matrix + single grid heatmap)
# ============================================================
section "[3/5] Year-wise overlap"
ts "Running overlap_by_year.py"

YEAR_MATRIX_CMD=("${PYTHON_BIN}" "${REPO_ROOT}/sparse_auto_encoder/yearly/overlap_by_year.py"
  --features-parquet "${YEAR_FEATURES_PARQUET}"
  --out-dir "${YEAR_OVERLAP_DIR}"
  --group-by "${YEAR_GROUP_BY}"
  --years "${YEARS}"
  --topk "${YEAR_TOPK}"
  --layers "${YEAR_LAYERS}"
  --samples-per-group "${YEAR_SAMPLES_PER_GROUP}"
  --font-size "${YEAR_FONT_SIZE}"
  --max-iters "${YEAR_MAX_ITERS}"
  --seed "${YEAR_SEED}"
  --dpi "${YEAR_DPI}"
  --grid-cols "${YEAR_GRID_COLS}"
  --grid-out-name "${YEAR_GRID_OUT_NAME}"
  --csv-subdir "${YEAR_CSV_SUBDIR}"
)
if [[ -n "${YEAR_MAX_GROUPS}" ]]; then
  YEAR_MATRIX_CMD+=(--max-groups "${YEAR_MAX_GROUPS}")
fi
"${YEAR_MATRIX_CMD[@]}"

ts "Saved year overlap outputs -> ${YEAR_OVERLAP_DIR}"
ts "Per-layer CSVs -> ${YEAR_CSV_DIR}"

# ============================================================
# [4/5] Year overlap graph
#   IMPORTANT: read per-layer CSVs from overlaps/csvs
# ============================================================
section "[4/5] Year overlap graph"
ts "Running overlap_by_year_graph.py"

"${PYTHON_BIN}" "${REPO_ROOT}/sparse_auto_encoder/yearly/overlap_by_year_graph.py" \
  --in-dir "${YEAR_CSV_DIR}" \
  --out-dir "${YEAR_OVERLAP_DIR}" \
  --font-size "${YEAR_GRAPH_FONT_SIZE}" \
  --alpha-lines "${YEAR_GRAPH_ALPHA_LINES}" \
  --lw-lines "${YEAR_GRAPH_LW_LINES}" \
  --lw-mean "${YEAR_GRAPH_LW_MEAN}" \
  --order "${YEAR_GRAPH_ORDER}" \
  --legend-max-pairs "${YEAR_GRAPH_LEGEND_MAX_PAIRS}" \
  --iqr-alpha "${YEAR_GRAPH_IQR_ALPHA}" \
  --dpi "${YEAR_GRAPH_DPI}"

ts "Saved year overlap graph -> ${YEAR_OVERLAP_DIR}"

# ============================================================
# [5/5] Year overlap difference graph
#   IMPORTANT: read per-layer CSVs from overlaps/csvs
# ============================================================
section "[5/5] Year overlap difference graph"
ts "Running overlap_by_year_diff_graph.py"

"${PYTHON_BIN}" "${REPO_ROOT}/sparse_auto_encoder/yearly/overlap_by_diff_year_graph.py" \
  --in-dir "${YEAR_CSV_DIR}" \
  --out-png "${DIFF_OUT_PNG}" \
  --layers "${DIFF_LAYERS}" \
  --agg "${DIFF_AGG}" \
  --alpha-lines "${DIFF_ALPHA_LINES}" \
  --lw-lines "${DIFF_LW_LINES}" \
  --lw-global "${DIFF_LW_GLOBAL}" \
  --iqr-alpha "${DIFF_IQR_ALPHA}" \
  --min-diff-count "${DIFF_MIN_COUNT}" \
  --dpi "${DIFF_DPI}"

ts "Saved year overlap diff graph -> ${YEAR_OVERLAP_DIR}"

  ts "Completed overlap analyses for target skill: ${TARGET_SKILL}"
done

# ============================================================
# DONE
# ============================================================
section "DONE"
ts "All overlap analyses completed successfully for: ${TARGET_SKILLS[*]}"
