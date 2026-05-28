#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ============================================================
# POSTING-LEVEL TREND-BASED EXPOSURE GAIN PIPELINE (2025)
#
# Purpose:
#  - Build substitute-pair counts, attach Google Trends values, and plot posting-level margins.
#
# Runs:
#  1) exposure_gain/get_count_by_pair.py
#     - Builds (truth, subs, year_month, count) pairs from prediction files.
#  2) exposure_gain/get_trend.py
#     - Fetches monthly Google Trends values for each pair.
#  3) exposure_gain/get_graph.py
#     - Computes posting-level margin ratios and writes a histogram.
#
# Input:
#  - PRED_DIR:          ${BASE_DATA_DIR}/bert_pred_new/pred/2025
#  - PRED_DIR_NEW:      ${BASE_DATA_DIR}/bert_pred_new/pred/2025
#  - PREPROCESSED_ROOT: ${BASE_DATA_DIR}/preprocessed_www_new/test/2025
#
# Output:
#  - COUNTS_BY_PAIR: ${BASE_DATA_DIR}/margin/counts_by_pair_2025.csv
#  - TRENDS_OUT:     ${BASE_DATA_DIR}/exposure_gain/counts_by_pair_with_trends_monthly.csv
#  - OUT_FIG:        ${BASE_DATA_DIR}/exposure_gain/posting_margin_ratio_hist_2025.png
# ============================================================

BLUE="\033[1;34m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
RESET="\033[0m"

section () {
  echo -e "\n${BLUE}══════════════════════════════════════════════════════════════${RESET}"
  echo -e "${BLUE}▶ $1${RESET}"
  echo -e "${BLUE}══════════════════════════════════════════════════════════════${RESET}\n"
}

ts () {
  echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${RESET} $1"
}

trap 'echo -e "${RED}✗ Failed at line $LINENO${RESET}"' ERR

PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_DATA_DIR="${BASE_DATA_DIR:-/home/jovyan/LEM_data2/data}"

# ============================================================
# Where the .py scripts live
# ============================================================
EXPOSURE_DIR="${REPO_ROOT}/exposure_gain"

cd "${EXPOSURE_DIR}"

# ============================================================
# Input / Output Paths
# ============================================================

# [1] Prediction → counts_by_pair
PRED_DIR="${PRED_DIR:-${BASE_DATA_DIR}/bert_pred_new/pred/2025}"
COUNTS_BY_PAIR="${COUNTS_BY_PAIR:-${BASE_DATA_DIR}/margin/counts_by_pair_2025.csv}"

# [2] Google Trends output
TRENDS_OUT="${TRENDS_OUT:-${BASE_DATA_DIR}/exposure_gain/counts_by_pair_with_trends_monthly.csv}"

# [3] Posting-level analysis inputs + output figure
PRED_DIR_NEW="${PRED_DIR_NEW:-${BASE_DATA_DIR}/bert_pred_new/pred/2025}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${BASE_DATA_DIR}/preprocessed_www_new/test/2025}"
OUT_FIG="${OUT_FIG:-${BASE_DATA_DIR}/exposure_gain/posting_margin_ratio_hist_2025.png}"
mkdir -p "$(dirname "${COUNTS_BY_PAIR}")" "$(dirname "${TRENDS_OUT}")" "$(dirname "${OUT_FIG}")"

# ============================================================
# [1/3] Build counts_by_pair_2025.csv
# ============================================================
section "[1/3] Build counts_by_pair (from predictions)"
ts "Running get_count_by_pair.py"

"${PYTHON_BIN}" "${EXPOSURE_DIR}/get_count_by_pair.py" \
  --pred-dir "${PRED_DIR}" \
  --out-csv "${COUNTS_BY_PAIR}"

ts "Saved counts_by_pair -> ${COUNTS_BY_PAIR}"

# ============================================================
# [2/3] Fetch Google Trends (monthly)
# ============================================================
section "[2/3] Fetch Google Trends (monthly)"
ts "Running get_trend.py"

"${PYTHON_BIN}" "${EXPOSURE_DIR}/get_trend.py" \
  --in-csv "${COUNTS_BY_PAIR}" \
  --out-csv "${TRENDS_OUT}" \
  --geo US

ts "Saved trends -> ${TRENDS_OUT} (+ .partial if interrupted)"

# ============================================================
# [3/3] Posting-level margin & visualization
# ============================================================
section "[3/3] Posting-level margin analysis & graph"
ts "Running get_graph.py"

"${PYTHON_BIN}" "${EXPOSURE_DIR}/get_graph.py" \
  --pred-dir "${PRED_DIR_NEW}" \
  --preprocessed-root "${PREPROCESSED_ROOT}" \
  --margin-csv "${TRENDS_OUT}" \
  --drop-bidirectional \
  --out-fig "${OUT_FIG}"

ts "Saved histogram -> ${OUT_FIG}"

section "DONE"
ts "Pipeline completed successfully"
