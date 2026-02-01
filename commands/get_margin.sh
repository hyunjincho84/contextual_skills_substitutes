#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Posting-level Trend-based Exposure Gain Pipeline (2025)
#
#  1) Build (truth, subs, year_month, count)
#     - get_count_by_pair.py
#
#  2) Fetch Google Trends (monthly, Jan–Jun 2025)
#     - get_trend.py
#
#  3) Compute posting-level margin & visualize
#     - get_graph.py
# ============================================================

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

ts () {
  echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${RESET} $1"
}

trap 'echo -e "${RED}✗ Failed at line $LINENO${RESET}"' ERR

PYTHON_BIN="python3"

# ============================================================
# Paths (EDIT IF NEEDED)
# ============================================================

# [1] Prediction → counts_by_pair
PRED_DIR="/home/jovyan/LEM_data2/hyunjincho/bert_pred/pred/2025"
COUNTS_BY_PAIR="/home/jovyan/LEM_data2/hyunjincho/margin/counts_by_pair_2025.csv"

# [2] Google Trends
TRENDS_OUT="./counts_by_pair_with_trends_monthly.csv"

# [3] Posting-level analysis & graph
PRED_DIR_NEW="/home/jovyan/LEM_data2/hyunjincho/bert_pred_new/pred/2025"
PREPROCESSED_ROOT="/home/jovyan/LEM_data2/hyunjincho/preprocessed_www_new/test/2025"
OUT_FIG="./posting_margin_ratio_hist_2025.png"

# ============================================================
# [1/3] Build (truth, subs, year_month, count)
# ============================================================
section "[1/3] Build counts_by_pair (from predictions)"
ts "Running get_count_by_pair.py"

$PYTHON_BIN get_count_by_pair.py \
  --pred-dir "${PRED_DIR}" \
  --out-csv "${COUNTS_BY_PAIR}"

ts "Saved counts_by_pair -> ${COUNTS_BY_PAIR}"

# ============================================================
# [2/3] Fetch Google Trends
# ============================================================
section "[2/3] Fetch Google Trends (monthly)"
ts "Running get_trend.py"

$PYTHON_BIN get_trend.py \
  --in-csv "${COUNTS_BY_PAIR}" \
  --out-csv "${TRENDS_OUT}" \
  --geo US

ts "Saved trends -> ${TRENDS_OUT} (+ .partial if interrupted)"

# ============================================================
# [3/3] Posting-level margin & visualization
# ============================================================
section "[3/3] Posting-level margin analysis & graph"
ts "Running get_graph.py"

$PYTHON_BIN get_graph.py \
  --pred-dir "${PRED_DIR_NEW}" \
  --preprocessed-root "${PREPROCESSED_ROOT}" \
  --margin-csv "${TRENDS_OUT}" \
  --drop-bidirectional \
  --out-fig "${OUT_FIG}"

ts "Saved histogram -> ${OUT_FIG}"

# ============================================================
# DONE
# ============================================================
section "DONE"
ts "Posting-level trend-based exposure gain pipeline completed successfully"