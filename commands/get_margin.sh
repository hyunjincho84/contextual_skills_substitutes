#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Trend-based Analysis Pipeline
#  1) Fetch Google Trends (get_trend.py)
#  2) Aggregate margin scores (get_margin / aggregate.py)
#  3) Visualize results (get_graph.py)
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

# ----------------------------
# Paths (adjust if needed)
# ----------------------------
COUNTS_BY_PAIR="../exposure_gain/counts_by_pair.csv"
TRENDS_OUT="../exposure_gain/counts_by_pair_with_trends_monthly.csv"
SAMPLE_SCORED="../exposure_gain/sample_1000_scored.csv"
MERGED_OUT="../exposure_gain/sample_1000_scored_with_trends.csv"
GRAPH_OUT_DIR="../exposure_gain/trend_graphs"

mkdir -p "${GRAPH_OUT_DIR}"

# ============================================================
# [1/3] Fetch Google Trends
# ============================================================
# section "[1/3] Fetch Google Trends"
# ts "Running get_trend.py"

# $PYTHON_BIN ../exposure_gain/get_trend.py \
#   --in-csv "${COUNTS_BY_PAIR}" \
#   --out-csv "${TRENDS_OUT}" \
#   --geo US

# ts "Saved trends -> ${TRENDS_OUT} (+ .partial if interrupted)"

# ============================================================
# [2/3] Aggregate margin scores
# ============================================================
section "[2/3] Aggregate trend margin"
ts "Running get_margin (aggregate.py)"

$PYTHON_BIN ../exposure_gain/get_margin.py \
  --partial "${TRENDS_OUT}.partial" \
  --sample "${SAMPLE_SCORED}" \
  --out "${MERGED_OUT}"

ts "Saved merged results -> ${MERGED_OUT}"

# ============================================================
# [3/3] Plot graphs
# ============================================================
section "[3/3] Plot trend graphs"
ts "Running get_graph.py"

$PYTHON_BIN ../exposure_gain/get_graph.py \
  --in-csv "${MERGED_OUT}" \
  --out-dir "${GRAPH_OUT_DIR}"

ts "Saved graphs -> ${GRAPH_OUT_DIR}"

# ============================================================
# DONE
# ============================================================
section "DONE"
ts "Trend-based analysis pipeline completed successfully"