#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Overlap Analysis Pipeline
#  1) Industry-wise overlap
#  2) Industry overlap graph
#  3) Year-wise overlap
#  4) Year overlap graph
#  5) Year overlap diff graph
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
IND_DIR="../sparse_auto_encoder/python_industry"
YEAR_DIR="../sparse_auto_encoder/python_yearly"

# ============================================================
# [1/5] Industry-wise overlap (matrix + heatmaps)
# ============================================================
section "[1/5] Industry-wise overlap"
ts "Running overlap_by_industry.py"

$PYTHON_BIN ../sparse_auto_encoder/industry/overlap_by_industry.py \
  --features-parquet "${IND_DIR}/features.parquet" \
  --out-dir "${IND_DIR}/overlaps" \
  --topk 128 \
  --layers 1-12 \
  --samples-per-industry 250 \
  --font-size 16

# ============================================================
# [2/5] Industry overlap graph
# ============================================================
section "[2/5] Industry overlap graph"
ts "Running overlap_by_industry_graph.py"

$PYTHON_BIN ../sparse_auto_encoder/industry/overlap_by_industry_graph.py \
  --in-dir "${IND_DIR}/overlaps" \
  --out-dir "${IND_DIR}/overlaps" \
  --alpha-lines 0.25

# ============================================================
# [3/5] Year-wise overlap
# ============================================================
section "[3/5] Year-wise overlap"
ts "Running overlap_by_year.py"

$PYTHON_BIN ../sparse_auto_encoder/yearly/overlap_by_year.py \
  --features-parquet "${YEAR_DIR}/features.parquet" \
  --out-dir "${YEAR_DIR}/overlaps" \
  --group-by year \
  --years 2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025 \
  --topk 128 \
  --layers 1-12 \
  --samples-per-group 1000 \
  --font-size 16

# ============================================================
# [4/5] Year overlap graph
# ============================================================
section "[4/5] Year overlap graph"
ts "Running overlap_by_year_graph.py"

$PYTHON_BIN ../sparse_auto_encoder/yearly/overlap_by_year_graph.py \
  --in-dir "${YEAR_DIR}/overlaps" \
  --out-dir "${YEAR_DIR}/overlaps" \
  --alpha-lines 0.20

# ============================================================
# [5/5] Year overlap difference graph
# ============================================================
section "[5/5] Year overlap difference graph"
ts "Running overlap_by_year_diff_graph.py"

$PYTHON_BIN ../sparse_auto_encoder/yearly/overlap_by_diff_year_graph.py \
  --in-dir "${YEAR_DIR}/overlaps" \
  --out-png "${YEAR_DIR}/overlaps/year_diff_overlap.png" \
  --layers 1,2,3,4,5,6,7,8,9,10,11,12 \
  --agg mean \
  --alpha-lines 0.25

# ============================================================
# DONE
# ============================================================
section "DONE"
ts "All overlap analyses completed successfully"