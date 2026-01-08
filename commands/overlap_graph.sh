#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Overlap Analysis Pipeline
#  1) Industry-wise overlap (single grid heatmap + per-layer CSVs)
#  2) Industry overlap graph
#  3) Year-wise overlap (single grid heatmap + per-layer CSVs)
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

step () { echo -e "${GREEN}➤ $1${RESET}"; }
ts () { echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${RESET} $1"; }

trap 'echo -e "${RED}✗ Failed at line $LINENO${RESET}"' ERR

PYTHON_BIN="python3"

# ---------- paths ----------
IND_DIR="../sparse_auto_encoder/python_industry"
YEAR_DIR="../sparse_auto_encoder/python_yearly"

IND_OVERLAP_DIR="${IND_DIR}/overlaps"
YEAR_OVERLAP_DIR="${YEAR_DIR}/overlaps"

# ✅ per-layer CSVs are saved here by the updated scripts
IND_CSV_DIR="${IND_OVERLAP_DIR}/csvs"
YEAR_CSV_DIR="${YEAR_OVERLAP_DIR}/csvs"

# Years list (avoid comma parsing issues)
YEARS="2010-2025"

# ============================================================
# [1/5] Industry-wise overlap (matrix + single grid heatmap)
# ============================================================
section "[1/5] Industry-wise overlap"
ts "Running overlap_by_industry.py"

$PYTHON_BIN ../sparse_auto_encoder/industry/overlap_by_industry.py \
  --features-parquet "${IND_DIR}/features.parquet" \
  --out-dir "${IND_OVERLAP_DIR}" \
  --topk 128 \
  --layers 1-12 \
  --samples-per-industry 250 \
  --font-size 16 \
  --grid-cols 4 \
  --grid-out-name heatmap_overlap_all_layers.png

ts "Saved industry overlap outputs -> ${IND_OVERLAP_DIR}"
ts "Per-layer CSVs -> ${IND_CSV_DIR}"

# ============================================================
# [2/5] Industry overlap graph
#   IMPORTANT: read per-layer CSVs from overlaps/csvs
# ============================================================
section "[2/5] Industry overlap graph"
ts "Running overlap_by_industry_graph.py"

$PYTHON_BIN ../sparse_auto_encoder/industry/overlap_by_industry_graph.py \
  --in-dir "${IND_CSV_DIR}" \
  --out-dir "${IND_OVERLAP_DIR}" \
  --alpha-lines 0.25

ts "Saved industry overlap graph -> ${IND_OVERLAP_DIR}"

# ============================================================
# [3/5] Year-wise overlap (matrix + single grid heatmap)
# ============================================================
section "[3/5] Year-wise overlap"
ts "Running overlap_by_year.py"

$PYTHON_BIN ../sparse_auto_encoder/yearly/overlap_by_year.py \
  --features-parquet "${YEAR_DIR}/features.parquet" \
  --out-dir "${YEAR_OVERLAP_DIR}" \
  --group-by year \
  --years "${YEARS}" \
  --topk 128 \
  --layers 1-12 \
  --samples-per-group 1000 \
  --font-size 16 \
  --grid-cols 4 \
  --grid-out-name heatmap_overlap_all_layers.png

ts "Saved year overlap outputs -> ${YEAR_OVERLAP_DIR}"
ts "Per-layer CSVs -> ${YEAR_CSV_DIR}"

# ============================================================
# [4/5] Year overlap graph
#   IMPORTANT: read per-layer CSVs from overlaps/csvs
# ============================================================
section "[4/5] Year overlap graph"
ts "Running overlap_by_year_graph.py"

$PYTHON_BIN ../sparse_auto_encoder/yearly/overlap_by_year_graph.py \
  --in-dir "${YEAR_CSV_DIR}" \
  --out-dir "${YEAR_OVERLAP_DIR}" \
  --alpha-lines 0.20

ts "Saved year overlap graph -> ${YEAR_OVERLAP_DIR}"

# ============================================================
# [5/5] Year overlap difference graph
#   IMPORTANT: read per-layer CSVs from overlaps/csvs
# ============================================================
section "[5/5] Year overlap difference graph"
ts "Running overlap_by_year_diff_graph.py"

$PYTHON_BIN ../sparse_auto_encoder/yearly/overlap_by_diff_year_graph.py \
  --in-dir "${YEAR_CSV_DIR}" \
  --out-png "${YEAR_OVERLAP_DIR}/yearly_diff.png" \
  --layers 1,2,3,4,5,6,7,8,9,10,11,12 \
  --agg mean \
  --alpha-lines 0.25 \

ts "Saved year overlap diff graph -> ${YEAR_OVERLAP_DIR}"

# ============================================================
# DONE
# ============================================================
section "DONE"
ts "All overlap analyses completed successfully"