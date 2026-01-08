#!/usr/bin/env bash
set -e

# ============================================================
# Substitute Analysis Pipeline
#  1) SOC2-wise substitutes (areawise)
#  2) Year-wise substitutes
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
TARGET_SKILL="python"

# BERT prediction roots
PRED_ROOT_SOC2="/home/jovyan/LEM_data2/hyunjincho/bert_pred_new/pred"
PRED_ROOT_YEAR="/home/jovyan/LEM_data2/hyunjincho/bert_pred_new/pred"

# outputs
OUT_SOC2="../substitute_by_area/subs_python_by_soc2.csv"
OUT_YEAR="../substitute_by_time/python_yearwise_top5.csv"

# ---------- run ----------
section "SUBSTITUTE ANALYSIS PIPELINE"
ts "Target skill = ${TARGET_SKILL}"

section "[1/2] SOC2-wise substitutes"
step "Running areawise_substitutes.py"

$PYTHON_BIN ../substitute_by_area/areawise_substitutes.py \
  --pred-root "${PRED_ROOT_SOC2}" \
  --target-skill "${TARGET_SKILL}" \
  --topk-per-soc2 5 \
  --out "${OUT_SOC2}" \
  --usecols

ts "Saved SOC2-wise results -> ${OUT_SOC2}"

section "[2/2] Year-wise substitutes"
step "Running yearwise_substitutes.py"

$PYTHON_BIN ../substitute_by_time/yearwise_substitutes.py \
  --pred-root "${PRED_ROOT_YEAR}" \
  --target-skill "${TARGET_SKILL}" \
  --out-topk 5 \
  --out "${OUT_YEAR}"

ts "Saved year-wise results -> ${OUT_YEAR}"

section "DONE"
ts "All substitute analyses completed successfully"