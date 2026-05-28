#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_DATA_DIR="${BASE_DATA_DIR:-/home/jovyan/LEM_data2/data}"
export BASE_DATA_DIR
cd "${REPO_ROOT}"

# ============================================================
# FULL BASELINE PREDICTION PIPELINE
#
# Purpose:
#  - Run BERT, Skill2Vec, and Conditional Probability predictions
#    on the full preprocessed test set.
#  - GPT sampling/prediction is intentionally separated and should
#    be run after LLaMA scoring creates sv_summary_* files.
#
# Runs:
#  1) model_tests/test_bert.py
#  2) model_tests/test_skill2vec.py
#  3) model_tests/test_conditional_prob.py
#
# Input:
#  - ${BASE_DATA_DIR}/preprocessed_www_new/test
#  - ${BASE_DATA_DIR}/checkpoints(www)_new/best_model.pt
#  - ${BASE_DATA_DIR}/skill2vec_new
#  - ${BASE_DATA_DIR}/condprob_new
#
# Output:
#  - ${BASE_DATA_DIR}/bert_pred_new/pred
#  - ${BASE_DATA_DIR}/skill2vec_pred_new/pred
#  - ${BASE_DATA_DIR}/condprob_pred_new/pred
# ============================================================

BLUE="[1;34m"
GREEN="[1;32m"
YELLOW="[1;33m"
RESET="[0m"

section () {
  echo -e "
${BLUE}══════════════════════════════════════════════════════════════${RESET}"
  echo -e "${BLUE}▶ $1${RESET}"
  echo -e "${BLUE}══════════════════════════════════════════════════════════════${RESET}
"
}

step () { echo -e "${GREEN}➤ $1${RESET}"; }
ts () { echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${RESET} $1"; }

section "FULL BASELINE PREDICTION PIPELINE"
ts "Starting baseline prediction pipeline"

step "[1/3] BERT"
"${PYTHON_BIN}" "${REPO_ROOT}/model_tests/test_bert.py"

step "[2/3] Skill2Vec"
"${PYTHON_BIN}" "${REPO_ROOT}/model_tests/test_skill2vec.py"

step "[3/3] Conditional Probability"
"${PYTHON_BIN}" "${REPO_ROOT}/model_tests/test_conditional_prob.py"

section "PIPELINE COMPLETED SUCCESSFULLY"
ts "Baseline predictions finished"
