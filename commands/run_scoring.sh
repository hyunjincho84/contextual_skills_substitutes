#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_DATA_DIR="${BASE_DATA_DIR:-/home/jovyan/LEM_data2/data}"
export BASE_DATA_DIR
cd "${REPO_ROOT}"

# ============================================================
# FULL MODEL SV SCORING PIPELINE
#
# Purpose:
#  - Score BERT / Skill2Vec / Conditional Probability predictions
#    with LLaMA-based semantic validity (SV).
#  - Write sv_summary_* files that downstream GPT sampling can read.
#
# Runs:
#  1) model_tests/likelihood_based_score/scoring.py
#     - Computes SV for BERT / Skill2Vec / Conditional Probability outputs.
#  2) model_tests/likelihood_based_score/compute_score_mean.py
#     - Computes mean SV per model on the full prediction set.
#
# Input:
#  - BERT_PRED_DIR: ${BASE_DATA_DIR}/bert_pred_new/pred
#  - W2V_PRED_DIR:  ${BASE_DATA_DIR}/skill2vec_pred_new/pred
#  - COND_PRED_DIR: ${BASE_DATA_DIR}/condprob_pred_new/pred
#  - LLaMA checkpoint: ${LLAMA_CKPT}
#
# Output:
#  - ${BERT_PRED_DIR}/20*/sv_summary_llama_full_bert_*.csv.gz
#  - ${W2V_PRED_DIR}/20*/sv_summary_llama_full_w2v_*.csv.gz
#  - ${COND_PRED_DIR}/20*/sv_summary_llama_full_conditional_*.csv.gz
#  - Mean summaries printed/written by compute_score_mean.py
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

LLAMA_CKPT="${LLAMA_CKPT:-meta-llama/Llama-3.2-3B}"
YEARS="${YEARS:-}"  # e.g. YEARS="2018 2019"
YEAR_ARGS=()
if [[ -n "${YEARS}" ]]; then
  read -r -a YEAR_ARGS <<< "${YEARS}"
fi

BERT_PRED_DIR="${BERT_PRED_DIR:-${BASE_DATA_DIR}/bert_pred_new/pred}"
W2V_PRED_DIR="${W2V_PRED_DIR:-${BASE_DATA_DIR}/skill2vec_pred_new/pred}"
COND_PRED_DIR="${COND_PRED_DIR:-${BASE_DATA_DIR}/condprob_pred_new/pred}"

START_FROM="${START_FROM:-all}"   # all | bert | w2v | conditional | bert,w2v
CAP="${CAP:-8000}"
WINDOW_SIZE="${WINDOW_SIZE:-256}"
SENT_BATCH="${SENT_BATCH:-4}"

SCORE_FULL_SCRIPT="${REPO_ROOT}/model_tests/likelihood_based_score/scoring.py"
MEAN_FULL_SCRIPT="${REPO_ROOT}/model_tests/likelihood_based_score/compute_score_mean.py"

section "CONFIG"
ts "Running full-data LLaMA SV scoring"
echo "PYTHON_BIN      = ${PYTHON_BIN}"
echo "LLAMA_CKPT      = ${LLAMA_CKPT}"
echo "YEARS           = ${YEARS:-all}"
echo "BERT_PRED_DIR   = ${BERT_PRED_DIR}"
echo "W2V_PRED_DIR    = ${W2V_PRED_DIR}"
echo "COND_PRED_DIR   = ${COND_PRED_DIR}"
echo "START_FROM      = ${START_FROM}"
echo "CAP             = ${CAP}"
echo "WINDOW_SIZE     = ${WINDOW_SIZE}"
echo "SENT_BATCH      = ${SENT_BATCH}"

section "[1/2] Compute SV(LLaMA)"
step "Running scoring.py"
SCORE_CMD=("${PYTHON_BIN}" "${SCORE_FULL_SCRIPT}"
  --bert_pred_dir "${BERT_PRED_DIR}"
  --w2v_pred_dir "${W2V_PRED_DIR}"
  --cond_pred_dir "${COND_PRED_DIR}"
  --llama_ckpt "${LLAMA_CKPT}"
  --start-from "${START_FROM}"
  --cap "${CAP}"
  --window-size "${WINDOW_SIZE}"
  --sent-batch "${SENT_BATCH}"
)
if [[ ${#YEAR_ARGS[@]} -gt 0 ]]; then
  SCORE_CMD+=(--years "${YEAR_ARGS[@]}")
fi
"${SCORE_CMD[@]}"

section "[2/2] Compute mean SV(LLaMA)"
step "Running compute_score_mean.py"
MEAN_CMD=("${PYTHON_BIN}" "${MEAN_FULL_SCRIPT}"
  --bert_pred_dir "${BERT_PRED_DIR}"
  --w2v_pred_dir "${W2V_PRED_DIR}"
  --cond_pred_dir "${COND_PRED_DIR}"
)
if [[ ${#YEAR_ARGS[@]} -gt 0 ]]; then
  MEAN_CMD+=(--years "${YEAR_ARGS[@]}")
fi
"${MEAN_CMD[@]}"

section "PIPELINE COMPLETED SUCCESSFULLY"
ts "Model SV scoring finished"
