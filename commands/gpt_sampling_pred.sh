#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_DATA_DIR="${BASE_DATA_DIR:-/home/jovyan/LEM_data2/data}"
export BASE_DATA_DIR
cd "${REPO_ROOT}"

# ============================================================
# GPT SAMPLING + PREDICTION PIPELINE
#
# Purpose:
#  - Sample GPT evaluation rows from BERT sv_summary_* files created
#    by commands/run_scoring.sh.
#  - Run GPT predictions on those sampled rows.
#
# Runs:
#  1) model_tests/get_samples_for_gpt.py
#     - Reads ${BERT_SV_ROOT}/20*/sv_summary_llama_full_bert_*.csv.gz
#  2) model_tests/test_gpt.py
#     - Writes *_with_gpt_pred.csv.gz next to sampled files.
#
# Input:
#  - BERT_SV_ROOT: ${BASE_DATA_DIR}/bert_pred_new/pred
#  - SKILL2IDX:    ${BASE_DATA_DIR}/preprocessed_www_new/skill2idx.json
#
# Output:
#  - ${GPT_SAMPLES_DIR}/20*/gpt_unique_samples_*_global*.csv.gz
#  - ${GPT_SAMPLES_DIR}/20*/gpt_unique_samples_*_global*_with_gpt_pred.csv.gz
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

BERT_SV_ROOT="${BERT_SV_ROOT:-${BASE_DATA_DIR}/bert_pred_new/pred}"
GPT_SAMPLES_DIR="${GPT_SAMPLES_DIR:-${BASE_DATA_DIR}/gpt_samples}"
SKILL2IDX="${SKILL2IDX:-${BASE_DATA_DIR}/preprocessed_www_new/skill2idx.json}"
SAMPLE_FRAC="${SAMPLE_FRAC:-0.01}"
SEED="${SEED:-42}"
GPT_MODEL="${GPT_MODEL:-gpt-5.1}"
YEARS="${YEARS:-}"  # e.g. YEARS="2018 2019"

YEAR_ARGS=()
if [[ -n "${YEARS}" ]]; then
  read -r -a YEAR_ARGS <<< "${YEARS}"
fi

GPT_IN_PATTERN="${GPT_IN_PATTERN:-${GPT_SAMPLES_DIR}/20*/gpt_unique_samples_*_global*0.csv.gz}"

section "GPT SAMPLING + PREDICTION"
echo "BERT_SV_ROOT    = ${BERT_SV_ROOT}"
echo "GPT_SAMPLES_DIR = ${GPT_SAMPLES_DIR}"
echo "SKILL2IDX       = ${SKILL2IDX}"
echo "SAMPLE_FRAC     = ${SAMPLE_FRAC}"
echo "SEED            = ${SEED}"
echo "GPT_MODEL       = ${GPT_MODEL}"
echo "YEARS           = ${YEARS:-all}"

step "[1/2] Sampling from BERT sv_summary_* files"
SAMPLE_CMD=("${PYTHON_BIN}" "${REPO_ROOT}/model_tests/get_samples_for_gpt.py"
  --in-root "${BERT_SV_ROOT}"
  --out-root "${GPT_SAMPLES_DIR}"
  --sample-frac "${SAMPLE_FRAC}"
  --seed "${SEED}"
)
if [[ ${#YEAR_ARGS[@]} -gt 0 ]]; then
  SAMPLE_CMD+=(--years "${YEAR_ARGS[@]}")
fi
"${SAMPLE_CMD[@]}"

step "[2/2] Running GPT prediction"
"${PYTHON_BIN}" "${REPO_ROOT}/model_tests/test_gpt.py" \
  --in-pattern "${GPT_IN_PATTERN}" \
  --skill2idx "${SKILL2IDX}" \
  --model "${GPT_MODEL}"

section "PIPELINE COMPLETED SUCCESSFULLY"
ts "GPT sampling and prediction finished"
