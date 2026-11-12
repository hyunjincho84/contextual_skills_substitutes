#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ============================================================
# CONFIG — edit values here
# ============================================================
# Enter values directly between the quotation marks.
PYTHON_BIN="python3"
BASE_DATA_DIR="/home/jovyan/LEM_data3/hyunjincho/LEM"

# Pipeline stages (true/false)
RUN_SAMPLING="true"
RUN_PREDICTION="true"

# Sampling
BERT_PRED_DIR="${BASE_DATA_DIR}/predictions_clean/bert_pred"
BERT_SAMPLE_PATH="${BASE_DATA_DIR}/predictions_clean/gpt_samplings/bert.csv"
GPT_OUTPUT_PATH="${BASE_DATA_DIR}/predictions_clean/gpt_samplings/gpt.csv"
SAMPLE_INPUT_GLOB="pred/20*/predictions_*.csv.gz"
SAMPLE_CHUNK_SIZE="200000"
SAMPLES_PER_YEAR="1000"
PARAMETER="text"
SEED="42"
YEARS=""  # e.g. "2018 2019"; empty = all years

# GPT prediction
GPT_IN_PATTERN="${BERT_SAMPLE_PATH}"
SKILL2IDX="${BASE_DATA_DIR}/all_filterd/skill2idx.json"
GPT_MODEL="gpt-5.1"
GPT_NUM_ROWS=""  # empty = full mode; integer = test mode
GPT_RESUME="true"
GPT_MAX_RETRIES="3"
GPT_RETRY_DELAY="3.0"
GPT_OUT_SUFFIX="_with_gpt_pred"
GPT_TEST_OUT="${BASE_DATA_DIR}/predictions_clean/gpt_samplings/gpt_test.csv"

export BASE_DATA_DIR
cd "${REPO_ROOT}"

# ============================================================
# GPT SAMPLING + PREDICTION PIPELINE
#
# Purpose:
#  - Sample 1,000 unique text pairs per year from BERT prediction files.
#  - Run GPT predictions on those sampled rows.
#
# Runs:
#  1) model_tests/get_samples_for_gpt.py
#     - Reads ${BERT_PRED_DIR}/pred/20*/predictions_*.csv.gz
#  2) model_tests/test_gpt.py
#     - Writes all GPT outputs to ${GPT_OUTPUT_PATH}.
#
# Input:
#  - BERT_PRED_DIR: ${BASE_DATA_DIR}/predictions_clean/bert_pred
#  - SKILL2IDX:     ${BASE_DATA_DIR}/all_filterd/skill2idx.json
#
# Output:
#  - ${BERT_SAMPLE_PATH} (up to 16,000 sampled rows)
#  - ${GPT_OUTPUT_PATH} (one prediction row per BERT sample)
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

is_true () {
  case "${1,,}" in
    true|1|yes|y) return 0 ;;
    false|0|no|n) return 1 ;;
    *)
      echo "[ERROR] Expected true/false, got: $1" >&2
      exit 2
      ;;
  esac
}

YEAR_ARGS=()
if [[ -n "${YEARS}" ]]; then
  read -r -a YEAR_ARGS <<< "${YEARS}"
fi


section "GPT SAMPLING + PREDICTION"
echo "BERT_PRED_DIR    = ${BERT_PRED_DIR}"
echo "BERT_SAMPLE_PATH = ${BERT_SAMPLE_PATH}"
echo "GPT_OUTPUT_PATH  = ${GPT_OUTPUT_PATH}"
echo "RUN_SAMPLING     = ${RUN_SAMPLING}"
echo "RUN_PREDICTION   = ${RUN_PREDICTION}"
echo "SAMPLE_INPUT_GLOB= ${SAMPLE_INPUT_GLOB}"
echo "SAMPLE_CHUNK_SIZE= ${SAMPLE_CHUNK_SIZE}"
echo "SAMPLES_PER_YEAR = ${SAMPLES_PER_YEAR}"
echo "PARAMETER        = ${PARAMETER}"
echo "SKILL2IDX        = ${SKILL2IDX}"
echo "SEED            = ${SEED}"
echo "GPT_MODEL       = ${GPT_MODEL}"
echo "GPT_NUM_ROWS    = ${GPT_NUM_ROWS:-all}"
echo "GPT_RESUME      = ${GPT_RESUME}"
echo "GPT_MAX_RETRIES = ${GPT_MAX_RETRIES}"
echo "GPT_RETRY_DELAY = ${GPT_RETRY_DELAY}"
echo "YEARS           = ${YEARS:-all}"

if is_true "${RUN_SAMPLING}"; then
  step "[1/2] Sampling ${SAMPLES_PER_YEAR} rows/year from BERT predictions"
  SAMPLE_CMD=("${PYTHON_BIN}" "${REPO_ROOT}/model_tests/get_samples_for_gpt.py"
    --in-root "${BERT_PRED_DIR}"
    --out-path "${BERT_SAMPLE_PATH}"
    --input-glob "${SAMPLE_INPUT_GLOB}"
    --chunk-size "${SAMPLE_CHUNK_SIZE}"
    --samples-per-year "${SAMPLES_PER_YEAR}"
    --parameter "${PARAMETER}"
    --seed "${SEED}"
  )
  if [[ ${#YEAR_ARGS[@]} -gt 0 ]]; then
    SAMPLE_CMD+=(--years "${YEAR_ARGS[@]}")
  fi
  "${SAMPLE_CMD[@]}"
else
  step "[1/2] Sampling skipped (RUN_SAMPLING=${RUN_SAMPLING})"
fi

if is_true "${RUN_PREDICTION}"; then
  step "[2/2] Running GPT prediction"
  GPT_CMD=("${PYTHON_BIN}" "${REPO_ROOT}/model_tests/test_gpt.py"
    --in-pattern "${GPT_IN_PATTERN}"
    --skill2idx "${SKILL2IDX}"
    --model "${GPT_MODEL}"
    --max-retries "${GPT_MAX_RETRIES}"
    --retry-delay "${GPT_RETRY_DELAY}"
    --out-suffix "${GPT_OUT_SUFFIX}"
    --out-path "${GPT_OUTPUT_PATH}"
    --test-out "${GPT_TEST_OUT}"
  )
  if [[ -n "${GPT_NUM_ROWS}" ]]; then
    GPT_CMD+=(--num-rows "${GPT_NUM_ROWS}")
  fi
  if is_true "${GPT_RESUME}"; then
    GPT_CMD+=(--resume)
  fi
  "${GPT_CMD[@]}"
else
  step "[2/2] GPT prediction skipped (RUN_PREDICTION=${RUN_PREDICTION})"
fi

section "PIPELINE COMPLETED SUCCESSFULLY"
ts "GPT sampling and prediction finished"
