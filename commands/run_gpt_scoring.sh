#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_DATA_DIR="${BASE_DATA_DIR:-/home/jovyan/LEM_data2/data}"
export BASE_DATA_DIR
cd "${REPO_ROOT}"

# ============================================================
# GPT SV SCORING + COMPARE PIPELINE
#
# Purpose:
#  - Score GPT sampled prediction files with LLaMA SV.
#  - Compare GPT SV with the BERT sv_summary_* rows sampled earlier.
#
# Runs:
#  1) model_tests/likelihood_based_score/scoring_gpt.py
#  2) model_tests/likelihood_based_score/compute_gpt_score_mean.py
#
# Input:
#  - GPT_IN_PATTERN: ${BASE_DATA_DIR}/gpt_samples/20*/*with_gpt_pred.csv*
#  - BERT_BASE:      ${BASE_DATA_DIR}/bert_pred_new/pred
#
# Output:
#  - ${BASE_DATA_DIR}/gpt_samples/20*/*with_sv_llama.csv.gz
#  - GPT/BERT sampled-row comparison summary printed by compute_gpt_score_mean.py
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
GPT_IN_PATTERN="${GPT_IN_PATTERN:-${BASE_DATA_DIR}/gpt_samples/20*/*with_gpt_pred.csv*}"
GPT_WITH_SV_PATTERN="${GPT_WITH_SV_PATTERN:-${BASE_DATA_DIR}/gpt_samples/20*/*with_sv_llama.csv.gz}"
BERT_BASE="${BERT_BASE:-${BASE_DATA_DIR}/bert_pred_new/pred}"
WINDOW_SIZE="${WINDOW_SIZE:-256}"
MAX_LEN="${MAX_LEN:-512}"
SENT_BATCH="${SENT_BATCH:-4}"
CHUNK_SIZE="${CHUNK_SIZE:-50000}"
RESUME="${RESUME:-0}"
USE_AUTH_TOKEN="${USE_AUTH_TOKEN:-0}"
MAX_MISMATCH_EXAMPLES="${MAX_MISMATCH_EXAMPLES:-5}"
YEARS="${YEARS:-}"  # e.g. YEARS="2018 2019"

YEAR_ARGS=()
if [[ -n "${YEARS}" ]]; then
  read -r -a YEAR_ARGS <<< "${YEARS}"
fi

section "GPT SV SCORING + COMPARE"
echo "GPT_IN_PATTERN      = ${GPT_IN_PATTERN}"
echo "GPT_WITH_SV_PATTERN = ${GPT_WITH_SV_PATTERN}"
echo "BERT_BASE           = ${BERT_BASE}"
echo "LLAMA_CKPT          = ${LLAMA_CKPT}"
echo "YEARS               = ${YEARS:-all}"

CMD=("${PYTHON_BIN}" "${REPO_ROOT}/model_tests/likelihood_based_score/scoring_gpt.py"
  --in-pattern "${GPT_IN_PATTERN}"
  --llama-ckpt "${LLAMA_CKPT}"
  --window-size "${WINDOW_SIZE}"
  --max-len "${MAX_LEN}"
  --sent-batch "${SENT_BATCH}"
  --chunk-size "${CHUNK_SIZE}"
)
if [[ ${#YEAR_ARGS[@]} -gt 0 ]]; then
  CMD+=(--years "${YEAR_ARGS[@]}")
fi
if [[ "${RESUME}" == "1" ]]; then
  CMD+=(--resume)
fi
if [[ "${USE_AUTH_TOKEN}" == "1" ]]; then
  CMD+=(--use-auth-token)
fi

step "[1/2] Scoring GPT predictions with LLaMA SV"
"${CMD[@]}"

step "[2/2] Comparing GPT sampled SV against BERT sv_summary_*"
"${PYTHON_BIN}" "${REPO_ROOT}/model_tests/likelihood_based_score/compute_gpt_score_mean.py" \
  --gpt-pattern "${GPT_WITH_SV_PATTERN}" \
  --bert-base "${BERT_BASE}" \
  --max-mismatch-examples "${MAX_MISMATCH_EXAMPLES}"

section "PIPELINE COMPLETED SUCCESSFULLY"
ts "GPT SV scoring and comparison finished"
