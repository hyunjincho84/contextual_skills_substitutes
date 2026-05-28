#!/usr/bin/env bash
#main 에 들어가야함
set -euo pipefail

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODE="${1:-full}"
if [[ "${MODE}" != "debug" && "${MODE}" != "full" ]]; then
  echo "Usage: $0 [debug|full]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ============================================================
# QWEN/GEMMA VALIDATION PIPELINE
#
# Purpose:
#  - Use Qwen2.5 and Gemma 2 as judge models for model prediction outputs.
#  - Supports debug mode for a small file and full mode for all prediction directories.
#
# Runs:
#  1) model_tests/likelihood_valid_with_other_models/valid_other_models.py
#     - Evaluates BERT / Skill2Vec / Conditional Probability outputs with Qwen2.5.
#  2) model_tests/likelihood_valid_with_other_models/valid_other_models.py
#     - Evaluates the same outputs with Gemma 2.
#  3) model_tests/likelihood_valid_with_other_models/get_mean.py
#     - Summarizes Qwen/Gemma judge scores for full runs.
#
# Input:
#  - BERT_PRED_DIR: ${BASE_DATA_DIR}/bert_pred_new/pred
#  - W2V_PRED_DIR:  ${BASE_DATA_DIR}/skill2vec_pred_new/pred
#  - COND_PRED_DIR: ${BASE_DATA_DIR}/condprob_pred_new/pred
#  - DEBUG_FILE:    ${BERT_PRED_DIR}/2016/sv_summary_llama_full_bert_2016-08.csv.gz
#  - Judge models:  ${QWEN_CKPT}, ${GEMMA_CKPT}
#
# Output:
#  - QWEN_OUTPUT_ROOT:  ${BASE_DATA_DIR}/qwen25_3b_eval_full
#  - GEMMA_OUTPUT_ROOT: ${BASE_DATA_DIR}/gemma2_2b_eval_full
#  - MEAN_OUTPUT_CSV:   ${BASE_DATA_DIR}/command_outputs/mean_summary_qwen25_gemma_${TIMESTAMP}.csv
#  - Log file:          ${BASE_DATA_DIR}/command_outputs/logs/eval_gemma_qwen_${MODE}_${TIMESTAMP}.log
# ============================================================

VALID_SCRIPT="${REPO_ROOT}/model_tests/likelihood_valid_with_other_models/valid_other_models.py"
MEAN_SCRIPT="${REPO_ROOT}/model_tests/likelihood_valid_with_other_models/get_mean.py"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_DATA_DIR="${BASE_DATA_DIR:-/home/jovyan/LEM_data2/data}"
export BASE_DATA_DIR
cd "${REPO_ROOT}"
BERT_PRED_DIR="${BERT_PRED_DIR:-${BASE_DATA_DIR}/bert_pred_new/pred}"
W2V_PRED_DIR="${W2V_PRED_DIR:-${BASE_DATA_DIR}/skill2vec_pred_new/pred}"
COND_PRED_DIR="${COND_PRED_DIR:-${BASE_DATA_DIR}/condprob_pred_new/pred}"

QWEN_CKPT="${QWEN_CKPT:-Qwen/Qwen2.5-3B-Instruct}"
GEMMA_CKPT="${GEMMA_CKPT:-google/gemma-2-2b-it}"

QWEN_OUTPUT_ROOT="${QWEN_OUTPUT_ROOT:-${BASE_DATA_DIR}/qwen25_3b_eval_full}"
GEMMA_OUTPUT_ROOT="${GEMMA_OUTPUT_ROOT:-${BASE_DATA_DIR}/gemma2_2b_eval_full}"

ANCHOR_SEARCH_RADIUS="${ANCHOR_SEARCH_RADIUS:-2}"
SENT_BATCH="${SENT_BATCH:-4}"
GPU_IDS="${GPU_IDS:-0,1,2}"
IFS="," read -r -a GPU_ID_LIST <<< "${GPU_IDS}"
NUM_WORKERS="${NUM_WORKERS:-${#GPU_ID_LIST[@]}}"
DEBUG_N="${DEBUG_N:-10}"
DEBUG_FILE="${DEBUG_FILE:-${BERT_PRED_DIR}/2016/sv_summary_llama_full_bert_2016-08.csv.gz}"
DEBUG_MODEL="${DEBUG_MODEL:-bert}"

LOG_DIR="${LOG_DIR:-${BASE_DATA_DIR}/command_outputs/logs}"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/eval_gemma_qwen_${MODE}_${TIMESTAMP}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

run_judge() {
  local judge_name="$1"
  local judge_ckpt="$2"
  local output_root="$3"

  echo
  echo "======================================"
  echo "[${MODE}] ${judge_name}: ${judge_ckpt}"
  echo "Output root: ${output_root}"
  echo "======================================"

  local common_args=(
    --judge_ckpt "${judge_ckpt}"
    --judge-name "${judge_name}"
    --output-root "${output_root}"
    --anchor-search-radius "${ANCHOR_SEARCH_RADIUS}"
    --sent-batch "${SENT_BATCH}"
  )

  local args=("${common_args[@]}")
  if [[ "${MODE}" == "debug" ]]; then
    args+=(
      --debug-file "${DEBUG_FILE}"
      --debug-model "${DEBUG_MODEL}"
      --debug-n "${DEBUG_N}"
    )
    CUDA_VISIBLE_DEVICES="${GPU_ID_LIST[0]}" "${PYTHON_BIN}" "${VALID_SCRIPT}" "${args[@]}"
    return
  fi

  args+=(
    --bert_pred_dir "${BERT_PRED_DIR}"
    --w2v_pred_dir "${W2V_PRED_DIR}"
    --cond_pred_dir "${COND_PRED_DIR}"
    --resume
  )

  if (( NUM_WORKERS <= 1 )); then
    CUDA_VISIBLE_DEVICES="${GPU_ID_LIST[0]}" "${PYTHON_BIN}" "${VALID_SCRIPT}" "${args[@]}"
    return
  fi

  local pids=()
  local rank
  for (( rank = 0; rank < NUM_WORKERS; rank++ )); do
    local gpu_id="${GPU_ID_LIST[rank]}"
    echo "[LAUNCH] ${judge_name} worker=${rank}/${NUM_WORKERS} gpu=${gpu_id}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" "${VALID_SCRIPT}" \
      "${args[@]}" \
      --task-rank "${rank}" \
      --task-world-size "${NUM_WORKERS}" &
    pids+=("$!")
  done

  local status=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done

  if (( status != 0 )); then
    echo "[ERROR] At least one ${judge_name} worker failed."
    exit "${status}"
  fi
}

echo "======================================"
echo "[START] Qwen2.5 + Gemma validation"
echo "Mode     : ${MODE}"
echo "GPU IDs  : ${GPU_IDS}"
echo "Workers  : ${NUM_WORKERS}"
echo "Log file : ${LOG_FILE}"
echo "======================================"

echo "BERT dir : ${BERT_PRED_DIR}"
echo "W2V dir  : ${W2V_PRED_DIR}"
echo "COND dir : ${COND_PRED_DIR}"

if (( NUM_WORKERS < 1 || NUM_WORKERS > ${#GPU_ID_LIST[@]} )); then
  echo "NUM_WORKERS must be between 1 and the number of GPU_IDS (${#GPU_ID_LIST[@]})."
  exit 1
fi

if [[ "${MODE}" == "debug" ]]; then
  echo "Debug file : ${DEBUG_FILE}"
  echo "Debug model: ${DEBUG_MODEL}"
  echo "Debug rows : ${DEBUG_N}"
fi

run_judge "qwen25" "${QWEN_CKPT}" "${QWEN_OUTPUT_ROOT}"
run_judge "gemma" "${GEMMA_CKPT}" "${GEMMA_OUTPUT_ROOT}"

if [[ "${MODE}" == "full" ]]; then
  MEAN_OUTPUT_CSV="${MEAN_OUTPUT_CSV:-${BASE_DATA_DIR}/command_outputs/mean_summary_qwen25_gemma_${TIMESTAMP}.csv}"
  mkdir -p "$(dirname "${MEAN_OUTPUT_CSV}")"
  echo
  echo "======================================"
  echo "[MEAN] Summarize qwen25/gemma outputs"
  echo "Mean output: ${MEAN_OUTPUT_CSV}"
  echo "======================================"
  "${PYTHON_BIN}" "${MEAN_SCRIPT}" \
    --qwen-root "${QWEN_OUTPUT_ROOT}" \
    --qwen-judge "qwen25" \
    --gemma-root "${GEMMA_OUTPUT_ROOT}" \
    --gemma-judge "gemma" \
    --output-csv "${MEAN_OUTPUT_CSV}"
fi

echo
echo "======================================"
echo "[DONE] ${MODE} run completed successfully."
echo "======================================"
