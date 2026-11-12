#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DEFAULT_MODE="${DEFAULT_MODE:-full}"
MODE="${1:-${DEFAULT_MODE}}"
if [[ "${MODE}" != "full" && "${MODE}" != "debug" ]]; then
  echo "Usage: $0 [full|debug]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ============================================================
# TOP-2 ROBUSTNESS PIPELINE
#
# Purpose:
#  - Test robustness of substitutes by sampling alternatives from BERT top-k predictions.
#  - Score sampled variants with LLaMA SV scoring and summarize the mean scores.
#
# Runs:
#  1) model_tests/top2_robustness/pred_bert_top10.py
#     - Generates BERT top-10 prediction summaries.
#  2) model_tests/top2_robustness/sampling.py
#     - Builds sampled substitute variants from top predictions.
#  3) model_tests/top2_robustness/scoring_samplings.py
#     - Scores sampled variants with LLaMA SV scoring.
#  4) model_tests/top2_robustness/get_mean.py
#     - Summarizes mean SV scores for sampled variants.
#
# Input:
#  - Prediction/source data paths and hyperparameters are configured below.
#  - Existing sampling outputs are detected with SAMPLING_OUTPUT_GLOB.
#  - Default glob: ${BASE_DATA_DIR}/bert_sampling_variants_from_summary/*/sv_summary_llama_full_bert_top10_*_sampling.csv.gz
#  - MODE: full or debug.
#
# Output:
#  - Top-10 prediction summaries under ${BASE_DATA_DIR}/bert_pred_new_top10_from_summary.
#  - Sampling CSV.GZ files under ${BASE_DATA_DIR}/bert_sampling_variants_from_summary.
#  - Scored sampling files and mean summaries from scoring_samplings.py/get_mean.py.
#  - Log file: ${BASE_DATA_DIR}/command_outputs/logs/top2_pipeline_${MODE}_${TIMESTAMP}.log
# ============================================================

TOP2_DIR="${REPO_ROOT}/model_tests/top2_robustness"
PRED_SCRIPT="${TOP2_DIR}/pred_bert_top10.py"
SAMPLING_SCRIPT="${TOP2_DIR}/sampling.py"
SCORING_SCRIPT="${TOP2_DIR}/scoring_samplings.py"
GET_MEAN_SCRIPT="${TOP2_DIR}/get_mean.py"

# ======================== EDITABLE CONFIG ========================
PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_DATA_DIR="${BASE_DATA_DIR:-/home/jovyan/LEM_data2/data}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/home/jovyan/LEM_data2/hf_cache}"
export BASE_DATA_DIR HF_CACHE_DIR
FALLBACK_PYTHON="${FALLBACK_PYTHON:-/home/jovyan/.venv/torch2.3.0-py3.11-cuda12.1/bin/python}"

if ! "${PYTHON_BIN}" -c "import torch, transformers" >/dev/null 2>&1; then
  if [[ -x "${FALLBACK_PYTHON}" ]] && "${FALLBACK_PYTHON}" -c "import torch, transformers" >/dev/null 2>&1; then
    echo "[WARN] ${PYTHON_BIN} cannot import torch/transformers; using fallback: ${FALLBACK_PYTHON}"
    PYTHON_BIN="${FALLBACK_PYTHON}"
  else
    echo "[ERROR] ${PYTHON_BIN} cannot import torch/transformers."
    echo "Install them in that environment or run with PYTHON_BIN=/path/to/python."
    exit 1
  fi
fi
GPU_IDS="${GPU_IDS:-auto}"
SENT_BATCH="${SENT_BATCH:-4}"
SAMPLE_PER_FILE="${SAMPLE_PER_FILE:-all}"
DEBUG_MONTHS="${DEBUG_MONTHS:-1}"
DEBUG_ROWS="${DEBUG_ROWS:-10}"
SCORING_RESUME="${SCORING_RESUME:-1}"
FORCE_STAGE12="${FORCE_STAGE12:-0}"

DATA_ROOT="${DATA_ROOT:-${BASE_DATA_DIR}/preprocessed_www_new}"
VOCAB_PATH="${VOCAB_PATH:-${DATA_ROOT}/skill2idx.json}"
TOP2_INPUT_GLOB="${TOP2_INPUT_GLOB:-${BASE_DATA_DIR}/bert_pred_new/pred/*/sv_summary_llama_full_bert_*.csv.gz}"
BERT_MODEL_NAME="${BERT_MODEL_NAME:-${BASE_DATA_DIR}/bert_pretrained}"
BEST_MODEL_PT="${BEST_MODEL_PT:-${BASE_DATA_DIR}/checkpoints(www)_new/best_model.pt}"
TOP10_OUT_DIR="${TOP10_OUT_DIR:-${BASE_DATA_DIR}/bert_pred_new_top10_from_summary}"
PRED_GPUS="${PRED_GPUS:-${GPU_IDS}}"
PRED_BATCH_SIZE="${PRED_BATCH_SIZE:-64}"
PRED_NUM_WORKERS="${PRED_NUM_WORKERS:-4}"
PRED_CHUNKSIZE="${PRED_CHUNKSIZE:-200000}"
PRED_MAX_LEN="${PRED_MAX_LEN:-512}"
PRED_NO_PARALLEL="${PRED_NO_PARALLEL:-0}"
PRED_RESUME="${PRED_RESUME:-1}"

SAMPLING_INPUT_GLOB="${SAMPLING_INPUT_GLOB:-${TOP10_OUT_DIR}/pred/*/sv_summary_llama_full_bert_top10_*.csv.gz}"
SAMPLING_OUTPUT_ROOT="${SAMPLING_OUTPUT_ROOT:-${BASE_DATA_DIR}/bert_sampling_variants_from_summary}"
SAMPLING_OUTPUT_GLOB="${SAMPLING_OUTPUT_GLOB:-${SAMPLING_OUTPUT_ROOT}/*/sv_summary_llama_full_bert_top10_*_sampling.csv.gz}"
TOP10_COL="${TOP10_COL:-bert_top10}"
TOP10_PROBS_COL="${TOP10_PROBS_COL:-bert_top10_probs}"
SAMPLING_RANDOM_SEED="${SAMPLING_RANDOM_SEED:-42}"
THRESHOLD_0005="${THRESHOLD_0005:-0.005}"
THRESHOLD_0001="${THRESHOLD_0001:-0.001}"
TEMPERATURE_15="${TEMPERATURE_15:-1.5}"
TEMPERATURE_20="${TEMPERATURE_20:-2.0}"

SCORING_INPUT_GLOB="${SCORING_INPUT_GLOB:-${SAMPLING_OUTPUT_GLOB}}"
LLAMA_CKPT="${LLAMA_CKPT:-meta-llama/Llama-3.2-3B}"
WINDOW_SIZE="${WINDOW_SIZE:-256}"
MAX_LEN="${MAX_LEN:-512}"
SCORING_RANDOM_SEED="${SCORING_RANDOM_SEED:-42}"
ANCHOR_SEARCH_RADIUS="${ANCHOR_SEARCH_RADIUS:-2}"
SAMPLING_METHODS="${SAMPLING_METHODS:-sample_top2_like sample_top10 sample_thresh_0005_sampling sample_thresh_0001_sampling sample_temp_15 sample_temp_20}"
SCORING_NO_PARALLEL="${SCORING_NO_PARALLEL:-0}"
DEBUG_FAIL_REASON="${DEBUG_FAIL_REASON:-0}"

MEAN_INPUT_GLOB="${MEAN_INPUT_GLOB:-${SAMPLING_OUTPUT_ROOT}/*/sv_summary_llama_all_methods_*.csv.gz}"
MEAN_METHODS="${MEAN_METHODS:-${SAMPLING_METHODS}}"
MEAN_BY_FILE="${MEAN_BY_FILE:-0}"
LOG_FILE="${LOG_FILE:-}"
LOG_DIR="${LOG_DIR:-${BASE_DATA_DIR}/command_outputs/logs}"
# ================================================================
read -r -a SAMPLING_METHOD_ARGS <<< "${SAMPLING_METHODS}"
read -r -a MEAN_METHOD_ARGS <<< "${MEAN_METHODS}"
for bool_value in "${FORCE_STAGE12}" "${PRED_NO_PARALLEL}" "${PRED_RESUME}" "${SCORING_RESUME}" "${SCORING_NO_PARALLEL}" "${DEBUG_FAIL_REASON}" "${MEAN_BY_FILE}"; do
  if [[ ! "${bool_value}" =~ ^[01]$ ]]; then
    echo "Boolean CONFIG values must be 0 or 1."
    exit 1
  fi
done

mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/top2_pipeline_${MODE}_${TIMESTAMP}.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "======================================"
echo "Top2 robustness pipeline"
echo "Mode: ${MODE}"
echo "Python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "GPUs: ${GPU_IDS}"
echo "Log: ${LOG_FILE}"
echo "======================================"

echo
echo "[0/4] Checking completed stages"
shopt -s nullglob
SAMPLING_OUTPUT_FILES=(${SAMPLING_OUTPUT_GLOB})
shopt -u nullglob
if [[ "${FORCE_STAGE12}" != "1" && ${#SAMPLING_OUTPUT_FILES[@]} -gt 0 ]]; then
  echo "[SKIP] Found ${#SAMPLING_OUTPUT_FILES[@]} sampling output files."
  echo "[SKIP] 1/4 and 2/4 already completed; starting from 3/4."
else
  echo
  echo "[1/4] Running BERT top-10 prediction"
  PRED_ARGS=(
    --input-glob "${TOP2_INPUT_GLOB}"
    --model-name "${BERT_MODEL_NAME}"
    --vocab-path "${VOCAB_PATH}"
    --checkpoint "${BEST_MODEL_PT}"
    --output-dir "${TOP10_OUT_DIR}"
    --max-len "${PRED_MAX_LEN}"
    --gpus "${PRED_GPUS}"
    --batch-size "${PRED_BATCH_SIZE}"
    --num-workers "${PRED_NUM_WORKERS}"
    --chunksize "${PRED_CHUNKSIZE}"
  )
  if [[ "${PRED_NO_PARALLEL}" == "1" ]]; then
    PRED_ARGS+=(--no-parallel)
  fi
  if [[ "${PRED_RESUME}" == "1" ]]; then
    PRED_ARGS+=(--resume)
  fi
  "${PYTHON_BIN}" "${PRED_SCRIPT}" "${PRED_ARGS[@]}"

  echo
  echo "[2/4] Running sampling variants"
  SAMPLING_ARGS=(
    --input-glob "${SAMPLING_INPUT_GLOB}"
    --output-root "${SAMPLING_OUTPUT_ROOT}"
    --top10-col "${TOP10_COL}"
    --top10-probs-col "${TOP10_PROBS_COL}"
    --random-seed "${SAMPLING_RANDOM_SEED}"
    --threshold-0005 "${THRESHOLD_0005}"
    --threshold-0001 "${THRESHOLD_0001}"
    --temperature-15 "${TEMPERATURE_15}"
    --temperature-20 "${TEMPERATURE_20}"
  )
  "${PYTHON_BIN}" "${SAMPLING_SCRIPT}" "${SAMPLING_ARGS[@]}"
fi

echo
echo "[3/4] Running LLaMA SV scoring"
SCORING_ARGS=(
  --input-glob "${SCORING_INPUT_GLOB}"
  --llama-ckpt "${LLAMA_CKPT}"
  --cache-dir "${HF_CACHE_DIR}"
  --window-size "${WINDOW_SIZE}"
  --max-len "${MAX_LEN}"
  --gpus "${GPU_IDS}"
  --sent-batch "${SENT_BATCH}"
  --sample-per-file "${SAMPLE_PER_FILE}"
  --random-seed "${SCORING_RANDOM_SEED}"
  --anchor-search-radius "${ANCHOR_SEARCH_RADIUS}"
  --methods "${SAMPLING_METHOD_ARGS[@]}"
)

if [[ "${SCORING_RESUME}" == "1" ]]; then
  SCORING_ARGS+=(--resume)
fi
if [[ "${SCORING_NO_PARALLEL}" == "1" ]]; then
  SCORING_ARGS+=(--no-parallel)
fi
if [[ "${DEBUG_FAIL_REASON}" == "1" ]]; then
  SCORING_ARGS+=(--debug-fail-reason)
fi

if [[ "${MODE}" == "debug" ]]; then
  SCORING_ARGS+=(
    --debug
    --debug-months "${DEBUG_MONTHS}"
    --debug-rows "${DEBUG_ROWS}"
  )
fi

"${PYTHON_BIN}" "${SCORING_SCRIPT}" "${SCORING_ARGS[@]}"

echo
echo "[4/4] Summarizing mean SV scores"
MEAN_ARGS=(
  --input-glob "${MEAN_INPUT_GLOB}"
  --methods "${MEAN_METHOD_ARGS[@]}"
)
if [[ "${MEAN_BY_FILE}" == "1" ]]; then
  MEAN_ARGS+=(--by-file)
fi
"${PYTHON_BIN}" "${GET_MEAN_SCRIPT}" "${MEAN_ARGS[@]}"

echo
echo "[DONE] Pipeline finished."
echo "Log: ${LOG_FILE}"
