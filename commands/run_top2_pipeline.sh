#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODE="${1:-full}"
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
#  - Prediction/source data paths are configured inside the Python scripts.
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

PYTHON_BIN="${PYTHON_BIN:-python3}"
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
RESUME_FLAG="${RESUME_FLAG:---resume}"
FORCE_STAGE12="${FORCE_STAGE12:-0}"
BASE_DATA_DIR="${BASE_DATA_DIR:-/home/jovyan/LEM_data2/data}"
export BASE_DATA_DIR
SAMPLING_OUTPUT_GLOB="${SAMPLING_OUTPUT_GLOB:-${BASE_DATA_DIR}/bert_sampling_variants_from_summary/*/sv_summary_llama_full_bert_top10_*_sampling.csv.gz}"

LOG_DIR="${LOG_DIR:-${BASE_DATA_DIR}/command_outputs/logs}"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/top2_pipeline_${MODE}_${TIMESTAMP}.log"
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
  "${PYTHON_BIN}" "${PRED_SCRIPT}" --resume

  echo
  echo "[2/4] Running sampling variants"
  "${PYTHON_BIN}" "${SAMPLING_SCRIPT}"
fi

echo
echo "[3/4] Running LLaMA SV scoring"
SCORING_ARGS=(
  --gpus "${GPU_IDS}"
  --sent-batch "${SENT_BATCH}"
  --sample-per-file "${SAMPLE_PER_FILE}"
)

if [[ -n "${RESUME_FLAG}" ]]; then
  SCORING_ARGS+=("${RESUME_FLAG}")
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
"${PYTHON_BIN}" "${GET_MEAN_SCRIPT}"

echo
echo "[DONE] Pipeline finished."
echo "Log: ${LOG_FILE}"
