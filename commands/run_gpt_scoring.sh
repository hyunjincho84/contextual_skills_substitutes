#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# EDIT PARAMETERS HERE
# =============================================================================

# Runtime
PYTHON_BIN="/home/jovyan/LEM_data2/hyunjincho/env/lem/bin/python"
CUDA_VISIBLE_DEVICES="0"
DEVICE="cuda"

# Run stages: 1 = run, 0 = skip
RUN_BERT_SCORING="1"
RUN_OTHER_MODEL_SCORING="1"
RUN_MEAN_SUMMARY="1"

# Data paths
BASE_DATA_DIR="/home/jovyan/LEM_data3/hyunjincho/LEM"
SAMPLE_ROOT="${BASE_DATA_DIR}/predictions_clean/gpt_samplings"
GPT_INPUT="${SAMPLE_ROOT}/gpt.csv"
BERT_INPUT="${SAMPLE_ROOT}/bert.csv"
BERT_REFERENCE_ROOT="${BASE_DATA_DIR}/evaluation_sv/bert_clean/bert"
OTHER_MODELS_OUTPUT_ROOT="${BASE_DATA_DIR}/evaluation_sv/gpt_other_models"
MEAN_OUTPUT_CSV="${OTHER_MODELS_OUTPUT_ROOT}/gpt_bert_mean_summary_raw.csv"

# Optional filter. Leave empty for all years, or use: "2018 2019"
YEARS=""

# Shared scoring parameters
WINDOW_SIZE="128"
MAX_LEN="320"
RANDOM_SEED="42"

# Step 1: BERT judge parameters
BERT_MODEL="bert-base-uncased"
BERT_CACHE_DIR="/home/jovyan/LEM_data2/hf_cache/transformers"
BERT_SENT_BATCH="128"
BERT_CHUNK_SIZE="8000"
BERT_RESUME="0"
BERT_STRICT="0"
BERT_LOCAL_FILES_ONLY="1"
BERT_SKIP_CONTROL="0"
BERT_SKIP_REFERENCE_CHECK="0"
BERT_REQUIRE_REFERENCE_MATCH="0"
BERT_SCORE_ATOL="5e-4"
BERT_SCORE_RTOL="1e-5"

# Step 2: LLaMA/Gemma/Qwen judge parameters
# OTHER_JUDGES accepts "all" or comma-separated values such as "llama,gemma,qwen".
OTHER_JUDGES="all"
LLAMA_CHECKPOINT="meta-llama/Llama-3.2-3B"
GEMMA_CHECKPOINT="google/gemma-2-2b-it"
QWEN_CHECKPOINT="Qwen/Qwen2.5-3B-Instruct"
OTHER_MODELS_CACHE_DIR="/home/jovyan/LEM_data2/hf_cache"
OTHER_SENT_BATCH="64"
# 8000 is divisible by OTHER_SENT_BATCH=64, avoiding partial chunk-boundary batches.
OTHER_CHUNK_SIZE="8000"
OTHER_RESUME="1"
OTHER_STRICT="0"
OTHER_LOCAL_FILES_ONLY="0"
OTHER_USE_AUTH_TOKEN="1"

# Step 3: mean-summary parameters
# Space-separated judge names chosen from: llama gemma qwen bert
MEAN_JUDGES="llama gemma qwen bert"

# =============================================================================
# PIPELINE IMPLEMENTATION
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export BASE_DATA_DIR CUDA_VISIBLE_DEVICES
cd "${REPO_ROOT}"

BOOLEAN_PARAMETERS=(
  RUN_BERT_SCORING
  RUN_OTHER_MODEL_SCORING
  RUN_MEAN_SUMMARY
  BERT_RESUME
  BERT_STRICT
  BERT_LOCAL_FILES_ONLY
  BERT_SKIP_CONTROL
  BERT_SKIP_REFERENCE_CHECK
  BERT_REQUIRE_REFERENCE_MATCH
  OTHER_RESUME
  OTHER_STRICT
  OTHER_LOCAL_FILES_ONLY
  OTHER_USE_AUTH_TOKEN
)
for parameter_name in "${BOOLEAN_PARAMETERS[@]}"; do
  parameter_value="${!parameter_name}"
  if [[ ! "${parameter_value}" =~ ^[01]$ ]]; then
    echo "ERROR: ${parameter_name} must be 0 or 1; got ${parameter_value}." >&2
    exit 2
  fi
done

YEAR_ARGS=()
if [[ -n "${YEARS}" ]]; then
  read -r -a YEAR_ARGS <<< "${YEARS}"
fi

MEAN_JUDGE_ARGS=()
if [[ -n "${MEAN_JUDGES}" ]]; then
  read -r -a MEAN_JUDGE_ARGS <<< "${MEAN_JUDGES}"
fi
if [[ "${RUN_MEAN_SUMMARY}" == "1" && ${#MEAN_JUDGE_ARGS[@]} -eq 0 ]]; then
  echo "ERROR: MEAN_JUDGES must not be empty when RUN_MEAN_SUMMARY=1." >&2
  exit 2
fi

print_stage() {
  echo
  echo "=================================================================="
  echo "$1"
  echo "=================================================================="
}

print_stage "GPT/BERT semantic-validation pipeline configuration"
echo "Python                 : ${PYTHON_BIN}"
echo "CUDA_VISIBLE_DEVICES   : ${CUDA_VISIBLE_DEVICES}"
echo "Device                 : ${DEVICE}"
echo "GPT input              : ${GPT_INPUT}"
echo "BERT input             : ${BERT_INPUT}"
echo "Years                  : ${YEARS:-all}"
echo "Window / max length    : ${WINDOW_SIZE} / ${MAX_LEN}"
echo "Run BERT judge         : ${RUN_BERT_SCORING}"
echo "Run other-model judges : ${RUN_OTHER_MODEL_SCORING} (${OTHER_JUDGES})"
echo "Run mean summary       : ${RUN_MEAN_SUMMARY} (${MEAN_JUDGES})"

if [[ "${RUN_BERT_SCORING}" == "1" ]]; then
  print_stage "[1/3] Score GPT and BERT samples with the BERT judge"
  BERT_CMD=(
    "${PYTHON_BIN}"
    "${REPO_ROOT}/model_tests/likelihood_based_score/scoring_gpt.py"
    --gpt-input "${GPT_INPUT}"
    --bert-input "${BERT_INPUT}"
    --reference-root "${BERT_REFERENCE_ROOT}"
    --model-name "${BERT_MODEL}"
    --cache-dir "${BERT_CACHE_DIR}"
    --device "${DEVICE}"
    --window-size "${WINDOW_SIZE}"
    --max-len "${MAX_LEN}"
    --sent-batch "${BERT_SENT_BATCH}"
    --chunk-size "${BERT_CHUNK_SIZE}"
    --random-seed "${RANDOM_SEED}"
    --score-atol "${BERT_SCORE_ATOL}"
    --score-rtol "${BERT_SCORE_RTOL}"
  )
  if [[ ${#YEAR_ARGS[@]} -gt 0 ]]; then
    BERT_CMD+=(--years "${YEAR_ARGS[@]}")
  fi
  if [[ "${BERT_RESUME}" == "1" ]]; then BERT_CMD+=(--resume); fi
  if [[ "${BERT_STRICT}" == "1" ]]; then BERT_CMD+=(--strict); fi
  if [[ "${BERT_LOCAL_FILES_ONLY}" == "1" ]]; then
    BERT_CMD+=(--local-files-only)
  fi
  if [[ "${BERT_SKIP_CONTROL}" == "1" ]]; then
    BERT_CMD+=(--skip-bert-control)
  fi
  if [[ "${BERT_SKIP_REFERENCE_CHECK}" == "1" ]]; then
    BERT_CMD+=(--skip-reference-check)
  fi
  if [[ "${BERT_REQUIRE_REFERENCE_MATCH}" == "1" ]]; then
    BERT_CMD+=(--require-reference-match)
  fi
  "${BERT_CMD[@]}"
else
  print_stage "[1/3] BERT judge scoring skipped"
fi

if [[ "${RUN_OTHER_MODEL_SCORING}" == "1" ]]; then
  print_stage "[2/3] Score GPT and BERT samples with LLaMA/Gemma/Qwen"
  OTHER_CMD=(
    "${PYTHON_BIN}"
    "${REPO_ROOT}/model_tests/likelihood_based_score/scoring_gpt_other_models.py"
    --gpt-input "${GPT_INPUT}"
    --bert-input "${BERT_INPUT}"
    --output-root "${OTHER_MODELS_OUTPUT_ROOT}"
    --judges "${OTHER_JUDGES}"
    --llama-ckpt "${LLAMA_CHECKPOINT}"
    --gemma-ckpt "${GEMMA_CHECKPOINT}"
    --qwen-ckpt "${QWEN_CHECKPOINT}"
    --cache-dir "${OTHER_MODELS_CACHE_DIR}"
    --device "${DEVICE}"
    --window-size "${WINDOW_SIZE}"
    --max-len "${MAX_LEN}"
    --sent-batch "${OTHER_SENT_BATCH}"
    --chunk-size "${OTHER_CHUNK_SIZE}"
    --random-seed "${RANDOM_SEED}"
  )
  if [[ ${#YEAR_ARGS[@]} -gt 0 ]]; then
    OTHER_CMD+=(--years "${YEAR_ARGS[@]}")
  fi
  if [[ "${OTHER_RESUME}" == "1" ]]; then OTHER_CMD+=(--resume); fi
  if [[ "${OTHER_STRICT}" == "1" ]]; then OTHER_CMD+=(--strict); fi
  if [[ "${OTHER_LOCAL_FILES_ONLY}" == "1" ]]; then
    OTHER_CMD+=(--local-files-only)
  fi
  if [[ "${OTHER_USE_AUTH_TOKEN}" == "1" ]]; then
    OTHER_CMD+=(--use-auth-token)
  else
    OTHER_CMD+=(--no-auth-token)
  fi
  "${OTHER_CMD[@]}"
else
  print_stage "[2/3] Other-model judge scoring skipped"
fi

if [[ "${RUN_MEAN_SUMMARY}" == "1" ]]; then
  print_stage "[3/3] Compute paired BERT/GPT raw-score summaries"
  MEAN_CMD=(
    "${PYTHON_BIN}"
    "${REPO_ROOT}/model_tests/likelihood_based_score/compute_gpt_score_mean.py"
    --judges "${MEAN_JUDGE_ARGS[@]}"
    --sample-root "${SAMPLE_ROOT}"
    --other-models-root "${OTHER_MODELS_OUTPUT_ROOT}"
    --output-csv "${MEAN_OUTPUT_CSV}"
  )
  if [[ ${#YEAR_ARGS[@]} -gt 0 ]]; then
    MEAN_CMD+=(--years "${YEAR_ARGS[@]}")
  fi
  "${MEAN_CMD[@]}"
else
  print_stage "[3/3] Mean summary skipped"
fi

print_stage "Pipeline completed successfully"
echo "Mean summary: ${MEAN_OUTPUT_CSV}"
