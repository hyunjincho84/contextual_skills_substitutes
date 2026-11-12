#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ========================== CONFIG ==========================
# Edit the values below before running this script.
PYTHON_BIN="/home/jovyan/LEM_data2/hyunjincho/env/lem/bin/python"
BASE_DATA_DIR="/home/jovyan/LEM_data3/hyunjincho/LEM"
DATA_ROOT="${BASE_DATA_DIR}/all_train_test"
TEST_ROOT="${DATA_ROOT}/test"
VOCAB_PATH="${BASE_DATA_DIR}/all_filterd/skill2idx.json"
INPUT_PATTERN="preprocessed_*.csv.gz"

BERT_MODEL_NAME="${BASE_DATA_DIR}/bert_pretrained"
BERT_CHECKPOINT_PATH="${BASE_DATA_DIR}/models/bert_fine_tuned_model/epoch_3.pt"
BERT_OUTPUT_DIR="${BASE_DATA_DIR}/predictions/bert_pred"
BERT_MAX_LENGTH=256
BERT_BATCH_SIZE=128
BERT_TOP_K=10
BERT_NUM_WORKERS=8
BERT_COMPRESSION_LEVEL=1
BERT_USE_AMP="true"
BERT_PREFETCH_FACTOR=4
BERT_STRICT="true"
BERT_OVERWRITE="false"
BERT_DEVICE="auto"

SKILL2VEC_MODEL_PATH="${BASE_DATA_DIR}/models/skill2vec_model/skill2vec_norm_sg1_d300_win50_neg10_ep5.model"
SKILL2VEC_TOKEN_MAP_PATH="${BASE_DATA_DIR}/models/skill2vec_model/skill2vec_norm_sg1_d300_win50_neg10_ep5.skill_tokens.json"
SKILL2VEC_OUTPUT_DIR="${BASE_DATA_DIR}/predictions/skill2vec_pred"
SKILL2VEC_CHUNK_SIZE=200000
SKILL2VEC_TOP_K=5
SKILL2VEC_WORKERS=8
SKILL2VEC_PROGRESS_EVERY=100
SKILL2VEC_COMPRESSION_LEVEL=1
SKILL2VEC_MAX_PRINT_PER_REASON=10
SKILL2VEC_JOINER="_"
SKILL2VEC_OVERWRITE="false"

CONDPROB_COOC_PROBS="${BASE_DATA_DIR}/models/condprob_model/cooc_probs.csv.gz"
CONDPROB_OUTPUT_DIR="${BASE_DATA_DIR}/predictions/condprob_pred"
CONDPROB_CHUNK_SIZE=200000
CONDPROB_TOP_K=5
CONDPROB_WORKERS=8
CONDPROB_PROGRESS_EVERY=25
CONDPROB_COMPRESSION_LEVEL=1
CONDPROB_MAX_PRINT_PER_REASON=10
CONDPROB_FORCE="false"
# ============================================================

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
#  - ${BASE_DATA_DIR}/all_train_test/test
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
"${PYTHON_BIN}" "${REPO_ROOT}/model_tests/test_bert.py" \
  --data-root "${DATA_ROOT}" \
  --test-dir "${TEST_ROOT}" \
  --vocab-path "${VOCAB_PATH}" \
  --pattern "${INPUT_PATTERN}" \
  --model-name "${BERT_MODEL_NAME}" \
  --checkpoint-path "${BERT_CHECKPOINT_PATH}" \
  --output-dir "${BERT_OUTPUT_DIR}" \
  --max-length "${BERT_MAX_LENGTH}" \
  --batch-size "${BERT_BATCH_SIZE}" \
  --top-k "${BERT_TOP_K}" \
  --num-workers "${BERT_NUM_WORKERS}" \
  --compression-level "${BERT_COMPRESSION_LEVEL}" \
  --use-amp "${BERT_USE_AMP}" \
  --prefetch-factor "${BERT_PREFETCH_FACTOR}" \
  --strict "${BERT_STRICT}" \
  --overwrite "${BERT_OVERWRITE}" \
  --device "${BERT_DEVICE}"

step "[2/3] Skill2Vec"
"${PYTHON_BIN}" "${REPO_ROOT}/model_tests/test_skill2vec.py" \
  --model-path "${SKILL2VEC_MODEL_PATH}" \
  --token-map-path "${SKILL2VEC_TOKEN_MAP_PATH}" \
  --test-root "${TEST_ROOT}" \
  --output-dir "${SKILL2VEC_OUTPUT_DIR}" \
  --pattern "${INPUT_PATTERN}" \
  --chunk-size "${SKILL2VEC_CHUNK_SIZE}" \
  --top-k "${SKILL2VEC_TOP_K}" \
  --workers "${SKILL2VEC_WORKERS}" \
  --progress-every "${SKILL2VEC_PROGRESS_EVERY}" \
  --compression-level "${SKILL2VEC_COMPRESSION_LEVEL}" \
  --max-print-per-reason "${SKILL2VEC_MAX_PRINT_PER_REASON}" \
  --joiner "${SKILL2VEC_JOINER}" \
  --overwrite "${SKILL2VEC_OVERWRITE}"

step "[3/3] Conditional Probability"
CONDPROB_ARGS=(
  "${PYTHON_BIN}" "${REPO_ROOT}/model_tests/test_conditional_prob.py"
  --cooc-probs "${CONDPROB_COOC_PROBS}" \
  --test-root "${TEST_ROOT}" \
  --output-dir "${CONDPROB_OUTPUT_DIR}" \
  --pattern "${INPUT_PATTERN}" \
  --chunk-size "${CONDPROB_CHUNK_SIZE}" \
  --top-k "${CONDPROB_TOP_K}" \
  --workers "${CONDPROB_WORKERS}" \
  --progress-every "${CONDPROB_PROGRESS_EVERY}" \
  --compression-level "${CONDPROB_COMPRESSION_LEVEL}" \
  --max-print-per-reason "${CONDPROB_MAX_PRINT_PER_REASON}"
)
if [[ "${CONDPROB_FORCE}" == "true" ]]; then
  CONDPROB_ARGS+=(--force)
fi
"${CONDPROB_ARGS[@]}"

section "PIPELINE COMPLETED SUCCESSFULLY"
ts "Baseline predictions finished"
