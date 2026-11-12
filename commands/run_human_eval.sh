#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ============================================================
# CONFIG — enter values directly between the quotation marks
# ============================================================
PYTHON_BIN="/home/jovyan/LEM_data2/hyunjincho/env/lem/bin/python"
BASE_DATA_DIR="/home/jovyan/LEM_data3/hyunjincho/LEM"

# Input data and vocabulary
TEST_ROOT="${BASE_DATA_DIR}/all_train_test/test"
INPUT_GLOB="${TEST_ROOT}/20*/20*/*.csv.gz"
SKILL_JSON="${BASE_DATA_DIR}/all_filterd/skill2idx.json"

# Existing BERT predictions
BERT_PREDICTION_GLOB="${BASE_DATA_DIR}/predictions/bert_pred/pred/20*/predictions_*.csv.gz"

# Skill2Vec and conditional-probability models
SKILL2VEC_MODEL="${BASE_DATA_DIR}/models/skill2vec_model/skill2vec_norm_sg1_d300_win50_neg10_ep5.kv"
SKILL2VEC_TOKEN_MAP="${BASE_DATA_DIR}/models/skill2vec_model/skill2vec_norm_sg1_d300_win50_neg10_ep5.skill_tokens.json"
CONDITIONAL_TABLE="${BASE_DATA_DIR}/models/condprob_model/cooc_probs.csv.gz"

# Output
OUTPUT_CSV="${BASE_DATA_DIR}/human_eval/skill_or_skill_shuffled.csv.gz"

# Runtime parameters
CHUNK_SIZE="200000"
SEED="42"
YEARS=""       # Example: "2020 2021"; empty = all years
MAX_FILES=""   # Example: "50" for a smoke test; empty = all files
OVERWRITE="true"
# ============================================================

case "${OVERWRITE,,}" in
  true|1|yes|y) OVERWRITE_ENABLED=true ;;
  false|0|no|n) OVERWRITE_ENABLED=false ;;
  *)
    echo "[ERROR] OVERWRITE must be true or false: ${OVERWRITE}" >&2
    exit 2
    ;;
esac

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

CMD=(
  "${PYTHON_BIN}"
  "${REPO_ROOT}/model_tests/human_eval/extract_skill_or_skill.py"
  --test-root "${TEST_ROOT}"
  --input-glob "${INPUT_GLOB}"
  --skill-json "${SKILL_JSON}"
  --bert-prediction-glob "${BERT_PREDICTION_GLOB}"
  --skill2vec-model "${SKILL2VEC_MODEL}"
  --skill2vec-token-map "${SKILL2VEC_TOKEN_MAP}"
  --conditional-table "${CONDITIONAL_TABLE}"
  --output "${OUTPUT_CSV}"
  --chunk-size "${CHUNK_SIZE}"
  --seed "${SEED}"
)

if [[ -n "${YEARS}" ]]; then
  read -r -a YEAR_ARGS <<< "${YEARS}"
  CMD+=(--years "${YEAR_ARGS[@]}")
fi

if [[ -n "${MAX_FILES}" ]]; then
  CMD+=(--max-files "${MAX_FILES}")
fi

if [[ "${OVERWRITE_ENABLED}" == true ]]; then
  CMD+=(--overwrite)
fi

echo "============================================================"
echo "HUMAN EVALUATION DATASET PIPELINE"
echo "============================================================"
echo "Input             : ${INPUT_GLOB}"
echo "Years             : ${YEARS:-all}"
echo "Max files         : ${MAX_FILES:-all}"
echo "BERT predictions  : ${BERT_PREDICTION_GLOB}"
echo "Skill2Vec model   : ${SKILL2VEC_MODEL}"
echo "Conditional table : ${CONDITIONAL_TABLE}"
echo "Output            : ${OUTPUT_CSV}"
echo "Seed              : ${SEED}"
echo "============================================================"

# extract_skill_or_skill.py extracts/shuffles the rows and imports
# human_eval_predictions.py to attach all three models' Top-5 predictions.
"${CMD[@]}"

echo "[DONE] Saved: ${OUTPUT_CSV}"
