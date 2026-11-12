#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SCORE_SCRIPT="${REPO_ROOT}/model_tests/likelihood_based_score/scoring_bert.py"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/LEM_data2/hyunjincho/env/lem/bin/python}"

BASE_DATA_DIR="${BASE_DATA_DIR:-/home/jovyan/LEM_data3/hyunjincho/LEM}"
INPUT_ROOT="${INPUT_ROOT:-${BASE_DATA_DIR}/predictions_clean}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE_DATA_DIR}/evaluation_sv/bert_clean}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/home/jovyan/LEM_data2/hf_cache/transformers}"
BERT_BASE_MODEL="${BERT_BASE_MODEL:-bert-base-uncased}"

START_FROM="${START_FROM:-all}"
DEVICE="${DEVICE:-auto}"
WINDOW_SIZE="${WINDOW_SIZE:-256}"
MAX_LEN="${MAX_LEN:-512}"
SENT_BATCH="${SENT_BATCH:-4}"
CHUNK_SIZE="${CHUNK_SIZE:-2000}"
RANDOM_SEED="${RANDOM_SEED:-42}"
TASK_RANK="${TASK_RANK:-0}"
TASK_WORLD_SIZE="${TASK_WORLD_SIZE:-1}"
RESUME="${RESUME:-0}"
STRICT="${STRICT:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

for flag_name in RESUME STRICT LOCAL_FILES_ONLY; do
  flag_value="${!flag_name}"
  if [[ "${flag_value}" != "0" && "${flag_value}" != "1" ]]; then
    echo "ERROR: ${flag_name} must be 0 or 1: ${flag_value}" >&2
    exit 2
  fi
done

if [[ ! -f "${SCORE_SCRIPT}" ]]; then
  echo "ERROR: scorer not found: ${SCORE_SCRIPT}" >&2
  exit 2
fi

CMD=(
  "${PYTHON_BIN}"
  "${SCORE_SCRIPT}"
  --input-root "${INPUT_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --model-name "${BERT_BASE_MODEL}"
  --judge-name bert
  --cache-dir "${HF_CACHE_DIR}"
  --start-from "${START_FROM}"
  --device "${DEVICE}"
  --window-size "${WINDOW_SIZE}"
  --max-len "${MAX_LEN}"
  --sent-batch "${SENT_BATCH}"
  --chunk-size "${CHUNK_SIZE}"
  --random-seed "${RANDOM_SEED}"
  --task-rank "${TASK_RANK}"
  --task-world-size "${TASK_WORLD_SIZE}"
)

if [[ -n "${YEARS:-}" ]]; then
  read -r -a YEAR_ARGS <<< "${YEARS}"
  CMD+=(--years "${YEAR_ARGS[@]}")
fi
if [[ "${RESUME}" == "1" ]]; then
  CMD+=(--resume)
fi
if [[ "${STRICT}" == "1" ]]; then
  CMD+=(--strict)
fi
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  CMD+=(--local-files-only)
fi

echo "BERT semantic-validation scoring"
echo "  input : ${INPUT_ROOT}/{bert_pred,skill2vec_pred,condprob_pred}/pred/<year>"
echo "  output: ${OUTPUT_ROOT}/{bert,w2v,conditional}/<year>"
echo "  model : ${BERT_BASE_MODEL}"
echo "  device: ${DEVICE}"
echo "  precision: AMP forward, FP32 pooling/cosine/attention reduction"

exec "${CMD[@]}" "$@"
