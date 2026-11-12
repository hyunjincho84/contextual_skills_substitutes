#!/usr/bin/env bash
set -euo pipefail

# Unified model-based validation entrypoint.
# By default, runs LLaMA -> Gemma -> Qwen sequentially.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SCORE_SCRIPT="${REPO_ROOT}/model_tests/likelihood_based_score/scoring_other_models.py"
MEAN_SCRIPT="${REPO_ROOT}/model_tests/likelihood_based_score/get_score_mean_w2v_one.py"

PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/LEM_data2/hyunjincho/env/lem/bin/python}"
BASE_DATA_DIR="${BASE_DATA_DIR:-/home/jovyan/LEM_data3/hyunjincho/LEM}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/home/jovyan/LEM_data2/hf_cache}"

MODE="full"
JUDGES="all"
RUN_SCORING="${RUN_SCORING:-1}"
RUN_MEAN="${RUN_MEAN:-1}"
RESUME="${RESUME:-0}"
STRICT_MEAN_MONTHS="${STRICT_MEAN_MONTHS:-1}"
YEAR_ARGS=()
if [[ -n "${YEARS:-}" ]]; then
  read -r -a YEAR_ARGS <<< "${YEARS}"
fi

BERT_PRED_DIR="${BERT_PRED_DIR:-${BASE_DATA_DIR}/predictions/bert_pred/pred}"
W2V_PRED_DIR="${W2V_PRED_DIR:-${BASE_DATA_DIR}/predictions/skill2vec_pred/pred}"
COND_PRED_DIR="${COND_PRED_DIR:-${BASE_DATA_DIR}/predictions/condprob_pred/pred}"

EVALUATION_ROOT="${EVALUATION_ROOT:-${BASE_DATA_DIR}/evaluation_sv}"
LLAMA_OUTPUT_ROOT="${LLAMA_OUTPUT_ROOT:-${EVALUATION_ROOT}/llama}"
GEMMA_OUTPUT_ROOT="${GEMMA_OUTPUT_ROOT:-${EVALUATION_ROOT}/gemma}"
QWEN_OUTPUT_ROOT="${QWEN_OUTPUT_ROOT:-${EVALUATION_ROOT}/qwen}"

LLAMA_CKPT="${LLAMA_CKPT:-meta-llama/Llama-3.2-3B}"
GEMMA_CKPT="${GEMMA_CKPT:-google/gemma-2-2b-it}"
QWEN_CKPT="${QWEN_CKPT:-Qwen/Qwen2.5-3B-Instruct}"
GEMMA_JUDGE_NAME="${GEMMA_JUDGE_NAME:-gemma}"
QWEN_JUDGE_NAME="${QWEN_JUDGE_NAME:-qwen25}"

# LLaMA scorer model selection: all, or a comma-separated subset of
# bert,w2v,conditional.
START_FROM="${START_FROM:-all}"

WINDOW_SIZE="${WINDOW_SIZE:-128}"
MAX_LEN="${MAX_LEN:-256}"
SENT_BATCH="${SENT_BATCH:-16}"
DEVICE="${DEVICE:-cuda}"
RANDOM_SEED="${RANDOM_SEED:-42}"
USE_AUTH_TOKEN="${USE_AUTH_TOKEN:-1}"

LLAMA_GPU_ID="${LLAMA_GPU_ID:-0}"
GPU_IDS="${GPU_IDS:-1}"
IFS=',' read -r -a JUDGE_GPU_IDS <<< "${GPU_IDS}"
NUM_WORKERS="${NUM_WORKERS:-1}"

DEBUG_FILE="${DEBUG_FILE:-${BERT_PRED_DIR}/2010/predictions_2010-01.csv.gz}"
DEBUG_N="${DEBUG_N:-10}"
DEBUG_MODEL="${DEBUG_MODEL:-bert}"

LOG_DIR="${LOG_DIR:-${EVALUATION_ROOT}/logs}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-${BASE_DATA_DIR}/all_train_test/train}"
MEAN_OUTPUT_CSV="${MEAN_OUTPUT_CSV:-${EVALUATION_ROOT}/all_judges_mean_summary_w2v_one_clean.csv}"
MEAN_OUTPUT_PLOT_DIR="${MEAN_OUTPUT_PLOT_DIR:-${EVALUATION_ROOT}}"

usage() {
  cat <<'EOF'
Usage:
  ./commands/run_scoring.sh [full|debug] [--judge NAME | --judges LIST]

Judge selection:
  --judge llama|gemma|qwen|all
  --judges llama,gemma,qwen
  --llama-only | --gemma-only | --qwen-only | --all

Examples:
  ./commands/run_scoring.sh
  ./commands/run_scoring.sh --judge gemma
  ./commands/run_scoring.sh --judges gemma,qwen
  ./commands/run_scoring.sh debug --judge llama

Environment switches:
  RUN_SCORING=0|1  RUN_MEAN=0|1  RESUME=0|1
  STRICT_MEAN_MONTHS=0|1
  START_FROM=all|bert|w2v|conditional|bert,w2v,...
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

while (( $# > 0 )); do
  case "$1" in
    --judge|--judges)
      if (( $# < 2 )); then
        die "$1 requires an argument"
      fi
      JUDGES="$2"
      shift 2
      ;;
    --judge=*|--judges=*)
      JUDGES="${1#*=}"
      shift
      ;;
    --mode)
      if (( $# < 2 )); then
        die "--mode requires full or debug"
      fi
      MODE="$2"
      shift 2
      ;;
    --mode=*)
      MODE="${1#*=}"
      shift
      ;;
    full|debug)
      MODE="$1"
      shift
      ;;
    --llama-only)
      JUDGES="llama"
      shift
      ;;
    --gemma-only)
      JUDGES="gemma"
      shift
      ;;
    --qwen-only)
      JUDGES="qwen"
      shift
      ;;
    --all)
      JUDGES="all"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1 (use --help)"
      ;;
  esac
done

[[ "${MODE}" == "full" || "${MODE}" == "debug" ]] ||
  die "mode must be full or debug: ${MODE}"

for flag_name in RUN_SCORING RUN_MEAN RESUME STRICT_MEAN_MONTHS; do
  flag_value="${!flag_name}"
  [[ "${flag_value}" == "0" || "${flag_value}" == "1" ]] ||
    die "${flag_name} must be 0 or 1: ${flag_value}"
done

JUDGES="${JUDGES//[[:space:]]/}"
RUN_LLAMA=0
RUN_GEMMA=0
RUN_QWEN=0

if [[ "${JUDGES}" == "all" ]]; then
  RUN_LLAMA=1
  RUN_GEMMA=1
  RUN_QWEN=1
else
  IFS=',' read -r -a REQUESTED_JUDGES <<< "${JUDGES}"
  (( ${#REQUESTED_JUDGES[@]} > 0 )) || die "judge selection is empty"
  for judge in "${REQUESTED_JUDGES[@]}"; do
    case "${judge}" in
      llama) RUN_LLAMA=1 ;;
      gemma) RUN_GEMMA=1 ;;
      qwen) RUN_QWEN=1 ;;
      "") die "judge selection contains an empty item: ${JUDGES}" ;;
      *) die "unknown judge '${judge}'; choose llama, gemma, qwen, or all" ;;
    esac
  done
fi

SELECTED_JUDGES=()
if (( RUN_LLAMA == 1 )); then
  SELECTED_JUDGES+=("llama")
fi
if (( RUN_GEMMA == 1 )); then
  SELECTED_JUDGES+=("gemma")
fi
if (( RUN_QWEN == 1 )); then
  SELECTED_JUDGES+=("qwen")
fi
saved_ifs="${IFS}"
IFS=','
SELECTED_LABEL="${SELECTED_JUDGES[*]}"
IFS="${saved_ifs}"

[[ -x "${PYTHON_BIN}" ]] || die "Python not found or not executable: ${PYTHON_BIN}"
[[ -f "${SCORE_SCRIPT}" ]] || die "missing script: ${SCORE_SCRIPT}"
[[ -f "${MEAN_SCRIPT}" ]] || die "missing script: ${MEAN_SCRIPT}"

if [[ "${RUN_SCORING}" == "1" && "${MODE}" == "full" ]]; then
  for input_dir in "${BERT_PRED_DIR}" "${W2V_PRED_DIR}" "${COND_PRED_DIR}"; do
    [[ -d "${input_dir}" ]] || die "prediction directory not found: ${input_dir}"
  done
fi

if [[ "${RUN_SCORING}" == "1" && "${MODE}" == "debug" ]]; then
  [[ -f "${DEBUG_FILE}" ]] || die "debug file not found: ${DEBUG_FILE}"
fi

if [[ "${RUN_SCORING}" == "1" &&
      ( "${RUN_GEMMA}" == "1" || "${RUN_QWEN}" == "1" ) ]]; then
  (( ${#JUDGE_GPU_IDS[@]} > 0 )) || die "JUDGE_GPU_IDS must not be empty"
  (( NUM_WORKERS >= 1 )) || die "NUM_WORKERS must be at least 1"
  (( NUM_WORKERS <= ${#JUDGE_GPU_IDS[@]} )) ||
    die "NUM_WORKERS (${NUM_WORKERS}) exceeds GPU count (${#JUDGE_GPU_IDS[@]})"
fi

export HF_HOME="${HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}/transformers"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_DIR}/hub"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "${LLAMA_OUTPUT_ROOT}" "${GEMMA_OUTPUT_ROOT}" "${QWEN_OUTPUT_ROOT}" "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/scoring_${SELECTED_LABEL}_${MODE}_${TIMESTAMP}.log"

cd "${REPO_ROOT}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================================"
echo "Unified validation"
echo "============================================================"
echo "Mode             : ${MODE}"
echo "Judges           : ${SELECTED_LABEL}"
echo "Execution order  : llama -> gemma -> qwen (selected only)"
echo "Run scoring      : ${RUN_SCORING}"
echo "Run mean         : ${RUN_MEAN}"
echo "Mean method      : paired judge SV / paired Skill2Vec SV"
echo "Score precision  : checkpoint-dtype forward, FP32 pooling/cosine reduction"
echo "Resume           : ${RESUME}"
echo "Years            : ${YEARS:-all}"
echo "Log               : ${LOG_FILE}"
echo "============================================================"

run_llama() {
  local cmd=(
    "${PYTHON_BIN}" "${SCORE_SCRIPT}"
    --judge-ckpt "${LLAMA_CKPT}"
    --judge-name "llama"
    --cache-dir "${HF_CACHE_DIR}"
    --window-size "${WINDOW_SIZE}"
    --max-len "${MAX_LEN}"
    --sent-batch "${SENT_BATCH}"
    --device "${DEVICE}"
    --random-seed "${RANDOM_SEED}"
    --start-from "${START_FROM}"
    --output-root "${LLAMA_OUTPUT_ROOT}"
  )

  if [[ "${USE_AUTH_TOKEN}" == "1" ]]; then
    cmd+=(--use-auth-token)
  else
    cmd+=(--no-auth-token)
  fi
  if [[ "${RESUME}" == "1" && "${MODE}" == "full" ]]; then
    cmd+=(--resume)
  fi
  if (( ${#YEAR_ARGS[@]} > 0 )); then
    cmd+=(--years "${YEAR_ARGS[@]}")
  fi

  if [[ "${MODE}" == "debug" ]]; then
    cmd+=(
      --debug-file "${DEBUG_FILE}"
      --debug-n "${DEBUG_N}"
      --debug-model "${DEBUG_MODEL}"
    )
  else
    cmd+=(
      --bert_pred_dir "${BERT_PRED_DIR}"
      --w2v_pred_dir "${W2V_PRED_DIR}"
      --cond_pred_dir "${COND_PRED_DIR}"
    )
  fi

  echo
  echo "============================================================"
  echo "[1/3] LLaMA validation"
  echo "Checkpoint        : ${LLAMA_CKPT}"
  echo "GPU               : ${LLAMA_GPU_ID}"
  echo "Output            : ${LLAMA_OUTPUT_ROOT}"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="${LLAMA_GPU_ID}" "${cmd[@]}"
}

run_other_judge() {
  local judge_ckpt="$1"
  local judge_name="$2"
  local output_root="$3"
  local display_name="$4"
  local order_label="$5"
  local worker_count="${NUM_WORKERS}"

  if [[ "${MODE}" == "debug" ]]; then
    worker_count=1
  fi

  echo
  echo "============================================================"
  echo "[${order_label}/3] ${display_name} validation"
  echo "Checkpoint        : ${judge_ckpt}"
  echo "Judge name        : ${judge_name}"
  echo "GPU IDs           : ${JUDGE_GPU_IDS[*]}"
  echo "Workers           : ${worker_count}"
  echo "Output            : ${output_root}"
  echo "============================================================"

  local worker_pids=()
  local worker_id gpu_id
  for ((worker_id = 0; worker_id < worker_count; worker_id++)); do
    gpu_id="${JUDGE_GPU_IDS[${worker_id}]}"
    local cmd=(
      "${PYTHON_BIN}" "${SCORE_SCRIPT}"
      --judge-ckpt "${judge_ckpt}"
      --judge-name "${judge_name}"
      --output-root "${output_root}"
      --cache-dir "${HF_CACHE_DIR}"
      --start-from all
      --window-size "${WINDOW_SIZE}"
      --max-len "${MAX_LEN}"
      --sent-batch "${SENT_BATCH}"
      --device "${DEVICE}"
      --random-seed "${RANDOM_SEED}"
      --task-world-size "${worker_count}"
      --task-rank "${worker_id}"
    )

    if [[ "${USE_AUTH_TOKEN}" == "1" ]]; then
      cmd+=(--use-auth-token)
    else
      cmd+=(--no-auth-token)
    fi

    if [[ "${RESUME}" == "1" && "${MODE}" == "full" ]]; then
      cmd+=(--resume)
    fi
    if (( ${#YEAR_ARGS[@]} > 0 )); then
      cmd+=(--years "${YEAR_ARGS[@]}")
    fi

    if [[ "${MODE}" == "debug" ]]; then
      local debug_output="${output_root}/debug/${judge_name}_debug.csv"
      mkdir -p "$(dirname "${debug_output}")"
      cmd+=(
        --debug-file "${DEBUG_FILE}"
        --debug-n "${DEBUG_N}"
        --debug-model "${DEBUG_MODEL}"
        --debug-output "${debug_output}"
      )
    else
      cmd+=(
        --bert_pred_dir "${BERT_PRED_DIR}"
        --w2v_pred_dir "${W2V_PRED_DIR}"
        --cond_pred_dir "${COND_PRED_DIR}"
      )
    fi

    echo "Launching ${display_name} worker ${worker_id}/$((worker_count - 1)) on GPU ${gpu_id}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}" &
    worker_pids+=("$!")
  done

  local failed=0
  local pid
  for pid in "${worker_pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  (( failed == 0 )) || die "${display_name} validation failed"
}

run_means() {
  local mean_judges=()
  if (( RUN_LLAMA == 1 )); then
    mean_judges+=("llama")
  fi
  if (( RUN_QWEN == 1 )); then
    mean_judges+=("qwen")
  fi
  if (( RUN_GEMMA == 1 )); then
    mean_judges+=("gemma")
  fi

  echo
  echo "Computing ${mean_judges[*]} summary..."
  local mean_cmd=(
    "${PYTHON_BIN}" "${MEAN_SCRIPT}"
    --judges "${mean_judges[@]}"
    --train-data-dir "${TRAIN_DATA_DIR}"
    --llama-root "${LLAMA_OUTPUT_ROOT}"
    --llama-judge "llama"
    --qwen-root "${QWEN_OUTPUT_ROOT}"
    --qwen-judge "${QWEN_JUDGE_NAME}"
    --gemma-root "${GEMMA_OUTPUT_ROOT}"
    --gemma-judge "${GEMMA_JUDGE_NAME}"
    --output-csv "${MEAN_OUTPUT_CSV}"
    --output-plot-dir "${MEAN_OUTPUT_PLOT_DIR}"
  )
  if (( STRICT_MEAN_MONTHS == 1 )); then
    mean_cmd+=(--strict-months)
  fi
  if (( ${#YEAR_ARGS[@]} > 0 )); then
    mean_cmd+=(--years "${YEAR_ARGS[@]}")
  fi
  "${mean_cmd[@]}"
}

if [[ "${RUN_SCORING}" == "1" ]]; then
  if (( RUN_LLAMA == 1 )); then
    run_llama
  fi
  if (( RUN_GEMMA == 1 )); then
    run_other_judge "${GEMMA_CKPT}" "${GEMMA_JUDGE_NAME}" "${GEMMA_OUTPUT_ROOT}" "Gemma" "2"
  fi
  if (( RUN_QWEN == 1 )); then
    run_other_judge "${QWEN_CKPT}" "${QWEN_JUDGE_NAME}" "${QWEN_OUTPUT_ROOT}" "Qwen" "3"
  fi
else
  echo "Skipping scoring (RUN_SCORING=0)."
fi

if [[ "${MODE}" == "full" && "${RUN_MEAN}" == "1" ]]; then
  run_means
elif [[ "${MODE}" == "debug" && "${RUN_MEAN}" == "1" ]]; then
  echo "Skipping mean calculation in debug mode."
else
  echo "Skipping mean calculation (RUN_MEAN=0)."
fi

echo
echo "============================================================"
echo "Completed judges: ${SELECTED_LABEL}"
echo "Log             : ${LOG_FILE}"
echo "============================================================"
