#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ============================================================
# EDITABLE CONFIGURATION
# Change the quoted text values in this block before running.
# Boolean options must be exactly "true" or "false".
# ============================================================

# Stage selection
# Body fingerprints are new pipeline data, so preprocessing, filtering, and
# splitting are all enabled for the first clean rebuild.
RUN_PREPROCESS="true"
RUN_FILTER="true"
RUN_SPLIT="true"
RUN_SUMMARY="true"

# Python executable
PYTHON_BIN="/home/jovyan/LEM_data2/hyunjincho/env/lem/bin/python"

# ---------------------- Stage 1: preprocess ------------------
RAW_INPUT_ROOT="/home/jovyan/LEM_data/us/csv/fortnightly/all/20250607"
TARGET_SKILLS_CSV="/home/jovyan/hyunjincho/LEM_journal/contextual_skills_substitutes/target_skills.csv"
PREPROCESS_OUTPUT_ROOT="/home/jovyan/LEM_data3/hyunjincho/LEM1"
PREPROCESS_WORKERS="16"
PREPROCESS_BERT_PATH="bert-base-uncased"
PREPROCESS_RESUME="false"
PREPROCESS_BODY_DEDUP_SEED="42"

# preprocess.py appends /all to PREPROCESS_OUTPUT_ROOT.
# Existing output: /home/jovyan/LEM_data3/hyunjincho/LEM1/all

# ---------------------- Stage 2: filter ----------------------
FILTER_INPUT_ROOT="/home/jovyan/LEM_data3/hyunjincho/LEM1/all"
FILTER_OUTPUT_ROOT="/home/jovyan/LEM_data3/hyunjincho/LEM1/all_filterd"
FILTER_PATTERN="preprocessed_*.csv.gz"
FILTER_WORKERS="16"
FILTER_DEDUP_SEED="42"
FILTER_MIN_SKILL_FREQUENCY="100"
FILTER_COMPRESSION_LEVEL="1"
FILTER_DUPLICATE_EXAMPLE_LIMIT="100"

# Optional manual exclusions. No-cooccurrence skills are excluded automatically.
# Example:
# FILTER_EXCLUDE_SKILLS=("skill one" "skill two")
FILTER_EXCLUDE_SKILLS=()

# ----------------------- Stage 3: split ----------------------
SPLIT_INPUT_ROOT="/home/jovyan/LEM_data3/hyunjincho/LEM1/all_filterd"
SPLIT_OUTPUT_ROOT="/home/jovyan/LEM_data3/hyunjincho/LEM1/all_train_test"
SPLIT_PATTERN="preprocessed_*.csv.gz"
SPLIT_WORKERS="16"
SPLIT_TEST_RATIO="0.2"
SPLIT_SEED="42"
SPLIT_COMPRESSION_LEVEL="1"

# ---------------------- Stage 4: summary --------------------
SUMMARY_OUTPUT_TXT="/home/jovyan/LEM_data3/hyunjincho/LEM1/pipeline_stage_counts.txt"

# ============================================================
# END EDITABLE CONFIGURATION
# ============================================================

require_boolean() {
  local option_name="$1"
  local option_value="$2"
  if [[ "${option_value}" != "true" && "${option_value}" != "false" ]]; then
    echo "${option_name} must be exactly true or false; got: ${option_value}" >&2
    exit 2
  fi
}

require_positive_integer() {
  local option_name="$1"
  local option_value="$2"
  if ! [[ "${option_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${option_name} must be a positive integer; got: ${option_value}" >&2
    exit 2
  fi
}

require_boolean "RUN_PREPROCESS" "${RUN_PREPROCESS}"
require_boolean "RUN_FILTER" "${RUN_FILTER}"
require_boolean "RUN_SPLIT" "${RUN_SPLIT}"
require_boolean "RUN_SUMMARY" "${RUN_SUMMARY}"
require_boolean "PREPROCESS_RESUME" "${PREPROCESS_RESUME}"
require_positive_integer "PREPROCESS_WORKERS" "${PREPROCESS_WORKERS}"
require_positive_integer "FILTER_WORKERS" "${FILTER_WORKERS}"
require_positive_integer "SPLIT_WORKERS" "${SPLIT_WORKERS}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

export RAW_INPUT_ROOT TARGET_SKILLS_CSV

echo "============================================================"
echo "Preprocessing pipeline configuration"
echo "RUN_PREPROCESS       = ${RUN_PREPROCESS}"
echo "RUN_FILTER           = ${RUN_FILTER}"
echo "RUN_SPLIT            = ${RUN_SPLIT}"
echo "RUN_SUMMARY          = ${RUN_SUMMARY}"
echo "PREPROCESS_OUTPUT    = ${PREPROCESS_OUTPUT_ROOT}/all"
echo "FILTER_INPUT         = ${FILTER_INPUT_ROOT}"
echo "FILTER_OUTPUT        = ${FILTER_OUTPUT_ROOT}"
echo "SPLIT_INPUT          = ${SPLIT_INPUT_ROOT}"
echo "SPLIT_OUTPUT         = ${SPLIT_OUTPUT_ROOT}"
echo "SUMMARY_OUTPUT       = ${SUMMARY_OUTPUT_TXT}"
echo "============================================================"

if [[ "${RUN_PREPROCESS}" == "true" ]]; then
  if [[ ! -d "${RAW_INPUT_ROOT}" ]]; then
    echo "Raw input root not found: ${RAW_INPUT_ROOT}" >&2
    exit 2
  fi
  if [[ ! -f "${TARGET_SKILLS_CSV}" ]]; then
    echo "Target skills CSV not found: ${TARGET_SKILLS_CSV}" >&2
    exit 2
  fi

  echo
  echo "==== [1/4] Running preprocess.py ===="
  echo "RAW_INPUT_ROOT          = ${RAW_INPUT_ROOT}"
  echo "TARGET_SKILLS_CSV       = ${TARGET_SKILLS_CSV}"
  echo "PREPROCESS_OUTPUT_ROOT  = ${PREPROCESS_OUTPUT_ROOT}"
  echo "PREPROCESS_WORKERS      = ${PREPROCESS_WORKERS}"
  echo "PREPROCESS_BERT_PATH    = ${PREPROCESS_BERT_PATH}"
  echo "PREPROCESS_RESUME       = ${PREPROCESS_RESUME}"
  echo "PREPROCESS_DEDUP_SEED   = ${PREPROCESS_BODY_DEDUP_SEED}"

  preprocess_command=(
    "${PYTHON_BIN}" "${REPO_ROOT}/data_preprocess/preprocess.py"
    --output_path "${PREPROCESS_OUTPUT_ROOT}"
    --workers "${PREPROCESS_WORKERS}"
    --bert_path "${PREPROCESS_BERT_PATH}"
    --body-dedup-seed "${PREPROCESS_BODY_DEDUP_SEED}"
  )
  if [[ "${PREPROCESS_RESUME}" == "true" ]]; then
    preprocess_command+=(--resume)
  else
    preprocess_command+=(--no-resume)
  fi
  "${preprocess_command[@]}"
else
  echo
  echo "==== [1/4] SKIP preprocess.py ===="
  echo "Reusing existing preprocessed data: ${FILTER_INPUT_ROOT}"
fi

if [[ "${RUN_FILTER}" == "true" ]]; then
  if [[ ! -d "${FILTER_INPUT_ROOT}" ]]; then
    echo "Filter input root not found: ${FILTER_INPUT_ROOT}" >&2
    exit 2
  fi

  echo
  echo "==== [2/4] Running filter_data.py ===="
  echo "FILTER_INPUT_ROOT             = ${FILTER_INPUT_ROOT}"
  echo "FILTER_OUTPUT_ROOT            = ${FILTER_OUTPUT_ROOT}"
  echo "FILTER_PATTERN                = ${FILTER_PATTERN}"
  echo "FILTER_WORKERS                = ${FILTER_WORKERS}"
  echo "FILTER_DEDUP_SEED             = ${FILTER_DEDUP_SEED}"
  echo "FILTER_MIN_SKILL_FREQUENCY    = ${FILTER_MIN_SKILL_FREQUENCY}"
  echo "FILTER_COMPRESSION_LEVEL      = ${FILTER_COMPRESSION_LEVEL}"
  echo "FILTER_EXCLUDE_SKILLS         = ${FILTER_EXCLUDE_SKILLS[*]:-none}"

  filter_command=(
    "${PYTHON_BIN}" "${REPO_ROOT}/data_preprocess/filter_data.py"
    --input-root "${FILTER_INPUT_ROOT}"
    --output-root "${FILTER_OUTPUT_ROOT}"
    --pattern "${FILTER_PATTERN}"
    --workers "${FILTER_WORKERS}"
    --dedup-seed "${FILTER_DEDUP_SEED}"
    --min-skill-frequency "${FILTER_MIN_SKILL_FREQUENCY}"
    --compression-level "${FILTER_COMPRESSION_LEVEL}"
    --duplicate-example-limit "${FILTER_DUPLICATE_EXAMPLE_LIMIT}"
  )
  for excluded_skill in "${FILTER_EXCLUDE_SKILLS[@]}"; do
    filter_command+=(--exclude-skill "${excluded_skill}")
  done
  "${filter_command[@]}"
else
  echo
  echo "==== [2/4] SKIP filter_data.py ===="
  echo "Reusing existing filtered data: ${SPLIT_INPUT_ROOT}"
fi

if [[ "${RUN_SPLIT}" == "true" ]]; then
  if [[ ! -d "${SPLIT_INPUT_ROOT}" ]]; then
    echo "Split input root not found: ${SPLIT_INPUT_ROOT}" >&2
    exit 2
  fi

  echo
  echo "==== [3/4] Running train_test_split.py ===="
  echo "SPLIT_INPUT_ROOT        = ${SPLIT_INPUT_ROOT}"
  echo "SPLIT_OUTPUT_ROOT       = ${SPLIT_OUTPUT_ROOT}"
  echo "SPLIT_PATTERN           = ${SPLIT_PATTERN}"
  echo "SPLIT_WORKERS           = ${SPLIT_WORKERS}"
  echo "SPLIT_TEST_RATIO        = ${SPLIT_TEST_RATIO}"
  echo "SPLIT_SEED              = ${SPLIT_SEED}"
  echo "SPLIT_COMPRESSION_LEVEL = ${SPLIT_COMPRESSION_LEVEL}"

  "${PYTHON_BIN}" "${REPO_ROOT}/data_preprocess/train_test_split.py" \
    --input-root "${SPLIT_INPUT_ROOT}" \
    --output-root "${SPLIT_OUTPUT_ROOT}" \
    --pattern "${SPLIT_PATTERN}" \
    --workers "${SPLIT_WORKERS}" \
    --test-ratio "${SPLIT_TEST_RATIO}" \
    --seed "${SPLIT_SEED}" \
    --compression-level "${SPLIT_COMPRESSION_LEVEL}"
else
  echo
  echo "==== [3/4] SKIP train_test_split.py ===="
fi

if [[ "${RUN_SUMMARY}" == "true" ]]; then
  echo
  echo "==== [4/4] Writing pipeline stage counts ===="
  "${PYTHON_BIN}" "${REPO_ROOT}/data_preprocess/summarize_preprocess_pipeline.py" \
    --preprocess-root "${PREPROCESS_OUTPUT_ROOT}" \
    --filter-root "${FILTER_OUTPUT_ROOT}" \
    --split-root "${SPLIT_OUTPUT_ROOT}" \
    --output-txt "${SUMMARY_OUTPUT_TXT}"
else
  echo
  echo "==== [4/4] SKIP pipeline stage counts ===="
fi

echo
echo "==== Pipeline completed ===="
