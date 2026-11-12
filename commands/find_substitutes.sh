#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ======================== EDITABLE CONFIG ========================
PYTHON_BIN="/home/jovyan/LEM_data2/hyunjincho/env/lem/bin/python"
BASE_DATA_DIR="/home/jovyan/LEM_data3/hyunjincho/LEM"

TARGET_SKILLS=("python" "java" "javascript" "c")

# Current prediction layout:
#   ${BASE_DATA_DIR}/predictions/bert_pred/pred/YYYY/predictions_YYYY-MM.csv.gz
PRED_ROOT_SOC2="${BASE_DATA_DIR}/predictions_clean/bert_pred/pred"
PRED_ROOT_YEAR="${BASE_DATA_DIR}/predictions_clean/bert_pred/pred"

# Current BERT prediction schema. `substitution` already excludes the truth.
SUBSTITUTION_COL="substitution"
POSTING_SOURCE_COL="preprocessed_file_path"

RUN_SOC2=1              # 1: run SOC2 analysis, 0: skip
RUN_YEAR=1              # 1: run year analysis, 0: skip

SOC2_TOPK=5
SOC2_YEARS=()           # all years; example: ("2018" "2019")
SOC2_USECOLS=1          # 1: required columns only, 0: full CSV

YEAR_TOPK=5
YEAR_YEARS=()           # all years; example: ("2018" "2019")
YEAR_USECOLS=1          # 1: required columns only, 0: full CSV

OUT_SOC2_DIR="${BASE_DATA_DIR}/top5_substitutions/substitute_by_area"
OUT_YEAR_DIR="${BASE_DATA_DIR}/top5_substitutions/substitute_by_time"
# ================================================================

BLUE="\033[1;34m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
RESET="\033[0m"

section () {
  echo -e "\n${BLUE}══════════════════════════════════════════════════════════════${RESET}"
  echo -e "${BLUE}▶ $1${RESET}"
  echo -e "${BLUE}══════════════════════════════════════════════════════════════${RESET}\n"
}

step () { echo -e "${GREEN}➤ $1${RESET}"; }
ts () { echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${RESET} $1"; }
fail () { echo -e "${RED}ERROR: $1${RESET}" >&2; exit 1; }

validate_flag () {
  local name="$1"
  local value="$2"
  [[ "${value}" == "0" || "${value}" == "1" ]] || fail "${name} must be 0 or 1."
}

validate_flag "RUN_SOC2" "${RUN_SOC2}"
validate_flag "RUN_YEAR" "${RUN_YEAR}"
validate_flag "SOC2_USECOLS" "${SOC2_USECOLS}"
validate_flag "YEAR_USECOLS" "${YEAR_USECOLS}"
[[ ${#TARGET_SKILLS[@]} -gt 0 ]] || fail "TARGET_SKILLS must contain at least one skill."
[[ "${RUN_SOC2}" == "1" || "${RUN_YEAR}" == "1" ]] || fail "At least one analysis must be enabled."
[[ -x "${PYTHON_BIN}" ]] || fail "Python executable not found: ${PYTHON_BIN}"
[[ -f "${REPO_ROOT}/substitute_by_area/areawise_substitutes.py" ]] || fail "Missing SOC2 script."
[[ -f "${REPO_ROOT}/substitute_by_time/yearwise_substitutes.py" ]] || fail "Missing year script."
if [[ "${RUN_SOC2}" == "1" ]]; then
  [[ -e "${PRED_ROOT_SOC2}" ]] || fail "SOC2 prediction root not found: ${PRED_ROOT_SOC2}"
fi
if [[ "${RUN_YEAR}" == "1" ]]; then
  [[ -e "${PRED_ROOT_YEAR}" ]] || fail "Year prediction root not found: ${PRED_ROOT_YEAR}"
fi

mkdir -p "${OUT_SOC2_DIR}" "${OUT_YEAR_DIR}"
cd "${REPO_ROOT}"

section "CONFIG"
echo "PYTHON_BIN        = ${PYTHON_BIN}"
echo "BASE_DATA_DIR     = ${BASE_DATA_DIR}"
echo "TARGET_SKILLS     = ${TARGET_SKILLS[*]}"
echo "SUBSTITUTION_COL  = ${SUBSTITUTION_COL}"
echo "POSTING_SOURCE    = ${POSTING_SOURCE_COL} + row_idx"
echo "PRED_ROOT_SOC2    = ${PRED_ROOT_SOC2}"
echo "PRED_ROOT_YEAR    = ${PRED_ROOT_YEAR}"
echo "RUN_SOC2          = ${RUN_SOC2}"
echo "RUN_YEAR          = ${RUN_YEAR}"
echo "SOC2_TOPK         = ${SOC2_TOPK}"
echo "SOC2_YEARS        = ${SOC2_YEARS[*]:-all}"
echo "YEAR_TOPK         = ${YEAR_TOPK}"
echo "YEAR_YEARS        = ${YEAR_YEARS[*]:-all}"
echo "OUT_SOC2_DIR      = ${OUT_SOC2_DIR}"
echo "OUT_YEAR_DIR      = ${OUT_YEAR_DIR}"

for TARGET_SKILL in "${TARGET_SKILLS[@]}"; do
  TARGET_SKILL_TAG="${TARGET_SKILL,,}"
  TARGET_SKILL_TAG="${TARGET_SKILL_TAG// /_}"
  TARGET_SKILL_TAG="${TARGET_SKILL_TAG//\//_}"

  OUT_SOC2="${OUT_SOC2_DIR}/subs_${TARGET_SKILL_TAG}_by_soc2.csv"
  OUT_YEAR="${OUT_YEAR_DIR}/${TARGET_SKILL_TAG}_yearwise_top${YEAR_TOPK}.csv"

  section "SKILL: ${TARGET_SKILL}"
  echo "OUT_SOC2 = ${OUT_SOC2}"
  echo "OUT_YEAR = ${OUT_YEAR}"

  if [[ "${RUN_SOC2}" == "1" ]]; then
    section "[1/2] SOC2-wise substitutes: ${TARGET_SKILL}"
    step "Running areawise_substitutes.py"

    SOC2_CMD=("${PYTHON_BIN}" "${REPO_ROOT}/substitute_by_area/areawise_substitutes.py"
      --pred-root "${PRED_ROOT_SOC2}"
      --target-skill "${TARGET_SKILL}"
      --candidate-col "${SUBSTITUTION_COL}"
      --posting-source-col "${POSTING_SOURCE_COL}"
      --topk-per-soc2 "${SOC2_TOPK}"
      --out "${OUT_SOC2}"
    )
    if [[ ${#SOC2_YEARS[@]} -gt 0 ]]; then
      SOC2_CMD+=(--years "${SOC2_YEARS[@]}")
    fi
    if [[ "${SOC2_USECOLS}" == "1" ]]; then
      SOC2_CMD+=(--usecols)
    fi
    "${SOC2_CMD[@]}"
    ts "Saved SOC2-wise results -> ${OUT_SOC2}"
  else
    ts "Skipping SOC2-wise analysis for ${TARGET_SKILL} (RUN_SOC2=0)"
  fi

  if [[ "${RUN_YEAR}" == "1" ]]; then
    section "[2/2] Year-wise substitutes: ${TARGET_SKILL}"
    step "Running yearwise_substitutes.py"

    YEAR_CMD=("${PYTHON_BIN}" "${REPO_ROOT}/substitute_by_time/yearwise_substitutes.py"
      --pred-root "${PRED_ROOT_YEAR}"
      --target-skill "${TARGET_SKILL}"
      --candidate-col "${SUBSTITUTION_COL}"
      --posting-source-col "${POSTING_SOURCE_COL}"
      --out-topk "${YEAR_TOPK}"
      --out "${OUT_YEAR}"
    )
    if [[ ${#YEAR_YEARS[@]} -gt 0 ]]; then
      YEAR_CMD+=(--years "${YEAR_YEARS[@]}")
    fi
    if [[ "${YEAR_USECOLS}" != "1" ]]; then
      YEAR_CMD+=(--no-usecols)
    fi
    "${YEAR_CMD[@]}"
    ts "Saved year-wise results -> ${OUT_YEAR}"
  else
    ts "Skipping year-wise analysis for ${TARGET_SKILL} (RUN_YEAR=0)"
  fi
done

section "DONE"
ts "All enabled substitute analyses completed successfully for: ${TARGET_SKILLS[*]}"
