#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXPOSURE_DIR="${REPO_ROOT}/exposure_gain"

BLUE="\033[1;34m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
RESET="\033[0m"

section () {
  echo -e "\n${BLUE}══════════════════════════════════════════════════════════════${RESET}"
  echo -e "${BLUE}▶ $1${RESET}"
  echo -e "${BLUE}══════════════════════════════════════════════════════════════${RESET}\n"
}

ts () {
  echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${RESET} $1"
}

trap 'echo -e "${RED}✗ Failed at line $LINENO${RESET}"' ERR

# ======================== EDITABLE CONFIG ========================
PYTHON_BIN="/home/jovyan/LEM_data2/hyunjincho/env/lem/bin/python"
BASE_DATA_DIR="/home/jovyan/LEM_data3/hyunjincho/LEM"
ANALYSIS_YEAR=2025

# 1: run, 0: skip. Skipped steps reuse the output paths below.
RUN_COUNT=0
RUN_TRENDS=0
RUN_GRAPH=1

# Clean BERT prediction structure:
#   ${BASE_DATA_DIR}/predictions_clean/bert_pred/pred/YYYY/predictions_YYYY-MM.csv.gz
PRED_DIR="${BASE_DATA_DIR}/predictions_clean/bert_pred/pred/${ANALYSIS_YEAR}"
COUNT_GLOB="predictions_${ANALYSIS_YEAR}-*.csv.gz"
GRAPH_PRED_GLOB="${COUNT_GLOB}"
CANDIDATE_COL="substitution"   # highest-probability candidate excluding truth
POSTING_SOURCE_COL="file_path" # posting id = (row_idx, this column)

OUTPUT_DIR="${BASE_DATA_DIR}/exposure_gain"
COUNTS_BY_PAIR="${OUTPUT_DIR}/counts_by_pair_${ANALYSIS_YEAR}.csv"
TRENDS_OUT="${OUTPUT_DIR}/trends_by_pair_${ANALYSIS_YEAR}.csv"
OUT_FIG="${OUTPUT_DIR}/posting_margin_ratio_hist_${ANALYSIS_YEAR}_clean.png"

COUNT_ENCODING="utf-8"

# Google Trends parameters. This step requires network access.
TREND_GEO="US"                         # US | empty string for worldwide
TREND_GPROP=""                         # empty | news | froogle | youtube | images
TREND_TIMEFRAME="2025-01-01 2025-06-30"
TREND_MIN_COUNT=1                      # only query pairs with count >= this value
TREND_SLEEP=1.0                        # seconds between retries/requests
TREND_RETRIES=3
TREND_CONNECT_TIMEOUT=15               # pytrends default is only 2 seconds
TREND_READ_TIMEOUT=30
TREND_SESSION_RETRIES=5                # initial/replacement session attempts
TREND_SESSION_RETRY_WAIT=15            # retry n waits this value*n seconds
TREND_WAIT_ON_429=600                  # seconds; progress is saved before waiting
TREND_ENCODING="utf-8"

GRAPH_DROP_BIDIRECTIONAL=1             # 1: drop A->B and B->A in same posting
GRAPH_ENCODING="utf-8"
# ================================================================

for flag_name in RUN_COUNT RUN_TRENDS RUN_GRAPH GRAPH_DROP_BIDIRECTIONAL; do
  flag_value="${!flag_name}"
  if [[ "${flag_value}" != "0" && "${flag_value}" != "1" ]]; then
    echo "${flag_name} must be 0 or 1 (current: ${flag_value})." >&2
    exit 2
  fi
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -d "${PRED_DIR}" ]]; then
  echo "Prediction directory not found: ${PRED_DIR}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}" "$(dirname "${OUT_FIG}")"
cd "${REPO_ROOT}"

section "CONFIG"
echo "PYTHON_BIN              = ${PYTHON_BIN}"
echo "PRED_DIR                = ${PRED_DIR}"
echo "COUNT_GLOB              = ${COUNT_GLOB}"
echo "CANDIDATE_COL           = ${CANDIDATE_COL}"
echo "POSTING_SOURCE_COL      = ${POSTING_SOURCE_COL}"
echo "COUNTS_BY_PAIR          = ${COUNTS_BY_PAIR}"
echo "TRENDS_OUT              = ${TRENDS_OUT}"
echo "OUT_FIG                 = ${OUT_FIG}"
echo "RUN_COUNT               = ${RUN_COUNT}"
echo "RUN_TRENDS              = ${RUN_TRENDS}"
echo "RUN_GRAPH               = ${RUN_GRAPH}"
echo "TREND_GEO               = ${TREND_GEO:-worldwide}"
echo "TREND_GPROP             = ${TREND_GPROP:-web}"
echo "TREND_TIMEFRAME         = ${TREND_TIMEFRAME}"
echo "TREND_MIN_COUNT         = ${TREND_MIN_COUNT}"
echo "TREND_CONNECT_TIMEOUT   = ${TREND_CONNECT_TIMEOUT}"
echo "TREND_READ_TIMEOUT      = ${TREND_READ_TIMEOUT}"
echo "TREND_SESSION_RETRIES   = ${TREND_SESSION_RETRIES}"
echo "TREND_SESSION_RETRY_WAIT= ${TREND_SESSION_RETRY_WAIT}"
echo "GRAPH_DROP_BIDIRECTIONAL= ${GRAPH_DROP_BIDIRECTIONAL}"

if [[ "${RUN_COUNT}" == "1" ]]; then
  section "[1/3] Build monthly unordered skill-pair counts"
  "${PYTHON_BIN}" "${EXPOSURE_DIR}/get_count_by_pair.py" \
    --pred-dir "${PRED_DIR}" \
    --out-csv "${COUNTS_BY_PAIR}" \
    --glob "${COUNT_GLOB}" \
    --candidate-col "${CANDIDATE_COL}" \
    --encoding "${COUNT_ENCODING}"
  ts "Saved pair counts -> ${COUNTS_BY_PAIR}"
else
  ts "Skipping pair counts (RUN_COUNT=0)"
fi

if [[ "${RUN_TRENDS}" == "1" ]]; then
  if [[ ! -f "${COUNTS_BY_PAIR}" ]]; then
    echo "Counts file not found: ${COUNTS_BY_PAIR}" >&2
    echo "Set RUN_COUNT=1 or point COUNTS_BY_PAIR to an existing file." >&2
    exit 2
  fi
  section "[2/3] Fetch monthly Google Trends proxy values"
  "${PYTHON_BIN}" "${EXPOSURE_DIR}/get_trend.py" \
    --in-csv "${COUNTS_BY_PAIR}" \
    --out-csv "${TRENDS_OUT}" \
    --geo "${TREND_GEO}" \
    --gprop "${TREND_GPROP}" \
    --timeframe "${TREND_TIMEFRAME}" \
    --min-count "${TREND_MIN_COUNT}" \
    --sleep "${TREND_SLEEP}" \
    --retries "${TREND_RETRIES}" \
    --connect-timeout "${TREND_CONNECT_TIMEOUT}" \
    --read-timeout "${TREND_READ_TIMEOUT}" \
    --session-retries "${TREND_SESSION_RETRIES}" \
    --session-retry-wait "${TREND_SESSION_RETRY_WAIT}" \
    --wait-on-429 "${TREND_WAIT_ON_429}" \
    --encoding "${TREND_ENCODING}"
  ts "Saved Trends values -> ${TRENDS_OUT}"
  ts "Interrupted runs resume from ${TRENDS_OUT} and ${TRENDS_OUT}.partial"
else
  ts "Skipping Google Trends (RUN_TRENDS=0)"
fi

if [[ "${RUN_GRAPH}" == "1" ]]; then
  if [[ ! -f "${TRENDS_OUT}" ]]; then
    echo "Trends file not found: ${TRENDS_OUT}" >&2
    echo "Set RUN_TRENDS=1 or point TRENDS_OUT to an existing file." >&2
    exit 2
  fi
  section "[3/3] Compute posting-level margins and plot"
  GRAPH_CMD=(
    "${PYTHON_BIN}" "${EXPOSURE_DIR}/get_graph.py"
    --pred-dir "${PRED_DIR}"
    --pred-glob "${GRAPH_PRED_GLOB}"
    --candidate-col "${CANDIDATE_COL}"
    --posting-source-col "${POSTING_SOURCE_COL}"
    --margin-csv "${TRENDS_OUT}"
    --out-fig "${OUT_FIG}"
    --encoding "${GRAPH_ENCODING}"
  )
  if [[ "${GRAPH_DROP_BIDIRECTIONAL}" == "1" ]]; then
    GRAPH_CMD+=(--drop-bidirectional)
  fi
  "${GRAPH_CMD[@]}"
  ts "Saved histogram -> ${OUT_FIG}"
else
  ts "Skipping graph (RUN_GRAPH=0)"
fi

section "DONE"
ts "Enabled margin pipeline stages completed successfully"
