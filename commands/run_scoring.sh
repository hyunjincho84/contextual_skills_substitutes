#!/usr/bin/env bash
set -e

# ============================================================
# FULL PIPELINE (A + B):
#
# PART A: FULL test set evaluation
#   - BERT / Skill2Vec / Conditional Probability
#   - Compute SV (LLaMA-based) with scoring.py
#   - Compute mean SV per model with compute_score_mean.py
#
# PART B: GPT-sample evaluation
#   - Input: gpt_samples/20*/*with_gpt_pred.csv*
#   - Compute SV (LLaMA-based) on those files using scoring_gpt.py
#   - Then compute GPT-sample mean SV + compare vs BERT full SV
#     using compute_gpt_score_mean.py
# ============================================================

# ---------- helpers (pretty logs) ----------
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
ts   () { echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${RESET} $1"; }
warn () { echo -e "${YELLOW}⚠ $1${RESET}"; }
die  () { echo -e "${RED}✖ $1${RESET}"; exit 1; }

# ============================================================
# USER CONFIGURATION
# ============================================================

PYTHON_BIN="python3"

# ---- LLaMA checkpoint (used in both A and B) ----
LLAMA_CKPT="meta-llama/Llama-3.2-3B"

# ---- Optional year filter (leave empty => all years) ----
YEARS=()   # e.g. YEARS=("2018" "2019")

# -------------------------
# PART A CONFIG (FULL DATA)
# -------------------------
BERT_PRED_DIR="/home/jovyan/LEM_data2/hyunjincho/bert_pred_new/pred"
W2V_PRED_DIR="/home/jovyan/LEM_data2/hyunjincho/skill2vec_pred_new/pred"
COND_PRED_DIR="/home/jovyan/LEM_data2/hyunjincho/condprob_pred_new/pred"

START_FROM="all"   # all | bert | w2v | conditional | bert,w2v
CAP=8000
A_WINDOW_SIZE=256
A_SENT_BATCH=4

SCORE_FULL_SCRIPT="../model_tests/likelihood_based_score/scoring.py"
MEAN_FULL_SCRIPT="../model_tests/likelihood_based_score/compute_score_mean.py"

# -------------------------
# PART B CONFIG (GPT SAMPLES, BERT-only SV)
# -------------------------
B_IN_PATTERN="/home/jovyan/LEM_data2/hyunjincho/gpt_samples/20*/*with_gpt_pred.csv*"

B_WINDOW_SIZE=256
B_MAX_LEN=512
B_SENT_BATCH=4
B_CHUNK_SIZE=50000
B_RESUME=0
B_USE_AUTH_TOKEN=0

BERT_ON_GPT_SV_SCRIPT="../model_tests/likelihood_based_score/scoring_gpt.py"

# ---- scoring_gpt.py outputs *with_sv_llama.csv.gz next to inputs ----
B_WITH_SV_PATTERN="/home/jovyan/LEM_data2/hyunjincho/gpt_samples/20*/*with_sv_llama.csv.gz"

# ---- Compare / summarize on GPT sample set ----
COMPUTE_GPT_MEAN_SCRIPT="../model_tests/likelihood_based_score/compute_gpt_score_mean.py"
BERT_BASE_FOR_GPT_COMPARE="${BERT_PRED_DIR}"   # where sv_summary_llama_full_bert_*.csv.gz live
GPT_COMPARE_MAX_MISMATCH_EX=5

# ============================================================
# Print config
# ============================================================
section "CONFIG"
ts "Running (A) FULL-DATA SV pipeline + (B) BERT-only SV on GPT samples + GPT mean/compare"

echo "PYTHON_BIN      = ${PYTHON_BIN}"
echo "LLAMA_CKPT      = ${LLAMA_CKPT}"
echo "YEARS           = ${YEARS[*]:-(all)}"
echo
echo "[PART A] BERT_PRED_DIR = ${BERT_PRED_DIR}"
echo "[PART A] W2V_PRED_DIR  = ${W2V_PRED_DIR}"
echo "[PART A] COND_PRED_DIR = ${COND_PRED_DIR}"
echo "[PART A] START_FROM    = ${START_FROM}"
echo "[PART A] CAP           = ${CAP}"
echo "[PART A] WINDOW_SIZE   = ${A_WINDOW_SIZE}"
echo "[PART A] SENT_BATCH    = ${A_SENT_BATCH}"
echo
echo "[PART B] IN_PATTERN    = ${B_IN_PATTERN}"
echo "[PART B] WITH_SV_PATTERN = ${B_WITH_SV_PATTERN}"
echo "[PART B] WINDOW_SIZE   = ${B_WINDOW_SIZE}"
echo "[PART B] MAX_LEN       = ${B_MAX_LEN}"
echo "[PART B] SENT_BATCH    = ${B_SENT_BATCH}"
echo "[PART B] CHUNK_SIZE    = ${B_CHUNK_SIZE}"
echo "[PART B] RESUME        = ${B_RESUME}"
echo "[PART B] USE_AUTH_TOKEN= ${B_USE_AUTH_TOKEN}"
echo
echo "[PART B] BERT_BASE_FOR_GPT_COMPARE = ${BERT_BASE_FOR_GPT_COMPARE}"
echo "[PART B] GPT_COMPARE_MAX_MISMATCH_EX = ${GPT_COMPARE_MAX_MISMATCH_EX}"

# ============================================================
# PART A — FULL TEST SET
# ============================================================
section "PART A — FULL TEST SET (BERT / Skill2Vec / ConditionalProb)"
ts "Computing SV(LLaMA) on FULL test-set predictions"

section "[A1/2] Compute SV(LLaMA)"
step "Running scoring.py"
$PYTHON_BIN "${SCORE_FULL_SCRIPT}" \
  --bert_pred_dir "${BERT_PRED_DIR}" \
  --w2v_pred_dir "${W2V_PRED_DIR}" \
  --cond_pred_dir "${COND_PRED_DIR}" \
  --llama_ckpt "${LLAMA_CKPT}" \
  --start-from "${START_FROM}" \
  ${YEARS:+--years "${YEARS[@]}"} \
  --cap "${CAP}" \
  --window-size "${A_WINDOW_SIZE}" \
  --sent-batch "${A_SENT_BATCH}"

section "[A2/2] Compute mean SV(LLaMA)"
step "Running compute_score_mean.py"
$PYTHON_BIN "${MEAN_FULL_SCRIPT}" \
  --bert_pred_dir "${BERT_PRED_DIR}" \
  --w2v_pred_dir "${W2V_PRED_DIR}" \
  --cond_pred_dir "${COND_PRED_DIR}" \
  ${YEARS:+--years "${YEARS[@]}"} \

# ============================================================
# PART B — GPT SAMPLES (BERT-only SV)
# ============================================================
section "PART B — GPT SAMPLES (BERT-only SV on *with_gpt_pred*)"
ts "Computing SV(LLaMA) on sampled GPT files (BERT predictions only)"

step "Building scoring_gpt.py command"
CMD=( $PYTHON_BIN "${BERT_ON_GPT_SV_SCRIPT}"
  --in-pattern "${B_IN_PATTERN}"
  --llama-ckpt "${LLAMA_CKPT}"
  ${YEARS:+--years "${YEARS[@]}"}
  --window-size "${B_WINDOW_SIZE}"
  --max-len "${B_MAX_LEN}"
  --sent-batch "${B_SENT_BATCH}"
  --chunk-size "${B_CHUNK_SIZE}"
)

if [[ "${B_RESUME}" -eq 1 ]]; then
  CMD+=( --resume )
fi
if [[ "${B_USE_AUTH_TOKEN}" -eq 1 ]]; then
  CMD+=( --use-auth-token )
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

section "[B2/2] Summarize / compare GPT-sample SV (compute_gpt_score_mean.py)"
step "Running compute_gpt_score_mean.py"
$PYTHON_BIN "${COMPUTE_GPT_MEAN_SCRIPT}" \
  --gpt-pattern "${B_WITH_SV_PATTERN}" \
  --bert-base "${BERT_BASE_FOR_GPT_COMPARE}" \
  --max-mismatch-examples "${GPT_COMPARE_MAX_MISMATCH_EX}"

# ============================================================
# DONE
# ============================================================
section "PIPELINE COMPLETED SUCCESSFULLY"
ts "All jobs finished"