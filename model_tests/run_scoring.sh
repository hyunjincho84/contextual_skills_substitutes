#!/bin/bash
set -e  # stop immediately if any error occurs

# ============================================================
# USER CONFIGURATION (CHANGE THESE FOR YOUR ENVIRONMENT)
# ------------------------------------------------------------
# NOTE:
#   These paths reflect the author's local setup.
#   If you clone this repository, PLEASE MODIFY the variables
#   below to match your own directory structure.
# ============================================================

# --- model prediction roots (BERT/W2V/COND) ---
DEFAULT_BERT_PRED_DIR="/home/jovyan/LEM_data2/hyunjincho/bert_pred_new/pred"
DEFAULT_W2V_PRED_DIR="/home/jovyan/LEM_data2/hyunjincho/skill2vec_pred_new/pred"
DEFAULT_COND_PRED_DIR="/home/jovyan/LEM_data2/hyunjincho/condprob_pred_new/pred"

# --- HuggingFace LLaMA checkpoint ---
DEFAULT_LLAMA_CHECKPOINT="meta-llama/Llama-3.2-3B"

# --- GPT I/O patterns ---
DEFAULT_GPT_IN_PATTERN="/home/jovyan/LEM_data2/hyunjincho/gpt_samples/20*/*with_gpt_pred.csv*"
DEFAULT_GPT_WITH_SV_PATTERN="/home/jovyan/LEM_data2/hyunjincho/gpt_samples/20*/*with_sv_llama.csv.gz"

# For compute_gpt_score_mean.py: BERT base dir (sv_summary files live here)
DEFAULT_BERT_BASE_FOR_GPT_COMPARE="/home/jovyan/LEM_data2/hyunjincho/bert_pred_new/pred"


# ---------------- runtime options ----------------
PYTHON_BIN="python3"

# score_with_llama.py options
START_FROM="all"     # all | bert,bert_freezed,w2v,conditional
CAP=8000
WINDOW_SIZE=256
SENT_BATCH=4

# Restrict years (leave empty = all)
YEARS=()             # e.g. YEARS=("2018" "2019")

# scoring_gpt.py options
GPT_WINDOW_SIZE=256
GPT_MAX_LEN=512
GPT_SENT_BATCH=4
GPT_CHUNK_SIZE=50000
GPT_RESUME=0
GPT_USE_AUTH_TOKEN=0

# compute_gpt_score_mean.py options
GPT_COMPARE_MAX_MISMATCH_EX=5
# -------------------------------------------------


echo "==== [paths] ===="
echo "BERT_PRED_DIR              = ${DEFAULT_BERT_PRED_DIR}"
echo "W2V_PRED_DIR               = ${DEFAULT_W2V_PRED_DIR}"
echo "COND_PRED_DIR              = ${DEFAULT_COND_PRED_DIR}"
echo "LLAMA_CHECKPOINT           = ${DEFAULT_LLAMA_CHECKPOINT}"
echo "GPT_IN_PATTERN             = ${DEFAULT_GPT_IN_PATTERN}"
echo "GPT_WITH_SV_PATTERN        = ${DEFAULT_GPT_WITH_SV_PATTERN}"
echo "BERT_BASE_FOR_GPT_COMPARE  = ${DEFAULT_BERT_BASE_FOR_GPT_COMPARE}"
echo "YEARS                      = ${YEARS[*]:-(all)}"
echo


echo "==== [1/4] Running score_with_llama.py ===="
$PYTHON_BIN scoring.py \
  --bert_pred_dir "${DEFAULT_BERT_PRED_DIR}" \
  --w2v_pred_dir "${DEFAULT_W2V_PRED_DIR}" \
  --cond_pred_dir "${DEFAULT_COND_PRED_DIR}" \
  --llama_ckpt "${DEFAULT_LLAMA_CHECKPOINT}" \
  --start-from "${START_FROM}" \
  ${YEARS:+--years "${YEARS[@]}"} \
  --cap "${CAP}" \
  --window-size "${WINDOW_SIZE}" \
  --sent-batch "${SENT_BATCH}"


echo
echo "==== [2/4] Running compute_llama_mean.py ===="
$PYTHON_BIN compute_score_mean.py \
  --bert_pred_dir "${DEFAULT_BERT_PRED_DIR}" \
  --w2v_pred_dir "${DEFAULT_W2V_PRED_DIR}" \
  --cond_pred_dir "${DEFAULT_COND_PRED_DIR}" \
  ${YEARS:+--years "${YEARS[@]}"} \


echo
echo "==== [3/4] Running scoring_gpt.py ===="
CMD_GPT=( $PYTHON_BIN scoring_gpt.py
  --in-pattern "${DEFAULT_GPT_IN_PATTERN}"
  --llama-ckpt "${DEFAULT_LLAMA_CHECKPOINT}"
  ${YEARS:+--years "${YEARS[@]}"}
  --window-size "${GPT_WINDOW_SIZE}"
  --max-len "${GPT_MAX_LEN}"
  --sent-batch "${GPT_SENT_BATCH}"
  --chunk-size "${GPT_CHUNK_SIZE}"
)
if [[ "${GPT_RESUME}" -eq 1 ]]; then
  CMD_GPT+=( --resume )
fi
if [[ "${GPT_USE_AUTH_TOKEN}" -eq 1 ]]; then
  CMD_GPT+=( --use-auth-token )
fi
echo "Running: ${CMD_GPT[*]}"
"${CMD_GPT[@]}"


echo
echo "==== [4/4] Running compute_gpt_score_mean.py ===="
$PYTHON_BIN compute_gpt_score_mean.py \
  --gpt-pattern "${DEFAULT_GPT_WITH_SV_PATTERN}" \
  --bert-base "${DEFAULT_BERT_BASE_FOR_GPT_COMPARE}" \
  --max-mismatch-examples "${GPT_COMPARE_MAX_MISMATCH_EX}"


echo "==== done ===="