#!/bin/bash
set -e

# ---------- helpers ----------
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

step () {
  echo -e "${GREEN}➤ $1${RESET}"
}

warn () {
  echo -e "${YELLOW}⚠ $1${RESET}"
}

ts () {
  echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${RESET} $1"
}

# ---------- pipeline ----------
section "FULL TEST SET PREDICTION PIPELINE"

ts "Starting full evaluation pipeline"

section "[1/2] Predicting on FULL test data"

step "[1/3] BERT"
python3 ../model_tests/test_bert.py

step "[2/3] Skill2Vec"
python3 ../model_tests/test_skill2vec.py

step "[3/3] Conditional Probability"
python3 ../model_tests/test_conditional_prob.py

section "[2/2] GPT-5.1 Evaluation"

step "Sampling examples for GPT-5.1"
python3 ../model_tests/get_samples_for_gpt.py

step "[1/2] BERT prediction on GPT samples"
python3 pred_bert_with_gpt_samples.py \
  --in-root /home/jovyan/LEM_data2/hyunjincho/gpt_samples \
  --out-root /home/jovyan/LEM_data2/hyunjincho/bert_pred_new/gpt_sample \
  --model-name /home/jovyan/LEM_data2/hyunjincho/bert_pretrained/checkpoint-165687 \
  --best-model-pt "/home/jovyan/LEM_data2/hyunjincho/checkpoints(www)_new/best_model.pt" \
  --vocab-path /home/jovyan/LEM_data2/hyunjincho/preprocessed_www_new/skill2idx.json \
  --topk 5 \
  --batch-size 64

step "[2/2] GPT-5.1 prediction"
python3 ../model_tests/test_gpt.py

section "PIPELINE COMPLETED SUCCESSFULLY"
ts "All jobs finished"