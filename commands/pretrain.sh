#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_DATA_DIR="${BASE_DATA_DIR:-/home/jovyan/LEM_data2/data}"
export BASE_DATA_DIR
cd "${REPO_ROOT}"

# ============================================================
# BERT MLM PRETRAINING PIPELINE
#
# Purpose:
#  - Pretrain a BERT masked-language model on sampled raw job-posting files.
#  - Save the sampled raw file list used by preprocessing to avoid overlap.
#
# Runs:
#  1) model_trains/pretrain.py
#     - Samples FILES_PER_MONTH raw CSV.GZ files per month from RAW_INPUT_ROOT.
#     - Writes used_files.csv.
#     - Runs MLM pretraining with HuggingFace Trainer.
#
# Input:
#  - RAW_INPUT_ROOT: /home/jovyan/LEM_data/us/csv/fortnightly/all/20250607 by default
#  - BERT_BASE_MODEL: bert-base-uncased by default
#
# Output:
#  - BERT_PRETRAIN_DIR: ${BASE_DATA_DIR}/bert_pretrained by default
#  - ${BASE_DATA_DIR}/bert_pretrained/used_files.csv
#  - ${BASE_DATA_DIR}/bert_pretrained/config.json, model weights, and tokenizer files
#  - ${BASE_DATA_DIR}/bert_pretrained/checkpoint-*/
# ============================================================

RAW_INPUT_ROOT="${RAW_INPUT_ROOT:-/home/jovyan/LEM_data/us/csv/fortnightly/all/20250607}"
BERT_BASE_MODEL="${BERT_BASE_MODEL:-bert-base-uncased}"
BERT_PRETRAIN_DIR="${BERT_PRETRAIN_DIR:-${BASE_DATA_DIR}/bert_pretrained}"
export RAW_INPUT_ROOT BERT_BASE_MODEL BERT_PRETRAIN_DIR

echo "==== BERT MLM pretraining ===="
echo "RAW_INPUT_ROOT    = ${RAW_INPUT_ROOT}"
echo "BERT_BASE_MODEL   = ${BERT_BASE_MODEL}"
echo "BERT_PRETRAIN_DIR = ${BERT_PRETRAIN_DIR}"

"${PYTHON_BIN}" "${REPO_ROOT}/model_trains/pretrain.py"
