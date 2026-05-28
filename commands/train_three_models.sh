#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_DATA_DIR="${BASE_DATA_DIR:-/home/jovyan/LEM_data2/data}"
TARGET_SKILLS_CSV="${TARGET_SKILLS_CSV:-${REPO_ROOT}/target_skills.csv}"
export BASE_DATA_DIR TARGET_SKILLS_CSV
cd "${REPO_ROOT}"

# ============================================================
# MODEL TRAINING PIPELINE
#
# Purpose:
#  - Train the three baseline/prediction models used by later evaluation scripts.
#
# Runs:
#  1) model_trains/fine_tune_bert.py
#     - Fine-tunes BERT for skill substitution prediction.
#  2) model_trains/skill2vec.py
#     - Trains the Skill2Vec baseline.
#  3) model_trains/conditional_prob.py
#     - Trains the conditional probability baseline.
#
# Input:
#  - ${BASE_DATA_DIR}/preprocessed_www_new/train
#  - ${BASE_DATA_DIR}/preprocessed_www_new/skill2idx.json
#  - TARGET_SKILLS_CSV: ./target_skills.csv by default for Skill2Vec/Conditional Probability
#  - Base BERT model defaults to ${BASE_DATA_DIR}/bert_pretrained
#
# Output:
#  - ${BASE_DATA_DIR}/checkpoints(www)_new/best_model.pt
#  - ${BASE_DATA_DIR}/skill2vec_new
#  - ${BASE_DATA_DIR}/condprob_new
# ============================================================

echo "==== [1/3] Training BERT ===="
"${PYTHON_BIN}" "${REPO_ROOT}/model_trains/fine_tune_bert.py"

echo "==== [2/3] Training Skill2Vec ===="
"${PYTHON_BIN}" "${REPO_ROOT}/model_trains/skill2vec.py"

echo "==== [3/3] Training conditional probability ===="
"${PYTHON_BIN}" "${REPO_ROOT}/model_trains/conditional_prob.py"
