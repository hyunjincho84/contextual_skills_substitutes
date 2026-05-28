#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ============================================================
# SPARSE AUTOENCODER TRAINING PIPELINE
#
# Purpose:
#  - Train one sparse autoencoder (SAE) per BERT layer using masked-token
#    representations from the fine-tuned BERT model.
#  - The trained SAE checkpoints are used by later SAE visualization scripts.
#
# Runs:
#  1) sparse_auto_encoder/train_sae.py
#     - Loads training masked sentences, the skill vocabulary, and the
#       fine-tuned BERT checkpoint.
#     - Freezes BERT and trains SAE models for layers 1-12.
#
# Input:
#  - Training files: ${BASE_DATA_DIR}/preprocessed_www_new/train
#  - Skill vocab:    ${BASE_DATA_DIR}/preprocessed_www_new/skill2idx.json
#  - BERT model:     ${BASE_DATA_DIR}/bert_pretrained
#  - Best model pt:  ${BASE_DATA_DIR}/checkpoints(www)_new/best_model.pt
#
# Output:
#  - ${BASE_DATA_DIR}/sae_layerwise_out_8192/layer_01/sae_best.pt
#  - ...
#  - ${BASE_DATA_DIR}/sae_layerwise_out_8192/layer_12/sae_best.pt
#  - Each layer directory also contains sae_last.pt.
# ============================================================

PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_DATA_DIR="${BASE_DATA_DIR:-/home/jovyan/LEM_data2/data}"
export BASE_DATA_DIR
cd "${REPO_ROOT}"
SAE_SCRIPT="${REPO_ROOT}/sparse_auto_encoder/train_sae.py"

"${PYTHON_BIN}" "${SAE_SCRIPT}"
