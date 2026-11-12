#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ============================================================
# SPARSE AUTOENCODER TRAINING PIPELINE
#
# Purpose:
#  - Freeze the fine-tuned BERT encoder.
#  - Extract each selected layer's contextual [MASK] representation.
#  - Train one sparse autoencoder per BERT layer.
#
# Inputs from preprocess.sh and train_three_models.sh:
#  - ${BASE_DATA_DIR}/all_train_test/train/{year}/{yyyy-mm}/preprocessed_*.csv.gz
#  - ${BASE_DATA_DIR}/all_filterd/skill2idx.json
#  - ${BASE_DATA_DIR}/bert_pretrained
#  - ${BASE_DATA_DIR}/models/bert_fine_tuned_model/best_model.pt
#
# Output:
#  - ${BASE_DATA_DIR}/models/sae_layerwise_out/layer_*/sae_best.pt
# ============================================================

# ========================== CONFIG ==========================
# Edit the text values below directly before running this script.
PYTHON_BIN="/home/jovyan/LEM_data2/hyunjincho/env/lem/bin/python"
BASE_DATA_DIR="/home/jovyan/LEM_data3/hyunjincho/LEM"

DATA_ROOT="${BASE_DATA_DIR}/all_train_test"
TRAIN_ROOT="${DATA_ROOT}/train"
INPUT_PATTERN="preprocessed_*.csv.gz"
TRAIN_PATTERN="${TRAIN_ROOT}/[0-9][0-9][0-9][0-9]/[0-9][0-9][0-9][0-9]-[0-9][0-9]/${INPUT_PATTERN}"
# Exact train_rows from ${DATA_ROOT}/split_summary.json.
TOTAL_SAMPLES=27237870
VOCAB_PATH="${BASE_DATA_DIR}/all_filterd/skill2idx.json"

BERT_MODEL_NAME="${BASE_DATA_DIR}/bert_pretrained"
BERT_CHECKPOINT_DIR="${BASE_DATA_DIR}/models/bert_fine_tuned_model"
BERT_BEST_MODEL_PATH="${BERT_CHECKPOINT_DIR}/best_model.pt"
SAE_OUTPUT_DIR="${BASE_DATA_DIR}/models/sae_layerwise_out"

TEXT_COL="masked_sentence"
LAYERS="1-12"
# Match the sequence length used by train_three_models.sh.
MAX_LEN=256
BATCH_SIZE=256
NUM_WORKERS=16
PIN_MEMORY=true
# Workers are created once per epoch by the streaming dataset.
PERSISTENT_WORKERS=false
PREFETCH_FACTOR=4
SHUFFLE_FILES=true
SHUFFLE_ROWS=true
DROP_LAST=false
SEED=42
DEVICE="cuda"

# SAE settings reported for the main analysis.
SAE_HIDDEN=4096
SAE_EPOCHS=3
SAE_LR="1e-4"
L1="1e-3"
L2="1e-4"

USE_AMP=true
AMP_DTYPE="bf16"
ALLOW_TF32=true
FUSED_OPTIMIZER=true
# ============================================================

cd "${REPO_ROOT}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -d "${TRAIN_ROOT}" ]]; then
  echo "Training directory not found: ${TRAIN_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${VOCAB_PATH}" ]]; then
  echo "Vocabulary file not found: ${VOCAB_PATH}" >&2
  exit 2
fi
if [[ ! -d "${BERT_MODEL_NAME}" ]]; then
  echo "BERT model directory not found: ${BERT_MODEL_NAME}" >&2
  exit 2
fi
if [[ ! -f "${BERT_BEST_MODEL_PATH}" ]]; then
  echo "Fine-tuned BERT checkpoint not found: ${BERT_BEST_MODEL_PATH}" >&2
  exit 2
fi

echo "==== Training layer-wise sparse autoencoders ===="
echo "PYTHON_BIN          = ${PYTHON_BIN}"
echo "TRAIN_ROOT          = ${TRAIN_ROOT}"
echo "TRAIN_PATTERN       = ${TRAIN_PATTERN}"
echo "TOTAL_SAMPLES       = ${TOTAL_SAMPLES}"
echo "VOCAB_PATH          = ${VOCAB_PATH}"
echo "BERT_MODEL_NAME     = ${BERT_MODEL_NAME}"
echo "BERT_BEST_MODEL_PATH= ${BERT_BEST_MODEL_PATH}"
echo "SAE_OUTPUT_DIR      = ${SAE_OUTPUT_DIR}"
echo "LAYERS              = ${LAYERS}"
echo "MAX_LEN             = ${MAX_LEN}"
echo "BATCH_SIZE          = ${BATCH_SIZE}"
echo "SAE_HIDDEN          = ${SAE_HIDDEN}"
echo "SAE_EPOCHS          = ${SAE_EPOCHS}"
echo "SAE_LR              = ${SAE_LR}"
echo "L1 / L2             = ${L1} / ${L2}"
echo "AMP_DTYPE           = ${AMP_DTYPE}"
echo "DEVICE              = ${DEVICE}"

sae_command=(
  "${PYTHON_BIN}" "${REPO_ROOT}/sparse_auto_encoder/train_sae.py"
  --train-pattern "${TRAIN_PATTERN}"
  --total-samples "${TOTAL_SAMPLES}"
  --vocab-path "${VOCAB_PATH}"
  --model-name "${BERT_MODEL_NAME}"
  --best-model-pt "${BERT_BEST_MODEL_PATH}"
  --out-dir "${SAE_OUTPUT_DIR}"
  --text-col "${TEXT_COL}"
  --layers "${LAYERS}"
  --max-len "${MAX_LEN}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --pin-memory "${PIN_MEMORY}"
  --persistent-workers "${PERSISTENT_WORKERS}"
  --prefetch-factor "${PREFETCH_FACTOR}"
  --shuffle-files "${SHUFFLE_FILES}"
  --shuffle-rows "${SHUFFLE_ROWS}"
  --drop-last "${DROP_LAST}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --sae-hidden "${SAE_HIDDEN}"
  --sae-epochs "${SAE_EPOCHS}"
  --sae-lr "${SAE_LR}"
  --l1 "${L1}"
  --l2 "${L2}"
  --use-amp "${USE_AMP}"
  --amp-dtype "${AMP_DTYPE}"
  --allow-tf32 "${ALLOW_TF32}"
  --fused-optimizer "${FUSED_OPTIMIZER}"
)

"${sae_command[@]}"

echo "==== SAE training completed ===="
echo "Output: ${SAE_OUTPUT_DIR}"
