#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ========================== CONFIG ==========================
# Edit the values below before running this script.
PYTHON_BIN="/home/jovyan/LEM_data2/hyunjincho/env/lem/bin/python"
BASE_DATA_DIR="/home/jovyan/LEM_data3/hyunjincho/LEM1"
TRAIN_SPLIT_ROOT="/home/jovyan/LEM_data3/hyunjincho/LEM1/all_train_test/train"
BERT_BASE_MODEL="bert-base-uncased"
BERT_PRETRAIN_DIR="${BASE_DATA_DIR}/bert_pretrained"
PRETRAIN_PATTERN="preprocessed_*.csv.gz"
PRETRAIN_POSTING_INDEX=""
PRETRAIN_MAX_LENGTH=256
PRETRAIN_RAW_CHUNK_SIZE=100000
PRETRAIN_BODY_START_MARKER="job description:"
PRETRAIN_MLM_PROBABILITY="0.15"
PRETRAIN_EPOCHS=1
PRETRAIN_NPROC_PER_NODE=2
# Optimized for 2 x A100-SXM4-40GB at max length 256.
# Effective global batch = 128 x 2 GPUs = 256.
PRETRAIN_BATCH_SIZE=128
PRETRAIN_LEARNING_RATE="1e-4"
PRETRAIN_MIXED_PRECISION="bf16"  # no | bf16 | fp16
PRETRAIN_OPTIM="adamw_torch_fused"
PRETRAIN_TF32="true"
PRETRAIN_WARMUP_RATIO="0.01"
PRETRAIN_SEED=42
PRETRAIN_SAVE_STEPS=5000
PRETRAIN_SAVE_TOTAL_LIMIT=2
PRETRAIN_LOGGING_STEPS=500
PRETRAIN_DATALOADER_WORKERS=16
PRETRAIN_DATALOADER_PREFETCH_FACTOR=4
PRETRAIN_PROGRESS_EVERY=500
PRETRAIN_REUSE_POSTING_INDEX="true"
PRETRAIN_INDEX_ONLY="false"
# ============================================================

cd "${REPO_ROOT}"

# ============================================================
# BERT MLM PRETRAINING PIPELINE
#
# Purpose:
#  - Pretrain BERT MLM on original body text from train postings only.
#  - Keep every test posting out of domain-adaptive pretraining.
#
# Runs:
#  1) model_trains/pretrain.py
#     - Reads unique (file_path, row_idx) IDs from TRAIN_SPLIT_ROOT.
#     - Recovers each selected posting's body from its original raw CSV.
#     - Uses only text after the first case-insensitive
#       PRETRAIN_BODY_START_MARKER match.
#     - Runs one MLM epoch by default with one body per train posting.
#
# Input:
#  - TRAIN_SPLIT_ROOT:
#      /home/jovyan/LEM_data3/hyunjincho/LEM/all_train_test/train by default
#  - BERT_BASE_MODEL: bert-base-uncased by default
#
# Output under BERT_PRETRAIN_DIR:
#  - used_files.csv: train split/raw source files and posting counts
#  - train_postings.sqlite: exact train posting membership index
#  - pretrain_corpus_summary.json: corpus and training configuration
#  - config.json, model weights, tokenizer files, and checkpoint-*/
# ============================================================

echo "==== BERT MLM pretraining ===="
echo "TRAIN_SPLIT_ROOT            = ${TRAIN_SPLIT_ROOT}"
echo "BERT_BASE_MODEL             = ${BERT_BASE_MODEL}"
echo "BERT_PRETRAIN_DIR           = ${BERT_PRETRAIN_DIR}"
echo "PRETRAIN_PATTERN            = ${PRETRAIN_PATTERN}"
echo "PRETRAIN_POSTING_INDEX      = ${PRETRAIN_POSTING_INDEX:-<output-dir>/train_postings.sqlite}"
echo "PRETRAIN_MAX_LENGTH         = ${PRETRAIN_MAX_LENGTH}"
echo "PRETRAIN_RAW_CHUNK_SIZE     = ${PRETRAIN_RAW_CHUNK_SIZE}"
echo "PRETRAIN_BODY_START_MARKER  = ${PRETRAIN_BODY_START_MARKER}"
echo "PRETRAIN_MLM_PROBABILITY    = ${PRETRAIN_MLM_PROBABILITY}"
echo "PRETRAIN_EPOCHS             = ${PRETRAIN_EPOCHS}"
echo "PRETRAIN_NPROC_PER_NODE     = ${PRETRAIN_NPROC_PER_NODE}"
echo "PRETRAIN_BATCH_SIZE/GPU     = ${PRETRAIN_BATCH_SIZE}"
echo "PRETRAIN_LEARNING_RATE      = ${PRETRAIN_LEARNING_RATE}"
echo "PRETRAIN_MIXED_PRECISION    = ${PRETRAIN_MIXED_PRECISION}"
echo "PRETRAIN_OPTIM              = ${PRETRAIN_OPTIM}"
echo "PRETRAIN_TF32               = ${PRETRAIN_TF32}"
echo "PRETRAIN_WARMUP_RATIO       = ${PRETRAIN_WARMUP_RATIO}"
echo "PRETRAIN_SEED               = ${PRETRAIN_SEED}"
echo "PRETRAIN_SAVE_STEPS         = ${PRETRAIN_SAVE_STEPS}"
echo "PRETRAIN_SAVE_TOTAL_LIMIT   = ${PRETRAIN_SAVE_TOTAL_LIMIT}"
echo "PRETRAIN_LOGGING_STEPS      = ${PRETRAIN_LOGGING_STEPS}"
echo "PRETRAIN_DATALOADER_WORKERS = ${PRETRAIN_DATALOADER_WORKERS}"
echo "PRETRAIN_PREFETCH_FACTOR    = ${PRETRAIN_DATALOADER_PREFETCH_FACTOR}"
echo "PRETRAIN_PROGRESS_EVERY     = ${PRETRAIN_PROGRESS_EVERY}"
echo "PRETRAIN_REUSE_INDEX        = ${PRETRAIN_REUSE_POSTING_INDEX}"
echo "PRETRAIN_INDEX_ONLY         = ${PRETRAIN_INDEX_ONLY}"

for boolean_name in PRETRAIN_TF32 PRETRAIN_REUSE_POSTING_INDEX PRETRAIN_INDEX_ONLY; do
  boolean_value="${!boolean_name}"
  if [[ "${boolean_value}" != "true" && "${boolean_value}" != "false" ]]; then
    echo "${boolean_name} must be true or false." >&2
    exit 2
  fi
done

case "${PRETRAIN_MIXED_PRECISION}" in
  no|bf16|fp16) ;;
  *)
    echo "PRETRAIN_MIXED_PRECISION must be no, bf16, or fp16." >&2
    exit 2
    ;;
esac

if ! [[ "${PRETRAIN_NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PRETRAIN_NPROC_PER_NODE must be a positive integer." >&2
  exit 2
fi

if [[ "${PRETRAIN_INDEX_ONLY}" == "true" || "${PRETRAIN_NPROC_PER_NODE}" == "1" ]]; then
  pretrain_launcher=("${PYTHON_BIN}")
else
  pretrain_launcher=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --standalone
    --nproc_per_node "${PRETRAIN_NPROC_PER_NODE}"
  )
fi

pretrain_command=(
  "${pretrain_launcher[@]}" "${REPO_ROOT}/model_trains/pretrain.py"
  --train-root "${TRAIN_SPLIT_ROOT}"
  --output-dir "${BERT_PRETRAIN_DIR}"
  --model-name "${BERT_BASE_MODEL}"
  --pattern "${PRETRAIN_PATTERN}"
  --max-length "${PRETRAIN_MAX_LENGTH}"
  --raw-chunk-size "${PRETRAIN_RAW_CHUNK_SIZE}"
  --body-start-marker "${PRETRAIN_BODY_START_MARKER}"
  --mlm-probability "${PRETRAIN_MLM_PROBABILITY}"
  --num-train-epochs "${PRETRAIN_EPOCHS}"
  --per-device-train-batch-size "${PRETRAIN_BATCH_SIZE}"
  --learning-rate "${PRETRAIN_LEARNING_RATE}"
  --mixed-precision "${PRETRAIN_MIXED_PRECISION}"
  --optim "${PRETRAIN_OPTIM}"
  --tf32 "${PRETRAIN_TF32}"
  --warmup-ratio "${PRETRAIN_WARMUP_RATIO}"
  --seed "${PRETRAIN_SEED}"
  --save-steps "${PRETRAIN_SAVE_STEPS}"
  --save-total-limit "${PRETRAIN_SAVE_TOTAL_LIMIT}"
  --logging-steps "${PRETRAIN_LOGGING_STEPS}"
  --dataloader-workers "${PRETRAIN_DATALOADER_WORKERS}"
  --dataloader-prefetch-factor "${PRETRAIN_DATALOADER_PREFETCH_FACTOR}"
  --progress-every "${PRETRAIN_PROGRESS_EVERY}"
)

if [[ -n "${PRETRAIN_POSTING_INDEX}" ]]; then
  pretrain_command+=(--posting-index "${PRETRAIN_POSTING_INDEX}")
fi
if [[ "${PRETRAIN_INDEX_ONLY}" == "true" ]]; then
  pretrain_command+=(--index-only)
fi
if [[ "${PRETRAIN_REUSE_POSTING_INDEX}" == "true" ]]; then
  pretrain_command+=(--reuse-posting-index)
fi

"${pretrain_command[@]}"
