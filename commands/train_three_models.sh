#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# ========================== CONFIG ==========================
# Edit the values below before running this script.
# This environment contains torch, transformers, pandas, and gensim together.
PYTHON_BIN="/home/jovyan/LEM_data2/hyunjincho/env/lem/bin/python"
BASE_DATA_DIR="/home/jovyan/LEM_data3/hyunjincho/LEM1"
DATA_ROOT="${BASE_DATA_DIR}/all_train_test"
TRAIN_ROOT="${DATA_ROOT}/train"
VOCAB_PATH="${BASE_DATA_DIR}/all_filterd/skill2idx.json"
INPUT_PATTERN="preprocessed_*.csv.gz"

BERT_MODEL_NAME="${BASE_DATA_DIR}/bert_pretrained"
BERT_CHECKPOINT_DIR="${BASE_DATA_DIR}/models/bert_fine_tuned_model"
BERT_BEST_MODEL_PATH="${BERT_CHECKPOINT_DIR}/best_model.pt"
BERT_RESUME_FROM=""
BERT_MAX_LENGTH=256
BERT_NPROC_PER_NODE=2
# Optimized for 2 x A100-SXM4-40GB at max length 256.
# Effective global batch = 128 x 2 GPUs = 256.
BERT_BATCH_SIZE=128
BERT_EPOCHS=3
BERT_LEARNING_RATE="4e-5"
BERT_WEIGHT_DECAY="0.01"
BERT_NUM_WORKERS=16
BERT_SHUFFLE_BUFFER_SIZE=10000
BERT_PREFETCH_FACTOR=4
BERT_PIN_MEMORY="true"
BERT_MIXED_PRECISION="bf16"  # no | bf16 | fp16
BERT_FUSED_OPTIMIZER="true"
BERT_TF32="true"
BERT_ATTENTION_IMPLEMENTATION="sdpa"  # eager | sdpa
BERT_SEED=42
BERT_DEVICE="auto"
BERT_LOSS_METRICS_PATH="${BERT_CHECKPOINT_DIR}/training_metrics.csv"
BERT_LOSS_PLOT_PATH="${BERT_CHECKPOINT_DIR}/training_loss.png"
BERT_LOSS_LOG_STEPS=500
BERT_LOSS_PLOT_EVERY_LOGS=10
BERT_LOSS_SMOOTHING_WINDOW=20

SKILL2VEC_PROGRESS_EVERY=100
SKILL2VEC_JOINER="_"
SKILL2VEC_VECTOR_SIZE=300
SKILL2VEC_WINDOW=50
SKILL2VEC_MIN_COUNT=1
SKILL2VEC_SG=1
SKILL2VEC_NEGATIVE=10
SKILL2VEC_EPOCHS=5
SKILL2VEC_WORKERS=8
SKILL2VEC_SEED=42
SKILL2VEC_OUTPUT_DIR="${BASE_DATA_DIR}/models/skill2vec_model"
SKILL2VEC_MODEL_NAME=""
SKILL2VEC_RESUME="true"
SKILL2VEC_EXPORT_TOPN=0
SKILL2VEC_CORPUS_PATH=""
SKILL2VEC_REBUILD_CORPUS="false"

CONDPROB_OUTPUT_DIR="${BASE_DATA_DIR}/models/condprob_model"
CONDPROB_CHUNK_SIZE=200000
CONDPROB_WORKERS=8
CONDPROB_PROGRESS_EVERY=100
CONDPROB_COMPRESSION_LEVEL=1
CONDPROB_MIN_EDGE_COUNT=1
CONDPROB_TOPN_PER_SKILL=0
CONDPROB_SMOOTH_ALPHA="0.0"
# ============================================================

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
#  - ${BASE_DATA_DIR}/all_train_test/train
#  - ${BASE_DATA_DIR}/all_filterd/skill2idx.json
#  - Base BERT model defaults to ${BASE_DATA_DIR}/bert_pretrained
#
# Output:
#  - ${BASE_DATA_DIR}/models/bert_fine_tuned_model/best_model.pt
#  - ${BASE_DATA_DIR}/models/skill2vec_model
#  - ${BASE_DATA_DIR}/models/condprob_model
# ============================================================

echo "==== [1/3] Training BERT ===="
echo "BERT_NPROC_PER_NODE = ${BERT_NPROC_PER_NODE}"
echo "BERT_BATCH_SIZE/GPU = ${BERT_BATCH_SIZE}"
echo "BERT_GLOBAL_BATCH   = $((BERT_BATCH_SIZE * BERT_NPROC_PER_NODE))"
echo "BERT_LEARNING_RATE  = ${BERT_LEARNING_RATE}"
echo "BERT_WEIGHT_DECAY   = ${BERT_WEIGHT_DECAY}"
echo "BERT_PRECISION      = ${BERT_MIXED_PRECISION}"
echo "BERT_ATTENTION_IMPL = ${BERT_ATTENTION_IMPLEMENTATION}"

if ! [[ "${BERT_NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BERT_NPROC_PER_NODE must be a positive integer." >&2
  exit 2
fi

if [[ "${BERT_NPROC_PER_NODE}" == "1" ]]; then
  bert_launcher=("${PYTHON_BIN}")
else
  bert_launcher=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --standalone
    --nproc_per_node "${BERT_NPROC_PER_NODE}"
  )
fi

bert_command=(
  "${bert_launcher[@]}" "${REPO_ROOT}/model_trains/fine_tune_bert.py"
  --data-root "${DATA_ROOT}" \
  --train-dir "${TRAIN_ROOT}" \
  --vocab-path "${VOCAB_PATH}" \
  --pattern "${INPUT_PATTERN}" \
  --model-name "${BERT_MODEL_NAME}" \
  --checkpoint-dir "${BERT_CHECKPOINT_DIR}" \
  --best-model-path "${BERT_BEST_MODEL_PATH}" \
  --max-length "${BERT_MAX_LENGTH}" \
  --batch-size "${BERT_BATCH_SIZE}" \
  --epochs "${BERT_EPOCHS}" \
  --learning-rate "${BERT_LEARNING_RATE}" \
  --weight-decay "${BERT_WEIGHT_DECAY}" \
  --num-workers "${BERT_NUM_WORKERS}" \
  --shuffle-buffer-size "${BERT_SHUFFLE_BUFFER_SIZE}" \
  --prefetch-factor "${BERT_PREFETCH_FACTOR}" \
  --pin-memory "${BERT_PIN_MEMORY}" \
  --mixed-precision "${BERT_MIXED_PRECISION}" \
  --fused-optimizer "${BERT_FUSED_OPTIMIZER}" \
  --tf32 "${BERT_TF32}" \
  --attention-implementation "${BERT_ATTENTION_IMPLEMENTATION}" \
  --seed "${BERT_SEED}" \
  --loss-metrics-path "${BERT_LOSS_METRICS_PATH}" \
  --loss-plot-path "${BERT_LOSS_PLOT_PATH}" \
  --loss-log-steps "${BERT_LOSS_LOG_STEPS}" \
  --loss-plot-every-logs "${BERT_LOSS_PLOT_EVERY_LOGS}" \
  --loss-smoothing-window "${BERT_LOSS_SMOOTHING_WINDOW}" \
  --device "${BERT_DEVICE}"
)
if [[ -n "${BERT_RESUME_FROM}" ]]; then
  bert_command+=(--resume-from "${BERT_RESUME_FROM}")
fi

"${bert_command[@]}"

echo "==== [2/3] Training Skill2Vec ===="
skill2vec_command=(
  "${PYTHON_BIN}" "${REPO_ROOT}/model_trains/skill2vec.py"
  --data-root "${DATA_ROOT}" \
  --train-root "${TRAIN_ROOT}" \
  --vocab-path "${VOCAB_PATH}" \
  --pattern "${INPUT_PATTERN}" \
  --progress-every "${SKILL2VEC_PROGRESS_EVERY}" \
  --joiner "${SKILL2VEC_JOINER}" \
  --vector-size "${SKILL2VEC_VECTOR_SIZE}" \
  --window "${SKILL2VEC_WINDOW}" \
  --min-count "${SKILL2VEC_MIN_COUNT}" \
  --sg "${SKILL2VEC_SG}" \
  --negative "${SKILL2VEC_NEGATIVE}" \
  --epochs "${SKILL2VEC_EPOCHS}" \
  --workers "${SKILL2VEC_WORKERS}" \
  --seed "${SKILL2VEC_SEED}" \
  --output-dir "${SKILL2VEC_OUTPUT_DIR}" \
  --model-name "${SKILL2VEC_MODEL_NAME}" \
  --resume "${SKILL2VEC_RESUME}" \
  --export-topn "${SKILL2VEC_EXPORT_TOPN}" \
  --rebuild-corpus "${SKILL2VEC_REBUILD_CORPUS}"
)
if [[ -n "${SKILL2VEC_CORPUS_PATH}" ]]; then
  skill2vec_command+=(--corpus-path "${SKILL2VEC_CORPUS_PATH}")
fi

"${skill2vec_command[@]}"

echo "==== [3/3] Training conditional probability ===="
"${PYTHON_BIN}" "${REPO_ROOT}/model_trains/conditional_prob.py" \
  --train-root "${TRAIN_ROOT}" \
  --vocab-path "${VOCAB_PATH}" \
  --output-dir "${CONDPROB_OUTPUT_DIR}" \
  --pattern "${INPUT_PATTERN}" \
  --chunk-size "${CONDPROB_CHUNK_SIZE}" \
  --workers "${CONDPROB_WORKERS}" \
  --progress-every "${CONDPROB_PROGRESS_EVERY}" \
  --compression-level "${CONDPROB_COMPRESSION_LEVEL}" \
  --min-edge-count-to-save "${CONDPROB_MIN_EDGE_COUNT}" \
  --topn-per-s1-to-save "${CONDPROB_TOPN_PER_SKILL}" \
  --smooth-alpha "${CONDPROB_SMOOTH_ALPHA}"
