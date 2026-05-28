#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_DATA_DIR="${BASE_DATA_DIR:-/home/jovyan/LEM_data2/data}"
export BASE_DATA_DIR

# ============================================================
# PREPROCESSING PIPELINE
#
# Purpose:
#  - Convert raw monthly job-posting CSV(.gz) files into masked-skill
#    train/test/findings datasets used by all downstream models.
#
# Runs:
#  1) model_trains/preprocess.py
#     - Scans raw posting files under RAW_INPUT_ROOT.
#     - Extracts sentences containing target IT skills.
#     - Creates one [MASK] sample per matched skill.
#     - Writes train/test/findings splits and skill2idx.json.
#
# Input:
#  - RAW_INPUT_ROOT:  /home/jovyan/LEM_data/us/csv/fortnightly/all/20250607 by default
#  - USED_FILES_CSV: ${BASE_DATA_DIR}/bert_pretrained/used_files.csv by default
#  - TARGET_SKILLS_CSV: ./target_skills.csv by default
#
# Output:
#  - PREPROCESSED_ROOT: ${BASE_DATA_DIR}/preprocessed_www_new by default
#  - sampled_files_train.csv / sampled_files_test.csv / sampled_files_findings.csv
#  - preprocess_global_log.txt
#  - skill2idx.json
# ============================================================

RAW_INPUT_ROOT="${RAW_INPUT_ROOT:-/home/jovyan/LEM_data/us/csv/fortnightly/all/20250607}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${BASE_DATA_DIR}/preprocessed_www_new}"
USED_FILES_CSV="${USED_FILES_CSV:-${BASE_DATA_DIR}/bert_pretrained/used_files.csv}"
TARGET_SKILLS_CSV="${TARGET_SKILLS_CSV:-${REPO_ROOT}/target_skills.csv}"
export RAW_INPUT_ROOT PREPROCESSED_ROOT USED_FILES_CSV TARGET_SKILLS_CSV

cd "${REPO_ROOT}"

echo "==== Preprocessing job postings ===="
echo "RAW_INPUT_ROOT     = ${RAW_INPUT_ROOT}"
echo "PREPROCESSED_ROOT  = ${PREPROCESSED_ROOT}"
echo "USED_FILES_CSV     = ${USED_FILES_CSV}"
echo "TARGET_SKILLS_CSV = ${TARGET_SKILLS_CSV}"

"${PYTHON_BIN}" "${REPO_ROOT}/model_trains/preprocess.py"
