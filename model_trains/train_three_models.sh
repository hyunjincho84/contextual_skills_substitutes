#!/bin/bash
set -e

echo "==== [1/3] Training BERT ===="
python3 fine_tune_bert.py

echo "==== [2/3] Training Skill2Vec ===="
python3 skill2vec.py

echo "==== [3/3] Training conditional probability ===="
python3 conditional_prob.py
