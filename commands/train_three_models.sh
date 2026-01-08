#!/bin/bash
set -e

echo "==== [1/3] Training BERT ===="
python3 ../model_trains/fine_tune_bert.py

echo "==== [2/3] Training Skill2Vec ===="
python3 ../model_trains/skill2vec.py

echo "==== [3/3] Training conditional probability ===="
python3 ../model_trains/conditional_prob.py
