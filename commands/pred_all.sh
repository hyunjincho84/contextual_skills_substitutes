#!/bin/bash
set -e

echo "==== [1/3] Predict using BERT ===="
python3 ../model_tests/test_bert.py

echo "==== [2/3] Predict using Skill2Vec ===="
python3 ../model_tests/test_skill2vec.py

echo "==== [3/3] Predict using conditional probability ===="
python3 ../model_tests/test_conditional_prob.py
