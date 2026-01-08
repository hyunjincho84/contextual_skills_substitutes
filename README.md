# Contextual Skill Substitutes
_A research codebase for extracting contextual skill substitution patterns from job postings using masked language models._

---

## Overview
We aim to quantify how **skills can substitute each other depending on the occupational and temporal context**, by fine-tuning a BERT-based model on millions of U.S. job postings (2010–2025) and analyzing contextual skill predictions, co-occurrence networks, and real-world trends.

---

## Setup

### 1. Create a virtual environment

All required dependencies are listed in `requirements.txt`.  
You can use either **conda** or **Python venv**.

## Training

### Train all models

After activating the virtual environment, run the following command to train all three models (BERT, Skill2vec, conditional probability):

```bash
 ./command/train_three_models.sh

## Prediction

### Run predictions for all models (BERT, Skill2vec, conditional probability, GPT-5.1)

After training is complete, generate predictions from **all models** by running:

```bash
./command/pred_all.sh

### Likelihood-based Scoring

To compute **likelihood-based validation scores** for the predicted substitute skills, run:

```bash
./command/run_scoring.sh

This script computes comparative likelihood scores for each model’s predicted substitutes, which serve as the primary quantitative metric in our analysis.

## Findings

### Extracting Contextual Skill Substitutes

After computing likelihood-based scores, run the following script to **extract final contextual skill substitutes** year-wise and industry-wise:

```bash
./command/find_substitutes.sh