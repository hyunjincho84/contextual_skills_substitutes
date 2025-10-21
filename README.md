# Contextual Skill Substitutes
_A research codebase for extracting contextual skill substitution patterns from job postings using masked language models._

---

## Overview
We aim to quantify how **skills can substitute each other depending on the occupational and temporal context**, by fine-tuning a BERT-based model on millions of U.S. job postings (2010–2025) and analyzing contextual skill predictions, co-occurrence networks, and real-world trends.

---

## Repository Structure
```
contextual_skills_substitutes/
│
├── exposure_gain/                  # External validation using Google Trends data
│   ├── get_graph.py                # Visualizes correlations between model scores and trend data
│   ├── get_margin.py               # Computes margin = (trend_diff - model_loss)
│   ├── get_trend.py                # Fetches and aggregates Google Trends time-series
│
├── model_tests/                    # Model evaluation and skill substitution testing
│   ├── scoring.py                  # Calculates Contextual Substitute Validation
│   ├── test_bert.py                # Tests our model
│   ├── test_conditional_prob.py    # Tests conditional probablility
│   ├── test_skill2vec.py           # Tests skill2vec
│
├── model_trains/                   # Training and pretraining modules for skill models
│   ├── conditional_prob.py         # Trains conditional probablility
│   ├── fine_tune_bert.py           # Fine-tunes our model
│   ├── model.py                    # Defines model architectures
│   ├── preprocess.py               # Preprocessing and skill masking from job postings
│   ├── pretrain.py                 # Domain-specific BERT pretraining on job posting corpus
│   ├── skill2vec.py                # Trains Skill2Vec
│
├── substitute_by_area/             # Occupational group–wise substitution analysis
│   └── areawise_substitute.py      # Aggregates substitutes by SOC MajorGroup
│
├── substitute_by_time/             # Temporal substitution dynamics (yearly evolution)
│   └── yearwise_top5_substitutes.py# Finds year-wise Top-5 contextual substitutes for a skill
│
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```