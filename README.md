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

### Train all models (`train_three_models.sh`)

After activating the virtual environment, run the following command:

```bash
bash ./train_three_models.sh