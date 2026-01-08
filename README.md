# Contextual Skill Substitutes
_A research codebase for extracting contextual skill substitution patterns from job postings using masked language models._

---

## Overview
We aim to quantify how **skills can substitute each other depending on the occupational and temporal context**, by fine-tuning a BERT-based model on millions of U.S. job postings (2010–2025) and analyzing contextual skill predictions, co-occurrence networks, and real-world trends.

---

## Setup

### 1. Create a virtual environment

All required dependencies are listed in `./requirements.txt`.  
You can use either **conda** or **Python venv**.

## Training

### Train all models

After activating the virtual environment, run the following command to train all three models (BERT, Skill2vec, conditional probability):

```bash
 ./command/train_three_models.sh
```
## Prediction

### Run predictions for all models (BERT, Skill2vec, conditional probability, GPT-5.1)

After training is complete, generate predictions from **all models** by running:

```bash
./command/pred_all.sh
```

### Likelihood-based Scoring

To compute **likelihood-based validation scores** for the predicted substitute skills, run:

```bash
./command/run_scoring.sh
```
This script computes comparative likelihood scores for each model’s predicted substitutes, which serve as the primary quantitative metric in our analysis.

## Findings

### Extracting Contextual Skill Substitutes

After computing likelihood-based scores, run the following script to **extract final contextual skill substitutes** year-wise and industry-wise:

```bash
./command/find_substitutes.sh
```

### SAE-based representation analysis

To visualize layer-wise representations using Sparse Autoencoders (SAEs) and visualize contextual clustering patterns, run:

```bash
./command/vis_sae_results.sh
```
This script:
	•	Extracts SAE features from BERT layers
	•	Visualizes UMAP embeddings
	•	Computes element-centric similarity across industries and years

### Overlap analysis across industries and years

To measure and visualize feature overlap patterns across industries and over time, run:

```bash
./command/overlap_graph.sh
```

This script generates:
	•	Industry-wise overlap heatmaps
	•	Year-wise overlap heatmaps
	•	Overlap trend graphs and year-difference analyses

### Feature-level activation visualization

To inspect unioned top-K feature activations and their industry-specific structure, run:

```bash
./command/vis_feature_id.sh
```
This script:
	•	Identifies top-K activated SAE features per industry
	•	Iteratively removes globally common features
	•	Produces unified heatmaps and feature metadata for interpretability analysis

