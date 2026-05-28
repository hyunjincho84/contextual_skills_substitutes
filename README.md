# Contextual Skill Substitutes

_A research codebase for extracting contextual skill substitution patterns from job postings using masked language models._

---

## Overview

We aim to quantify how **skills can substitute each other depending on the occupational and temporal context**, by fine-tuning a BERT-based model on millions of U.S. job postings (2010-2025) and analyzing contextual skill predictions, semantic-validity scores, SAE representations, and real-world trend signals.

---

## Setup

All required dependencies are listed in `requirements.txt`. You can use either **conda** or **Python venv**.

Before running any pipeline command, make the command scripts executable:

```bash
cd contextual_skills_substitutes
chmod +x commands/*.sh
```

Most command scripts resolve repository paths from the `commands/` directory, so they can be launched from the repository root. Real full-run generated data defaults to `/home/jovyan/LEM_data2/data` and can be overridden with `BASE_DATA_DIR=/path/to/data` or the script-specific environment variables shown in each command header.

Before the first run, place `target_skills.csv` in the repository root or run commands with `TARGET_SKILLS_CSV=/path/to/target_skills.csv`. LLaMA/Qwen/Gemma steps also require Hugging Face access for gated models, GPT prediction requires `OPENAI_API_KEY`, and Google Trends margin analysis requires network access.

---

## Command Pipelines

Each script in `commands/` now has a header documenting its **Purpose**, executed Python files, **Input**, and **Output**. The main workflows are summarized below.

### 1. Pretrain BERT MLM

Pretrain a BERT masked-language model on sampled raw job-posting files. This step also writes `used_files.csv`, which preprocessing uses to avoid reusing pretraining files in downstream splits.

```bash
bash commands/pretrain.sh
```

Inputs:
- `RAW_INPUT_ROOT`, default `/home/jovyan/LEM_data/us/csv/fortnightly/all/20250607`
- `BERT_BASE_MODEL`, default `bert-base-uncased`

Runs:
- `model_trains/pretrain.py`

Outputs:
- `${BASE_DATA_DIR}/bert_pretrained/used_files.csv`
- `${BASE_DATA_DIR}/bert_pretrained/config.json`, model weights, and tokenizer files
- `${BASE_DATA_DIR}/bert_pretrained/checkpoint-*/`
- `${BASE_DATA_DIR}/bert_pretrained/logs`

### 2. Preprocess Job Postings

Create the masked-skill train/test/findings datasets used by every downstream model.

```bash
bash commands/preprocess.sh
```

Inputs:
- `RAW_INPUT_ROOT`, default `/home/jovyan/LEM_data/us/csv/fortnightly/all/20250607`
- `USED_FILES_CSV`, default `${BASE_DATA_DIR}/bert_pretrained/used_files.csv`
- `TARGET_SKILLS_CSV`, default `./target_skills.csv` in the repository root

Runs:
- `model_trains/preprocess.py`

Outputs:
- `${BASE_DATA_DIR}/preprocessed_www_new/train`
- `${BASE_DATA_DIR}/preprocessed_www_new/test`
- `${BASE_DATA_DIR}/preprocessed_www_new/findings`
- `${BASE_DATA_DIR}/preprocessed_www_new/skill2idx.json`
- `${BASE_DATA_DIR}/preprocessed_www_new/sampled_files_*.csv`

### 3. Train Models

Train the three prediction baselines: BERT, Skill2Vec, and conditional probability.

```bash
bash commands/train_three_models.sh
```

Runs:
- `model_trains/fine_tune_bert.py`
- `model_trains/skill2vec.py`
- `model_trains/conditional_prob.py`

Inputs include `${BASE_DATA_DIR}/bert_pretrained`, `${BASE_DATA_DIR}/preprocessed_www_new/train`, `${BASE_DATA_DIR}/preprocessed_www_new/skill2idx.json`, and `TARGET_SKILLS_CSV` for Skill2Vec/Conditional Probability. Outputs are written under `${BASE_DATA_DIR}` by default: `${BASE_DATA_DIR}/checkpoints(www)_new`, `${BASE_DATA_DIR}/skill2vec_new`, and `${BASE_DATA_DIR}/condprob_new`.

### 4. Run Baseline Predictions

Generate full-test predictions from BERT, Skill2Vec, and Conditional Probability. GPT sampling/prediction is intentionally not run here because it uses `sv_summary_*` outputs from the next LLaMA scoring step.

```bash
bash commands/pred_all.sh
```

Runs:
- `model_tests/test_bert.py`
- `model_tests/test_skill2vec.py`
- `model_tests/test_conditional_prob.py`

Outputs are written under `${BASE_DATA_DIR}` by default:
- `${BASE_DATA_DIR}/bert_pred_new/pred`
- `${BASE_DATA_DIR}/skill2vec_pred_new/pred`
- `${BASE_DATA_DIR}/condprob_pred_new/pred`

### 5. LLaMA-Based Model SV Scoring

Compute LLaMA-based semantic-validity scores for the baseline prediction outputs. This creates the `sv_summary_*` files used by GPT sampling.

```bash
bash commands/run_scoring.sh
```

Inputs:
- `BERT_PRED_DIR`, default `${BASE_DATA_DIR}/bert_pred_new/pred`
- `W2V_PRED_DIR`, default `${BASE_DATA_DIR}/skill2vec_pred_new/pred`
- `COND_PRED_DIR`, default `${BASE_DATA_DIR}/condprob_pred_new/pred`

Runs:
- `model_tests/likelihood_based_score/scoring.py`
- `model_tests/likelihood_based_score/compute_score_mean.py`

Outputs:
- `${BASE_DATA_DIR}/bert_pred_new/pred/20*/sv_summary_llama_full_bert_*.csv.gz`
- `${BASE_DATA_DIR}/skill2vec_pred_new/pred/20*/sv_summary_llama_full_w2v_*.csv.gz`
- `${BASE_DATA_DIR}/condprob_pred_new/pred/20*/sv_summary_llama_full_conditional_*.csv.gz`
- Model-level mean SV summaries

### 6. GPT Sampling And Prediction

Sample GPT evaluation rows from BERT `sv_summary_*` files and run GPT predictions. This step must run after `commands/run_scoring.sh`.

```bash
bash commands/gpt_sampling_pred.sh
```

Inputs:
- `BERT_SV_ROOT`, default `${BASE_DATA_DIR}/bert_pred_new/pred`
- `SKILL2IDX`, default `${BASE_DATA_DIR}/preprocessed_www_new/skill2idx.json`
- BERT SV files matching `${BERT_SV_ROOT}/20*/sv_summary_llama_full_bert_*.csv.gz`

Runs:
- `model_tests/get_samples_for_gpt.py`
- `model_tests/test_gpt.py`

Outputs:
- `${BASE_DATA_DIR}/gpt_samples/20*/gpt_unique_samples_*_global*.csv.gz`
- `${BASE_DATA_DIR}/gpt_samples/20*/gpt_unique_samples_*_global*_with_gpt_pred.csv.gz`

### 7. GPT SV Scoring And Compare

Optionally score GPT predictions with LLaMA SV and compare the sampled rows against the corresponding BERT `sv_summary_*` rows.

```bash
bash commands/run_gpt_scoring.sh
```

Runs:
- `model_tests/likelihood_based_score/scoring_gpt.py`
- `model_tests/likelihood_based_score/compute_gpt_score_mean.py`

Outputs:
- `${BASE_DATA_DIR}/gpt_samples/20*/*with_sv_llama.csv.gz`
- GPT-vs-BERT sampled-row mean SV comparison printed by `compute_gpt_score_mean.py`

### 8. Qwen/Gemma Judge Validation

Evaluate prediction outputs with Qwen2.5 and Gemma 2 judge models. Use `debug` for a small run or `full` for full prediction directories.

```bash
bash commands/eval_gemma_qwen_full.sh debug
bash commands/eval_gemma_qwen_full.sh full
```

Inputs:
- `BERT_PRED_DIR`, default `${BASE_DATA_DIR}/bert_pred_new/pred`
- `W2V_PRED_DIR`, default `${BASE_DATA_DIR}/skill2vec_pred_new/pred`
- `COND_PRED_DIR`, default `${BASE_DATA_DIR}/condprob_pred_new/pred`
- `QWEN_CKPT`, default `Qwen/Qwen2.5-3B-Instruct`
- `GEMMA_CKPT`, default `google/gemma-2-2b-it`

Runs:
- `model_tests/likelihood_valid_with_other_models/valid_other_models.py`
- `model_tests/likelihood_valid_with_other_models/get_mean.py`

Outputs:
- `${BASE_DATA_DIR}/qwen25_3b_eval_full`
- `${BASE_DATA_DIR}/gemma2_2b_eval_full`
- `${BASE_DATA_DIR}/command_outputs/mean_summary_qwen25_gemma_*.csv`
- `${BASE_DATA_DIR}/command_outputs/logs/eval_gemma_qwen_*.log`

### 9. Top-2 Robustness Analysis

Generate BERT top-10 predictions, sample substitute variants, score them with LLaMA SV scoring, and summarize mean robustness scores.

```bash
bash commands/run_top2_pipeline.sh debug
bash commands/run_top2_pipeline.sh full
```

Runs:
- `model_tests/top2_robustness/pred_bert_top10.py`
- `model_tests/top2_robustness/sampling.py`
- `model_tests/top2_robustness/scoring_samplings.py`
- `model_tests/top2_robustness/get_mean.py`

Outputs include top-10 summaries under `${BASE_DATA_DIR}/bert_pred_new_top10_from_summary`, sampling/scored CSV.GZ files under `${BASE_DATA_DIR}/bert_sampling_variants_from_summary`, mean summaries printed by `get_mean.py`, and logs under `${BASE_DATA_DIR}/command_outputs/logs`.

### 10. Extract Contextual Skill Substitutes

Extract final contextual skill substitutes by SOC2 area and by year.

```bash
bash commands/find_substitutes.sh
```

Inputs:
- `TARGET_SKILL`, default `python`
- `PRED_ROOT_SOC2`, default `${BASE_DATA_DIR}/bert_pred_new/pred`
- `PRED_ROOT_YEAR`, default `${BASE_DATA_DIR}/bert_pred_new/pred`

Runs:
- `substitute_by_area/areawise_substitutes.py`
- `substitute_by_time/yearwise_substitutes.py`

Outputs:
- `${BASE_DATA_DIR}/substitute_by_area/subs_python_by_soc2.csv`
- `${BASE_DATA_DIR}/substitute_by_time/python_yearwise_top5.csv`

### 11. Exposure Gain / Margin Analysis

Combine substitute-pair counts with Google Trends signals and generate posting-level margin visualizations for 2025.

```bash
bash commands/get_margin.sh
```

Inputs:
- `PRED_DIR`, default `${BASE_DATA_DIR}/bert_pred_new/pred/2025`
- `PREPROCESSED_ROOT`, default `${BASE_DATA_DIR}/preprocessed_www_new/test/2025`

Runs:
- `exposure_gain/get_count_by_pair.py`
- `exposure_gain/get_trend.py`
- `exposure_gain/get_graph.py`

Outputs:
- `${BASE_DATA_DIR}/margin/counts_by_pair_2025.csv`
- `${BASE_DATA_DIR}/exposure_gain/counts_by_pair_with_trends_monthly.csv`
- `${BASE_DATA_DIR}/exposure_gain/posting_margin_ratio_hist_2025.png`

### 12. Train Sparse Autoencoders

Train layer-wise sparse autoencoders from the fine-tuned BERT representations. These SAE checkpoints are used by the SAE visualization and feature-analysis scripts.

```bash
bash commands/train_sae.sh
```

Runs:
- `sparse_auto_encoder/train_sae.py`

Inputs are configured inside `train_sae.py` and default to:
- `${BASE_DATA_DIR}/preprocessed_www_new/train`
- `${BASE_DATA_DIR}/preprocessed_www_new/skill2idx.json`
- `${BASE_DATA_DIR}/bert_pretrained`
- `${BASE_DATA_DIR}/checkpoints(www)_new/best_model.pt`

Outputs:
- `${BASE_DATA_DIR}/sae_layerwise_out_8192/layer_01/sae_best.pt`
- `...`
- `${BASE_DATA_DIR}/sae_layerwise_out_8192/layer_12/sae_best.pt`
- Each layer directory also contains `sae_last.pt`.

### 13. SAE Representation Visualization

Build industry- and year-grouped SAE UMAP visualizations and similarity curves.

```bash
bash commands/vis_sae_results.sh
```

Inputs include preprocessed test files, BERT checkpoint, skill vocabulary, best model checkpoint, and SAE layerwise outputs from `${BASE_DATA_DIR}/sae_layerwise_out_8192` by default.

Runs:
- `sparse_auto_encoder/industry/umap_cluster_sim_industry.py`
- `sparse_auto_encoder/yearly/umap_cluster_sim_yearly.py`
- `sparse_auto_encoder/vis_umap.py`
- `sparse_auto_encoder/vis_cluster_sim_graph.py`

Outputs:
- `${BASE_DATA_DIR}/sparse_auto_encoder/python_industry`
- `${BASE_DATA_DIR}/sparse_auto_encoder/python_yearly`
- UMAP CSV/PNGs, legends/colorbars, feature parquet files, and similarity plots.

### 14. SAE Feature Overlap Analysis

Measure and visualize SAE feature overlap across industries and years.

```bash
bash commands/overlap_graph.sh
```

Runs:
- `sparse_auto_encoder/industry/overlap_by_industry.py`
- `sparse_auto_encoder/industry/overlap_by_industry_graph.py`
- `sparse_auto_encoder/yearly/overlap_by_year.py`
- `sparse_auto_encoder/yearly/overlap_by_year_graph.py`
- `sparse_auto_encoder/yearly/overlap_by_diff_year_graph.py`

Outputs include overlap heatmaps, per-layer CSVs, overlap graphs, and year-difference plots under:
- `${BASE_DATA_DIR}/sparse_auto_encoder/python_industry/overlaps`
- `${BASE_DATA_DIR}/sparse_auto_encoder/python_yearly/overlaps`

### 15. SAE Feature-ID Heatmaps

Visualize unioned top-K SAE feature activations and industry-specific activation structure.

```bash
bash commands/vis_feature_id.sh
```

Runs:
- `sparse_auto_encoder/industry/vis_union_mean_activation_heatmap.py`
- `sparse_auto_encoder/industry/vis_union_mean_activation_heatmap_threshold.py`

Inputs:
- `${BASE_DATA_DIR}/sparse_auto_encoder/python_industry/features.parquet`

Outputs:
- `${BASE_DATA_DIR}/sparse_auto_encoder/python_industry/feature_id_vis`
- Raw and quantile-thresholded heatmap outputs.

---

## Suggested Workflow

A typical end-to-end run is:

```bash
bash commands/pretrain.sh
bash commands/preprocess.sh
bash commands/train_three_models.sh
bash commands/pred_all.sh
bash commands/run_scoring.sh
bash commands/gpt_sampling_pred.sh
bash commands/find_substitutes.sh
```

Then run optional analysis pipelines depending on the experiment:

```bash
bash commands/run_gpt_scoring.sh
bash commands/eval_gemma_qwen_full.sh full
bash commands/run_top2_pipeline.sh full
bash commands/get_margin.sh
bash commands/train_sae.sh
bash commands/vis_sae_results.sh
bash commands/overlap_graph.sh
bash commands/vis_feature_id.sh
```
---
