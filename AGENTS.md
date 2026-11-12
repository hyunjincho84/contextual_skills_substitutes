# Codex Project Context: Contextual Skill Substitution

This file defines persistent project instructions for follow-up coding, analysis, experimental design, paper revision, and result interpretation based on the paper **"Skill Substitutability is Contextual: Extracting Dynamic Skill Substitutes from Job Postings."**

Authors: Hyunjin Cho, Woo-Sung Jung, Jaehyuk Park, and Jisung Yoon

Source PDF: `contextual_alternative_skills(5).pdf`

## 0. Instruction Priority and General Principles

1. Follow the user's current request first.
2. When repository code, configuration, or actual data schemas are more specific than this document, follow the repository while reporting any material discrepancy.
3. If this document conflicts with the source paper, inspect the paper and tell the user about the difference.
4. Never fill in details that the paper does not report. Clearly distinguish among `reported in the paper`, `interpretation`, `proposal`, and `unverified`.
5. Separate replication from methodological extensions. Preserve the paper's original method as a baseline whenever testing a new approach.

---

## 1. Central Research Question

The paper asks:

> Given a specific job context, occupational group, and point in time, which skill can most appropriately replace the originally required skill while preserving the intended job function?

The central claims are:

- Skill **relatedness or similarity** is not the same as **functional substitutability**.
- Frequent co-occurrence or proximity in an embedding space does not by itself establish interchangeability.
- Substitutability is not a fixed global property of a skill pair. It varies across individual job-posting contexts.
- The same target skill can have different substitutes across occupations and years.
- The basic analytical unit is therefore a contextual relation rather than a global `skill -> skill` mapping:

\[
(\text{target skill},\ \text{posting context},\ \text{occupation},\ \text{time})
\rightarrow \text{contextual substitute}
\]

Treat this as a retrospective study. Because the model is trained on data spanning 2010-2025, do not describe historical outputs as point-in-time forecasts made using only information available in each historical year. Do not use claims such as `prospective prediction` or `real-time historical forecast` unless a separate future-information-blocked training design is implemented.

---

## 2. Core Terms and Mathematical Definition

| Term or symbol | Meaning |
| --- | --- |
| \(X\) | Job-posting context in which the original skill has been replaced by `[MASK]` |
| \(s^*\) | Original target or ground-truth skill in the posting |
| \(V\) | Fixed candidate-skill vocabulary; \(|V|=2{,}012\) in the paper |
| \(P(s\mid X)\) | Model probability assigned to candidate skill \(s\) given context \(X\) |
| Contextual substitute | Highest-probability candidate after explicitly excluding \(s^*\) |
| Masked-skill instance | One training or inference observation created from one target-skill occurrence |
| Posting-level aggregation | Procedure that prevents repeated instances from one posting from receiving excessive weight |

Always implement substitute selection as:

\[
\hat{s}_{sub}=\arg\max_{s\in V\setminus\{s^*\}}P(s\mid X)
\]

### Critical implementation rule

- Do not implement this rule as an unconditional selection of the second array element in a sorted probability list.
- The paper informally calls the result the “second-highest candidate” because the original skill is usually Top-1.
- The exact procedure is to remove the original skill explicitly and then take the argmax.
- Apply the same exclusion rule even when the original skill is not Top-1.
- A contextual substitute is not necessarily a synonym or a universally equivalent skill. It is a functional alternative under the specific posting context.

---

## 3. Data Scope and Preprocessing

### 3.1 Dataset

- Source: Lightcast longitudinal job postings
- Geography and period: United States, 2010-2025
- Overall scale reported in the paper: more than 3.7 billion postings
- Structured variables: 139
- Main variable categories:
  - metadata
  - job classification
  - skills and certifications
- Main analytical focus: IT and software-development postings
- Occupational classification: 23 SOC Major Groups
- Candidate-skill vocabulary: 2,012 skills

Examples of reported fields:

| Category | Example columns |
| --- | --- |
| Metadata | `id`, `posted`, `expired`, `url`, `title_raw`, `body`, `company_name`, `education_levels`, `salary`, `location` |
| Job classification | `nasics6`, `onet_2019`, `cip6`, `soc_5`, `lot_career_area`, `lot_occupation` |
| Skills and certifications | `skills_name`, `specialized_skills_name`, `common_skills_name`, `software_skills_name`, `certifications_name` |

Inspect the actual schema before using these names. For example, verify whether the paper's `nasics6` field is actually `naics6` in the data.

### 3.2 Preprocessing pipeline

Follow the sequence described in the paper:

1. Retain only the raw-posting text after `Job Description:`.
2. Match description text against the standardized Lightcast skill taxonomy without case sensitivity.
3. Replace the matched target-skill span with `[MASK]`.
4. Construct a balanced context window around the target under BERT's 512-token limit.
5. When one side has insufficient context, use additional context from the other side to fill the available sequence length.
6. Create a separate instance for each masked skill occurrence.
7. Split the instances into training and held-out sets. Use the training set for fine-tuning and held-out data for validation and downstream analysis.

### 3.3 Preprocessing invariants

Preserve the following fields through the full pipeline:

- original text
- original skill span
- normalized skill ID
- character and token offsets
- posting ID
- posting date and year
- SOC code and SOC Major Group

Additional requirements:

- Treat a multi-token skill as one target skill.
- Define policies for overlapping skill phrases, substring matches, aliases, and ambiguous names.
- Retain posting IDs because one posting can generate multiple masked instances.
- Track possible split leakage if instances from the same posting can enter different splits.
- Derive temporal labels from the posting's `posted` date, not the model execution date.
- SOC Major Group is an **occupational domain**, not an industry classification. Do not conflate SOC with NAICS. When the paper uses `industry` in a figure caption, verify the actual label source before reproducing that terminology.

### 3.4 Sample-size discrepancies that must remain visible

The PDF reports different counts at different stages:

- Preprocessing section: approximately 2.9M valid instances in each of the training and test sets
- Masked-skill accuracy: a held-out test set of 1.4M samples
- Employer-stated pair extraction: a 2.9M-instance held-out test set
- Selection-strategy appendix: the text mentions a common valid subset of 0.85M, while Table A.1 describes a sampled set of 37K

Do not silently harmonize these numbers. Recompute and report separately:

- instance counts immediately after splitting
- posting counts and masked-instance counts
- sample size used for accuracy
- sample size before and after sampling
- common valid sample size across evaluators or strategies
- counts of failed or discarded observations

---

## 4. Model and Training Specification

### 4.1 Architecture

- BERT encoder
- MLP applied to the masked-token representation
- Softmax classifier over the fixed 2,012-skill vocabulary
- Objective: recover the original masked skill from the surrounding job-description context

### 4.2 Two-stage training

1. Domain pre-training
   - Adapt the encoder to job-posting language.
   - Use 15% random token masking.
2. Masked-skill fine-tuning
   - Train on instances in which a recognized skill span is masked.
   - Minimize cross-entropy loss for the original skill.

\[
\mathcal{L}_{CE}=-\log \hat{y}_{s^*}
\]

Fine-tuning settings reported in the paper:

| Setting | Value |
| --- | ---: |
| Epochs | 3 |
| Optimizer | AdamW |
| Maximum sequence length | 512 |
| Batch size | 64 |
| Learning rate | \(2\times10^{-5}\) |
| Hardware | One NVIDIA A100 GPU |
| Reported runtime | Approximately one day |

Do not guess the base BERT checkpoint, cased/uncased choice, tokenizer, warmup, scheduler, weight decay, gradient accumulation, or random seed. Use repository values when they exist; otherwise mark them as unverified.

### 4.3 Minimum inference output

Store at least the following for every instance:

```text
posting_id
target_skill_id
target_skill_text
mask_start / mask_end or token_offset
posted_year
soc_major_group
predicted_skill_ids (ordered)
predicted_probabilities (ordered)
selected_substitute_id
target_rank
model_checkpoint
```

Retain Top-k candidates and probabilities rather than saving only the selected substitute. This is necessary for later selection-rule robustness checks.

---

## 5. Aggregation Rules

Do not construct temporal trajectories or SOC-level tables by simply counting instance-level predictions.

If the same target skill occurs \(n\) times in one posting, assign each occurrence a weight of \(1/n\). The total contribution of that posting for the target skill must equal one.

\[
w_{instance}=\frac{1}{\#\{\text{target-skill occurrences in the posting}\}}
\]

Then sum the weights within the relevant aggregation group:

- Temporal analysis: `(target skill, year, substitute)`
- Occupational analysis: `(target skill, SOC Major Group, substitute)`

Rank substitutes within each group by their summed weights.

The GPT-5.1 baseline does not provide candidate probabilities. The paper assigns uniform weights to its Top-5 outputs and then applies the same aggregation procedure. Do not interpret this as confidence weighting equivalent to the probability-based BERT aggregation.

---

## 6. Model Validation

### 6.1 Masked-skill recovery

This evaluation measures whether the model can recover the original skill. It does not directly measure substitute correctness.

| Metric | Result |
| --- | ---: |
| Top-1 accuracy | 0.808 |
| Top-3 accuracy | 0.917 |
| Top-5 accuracy | 0.944 |

### 6.2 Semantic Validation

The LLaMA-based validator compares the original sentence \(X\) with the substituted sentence \(X'\) to assess preservation of meaning and hiring intent.

The paper defines:

\[
s(X',X,k)=\sum_{i=1}^{L}w_{i,k}\cdot
f\left(h(x_i\mid X),h(x'_i\mid X')\right)
\]

where:

- \(k\) is the anchor position, defined as the first token of the skill span.
- \(h\) concatenates the final four hidden layers of LLaMA.
- \(f\) is cosine similarity.
- \(w_{i,k}\) is self-attention from token \(i\) to anchor \(k\), averaged across all layers and heads.
- A higher score indicates better preservation of the original contextual meaning.

The PDF does not fully specify alignment when the original and replacement skills tokenize to different lengths. Preserve a working existing implementation unless an explicit change is requested.

If the repository's established Semantic Validation pipeline uses `_anchor_by_offset` with local windowing, treat that implementation as the working baseline. Do not replace it with a full-sequence anchor implementation without an explicit request. When testing an alternative, report the offset-based baseline alongside it.

### 6.3 Semantic-validation results

Do not merge the following two comparison blocks as though they were calculated on one identical dataset.

| Comparison block | Model | Mean SV |
| --- | --- | ---: |
| Baseline comparison | Ours | 0.8118 |
| Baseline comparison | word2vec | 0.6491 |
| Baseline comparison | Conditional probability | 0.7996 |
| LLM comparison | Ours | 0.8143 |
| LLM comparison | GPT-5.1 | 0.7657 |

Evaluator robustness results:

| Comparison block | Model | Qwen2.5-3B-Instruct | Gemma-2-2B-it |
| --- | --- | ---: | ---: |
| Baseline | Ours | 0.8827 | 0.8663 |
| Baseline | word2vec | 0.7854 | 0.7490 |
| Baseline | Conditional probability | 0.8789 | 0.8602 |
| LLM | Ours | 0.8320 | 0.8982 |
| LLM | GPT-5.1 | 0.7886 | 0.8710 |

Treat LLM-evaluator scores as comparative indicators, not absolute ground truth.

### 6.4 Employer-stated `A or B` validation

The paper uses patterns such as `skill A or skill B` in real job postings as a data-driven proxy for employer-stated substitutability.

Procedure:

1. Extract 17,776 candidate `A or B` cases from held-out postings.
2. Manually filter them to 1,000 high-quality, genuinely substitutable pairs.
3. Replace one side with `[MASK]` while retaining the other skill and surrounding context.
4. Treat the masked side as the ground-truth substitute and compute Top-1/3/5 recovery.

| Model | Top-1 | Top-3 | Top-5 |
| --- | ---: | ---: | ---: |
| Ours | 0.831 | 0.957 | 0.982 |
| GPT-5.1 | 0.533 | 0.675 | 0.730 |
| Conditional probability | 0.356 | 0.557 | 0.582 |
| word2vec | 0.011 | 0.013 | 0.014 |

Use precise descriptions such as `employer-stated alternatives` or `an expert-grounded proxy extracted from HR-authored postings`. Do not describe this as a direct panel evaluation in which HR experts independently rated model outputs. The PDF does not fully report the manual-filtering protocol, number of annotators, or inter-annotator agreement.

### 6.5 Substitute-selection robustness

| Selection strategy | Mean SV |
| --- | ---: |
| Highest-probability non-original | 0.8242 |
| Top-k, \(k=10\) | 0.8190 |
| Threshold, \(\tau=0.005\) | 0.8216 |
| Threshold, \(\tau=0.001\) | 0.8200 |
| Temperature, \(T=1.5\), Top-10 | 0.8143 |
| Temperature, \(T=2.0\), Top-10 | 0.8095 |

Use the highest-probability non-original rule for the main analysis. Describe the broader result as relatively robust to the selection strategy, not identical across strategies.

---

## 7. Baseline Definitions

### 7.1 Conditional probability

\[
P(s_2\mid s_1)=\frac{\operatorname{count}(s_1,s_2)}{\operatorname{count}(s_1)}
\]

- `count(s1, s2)` is the number of postings containing both skills.
- `count(s1)` is the number of postings containing the target skill.
- Select the candidate with the highest conditional probability.
- This is a global co-occurrence measure and does not model posting-specific contextual variation.

### 7.2 word2vec

- Use Skip-gram with Negative Sampling.
- Treat skill requirements in one posting as co-occurring terms.
- Shuffle their order to reduce positional bias.
- Select the skill with the highest cosine similarity to the target.
- This produces a global embedding and does not directly represent context-specific interchangeability.

### 7.3 GPT-5.1

The core prompt constraints are:

```text
You are given a job posting sentence containing a [MASK] token.
Infer the FIVE most appropriate skills to replace [MASK].

HARD CONSTRAINT:
- Choose ONLY from the allowed skill list.
- Do NOT invent new skills.
- Use EXACT spelling from the list.
- Output ONLY five skills separated by "|".
- No extra text.
```

Do not invent the exact model snapshot, decoding parameters, batching setup, or malformed-output retry policy. Mark them as unverified unless the code or experiment logs specify them.

### 7.4 Rules for fair baseline comparisons

- Match candidate vocabularies and evaluation samples whenever possible.
- State the input-information difference between context-aware and context-free models.
- Record the output-structure difference between GPT Top-5 lists and BERT probability distributions.
- Do not present results from different evaluation subsets as if they came from identical conditions.
- Preserve the original three baselines when adding new lexical-substitution baselines.

---

## 8. Main Empirical Findings

### 8.1 Occupational variation

The same target skill can map to different substitutes across SOC Major Groups.

Reported examples include:

- `C -> Embedded C` in some computing and engineering contexts
- `C -> R` in healthcare-research contexts
- `Python -> SQL`, `Java`, `Perl`, `C++`, `R`, or `DevOps`, depending on SOC context
- `JavaScript -> jQuery`, `Java`, `HTML5`, `Scripting`, `JScript`, or `Apache Camel`, depending on SOC context

Treat these as results of the paper's specific aggregation procedure, not universal equivalences.

### 8.2 Temporal variation

Representative trajectories from 2010-2025 include:

- Python: movement from Perl toward SQL
- JavaScript: movement from jQuery toward TypeScript
- C: increasing alignment with Embedded C and C++

Do not interpret a short-lived change in instance-level Top-1 predictions as technological change by itself. Check annual sample composition, weighted aggregation, uncertainty, and taxonomy drift.

---

## 9. Sparse Autoencoder Analysis

### 9.1 Purpose

The SAE analysis asks whether the model represents the same surface skill differently across occupations and time. It examines alignment between internal representations and context labels; it does not establish that SAE features causally determine substitute rankings.

### 9.2 Procedure

1. Freeze the fine-tuned BERT encoder.
2. Extract the contextual hidden representation at the `[MASK]` position from each BERT layer.
3. Train a separate encoder-decoder SAE for each layer.
4. Combine reconstruction loss with sparsity and stability penalties.
5. Compare SAE features against occupation and year labels.

Reported SAE settings:

| Setting | Value |
| --- | ---: |
| Latent dimension \(m\) | 4,096 |
| Epochs | 3 |
| Optimizer | AdamW |
| Learning rate | \(1\times10^{-4}\) |
| \(\lambda_{\ell_1}\) | \(10^{-3}\) |
| \(\lambda_{\ell_2}\) | \(10^{-4}\) |

### 9.3 Analyses and interpretation

- Project SAE activations with UMAP to visualize SOC- and year-related organization.
- Apply k-means to layer-wise SAE activations.
- Measure alignment with SOC or year labels using element-centric cluster similarity.
- Compare the observed clustering with a random-null baseline.
- Compute overlap among highly activated Top-K SAE features across occupations or years.
- Interpret increasing cluster alignment and decreasing feature overlap in deeper layers as evidence of contextual specialization.

### 9.4 Limitations

- The SAE evidence is correlational.
- Do not claim that a particular feature causes a substitution decision.
- Causal claims require additional interventions such as feature ablation, activation patching, or steering.
- Do not guess the exact SAE loss, activation function, normalization, Top-K value, threshold \(q\), cluster count, UMAP parameters, or random seeds when they are absent from the paper and code.

---

## 10. Market-Exposure Simulation

### 10.1 Main metric

Let \(x\) be the Google Trends score for the original skill and \(y\) the score for its substitute:

\[
\text{relative exposure gain}=\frac{y-x}{x}
\]

- Reported posting period: January-June 2025
- A positive value means the substitute has higher search interest than the original skill.
- Approximately 64.09% of postings contain at least one positive-margin substitution.
- Some cases exhibit relative increases of up to three orders of magnitude.

### 10.2 Robustness metrics

| Metric | Definition or result |
| --- | --- |
| Median original trend score | \(x=21.5\) |
| Absolute gain | \(y-x\); mean 18.59 |
| Clipped relative gain | \((y-x)/\max(x,c)\), with \(c=10\); mean 259.70% |
| Log-ratio gain | \(\log((y+\epsilon)/(x+\epsilon))\); mean 0.95 |

The PDF does not report the value of \(\epsilon\). Do not supply one without checking the code.

### 10.3 Required interpretation

- Google Trends measures skill-level public search interest and is only a proxy.
- Do not describe the result as an observed increase in posting impressions, clicks, applications, interviews, or hires.
- Use terms such as `market-exposure proxy` or `search-interest-based visibility proxy`.
- Google Trends scores are normalized within a query, time window, and geography. If different skills were retrieved through separate requests, verify their comparability.
- Direct validation requires job-board impression, click, application, or hiring outcomes, ideally through a field experiment.

---

## 11. Limitations and Reproducibility Gaps

Limitations explicitly acknowledged in the paper:

1. Semantic validation depends on LLM evaluators.
2. SAE analysis is correlational and does not identify a causal mechanism.
3. Google Trends is not a direct measure of job-posting exposure or hiring outcomes.
4. Vague or underspecified postings can produce poorly matched substitutes.

Additional items Codex must verify during replication:

- Base BERT checkpoint and tokenizer
- Construction and filtering of the 2,012-skill vocabulary
- Skill-span matching, overlap resolution, and alias handling
- Whether splitting occurs by posting or by instance
- Possible leakage of instances from one posting across splits
- Random seeds, scheduler, warmup, and weight decay
- Domain pre-training hyperparameters and corpus sampling
- word2vec hyperparameters
- Exact LLaMA evaluator checkpoint and token-alignment procedure
- GPT-5.1 snapshot, decoding configuration, and malformed-output handling
- Manual-filtering protocol and annotator agreement for `A or B` pairs
- SAE architecture, loss, Top-K, threshold, clustering, and UMAP parameters
- Google Trends query construction, geography, normalization, and \(\epsilon\)
- Confidence intervals, bootstrap estimates, or statistical significance
- Harmonization of changing Lightcast taxonomy versions across years

---

## 12. Codex Working Rules

### 12.1 Paper explanation and writing

- Distinguish `similar`, `related`, `co-occurring`, and `substitutable`.
- When using `dynamic`, identify the occupational and temporal dimensions.
- If labels come from SOC, use `occupation` or `SOC Major Group`, not `industry`.
- When discussing human validation, explain that the benchmark is extracted from employer-authored `A or B` statements rather than a direct HR-rating panel.
- Do not use causal verbs such as `causes` or `drives` for SAE results without an intervention experiment.
- Always describe the Google Trends analysis as a simulation or proxy.

### 12.2 Code changes

- Inspect the existing file structure, schema, configuration, functions, and output format before editing.
- Preserve unrelated user changes.
- Expose paths and hyperparameters through configuration files or CLI arguments rather than hard-coding them.
- Record random seeds and software/model versions.
- Retain posting IDs in all intermediate artifacts.
- Apply inverse-occurrence weighting for repeated target skills within a posting.
- Explicitly exclude the target skill before selecting a substitute.
- Test on a small deterministic sample before large-scale execution.
- Log validation sample sizes, failed instances, and dropped instances.
- Store numerator, denominator, and sample-construction details for reported metrics.

### 12.3 Reporting results

Separate each result into four layers whenever possible:

1. **Observation:** value directly observed in data or model output
2. **Interpretation:** what the observation may imply
3. **Alternative explanation:** co-occurrence, sample composition, taxonomy drift, evaluator bias, or another mechanism
4. **Validation needed:** additional data or experiment required

For every model-comparison table, report:

- evaluation set
- sample size
- candidate vocabulary
- whether contextual text was available
- selection rule
- evaluator
- uncertainty or repeated-run variation

### 12.4 Proposing new experiments

Separate applications by stakeholder:

- **Worker:** reachable alternative jobs, reskilling paths, wages, and employment outcomes
- **HR or firm:** requirement relaxation, applicant-pool expansion, application conversion, and time-to-fill
- **Policymaker:** regional or occupational skill mismatch, training allocation, and transition or displacement risk

Prefer direct outcomes over Google Trends when possible:

- posting impressions
- search-result rank
- click-through rate
- application starts and completions
- interview callbacks
- vacancy duration or time-to-fill
- offered or realized wages
- worker transitions and employment duration

Do not infer causality from observational associations alone. Consider A/B tests, staggered rollout, difference-in-differences, matching or weighting, and other appropriate quasi-experimental designs.

---

## 13. Recommended Repository Structure

Use this structure only when the repository does not already have an established organization:

```text
configs/
  data.yaml
  model.yaml
  evaluation.yaml
src/
  data/
    schema.py
    extract_descriptions.py
    match_skills.py
    build_masked_instances.py
    split.py
  modeling/
    pretrain.py
    finetune.py
    predict.py
    select_substitute.py
  aggregation/
    posting_weights.py
    temporal_rankings.py
    soc_rankings.py
  evaluation/
    masked_skill_accuracy.py
    semantic_validation.py
    explicit_or_pairs.py
    baselines.py
  interpretability/
    extract_hidden_states.py
    train_sae.py
    cluster_alignment.py
    feature_overlap.py
  applications/
    exposure_proxy.py
tests/
scripts/
outputs/
```

Record a configuration hash, code version, model checkpoint, and data version or date in intermediate outputs so every table and figure can be reproduced.

---

## 14. Completion Checklist

- [ ] Did the analysis avoid treating a substitute as merely a related skill or synonym?
- [ ] Was the target skill explicitly excluded before argmax selection?
- [ ] Were instance-level and posting-level results kept distinct?
- [ ] Was inverse-occurrence weighting applied within each posting?
- [ ] Was SOC kept distinct from industry classification?
- [ ] Did temporal labels come from posting dates?
- [ ] Were results from different evaluation subsets kept separate?
- [ ] Were sample sizes and dropped-instance counts recorded?
- [ ] Was the employer-stated proxy described accurately rather than as a direct HR panel?
- [ ] Were LLM-evaluator scores treated as comparative indicators rather than absolute ground truth?
- [ ] Were causal claims avoided in the SAE interpretation?
- [ ] Was Google Trends kept distinct from actual application and hiring outcomes?
- [ ] Were unreported hyperparameters and preprocessing rules left unverified rather than invented?
- [ ] Were replication results separated from extension results?

---

## 15. One-Sentence Summary

The project's guiding question is not **“Which skills are globally similar?”** but **“Which skill can most plausibly replace the original skill in this posting's specific occupational, temporal, and textual context?”**
