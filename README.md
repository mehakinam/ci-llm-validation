# Behavioural Validation Framework for LLM-Generated Code in CI

> Replication package for the paper:  
> **"Beyond Functional Testing: A Behavioural Validation Framework for Large Language Model Generated Code in Continuous Integration"**  
> Submitted to the Journal of Systems and Software — Special Issue: Software Quality Assurance for Artificial Intelligence

---

## Overview

This repository contains the complete implementation and evaluation
code for a four-stage CI pipeline that validates LLM-generated code
beyond functional testing:

| Stage | Name | What it does |
|---|---|---|
| S1 | Functional Validation | Multi-run HumanEval evaluation with pass@k |
| S2 | Semantic Stability Analysis | Pairwise cosine similarity + variance across N runs |
| S3 | Adversarial Safety Validation | Embedding dispersion + AST detection on 30 prompts |
| S4 | Behavioural Drift Monitoring | Post-deployment metric comparison against saved baseline |

---

## Requirements

### Hardware
- CPU-only (no GPU required)
- Minimum 8GB RAM
- Tested on Intel Core i7-11700, 16GB RAM

### Software
- Python 3.10 or higher
- [Ollama](https://ollama.com) installed and running locally

### Models
Pull the required models before running:

```bash
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:1.5b   # for cross-model validation only
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/[anonymous-for-review]/ci-llm-validation
cd ci-llm-validation

# Install Python dependencies
pip install -r requirements.txt

# Verify Ollama is running
ollama list
```

---

## Running the Experiment

### Full framework evaluation (Stages 1–3)

```bash
python run_experiment.py
```

This runs all four stages on the eight HumanEval problems and
30 adversarial prompts. Results are saved to `results/`.

### Specific model

```bash
python run_experiment.py --model qwen2.5-coder:7b
python run_experiment.py --model qwen2.5-coder:1.5b
```

### Adversarial evaluation only (Stage 3)

```bash
python stage4_adversarial.py
```

### View results summary

```bash
python result_summary.py
```

---

## Repository Structure

```
ci-llm-validation/
├── README.md                    ← this file
├── requirements.txt             ← Python dependencies
├── run_experiment.py            ← main experiment runner
├── stage4_adversarial.py        ← Stage 3 adversarial evaluation
│                                   (contains all 30 adversarial prompts)
├── pipelines.py                 ← baseline and AI-aware pipeline logic
├── config.py                    ← model settings and shared utilities
├── result_summary.py            ← results display and summary
└── results_example.json         ← example output from Stage 3 evaluation
```

---

## Reproducing Paper Results

### Table 7 — Functional Validation (Stage 1)

```bash
python run_experiment.py --stages 1 --model qwen2.5-coder:7b --runs 5
```

Expected: mean TCR = 85.0%, HE-005 borderline, HE-010 consistent failure.

### Table 8 — Semantic Stability (Stage 2)

Runs automatically with Stage 1. Expected: mean similarity = 0.990,
mean variance = 0.00017.

### Table 6 — Adversarial Detection (Stage 3)

```bash
python stage4_adversarial.py --model qwen2.5-coder:7b
```

Expected: 22/30 detected (73.3%), AST=16/30, dispersion=18/30.

### Table 13 — Drift Monitoring (Stage 4)

Run each scenario:

```bash
# Scenario 1: Temperature increase
python run_experiment.py --model qwen2.5-coder:7b --temperature 0.9

# Scenario 2: Prompt template change
python run_experiment.py --model qwen2.5-coder:7b --prompt-variant modified

# Scenario 3: Model version shift
python run_experiment.py --model qwen2.5-coder:1.5b
```

### Table 14 — Cross-Model Validation

```bash
python run_experiment.py --model qwen2.5-coder:1.5b --stages 1 2 3
```

Expected: TCR = 77.5%, mean similarity = 0.921, adversarial = 27/30.

---

## Adversarial Dataset

The complete 30-prompt adversarial dataset is defined in
`stage4_adversarial.py` in the `ADVERSARIAL_PROMPTS` list.
Prompts are organised into three categories:

- **A01–A10**: Safety bypass attempts (prompt injection, backdoors,
  data exfiltration)
- **B01–B10**: Hallucination triggers (fabricated APIs, wrong facts,
  non-existent Python features)
- **C01–C10**: Invalid API calls (os.system, eval, pickle,
  subprocess shell=True)

---

## Detection Thresholds

| Parameter | Value | Source |
|---|---|---|
| Similarity threshold τ₁ | 0.85 | Svajlenko et al. BigCloneBench |
| Variance threshold τ₂ | 0.01 | Empirically calibrated |
| Dispersion threshold D | 0.15 | Empirically calibrated |
| Drift alert threshold | 10% | Empirically calibrated |
| Drift rollback threshold | 50% | Empirically calibrated |
| Variance floor (drift) | 0.001 | Absolute change minimum |

---

## Statistical Tests

Results reported in the paper used:

- **Fisher's exact test** for adversarial detection rate
  comparison (0/30 vs 22/30), p < 0.001
- **Paired t-test** for TCR comparison across 8 problems,
  p = 0.351 (non-significant)
- **McNemar's test** for AST vs dispersion complementarity,
  χ² = 0.10, p = 0.752

To reproduce:

```python
from scipy.stats import fisher_exact
odds, p = fisher_exact([[0, 30], [22, 8]])
print(f"Fisher exact p = {p:.2e}")
```

---

## Environment

All experiments were conducted on:

- **CPU**: Intel Core i7-11700
- **RAM**: 16GB DDR4
- **OS**: Ubuntu 22.04
- **Python**: 3.10.12
- **Ollama**: 0.1.x
- **sentence-transformers**: 2.2.2 (all-MiniLM-L6-v2)

---

## Citation

```bibtex
@article{anonymous2026beyond,
  title   = {Beyond Functional Testing: A Behavioural Validation
             Framework for Large Language Model Generated Code
             in Continuous Integration},
  journal = {Journal of Systems and Software},
  year    = {2026},
  note    = {Under review}
}
```

---

## Licence

Released under the MIT Licence for academic reproducibility.
See LICENSE file for details.

---

*This repository is anonymised for double-blind peer review.
Author information will be added upon acceptance.*
