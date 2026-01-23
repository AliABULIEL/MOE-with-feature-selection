# MoE Router Analysis Pipelines

End-to-end pipelines for analyzing Mixture-of-Experts (MoE) router behavior.

## Overview

These pipelines **use the existing `RouterLogger` classes** from your PR files:
- `run_deepseek_pipeline.py` → imports from `moe_internal_logging_deepseek.py`
- `run_qwen_pipeline.py` → imports from `moe_internal_logging_qwen.py`

## What Each Stage Does

| Stage | What It Does | Equivalent in logs_eda.ipynb |
|-------|--------------|------------------------------|
| **Stage 1: Evaluation** | Load model, run inference, capture router logits using `RouterLogger` hooks, save JSON | JSON files are the **input** to logs_eda.ipynb |
| **Stage 2: KDE Training** | Load JSON, extract `router_logits_sample`, train `gaussian_kde` per layer, save `.pkl` | `gaussian_kde(train_data.T)` + `pickle.dump` cells |
| **Stage 3: Basic Plots** | Expert weight sums, choice counts, softmax/raw logit distributions | First half of notebook |
| **Stage 4: Per-Expert Plots** | Distribution per (layer × expert) combination | `softmax_router_logits_per_expert` loops |
| **Stage 5: KDE Plots** | Cumulative density, trimmed comparisons, kernel comparisons | KDE and trimming cells |
| **Stage 6: P-Value Plots** | P-value distributions, vs uniform, by k-th best expert, by popularity | P-value cells at the end |

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: Evaluation with Internal Routing Logging               │
│ Uses: RouterLogger from moe_internal_logging_{model}.py         │
├─────────────────────────────────────────────────────────────────┤
│ Input:  Model + Datasets (LAMBADA, HellaSwag, WikiText)         │
│ Output: JSON files with router_logits_sample for ALL experts    │
│         {model}_{dataset}_internal_routing.json                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: KDE Model Training                                      │
├─────────────────────────────────────────────────────────────────┤
│ Input:  JSON from LAMBADA (training dataset)                    │
│ Output: KDE models per layer                                    │
│         {model}_distribution_model_layer_{i}.pkl                │
│         {model}_distribution_model_layer_{i}_trimmed_{k}.pkl    │
│         {model}_distribution_model_layer_{i}_{kernel}.pkl       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGES 3-6: Visualization                                        │
├─────────────────────────────────────────────────────────────────┤
│ Input:  JSON logs + KDE models                                  │
│ Output: All the plots from logs_eda.ipynb                       │
└─────────────────────────────────────────────────────────────────┘
```

## JSON Format (Compatible with logs_eda.ipynb)

```json
{
  "config": "deepseek-ai/DeepSeek-V2-Lite",
  "strategy": "topk_baseline",
  "num_experts": 64,
  "num_layers": 27,
  "samples": [
    {
      "sample_id": 0,
      "num_tokens": 45,
      "loss": 2.341,
      "layers": [
        {
          "layer": 0,
          "router_logits_shape": [45, 64],
          "router_logits_sample": [[...], ...],  // ← ALL 64 experts (critical for KDE)
          "selected_experts": [[12, 5, 33, 8, 41, 2], ...],
          "expert_weights": [[0.18, 0.15, ...], ...]
        }
      ]
    }
  ]
}
```

The key field is `router_logits_sample` which contains the **raw logits for ALL experts** (not just top-k). This is what the KDE models are trained on.

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipelines
python run_deepseek_pipeline.py
python run_qwen_pipeline.py
```

## Configuration

Edit the `CONFIG` dict at the top of each script:

```python
CONFIG = {
    # Toggle stages
    "run_evaluation": True,       # Stage 1
    "run_kde_training": True,     # Stage 2
    "run_basic_plots": True,      # Stage 3
    "run_per_expert_plots": True, # Stage 4 (many plots!)
    "run_kde_plots": True,        # Stage 5
    "run_pvalue_plots": True,     # Stage 6
    
    # Sample counts
    "max_samples": {
        "lambada": 500,
        "hellaswag": 500,
        "wikitext": 300,
    },
}
```

## Quick Test

```python
def main():
    config = CONFIG.copy()
    config["max_samples"] = {"lambada": 50, "hellaswag": 50, "wikitext": 30}
    config["run_per_expert_plots"] = False  # Skip 3000+ plots
    run_pipeline(config)
```

## Model Specifications

| Model | Script | RouterLogger Source | Experts | Hook Location |
|-------|--------|---------------------|---------|---------------|
| DeepSeek-V2-Lite | `run_deepseek_pipeline.py` | `moe_internal_logging_deepseek.py` | 64 | `layer.mlp.gate` |
| Qwen3-30B-A3B | `run_qwen_pipeline.py` | `moe_internal_logging_qwen.py` | 128 | `layer.mlp` → `module.gate()` |



# 1. Clone your repo
git clone <your-repo-url>
cd MOE-with-feature-selection

# 2. Install dependencies
pip install -r pipelines/requirements.txt

# 3. Run DeepSeek pipeline
python pipelines/run_deepseek_pipeline.py

# 4. Run Qwen pipeline  
python pipelines/run_qwen_pipeline.py
