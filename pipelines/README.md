# MoE Router Analysis Pipelines

End-to-end pipelines for analyzing Mixture-of-Experts (MoE) router behavior in DeepSeek-V2-Lite and Qwen3-30B-A3B models.

## Overview

These pipelines capture internal router logits during inference, train KDE models to estimate probability distributions, and generate comprehensive visualizations for analyzing expert selection patterns.

## Scripts

| Script | Model | Experts | Top-K |
|--------|-------|---------|-------|
| `run_deepseek_pipeline.py` | DeepSeek-V2-Lite | 64 | 6 |
| `run_qwen_pipeline.py` | Qwen3-30B-A3B | 128 | 8 |

## Installation

```bash
pip install -r requirements.txt
```

**Note:** Requires `transformers>=4.57.0` for Qwen3 support.

## Usage

### Basic Usage

```bash
# Run DeepSeek pipeline
python run_deepseek_pipeline.py

# Run Qwen pipeline
python run_qwen_pipeline.py
```

### Configuration

Each script has a `CONFIG` dictionary at the top that can be modified:

```python
CONFIG = {
    # Model Configuration
    "model_id": "deepseek-ai/DeepSeek-V2-Lite",
    "model_name": "deepseek",
    "num_experts": 64,
    "default_top_k": 6,
    
    # Dataset Configuration
    "datasets": ["lambada", "hellaswag", "wikitext"],
    "max_samples": {
        "lambada": 500,
        "hellaswag": 500,
        "wikitext": 300,
    },
    
    # Pipeline Stage Toggles
    "run_evaluation": True,      # Stage 1: Run model on datasets
    "run_kde_training": True,    # Stage 2: Train KDE models
    "run_basic_plots": True,     # Stage 3: Summary plots
    "run_per_expert_plots": True,# Stage 4: Per-expert plots (many!)
    "run_kde_plots": True,       # Stage 5: KDE analysis plots
    "run_pvalue_plots": True,    # Stage 6: P-value distribution plots
    
    # Hardware Configuration
    "device": "cuda",
    "dtype": "bfloat16",
}
```

### Quick Testing

For quick testing with fewer samples:

```python
def main():
    config = CONFIG.copy()
    
    # Override for quick testing
    config["max_samples"] = {"lambada": 50, "hellaswag": 50, "wikitext": 30}
    config["run_per_expert_plots"] = False  # Skip the many per-expert plots
    
    run_pipeline(config)
```

## Pipeline Stages

### Stage 1: Model Evaluation with Internal Logging
- Loads the model and tokenizer
- Registers forward hooks on router gates
- Runs inference on LAMBADA, HellaSwag, and WikiText datasets
- Captures full router logits (all experts) for each token
- Saves JSON logs with routing data

### Stage 2: KDE Model Training
- Trains Gaussian KDE models per layer using LAMBADA data
- Creates CDF lookup tables for p-value computation
- Trains trimmed variants (removing top/bottom 1, 2, 4 experts)
- Trains with multiple kernels (gaussian, tophat, epanechnikov, linear)

### Stage 3: Basic Plots
- Expert weight sum distributions per layer
- Expert choice count bar plots
- Router softmax output distributions
- Raw router logit distributions

### Stage 4: Per-Expert Plots (Optional)
- Per-layer, per-expert softmax distributions
- Per-layer, per-expert raw logit distributions
- **Warning:** Generates many plots (DeepSeek: ~3,500, Qwen: ~12,500)

### Stage 5: KDE Analysis Plots
- KDE cumulative density histograms
- Trimmed logit comparisons (train vs test datasets)
- Kernel comparison plots

### Stage 6: P-Value Distribution Plots
- Basic p-value distributions per layer
- P-value vs uniform distribution comparisons
- P-value by k-th best expert
- P-value by expert popularity (most chosen to least chosen)

## Output Structure

```
outputs/
├── deepseek/
│   ├── logs/
│   │   ├── deepseek_lambada_internal_routing.json
│   │   ├── deepseek_hellaswag_internal_routing.json
│   │   └── deepseek_wikitext_internal_routing.json
│   ├── kde_models/
│   │   ├── deepseek_distribution_model_layer_0.pkl
│   │   ├── deepseek_distribution_model_layer_0_trimmed_2.pkl
│   │   ├── deepseek_distribution_model_layer_0_gaussian.pkl
│   │   └── ...
│   └── plots/
│       ├── basic/
│       │   ├── expert_weights_sum/
│       │   ├── expert_choice_counts/
│       │   ├── router_softmax/
│       │   └── router_raw_logits/
│       ├── per_expert/
│       │   ├── per_expert_softmax/
│       │   └── per_expert_raw_logits/
│       ├── kde/
│       │   ├── kde_cumulative_density/
│       │   ├── trimmed_comparison/
│       │   └── kernel_comparison/
│       └── pvalue/
│           ├── pvalue_basic/
│           ├── pvalue_vs_uniform/
│           └── pvalue_by_popularity/
│
└── qwen/
    └── ... (same structure)
```

## JSON Log Format

```json
{
  "config": "deepseek-ai/DeepSeek-V2-Lite",
  "strategy": "topk_baseline",
  "num_experts": 64,
  "top_k": 6,
  "dataset": "lambada",
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
          "router_logits_sample": [[...], ...],  // All 64 experts
          "selected_experts": [[12, 5, 33, 8, 41, 2], ...],  // Top-6
          "expert_weights": [[0.18, 0.15, ...], ...]  // Top-6 weights
        }
      ]
    }
  ]
}
```

## Hardware Requirements

| Model | VRAM Required |
|-------|---------------|
| DeepSeek-V2-Lite | ~40GB |
| Qwen3-30B-A3B | ~60GB |

Both models work well on A100 (80GB).

## Estimated Runtime

| Stage | DeepSeek | Qwen |
|-------|----------|------|
| Model Loading | ~2 min | ~5 min |
| Dataset Evaluation | ~30 min | ~60 min |
| KDE Training | ~5 min | ~10 min |
| Plot Generation | ~30 min | ~60 min |
| **Total** | **~1 hour** | **~2.5 hours** |

## Model-Specific Details

### DeepSeek-V2-Lite
- Router gate location: `model.layers[i].mlp.gate` (nn.Linear)
- Logit computation: `F.linear(hidden_states, gate.weight)`
- 27 MoE layers, 64 experts per layer

### Qwen3-30B-A3B
- Router gate location: `model.layers[i].mlp.gate` (inside MoE block)
- Logit computation: `mlp.gate(hidden_states.view(-1, hidden_dim))`
- 48 MoE layers, 128 experts per layer

## Compatibility with logs_eda.ipynb

The JSON format produced by these pipelines is compatible with the `logs_eda.ipynb` notebook. The key fields are:
- `router_logits_sample`: Full router logits for all experts (used for KDE training)
- `selected_experts`: Top-k selected expert indices
- `expert_weights`: Top-k expert weights (post-softmax)

## License

MIT License
