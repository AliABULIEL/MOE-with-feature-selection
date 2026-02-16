# Multi-Model HC Routing Workflow

## Overview
Python scripts for running HC routing experiments on DeepSeek-v2-Lite and Qwen3-30B-A3B with configurable estimators.

## Quick Start

### 1. Install Dependencies
```bash
pip install torch transformers datasets pydantic pyyaml numpy scipy scikit-learn
```

### 2. Run Experiments

**DeepSeek with KDE:**
```bash
python run_hc_workflow.py --config configs/deepseek_kde.yaml
```

**DeepSeek with Student-T:**
```bash
python run_hc_workflow.py --config configs/deepseek_student_t.yaml
```

**Qwen with KDE:**
```bash
python run_hc_workflow.py --config configs/qwen_kde.yaml
```

**Qwen with GMM:**
```bash
python run_hc_workflow.py --config configs/qwen_gmm.yaml
```

## Configuration

Configuration files are in YAML format. Key sections:

- **model**: Model selection (deepseek/qwen), model ID, dtype
- **estimator**: Estimator type (kde/student_t/gmm/etc.), parameters
- **hc_routing**: HC parameters (beta, min_k, max_k, patch_mode)
- **evaluation**: Datasets, sample counts, batch size
- **output**: Results directory, plotting options

## Estimator Types

Supported estimators (from `estimators/` directory):

- `kde`: Kernel Density Estimation
- `student_t`: Student-T distribution
- `student_t_skewed`: Skewed Student-T
- `gmm`: Gaussian Mixture Model
- `lindsey_glm`: Lindsey GLM method
- `advanced`: Advanced estimator

Configure via `estimator.type` in config file.

## Output Structure

```
results/{experiment_name}/
├── config.yaml           # Saved configuration
├── all_results.json      # Combined results
├── logs/                 # Detailed logs
├── metrics/             # Per-dataset metrics
└── plots/               # Visualizations
```

## Cloud Deployment

Scripts are designed for cloud environments:
- No interactive prompts
- Command-line driven
- Configurable via YAML
- Automatic logging and metrics

Example cloud usage:
```bash
# On compute cluster
python run_hc_workflow.py --config my_experiment.yaml
```

## Architecture

### Core Modules

- **config_schema.py**: Pydantic configuration validation
- **model_loader.py**: Model loading (DeepSeek/Qwen)
- **estimator_factory.py**: Estimator creation and management
- **run_hc_workflow.py**: Main workflow script

### Integration

Uses existing modules:
- `hc_routing.py`: HC routing logic
- `hc_routing_evaluation.py`: Dataset loading
- `hc_routing_metrics.py`: Metrics computation
- `moe_internal_logging_{deepseek,qwen}.py`: Model-specific logging

## Example: Custom Configuration

```yaml
model:
  name: "deepseek"
  model_id: "deepseek-ai/DeepSeek-V2-Lite"
  num_experts: 64
  default_top_k: 6

estimator:
  type: "student_t"
  quantile_range: [0.05, 0.95]
  bins: 71

hc_routing:
  beta: "auto"  # Adaptive search
  min_k: 2
  max_k: 12
  patch_mode: 4  # Last 4 layers only

evaluation:
  datasets: ["wikitext", "lambada", "hellaswag"]
  max_samples: 500

output:
  results_dir: "./results/my_experiment"
```

## Notes

- Quantile range is `[0.05, 0.95]` as recommended
- Model loading follows exact patterns from `pipelines/run_{model}_pipeline.py`
- Estimators use `.cdf()` method for p-value computation
- Router patching requires further integration work (TODO)

## Next Steps

1. Implement full router patching for DeepSeek/Qwen
2. Add estimator training on collected routing data
3. Add comprehensive visualization generation
4. Add support for saving/loading fitted estimators
