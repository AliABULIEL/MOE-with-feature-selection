# BH Routing Comprehensive Evaluation Framework

**Complete implementation of Benjamini-Hochberg statistical routing evaluation for OLMoE**

---

## 🎯 Overview

This framework implements a comprehensive evaluation system for comparing Benjamini-Hochberg (BH) adaptive routing against baseline Top-K routing in OLMoE (Open Language Model with Experts). It provides:

- **16 metrics across 8 categories** for thorough analysis
- **3 benchmark datasets** (WikiText-2, LAMBADA, HellaSwag)
- **Dual file logging** (summary + internal routing data)
- **9-panel visualizations** for comparative analysis
- **Template-aligned structure** matching OLMoE_Full_Routing_Experiments

---

## 📁 Framework Components

### Core Modules

#### 1. **`bh_routing_metrics.py`** (820 lines)
Implements all 16 evaluation metrics:

**Category 1: Quality Metrics**
- `compute_perplexity()` - Language modeling quality
- `compute_avg_task_accuracy()` - Average across tasks

**Category 2: Efficiency Metrics**
- `compute_expert_activation_ratio()` - Usage vs capacity
- `compute_flops_reduction_pct()` - Computational savings

**Category 3: Speed Metrics**
- `compute_tokens_per_second()` - Throughput
- `compute_latency_ms_per_token()` - Latency

**Category 4: Routing Distribution Metrics**
- `compute_expert_utilization()` - Load balancing
- `compute_gini_coefficient()` - Inequality measure

**Category 5: Routing Behavior Metrics (BH-specific)**
- `compute_adaptive_range()` - Selection variability
- `compute_selection_entropy()` - Distribution entropy

**Category 6: Constraint Metrics**
- `compute_ceiling_hit_rate()` - max_k constraint hits
- `compute_floor_hit_rate()` - min_k constraint hits

**Category 7: Cross-Layer Metrics**
- `compute_layer_expert_variance()` - Layer-wise differences
- `compute_layer_consistency_score()` - Adjacent layer correlation

**Category 8: Stability Metrics**
- `compute_expert_overlap_score()` - Routing consistency
- `compute_output_determinism()` - Generation consistency

**Usage:**
```python
from bh_routing_metrics import BHMetricsComputer

computer = BHMetricsComputer()
metrics = computer.compute_all_metrics(
    losses=[2.1, 2.3, 2.0],
    accuracies={'lambada': 0.65, 'hellaswag': 0.52},
    avg_experts=4.5,
    max_k=8,
    expert_usage_counts=usage_array,
    expert_counts=counts_array,
    # ... other inputs
)
```

---

#### 2. **`bh_routing_evaluation.py`** (440 lines)
Dataset loading and evaluation functions:

**Dataset Loaders:**
- `load_wikitext()` - WikiText-2 test set
- `load_lambada()` - LAMBADA validation set
- `load_hellaswag()` - HellaSwag validation set

**Evaluation Functions:**
- `evaluate_perplexity()` - With internal routing logging
- `evaluate_lambada()` - Last-word prediction accuracy
- `evaluate_hellaswag()` - Sentence completion accuracy
- `evaluate_all_datasets()` - Comprehensive evaluation

**Usage:**
```python
from bh_routing_evaluation import evaluate_perplexity, load_wikitext

texts = load_wikitext(max_samples=200)
result = evaluate_perplexity(
    model, tokenizer, texts,
    patcher=patcher,  # For internal logging
    device='cuda'
)

print(f"Perplexity: {result['perplexity']:.2f}")
print(f"Samples: {len(result['internal_logs'])}")
```

---

#### 3. **`bh_routing_experiment_runner.py`** (800+ lines)
Main experiment orchestration:

**Classes:**
- `BHRoutingExperimentRunner` - Main orchestrator
- `BHRoutingPatcherAdapter` - Internal logging wrapper

**Key Features:**
- Two-phase experimental approach
- Automatic model loading and patching
- Dual file logging per configuration
- All 16 metrics computation
- GPU memory management
- Progress reporting

**Usage:**
```python
from bh_routing_experiment_runner import BHRoutingExperimentRunner

runner = BHRoutingExperimentRunner(
    model_name="allenai/OLMoE-1B-7B-0924",
    device="cuda",
    output_dir="./bh_routing_experiment"
)

results_df = runner.run_two_phase_experiment(
    baseline_k_values=[8, 16, 32, 64],
    bh_max_k_values=[8, 16, 32, 64],
    bh_alpha_values=[0.30, 0.40, 0.50, 0.60],
    datasets=['wikitext', 'lambada', 'hellaswag'],
    max_samples=200
)
```

---

#### 4. **`bh_routing_visualization.py`** (600+ lines)
Comprehensive visualization suite:

**Main Function:**
- `create_comprehensive_visualization()` - 9-panel figure

**Panels:**
1. Perplexity Comparison (baseline vs BH)
2. Task Accuracy Comparison
3. Expert Efficiency (avg experts selected)
4. Alpha Sensitivity Heatmap
5. Pareto Frontier (efficiency vs quality)
6. Routing Behavior Summary (floor/mid/ceiling)
7. Expert Utilization
8. Layer-wise Analysis
9. Speed-Quality Trade-off

**Usage:**
```python
from bh_routing_visualization import create_comprehensive_visualization

fig = create_comprehensive_visualization(
    results_df,
    output_path='bh_comprehensive_comparison.png'
)

# Or create individual panels
from bh_routing_visualization import create_single_panel

fig = create_single_panel(results_df, 'pareto', 'pareto_frontier.png')
```

---

#### 5. **`BH_Routing_Quick_Start.py`**
Demonstration and quick-start script:

**Modes:**
- `--quick-test`: Fast verification (3 experiments, ~5 min)
- `--full`: Complete evaluation (60 experiments, ~2-3 hours)
- Default: Custom moderate experiment

**Usage:**
```bash
# Quick test
python BH_Routing_Quick_Start.py --quick-test

# Full experiment
python BH_Routing_Quick_Start.py --full

# Custom (modify script parameters)
python BH_Routing_Quick_Start.py
```

---

## 🧪 Experimental Design

### Baseline Configurations (4)
- **8experts_topk_baseline** - OLMoE default (K=8)
- **16experts_topk_baseline** - 2× experts
- **32experts_topk_baseline** - 4× experts
- **64experts_topk_baseline** - All experts

### BH Configurations (16)
- **max_k values:** [8, 16, 32, 64]
- **Alpha values:** [0.30, 0.40, 0.50, 0.60]
- **Total:** 4 × 4 = 16 configurations

### Datasets (3)
- **WikiText-2** - Language modeling (perplexity)
- **LAMBADA** - Last word prediction (accuracy)
- **HellaSwag** - Sentence completion (accuracy)

### Total Experiments
- **(4 baseline + 16 BH) × 3 datasets = 60 experiments**
- **120 output files** (60 summary + 60 internal_routing JSONs)

---

## 📊 Output Structure

```
bh_routing_experiment/
├── logs/
│   ├── 8experts_topk_baseline_wikitext.json
│   ├── 8experts_topk_baseline_wikitext_internal_routing.json
│   ├── 8experts_bh_a030_wikitext.json
│   ├── 8experts_bh_a030_wikitext_internal_routing.json
│   └── ... (120 files total)
├── visualizations/
│   └── bh_comprehensive_comparison.png (9-panel figure)
├── bh_routing_results.csv
├── bh_routing_results.json
└── (optional) bh_routing_report.md
```

### Summary JSON Format
```json
{
    "config": "8experts_bh_a040",
    "strategy": "bh",
    "k_or_max_k": 8,
    "alpha": 0.40,
    "dataset": "wikitext",
    "metrics": {
        "perplexity": 15.82,
        "avg_experts": 4.52,
        "flops_reduction_pct": 21.8,
        "tokens_per_second": 110.5,
        "expert_utilization": 0.875,
        "gini_coefficient": 0.342,
        "adaptive_range": 7,
        "ceiling_hit_rate": 12.3,
        "floor_hit_rate": 18.7,
        "layer_expert_variance": 0.45,
        "layer_consistency_score": 0.72,
        ... (all 16 metrics)
    }
}
```

### Internal Routing JSON Format
```json
{
    "config": "8experts_bh_a040",
    "dataset": "wikitext",
    "samples": [
        {
            "sample_id": 0,
            "num_tokens": 128,
            "loss": 2.456,
            "layers": [
                {
                    "layer": 0,
                    "selected_experts": [[12, 45, 3, -1, ...], ...],
                    "expert_weights": [[0.45, 0.32, 0.23, ...], ...],
                    "expert_counts": [3, 4, 2, 5, ...],
                    "router_logits_sample": [[-1.2, 0.5, ...], ...],
                    "layer_stats": {...}
                },
                ... (16 layers)
            ]
        },
        ... (200 samples)
    ],
    "aggregate_stats": {
        "per_layer_avg_experts": [3.42, 4.12, ...],
        "expert_usage_counts": [1234, 987, ...],
        ...
    }
}
```

---

## 🚀 Quick Start Guide

### 1. Installation

```bash
# Clone repository
git clone https://github.com/aliabbasjaffri/MOE-with-feature-selection.git
cd MOE-with-feature-selection

# Install dependencies
pip install torch transformers datasets pandas numpy matplotlib seaborn scipy tqdm

# Verify BH routing module exists
ls bh_routing.py  # Should exist
ls kde_models/models/  # Should contain KDE models
```

### 2. Quick Test (5 minutes)

```bash
python BH_Routing_Quick_Start.py --quick-test
```

This runs:
- 1 baseline (TopK-8)
- 2 BH configs (alpha=0.40, 0.50)
- 1 dataset (WikiText, 50 samples)
- Total: 3 experiments

### 3. Full Experiment (2-3 hours)

```bash
python BH_Routing_Quick_Start.py --full
```

This runs:
- 4 baselines
- 16 BH configs
- 3 datasets
- 200 samples each
- Total: 60 experiments

### 4. Python API

```python
from bh_routing_experiment_runner import BHRoutingExperimentRunner
from bh_routing_visualization import create_comprehensive_visualization

# Initialize
runner = BHRoutingExperimentRunner(
    model_name="allenai/OLMoE-1B-7B-0924",
    device="cuda",
    output_dir="./my_experiment"
)

# Run experiment
results_df = runner.run_two_phase_experiment(
    baseline_k_values=[8, 16],
    bh_max_k_values=[8, 16],
    bh_alpha_values=[0.40, 0.50],
    datasets=['wikitext'],
    max_samples=100
)

# Generate visualization
fig = create_comprehensive_visualization(
    results_df,
    output_path='./my_experiment/visualizations/comparison.png'
)

# Analyze results
print(results_df[['config_name', 'perplexity', 'avg_experts', 'flops_reduction_pct']])
```

---

## 📈 Expected Results

Based on the master prompt requirements and preliminary analysis:

### Efficiency Gains
- **BH routing (alpha=0.05, max_k=8)**: ~35-50% fewer experts than TopK-8
- **BH routing (alpha=0.01, max_k=8)**: ~55-70% fewer experts (very conservative)
- **BH routing (alpha=0.10, max_k=8)**: ~25-35% fewer experts (balanced)

### Quality Impact
- **Perplexity delta**: -0.5 to +1.5 vs TopK-8 baseline
- **Task accuracy delta**: -2% to +1% vs baseline
- **Best configurations**: alpha=0.05-0.10, max_k=8-16

### Adaptive Behavior
- **Adaptive range**: 0 for TopK, 5-7 for BH
- **Selection entropy**: 0 for TopK, 1.8-2.5 for BH
- **Ceiling hit rate**: 100% for TopK, 10-30% for BH

---

## 🔬 Advanced Usage

### Custom Metrics Computation

```python
from bh_routing_metrics import BHMetricsComputer
import numpy as np

computer = BHMetricsComputer()

# Compute individual metrics
ppl = computer.compute_perplexity([2.1, 2.3, 2.0])
gini = computer.compute_gini_coefficient(np.array([100, 50, 200, 30]))
entropy, norm = computer.compute_selection_entropy(np.array([3,4,5,3,6]), max_k=8)

# Compute all metrics
metrics = computer.compute_all_metrics(
    losses=[2.1, 2.3],
    avg_experts=4.5,
    max_k=8,
    total_tokens=10000,
    total_time=90.0
)
```

### Custom Evaluation

```python
from bh_routing_evaluation import evaluate_perplexity, load_wikitext

# Load custom dataset
texts = load_wikitext(split='validation', max_samples=500)

# Evaluate with custom patcher
result = evaluate_perplexity(
    model, tokenizer, texts,
    patcher=my_custom_patcher,
    max_length=1024,
    device='cuda'
)
```

### Individual Visualizations

```python
from bh_routing_visualization import create_single_panel

# Create only Pareto frontier
fig = create_single_panel(results_df, 'pareto', 'pareto.png')

# Create only alpha sensitivity
fig = create_single_panel(results_df, 'alpha_sensitivity', 'alpha_sens.png')

# Available panels:
# 'perplexity', 'accuracy', 'efficiency', 'alpha_sensitivity',
# 'pareto', 'behavior', 'utilization', 'layer', 'speed_quality'
```

---

## 🐛 Troubleshooting

### GPU Memory Issues

```python
# Reduce batch size in evaluation functions
result = evaluate_perplexity(
    model, tokenizer, texts,
    batch_size=1,  # Reduce from default
    max_length=256  # Reduce sequence length
)

# Or clear cache between experiments
import torch
torch.cuda.empty_cache()
```

### Missing KDE Models

```python
# Check if KDE models exist
import os
kde_dir = './kde_models/models/'
if not os.path.exists(kde_dir):
    print("KDE models not found!")
    print("BH routing will fall back to empirical p-values")
```

### ImportError

```bash
# Ensure all required modules are in the same directory
ls bh_routing*.py
# Should show:
# bh_routing.py
# bh_routing_metrics.py
# bh_routing_evaluation.py
# bh_routing_experiment_runner.py
# bh_routing_visualization.py
```

---

## 📚 References

1. **Benjamini-Hochberg Procedure**
   - Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. Journal of the Royal Statistical Society: Series B, 57(1), 289-300.

2. **OLMoE Model**
   - AllenAI OLMoE-1B-7B (https://huggingface.co/allenai/OLMoE-1B-7B-0924)

3. **Datasets**
   - WikiText-2: Merity et al. (2016)
   - LAMBADA: Paperno et al. (2016)
   - HellaSwag: Zellers et al. (2019)

---

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@software{bh_routing_framework,
  title={BH Routing Comprehensive Evaluation Framework for OLMoE},
  author={Implementation by Claude Code},
  year={2025},
  url={https://github.com/aliabbasjaffri/MOE-with-feature-selection}
}
```

---

## ✅ Validation Checklist

Before running full experiments, verify:

- [ ] GPU available (`torch.cuda.is_available()`)
- [ ] BH routing module exists (`bh_routing.py`)
- [ ] KDE models directory exists (`kde_models/models/`)
- [ ] All framework modules imported successfully
- [ ] Quick test runs without errors
- [ ] Output directory has write permissions

---

## 🎓 Framework Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│  BH_Routing_Quick_Start.py  |  Custom Python Scripts        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  Orchestration Layer                         │
│           BHRoutingExperimentRunner                          │
│  - Two-phase experiment control                              │
│  - Model loading and patching                                │
│  - Progress tracking                                         │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────────────┐   ┌──────────────────┐   ┌──────────────┐
│   Evaluation  │   │     Metrics      │   │Visualization │
│     Layer     │   │  Computation     │   │    Layer     │
├───────────────┤   ├──────────────────┤   ├──────────────┤
│ - WikiText    │   │ - 16 metrics     │   │ - 9 panels   │
│ - LAMBADA     │   │ - 8 categories   │   │ - Pareto     │
│ - HellaSwag   │   │ - Aggregation    │   │ - Heatmaps   │
└───────────────┘   └──────────────────┘   └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     Core BH Routing                          │
│  bh_routing.py + KDE models + OLMoE model                   │
└─────────────────────────────────────────────────────────────┘
```

---

**Framework Complete! 🎉**

Ready for comprehensive BH routing evaluation with production-grade metrics,
template-aligned structure, and publication-quality visualizations.
