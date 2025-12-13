# OLMoE Routing Analysis - Notebook Suite Plan

**Date**: 2025-12-13
**Purpose**: Align all notebooks with multi-expert analysis (max_k = 8, 16, 32, 64)

---

## Notebook Suite Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OLMOE ROUTING NOTEBOOK SUITE                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. OLMoE_Baseline_Colab.ipynb          [NEW - TO CREATE]          │
│     └─► Understand baseline OLMoE routing (Top-8)                  │
│         - No modifications, pure observation                        │
│         - Capture routing stats, distributions, patterns            │
│         - Identify optimization opportunities                       │
│                                                                     │
│  2. OLMoE_BenjaminiHochberg_Routing.ipynb  [UPDATE]                │
│     └─► Deep dive into BH routing implementation                   │
│         - BH routing with max_k = 8, 16, 32, 64                     │
│         - Multi-expert analysis and saturation study                │
│         - 4 new visualization plots                                 │
│                                                                     │
│  3. OLMoE_Full_Routing_Experiments.ipynb   [UPDATE]                │
│     └─► Comprehensive comparison of ALL routing strategies         │
│         - Regular, Normalized, Uniform, Adaptive, BH                │
│         - BH with max_k = 4, 8, 16, 32, 64                          │
│         - Test on WikiText, HellaSwag, LAMBADA                      │
│         - Full comparison matrix (57+ experiments)                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. OLMoE_Baseline_Colab.ipynb [NEW]

### Purpose
Understand how OLMoE works BEFORE applying BH routing

### Structure (12 cells)

```
Cell 1: Title & Introduction
────────────────────────────
# OLMoE Baseline Analysis
Understanding Standard Top-8 Routing Behavior

Cell 2: Environment Setup
────────────────────────────
!pip install transformers torch accelerate matplotlib seaborn

Cell 3: Load Model
────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924")

Cell 4: Register Routing Hooks
────────────────────────────
# Capture routing stats without modification
routing_stats = {
    'router_logits': [],
    'expert_indices': [],
    'routing_weights': []
}

Cell 5: Run Inference
────────────────────────────
# Test prompts of varying complexity
prompts = [
    "The capital of France is",
    "In quantum mechanics, the uncertainty principle states that",
    "Once upon a time, there was a"
]

Cell 6: Analyze Routing Statistics
────────────────────────────
# Compute per-layer statistics
for layer_idx in range(16):
    stats = {
        'avg_max_weight': ...,
        'entropy': ...,
        'top_expert_usage': ...
    }

Cell 7: Visualization 1 - Router Logit Distributions
────────────────────────────
# Plot logit distributions per layer

Cell 8: Visualization 2 - Expert Usage Heatmap
────────────────────────────
# Which experts are used most frequently

Cell 9: Visualization 3 - Routing Entropy
────────────────────────────
# Entropy across layers

Cell 10: Visualization 4 - Weight Concentration
────────────────────────────
# How concentrated are the routing weights

Cell 11: Key Observations
────────────────────────────
## Baseline OLMoE Behavior
1. Top-8 routing is FIXED (always 8 experts)
2. Expert usage is imbalanced
3. Simple tokens could use fewer experts
4. Opportunity for adaptive routing!

Cell 12: Next Steps
────────────────────────────
Continue to:
- OLMoE_BenjaminiHochberg_Routing.ipynb (BH implementation)
- OLMoE_Full_Routing_Experiments.ipynb (comprehensive comparison)
```

### Expected Output

```
BASELINE ROUTING STATISTICS (Standard Top-8)
═══════════════════════════════════════════════════════════════
Layer    Avg Max Weight    Entropy    Top Expert Usage
  0         0.342          1.82       Expert 12 (15.2%)
  1         0.298          1.94       Expert 7  (12.8%)
  ...
 15         0.267          2.01       Expert 23 (11.3%)
═══════════════════════════════════════════════════════════════

KEY FINDINGS:
- Some experts are used 3x more than others
- Routing entropy varies by layer (early layers more confident)
- Simple tokens have higher max weight (fewer experts needed)
```

---

## 2. OLMoE_BenjaminiHochberg_Routing.ipynb [UPDATE]

### Updates Needed

**Add after Cell 5 (Analyzer Setup):**

```python
# ============================================================================
# CELL 5b: Multi-Expert Configuration
# ============================================================================

MAX_K_VALUES = [8, 16, 32, 64]
ALPHA_VALUES = [0.01, 0.05, 0.10]

print("Multi-Expert Experiment Configuration")
print("=" * 60)
print(f"max_k values: {MAX_K_VALUES}")
print(f"alpha values: {ALPHA_VALUES}")
print(f"Total BH configurations: {len(MAX_K_VALUES) * len(ALPHA_VALUES)}")
```

**Add after Cell 10 (Comparative Analysis):**

```python
# ============================================================================
# CELL 10b: Multi-Expert BH Routing Analysis
# ============================================================================

test_prompt = "The theory of relativity fundamentally changed our understanding of"

multi_expert_results = []

# Run experiments for each max_k and alpha combination
for max_k in tqdm(MAX_K_VALUES, desc="Testing max_k values"):
    # Baseline TopK
    baseline_stats = run_topk_baseline(model, tokenizer, test_prompt, k=max_k)

    # BH with different alphas
    for alpha in ALPHA_VALUES:
        analyzer = BHRoutingAnalyzer(model, alpha=alpha, max_k=max_k)
        analyzer.patch_model()

        # Run inference and collect stats
        outputs = model.generate(...)
        stats = analyzer.get_stats()

        multi_expert_results.append({
            'method': 'BH',
            'max_k': max_k,
            'alpha': alpha,
            'avg_experts': stats['mean_experts']
        })

        analyzer.unpatch_model()

# Display summary table
multi_df = pd.DataFrame(multi_expert_results)
summary_table = multi_df.pivot_table(
    values='avg_experts',
    index='alpha',
    columns='max_k'
)
print(summary_table)
```

**Add Cell 10c: Multi-Expert Visualizations**

```python
# 4 new plots:
# 1. BH vs TopK by max_k (bar chart)
# 2. Heatmap - Alpha x max_k
# 3. Line plot - Saturation analysis
# 4. Reduction percentage
```

**Update Conclusions Cell:**

```markdown
### Multi-Expert Findings

1. **Saturation Point**: BH routing shows diminishing returns beyond max_k=16
2. **Alpha Interaction**: Lower alpha benefits less from higher max_k
3. **Recommendation**:
   - For efficiency: max_k=8, α=0.05
   - For quality: max_k=16, α=0.10
```

---

## 3. OLMoE_Full_Routing_Experiments.ipynb [UPDATE]

### Current Routing Strategies
- Regular, Normalized, Uniform, Adaptive

### **ADD: Benjamini-Hochberg with Multi-Expert Support**

```python
ROUTING_CONFIGS = {
    # Existing strategies
    'regular': {...},
    'normalized': {...},
    'uniform': {...},
    'adaptive': {...},

    # NEW: BH with multiple max_k values
    'bh_k4_a005': {'method': 'bh', 'max_k': 4, 'alpha': 0.05},
    'bh_k8_a001': {'method': 'bh', 'max_k': 8, 'alpha': 0.01},
    'bh_k8_a005': {'method': 'bh', 'max_k': 8, 'alpha': 0.05},
    'bh_k8_a010': {'method': 'bh', 'max_k': 8, 'alpha': 0.10},
    'bh_k16_a005': {'method': 'bh', 'max_k': 16, 'alpha': 0.05},
    'bh_k32_a005': {'method': 'bh', 'max_k': 32, 'alpha': 0.05},
    'bh_k64_a005': {'method': 'bh', 'max_k': 64, 'alpha': 0.05},
}

# Total configurations: ~15 strategies
```

### Updated Experiment Matrix

| Strategy | Expert Counts | Datasets | Experiments |
|----------|---------------|----------|-------------|
| Regular | 2, 4, 6, 8 | WikiText, HellaSwag, LAMBADA | 12 |
| Normalized | 2, 4, 6, 8 | WikiText, HellaSwag, LAMBADA | 12 |
| Uniform | 2, 4, 6, 8 | WikiText, HellaSwag, LAMBADA | 12 |
| Adaptive | 2, 4, 6, 8 | WikiText, HellaSwag, LAMBADA | 12 |
| **BH (NEW)** | **4, 8, 16, 32, 64** | **WikiText, HellaSwag, LAMBADA** | **21** |

**Total**: 69 experiments (was 48)

### Expected Results Table

```
COMPREHENSIVE ROUTING COMPARISON
════════════════════════════════════════════════════════════════════════
Strategy        Experts   WikiText PPL   HellaSwag Acc   LAMBADA Acc
────────────────────────────────────────────────────────────────────────
Regular            8         12.45          58.2%          65.3%
Normalized         8         12.41          58.4%          65.5%
Uniform            8         12.89          57.1%          64.2%
Adaptive          4-8        12.52          58.0%          65.1%

BH (α=0.05, k=4)   4         12.98          57.5%          64.8%
BH (α=0.05, k=8)  4-6        12.48          58.1%          65.2%
BH (α=0.05, k=16) 6-8        12.43          58.3%          65.4%
BH (α=0.05, k=32) 8-12       12.41          58.4%          65.5%
BH (α=0.05, k=64) 10-15      12.40          58.5%          65.6%
════════════════════════════════════════════════════════════════════════

KEY FINDING: BH with max_k=16 achieves best balance
- Similar quality to TopK-8
- 40-50% fewer experts actually used
- Minimal performance degradation
```

---

## Implementation Plan

### Phase 1: Create Baseline Notebook ✅

```bash
# Create OLMoE_Baseline_Colab.ipynb
# - 12 cells
# - Baseline analysis only
# - No BH routing
# - Focus on understanding standard Top-8 behavior
```

### Phase 2: Update BH Routing Notebook ✅

```bash
# Update OLMoE_BenjaminiHochberg_Routing.ipynb
# - Add Cell 5b: Multi-expert configuration
# - Add Cell 10b: Multi-expert BH analysis
# - Add Cell 10c: Multi-expert visualizations (4 plots)
# - Update conclusions with multi-expert findings
```

### Phase 3: Update Full Routing Experiments ✅

```bash
# Update OLMoE_Full_Routing_Experiments.ipynb
# - Add BH routing configs with max_k = 4, 8, 16, 32, 64
# - Update experiment matrix (69 total experiments)
# - Add BH results to comparison table
# - Create comprehensive comparison visualizations
```

---

## Notebook Dependencies

All notebooks require:
```python
!pip install transformers==4.36.0 \
             torch>=2.0.0 \
             accelerate \
             datasets \
             matplotlib \
             seaborn \
             pandas \
             numpy \
             scipy \
             tqdm
```

---

## Colab Compatibility

All notebooks designed for Google Colab:
- ✅ GPU detection and configuration
- ✅ Package installation cells
- ✅ Progress bars (tqdm)
- ✅ Automatic model download
- ✅ Results export to Google Drive (optional)

---

## Expected Outputs

### From Baseline Notebook
- `baseline_routing_stats.csv`
- `baseline_plots/` (4 visualizations)

### From BH Routing Notebook
- `bh_routing_results.csv`
- `bh_plots/` (7 visualizations, including 4 new multi-expert plots)

### From Full Routing Experiments
- `full_routing_results.csv` (69 rows)
- `comparison_plots/` (comprehensive comparison)
- `ROUTING_COMPARISON_REPORT.md`

---

## Timeline

1. **Create Baseline Notebook**: ~30 min
2. **Update BH Notebook**: ~20 min
3. **Update Full Experiments**: ~40 min
4. **Test on Colab**: ~30 min
5. **Documentation**: ~20 min

**Total**: ~2-3 hours

---

## Success Criteria

- ✅ All 3 notebooks run successfully on Colab
- ✅ Multi-expert analysis (8, 16, 32, 64) in all relevant notebooks
- ✅ Comprehensive comparison of all routing strategies
- ✅ Clear progression: Baseline → BH Deep Dive → Comprehensive Comparison
- ✅ All visualizations render correctly
- ✅ Results exportable and reproducible

---

*Last updated: 2025-12-13*
