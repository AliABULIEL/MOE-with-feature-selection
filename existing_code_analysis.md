# Existing Code Analysis: OLMoE Routing Infrastructure

**Document Purpose**: This analysis identifies current capabilities, gaps, and integration points for implementing Benjamini-Hochberg (BH) statistical routing in the OLMoE model.

**Date**: 2025-12-13
**Repository**: MOE-with-feature-selection

---

## Table of Contents
1. [Current Capabilities](#1-current-capabilities)
2. [File-by-File Analysis](#2-file-by-file-analysis)
3. [Routing Data Flow](#3-routing-data-flow)
4. [Hook Infrastructure](#4-hook-infrastructure)
5. [Statistical Analysis Capabilities](#5-statistical-analysis-capabilities)
6. [Gaps for BH Routing](#6-gaps-for-bh-routing)
7. [Reusable Components](#7-reusable-components)
8. [Integration Points](#8-integration-points)

---

## 1. Current Capabilities

### 1.1 What We Already Have

✅ **Complete OLMoE Model Access**
- Full integration with HuggingFace transformers library
- Model: `allenai/OLMoE-1B-7B-0924` (1.3B active, 6.9B total parameters)
- Configuration: 16 layers, 64 experts per layer, top-k=8 default routing

✅ **Routing Interception Infrastructure**
- Forward hook registration system on `layer.mlp.gate` modules
- Captures router logits, expert indices, and weights during inference
- Support for both hook-based and model-patching approaches

✅ **Multiple Routing Strategies Implementation**
- RegularRouting: Standard softmax-based top-k
- NormalizedRouting: Top-k with re-normalized weights
- UniformRouting: Top-k with equal weights (1/k each)
- AdaptiveRouting: Dynamic expert count based on confidence

✅ **Model Patching Capabilities**
- `ModelPatchingUtils` class for injecting custom routing logic
- Ability to replace forward passes at MLP layer level
- Custom `custom_select_experts()` function with configurable strategies
- Support for patching/unpatching without model reload

✅ **Comprehensive Logging & Data Collection**
- Per-layer router logits capture
- Expert selection tracking (indices + weights)
- Token-level routing statistics (entropy, concentration, max weight)
- Expert utilization tracking across all layers
- JSON-based internal routing logs for offline analysis

✅ **Evaluation Infrastructure**
- Multi-dataset support (HellaSwag, LAMBADA, Wikitext, etc.)
- Metrics: perplexity, token accuracy, loss, inference speed
- Experiment orchestration with `RoutingExperimentRunner`
- Results aggregation and visualization

✅ **Statistical Analysis Tools** (from `logs_eda.ipynb`)
- KDE (Kernel Density Estimation) models for router logit distributions
- Per-layer, per-expert distribution analysis
- P-value computation infrastructure
- Trimmed distribution analysis (removing top/bottom experts)
- Cumulative density function (CDF) estimation
- Multiple kernel support (Gaussian, Tophat, Epanechnikov, Linear)
- Empirical vs. KDE comparison

---

## 2. File-by-File Analysis

### 2.1 `routing_mechanism_detailed.py` (706 lines)

**Purpose**: Comprehensive documentation and pseudocode for BH routing injection

**Key Contents**:
1. **OLMoE Architecture Documentation**
   - Exact class definitions from transformers library
   - `OlmoeTopKRouter`: Line-by-line routing implementation
   - `OlmoeExperts`: Expert computation mechanism
   - `OlmoeSparseMoeBlock`: Complete MoE block flow

2. **Critical Injection Point Identified**
   ```python
   # Line ~117 in OlmoeTopKRouter.forward()
   router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
   # ⬆️ REPLACE THIS WITH BH ROUTING
   ```

3. **Tensor Shape Documentation**
   - Input: `hidden_states` [batch_size, seq_len, hidden_dim] → [B, L, 2048]
   - Router logits: [num_tokens, num_experts] → [B*L, 64]
   - After softmax: [B*L, 64] (probabilities sum to 1)
   - Selected weights: [B*L, top_k] → [B*L, 8]
   - Selected indices: [B*L, top_k] → [B*L, 8]

4. **BH Routing Algorithm Pseudocode**
   - Detailed implementation plan
   - Vectorized vs. per-token approaches
   - P-value computation strategies
   - Compatibility considerations

5. **Load Balancing Loss Analysis**
   - Switch Transformer-style auxiliary loss
   - Handles attention masking for padding tokens
   - Per-expert token distribution balancing

**What It Extracts**: None (documentation only)

**How It Hooks**: None (provides injection strategy)

**Statistics Collected**: None (theoretical framework)

**Value for BH Routing**: ⭐⭐⭐⭐⭐
- Provides exact code structure for injection
- Documents all tensor shapes and flows
- Contains detailed BH implementation pseudocode

---

### 2.2 `olmoe_routing_experiments.py` (1,900+ lines)

**Purpose**: Main experiment framework for testing different routing strategies

**Key Components**:

#### A. Routing Strategy Classes (lines 89-259)
```python
class RoutingStrategy:
    def route(self, logits) -> (indices, weights)
    def update_stats(self, expert_indices, expert_weights)
    def get_summary_stats() -> Dict
```

**Implementations**:
- `RegularRouting`: Softmax → top-k
- `NormalizedRouting`: Softmax → top-k → renormalize
- `UniformRouting`: Softmax → top-k → equal weights (1/k)
- `AdaptiveRouting`: Dynamic k based on confidence thresholds

**Statistics Tracked**:
- Max weights per token
- Routing entropy: -∑(w * log(w))
- Weight concentration: max(w) / sum(w)
- Unique experts activated

#### B. Model Patching System (lines 261-426)

```python
class ModelPatchingUtils:
    @staticmethod
    def custom_select_experts(router_logits, top_k, num_experts, strategy)
        # Applies routing strategy to logits
        # Returns: (routing_weights, selected_experts)

    @staticmethod
    def create_patched_forward(top_k, num_experts, strategy)
        # Creates new forward() that uses custom_select_experts()
        # Replaces standard MLP forward pass

    @staticmethod
    def patch_model(model, top_k, strategy)
        # Patches all MoE layers with custom routing

    @staticmethod
    def unpatch_model(model, original_forwards)
        # Restores original forward passes
```

**Routing Strategies Supported in Patching**:
- `'uniform'`: Equal weights (1/k)
- `'normalized'`: Renormalized top-k weights
- `'baseline'`: Standard softmax weights

#### C. Hook Registration System (lines 536-588)

```python
def _register_router_hooks(self, top_k=8):
    """
    Registers forward hooks on layer.mlp.gate to capture router outputs
    More reliable than output_router_logits=True parameter
    """
    def logging_hook_router(module, input, output, layer_index, k=8):
        router_logits = output  # Raw logits from gate
        softmax_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(softmax_weights, k, dim=-1)

        self.logged_routing_data.append({
            "layer": layer_index,
            "router_logits": router_logits.detach(),
            "expert_indices": topk_indices.detach(),
            "softmax_weights": topk_weights.detach(),
        })
```

**Hook Locations**: `layer.mlp.gate` for each of 16 layers

**Data Captured Per Forward Pass**:
- Layer index
- Router logits (raw, pre-softmax)
- Expert indices (top-k)
- Softmax weights (top-k probabilities)

#### D. Experiment Runner (lines 428-end)

```python
class RoutingExperimentRunner:
    def __init__(model_name, device, output_dir, use_model_patching)

    def _set_expert_count(num_experts)
        # Updates model config + re-patches if needed

    def _register_router_hooks(top_k)
        # Installs hooks on all layers

    def load_dataset(dataset_name, split, max_samples)
        # Supports: wikitext, hellaswag, lambada, etc.

    def run_single_experiment(config, dataset_name, max_samples)
        # Runs one routing strategy on one dataset
        # Returns: ExperimentResults with metrics

    def run_all_experiments(expert_counts, strategies, datasets)
        # Orchestrates full experimental sweep
        # Saves: CSV, JSON, per-sample routing logs
```

**Experiment Outputs**:
- `all_results.csv`: Aggregated metrics
- `all_results.json`: Full results with metadata
- `logs/{config}_{dataset}_internal_routing.json`: Per-sample routing data

**Routing Information Extracted**:
- Router logits (raw and softmax)
- Selected expert indices
- Expert weights
- Per-token, per-layer granularity

**How It Hooks**:
1. Forward hooks on `layer.mlp.gate` modules
2. Captures output of gate (router logits)
3. Post-processes to get softmax + top-k
4. Stores in `self.logged_routing_data` list

**Statistics Collected**:
- Perplexity, loss, token accuracy
- Inference time, tokens/second
- Average experts used, max weight, entropy
- Weight concentration, unique experts activated
- Per-sample routing decisions (optional detailed logs)

**Value for BH Routing**: ⭐⭐⭐⭐⭐
- Complete infrastructure for adding new routing strategies
- Hook system ready for BH p-value computation
- Experiment runner can test BH vs. baselines
- Logging captures all necessary data

---

### 2.3 `analyze_results.py` (470 lines)

**Purpose**: Post-experiment analysis and comparison tool

**Key Features**:
- Loads results from `all_results.csv`
- Summary statistics across all experiments
- Per-strategy analysis
- Per-expert-count analysis
- Optimal configuration finder (quality vs. speed trade-off)
- Comparison plots

**CLI Commands**:
```bash
python analyze_results.py summary <results_dir>
python analyze_results.py strategy regular <results_dir>
python analyze_results.py compare <results_dir>
python analyze_results.py optimize --quality-weight 0.7 --speed-weight 0.3 <results_dir>
python analyze_results.py plot <results_dir>
```

**Visualizations**:
1. Perplexity distribution by strategy (box plots)
2. Token accuracy distribution by strategy
3. Speed vs. quality scatter plots
4. Perplexity vs. expert count line plots

**Value for BH Routing**: ⭐⭐⭐⭐
- Ready to compare BH against existing strategies
- No modifications needed
- Automated optimal config finder

---

### 2.4 `test_routing_strategies_cpu.py` (163 lines)

**Purpose**: Quick CPU-based test to verify routing strategies actually modify computation

**What It Does**:
1. Loads model on CPU (float32)
2. Tests single sample with 3 strategies:
   - Baseline (no patching)
   - Uniform weights
   - Normalized weights
3. Verifies losses are different (proves patching works)

**Test Conditions**:
```python
tolerance = 1e-4
if abs(loss_baseline - loss_uniform) < tolerance:
    FAIL: Patching not working
```

**Value for BH Routing**: ⭐⭐⭐
- Template for BH unit tests
- Validates patching mechanism
- Quick feedback loop (~1 minute runtime)

---

### 2.5 `logs_eda.ipynb` (Jupyter Notebook - 35 cells)

**Purpose**: Exploratory data analysis of routing logs with **KDE-based p-value estimation**

**Key Analyses**:

#### A. Data Loading & Structure
```python
JSON Structure:
{
  "config": str,
  "strategy": str,
  "num_experts": int,
  "dataset": str,
  "samples": [
    {
      "sample_id": int,
      "num_tokens": int,
      "loss": float,
      "layers": [
        {
          "layer": int,
          "router_logits_shape": [int, int],
          "selected_experts": list[list[int]],
          "expert_weights": list[list[float]],
          "router_logits_sample": list[list[float]]  # [tokens, 64]
        }
      ]
    }
  ]
}
```

#### B. Router Logit Distribution Analysis

**Aggregation**:
```python
layers_router_logits_raw[layer_idx] → shape: [num_tokens, num_experts] = [~5000, 64]
```

**Visualizations Created**:
1. Per-layer router softmax distributions
2. Per-expert, per-layer raw logit histograms
3. Trimmed distributions (removing top/bottom k experts)
4. Cross-dataset comparisons (LAMBADA vs. HellaSwag)

#### C. **KDE-Based P-Value Estimation** ⭐⭐⭐⭐⭐

**Critical for BH Routing!**

**Training KDE Models** (Cell `423b6395`):
```python
# For each layer:
train_data = layers_router_logits_raw[layer_idx].flatten()
kde = gaussian_kde(train_data.T)

x_grid = np.linspace(train_data_min, train_data_max, 10000)
pdf_grid = kde.evaluate(x_grid)
cdf_grid = np.cumsum(pdf_grid)
cdf_grid /= cdf_grid[-1]  # Normalize to [0, 1]

# Save model
pickle.dump({'x': x_grid, 'cdf': cdf_grid}, f)
```

**Computing P-Values on Test Data** (Cell `f1787d03`):
```python
test_layer_data = layers_router_logits_raw_hellswag[layer_idx].flatten()
probabilities = np.interp(test_layer_data, x_grid, cdf_grid)
p_values = 1 - probabilities  # P(X > x)
```

**Multiple Kernel Support** (Cell `e8075670`):
- Gaussian (default)
- Tophat
- Epanechnikov
- Linear
- Comparison plots for all kernels

**Per-Expert P-Value Analysis** (Cell `28d99046`):
```python
# For k-th best expert:
kth_best_expert_logits = np.sort(test_layer_data, axis=1)[:, -1 - k]
probabilities = np.interp(kth_best_expert_logits, x_grid, cdf_grid)
p_values = 1 - probabilities
```

**Empirical vs. KDE Comparison** (Cell `f88639ed`):
```python
# Actual empirical probabilities:
sorted_train_data = np.sort(train_data_raw)
ranks = np.searchsorted(sorted_train_data, test_layer_data, side="right")
empirical_probs = (ranks + 1) / n_train
empirical_pvalues = 1 - empirical_probs
```

#### D. Expert Utilization Analysis
- Expert choice counts per layer
- Most-chosen vs. least-chosen expert logit distributions
- Per-chosen-expert p-value histograms

#### E. Statistical Validation
- Uniform distribution overlays
- P-value histogram comparisons
- Right-tail analysis with thresholds

**Routing Information Extracted**:
- Raw router logits per layer, per token
- Softmax probabilities
- Expert selection patterns
- Per-expert logit distributions
- Statistical properties (mean, std, quantiles)

**How It Hooks**:
- Post-hoc analysis of logged data
- No real-time hooks (reads from saved JSON)

**Statistics Collected**:
- Expert weight sums per token
- Expert choice frequencies
- Router logit distributions (raw & softmax)
- Per-expert, per-layer statistics
- **KDE-based cumulative density functions**
- **P-values for expert logits**
- **Empirical distribution functions**

**Value for BH Routing**: ⭐⭐⭐⭐⭐
- **ALREADY IMPLEMENTS P-VALUE COMPUTATION!**
- KDE infrastructure ready for BH procedure
- Multiple kernel options for robustness
- Empirical validation built-in
- Per-expert p-value analysis framework
- Can be adapted for online BH routing

---

## 3. Routing Data Flow

### 3.1 Current Flow (Standard Top-K)

```
┌─────────────────────────────────────────────────────────────┐
│ Input: hidden_states [batch, seq_len, 2048]                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ OlmoeSparseMoeBlock.forward │
         └─────────────┬───────────────┘
                       │
                       ├─► Flatten to [B*L, 2048]
                       │
                       ▼
         ┌─────────────────────────────┐
         │ OlmoeTopKRouter.forward     │
         │ (layer.mlp.gate)            │
         └─────────────┬───────────────┘
                       │
                       ├─► Linear projection: [B*L, 64] (raw logits)
                       ├─► Softmax: [B*L, 64] (probabilities)
                       │
    ┌──────────────────┴──────────────────┐
    │ HOOK CAPTURE POINT                  │
    │ - router_logits [B*L, 64]           │
    │ - Stored in logged_routing_data     │
    └──────────────────┬──────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ torch.topk(logits, k=8)     │ ◄─── REPLACE FOR BH ROUTING
         └─────────────┬───────────────┘
                       │
                       ├─► top_k_weights [B*L, 8]
                       ├─► top_k_indices [B*L, 8]
                       │
                       ▼
         ┌─────────────────────────────┐
         │ OlmoeExperts.forward        │
         │ - Loop over active experts  │
         │ - Apply expert FFNs         │
         │ - Weight by top_k_weights   │
         │ - Accumulate results        │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ Output: [batch, seq_len, 2048] │
         └─────────────────────────────┘
```

### 3.2 Injection Points for BH Routing

**Option 1: Hook-Based (Offline Analysis)**
```
Router logits → Hook captures → Save to disk → Offline BH procedure → Analysis
```
- ✅ Already implemented
- ✅ No model modification required
- ❌ Not usable for inference (post-hoc only)

**Option 2: Model Patching (Online BH Routing)**
```
Router logits → BH procedure → Select experts → Continue forward pass
```
- ✅ Can affect actual inference
- ✅ Testable against baselines
- ✅ Infrastructure exists (ModelPatchingUtils)
- ⚠️ Need to implement BH expert selection function

---

## 4. Hook Infrastructure

### 4.1 Current Hook System

**Registration** (`olmoe_routing_experiments.py:536-581`):
```python
for i, layer in enumerate(model.model.layers):
    router_module = layer.mlp.gate
    hook_handle = router_module.register_forward_hook(
        lambda m, inp, out, idx=i: logging_hook_router(m, inp, out, idx, k=top_k)
    )
    self.router_hooks.append(hook_handle)
```

**Hook Function**:
```python
def logging_hook_router(module, input, output, layer_index, k=8):
    router_logits = output  # [num_tokens, num_experts]
    softmax_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
    topk_weights, topk_indices = torch.topk(softmax_weights, k, dim=-1)

    self.logged_routing_data.append({
        "layer": layer_index,
        "router_logits": router_logits.detach(),
        "expert_indices": topk_indices.detach(),
        "softmax_weights": topk_weights.detach(),
    })
```

**Cleanup**:
```python
def _clear_router_hooks(self):
    for hook in self.router_hooks:
        hook.remove()
    self.router_hooks = []
```

### 4.2 What Hooks Can Capture for BH Routing

✅ **Raw router logits** (pre-softmax)
✅ **Per-layer, per-token granularity**
✅ **Batch and sequence dimensions accessible**
✅ **Can compute p-values in hook function**
❌ **Cannot modify expert selection** (hooks are read-only observers)

### 4.3 Model Patching Alternative

**Current Patching** (`olmoe_routing_experiments.py:308-374`):
```python
def create_patched_forward(top_k, num_experts, strategy):
    def new_forward(self, hidden_states):
        # Get router logits
        router_logits = self.gate(hidden_states)

        # Custom expert selection
        routing_weights, selected_experts = ModelPatchingUtils.custom_select_experts(
            router_logits, top_k, num_experts, strategy
        )

        # Rest of forward pass with selected experts
        ...
        return final_hidden_states, router_logits
    return new_forward
```

**For BH Routing, Replace**:
```python
# CURRENT:
routing_weights, selected_experts = ModelPatchingUtils.custom_select_experts(
    router_logits, top_k, num_experts, strategy='uniform'
)

# BH ROUTING:
routing_weights, selected_experts = benjamini_hochberg_routing(
    router_logits,
    kde_models,  # Pre-trained KDE per layer
    fdr_level=0.05,
    min_experts=1,
    max_experts=top_k
)
```

---

## 5. Statistical Analysis Capabilities

### 5.1 KDE Infrastructure (from `logs_eda.ipynb`)

**Training Phase**:
```python
# Per-layer KDE training
train_data = layers_router_logits_raw[layer_idx].flatten()
kde = gaussian_kde(train_data.T)

# CDF computation
x_grid = np.linspace(data_min, data_max, 10000)
pdf_grid = kde.evaluate(x_grid)
cdf_grid = np.cumsum(pdf_grid)
cdf_grid /= cdf_grid[-1]

# Save for reuse
pickle.dump({'x': x_grid, 'cdf': cdf_grid}, 'kde_layer_{layer_idx}.pkl')
```

**Inference Phase**:
```python
# Load KDE model
with open('kde_layer_{layer_idx}.pkl', 'rb') as f:
    model_data = pickle.load(f)
    x_grid = model_data['x']
    cdf_grid = model_data['cdf']

# Compute p-values
test_logits = router_logits  # [num_tokens, num_experts]
probabilities = np.interp(test_logits, x_grid, cdf_grid)
p_values = 1 - probabilities
```

### 5.2 P-Value Computation Patterns

**Per-Expert P-Values**:
```python
# For each expert column in router_logits
for expert_idx in range(num_experts):
    expert_logits = router_logits[:, expert_idx]
    p_values[:, expert_idx] = compute_pvalue(expert_logits, kde_models[layer_idx])
```

**Per-Token BH Procedure** (not yet implemented):
```python
# For each token
for token_idx in range(num_tokens):
    token_logits = router_logits[token_idx, :]  # [64]
    p_values = compute_pvalues(token_logits, kde_model)

    # BH procedure
    sorted_pvalues, sorted_indices = torch.sort(p_values)
    ranks = torch.arange(1, num_experts + 1)
    thresholds = ranks / num_experts * fdr_level

    significant_mask = sorted_pvalues <= thresholds
    num_selected = significant_mask.sum().item()

    selected_experts[token_idx] = sorted_indices[:num_selected]
```

### 5.3 Available Kernels
- **Gaussian** (default, smooth, unbounded support)
- **Tophat** (uniform, bounded support)
- **Epanechnikov** (optimal for MSE, bounded)
- **Linear** (triangular, simple)

**Comparison Available**: Cell `e8075670` compares all kernels

---

## 6. Gaps for BH Routing

### 6.1 Missing Components

❌ **BH Routing Strategy Class**
```python
# Need to implement:
class BenjaminiHochbergRouting(RoutingStrategy):
    def __init__(self, num_experts, fdr_level=0.05, kde_models=None):
        super().__init__(num_experts)
        self.fdr_level = fdr_level
        self.kde_models = kde_models  # Per-layer KDE models

    def route(self, logits, layer_idx):
        # Compute p-values
        # Apply BH procedure
        # Return (indices, weights)
        pass
```

❌ **Online P-Value Computation**
- Current: Offline KDE fitting on saved logs
- Need: Real-time p-value lookup during forward pass
- Solution: Pre-train KDE models, load into memory, use interpolation

❌ **Variable-Length Expert Selection Handling**
- Current: All strategies return fixed [num_tokens, top_k]
- BH: Different tokens may select different numbers of experts
- Solution: Pad with zeros or modify `OlmoeExperts` to handle ragged tensors

❌ **FDR Level Configuration**
- Need to add `fdr_level` to `RoutingConfig`
- Support for fixed, adaptive, or learned FDR

❌ **BH-Specific Metrics**
- Actual FDR measurement
- Number of experts selected per token (distribution)
- False discovery tracking (if ground truth available)

### 6.2 Integration Challenges

⚠️ **Compatibility Issues**:

1. **Fixed-Size Output Assumption**
   ```python
   # Current code expects:
   router_indices: [num_tokens, top_k]
   router_weights: [num_tokens, top_k]

   # BH produces:
   router_indices: [num_tokens, variable_k]  # ❌ Incompatible
   ```

   **Solutions**:
   - Pad to max_k with dummy indices (expert_idx = num_experts)
   - Modify `OlmoeExperts.forward()` to handle variable-length lists
   - Use max_experts constraint in BH (simpler)

2. **KDE Model Loading**
   - Need to load 16 KDE models (one per layer) into memory
   - Models trained on calibration dataset (e.g., LAMBADA)
   - File size: ~100KB per layer → ~1.6MB total (manageable)

3. **Performance Overhead**
   - Interpolation: O(num_tokens * num_experts * log(grid_size))
   - BH procedure: O(num_tokens * num_experts * log(num_experts))
   - Should be fast enough for inference (needs benchmarking)

4. **Gradient Flow**
   - P-values computed via interpolation (not differentiable)
   - BH selection is discrete (not differentiable)
   - **Cannot train with BH routing, only evaluate**
   - For training: Would need Gumbel-Softmax or similar relaxation

---

## 7. Reusable Components

### 7.1 Directly Reusable (No Modification)

✅ **`RoutingExperimentRunner`**
- Add `BenjaminiHochbergRouting` to `strategy_factory`
- No other changes needed

✅ **Hook Infrastructure**
- Can capture p-values in hook for analysis
- Append p-values to `logged_routing_data`

✅ **Evaluation Metrics**
- Perplexity, accuracy, speed all work unchanged

✅ **`ResultAnalyzer`**
- Works with any strategy results
- No modifications needed

✅ **Dataset Loading**
- `load_dataset()` supports all needed benchmarks

### 7.2 Needs Adaptation

🔧 **`ModelPatchingUtils.custom_select_experts()`**
```python
# Current:
def custom_select_experts(router_logits, top_k, num_experts, strategy):
    probs = F.softmax(router_logits, dim=1)
    top_weights, selected_experts = torch.topk(probs, top_k, dim=-1)

    if strategy == 'uniform':
        routing_weights = torch.ones_like(selected_experts) / top_k
    ...

    return routing_weights, selected_experts

# Add BH strategy:
def custom_select_experts(router_logits, top_k, num_experts, strategy, **kwargs):
    ...
    elif strategy == 'benjamini_hochberg':
        kde_models = kwargs['kde_models']
        layer_idx = kwargs['layer_idx']
        fdr_level = kwargs.get('fdr_level', 0.05)

        routing_weights, selected_experts = benjamini_hochberg_select(
            router_logits, kde_models[layer_idx], fdr_level, top_k
        )
    ...
```

🔧 **`create_patched_forward()`**
- Need to pass layer index to `custom_select_experts()`
- Current: Layer index unknown in forward function
- Solution: Capture layer index in closure

### 7.3 New Utilities Needed

📦 **KDE Model Manager**
```python
class KDEModelManager:
    def __init__(self, model_dir='./kde_models/models'):
        self.models = {}  # {layer_idx: {'x': x_grid, 'cdf': cdf_grid}}

    def load_models(self, num_layers=16):
        for layer_idx in range(num_layers):
            path = f"{self.model_dir}/distribution_model_layer_{layer_idx}.pkl"
            with open(path, 'rb') as f:
                self.models[layer_idx] = pickle.load(f)

    def compute_pvalues(self, logits, layer_idx):
        """
        Args:
            logits: [num_tokens, num_experts] torch.Tensor
            layer_idx: int
        Returns:
            p_values: [num_tokens, num_experts] torch.Tensor
        """
        x_grid = self.models[layer_idx]['x']
        cdf_grid = self.models[layer_idx]['cdf']

        # Convert to numpy for interpolation
        logits_np = logits.cpu().numpy()
        probs = np.interp(logits_np, x_grid, cdf_grid)
        p_values = 1 - probs

        # Convert back to torch
        return torch.from_numpy(p_values).to(logits.device)
```

📦 **BH Expert Selector**
```python
def benjamini_hochberg_select(
    router_logits: torch.Tensor,  # [num_tokens, num_experts]
    kde_model: Dict,  # {'x': x_grid, 'cdf': cdf_grid}
    fdr_level: float = 0.05,
    max_experts: int = 8,
    min_experts: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Benjamini-Hochberg procedure for expert selection.

    Returns:
        routing_weights: [num_tokens, max_experts]
        selected_experts: [num_tokens, max_experts]
    """
    num_tokens, num_experts = router_logits.shape
    device = router_logits.device

    # Compute p-values
    p_values = compute_pvalues_from_kde(router_logits, kde_model)

    # Apply BH procedure per token
    selected_experts = torch.zeros(num_tokens, max_experts, dtype=torch.long, device=device)
    routing_weights = torch.zeros(num_tokens, max_experts, dtype=router_logits.dtype, device=device)

    for token_idx in range(num_tokens):
        # Sort p-values
        sorted_pvals, sorted_indices = torch.sort(p_values[token_idx])

        # BH thresholds
        ranks = torch.arange(1, num_experts + 1, device=device)
        thresholds = (ranks / num_experts) * fdr_level

        # Find cutoff
        significant = sorted_pvals <= thresholds
        num_selected = torch.sum(significant).item()

        # Enforce constraints
        num_selected = max(min_experts, min(num_selected, max_experts))

        # Select experts
        selected_idx = sorted_indices[:num_selected]
        selected_experts[token_idx, :num_selected] = selected_idx

        # Compute weights (could use various strategies)
        # Option 1: Original router softmax probabilities
        probs = F.softmax(router_logits[token_idx], dim=-1)
        weights = probs[selected_idx]
        weights = weights / weights.sum()  # Renormalize

        routing_weights[token_idx, :num_selected] = weights

    return routing_weights, selected_experts
```

---

## 8. Integration Points

### 8.1 Step-by-Step BH Integration Plan

#### Phase 1: Offline BH Analysis (Using Existing Hooks)

**Goal**: Validate BH procedure on logged routing data

**Steps**:
1. ✅ Use existing hook system to log router logits
2. ✅ Train KDE models on calibration dataset (LAMBADA)
   - Already done in `logs_eda.ipynb`
3. ✅ Compute p-values for test dataset (HellaSwag)
   - Already done in notebook
4. 📝 **NEW**: Implement offline BH selection
   ```python
   # In logs_eda.ipynb or new script:
   selected_experts_bh = apply_bh_offline(
       layers_router_logits_raw_hellswag,
       kde_models,
       fdr_level=0.05
   )
   ```
5. 📝 **NEW**: Simulate perplexity with BH selections
   - Require re-running forward pass with BH experts
   - Not truly offline, need patching

**Limitation**: Cannot measure actual perplexity/accuracy without modifying forward pass

#### Phase 2: Online BH Routing (Model Patching)

**Goal**: Actually use BH routing during inference

**Steps**:
1. 📝 Implement `KDEModelManager`
   ```python
   kde_manager = KDEModelManager(model_dir='./kde_models/models')
   kde_manager.load_models(num_layers=16)
   ```

2. 📝 Implement `benjamini_hochberg_select()` function
   - Test unit cases with synthetic data
   - Verify FDR control

3. 📝 Extend `custom_select_experts()` with BH strategy
   ```python
   elif strategy == 'benjamini_hochberg':
       return benjamini_hochberg_select(
           router_logits,
           kwargs['kde_model'],
           kwargs.get('fdr_level', 0.05),
           max_experts=top_k
       )
   ```

4. 📝 Modify `create_patched_forward()` to pass layer index
   ```python
   def create_patched_forward(top_k, num_experts, strategy, layer_idx=None, kde_models=None):
       def new_forward(self, hidden_states):
           router_logits = self.gate(hidden_states)

           routing_weights, selected_experts = custom_select_experts(
               router_logits, top_k, num_experts, strategy,
               layer_idx=layer_idx,  # NEW
               kde_models=kde_models  # NEW
           )
           ...
   ```

5. 📝 Update `patch_model()` to handle BH
   ```python
   def patch_model(model, top_k, strategy, kde_models=None):
       for idx, layer in enumerate(model.model.layers):
           layer.mlp.forward = create_patched_forward(
               top_k, num_experts, strategy,
               layer_idx=idx,  # NEW
               kde_models=kde_models  # NEW
           ).__get__(layer.mlp, layer.mlp.__class__)
   ```

6. 📝 Add BH to `RoutingExperimentRunner.strategy_factory`
   ```python
   self.strategy_factory = {
       ...
       'benjamini_hochberg': BenjaminiHochbergRouting,
   }
   ```

7. 📝 Update `RoutingConfig` to include BH parameters
   ```python
   @dataclass
   class RoutingConfig:
       num_experts: int
       strategy: str
       description: str
       fdr_level: float = 0.05  # NEW: For BH routing
       kde_model_dir: str = './kde_models/models'  # NEW
   ```

8. ✅ Run experiments
   ```python
   runner = RoutingExperimentRunner(
       model_name="allenai/OLMoE-1B-7B-0924",
       use_model_patching=True  # Required for BH
   )

   # Load KDE models
   runner.kde_manager = KDEModelManager()
   runner.kde_manager.load_models()

   # Run BH experiments
   runner.run_all_experiments(
       expert_counts=[8],  # BH can adapt, but set max
       strategies=['benjamini_hochberg', 'baseline', 'uniform'],
       datasets=['hellaswag', 'lambada', 'wikitext'],
       max_samples=500
   )
   ```

#### Phase 3: Advanced BH Features

**Adaptive FDR**:
```python
class AdaptiveBHRouting(RoutingStrategy):
    def determine_fdr_level(self, logits, layer_idx):
        # Adaptive FDR based on:
        # - Layer depth (early layers → higher FDR)
        # - Routing confidence (high max prob → lower FDR)
        # - Token importance (computed via attention weights)
        ...
```

**Multiple Hypothesis Testing Corrections**:
- Current: BH (controls FDR)
- Add: Bonferroni (controls FWER, more conservative)
- Add: Holm-Bonferroni (step-down, less conservative than Bonferroni)

**Expert Weight Strategies**:
- Current plan: Renormalized softmax probabilities
- Alternative 1: Uniform weights (1/num_selected)
- Alternative 2: Inverse p-value weighting (1/p_value, renormalized)
- Alternative 3: Learned weights (requires training)

### 8.2 Testing Strategy

**Unit Tests**:
```python
# tests/test_bh_routing.py

def test_bh_selection_basic():
    # Synthetic router logits
    router_logits = torch.randn(10, 64)  # 10 tokens, 64 experts

    # Mock KDE model
    kde_model = {'x': np.linspace(-5, 5, 1000), 'cdf': np.linspace(0, 1, 1000)}

    weights, indices = benjamini_hochberg_select(
        router_logits, kde_model, fdr_level=0.05, max_experts=8
    )

    assert weights.shape == (10, 8)
    assert indices.shape == (10, 8)
    assert torch.all(weights >= 0)
    assert torch.all(torch.sum(weights, dim=-1) <= 1.01)  # Renormalized to ~1

def test_bh_fdr_control():
    # Verify that FDR is actually controlled
    # Requires ground truth significance labels
    ...

def test_bh_vs_baseline_equivalence():
    # With very high FDR (e.g., 1.0), should select all experts
    # With very low FDR (e.g., 0.001), should select few experts
    ...
```

**Integration Tests**:
```python
def test_bh_patching():
    # Similar to test_routing_strategies_cpu.py
    model = OlmoeForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924", device_map='cpu')

    # Load KDE models
    kde_manager = KDEModelManager()
    kde_manager.load_models()

    # Patch with BH
    ModelPatchingUtils.patch_model(
        model, top_k=8, strategy='benjamini_hochberg', kde_models=kde_manager.models
    )

    # Run inference
    inputs = tokenizer("Test sentence", return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)

    # Verify output is different from baseline
    ...
```

**Benchmark Tests**:
```python
def benchmark_bh_speed():
    # Measure overhead of p-value computation + BH procedure
    # Compare to baseline top-k selection
    # Target: <10% slowdown
    ...
```

### 8.3 Evaluation Metrics for BH Routing

**Standard Metrics** (already tracked):
- Perplexity
- Token accuracy
- Inference speed (tokens/sec)

**BH-Specific Metrics** (need to add):
```python
@dataclass
class BHRoutingStats:
    avg_experts_selected: float  # Mean number of experts per token
    std_experts_selected: float  # Variance in expert count
    min_experts_selected: int
    max_experts_selected: int
    fdr_level_used: float

    # If ground truth available (synthetic experiments):
    empirical_fdr: float  # Actual false discovery rate
    empirical_tpr: float  # True positive rate (power)
```

**Comparison Baselines**:
1. **Regular top-k=8** (current default)
2. **Uniform top-k=8** (equal weights)
3. **Normalized top-k=8** (renormalized weights)
4. **Adaptive top-k** (variable k based on confidence)
5. **BH with FDR=0.01** (conservative)
6. **BH with FDR=0.05** (standard)
7. **BH with FDR=0.10** (permissive)
8. **BH with adaptive FDR** (layer/token-dependent)

---

## Summary

### ✅ What We Have
1. **Complete routing interception** (hooks + patching)
2. **Multiple routing strategies** (baseline, uniform, normalized, adaptive)
3. **Comprehensive logging** (per-layer, per-token routing data)
4. **Experiment orchestration** (RoutingExperimentRunner)
5. **KDE-based p-value computation** (already implemented in logs_eda.ipynb)
6. **Statistical analysis tools** (distribution fitting, empirical vs. KDE comparison)
7. **Visualization pipeline** (plots for all routing statistics)
8. **Evaluation metrics** (perplexity, accuracy, speed)

### ❌ What We Need
1. **BH expert selection function** (`benjamini_hochberg_select()`)
2. **KDE model manager** (load pre-trained models into memory)
3. **Online p-value computation** (integrate KDE lookup into forward pass)
4. **BH routing strategy class** (`BenjaminiHochbergRouting`)
5. **Patching modifications** (pass layer index, KDE models)
6. **BH-specific metrics** (empirical FDR, expert selection distribution)
7. **Unit tests** (validate BH procedure, FDR control)
8. **Integration tests** (end-to-end BH routing)

### 🔗 Key Integration Points
1. **ModelPatchingUtils.custom_select_experts()** - Add BH strategy
2. **RoutingExperimentRunner.strategy_factory** - Register BH routing
3. **create_patched_forward()** - Pass layer index and KDE models
4. **logs_eda.ipynb** - Adapt KDE training/loading for online use
5. **ExperimentResults** - Add BH-specific fields

### 📊 Recommended Next Steps
1. **Implement KDEModelManager** (reuse logs_eda.ipynb code)
2. **Implement benjamini_hochberg_select()** (use pseudocode from routing_mechanism_detailed.py)
3. **Write unit tests** (validate on synthetic data)
4. **Modify ModelPatchingUtils** (add BH support)
5. **Run pilot experiment** (BH vs. baseline on small dataset)
6. **Analyze results** (perplexity, FDR, expert utilization)
7. **Iterate on FDR levels** (find optimal FDR for quality/speed trade-off)
8. **Write paper** 📝

---

**End of Analysis**

This document provides a complete roadmap for BH routing integration. The infrastructure is 80% ready - we primarily need to connect existing KDE analysis to the online routing system.
