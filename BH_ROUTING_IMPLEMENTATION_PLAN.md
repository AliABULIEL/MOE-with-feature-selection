# BH Routing Comprehensive Evaluation Framework - Implementation Plan

**Generated:** 2025-12-13
**Status:** Gap Analysis and Implementation Roadmap

---

## Executive Summary

This document outlines the implementation plan for building a comprehensive BH routing evaluation framework aligned with the master prompt requirements. The framework will evaluate 20 BH routing configurations + 4 baselines across 3 datasets with 16 comprehensive metrics.

---

## Current State Analysis

### What We HAVE ✅

1. **Core BH Routing Implementation** (`bh_routing.py` - 813 lines)
   - ✅ BH algorithm with KDE-based p-values
   - ✅ TopK routing implementation
   - ✅ Vectorized PyTorch implementation
   - ✅ KDE models (144+ models for 16 layers)
   - ✅ Basic statistics computation

2. **Model Integration** (`olmoe_bh_integration.py` - 541 lines
   - ✅ OLMoERouterPatcher class
   - ✅ Direct Method Replacement patching
   - ✅ BH routing integration
   - ✅ TopK routing integration
   - ✅ Basic statistics collection

3. **BH Experiments Notebook** (`OLMoE_BH_Routing_Experiments.ipynb`)
   - ✅ 21 configurations (1 baseline + 20 BH variants)
   - ✅ GPU setup and model loading
   - ✅ OLMoERouterPatcher integration
   - ✅ Test prompts across 4 complexity levels
   - ✅ Basic metrics collection
   - ✅ Visualization (9-panel)
   - ✅ Report generation

4. **Template Reference** (`OLMoE_Full_Routing_Experiments.ipynb`)
   - ✅ Two-phase experimental structure
   - ✅ Dual file logging (summary + internal_routing)
   - ✅ Internal routing log format defined
   - ✅ Dataset evaluation infrastructure
   - ✅ Visualization patterns

5. **Framework File** (`olmoe_routing_experiments.py` - 1672 lines)
   - ✅ RoutingExperimentRunner class
   - ✅ Multiple routing strategies
   - ✅ Dataset loading utilities
   - ✅ Basic experiment orchestration

---

## What We NEED (Gaps) ❌

### CRITICAL GAPS

#### 1. **Internal Routing Data Collection** ❌
**Current:** Only aggregated statistics (avg_experts, std, etc.)
**Needed:** Full router_logits, selected_experts, p_values per sample/layer

**Impact:** Cannot analyze:
- Layer-wise behavior patterns
- Per-token routing decisions
- P-value distributions
- Expert selection consistency
- Routing evolution through layers

**Required Implementation:**
```python
class OLMoERouterPatcher:
    def __init__(self, model):
        # MISSING:
        self.internal_logs = []  # Per-sample routing data
        self.current_sample = None  # Current sample being logged

    def start_sample_logging(self, sample_id, text_preview):
        """NEW METHOD: Initialize logging for a sample"""
        pass

    def end_sample_logging(self, loss):
        """NEW METHOD: Finalize sample logging"""
        pass

    def get_internal_logs(self):
        """NEW METHOD: Return collected logs"""
        pass

    def get_aggregate_stats(self):
        """NEW METHOD: Compute aggregate statistics"""
        pass
```

#### 2. **Comprehensive Metrics (16 metrics across 8 categories)** ❌
**Current:** Only ~6 metrics implemented:
- avg_experts
- std_experts
- min/max_experts
- reduction_vs_baseline
- tokens_per_second

**MISSING Categories:**
1. ❌ QUALITY METRICS (2 metrics)
   - perplexity (for WikiText)
   - avg_task_accuracy (for LAMBADA, HellaSwag, ARC)

2. ❌ EFFICIENCY METRICS (1 metric missing)
   - ✅ expert_activation_ratio (have this)
   - ❌ flops_reduction_pct (formula-based computation)

3. ❌ SPEED METRICS (partially implemented)
   - ✅ tokens_per_second (have this)
   - ❌ latency_ms_per_token (need proper measurement)

4. ❌ ROUTING DISTRIBUTION METRICS (2 metrics)
   - ❌ expert_utilization (from usage_counts)
   - ❌ gini_coefficient (inequality measure)

5. ❌ ROUTING BEHAVIOR METRICS (2 metrics - BH specific)
   - ❌ adaptive_range (max - min expert counts)
   - ❌ selection_entropy (entropy of count distribution)

6. ✅ CONSTRAINT METRICS (have these)
   - ✅ ceiling_hit_rate
   - ✅ floor_hit_rate

7. ❌ CROSS-LAYER METRICS (2 metrics)
   - ❌ layer_expert_variance
   - ❌ layer_consistency_score

8. ❌ STABILITY METRICS (2 metrics)
   - ❌ expert_overlap_score
   - ❌ output_determinism

#### 3. **Dataset Evaluation Infrastructure** ❌
**Current:** Only test prompts (12 prompts)
**Needed:** Real benchmark datasets:
- WikiText-2 (200 samples, compute perplexity)
- LAMBADA (200 samples, compute accuracy)
- HellaSwag (200 samples, compute accuracy)
- Optional: ARC-Challenge (200 samples)

**Required Implementation:**
```python
def evaluate_perplexity(model, dataset, max_samples=200):
    """Compute perplexity on WikiText-2"""
    pass

def evaluate_task_accuracy(model, dataset_name, max_samples=200):
    """Compute accuracy on LAMBADA/HellaSwag/ARC"""
    pass
```

#### 4. **Dual File Logging System** ❌
**Current:** Single results JSON file
**Needed:** TWO files per configuration per dataset:
- `{config}_{dataset}.json` - Summary metrics
- `{config}_{dataset}_internal_routing.json` - Full routing logs

**Structure:**
- 20 configs × 3 datasets = 60 experiments
- 60 × 2 files = **120 log files total**

#### 5. **Experiment Configuration Mismatch** ⚠️
**Current (BH Notebook):**
- max_k: [4, 8, 16, 32, 64]
- alpha: [0.01, 0.05, 0.10, 0.20]
- Total: 20 BH configs

**Master Prompt Requires:**
- **4 BASELINE configs:** TopK with K=[8, 16, 32, 64]
- **16 BH configs:** max_k=[8, 16, 32, 64] × alpha=[0.30, 0.40, 0.50, 0.60]
- Total: **4 + 16 = 20 total configs**

**MISMATCH:** Alpha values different! Master wants [0.30, 0.40, 0.50, 0.60]

#### 6. **9-Panel Visualization Not Matching Template** ⚠️
**Current:** Has 9-panel visualization but different panels
**Needed (from master prompt):**
1. Perplexity comparison (baseline vs BH)
2. Task accuracy comparison
3. Expert efficiency (avg experts)
4. Alpha sensitivity heatmap
5. Pareto frontier (efficiency vs quality)
6. Routing behavior summary (floor/mid/ceiling)
7. Expert utilization
8. Layer-wise analysis
9. Speed-quality trade-off

---

## Implementation Roadmap

### Phase 1: Core Infrastructure (Priority: CRITICAL)

#### Task 1.1: Add Internal Routing Logging to OLMoERouterPatcher
**File:** `olmoe_bh_integration.py`
**Estimated Lines:** +200

**Implementation:**
```python
class OLMoERouterPatcher:
    def __init__(self, model):
        # ... existing code ...
        self.internal_logs = []
        self.current_sample = None
        self.layer_stats = defaultdict(list)

    def start_sample_logging(self, sample_id, text_preview):
        self.current_sample = {
            'sample_id': sample_id,
            'text_preview': text_preview[:100],
            'layers': []
        }

    def end_sample_logging(self, loss):
        self.current_sample['loss'] = loss
        self.current_sample['num_tokens'] = len(
            self.current_sample['layers'][0]['expert_counts']
        ) if self.current_sample['layers'] else 0
        self.internal_logs.append(self.current_sample)

    def _modify_bh_forward(self, ...):
        # In the forward pass, add:
        if hasattr(self, 'current_sample'):
            layer_data = {
                'layer': layer_idx,
                'router_logits_shape': list(router_logits.shape),
                'selected_experts': selected_experts.cpu().tolist(),
                'expert_weights': routing_weights[routing_weights > 0].cpu().tolist(),
                'expert_counts': expert_counts.cpu().tolist(),
                'router_logits_sample': router_logits[:5].cpu().tolist(),
                'p_values_sample': p_values[:5].cpu().tolist() if p_values else None,
                'layer_stats': {
                    'avg_experts': float(expert_counts.float().mean()),
                    'std_experts': float(expert_counts.float().std()),
                    'ceiling_hits': int((expert_counts == max_k).sum()),
                    'floor_hits': int((expert_counts == 1).sum())
                }
            }
            self.current_sample['layers'].append(layer_data)

    def get_internal_logs(self):
        return self.internal_logs

    def get_aggregate_stats(self):
        # Compute per-layer averages, expert usage counts, etc.
        pass

    def clear_internal_logs(self):
        self.internal_logs = []
```

**Testing:**
```python
patcher = OLMoERouterPatcher(model)
patcher.patch_with_bh(alpha=0.05, max_k=8)

for i, sample in enumerate(dataset[:10]):
    patcher.start_sample_logging(i, sample['text'][:100])
    outputs = model.generate(...)
    loss = compute_loss(...)
    patcher.end_sample_logging(loss)

internal_logs = patcher.get_internal_logs()
assert len(internal_logs) == 10
assert 'layers' in internal_logs[0]
assert len(internal_logs[0]['layers']) == 16  # All layers logged
```

#### Task 1.2: Implement Comprehensive Metrics
**File:** New file `bh_routing_metrics.py`
**Estimated Lines:** +500

**Structure:**
```python
# Category 1: Quality Metrics
def compute_perplexity(losses: List[float]) -> float:
    """Compute perplexity from losses"""
    return np.exp(np.mean(losses))

def compute_avg_task_accuracy(accuracies: Dict[str, float]) -> float:
    """Average accuracy across tasks"""
    return np.mean(list(accuracies.values()))

# Category 2: Efficiency Metrics
def compute_flops_reduction(avg_experts, baseline_k=8, num_experts=64):
    """Estimate FLOPs reduction"""
    expert_fraction = 0.6
    baseline_cost = (baseline_k / num_experts) * expert_fraction + (1 - expert_fraction)
    bh_cost = (avg_experts / num_experts) * expert_fraction + (1 - expert_fraction)
    return (baseline_cost - bh_cost) / baseline_cost * 100

# Category 4: Routing Distribution Metrics
def compute_expert_utilization(usage_counts: np.ndarray, num_experts=64):
    """Fraction of experts with any usage"""
    return np.sum(usage_counts > 0) / num_experts

def compute_gini_coefficient(usage_counts: np.ndarray):
    """Gini coefficient for expert usage inequality"""
    sorted_counts = np.sort(usage_counts)
    n = len(sorted_counts)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n

# Category 5: Routing Behavior Metrics
def compute_adaptive_range(expert_counts: np.ndarray):
    """Range of expert counts selected"""
    return int(np.max(expert_counts) - np.min(expert_counts))

def compute_selection_entropy(expert_counts: np.ndarray, max_k: int):
    """Entropy of expert count distribution"""
    hist, _ = np.histogram(expert_counts, bins=range(1, max_k + 2))
    probs = hist / np.sum(hist)
    probs = probs[probs > 0]  # Remove zeros
    entropy = -np.sum(probs * np.log(probs))
    normalized_entropy = entropy / np.log(max_k) if max_k > 1 else 0
    return entropy, normalized_entropy

# Category 7: Cross-Layer Metrics
def compute_layer_expert_variance(per_layer_avg: List[float]):
    """Variance in average experts across layers"""
    return float(np.var(per_layer_avg))

def compute_layer_consistency_score(internal_logs):
    """Average correlation between adjacent layers"""
    # Extract expert counts per layer per token
    # Compute Pearson correlation between adjacent layers
    # Return average correlation
    pass

# Category 8: Stability Metrics
def compute_expert_overlap_score(internal_logs):
    """Jaccard similarity of selected experts"""
    pass

def compute_output_determinism(model, prompts, num_runs=2):
    """Consistency of outputs across runs"""
    pass
```

#### Task 1.3: Dataset Evaluation Infrastructure
**File:** New file `bh_routing_evaluation.py`
**Estimated Lines:** +400

**Implementation:**
```python
from datasets import load_dataset
from torch.utils.data import DataLoader

def load_wikitext(max_samples=200):
    """Load WikiText-2 test set"""
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    return dataset.select(range(min(len(dataset), max_samples)))

def load_lambada(max_samples=200):
    """Load LAMBADA dataset"""
    dataset = load_dataset('lambada', split='validation')
    return dataset.select(range(min(len(dataset), max_samples)))

def load_hellaswag(max_samples=200):
    """Load HellaSwag dataset"""
    dataset = load_dataset('hellaswag', split='validation')
    return dataset.select(range(min(len(dataset), max_samples)))

def evaluate_perplexity(model, tokenizer, dataset, patcher=None):
    """
    Compute perplexity on WikiText-2

    Returns:
        perplexity: float
        losses: List[float] (per sample)
        internal_logs: List[Dict] (if patcher provided)
    """
    losses = []

    for i, sample in enumerate(tqdm(dataset)):
        if patcher:
            patcher.start_sample_logging(i, sample['text'][:100])

        inputs = tokenizer(sample['text'], return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss.item()
            losses.append(loss)

        if patcher:
            patcher.end_sample_logging(loss)

    perplexity = np.exp(np.mean(losses))

    return {
        'perplexity': perplexity,
        'avg_loss': np.mean(losses),
        'losses': losses,
        'internal_logs': patcher.get_internal_logs() if patcher else None
    }

def evaluate_lambada(model, tokenizer, dataset, patcher=None):
    """
    Compute accuracy on LAMBADA (last word prediction)

    Returns:
        accuracy: float
        predictions: List[Dict]
        internal_logs: List[Dict]
    """
    correct = 0
    total = 0
    predictions = []

    for i, sample in enumerate(tqdm(dataset)):
        text = sample['text']
        # Split into context and last word
        words = text.strip().split()
        context = ' '.join(words[:-1])
        target_word = words[-1]

        if patcher:
            patcher.start_sample_logging(i, context[:100])

        # Generate next word
        inputs = tokenizer(context, return_tensors='pt')
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False)
            predicted_word = tokenizer.decode(outputs[0][-1], skip_special_tokens=True)

        is_correct = predicted_word.strip().lower() == target_word.strip().lower()
        correct += int(is_correct)
        total += 1

        predictions.append({
            'context': context,
            'target': target_word,
            'predicted': predicted_word,
            'correct': is_correct
        })

        if patcher:
            patcher.end_sample_logging(0.0)  # No loss for generation

    accuracy = correct / total if total > 0 else 0.0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'predictions': predictions,
        'internal_logs': patcher.get_internal_logs() if patcher else None
    }

def evaluate_hellaswag(model, tokenizer, dataset, patcher=None):
    """
    Compute accuracy on HellaSwag (sentence completion)

    Returns:
        accuracy: float
        predictions: List[Dict]
        internal_logs: List[Dict]
    """
    # Similar to LAMBADA but with multiple choice
    pass
```

### Phase 2: Experiment Runner (Priority: HIGH)

#### Task 2.1: Create Comprehensive Experiment Runner
**File:** New file `bh_routing_experiment_runner.py`
**Estimated Lines:** +800

**Implementation:**
```python
class BHRoutingExperimentRunner:
    """
    Two-phase experiment runner for BH routing evaluation.
    Aligned with OLMoE_Full_Routing_Experiments template.
    """

    def __init__(self, model_name, device, output_dir):
        self.model_name = model_name
        self.device = device
        self.output_dir = Path(output_dir)
        self.logs_dir = self.output_dir / 'logs'
        self.viz_dir = self.output_dir / 'visualizations'

        # Create directories
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        self.model, self.tokenizer = self._load_model()
        self.patcher = OLMoERouterPatcher(self.model)

    def run_two_phase_experiment(
        self,
        baseline_k_values=[8, 16, 32, 64],
        bh_max_k_values=[8, 16, 32, 64],
        bh_alpha_values=[0.30, 0.40, 0.50, 0.60],
        datasets=['wikitext', 'lambada', 'hellaswag'],
        max_samples=200
    ):
        """
        Run full two-phase experiment.

        Phase 1: Evaluate all baseline TopK configurations
        Phase 2: Evaluate all BH configurations
        Phase 3: Generate comparisons and visualizations
        """
        all_results = []

        # PHASE 1: BASELINE ANALYSIS
        print("=" * 70)
        print("PHASE 1: BASELINE TOP-K ANALYSIS")
        print("=" * 70)

        for k in baseline_k_values:
            for dataset in datasets:
                config_name = f"{k}experts_topk_baseline"
                print(f"\n[{config_name}] Dataset: {dataset}")

                result = self._run_single_experiment(
                    config_name=config_name,
                    routing_type='topk',
                    k=k,
                    alpha=None,
                    dataset=dataset,
                    max_samples=max_samples
                )
                all_results.append(result)

        # PHASE 2: BH ROUTING ANALYSIS
        print("\n" + "=" * 70)
        print("PHASE 2: BH ROUTING ANALYSIS")
        print("=" * 70)

        for max_k in bh_max_k_values:
            for alpha in bh_alpha_values:
                for dataset in datasets:
                    config_name = f"{max_k}experts_bh_a{int(alpha*100):03d}"
                    print(f"\n[{config_name}] Dataset: {dataset}")

                    result = self._run_single_experiment(
                        config_name=config_name,
                        routing_type='bh',
                        k=max_k,
                        alpha=alpha,
                        dataset=dataset,
                        max_samples=max_samples
                    )
                    all_results.append(result)

        # PHASE 3: COMPARATIVE ANALYSIS
        print("\n" + "=" * 70)
        print("PHASE 3: COMPARATIVE ANALYSIS")
        print("=" * 70)

        results_df = pd.DataFrame(all_results)

        # Save results
        results_df.to_csv(self.output_dir / 'bh_routing_results.csv', index=False)
        results_df.to_json(self.output_dir / 'bh_routing_results.json', orient='records', indent=2)

        # Generate visualizations
        self._generate_visualizations(results_df)

        # Generate report
        self._generate_report(results_df)

        return results_df

    def _run_single_experiment(
        self,
        config_name,
        routing_type,
        k,
        alpha,
        dataset,
        max_samples
    ):
        """Run single configuration on single dataset."""

        # Setup routing
        self.patcher.clear_internal_logs()

        if routing_type == 'topk':
            if k == 8:
                self.patcher.unpatch()  # Native OLMoE
            else:
                self.patcher.patch_with_topk(k=k)
        else:  # bh
            self.patcher.patch_with_bh(alpha=alpha, max_k=k, min_k=1)

        # Load dataset
        eval_data = self._load_dataset(dataset, max_samples)

        # Run evaluation with internal logging
        metrics = self._evaluate_with_logging(eval_data, dataset)

        # Collect internal logs
        internal_logs = self.patcher.get_internal_logs()
        aggregate_stats = self.patcher.get_aggregate_stats()

        # Save summary JSON
        summary = {
            'config': config_name,
            'strategy': routing_type,
            'k_or_max_k': k,
            'alpha': alpha,
            'num_experts': 64,
            'dataset': dataset,
            'metrics': metrics,
            'summary': {
                'total_samples': len(internal_logs),
                'total_tokens': sum(s['num_tokens'] for s in internal_logs),
            }
        }

        summary_path = self.logs_dir / f"{config_name}_{dataset}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Save internal routing JSON
        internal_routing = {
            'config': config_name,
            'strategy': routing_type,
            'k_or_max_k': k,
            'alpha': alpha,
            'num_experts': 64,
            'dataset': dataset,
            'samples': internal_logs,
            'aggregate_stats': aggregate_stats
        }

        internal_path = self.logs_dir / f"{config_name}_{dataset}_internal_routing.json"
        with open(internal_path, 'w') as f:
            json.dump(internal_routing, f, indent=2)

        print(f"   ✅ Saved: {summary_path.name}")
        print(f"   ✅ Saved: {internal_path.name}")

        # Return result row
        result = {
            'config_name': config_name,
            'routing_type': routing_type,
            'k_or_max_k': k,
            'alpha': alpha,
            'dataset': dataset,
            **metrics
        }

        return result

    def _load_dataset(self, dataset_name, max_samples):
        """Load evaluation dataset"""
        if dataset_name == 'wikitext':
            return load_wikitext(max_samples)
        elif dataset_name == 'lambada':
            return load_lambada(max_samples)
        elif dataset_name == 'hellaswag':
            return load_hellaswag(max_samples)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def _evaluate_with_logging(self, dataset, dataset_name):
        """Evaluate model on dataset with internal logging"""
        if dataset_name == 'wikitext':
            result = evaluate_perplexity(self.model, self.tokenizer, dataset, self.patcher)
            return {
                'perplexity': result['perplexity'],
                'avg_loss': result['avg_loss']
            }
        elif dataset_name == 'lambada':
            result = evaluate_lambada(self.model, self.tokenizer, dataset, self.patcher)
            return {
                'lambada_accuracy': result['accuracy']
            }
        elif dataset_name == 'hellaswag':
            result = evaluate_hellaswag(self.model, self.tokenizer, dataset, self.patcher)
            return {
                'hellaswag_accuracy': result['accuracy']
            }

    def _generate_visualizations(self, results_df):
        """Generate 9-panel comparison figure"""
        # Implementation of 9-panel visualization
        pass

    def _generate_report(self, results_df):
        """Generate markdown report"""
        # Implementation of markdown report
        pass
```

### Phase 3: Visualization & Reporting (Priority: MEDIUM)

#### Task 3.1: Implement 9-Panel Visualization
**File:** New file `bh_routing_visualizations.py`
**Estimated Lines:** +600

#### Task 3.2: Implement Markdown Report Generator
**File:** In `bh_routing_experiment_runner.py`
**Estimated Lines:** +300

---

## Testing Strategy

### Unit Tests
1. Test internal logging mechanism
2. Test each metric computation
3. Test dataset loading
4. Test dual file saving

### Integration Tests
1. Run single configuration end-to-end
2. Verify 2 files generated (summary + internal)
3. Verify all 16 metrics present
4. Verify internal logs have correct structure

### Full System Test
1. Run 2 baseline configs × 1 dataset = 2 experiments (4 files)
2. Run 2 BH configs × 1 dataset = 2 experiments (4 files)
3. Verify visualizations generated
4. Verify report generated

---

## Timeline Estimate

| Phase | Tasks | Estimated Time | Priority |
|-------|-------|----------------|----------|
| Phase 1.1 | Internal Logging | 4 hours | CRITICAL |
| Phase 1.2 | Metrics Implementation | 6 hours | CRITICAL |
| Phase 1.3 | Dataset Evaluation | 4 hours | CRITICAL |
| Phase 2.1 | Experiment Runner | 6 hours | HIGH |
| Phase 3.1 | Visualization | 4 hours | MEDIUM |
| Phase 3.2 | Report Generator | 3 hours | MEDIUM |
| Testing | All phases | 3 hours | HIGH |
| **TOTAL** | | **30 hours** | |

---

## Next Steps

1. **Immediate:** Implement internal logging mechanism (Phase 1.1)
2. **Then:** Implement metrics (Phase 1.2)
3. **Then:** Dataset evaluation (Phase 1.3)
4. **Then:** Experiment runner (Phase 2.1)
5. **Finally:** Visualization and reporting (Phase 3)

---

## Success Criteria

✅ Implementation complete when:
1. All 60 experiments run successfully (20 configs × 3 datasets)
2. 120 log files generated (60 summary + 60 internal_routing)
3. All 16 metrics computed for each configuration
4. Internal routing logs contain full router_logits data
5. 9-panel visualization generated
6. Markdown report with actionable recommendations
7. File naming matches template convention

---

**END OF IMPLEMENTATION PLAN**
