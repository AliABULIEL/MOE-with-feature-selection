"""
Metrics Computation for Higher Criticism Routing
=================================================

Computes quality and efficiency metrics for HC routing including:
- Standard routing metrics (expert counts, entropy)
- HC-specific metrics (threshold stability, signal strength)
- Comparison metrics (vs baseline, vs BH)
- Unified evaluation metrics across all datasets
- Metrics aggregation and comparison utilities
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict


@dataclass
class HCMetrics:
    """
    Container for HC routing metrics.

    Includes literature-accurate parameters from Donoho & Jin (2004).
    """
    # Expert selection stats
    avg_experts: float = 0.0
    std_experts: float = 0.0
    min_experts: int = 0
    max_experts: int = 0
    adaptive_range: int = 0

    # Constraint hit rates
    floor_hit_rate: float = 0.0
    ceiling_hit_rate: float = 0.0
    mid_range_rate: float = 0.0

    # HC-specific (Donoho & Jin 2004 parameters)
    beta: float = 0.5                    # Search fraction β ∈ (0, 1]
    hc_variant: str = 'plus'             # 'standard', 'plus', or 'star'
    avg_hc_threshold: float = 0.0
    std_hc_threshold: float = 0.0
    avg_hc_max_value: float = 0.0
    hc_signal_rate: float = 0.0  # % of tokens with positive HC signal

    # Efficiency
    expert_activation_ratio: float = 0.0
    selection_entropy: float = 0.0
    normalized_entropy: float = 0.0

    # Comparison (filled externally)
    reduction_vs_baseline: float = 0.0
    improvement_vs_bh: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'avg_experts': self.avg_experts,
            'std_experts': self.std_experts,
            'min_experts': self.min_experts,
            'max_experts': self.max_experts,
            'adaptive_range': self.adaptive_range,
            'floor_hit_rate': self.floor_hit_rate,
            'ceiling_hit_rate': self.ceiling_hit_rate,
            'mid_range_rate': self.mid_range_rate,
            'beta': self.beta,
            'hc_variant': self.hc_variant,
            'avg_hc_threshold': self.avg_hc_threshold,
            'std_hc_threshold': self.std_hc_threshold,
            'avg_hc_max_value': self.avg_hc_max_value,
            'hc_signal_rate': self.hc_signal_rate,
            'expert_activation_ratio': self.expert_activation_ratio,
            'selection_entropy': self.selection_entropy,
            'normalized_entropy': self.normalized_entropy,
            'reduction_vs_baseline': self.reduction_vs_baseline,
            'improvement_vs_bh': self.improvement_vs_bh,
        }


class HCMetricsComputer:
    """
    Compute comprehensive metrics for HC routing.

    Example:
        computer = HCMetricsComputer(baseline_k=8)
        metrics = computer.compute(expert_counts, routing_weights, hc_stats)
    """

    def __init__(
        self,
        baseline_k: int = 8,
        num_experts: int = 64
    ):
        """
        Initialize metrics computer.

        Args:
            baseline_k: Baseline TopK for comparison (default: 8)
            num_experts: Total number of experts (default: 64)
        """
        self.baseline_k = baseline_k
        self.num_experts = num_experts

    def compute(
        self,
        expert_counts: torch.Tensor,
        routing_weights: Optional[torch.Tensor] = None,
        hc_stats: Optional[Dict[str, torch.Tensor]] = None,
        min_k: int = 1,
        max_k: int = 16,
        beta: float = 0.5,
        hc_variant: str = 'plus'
    ) -> HCMetrics:
        """
        Compute all HC routing metrics.

        Args:
            expert_counts: Number of experts per token [num_tokens]
            routing_weights: Routing weights [num_tokens, num_experts]
            hc_stats: Optional dict with 'threshold_ranks', 'max_hc_values'
            min_k: Minimum experts configuration
            max_k: Maximum experts configuration
            beta: Search fraction parameter β ∈ (0, 1]
            hc_variant: HC variant used - 'standard', 'plus', or 'star'

        Returns:
            HCMetrics dataclass with all computed metrics
        """
        metrics = HCMetrics(beta=beta, hc_variant=hc_variant)

        counts = expert_counts.float().cpu()
        num_tokens = len(counts)

        # Basic statistics
        metrics.avg_experts = counts.mean().item()
        metrics.std_experts = counts.std().item() if num_tokens > 1 else 0.0
        metrics.min_experts = int(counts.min().item())
        metrics.max_experts = int(counts.max().item())
        metrics.adaptive_range = metrics.max_experts - metrics.min_experts

        # Constraint hit rates
        metrics.floor_hit_rate = (counts == min_k).sum().item() / num_tokens * 100
        metrics.ceiling_hit_rate = (counts == max_k).sum().item() / num_tokens * 100
        metrics.mid_range_rate = 100 - metrics.floor_hit_rate - metrics.ceiling_hit_rate

        # Reduction vs baseline
        metrics.reduction_vs_baseline = (1 - metrics.avg_experts / self.baseline_k) * 100

        # HC-specific metrics
        if hc_stats is not None:
            if 'threshold_ranks' in hc_stats:
                ranks = hc_stats['threshold_ranks'].float().cpu()
                metrics.avg_hc_threshold = ranks.mean().item()
                metrics.std_hc_threshold = ranks.std().item() if len(ranks) > 1 else 0.0

            if 'max_hc_values' in hc_stats:
                hc_max = hc_stats['max_hc_values'].float().cpu()
                metrics.avg_hc_max_value = hc_max.mean().item()
                metrics.hc_signal_rate = (hc_max > 0).sum().item() / num_tokens * 100

        # Weight-based metrics
        if routing_weights is not None:
            weights = routing_weights.float().cpu()

            # Selection entropy
            eps = 1e-10
            nonzero_weights = weights[weights > eps]
            if len(nonzero_weights) > 0:
                # Per-token entropy
                token_entropy = -(weights * torch.log(weights + eps)).sum(dim=-1)
                metrics.selection_entropy = token_entropy.mean().item()

                # Normalized entropy (0 to 1)
                max_entropy = np.log(metrics.avg_experts) if metrics.avg_experts > 1 else 1.0
                metrics.normalized_entropy = metrics.selection_entropy / max_entropy

            # Expert activation ratio
            ever_selected = (weights > 0.01).any(dim=0)
            metrics.expert_activation_ratio = ever_selected.sum().item() / self.num_experts

        return metrics

    def compute_comparison(
        self,
        hc_counts: torch.Tensor,
        bh_counts: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compare HC vs BH routing on same inputs.

        Args:
            hc_counts: Expert counts from HC [num_tokens]
            bh_counts: Expert counts from BH [num_tokens]

        Returns:
            Comparison statistics
        """
        hc = hc_counts.float()
        bh = bh_counts.float()

        return {
            'hc_avg': hc.mean().item(),
            'bh_avg': bh.mean().item(),
            'hc_minus_bh_avg': (hc - bh).mean().item(),
            'hc_wins_pct': (hc > bh).float().mean().item() * 100,
            'bh_wins_pct': (bh > hc).float().mean().item() * 100,
            'ties_pct': (hc == bh).float().mean().item() * 100,
        }


# =============================================================================
# UNIFIED EVALUATION METRICS (Cross-Dataset)
# =============================================================================

@dataclass
class UnifiedEvaluationMetrics:
    """
    Unified metrics dataclass for comprehensive HC routing evaluation.

    Captures ALL metrics across datasets with consistent schema:
    - Dataset identification
    - Task-specific metrics (perplexity, accuracy)
    - Routing metrics (expert selection behavior)
    - TopK agreement (comparison with TopK-8)
    - Diagnostic metrics (loss distribution, timing)

    This is the PRIMARY metrics class for cross-dataset comparisons.
    """

    # === DATASET IDENTIFICATION ===
    dataset_name: str  # 'wikitext', 'lambada', 'hellaswag'
    routing_type: str  # 'topk' or 'hc'
    config_name: str  # e.g., 'topk_8', 'hc_beta0.5_maxk16'
    beta: Optional[float] = None  # HC beta parameter (None for TopK)
    min_k: int = 1
    max_k: int = 64

    # === TASK METRICS ===
    # WikiText
    perplexity: Optional[float] = None  # Token-weighted (primary)
    perplexity_token_weighted: Optional[float] = None  # Explicit
    perplexity_sample_weighted: Optional[float] = None  # For comparison
    avg_loss: Optional[float] = None
    avg_loss_token_weighted: Optional[float] = None
    avg_loss_sample_weighted: Optional[float] = None

    # LAMBADA & HellaSwag
    accuracy: Optional[float] = None  # Raw accuracy
    accuracy_raw: Optional[float] = None  # Explicit (may favor shorter endings)
    accuracy_normalized: Optional[float] = None  # Length-normalized (HellaSwag only)

    # === ROUTING METRICS (all datasets) ===
    avg_experts: float = 0.0
    std_experts: float = 0.0
    min_experts_observed: int = 0
    max_experts_observed: int = 0

    # Expert selection distribution
    floor_hit_rate: float = 0.0  # % hitting min_k
    ceiling_hit_rate: float = 0.0  # % hitting max_k
    mid_range_rate: float = 0.0  # % in adaptive range

    # Weight analysis (CRITICAL for diagnosis)
    avg_weight_sum: float = 0.0
    std_weight_sum: float = 0.0
    min_weight_sum: float = 0.0
    max_weight_sum: float = 0.0

    # Selection diversity
    selection_entropy: float = 0.0
    expert_utilization: float = 0.0  # Fraction of 64 experts ever used

    # === TOPK AGREEMENT METRICS ===
    avg_topk_jaccard: float = 0.0  # Average Jaccard similarity with TopK-8
    avg_topk_intersection: float = 0.0  # Average experts in common
    topk_agreement_by_layer: Dict[int, float] = field(default_factory=dict)  # Per-layer Jaccard

    # === DIAGNOSTIC METRICS ===
    loss_std: Optional[float] = None  # Std dev of losses
    loss_p95: Optional[float] = None  # 95th percentile loss
    loss_p99: Optional[float] = None  # 99th percentile loss

    num_samples: int = 0  # Number of samples evaluated
    total_tokens: int = 0  # Total tokens processed
    evaluation_time_seconds: float = 0.0  # Wall clock time

    # === METADATA ===
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with all fields."""
        return asdict(self)

    def to_dataframe_row(self) -> pd.DataFrame:
        """Convert to single-row DataFrame for easy concatenation."""
        return pd.DataFrame([self.to_dict()])

    def save_json(self, filepath: str):
        """Save metrics to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        print(f"✅ Saved metrics to: {filepath}")

    @classmethod
    def load_json(cls, filepath: str) -> 'UnifiedEvaluationMetrics':
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


def save_metrics(
    metrics: UnifiedEvaluationMetrics,
    output_dir: str = './metrics',
    experiment_name: str = 'experiment',
    append_to_cumulative: bool = True
):
    """
    Save metrics in multiple formats with summary output.

    Args:
        metrics: UnifiedEvaluationMetrics instance
        output_dir: Directory to save metrics
        experiment_name: Name for this experiment
        append_to_cumulative: Whether to append to cumulative CSV
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save JSON with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = output_dir / f'{experiment_name}_metrics_{timestamp}.json'
    metrics.save_json(json_path)

    # 2. Append to cumulative CSV
    if append_to_cumulative:
        cumulative_csv = output_dir / 'all_experiments_metrics.csv'
        df = metrics.to_dataframe_row()

        if cumulative_csv.exists():
            existing_df = pd.read_csv(cumulative_csv)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_csv(cumulative_csv, index=False)
        else:
            df.to_csv(cumulative_csv, index=False)

        print(f"✅ Appended to cumulative CSV: {cumulative_csv}")

    # 3. Print summary table
    print("\n" + "="*70)
    print(f"METRICS SUMMARY: {experiment_name}")
    print("="*70)

    summary_data = {}

    if metrics.perplexity is not None:
        summary_data['Perplexity'] = f"{metrics.perplexity:.2f}"
        if metrics.perplexity_sample_weighted:
            summary_data['PPL (sample-weighted)'] = f"{metrics.perplexity_sample_weighted:.2f}"

    if metrics.accuracy is not None:
        summary_data['Accuracy'] = f"{metrics.accuracy:.4f}"
    if metrics.accuracy_normalized is not None:
        summary_data['Accuracy (normalized)'] = f"{metrics.accuracy_normalized:.4f}"

    summary_data['Avg Experts'] = f"{metrics.avg_experts:.2f}"
    summary_data['Avg Weight Sum'] = f"{metrics.avg_weight_sum:.4f}"

    if metrics.avg_topk_jaccard > 0:
        summary_data['TopK Agreement'] = f"{metrics.avg_topk_jaccard:.4f}"

    summary_data['Floor Hit Rate'] = f"{metrics.floor_hit_rate:.1f}%"
    summary_data['Ceiling Hit Rate'] = f"{metrics.ceiling_hit_rate:.1f}%"

    max_key_len = max(len(k) for k in summary_data.keys())
    for key, value in summary_data.items():
        print(f"  {key:<{max_key_len}} : {value}")

    print("="*70 + "\n")
