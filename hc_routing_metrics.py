"""
Metrics Computation for Higher Criticism Routing
=================================================

Computes quality and efficiency metrics for HC routing including:
- Standard routing metrics (expert counts, entropy)
- HC-specific metrics (threshold stability, signal strength)
- Comparison metrics (vs baseline, vs BH)
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


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
