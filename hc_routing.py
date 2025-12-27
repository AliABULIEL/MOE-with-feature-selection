"""
Higher Criticism Routing for Mixture-of-Experts Models
======================================================

Implements Higher Criticism (HC) statistical method for adaptive expert
selection in sparse MoE models. HC automatically finds the signal-noise
boundary by detecting where p-values deviate most from uniform distribution.

Key Advantages over Benjamini-Hochberg:
- Adaptive: finds natural threshold instead of fixed formula
- No alpha parameter to tune
- Selects more experts when signal is strong

References:
    Donoho, D., & Jin, J. (2004). Higher criticism for detecting sparse
    heterogeneous mixtures. The Annals of Statistics, 32(3), 962-994.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, TYPE_CHECKING
import warnings

# REUSE from existing BH implementation
from bh_routing import (
    load_kde_models,
    compute_pvalues_kde,
    compute_pvalues_empirical
)

if TYPE_CHECKING:
    from hc_routing_logging import HCRoutingLogger


# =============================================================================
# CORE HC FUNCTIONS
# =============================================================================

def compute_hc_statistic(
    p_values_sorted: torch.Tensor,
    n: int,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Compute Higher Criticism statistic for each rank.

    HC(i) = sqrt(N) * (i/N - p_(i)) / sqrt(p_(i) * (1 - p_(i)))

    Args:
        p_values_sorted: Sorted p-values [num_tokens, num_experts], ascending order
        n: Total number of experts (64 for OLMoE)
        eps: Small constant for numerical stability

    Returns:
        hc_stats: HC statistic at each rank [num_tokens, num_experts]

    Note:
        - Positive HC means p-value is BELOW expected (signal)
        - Negative HC means p-value is ABOVE expected (noise)
        - We look for maximum POSITIVE HC to find signal-noise boundary
    """
    device = p_values_sorted.device
    dtype = p_values_sorted.dtype
    num_tokens = p_values_sorted.shape[0]

    # Expected p-values under uniform: i/N for i = 1, 2, ..., N
    ranks = torch.arange(1, n + 1, device=device, dtype=dtype)  # [1, 2, ..., 64]
    expected = ranks / n  # [1/64, 2/64, ..., 64/64]

    # Expand for batch computation
    expected = expected.unsqueeze(0).expand(num_tokens, -1)  # [num_tokens, 64]

    # Clamp p-values for numerical stability
    p_clamped = torch.clamp(p_values_sorted, min=eps, max=1.0 - eps)

    # Compute HC statistic
    # Numerator: deviation from expected
    numerator = expected - p_clamped  # positive when p < expected (signal)

    # Denominator: standard error under uniform
    denominator = torch.sqrt(p_clamped * (1.0 - p_clamped) + eps)

    # HC statistic
    sqrt_n = np.sqrt(n)
    hc_stats = sqrt_n * numerator / denominator

    return hc_stats


def find_hc_threshold(
    hc_stats: torch.Tensor,
    p_values_sorted: torch.Tensor,
    n: int,
    min_k: int = 1,
    max_k: int = 16,
    beta: float = 0.5,
    hc_variant: str = 'plus'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find the HC threshold - the rank with maximum HC statistic.

    Implements three variants from Donoho & Jin (2004):
    - 'standard': HC_n = max over all ranks
    - 'plus': HC⁺_n = max where p < expected (RECOMMENDED)
    - 'star': HC*_{n,β} = max over first β fraction of ranks

    Args:
        hc_stats: HC statistics [num_tokens, num_experts]
        p_values_sorted: Sorted p-values [num_tokens, num_experts]
        n: Number of experts
        min_k: Minimum experts to select (safety floor)
        max_k: Maximum experts to select (ceiling)
        beta: Search fraction β ∈ (0, 1] for 'star' variant
        hc_variant: Which HC formula - 'standard', 'plus', or 'star'

    Returns:
        num_selected: Number of experts per token [num_tokens]
        threshold_ranks: Which rank had max HC [num_tokens]
        max_hc_values: Maximum HC value per token [num_tokens]

    References:
        Donoho, D., & Jin, J. (2004). Higher criticism for detecting sparse
        heterogeneous mixtures. The Annals of Statistics, 32(3), 962-994.
    """
    device = hc_stats.device
    num_tokens = hc_stats.shape[0]

    # Create mask based on HC variant (from Donoho & Jin 2004)
    if hc_variant == 'plus':
        # HC⁺: Only consider ranks where p < expected (i.e., positive HC)
        # This focuses on the "signal" region - most robust variant
        ranks = torch.arange(1, n + 1, device=device).float()
        expected = ranks / n
        expected = expected.unsqueeze(0).expand(num_tokens, -1)
        valid_mask = p_values_sorted < expected  # [num_tokens, num_experts]

    elif hc_variant == 'star':
        # HC*_{n,β}: Only search first β fraction of ranks
        # β controls how conservative the search is
        max_search = int(n * beta)
        valid_mask = torch.zeros_like(hc_stats, dtype=torch.bool)
        valid_mask[:, :max_search] = True

    else:  # 'standard'
        # HC_n: Standard HC - consider all ranks
        valid_mask = torch.ones_like(hc_stats, dtype=torch.bool)

    # Apply min_k and max_k constraints to valid mask
    # Don't search below min_k or above max_k
    valid_mask[:, :min_k-1] = False  # Must select at least min_k
    valid_mask[:, max_k:] = False     # Can't select more than max_k

    # Set invalid positions to -inf so they're never selected
    hc_masked = hc_stats.clone()
    hc_masked[~valid_mask] = float('-inf')

    # Find maximum HC and its position
    max_hc_values, threshold_ranks = torch.max(hc_masked, dim=1)

    # threshold_ranks is 0-indexed, so num_selected = threshold_ranks + 1
    num_selected = threshold_ranks + 1

    # Handle edge case: if all positions were invalid (no signal found)
    # Fall back to min_k selection
    no_valid = (max_hc_values == float('-inf'))
    num_selected[no_valid] = min_k
    threshold_ranks[no_valid] = min_k - 1
    max_hc_values[no_valid] = 0.0

    # Enforce constraints
    num_selected = torch.clamp(num_selected, min=min_k, max=max_k)

    return num_selected, threshold_ranks, max_hc_values


def higher_criticism_routing(
    router_logits: torch.Tensor,
    temperature: float = 1.0,
    min_k: int = 1,
    max_k: int = 16,
    layer_idx: int = 0,
    kde_models: Optional[Dict[int, Dict]] = None,
    beta: float = 0.5,
    hc_variant: str = 'plus',
    return_stats: bool = False,
    logger: Optional['HCRoutingLogger'] = None,
    log_every_n_tokens: int = 100,
    sample_idx: int = 0,
    token_idx: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict]]:
    """
    Higher Criticism routing for adaptive expert selection in MoE models.

    HC automatically finds where p-values deviate most from uniform distribution,
    identifying the natural boundary between "signal" (relevant experts) and
    "noise" (irrelevant experts).

    Implements the HC method from Donoho & Jin (2004) with three variants:
    - HC (standard): max over all ranks
    - HC⁺ (plus): max where p < expected [RECOMMENDED]
    - HC* (star): max over first β fraction of ranks

    Args:
        router_logits: Router output logits
            Shape: [batch, seq_len, num_experts] or [num_tokens, num_experts]
        temperature: Temperature for softmax weight computation (default: 1.0)
        min_k: Minimum experts to select per token (safety floor)
        max_k: Maximum experts to select per token (ceiling)
        layer_idx: Which transformer layer (0-15) for KDE model lookup
        kde_models: Pre-loaded KDE models. If None, loads automatically.
        beta: Search fraction β ∈ (0, 1] for 'star' variant.
              Controls how much of the ranking to search (e.g., β=0.5 → first half)
        hc_variant: HC variant to use:
            - 'plus': HC⁺ - Only where p < expected (RECOMMENDED, most robust)
            - 'standard': HC - All ranks considered
            - 'star': HC* - Only first β fraction of ranks
        return_stats: If True, return additional statistics dict
        logger: Optional HCRoutingLogger for detailed logging
        log_every_n_tokens: How often to log (every N tokens)
        sample_idx: Sample index for logging
        token_idx: Starting token index for logging

    Returns:
        routing_weights: Normalized weights for selected experts
            Shape: [batch, seq_len, num_experts]
        selected_experts: Indices of selected experts (padded with -1)
            Shape: [batch, seq_len, max_k]
        expert_counts: Number of experts selected per token
            Shape: [batch, seq_len]
        stats (optional): Dict with 'p_values', 'hc_stats', 'threshold_ranks', etc.

    Example:
        >>> logits = torch.randn(1, 10, 64)  # batch=1, seq=10, experts=64
        >>> weights, experts, counts = higher_criticism_routing(
        ...     logits, min_k=4, max_k=12, beta=0.5, hc_variant='plus'
        ... )[:3]
        >>> print(counts)  # Adaptive: might be [6, 8, 5, 7, ...] not fixed 8

    References:
        Donoho, D., & Jin, J. (2004). Higher criticism for detecting sparse
        heterogeneous mixtures. The Annals of Statistics, 32(3), 962-994.
    """
    # =========================================================================
    # STEP 1: Input validation and reshaping
    # =========================================================================
    original_shape = router_logits.shape
    device = router_logits.device

    # Handle different input shapes
    if router_logits.dim() == 2:
        # [num_tokens, num_experts]
        router_logits = router_logits.unsqueeze(0)  # [1, num_tokens, num_experts]
        was_2d = True
    else:
        was_2d = False

    batch_size, seq_len, num_experts = router_logits.shape

    # Flatten to [num_tokens, num_experts] for batch processing
    logits_flat = router_logits.view(-1, num_experts)  # [B*S, E]
    num_tokens = logits_flat.shape[0]

    # =========================================================================
    # STEP 2: Load KDE models if not provided
    # =========================================================================
    if kde_models is None:
        kde_models = load_kde_models()
        if not kde_models:
            warnings.warn("KDE models not found. Using empirical p-values.")

    # =========================================================================
    # STEP 3: Compute p-values using KDE (reuse from BH)
    # =========================================================================
    if kde_models and layer_idx in kde_models:
        p_values = compute_pvalues_kde(logits_flat, layer_idx, kde_models)
    else:
        p_values = compute_pvalues_empirical(logits_flat)

    # =========================================================================
    # STEP 4: Sort p-values ascending (smallest first = most significant)
    # =========================================================================
    p_sorted, sort_indices = torch.sort(p_values, dim=1)  # [num_tokens, 64]

    # =========================================================================
    # STEP 5: Compute HC statistics
    # =========================================================================
    hc_stats = compute_hc_statistic(p_sorted, num_experts)  # [num_tokens, 64]

    # =========================================================================
    # STEP 6: Find HC threshold (where to cut off selection)
    # =========================================================================
    num_selected, threshold_ranks, max_hc_values = find_hc_threshold(
        hc_stats, p_sorted, num_experts,
        min_k=min_k, max_k=max_k, beta=beta, hc_variant=hc_variant
    )

    # =========================================================================
    # STEP 7: Create selection mask
    # =========================================================================
    # For each token, select experts with rank <= num_selected
    ranks = torch.arange(num_experts, device=device).unsqueeze(0)  # [1, 64]
    ranks = ranks.expand(num_tokens, -1)  # [num_tokens, 64]

    # Mask: True for selected experts (in sorted order)
    selection_mask_sorted = ranks < num_selected.unsqueeze(1)  # [num_tokens, 64]

    # Map back to original expert indices
    selection_mask = torch.zeros_like(p_values, dtype=torch.bool)
    selection_mask.scatter_(1, sort_indices, selection_mask_sorted)

    # =========================================================================
    # STEP 8: Get selected expert indices (padded with -1)
    # =========================================================================
    # Find which experts are selected for each token
    selected_experts = torch.full((num_tokens, max_k), -1, device=device, dtype=torch.long)

    for t in range(num_tokens):
        # Get indices of selected experts (sorted by p-value, i.e., best first)
        selected_idx = sort_indices[t, :num_selected[t]]
        selected_experts[t, :len(selected_idx)] = selected_idx

    # =========================================================================
    # STEP 9: Compute routing weights
    # =========================================================================
    # Apply temperature
    scaled_logits = logits_flat / temperature

    # Compute softmax weights
    weights = F.softmax(scaled_logits, dim=-1)

    # Zero out non-selected experts
    routing_weights = weights * selection_mask.float()

    # Renormalize to sum to 1
    weight_sum = routing_weights.sum(dim=-1, keepdim=True)
    weight_sum = torch.clamp(weight_sum, min=1e-10)
    routing_weights = routing_weights / weight_sum

    # =========================================================================
    # STEP 10: Logging (if logger provided)
    # =========================================================================
    if logger is not None:
        for t in range(num_tokens):
            if (token_idx + t) % log_every_n_tokens == 0:
                log_entry = {
                    'sample_idx': sample_idx,
                    'token_idx': token_idx + t,
                    'layer_idx': layer_idx,
                    'router_logits': logits_flat[t].cpu().tolist(),
                    'p_values': p_values[t].cpu().tolist(),
                    'p_values_sorted': p_sorted[t].cpu().tolist(),
                    'hc_statistics': hc_stats[t].cpu().tolist(),
                    'hc_threshold_rank': threshold_ranks[t].item(),
                    'hc_max_value': max_hc_values[t].item(),
                    'num_selected': num_selected[t].item(),
                    'selected_experts': selected_experts[t].cpu().tolist(),
                    'routing_weights': routing_weights[t].cpu().tolist(),
                    'beta': beta,
                    'hc_variant': hc_variant,
                    'min_k': min_k,
                    'max_k': max_k,
                }
                logger.log_routing_decision(log_entry)

    # =========================================================================
    # STEP 11: Reshape outputs to match input shape
    # =========================================================================
    routing_weights = routing_weights.view(batch_size, seq_len, num_experts)
    selected_experts = selected_experts.view(batch_size, seq_len, max_k)
    expert_counts = num_selected.view(batch_size, seq_len)

    if was_2d:
        routing_weights = routing_weights.squeeze(0)
        selected_experts = selected_experts.squeeze(0)
        expert_counts = expert_counts.squeeze(0)

    # =========================================================================
    # STEP 12: Return results
    # =========================================================================
    if return_stats:
        stats = {
            'p_values': p_values.view(batch_size, seq_len, num_experts) if not was_2d else p_values,
            'hc_statistics': hc_stats.view(batch_size, seq_len, num_experts) if not was_2d else hc_stats,
            'threshold_ranks': threshold_ranks.view(batch_size, seq_len) if not was_2d else threshold_ranks,
            'max_hc_values': max_hc_values.view(batch_size, seq_len) if not was_2d else max_hc_values,
        }
        return routing_weights, selected_experts, expert_counts, stats

    return routing_weights, selected_experts, expert_counts, None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_hc_routing_statistics(
    expert_counts: torch.Tensor,
    routing_weights: torch.Tensor,
    threshold_ranks: Optional[torch.Tensor] = None,
    max_hc_values: Optional[torch.Tensor] = None,
    min_k: int = 1,
    max_k: int = 16
) -> Dict[str, float]:
    """
    Compute summary statistics for HC routing.

    Args:
        expert_counts: Number of experts per token [num_tokens]
        routing_weights: Routing weights [num_tokens, num_experts]
        threshold_ranks: HC threshold rank per token [num_tokens]
        max_hc_values: Maximum HC value per token [num_tokens]
        min_k: Minimum experts configuration
        max_k: Maximum experts configuration

    Returns:
        Dict with statistics:
        - avg_experts, std_experts, min_experts, max_experts
        - floor_hit_rate, ceiling_hit_rate, mid_range_rate
        - avg_hc_threshold, std_hc_threshold (if provided)
        - avg_hc_max, std_hc_max (if provided)
        - selection_entropy, expert_activation_ratio
    """
    counts = expert_counts.float()

    stats = {
        'avg_experts': counts.mean().item(),
        'std_experts': counts.std().item() if len(counts) > 1 else 0.0,
        'min_experts': counts.min().item(),
        'max_experts': counts.max().item(),
        'adaptive_range': (counts.max() - counts.min()).item(),
    }

    # Floor/ceiling hit rates
    num_tokens = len(counts)
    stats['floor_hit_rate'] = (counts == min_k).sum().item() / num_tokens * 100
    stats['ceiling_hit_rate'] = (counts == max_k).sum().item() / num_tokens * 100
    stats['mid_range_rate'] = 100 - stats['floor_hit_rate'] - stats['ceiling_hit_rate']

    # HC-specific statistics
    if threshold_ranks is not None:
        ranks = threshold_ranks.float()
        stats['avg_hc_threshold'] = ranks.mean().item()
        stats['std_hc_threshold'] = ranks.std().item() if len(ranks) > 1 else 0.0

    if max_hc_values is not None:
        hc_max = max_hc_values.float()
        stats['avg_hc_max'] = hc_max.mean().item()
        stats['std_hc_max'] = hc_max.std().item() if len(hc_max) > 1 else 0.0
        stats['hc_signal_strength'] = (hc_max > 0).sum().item() / num_tokens * 100

    # Selection entropy (how spread out are the selections)
    if routing_weights is not None:
        # Compute entropy of routing weights
        eps = 1e-10
        entropy = -(routing_weights * torch.log(routing_weights + eps)).sum(dim=-1)
        stats['selection_entropy'] = entropy.mean().item()

        # Expert activation ratio (what fraction of experts ever get used)
        ever_selected = (routing_weights > 0.01).any(dim=0)
        stats['expert_activation_ratio'] = ever_selected.sum().item() / routing_weights.shape[-1]

    return stats


# =============================================================================
# COMPARISON HELPER
# =============================================================================

def compare_hc_vs_bh(
    router_logits: torch.Tensor,
    layer_idx: int = 0,
    kde_models: Optional[Dict] = None,
    alpha: float = 0.6,
    min_k: int = 1,
    max_k: int = 16
) -> Dict[str, Any]:
    """
    Compare HC and BH routing on same input.

    Useful for debugging and understanding when HC outperforms BH.

    Args:
        router_logits: [num_tokens, num_experts]
        layer_idx: Which layer
        kde_models: Pre-loaded KDE models
        alpha: BH alpha parameter
        min_k, max_k: Expert count constraints

    Returns:
        Dict with 'hc_counts', 'bh_counts', 'hc_selected', 'bh_selected',
        'hc_wins', 'bh_wins', 'ties'
    """
    from bh_routing import benjamini_hochberg_routing

    if kde_models is None:
        kde_models = load_kde_models()

    # Run HC
    _, hc_selected, hc_counts, _ = higher_criticism_routing(
        router_logits, min_k=min_k, max_k=max_k,
        layer_idx=layer_idx, kde_models=kde_models
    )

    # Run BH
    _, bh_selected, bh_counts = benjamini_hochberg_routing(
        router_logits, alpha=alpha, min_k=min_k, max_k=max_k,
        layer_idx=layer_idx, kde_models=kde_models
    )

    # Compare
    hc_c = hc_counts.float()
    bh_c = bh_counts.float()

    return {
        'hc_counts': hc_counts,
        'bh_counts': bh_counts,
        'hc_avg': hc_c.mean().item(),
        'bh_avg': bh_c.mean().item(),
        'hc_wins': (hc_c > bh_c).sum().item(),
        'bh_wins': (bh_c > hc_c).sum().item(),
        'ties': (hc_c == bh_c).sum().item(),
        'hc_more_experts': (hc_c > bh_c).float().mean().item() * 100,
    }
