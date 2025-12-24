"""
Higher Criticism Routing for Mixture-of-Experts Models
=======================================================

This module implements the Higher Criticism (HC) procedure for expert selection
in sparse MoE models, providing optimal sparse signal detection.

HC is particularly effective when:
- Most experts are irrelevant (null) for a given token
- A few experts are relevant (non-null)
- Signal strength varies across experts

This matches expert routing scenarios perfectly!

Key Features:
- KDE-based p-values (reuses existing KDE models from BH)
- Fully vectorized PyTorch implementation
- GPU-compatible
- Configurable beta parameter (search fraction)
- Min/max expert constraints
- Numerical stability guarantees

References:
    Donoho, D. & Jin, J. (2004). Higher criticism for detecting sparse
    heterogeneous mixtures. Annals of Statistics, 32(3), 962-994.

    Donoho, D. & Jin, J. (2015). Higher Criticism for Large-Scale Inference,
    Especially for Rare and Weak Effects. Statistical Science, 30(1), 1-25.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, TYPE_CHECKING
import warnings
import numpy as np

# Reuse from BH routing (DO NOT DUPLICATE - import!)
from bh_routing import (
    load_kde_models,
    compute_pvalues_kde,
    compute_pvalues_empirical,
    compute_routing_statistics
)

if TYPE_CHECKING:
    from bh_routing_logging import BHRoutingLogger


def compute_hc_scores(
    sorted_pvalues: torch.Tensor,
    beta: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Higher Criticism scores for each rank.

    The HC statistic measures the standardized deviation between expected
    and observed p-value distributions:

    HC(i) = √n × (i/n - p₍ᵢ₎) / √(p₍ᵢ₎(1 - p₍ᵢ₎))

    Where:
    - n = number of tests (experts)
    - i/n = expected fraction under uniform null
    - p₍ᵢ₎ = i-th sorted p-value (observed)
    - Denominator = standard error

    Args:
        sorted_pvalues: [num_tokens, num_experts] p-values ALREADY SORTED ascending
                       sorted_pvalues[t, 0] is smallest p-value for token t
        beta: Search fraction in (0, 1]. Only compute HC for ranks 1 to floor(β×n)
              Default 0.5 means search first half of ranks.

    Returns:
        hc_scores: [num_tokens, num_experts] HC score at each rank
                  Ranks > floor(β×n) have score = -inf (excluded from search)
        i_star: [num_tokens] optimal rank (1-indexed) where HC is maximized
                This is the number of experts to select.

    Example:
        >>> p_sorted = torch.tensor([[0.01, 0.05, 0.15, 0.35, 0.50, 0.65, 0.75, 0.85]])
        >>> hc_scores, i_star = compute_hc_scores(p_sorted, beta=0.5)
        >>> # HC computed for ranks 1-4 (half of 8)
        >>> # i_star[0] = rank with maximum HC

    Notes:
        - Higher HC score indicates stronger evidence of signal
        - i_star = 1 means only 1 expert should be selected
        - Numerical stability: p-values clamped to (eps, 1-eps)
    """
    device = sorted_pvalues.device
    dtype = sorted_pvalues.dtype
    num_tokens, num_experts = sorted_pvalues.shape

    # Numerical stability
    eps = 1e-10
    sorted_pvalues = torch.clamp(sorted_pvalues, min=eps, max=1.0 - eps)

    # Maximum rank to search
    max_rank = max(1, int(beta * num_experts))

    # Create rank indices [1, 2, 3, ..., n] (1-indexed)
    ranks = torch.arange(1, num_experts + 1, device=device, dtype=torch.float32)
    ranks = ranks.view(1, -1).expand(num_tokens, -1)  # [num_tokens, num_experts]

    # Expected fraction under uniform null: i/n
    expected = ranks / num_experts  # [num_tokens, num_experts]

    # Observed: sorted p-values
    observed = sorted_pvalues  # [num_tokens, num_experts]

    # Standard error: sqrt(p * (1 - p))
    se = torch.sqrt(observed * (1.0 - observed) + eps)  # [num_tokens, num_experts]

    # HC score: sqrt(n) * (expected - observed) / se
    sqrt_n = np.sqrt(num_experts)
    hc_scores = sqrt_n * (expected - observed) / se  # [num_tokens, num_experts]

    # Mask out ranks > max_rank (set to -inf so they won't be selected)
    rank_mask = ranks <= max_rank  # [num_tokens, num_experts]
    hc_scores = torch.where(rank_mask, hc_scores, torch.tensor(float('-inf'), device=device))

    # Find i* = argmax(HC) for each token
    # Note: argmax returns 0-indexed, but we want 1-indexed rank
    i_star_0indexed = hc_scores.argmax(dim=-1)  # [num_tokens]
    i_star = i_star_0indexed + 1  # Convert to 1-indexed rank

    # Handle edge case: if all HC scores are -inf or negative, default to 1
    max_hc = hc_scores.max(dim=-1).values
    i_star = torch.where(max_hc > float('-inf'), i_star, torch.ones_like(i_star))

    return hc_scores, i_star


def higher_criticism_routing(
    router_logits: torch.Tensor,
    beta: float = 0.5,
    temperature: float = 1.0,
    min_k: int = 1,
    max_k: int = 16,
    layer_idx: int = 0,
    kde_models: Optional[Dict[int, Dict]] = None,
    return_stats: bool = False,
    logger: Optional['BHRoutingLogger'] = None,
    log_every_n_tokens: int = 100,
    sample_idx: int = 0,
    token_idx: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Implements Higher Criticism procedure for expert selection in MoE models.

    HC is optimal for sparse signal detection - finding the few relevant experts
    among many irrelevant ones. Unlike BH which uses a fixed threshold (α),
    HC automatically finds the optimal cutoff by maximizing a test statistic.

    Algorithm:
    1. Compute p-values using KDE: p_i = 1 - CDF(logit_i)
       (Higher logit → higher CDF → lower p-value → more significant)
    2. Sort p-values ascending (smallest = most significant first)
    3. Compute HC score at each rank i for i ∈ [1, β×n]
    4. Find i* = argmax(HC) - the optimal number of experts
    5. Select top i* experts (those with smallest p-values)
    6. Apply constraints: clamp i* to [min_k, max_k]
    7. Compute routing weights via softmax and renormalize

    Args:
        router_logits: Router output logits
            Shape: [batch_size, seq_len, num_experts] or [num_tokens, num_experts]
        beta: HC search fraction in (0, 1]. Default: 0.5
            - β=0.3: Conservative, search first 30% of ranks → fewer experts
            - β=0.5: Balanced, search first half → moderate selection
            - β=0.7: Inclusive, search first 70% → more experts
            HC statistic is computed for ranks 1 to floor(β × num_experts).
        temperature: Softmax temperature for probability computation. Default: 1.0
            Higher → more uniform weights, Lower → sharper weights
        min_k: Minimum experts to select per token. Default: 1
            Enforced after HC selection (i* clamped to ≥ min_k)
        max_k: Maximum experts to select per token. Default: 16
            Enforced after HC selection (i* clamped to ≤ max_k)
        layer_idx: Layer index (0-15) for KDE model lookup. Default: 0
        kde_models: Pre-loaded KDE models dict. If None, loads from default path.
        return_stats: If True, return additional statistics dict. Default: False
        logger: Optional BHRoutingLogger for detailed logging
        log_every_n_tokens: Logging sampling rate. Default: 100
        sample_idx, token_idx: Position identifiers for logging

    Returns:
        routing_weights: Sparse routing weights
            Shape: [batch_size, seq_len, num_experts] or [num_tokens, num_experts]
            Non-selected experts have weight 0. Selected weights sum to 1.
        selected_experts: Indices of selected experts, padded with -1
            Shape: [batch_size, seq_len, max_k] or [num_tokens, max_k]
        expert_counts: Number of experts selected per token
            Shape: [batch_size, seq_len] or [num_tokens]

        If return_stats=True, also returns:
            stats: Dict containing:
                - 'hc_scores': HC scores at each rank
                - 'i_star': Optimal rank before constraints
                - 'p_values': Computed p-values
                - 'sorted_pvalues': Sorted p-values
                - 'sorted_indices': Expert indices in sorted order
                - 'beta': Beta parameter used

    Example:
        >>> logits = torch.randn(2, 10, 64)  # [batch, seq, experts]
        >>> weights, experts, counts = higher_criticism_routing(
        ...     logits, beta=0.5, max_k=8
        ... )
        >>> print(weights.sum(dim=-1))  # All 1.0
        >>> print(counts)  # Variable, typically 1-8

    Comparison with BH Routing:
        - BH: Uses threshold p₍ₖ₎ ≤ (k/n)×α, requires tuning α
        - HC: Uses argmax of HC statistic, fully adaptive
        - Both use same KDE p-value computation
        - HC often selects fewer experts for "easy" tokens
    """
    # =========================================================================
    # Input Validation
    # =========================================================================
    if not isinstance(router_logits, torch.Tensor):
        raise TypeError(f"router_logits must be torch.Tensor, got {type(router_logits)}")

    if router_logits.ndim not in [2, 3]:
        raise ValueError(
            f"router_logits must be 2D or 3D, got shape {router_logits.shape}"
        )

    # Handle 2D input
    input_is_2d = router_logits.ndim == 2
    if input_is_2d:
        router_logits = router_logits.unsqueeze(0)  # [1, num_tokens, num_experts]

    batch_size, seq_len, num_experts = router_logits.shape

    # Validate parameters
    if not 0.0 < beta <= 1.0:
        raise ValueError(f"beta must be in (0, 1], got {beta}")

    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    if not 1 <= min_k <= num_experts:
        raise ValueError(f"min_k must be in [1, {num_experts}], got {min_k}")

    if not min_k <= max_k <= num_experts:
        raise ValueError(f"max_k must be in [{min_k}, {num_experts}], got {max_k}")

    device = router_logits.device
    dtype = router_logits.dtype
    eps = 1e-10

    # =========================================================================
    # Step 1: Compute Softmax Probabilities (for final weights)
    # =========================================================================
    scaled_logits = router_logits / temperature
    probs = F.softmax(scaled_logits, dim=-1, dtype=torch.float32).to(dtype)

    # =========================================================================
    # Step 2: Load KDE Models and Compute P-Values
    # =========================================================================
    if kde_models is None:
        kde_models = load_kde_models()

    # Flatten to 2D for p-value computation
    router_logits_2d = router_logits.view(-1, num_experts)

    # Compute p-values: p = 1 - CDF(logit)
    p_values = compute_pvalues_kde(router_logits_2d, layer_idx, kde_models)
    p_values = p_values.view(batch_size, seq_len, num_experts)

    # =========================================================================
    # Step 3: Sort P-Values Ascending
    # =========================================================================
    sorted_pvals, sorted_indices = torch.sort(p_values, dim=-1)
    # sorted_pvals[b, s, 0] = smallest p-value (most significant)
    # sorted_indices[b, s, k] = original expert index for rank k

    # =========================================================================
    # Step 4: Compute HC Scores and Find i*
    # =========================================================================
    # Flatten for HC computation
    sorted_pvals_2d = sorted_pvals.view(-1, num_experts)

    hc_scores, i_star = compute_hc_scores(sorted_pvals_2d, beta=beta)

    # Reshape back
    hc_scores = hc_scores.view(batch_size, seq_len, num_experts)
    i_star = i_star.view(batch_size, seq_len)

    # =========================================================================
    # Step 5: Apply Constraints [min_k, max_k]
    # =========================================================================
    num_selected = i_star.clone()
    num_selected = num_selected.clamp(min=min_k, max=max_k)

    # =========================================================================
    # Step 6: Create Selection Mask and Compute Weights
    # =========================================================================
    # Create mask in sorted order
    expert_ranks = torch.arange(num_experts, device=device).view(1, 1, -1)
    expert_ranks = expert_ranks.expand(batch_size, seq_len, -1)

    # selected_mask_sorted[b, s, k] = True if k < num_selected[b, s]
    selected_mask_sorted = expert_ranks < num_selected.unsqueeze(-1)

    # Convert mask from sorted order to original expert order
    selected_mask = torch.zeros_like(probs, dtype=torch.bool)
    selected_mask.scatter_(dim=-1, index=sorted_indices, src=selected_mask_sorted)

    # Apply mask to probabilities
    routing_weights = torch.where(selected_mask, probs, torch.zeros_like(probs))

    # Renormalize to sum to 1
    weight_sums = routing_weights.sum(dim=-1, keepdim=True).clamp(min=eps)
    routing_weights = routing_weights / weight_sums

    # =========================================================================
    # Step 7: Extract Selected Expert Indices (padded to max_k)
    # =========================================================================
    selected_experts = torch.full(
        (batch_size, seq_len, max_k),
        fill_value=-1,
        dtype=torch.long,
        device=device
    )

    for k_idx in range(max_k):
        slot_active = k_idx < num_selected
        expert_idx = sorted_indices[:, :, k_idx]
        selected_experts[:, :, k_idx] = torch.where(
            slot_active, expert_idx, torch.full_like(expert_idx, -1)
        )

    # =========================================================================
    # Step 8: Logging (if logger provided)
    # =========================================================================
    if logger is not None and token_idx % log_every_n_tokens == 0:
        if input_is_2d:
            if token_idx < seq_len:
                log_entry = {
                    'sample_idx': sample_idx,
                    'token_idx': token_idx,
                    'layer_idx': layer_idx,
                    'method': 'hc',
                    'beta': beta,
                    'router_logits': router_logits[0, token_idx, :].detach().cpu().numpy(),
                    'p_values': p_values[0, token_idx, :].detach().cpu().numpy(),
                    'hc_scores': hc_scores[0, token_idx, :].detach().cpu().numpy(),
                    'i_star': int(i_star[0, token_idx].item()),
                    'num_selected': int(num_selected[0, token_idx].item()),
                    'selected_experts': selected_experts[0, token_idx, :].detach().cpu().tolist(),
                    'max_k': max_k,
                    'min_k': min_k,
                }
                logger.log_routing_decision(log_entry)

    # =========================================================================
    # Step 9: Prepare Output
    # =========================================================================
    if input_is_2d:
        routing_weights = routing_weights.squeeze(0)
        selected_experts = selected_experts.squeeze(0)
        num_selected = num_selected.squeeze(0)
        if return_stats:
            hc_scores = hc_scores.squeeze(0)
            p_values = p_values.squeeze(0)
            sorted_pvals = sorted_pvals.squeeze(0)
            sorted_indices = sorted_indices.squeeze(0)
            i_star = i_star.squeeze(0)

    if return_stats:
        stats = {
            'hc_scores': hc_scores,
            'i_star': i_star,
            'p_values': p_values,
            'sorted_pvalues': sorted_pvals,
            'sorted_indices': sorted_indices,
            'beta': beta,
            'temperature': temperature,
            'layer_idx': layer_idx,
            'kde_available': layer_idx in kde_models if kde_models else False
        }
        return routing_weights, selected_experts, num_selected, stats

    return routing_weights, selected_experts, num_selected


def run_hc_multi_beta(
    router_logits: torch.Tensor,
    beta_values: list = None,
    temperature: float = 1.0,
    min_k: int = 1,
    max_k: int = 16,
    kde_models: Optional[Dict] = None
) -> Dict[float, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Run HC routing with multiple beta values for comparison.

    Args:
        router_logits: [batch, seq, experts] or [tokens, experts]
        beta_values: List of beta values to test (default: [0.3, 0.5, 0.7, 1.0])
        temperature: Softmax temperature
        min_k: Minimum experts
        max_k: Maximum experts
        kde_models: Pre-loaded KDE models

    Returns:
        Dict mapping beta -> (routing_weights, selected_experts, expert_counts)
    """
    if beta_values is None:
        beta_values = [0.3, 0.5, 0.7, 1.0]

    if kde_models is None:
        kde_models = load_kde_models()

    results = {}
    for beta in beta_values:
        weights, experts, counts = higher_criticism_routing(
            router_logits,
            beta=beta,
            temperature=temperature,
            min_k=min_k,
            max_k=max_k,
            kde_models=kde_models
        )
        results[beta] = (weights, experts, counts)

    return results


def compare_hc_bh(
    router_logits: torch.Tensor,
    hc_beta: float = 0.5,
    bh_alpha: float = 0.30,
    max_k: int = 16,
    kde_models: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Compare HC and BH routing on the same input.

    Args:
        router_logits: Router logits
        hc_beta: Beta parameter for HC
        bh_alpha: Alpha parameter for BH
        max_k: Maximum experts for both
        kde_models: Pre-loaded KDE models

    Returns:
        Dict with comparison results:
        - hc_counts, bh_counts: Expert count tensors
        - hc_avg, bh_avg: Average experts
        - hc_weights, bh_weights: Routing weight tensors
        - agreement: Fraction of tokens with same count
    """
    from bh_routing import benjamini_hochberg_routing

    if kde_models is None:
        kde_models = load_kde_models()

    # Run HC
    hc_weights, hc_experts, hc_counts = higher_criticism_routing(
        router_logits, beta=hc_beta, max_k=max_k, kde_models=kde_models
    )

    # Run BH
    bh_weights, bh_experts, bh_counts = benjamini_hochberg_routing(
        router_logits, alpha=bh_alpha, max_k=max_k, kde_models=kde_models
    )

    # Compute comparison metrics
    hc_counts_flat = hc_counts.flatten().float()
    bh_counts_flat = bh_counts.flatten().float()

    agreement = (hc_counts.flatten() == bh_counts.flatten()).float().mean().item()

    return {
        'hc_counts': hc_counts,
        'bh_counts': bh_counts,
        'hc_avg': hc_counts_flat.mean().item(),
        'bh_avg': bh_counts_flat.mean().item(),
        'hc_std': hc_counts_flat.std().item(),
        'bh_std': bh_counts_flat.std().item(),
        'hc_weights': hc_weights,
        'bh_weights': bh_weights,
        'agreement': agreement,
        'hc_beta': hc_beta,
        'bh_alpha': bh_alpha,
    }


def compare_multi_beta_statistics(
    results: Dict[float, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """
    Generate comparison statistics for different beta values.

    Args:
        results: Output from run_hc_multi_beta()

    Returns:
        DataFrame with columns: beta, mean_experts, std_experts, etc.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required. Install with: pip install pandas")

    rows = []
    for beta, (weights, experts, counts) in results.items():
        counts_flat = counts.flatten().float()
        max_k = experts.shape[-1]

        rows.append({
            'beta': beta,
            'mean_experts': counts_flat.mean().item(),
            'std_experts': counts_flat.std().item(),
            'min_experts': counts_flat.min().item(),
            'max_experts': counts_flat.max().item(),
            'pct_at_ceiling': (counts_flat == max_k).float().mean().item() * 100,
            'pct_at_floor': (counts_flat == 1).float().mean().item() * 100,
        })

    return pd.DataFrame(rows)


# =============================================================================
# Main Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HIGHER CRITICISM ROUTING DEMONSTRATION")
    print("=" * 70)

    torch.manual_seed(42)

    # Create sample input
    batch_size, seq_len, num_experts = 2, 4, 64
    router_logits = torch.randn(batch_size, seq_len, num_experts)

    print(f"\nInput shape: {router_logits.shape}")
    print(f"Testing with beta=0.5, max_k=8\n")

    # Run HC routing
    weights, experts, counts, stats = higher_criticism_routing(
        router_logits,
        beta=0.5,
        max_k=8,
        return_stats=True
    )

    print(f"Output shapes:")
    print(f"  routing_weights: {weights.shape}")
    print(f"  selected_experts: {experts.shape}")
    print(f"  expert_counts: {counts.shape}")

    print(f"\nExpert counts per token:")
    print(counts)

    print(f"\nWeight sums (should all be ~1.0):")
    print(weights.sum(dim=-1))

    print(f"\nRouting statistics:")
    routing_stats = compute_routing_statistics(weights, counts)
    for key, value in routing_stats.items():
        print(f"  {key}: {value:.4f}")

    # Compare with BH
    print(f"\n" + "=" * 70)
    print("COMPARISON: HC vs BH")
    print("=" * 70)

    comparison = compare_hc_bh(router_logits, hc_beta=0.5, bh_alpha=0.30, max_k=8)

    print(f"\nHC (β=0.5): avg={comparison['hc_avg']:.2f}, std={comparison['hc_std']:.2f}")
    print(f"BH (α=0.30): avg={comparison['bh_avg']:.2f}, std={comparison['bh_std']:.2f}")
    print(f"Agreement: {comparison['agreement']*100:.1f}% of tokens have same count")

    # Multi-beta comparison
    print(f"\n" + "=" * 70)
    print("MULTI-BETA COMPARISON")
    print("=" * 70)

    multi_results = run_hc_multi_beta(router_logits, max_k=8)
    comparison_df = compare_multi_beta_statistics(multi_results)
    print("\n" + comparison_df.to_string(index=False))
