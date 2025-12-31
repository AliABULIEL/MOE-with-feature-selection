"""
Higher Criticism Routing for Mixture-of-Experts Models
======================================================

Implements Higher Criticism (HC) statistical method for adaptive expert
selection in sparse MoE models. HC automatically finds the signal-noise
boundary by detecting where p-values deviate most from uniform distribution.

Key Advantages over Benjamini-Hochberg:
- Adaptive: finds natural threshold instead of fixed formula
- Single tuning parameter (beta) controls search range
- Selects more experts when signal is strong

SIMPLIFIED API (Option B):
--------------------------
Beta is the ONLY tuning parameter:
    - beta='auto'  → Adaptive search (HC⁺) - searches where p < expected
    - beta=1.0     → Full search (HC) - searches all 64 ranks
    - beta=0.3-0.9 → Partial search (HC*) - searches first β fraction of ranks

Example:
    # Adaptive (recommended for most cases)
    weights, experts, counts, _ = higher_criticism_routing(logits, beta='auto')
    
    # Conservative (search first 30% of ranks)
    weights, experts, counts, _ = higher_criticism_routing(logits, beta=0.3)
    
    # Moderate (search first 50% of ranks)  
    weights, experts, counts, _ = higher_criticism_routing(logits, beta=0.5)
    
    # Full search (all ranks)
    weights, experts, counts, _ = higher_criticism_routing(logits, beta=1.0)

References:
    Donoho, D., & Jin, J. (2004). Higher criticism for detecting sparse
    heterogeneous mixtures. The Annals of Statistics, 32(3), 962-994.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union, TYPE_CHECKING
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
    beta: Union[float, str] = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find the HC threshold - the rank with maximum HC statistic.

    SIMPLIFIED API - Beta controls everything:
    - beta='auto' → HC⁺ (only search where p < expected) - ADAPTIVE
    - beta=1.0    → HC (search all ranks) - FULL
    - beta=0.3-0.9 → HC* (search first β fraction) - PARTIAL

    Args:
        hc_stats: HC statistics [num_tokens, num_experts]
        p_values_sorted: Sorted p-values [num_tokens, num_experts]
        n: Number of experts
        min_k: Minimum experts to select (safety floor)
        max_k: Maximum experts to select (ceiling)
        beta: Search fraction β ∈ (0, 1] OR 'auto' for adaptive

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

    # ==========================================================================
    # SIMPLIFIED LOGIC: Derive behavior from beta value
    # ==========================================================================
    if beta == 'auto':
        # HC⁺ (plus): Adaptive - only consider ranks where p < expected
        # This focuses on the "signal" region - most robust variant
        ranks = torch.arange(1, n + 1, device=device).float()
        expected = ranks / n
        expected = expected.unsqueeze(0).expand(num_tokens, -1)
        valid_mask = p_values_sorted < expected  # [num_tokens, num_experts]
        
    elif isinstance(beta, (int, float)) and beta >= 1.0:
        # HC (standard): Full search - consider all ranks
        valid_mask = torch.ones_like(hc_stats, dtype=torch.bool)
        
    elif isinstance(beta, (int, float)) and 0 < beta < 1.0:
        # HC* (star): Partial search - only first β fraction of ranks
        max_search = max(1, int(n * beta))  # At least 1 rank
        valid_mask = torch.zeros_like(hc_stats, dtype=torch.bool)
        valid_mask[:, :max_search] = True
        
    else:
        raise ValueError(
            f"beta must be 'auto' or a float in (0, 1], got {beta}"
        )

    # CRITICAL FIX: Do NOT apply max_k to search range
    # β defines search range independently of final selection constraints
    # Only prevent searching below min_k (if min_k > 1)
    if min_k > 1:
        valid_mask[:, :min_k-1] = False  # Must select at least min_k

    # NOTE: Removed the line `valid_mask[:, max_k:] = False` which was
    # incorrectly limiting the search range. This caused all β ≥ max_k/n
    # to behave identically. max_k should only clamp the FINAL selection,
    # not restrict where we search for the HC threshold.

    # Set invalid positions to -inf so they're never selected
    hc_masked = hc_stats.clone()
    hc_masked[~valid_mask] = float('-inf')

    # Find maximum HC and its position (within β-defined search range)
    max_hc_values, threshold_ranks = torch.max(hc_masked, dim=1)

    # threshold_ranks is 0-indexed, so num_selected = threshold_ranks + 1
    num_selected = threshold_ranks + 1

    # Handle edge case: if all positions were invalid (no signal found)
    # Fall back to min_k selection
    no_valid = (max_hc_values == float('-inf'))
    num_selected[no_valid] = min_k
    threshold_ranks[no_valid] = min_k - 1
    max_hc_values[no_valid] = 0.0

    # APPLY min_k/max_k constraints AFTER finding threshold
    # This is the ONLY place where max_k should affect the result
    num_selected_unclamped = num_selected.clone()  # Save for diagnostics
    num_selected = torch.clamp(num_selected, min=min_k, max=max_k)

    # Diagnostic: warn if clamping occurs frequently
    clamped_down = (num_selected_unclamped > max_k).sum().item()
    clamped_up = (num_selected_unclamped < min_k).sum().item()
    if clamped_down > num_tokens * 0.3:  # >30% tokens clamped down
        warnings.warn(
            f"{clamped_down}/{num_tokens} tokens had HC threshold > max_k={max_k}. "
            f"Consider increasing max_k or using lower β.",
            UserWarning
        )
    if clamped_up > num_tokens * 0.3:  # >30% tokens clamped up
        warnings.warn(
            f"{clamped_up}/{num_tokens} tokens had HC threshold < min_k={min_k}. "
            f"Consider decreasing min_k or using higher β.",
            UserWarning
        )

    return num_selected, threshold_ranks, max_hc_values


def higher_criticism_routing(
    router_logits: torch.Tensor,
    beta: Union[float, str] = 0.5,
    temperature: float = 1.0,
    min_k: int = 1,
    max_k: int = 16,
    layer_idx: int = 0,
    kde_models: Optional[Dict[int, Dict]] = None,
    return_stats: bool = False,
    logger: Optional['HCRoutingLogger'] = None,
    log_every_n_tokens: int = 100,
    sample_idx: int = 0,
    token_idx: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict]]:
    """
    Higher Criticism routing for adaptive expert selection in MoE models.

    SIMPLIFIED API - Beta is the ONLY tuning parameter!
    ===================================================
    
    Beta controls the search range for finding the optimal threshold:
    
    | Beta Value | Behavior | Searches | Use Case |
    |------------|----------|----------|----------|
    | 'auto'     | Adaptive | Where p < expected | Recommended - self-tuning |
    | 1.0        | Full     | All 64 ranks | Maximum coverage |
    | 0.7        | Wide     | First 45 ranks | Balanced |
    | 0.5        | Medium   | First 32 ranks | Moderate |
    | 0.3        | Narrow   | First 19 ranks | Conservative |

    Args:
        router_logits: Router output logits
            Shape: [batch, seq_len, num_experts] or [num_tokens, num_experts]
        beta: Search fraction - THE MAIN TUNING PARAMETER
            - 'auto': Adaptive search (only where p < expected) [RECOMMENDED]
            - 1.0: Search all ranks
            - 0.3-0.9: Search first β fraction of ranks
        temperature: Softmax temperature for weight computation (default: 1.0)
        min_k: Minimum experts to select per token (safety floor)
        max_k: Maximum experts to select per token (ceiling)
        layer_idx: Which transformer layer (0-15) for KDE model lookup
        kde_models: Pre-loaded KDE models. If None, loads automatically.
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
        >>> import torch
        >>> logits = torch.randn(1, 10, 64)  # batch=1, seq=10, experts=64
        
        >>> # Adaptive (recommended)
        >>> weights, experts, counts, _ = higher_criticism_routing(
        ...     logits, beta='auto', min_k=4, max_k=12
        ... )
        
        >>> # Conservative (fewer experts)
        >>> weights, experts, counts, _ = higher_criticism_routing(
        ...     logits, beta=0.3, min_k=4, max_k=12
        ... )
        
        >>> # Moderate
        >>> weights, experts, counts, _ = higher_criticism_routing(
        ...     logits, beta=0.5, min_k=4, max_k=12
        ... )
        
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

    # Validate inputs
    assert router_logits.dim() in [2, 3], f"Expected 2D or 3D tensor, got {router_logits.dim()}D"
    assert 0 < min_k <= max_k, f"Invalid constraints: min_k={min_k}, max_k={max_k}"
    assert temperature > 0, f"Temperature must be positive, got {temperature}"

    # Validate beta
    if beta != 'auto' and not isinstance(beta, (int, float)):
        raise ValueError(f"beta must be 'auto' or a number, got {type(beta)}")
    if isinstance(beta, (int, float)) and not (0 < beta <= 1.0):
        raise ValueError(f"beta must be in (0, 1], got {beta}")

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

    # Validate HC computation
    assert hc_stats.shape == (num_tokens, num_experts), f"HC stats shape mismatch: {hc_stats.shape}"
    assert not torch.isnan(hc_stats).any(), "HC statistics contain NaN values"
    assert not torch.isinf(hc_stats).any(), "HC statistics contain Inf values"

    # =========================================================================
    # STEP 6: Find HC threshold (where to cut off selection)
    # =========================================================================
    num_selected, threshold_ranks, max_hc_values = find_hc_threshold(
        hc_stats, p_sorted, num_experts,
        min_k=min_k, max_k=max_k, beta=beta
    )

    # Validate threshold selection
    assert num_selected.shape == (num_tokens,), f"num_selected shape mismatch: {num_selected.shape}"
    assert (num_selected >= min_k).all(), f"Some selections below min_k: {num_selected.min()}"
    assert (num_selected <= max_k).all(), f"Some selections above max_k: {num_selected.max()}"
    assert (threshold_ranks >= 0).all() and (threshold_ranks < num_experts).all(), \
        "threshold_ranks out of bounds"

    # Warn if no positive HC signal detected
    no_signal_count = (max_hc_values <= 0).sum().item()
    if no_signal_count > 0:
        warnings.warn(
            f"Layer {layer_idx}: No positive HC signal in {no_signal_count}/{num_tokens} tokens. "
            f"Using min_k={min_k} fallback.",
            UserWarning
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
        num_to_select = len(selected_idx)
        assert num_to_select == num_selected[t].item(), "Selection count mismatch"
        assert all(0 <= idx < num_experts for idx in selected_idx), "Expert index out of bounds"
        selected_experts[t, :num_to_select] = selected_idx

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

    # Validate final weights
    final_sums = routing_weights.sum(dim=-1)
    assert torch.allclose(final_sums, torch.ones_like(final_sums), atol=1e-5), \
        f"Routing weights don't sum to 1: min={final_sums.min():.6f}, max={final_sums.max():.6f}"
    assert (routing_weights >= 0).all(), "Routing weights contain negative values"
    assert (routing_weights <= 1).all(), "Routing weights exceed 1.0"

    # =========================================================================
    # STEP 10: Logging (if logger provided) - COMPLETE SCHEMA
    # =========================================================================
    if logger is not None:
        for t in range(num_tokens):
            if (token_idx + t) % log_every_n_tokens == 0:
                # Get data for this token
                token_logits = logits_flat[t]
                token_p_values = p_values[t]
                token_p_sorted = p_sorted[t]
                token_sort_indices = sort_indices[t]
                token_hc_stats = hc_stats[t]
                token_num_selected = num_selected[t].item()
                token_threshold_rank = threshold_ranks[t].item()
                token_max_hc = max_hc_values[t].item()
                token_selected_experts = selected_experts[t].cpu().tolist()
                token_routing_weights = routing_weights[t]

                # Compute statistics
                logit_stats = {
                    'min': float(token_logits.min()),
                    'max': float(token_logits.max()),
                    'mean': float(token_logits.mean()),
                    'std': float(token_logits.std())
                }

                # Determine selection reason
                if token_num_selected == min_k:
                    if token_max_hc == 0.0:
                        selection_reason = 'no_signal'
                    else:
                        selection_reason = 'min_k_floor'
                elif token_num_selected == max_k:
                    selection_reason = 'max_k_ceiling'
                else:
                    selection_reason = 'hc_threshold'

                # Count positive HC values
                hc_positive_count = int((token_hc_stats > 0).sum())

                # Determine search range based on beta
                if beta == 'auto':
                    # For auto, we search where p < expected
                    ranks = torch.arange(1, num_experts + 1, device=device).float()
                    expected = ranks / num_experts
                    search_start = 1
                    search_end = int((token_p_sorted < expected).sum())
                    beta_str = 'auto'
                elif isinstance(beta, (int, float)) and beta >= 1.0:
                    search_start = 1
                    search_end = num_experts
                    beta_str = str(beta)
                else:
                    search_start = 1
                    search_end = max(1, int(num_experts * beta))
                    beta_str = str(beta)

                # Validate weights sum
                weights_sum = float(token_routing_weights.sum())

                # Create complete log entry with FULL schema
                log_entry = {
                    # === IDENTIFICATION ===
                    'sample_idx': sample_idx,
                    'token_idx': token_idx + t,
                    'layer_idx': layer_idx,

                    # === ROUTER OUTPUTS ===
                    'router_logits': token_logits.cpu().tolist(),
                    'router_logits_stats': logit_stats,

                    # === P-VALUE COMPUTATION ===
                    'kde_model_id': f"layer_{layer_idx}",
                    'p_values': token_p_values.cpu().tolist(),
                    'p_values_sorted': token_p_sorted.cpu().tolist(),
                    'sort_indices': token_sort_indices.cpu().tolist(),

                    # === HC STATISTICS (CRITICAL) ===
                    'hc_statistics': token_hc_stats.cpu().tolist(),
                    'hc_max_rank': token_threshold_rank + 1,  # Convert to 1-indexed
                    'hc_max_value': token_max_hc,
                    'hc_positive_count': hc_positive_count,
                    'search_range': {
                        'beta': beta_str,
                        'start_rank': search_start,
                        'end_rank': search_end
                    },

                    # === SELECTION DECISION ===
                    'num_selected': token_num_selected,
                    'selected_experts': token_selected_experts,
                    'routing_weights': token_routing_weights.cpu().tolist(),
                    'selection_reason': selection_reason,

                    # === CONSTRAINT FLAGS ===
                    'hit_min_k': (token_num_selected == min_k),
                    'hit_max_k': (token_num_selected == max_k),
                    'fallback_triggered': (token_max_hc == 0.0),

                    # === VALIDATION ===
                    'weights_sum': weights_sum,
                    'config': {
                        'min_k': min_k,
                        'max_k': max_k,
                        'temperature': temperature
                    }
                }

                # Log with error handling
                try:
                    logger.log_routing_decision(log_entry)
                except Exception as e:
                    warnings.warn(f"Logging failed: {e}", UserWarning)

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
            'beta': beta,
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
    beta: Union[float, str] = 0.5,
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
        beta: HC beta parameter ('auto' or float)
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
        router_logits, beta=beta, min_k=min_k, max_k=max_k,
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
        'beta': beta,
        'alpha': alpha,
    }


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HC ROUTING - SIMPLIFIED BETA API TEST")
    print("=" * 70)
    
    # Create test logits
    torch.manual_seed(42)
    logits = torch.randn(20, 64)
    
    print("\nTesting different beta values:\n")
    
    # Test different beta values
    for beta in ['auto', 0.3, 0.5, 0.7, 1.0]:
        weights, experts, counts, stats = higher_criticism_routing(
            logits,
            beta=beta,
            min_k=4,
            max_k=12,
            return_stats=True
        )
        
        avg = counts.float().mean().item()
        std = counts.float().std().item()
        min_c = counts.min().item()
        max_c = counts.max().item()
        
        beta_str = f"beta={beta}" if beta != 'auto' else "beta='auto'"
        print(f"  {beta_str:15s} → avg={avg:.1f} ± {std:.1f}, range=[{min_c}, {max_c}]")
    
    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
