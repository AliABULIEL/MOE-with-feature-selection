"""
Benjamini-Hochberg Routing for Mixture-of-Experts Models
==========================================================

This module implements the Benjamini-Hochberg (BH) procedure for expert selection
in sparse MoE models, providing statistical control of the False Discovery Rate (FDR).

Key Features:
- Fully vectorized PyTorch implementation
- GPU-compatible (no CPU transfers)
- Configurable FDR level (alpha)
- Temperature-based calibration
- Min/max expert constraints
- Numerical stability guarantees

References:
    Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate:
    a practical and powerful approach to multiple testing. Journal of the Royal
    Statistical Society: Series B, 57(1), 289-300.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import warnings


def benjamini_hochberg_routing(
    router_logits: torch.Tensor,
    alpha: float = 0.05,
    temperature: float = 1.0,
    min_k: int = 1,
    max_k: int = 16,
    return_stats: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Implements Benjamini-Hochberg procedure for expert selection in MoE models.

    The BH procedure controls the False Discovery Rate (FDR) when selecting which
    experts to activate for each token. Unlike fixed top-k routing, BH adapts the
    number of experts based on statistical significance.

    Mathematical Formulation:
    -------------------------
    Given router logits r ∈ R^N for N experts:

    1. Compute probabilities: π = softmax(r / τ)
    2. Compute pseudo p-values: p_i = 1 - π_i  (higher prob → lower p-value)
    3. Sort p-values: p_(1) ≤ p_(2) ≤ ... ≤ p_(N)
    4. Find largest k such that: p_(k) ≤ (k/N) × α
    5. Select experts with k smallest p-values
    6. Renormalize selected probabilities to sum to 1

    The critical values c_k = (k/N) × α form a linear sequence, and the BH
    procedure selects all hypotheses up to the last one that falls below its
    critical value.

    Args:
        router_logits: Router logits from MoE gating network
            Shape: [batch_size, seq_len, num_experts] or [num_tokens, num_experts]
        alpha: FDR control level (default: 0.05)
            Lower values → more conservative → fewer experts selected
            Typical values: 0.01 (strict), 0.05 (standard), 0.10 (permissive)
        temperature: Softmax temperature for probability calibration (default: 1.0)
            Higher values → more uniform distribution → more experts selected
            Lower values → sharper distribution → fewer experts selected
        min_k: Minimum number of experts to select per token (default: 1)
            Ensures at least min_k experts are always activated
        max_k: Maximum number of experts to select per token (default: 16)
            Caps selection for compatibility with fixed-size expert computation
        return_stats: If True, return additional statistics (default: False)

    Returns:
        routing_weights: Sparse routing weights, summing to 1 per token
            Shape: [batch_size, seq_len, num_experts]
            Non-selected experts have weight 0
        selected_experts: Indices of selected experts, padded with -1
            Shape: [batch_size, seq_len, max_k]
            Padding value: -1 indicates unused slot
        expert_counts: Number of experts selected per token
            Shape: [batch_size, seq_len]

        If return_stats=True, also returns:
            stats: Dict with keys:
                - 'p_values': Computed pseudo p-values [B, S, N]
                - 'bh_threshold': BH threshold used [B, S]
                - 'significant_mask': Boolean mask of selected experts [B, S, N]

    Raises:
        ValueError: If input shapes are invalid or parameters out of range
        RuntimeError: If numerical instability is detected

    Examples:
        >>> import torch
        >>> router_logits = torch.randn(2, 10, 64)  # 2 samples, 10 tokens, 64 experts
        >>> weights, experts, counts = benjamini_hochberg_routing(
        ...     router_logits, alpha=0.05, temperature=1.0, max_k=8
        ... )
        >>> print(weights.shape)  # [2, 10, 64] - sparse weights
        >>> print(experts.shape)  # [2, 10, 8] - selected expert indices
        >>> print(counts.shape)   # [2, 10] - number selected per token
        >>> print(weights.sum(dim=-1))  # All sum to 1.0

    Notes:
        - The function is fully vectorized and GPU-compatible
        - No Python loops are used for batch/sequence dimensions
        - Numerical stability is ensured via epsilon constants
        - If BH procedure selects 0 experts, min_k experts with lowest p-values are used
        - If BH selects > max_k, only top max_k by probability are kept
    """
    # =========================================================================
    # Input Validation
    # =========================================================================
    if not isinstance(router_logits, torch.Tensor):
        raise TypeError(f"router_logits must be a torch.Tensor, got {type(router_logits)}")

    if router_logits.ndim not in [2, 3]:
        raise ValueError(
            f"router_logits must be 2D [num_tokens, num_experts] or "
            f"3D [batch, seq_len, num_experts], got shape {router_logits.shape}"
        )

    # Handle 2D input by adding batch dimension
    input_is_2d = router_logits.ndim == 2
    if input_is_2d:
        router_logits = router_logits.unsqueeze(0)  # [1, num_tokens, num_experts]

    batch_size, seq_len, num_experts = router_logits.shape

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    if not 1 <= min_k <= num_experts:
        raise ValueError(f"min_k must be in [1, {num_experts}], got {min_k}")

    if not min_k <= max_k <= num_experts:
        raise ValueError(f"max_k must be in [{min_k}, {num_experts}], got {max_k}")

    # Warn if temperature is far from 1.0
    if temperature > 2.0 or temperature < 0.5:
        warnings.warn(
            f"Unusual temperature value: {temperature}. "
            f"Typical range is [0.5, 2.0]. This may lead to unexpected behavior.",
            UserWarning
        )

    device = router_logits.device
    dtype = router_logits.dtype

    # Epsilon for numerical stability
    eps = torch.finfo(dtype).eps if dtype.is_floating_point else 1e-8

    # =========================================================================
    # Step 1: Compute Softmax Probabilities
    # =========================================================================
    # Apply temperature scaling and softmax
    # probs: [batch_size, seq_len, num_experts]
    scaled_logits = router_logits / temperature
    probs = F.softmax(scaled_logits, dim=-1, dtype=torch.float32)

    # Convert back to original dtype
    probs = probs.to(dtype)

    # Ensure probabilities are valid (sum to 1, no negatives)
    # Note: Softmax guarantees this, but we check for numerical errors
    prob_sums = probs.sum(dim=-1)
    if not torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5):
        warnings.warn(
            f"Probability sums deviate from 1.0. Max deviation: {(prob_sums - 1.0).abs().max().item():.2e}",
            RuntimeWarning
        )

    # =========================================================================
    # Step 2: Compute Pseudo P-Values
    # =========================================================================
    # Higher probability → more significant → lower p-value
    # p-value interpretation: probability of observing this expert by chance
    # p_values: [batch_size, seq_len, num_experts]
    p_values = 1.0 - probs

    # Clamp to ensure p-values are in (0, 1) for numerical stability
    p_values = torch.clamp(p_values, min=eps, max=1.0 - eps)

    # =========================================================================
    # Step 3: Sort P-Values (Ascending)
    # =========================================================================
    # sorted_pvals: [batch_size, seq_len, num_experts] - sorted in ascending order
    # sorted_indices: [batch_size, seq_len, num_experts] - original indices
    sorted_pvals, sorted_indices = torch.sort(p_values, dim=-1, descending=False)

    # =========================================================================
    # Step 4: Compute BH Critical Values
    # =========================================================================
    # For each rank k ∈ {1, 2, ..., N}, compute critical value c_k = (k/N) × α
    # ranks: [num_experts]
    ranks = torch.arange(1, num_experts + 1, device=device, dtype=torch.float32)

    # critical_values: [num_experts]
    critical_values = (ranks / num_experts) * alpha

    # Broadcast to [1, 1, num_experts] for comparison
    critical_values = critical_values.view(1, 1, -1)

    # =========================================================================
    # Step 5: Find BH Cutoff (Largest k where p_(k) ≤ c_k)
    # =========================================================================
    # significant: [batch_size, seq_len, num_experts] - True if p_(k) ≤ c_k
    significant = sorted_pvals <= critical_values

    # Find the last (rightmost) True value in each row
    # Method: Convert boolean to int, then find argmax of cumulative sum from right
    # This gives us the index of the last True value

    # First, find if ANY expert is significant
    any_significant = significant.any(dim=-1, keepdim=True)  # [B, S, 1]

    # Find the rightmost True by reversing, finding first True, then un-reversing
    # reversed_significant: [B, S, N]
    reversed_significant = torch.flip(significant, dims=[-1])

    # Find first True in reversed array (this is the last True in original)
    # argmax returns the first occurrence of max value (which is 1 for True)
    reversed_positions = torch.argmax(reversed_significant.to(torch.int32), dim=-1)  # [B, S]

    # Convert back to original positions
    # If reversed_positions = k, original position = (N - 1) - k
    num_selected = num_experts - reversed_positions  # [B, S]

    # If no expert is significant, set num_selected to 0
    # We'll enforce min_k later
    num_selected = torch.where(
        any_significant.squeeze(-1),
        num_selected,
        torch.zeros_like(num_selected)
    )

    # =========================================================================
    # Step 6: Enforce Constraints (min_k, max_k)
    # =========================================================================
    # Ensure at least min_k experts are selected
    num_selected = torch.clamp(num_selected, min=min_k, max=max_k)

    # =========================================================================
    # Step 7: Select Experts and Compute Weights
    # =========================================================================
    # For each token, select the top num_selected[token] experts
    # Since num_selected varies per token, we need to handle this carefully

    # Create a mask for selected experts in the SORTED order
    # selected_mask_sorted: [batch_size, seq_len, num_experts]
    expert_ranks = torch.arange(num_experts, device=device).view(1, 1, -1)
    expert_ranks = expert_ranks.expand(batch_size, seq_len, -1)

    # selected_mask_sorted[b, s, k] = True if k < num_selected[b, s]
    selected_mask_sorted = expert_ranks < num_selected.unsqueeze(-1)

    # Convert mask from sorted order to original order
    # We need to scatter the mask using sorted_indices
    # selected_mask: [batch_size, seq_len, num_experts]
    selected_mask = torch.zeros_like(p_values, dtype=torch.bool)
    selected_mask.scatter_(
        dim=-1,
        index=sorted_indices,
        src=selected_mask_sorted
    )

    # Extract selected probabilities and renormalize
    # routing_weights: [batch_size, seq_len, num_experts]
    routing_weights = torch.where(
        selected_mask,
        probs,
        torch.zeros_like(probs)
    )

    # Renormalize weights to sum to 1 per token
    weight_sums = routing_weights.sum(dim=-1, keepdim=True)
    # Avoid division by zero (shouldn't happen since min_k >= 1, but be safe)
    weight_sums = torch.clamp(weight_sums, min=eps)
    routing_weights = routing_weights / weight_sums

    # =========================================================================
    # Step 8: Extract Selected Expert Indices (Padded to max_k)
    # =========================================================================
    # selected_experts: [batch_size, seq_len, max_k]
    # Padding value: -1 (indicates unused slot)

    selected_experts = torch.full(
        (batch_size, seq_len, max_k),
        fill_value=-1,
        dtype=torch.long,
        device=device
    )

    # For each token, fill in the selected expert indices
    # Method: Take the first max_k indices from sorted_indices where mask is True

    for k_idx in range(max_k):
        # Check if this slot should be filled (k_idx < num_selected)
        slot_active = k_idx < num_selected  # [B, S]

        # Get the k_idx-th selected expert (in sorted order)
        # This is sorted_indices[:, :, k_idx] where selected_mask_sorted[:, :, k_idx] is True
        # Since selected_mask_sorted[:, :, k_idx] is True iff k_idx < num_selected,
        # we can directly take sorted_indices[:, :, k_idx]

        expert_idx = sorted_indices[:, :, k_idx]  # [B, S]

        # Only assign if slot is active
        selected_experts[:, :, k_idx] = torch.where(
            slot_active,
            expert_idx,
            torch.full_like(expert_idx, fill_value=-1)
        )

    # =========================================================================
    # Step 9: Prepare Outputs
    # =========================================================================
    # If input was 2D, remove batch dimension from outputs
    if input_is_2d:
        routing_weights = routing_weights.squeeze(0)  # [seq_len, num_experts]
        selected_experts = selected_experts.squeeze(0)  # [seq_len, max_k]
        num_selected = num_selected.squeeze(0)  # [seq_len]

    # Prepare statistics if requested
    if return_stats:
        # Compute BH threshold used for each token
        # bh_threshold[b, s] = critical_values[num_selected[b, s] - 1]
        # Need to gather the critical value at position num_selected - 1

        if input_is_2d:
            # Add back batch dimension for indexing
            num_selected_3d = num_selected.unsqueeze(0)
        else:
            num_selected_3d = num_selected

        # Clamp indices to valid range [0, num_experts - 1]
        gather_indices = torch.clamp(num_selected_3d - 1, min=0, max=num_experts - 1)
        gather_indices = gather_indices.unsqueeze(-1)  # [B, S, 1]

        # critical_values is [1, 1, N], expand to [B, S, N]
        critical_values_expanded = critical_values.expand(batch_size, seq_len, -1)

        # Gather the threshold
        bh_threshold = torch.gather(critical_values_expanded, dim=-1, index=gather_indices)
        bh_threshold = bh_threshold.squeeze(-1)  # [B, S]

        if input_is_2d:
            bh_threshold = bh_threshold.squeeze(0)
            p_values = p_values.squeeze(0)
            selected_mask = selected_mask.squeeze(0)

        stats = {
            'p_values': p_values,
            'bh_threshold': bh_threshold,
            'significant_mask': selected_mask,
            'alpha': alpha,
            'temperature': temperature
        }

        return routing_weights, selected_experts, num_selected, stats

    return routing_weights, selected_experts, num_selected


def topk_routing(
    router_logits: torch.Tensor,
    k: int = 8,  # Now explicitly parameterized with default
    temperature: float = 1.0,
    normalize: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standard top-k routing for comparison with BH routing.

    Args:
        router_logits: Router logits [batch, seq_len, num_experts] or [num_tokens, num_experts]
        k: Number of experts to select (default: 8)
        temperature: Softmax temperature (default: 1.0)
        normalize: Whether to renormalize selected weights (default: True)

    Returns:
        routing_weights: Sparse routing weights [batch, seq_len, num_experts]
        selected_experts: Selected expert indices [batch, seq_len, k]
        expert_counts: Number of experts per token [batch, seq_len] (always k)
    """
    input_is_2d = router_logits.ndim == 2
    if input_is_2d:
        router_logits = router_logits.unsqueeze(0)

    batch_size, seq_len, num_experts = router_logits.shape

    if k > num_experts:
        raise ValueError(f"k={k} cannot exceed num_experts={num_experts}")

    # Compute softmax probabilities
    scaled_logits = router_logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)

    # Select top-k
    topk_weights, topk_indices = torch.topk(probs, k, dim=-1)

    # Renormalize if requested
    if normalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    # Create sparse weight tensor
    routing_weights = torch.zeros_like(probs)
    routing_weights.scatter_(dim=-1, index=topk_indices, src=topk_weights)

    # Expert counts (constant k for all tokens)
    expert_counts = torch.full(
        (batch_size, seq_len),
        fill_value=k,
        dtype=torch.long,
        device=router_logits.device
    )

    if input_is_2d:
        routing_weights = routing_weights.squeeze(0)
        topk_indices = topk_indices.squeeze(0)
        expert_counts = expert_counts.squeeze(0)

    return routing_weights, topk_indices, expert_counts


# ============================================================================
# Utility Functions
# ============================================================================

def compute_routing_statistics(
    routing_weights: torch.Tensor,
    expert_counts: torch.Tensor
) -> Dict[str, float]:
    """
    Compute statistics about routing decisions.

    Args:
        routing_weights: Routing weights [batch, seq_len, num_experts]
        expert_counts: Number of experts selected per token [batch, seq_len]

    Returns:
        Dictionary with statistics:
            - mean_experts: Average experts per token
            - std_experts: Std dev of experts per token
            - min_experts: Minimum experts selected
            - max_experts: Maximum experts selected
            - sparsity: Fraction of zero weights
            - weight_entropy: Average entropy of weight distribution
    """
    stats = {}

    # Expert count statistics
    stats['mean_experts'] = expert_counts.float().mean().item()
    stats['std_experts'] = expert_counts.float().std().item()
    stats['min_experts'] = expert_counts.min().item()
    stats['max_experts'] = expert_counts.max().item()

    # Sparsity (fraction of zero weights)
    total_weights = routing_weights.numel()
    zero_weights = (routing_weights == 0).sum().item()
    stats['sparsity'] = zero_weights / total_weights

    # Weight entropy (measure of concentration)
    # H = -sum(p * log(p)) for p > 0
    eps = 1e-10
    nonzero_weights = routing_weights + eps
    entropy = -(routing_weights * torch.log(nonzero_weights)).sum(dim=-1)
    stats['weight_entropy'] = entropy.mean().item()

    return stats


def run_bh_multi_k(
    router_logits: torch.Tensor,
    max_k_values: list = None,
    alpha: float = 0.05,
    temperature: float = 1.0,
    min_k: int = 1
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Run BH routing with multiple max_k values for comparison.

    Args:
        router_logits: [batch, seq_len, num_experts] or [num_tokens, num_experts]
        max_k_values: List of maximum expert counts to test (default: [8, 16, 32, 64])
        alpha: FDR control level
        temperature: Softmax temperature
        min_k: Minimum experts to select

    Returns:
        Dict mapping max_k to (routing_weights, selected_experts, expert_counts)

    Example:
        >>> results = run_bh_multi_k(logits, max_k_values=[8, 16, 32, 64])
        >>> for k, (weights, experts, counts) in results.items():
        ...     print(f"max_k={k}: avg experts = {counts.float().mean():.2f}")
    """
    if max_k_values is None:
        max_k_values = [8, 16, 32, 64]

    results = {}
    for max_k in max_k_values:
        weights, experts, counts = benjamini_hochberg_routing(
            router_logits,
            alpha=alpha,
            temperature=temperature,
            min_k=min_k,
            max_k=max_k
        )
        results[max_k] = (weights, experts, counts)
    return results


def compare_multi_k_statistics(
    results: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> 'pd.DataFrame':
    """
    Generate comparison statistics for different max_k values.

    Args:
        results: Output from run_bh_multi_k()

    Returns:
        DataFrame with columns:
            - max_k
            - mean_experts
            - std_experts
            - min_experts
            - max_experts
            - pct_at_ceiling (% hitting max_k limit)
            - pct_at_floor (% at min_k)
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for compare_multi_k_statistics. Install with: pip install pandas")

    rows = []
    for max_k, (weights, experts, counts) in results.items():
        counts_flat = counts.flatten().float()
        rows.append({
            'max_k': max_k,
            'mean_experts': counts_flat.mean().item(),
            'std_experts': counts_flat.std().item(),
            'min_experts': counts_flat.min().item(),
            'max_experts': counts_flat.max().item(),
            'pct_at_ceiling': (counts_flat == max_k).float().mean().item() * 100,
            'pct_at_floor': (counts_flat == 1).float().mean().item() * 100,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Quick demonstration
    print("BH Routing Demonstration")
    print("=" * 70)

    # Create sample router logits
    torch.manual_seed(42)
    batch_size, seq_len, num_experts = 2, 4, 16
    router_logits = torch.randn(batch_size, seq_len, num_experts)

    print(f"\nInput shape: {router_logits.shape}")
    print(f"Testing with alpha=0.05, temperature=1.0, max_k=8\n")

    # Run BH routing
    weights, experts, counts, stats = benjamini_hochberg_routing(
        router_logits,
        alpha=0.05,
        temperature=1.0,
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

    print(f"\nBH statistics:")
    print(f"  Mean p-value: {stats['p_values'].mean().item():.4f}")
    print(f"  Mean BH threshold: {stats['bh_threshold'].mean().item():.4f}")

    # Compare with top-k
    print(f"\n" + "=" * 70)
    print("Comparison with Top-K routing (k=8)")
    print("=" * 70)

    topk_weights, topk_experts, topk_counts = topk_routing(
        router_logits, k=8, temperature=1.0
    )

    topk_stats = compute_routing_statistics(topk_weights, topk_counts)
    print(f"\nTop-K statistics:")
    for key, value in topk_stats.items():
        print(f"  {key}: {value:.4f}")

    print(f"\nDifference in mean experts: {routing_stats['mean_experts'] - topk_stats['mean_experts']:.2f}")
    print("(Negative = BH uses fewer experts on average)")

    # =========================================================================
    # NEW: Multi-K Demonstration
    # =========================================================================
    print(f"\n" + "=" * 70)
    print("Multi-K BH Routing Comparison")
    print("=" * 70)

    results = run_bh_multi_k(
        router_logits,
        max_k_values=[8, 16, 32, 64],
        alpha=0.05
    )

    comparison_df = compare_multi_k_statistics(results)
    print("\nComparison across max_k values:")
    print(comparison_df.to_string(index=False))

    print("\nKey Insights:")
    print("  - Higher max_k allows more experts when needed")
    print("  - pct_at_ceiling shows how often BH hits the max_k limit")
    print("  - pct_at_floor shows how conservative BH is being")
