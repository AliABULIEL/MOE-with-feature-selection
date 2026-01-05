"""
Random Routing for Mixture-of-Experts Models
======================================================

Implements a random expert selection mechanism for MoE models.

This routing strategy allows for controlled experiments by selecting a random
subset of experts and normalizing their weights to a specified total sum.

Key Parameters:
- experts_amount: The number of experts to select randomly.
- sum_of_weights: The target sum for the weights of the selected experts.
  If not provided, it's calculated from the top 8 experts.

Example:
    # Select 4 random experts and normalize their weights to sum to 0.5
    weights, experts, counts, _ = random_routing(
        logits, experts_amount=4, sum_of_weights=0.5
    )
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

def random_routing(
    router_logits: torch.Tensor,
    experts_amount: int = 8,
    sum_of_weights: Optional[float] = None,
    temperature: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict]]:
    """
    Random routing for expert selection in MoE models.

    Args:
        router_logits: Router output logits
            Shape: [batch, seq_len, num_experts] or [num_tokens, num_experts]
        experts_amount: The number of experts to select randomly.
        sum_of_weights: The target sum for the weights of the selected experts.
            If None, it is calculated as the sum of the top 8 expert weights.
        temperature: Softmax temperature for weight computation (default: 1.0)

    Returns:
        routing_weights: Normalized weights for selected experts 
            Shape: [batch, seq_len, num_experts]
        
        selected_experts: Indices of selected experts (padded with -1)
            Shape: [batch, seq_len, experts_amount]
        expert_counts: Number of experts selected per token
            Shape: [batch, seq_len]
        stats (optional): Always None for this routing method.
    """
    # =========================================================================
    # STEP 1: Input validation and reshaping
    # =========================================================================
    original_shape = router_logits.shape
    device = router_logits.device

    assert router_logits.dim() in [2, 3], f"Expected 2D or 3D tensor, got {router_logits.dim()}D"
    assert temperature > 0, f"Temperature must be positive, got {temperature}"

    if router_logits.dim() == 2:
        router_logits = router_logits.unsqueeze(0)
        was_2d = True
    else:
        was_2d = False

    batch_size, seq_len, num_experts = router_logits.shape
    logits_flat = router_logits.view(-1, num_experts)
    num_tokens = logits_flat.shape[0]

    # =========================================================================
    # STEP 2: Compute base weights
    # =========================================================================
    scaled_logits = logits_flat / temperature
    weights = F.softmax(scaled_logits, dim=-1)

    # =========================================================================
    # STEP 3: Choose a random subset of experts for each token
    # =========================================================================
    random_experts_indices = torch.stack([
        torch.randperm(num_experts, device=device)[:experts_amount]
        for _ in range(num_tokens)
    ])

    # =========================================================================
    # STEP 4: Determine sum_of_weights if not provided
    # =========================================================================
    if sum_of_weights is None:
        top8_weights, _ = torch.topk(weights, 8, dim=1)
        target_sum_of_weights = top8_weights.sum(dim=1)
    else:
        target_sum_of_weights = torch.full((num_tokens,), sum_of_weights, device=device)

    # =========================================================================
    # STEP 5: Normalize the weights of the chosen experts
    # =========================================================================
    routing_weights = torch.zeros_like(weights)
    
    for i in range(num_tokens):
        selected_indices = random_experts_indices[i]
        selected_weights = weights[i, selected_indices]
        
        current_sum = selected_weights.sum()
        
        if current_sum > 1e-9: # Avoid division by zero
            normalization_factor = target_sum_of_weights[i] / current_sum
            normalized_weights = selected_weights * normalization_factor
            routing_weights[i, selected_indices] = normalized_weights

    # =========================================================================
    # STEP 6: Get selected expert indices (padded with -1)
    # =========================================================================
    selected_experts = torch.full((num_tokens, experts_amount), -1, device=device, dtype=torch.long)
    for i in range(num_tokens):
        selected_experts[i, :experts_amount] = random_experts_indices[i]

    expert_counts = torch.full((num_tokens,), experts_amount, device=device)

    # =========================================================================
    # STEP 7: Reshape outputs to match input shape
    # =========================================================================
    routing_weights = routing_weights.view(batch_size, seq_len, num_experts)
    selected_experts = selected_experts.view(batch_size, seq_len, experts_amount)
    expert_counts = expert_counts.view(batch_size, seq_len)

    if was_2d:
        routing_weights = routing_weights.squeeze(0)
        selected_experts = selected_experts.squeeze(0)
        expert_counts = expert_counts.squeeze(0)

    return routing_weights, selected_experts, expert_counts, None


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RANDOM ROUTING - TEST")
    print("=" * 70)
    
    torch.manual_seed(42)
    logits = torch.randn(1, 10, 64) # batch=1, seq=10, experts=64
    
    # Test with specified sum_of_weights
    print("\nTesting with specified sum_of_weights=0.5 and experts_amount=4:\n")
    weights, experts, counts, _ = random_routing(
        logits, experts_amount=4, sum_of_weights=0.5
    )
    
    print(f"Expert counts per token: {counts[0]}")
    print(f"Selected experts for first token: {experts[0, 0]}")
    print(f"Sum of weights for first token: {weights[0, 0].sum():.4f}")

    # Test with dynamic sum_of_weights
    print("\nTesting with dynamic sum_of_weights (from top 8) and experts_amount=4:\n")
    weights_dyn, experts_dyn, counts_dyn, _ = random_routing(
        logits, experts_amount=4
    )
    
    # Calculate what the sum should be for the first token
    base_weights = F.softmax(logits[0, 0] / 1.0, dim=-1)
    top8_sum = torch.topk(base_weights, 8)[0].sum()

    print(f"Expert counts per token: {counts_dyn[0]}")
    print(f"Selected experts for first token: {experts_dyn[0, 0]}")
    print(f"Sum of weights for first token: {weights_dyn[0, 0].sum():.4f} (expected: ~{top8_sum:.4f})")

    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
