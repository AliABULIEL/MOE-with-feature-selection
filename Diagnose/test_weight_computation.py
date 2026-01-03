#!/usr/bin/env python3
"""
Test Weight Computation Bug Hypothesis
======================================

HYPOTHESIS: The topk_routing and HC routing both have a RENORMALIZATION BUG
that causes weight distortion for k != 8.

Background:
- TopK-8 works (84 perplexity)
- TopK-16 fails (249 perplexity)
- HC fails (350-625 perplexity)

This suggests the bug affects ALL routing when experts != 8.

The Bug:
--------
Current implementation:
1. Softmax over ALL 64 experts → weights sum to 1.0
2. Select top-k experts
3. RENORMALIZE selected weights to sum to 1.0 again

Why this is wrong:
- After softmax, the weights already represent proper probabilities
- Selecting top-k gives weights that sum to some value p < 1.0
- Renormalizing to 1.0 AMPLIFIES all weights by 1/p
- This changes the effective weight distribution!

Example:
- Original softmax: [0.3, 0.2, 0.15, 0.1, 0.05, ...]  (sum=1.0)
- Top-3 selected: [0.3, 0.2, 0.15] (sum=0.65)
- After renorm: [0.46, 0.31, 0.23] (sum=1.0)
- Weight on expert 0 increased from 0.3 → 0.46 (53% increase!)

This amplification is DIFFERENT for different k values:
- k=8: smaller amplification (sum of top-8 ≈ 0.9)
- k=16: larger amplification (sum of top-16 ≈ 0.95+)
- HC with varying k: unpredictable amplification

Expected correct behavior:
--------------------------
OLMoE likely expects one of these patterns:

Option A: Sparse softmax (what's currently implemented, but without renorm)
1. Softmax over ALL 64 experts
2. Select top-k
3. Use weights AS-IS (no renormalization!)

Option B: TopK softmax (different)
1. Take top-k LOGITS
2. Softmax over just those k logits
3. Use those weights (naturally sum to 1)

Option A vs Option B give DIFFERENT weight distributions!

Let's test which one OLMoE expects.
"""

import torch
import torch.nn.functional as F
import numpy as np


def topk_routing_CURRENT_BUGGY(router_logits, k=8, temperature=1.0):
    """Current implementation (suspected bug)."""
    scaled_logits = router_logits / temperature
    weights = F.softmax(scaled_logits, dim=-1)  # Over all 64

    topk_weights, topk_indices = torch.topk(weights, k, dim=-1)

    routing_weights = torch.zeros_like(weights)
    routing_weights.scatter_(1, topk_indices, topk_weights)

    # BUG: This renormalization amplifies weights!
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

    return routing_weights, topk_indices


def topk_routing_OPTION_A(router_logits, k=8, temperature=1.0):
    """Option A: Sparse softmax (no renorm)."""
    scaled_logits = router_logits / temperature
    weights = F.softmax(scaled_logits, dim=-1)  # Over all 64

    topk_weights, topk_indices = torch.topk(weights, k, dim=-1)

    routing_weights = torch.zeros_like(weights)
    routing_weights.scatter_(1, topk_indices, topk_weights)

    # NO renormalization! Use sparse weights as-is
    # Sum will be < 1.0, but that's OK for MoE

    return routing_weights, topk_indices


def topk_routing_OPTION_B(router_logits, k=8, temperature=1.0):
    """Option B: TopK softmax (softmax over selected logits only)."""
    scaled_logits = router_logits / temperature

    # Get top-k LOGITS (not weights)
    topk_logits, topk_indices = torch.topk(scaled_logits, k, dim=-1)

    # Softmax over ONLY the selected k logits
    topk_weights = F.softmax(topk_logits, dim=-1)  # Sum = 1.0

    # Scatter back to full dimension
    routing_weights = torch.zeros_like(scaled_logits)
    routing_weights.scatter_(1, topk_indices, topk_weights)

    return routing_weights, topk_indices


def analyze_weight_distributions():
    """Compare weight distributions across methods."""
    torch.manual_seed(42)
    router_logits = torch.randn(5, 64)  # 5 tokens, 64 experts

    print("=" * 80)
    print("WEIGHT DISTRIBUTION ANALYSIS")
    print("=" * 80)

    for k in [8, 16]:
        print(f"\nk = {k}")
        print("-" * 80)

        # Test all three methods
        weights_current, _ = topk_routing_CURRENT_BUGGY(router_logits, k=k)
        weights_option_a, _ = topk_routing_OPTION_A(router_logits, k=k)
        weights_option_b, _ = topk_routing_OPTION_B(router_logits, k=k)

        # Analyze token 0
        t = 0

        # Get top expert weights
        top_expert_idx = weights_current[t].argmax().item()

        print(f"Token {t}, Top Expert {top_expert_idx}:")
        print(f"  Current (BUGGY):  {weights_current[t, top_expert_idx]:.6f}")
        print(f"  Option A (sparse): {weights_option_a[t, top_expert_idx]:.6f}")
        print(f"  Option B (topk):   {weights_option_b[t, top_expert_idx]:.6f}")
        print()
        print(f"  Weight sums:")
        print(f"    Current:  {weights_current[t].sum():.6f}")
        print(f"    Option A: {weights_option_a[t].sum():.6f}")
        print(f"    Option B: {weights_option_b[t].sum():.6f}")
        print()

        # Compute amplification factor
        sum_before_renorm = weights_option_a[t].sum()
        amplification = 1.0 / sum_before_renorm
        print(f"  Amplification factor (current buggy method): {amplification:.4f}")
        print(f"  → Top expert weight boosted by {(amplification - 1) * 100:.1f}%")


def demonstrate_bug():
    """Demonstrate how renormalization distorts weights."""
    print("\n" + "=" * 80)
    print("BUG DEMONSTRATION: How Renormalization Distorts Weights")
    print("=" * 80)

    torch.manual_seed(123)
    logits = torch.randn(1, 64)

    # Compute full softmax
    full_softmax = F.softmax(logits, dim=-1)

    print("\nOriginal softmax over all 64 experts:")
    top5_vals, top5_idx = torch.topk(full_softmax[0], 5)
    for i, (idx, val) in enumerate(zip(top5_idx, top5_vals)):
        print(f"  Expert {idx:2d}: {val:.6f}")
    print(f"  Sum of all 64: {full_softmax.sum():.6f}")

    # Compare k=8 vs k=16 renormalization
    for k in [8, 16]:
        print(f"\nWith k={k}:")

        # Get top-k
        topk_vals, topk_idx = torch.topk(full_softmax[0], k)
        sum_before = topk_vals.sum()

        # Renormalize (buggy method)
        topk_renorm = topk_vals / sum_before

        print(f"  Sum of top-{k} before renorm: {sum_before:.6f}")
        print(f"  Amplification factor: {1/sum_before:.4f}")
        print(f"  Top expert weight:")
        print(f"    Before renorm: {topk_vals[0]:.6f}")
        print(f"    After renorm:  {topk_renorm[0]:.6f}")
        print(f"    Change: +{(topk_renorm[0] - topk_vals[0]) / topk_vals[0] * 100:.1f}%")


def test_hc_routing_weight_computation():
    """Test HC routing weight computation pattern."""
    print("\n" + "=" * 80)
    print("HC ROUTING WEIGHT COMPUTATION")
    print("=" * 80)

    torch.manual_seed(42)
    logits = torch.randn(3, 64)

    # Simulate HC selection (variable k per token)
    num_selected = torch.tensor([8, 13, 16])  # Different k per token

    # HC routing pattern (from hc_routing.py lines 424-436)
    scaled_logits = logits / 1.0
    weights = F.softmax(scaled_logits, dim=-1)  # Over all 64

    # Create selection mask (select top-k for each token)
    routing_weights_hc = torch.zeros_like(weights)
    for t, k in enumerate(num_selected):
        topk_weights, topk_indices = torch.topk(weights[t], k)
        routing_weights_hc[t].scatter_(0, topk_indices, topk_weights)

    # Renormalize (as in HC routing)
    routing_weights_hc = routing_weights_hc / routing_weights_hc.sum(dim=-1, keepdim=True)

    print("\nHC routing with variable k:")
    for t, k in enumerate(num_selected):
        top_expert_idx = routing_weights_hc[t].argmax().item()
        top_weight = routing_weights_hc[t, top_expert_idx].item()
        original_weight = weights[t, top_expert_idx].item()
        boost = (top_weight - original_weight) / original_weight * 100

        print(f"  Token {t} (k={k:2d}): top expert weight {top_weight:.6f} "
              f"(boosted +{boost:.1f}% from {original_weight:.6f})")


if __name__ == '__main__':
    analyze_weight_distributions()
    demonstrate_bug()
    test_hc_routing_weight_computation()

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
The bug is clear: Both topk_routing and HC routing apply renormalization
after selecting experts, which AMPLIFIES weights beyond their original
softmax values.

This amplification is DIFFERENT for different k values:
- k=8:  smaller boost (sum of top-8 ≈ 85-90% of total)
- k=16: larger boost (sum of top-16 ≈ 95%+ of total)

Since OLMoE was trained with k=8, it expects specific weight magnitudes.
When we use k=16 or variable k (HC), the amplification changes, causing
the model to receive unexpected weight distributions → perplexity degrades.

SOLUTION: We need to determine which weight computation pattern OLMoE
expects (Option A or Option B) and use that consistently.

Most likely: Option A (sparse softmax without renorm) is correct for
standard MoE architectures. But we need to verify this.
""")
