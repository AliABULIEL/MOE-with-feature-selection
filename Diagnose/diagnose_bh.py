#!/usr/bin/env python3
"""
Diagnostic script to understand BH routing behavior.

This script shows what's happening inside the BH algorithm,
specifically checking if the ceiling (max_k) is being hit.
"""

import torch
import sys
import os
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bh_routing import benjamini_hochberg_routing

def diagnose_bh_algorithm():
    """Diagnose BH algorithm behavior with different alpha and max_k values."""

    print("=" * 70)
    print("BH ALGORITHM DIAGNOSTIC")
    print("=" * 70)

    # Create test logits
    torch.manual_seed(42)
    num_tokens = 10  # Just 10 tokens for easier inspection
    num_experts = 64

    router_logits = torch.randn(num_tokens, num_experts)
    router_logits[:, :8] += 2.0  # Boost top 8
    router_logits[:, :3] += 1.0  # Boost top 3 even more

    print(f"\nTest setup: {num_tokens} tokens, {num_experts} experts")
    print("Top 8 experts have boosted logits to be clearly better\n")

    # Test 1: Show what happens without ceiling constraints
    print("=" * 70)
    print("TEST 1: HIGH max_k (no ceiling constraint)")
    print("=" * 70)

    for alpha in [0.01, 0.05, 0.10, 0.20]:
        _, selected_experts, expert_counts = benjamini_hochberg_routing(
            router_logits,
            alpha=alpha,
            temperature=1.0,
            min_k=1,
            max_k=64  # Very high ceiling
        )

        avg = expert_counts.float().mean().item()
        min_exp = expert_counts.min().item()
        max_exp = expert_counts.max().item()

        print(f"\nα={alpha:4.2f} (max_k=64):")
        print(f"  Avg: {avg:5.2f}  Range: [{min_exp:2d}, {max_exp:2d}]")
        print(f"  Per token: {expert_counts.tolist()}")

    # Test 2: Show ceiling effect
    print("\n\n" + "=" * 70)
    print("TEST 2: LOW max_k (ceiling constraint active)")
    print("=" * 70)

    for alpha in [0.01, 0.05, 0.10, 0.20]:
        _, selected_experts, expert_counts = benjamini_hochberg_routing(
            router_logits,
            alpha=alpha,
            temperature=1.0,
            min_k=1,
            max_k=8  # Standard OLMoE ceiling
        )

        avg = expert_counts.float().mean().item()
        min_exp = expert_counts.min().item()
        max_exp = expert_counts.max().item()
        ceiling_hits = (expert_counts == 8).sum().item()

        print(f"\nα={alpha:4.2f} (max_k=8):")
        print(f"  Avg: {avg:5.2f}  Range: [{min_exp:2d}, {max_exp:2d}]")
        print(f"  Ceiling hits: {ceiling_hits}/{num_tokens} tokens hit max_k")
        print(f"  Per token: {expert_counts.tolist()}")

    # Test 3: Manual p-value inspection for one token
    print("\n\n" + "=" * 70)
    print("TEST 3: P-VALUE INSPECTION (Single Token)")
    print("=" * 70)

    single_logits = router_logits[0:1]  # Just first token
    print(f"\nInspecting first token...")

    # Manually compute p-values to see what's happening
    temperature = 1.0
    alpha = 0.05

    scaled_logits = single_logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)

    # Compute p-values using the corrected algorithm
    min_logits = scaled_logits.min(dim=-1, keepdim=True).values
    normalized_logits = scaled_logits - min_logits
    p_values_unnorm = torch.exp(-normalized_logits)
    p_values = p_values_unnorm / p_values_unnorm.sum(dim=-1, keepdim=True)

    # Sort p-values
    sorted_p, sorted_indices = torch.sort(p_values[0])

    print(f"\nTop 10 experts by p-value (lowest = most significant):")
    print(f"{'Rank':<6} {'Expert':<8} {'P-value':<12} {'Logit':<10} {'Prob':<10} {'BH Threshold':<15}")
    print("-" * 70)

    N = num_experts
    for rank in range(10):
        expert_idx = sorted_indices[rank].item()
        p_val = sorted_p[rank].item()
        logit = scaled_logits[0, expert_idx].item()
        prob = probs[0, expert_idx].item()
        threshold = ((rank + 1) / N) * alpha

        passes = "✅" if p_val <= threshold else "❌"

        print(f"{rank+1:<6} {expert_idx:<8} {p_val:<12.6f} {logit:<10.4f} {prob:<10.6f} {threshold:<15.6f} {passes}")

    # Count how many would pass
    k_max = 0
    for k in range(N):
        threshold = ((k + 1) / N) * alpha
        if sorted_p[k] <= threshold:
            k_max = k + 1

    print(f"\n📊 BH procedure would select: {k_max} experts (α={alpha})")
    print(f"   With max_k=8 constraint: {min(k_max, 8)} experts actually selected")

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)

    # Interpretation
    print("\nINTERPRETATION:")
    print("If higher alpha values all show max_k ceiling hits,")
    print("this means BH wants to select MORE than max_k experts,")
    print("which is CORRECT behavior (BH is being constrained by ceiling).")
    print("\nThis is actually GOOD - it means:")
    print("  ✅ P-value computation is working correctly")
    print("  ✅ BH threshold is working as expected")
    print("  ✅ max_k ceiling is properly constraining selection")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    diagnose_bh_algorithm()
