#!/usr/bin/env python3
"""
Test script to verify BH routing algorithm fix.

This script tests that the corrected p-value computation in benjamini_hochberg_routing()
produces variable expert counts based on alpha values.

Expected behavior after fix:
- α=0.01 (strict):   ~2-4 experts per token
- α=0.05 (moderate): ~4-6 experts per token
- α=0.10 (loose):    ~5-7 experts per token
- α=0.20 (very loose): ~6-8 experts per token

Before fix (BROKEN):
- All alpha values: exactly 1 expert per token
"""

import torch
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deprecated.bh_routing import benjamini_hochberg_routing

def test_bh_routing_fix():
    """Test that BH routing produces variable expert counts based on alpha."""

    print("=" * 70)
    print("BH ROUTING ALGORITHM FIX VERIFICATION")
    print("=" * 70)

    print("\nTesting corrected p-value computation...")
    print("Checking if expert counts vary with alpha parameter.\n")

    # Create realistic router logits (simulating 64 experts for OLMoE)
    torch.manual_seed(42)
    num_tokens = 100
    num_experts = 64

    # Generate logits with realistic distribution
    # Top experts have higher logits, rest are lower
    router_logits = torch.randn(num_tokens, num_experts)
    # Make some experts clearly better (will have low p-values)
    router_logits[:, :8] += 2.0  # Top 8 experts get boost
    router_logits[:, :3] += 1.0  # Top 3 get even more boost

    alpha_configs = [
        (0.01, "strict", 2, 4),
        (0.05, "moderate", 4, 6),
        (0.10, "loose", 5, 7),
        (0.20, "very loose", 6, 8)
    ]

    results = []
    all_pass = True

    print(f"Router logits shape: {router_logits.shape}")
    print(f"Testing {len(alpha_configs)} configurations...\n")
    print("-" * 70)

    for alpha, desc, min_expected, max_expected in alpha_configs:
        # Run BH routing
        routing_weights, selected_experts, expert_counts = benjamini_hochberg_routing(
            router_logits,
            alpha=alpha,
            temperature=1.0,
            min_k=1,
            max_k=8
        )

        # Compute statistics
        avg_experts = expert_counts.float().mean().item()
        std_experts = expert_counts.float().std().item()
        min_experts = expert_counts.min().item()
        max_experts = expert_counts.max().item()

        # Check if within expected range
        in_range = min_expected <= avg_experts <= max_expected

        # Print results
        status = "✅ PASS" if in_range else "❌ FAIL"
        print(f"\nα={alpha:4.2f} ({desc:12s}): {status}")
        print(f"  Avg experts: {avg_experts:.2f}  (expected: {min_expected}-{max_expected})")
        print(f"  Range: [{min_experts}, {max_experts}]")
        print(f"  Std dev: {std_experts:.2f}")

        # Additional checks
        if avg_experts <= 1.1:
            print(f"  ⚠️  WARNING: Still selecting ~1 expert (BUG NOT FIXED!)")
            all_pass = False
        elif not in_range:
            print(f"  ⚠️  WARNING: Outside expected range")
            all_pass = False

        results.append({
            'alpha': alpha,
            'desc': desc,
            'avg_experts': avg_experts,
            'in_range': in_range
        })

    print("\n" + "-" * 70)

    # Additional test: alpha effect (higher alpha should select more experts)
    print("\nTesting alpha sensitivity (higher α should select more experts):")
    for i in range(len(results) - 1):
        curr = results[i]
        next_result = results[i + 1]

        if next_result['avg_experts'] > curr['avg_experts']:
            print(f"  ✅ α={next_result['alpha']} > α={curr['alpha']}: "
                  f"{next_result['avg_experts']:.2f} > {curr['avg_experts']:.2f}")
        else:
            print(f"  ⚠️  α={next_result['alpha']} ≤ α={curr['alpha']}: "
                  f"{next_result['avg_experts']:.2f} ≤ {curr['avg_experts']:.2f}")
            all_pass = False

    # Check variability (not fixed 1 expert)
    print("\nTesting expert count variability:")
    first_config_avg = results[0]['avg_experts']
    if first_config_avg > 1.5:
        print(f"  ✅ Expert counts are VARIABLE ({first_config_avg:.2f} experts, not fixed 1)")
    else:
        print(f"  ❌ Expert counts are FIXED (~1 expert) - BUG STILL EXISTS!")
        all_pass = False

    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    if all_pass:
        print("\n🎉 SUCCESS! BH routing algorithm is FIXED!")
        print("\n✅ P-value computation is working correctly")
        print("✅ Expert counts vary based on alpha parameter")
        print("✅ Higher alpha selects more experts as expected")
        print("\n🎯 The BH routing algorithm is ready to use in experiments!")
    else:
        print("\n❌ FAILURE! BH routing algorithm still has issues!")
        print("\n⚠️  The p-value computation may still be incorrect")
        print("⚠️  Expert counts are not varying as expected")
        print("\nPlease review the benjamini_hochberg_routing() function in bh_routing.py")
        print("Specifically check the p-value computation section.")

    print("\n" + "=" * 70)

    return all_pass


if __name__ == "__main__":
    success = test_bh_routing_fix()
    sys.exit(0 if success else 1)
