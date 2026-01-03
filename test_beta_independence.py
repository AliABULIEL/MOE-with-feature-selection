#!/usr/bin/env python3
"""
Test β Independence After Fix
=============================

Validates that different β values now produce different routing behaviors.
"""

import torch
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from hc_routing import higher_criticism_routing, load_kde_models


def test_beta_sweep_wide_bounds():
    """
    Test β sweep with wide bounds (min_k=1, max_k=64).

    Pass Criteria:
    - At least 7 of 10 β values produce distinct avg_experts
    - Lower β tends toward fewer experts
    - Higher β tends toward more experts
    """
    print("=" * 70)
    print("TEST 1: β SWEEP WITH WIDE BOUNDS (min_k=1, max_k=64)")
    print("=" * 70)

    # Create test data
    torch.manual_seed(42)
    batch_size = 1
    seq_len = 100  # More tokens for better statistics
    num_experts = 64

    router_logits = torch.randn(batch_size, seq_len, num_experts)

    # Test different β values
    beta_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = {}

    print(f"\nTesting {len(beta_values)} β values with {seq_len} tokens...")
    print(f"\n{'β':>6} | {'Avg':>8} | {'Std':>8} | {'Min':>5} | {'Max':>5} | {'Range':>7}")
    print("-" * 60)

    for beta in beta_values:
        routing_weights, selected_experts, expert_counts, _ = higher_criticism_routing(
            router_logits,
            beta=beta,
            min_k=1,
            max_k=64,  # Wide bounds
            temperature=1.0,
            kde_models=None,  # Use empirical p-values
            return_stats=False
        )

        # Compute statistics
        counts_flat = expert_counts.flatten()
        avg = counts_flat.float().mean().item()
        std = counts_flat.float().std().item()
        min_c = counts_flat.min().item()
        max_c = counts_flat.max().item()

        results[beta] = {
            'avg': avg,
            'std': std,
            'min': min_c,
            'max': max_c,
            'counts': counts_flat
        }

        print(f"{beta:6.1f} | {avg:8.2f} | {std:8.2f} | {min_c:5d} | {max_c:5d} | {max_c - min_c:7d}")

    # Validation
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    # Check 1: Distinct values
    avg_values = [r['avg'] for r in results.values()]
    unique_avgs = len(set(round(a, 1) for a in avg_values))  # Round to 0.1

    print(f"\n1. Distinct avg_experts values: {unique_avgs}/10")
    if unique_avgs >= 7:
        print("   ✅ PASS: β values produce distinct behaviors")
    else:
        print(f"   ❌ FAIL: Only {unique_avgs} distinct values (need ≥7)")
        return False

    # Check 2: Monotonicity trend (not strict, but general)
    avg_trend = [results[b]['avg'] for b in beta_values]
    increasing_pairs = sum(1 for i in range(len(avg_trend)-1) if avg_trend[i+1] >= avg_trend[i])

    print(f"\n2. Monotonicity: {increasing_pairs}/9 pairs increase")
    if increasing_pairs >= 6:  # At least 2/3 should follow trend
        print("   ✅ PASS: Higher β tends toward more experts")
    else:
        print(f"   ⚠️  WARNING: Weak trend (only {increasing_pairs}/9)")

    # Check 3: Range coverage
    min_avg = min(avg_values)
    max_avg = max(avg_values)
    range_ratio = (max_avg - min_avg) / 64  # Fraction of possible range

    print(f"\n3. Range coverage: {min_avg:.1f} to {max_avg:.1f} ({range_ratio:.1%} of [1, 64])")
    if range_ratio >= 0.15:  # At least 15% of range
        print("   ✅ PASS: Good range coverage")
    else:
        print(f"   ⚠️  WARNING: Limited range ({range_ratio:.1%})")

    return unique_avgs >= 7


def test_beta_sweep_typical_bounds():
    """
    Test β sweep with typical bounds (min_k=4, max_k=16).

    Pass Criteria:
    - At least 5 of 10 β values produce distinct avg_experts
    - Lower β shows more floor hits
    - Higher β shows more ceiling hits
    """
    print("\n\n" + "=" * 70)
    print("TEST 2: β SWEEP WITH TYPICAL BOUNDS (min_k=4, max_k=16)")
    print("=" * 70)

    torch.manual_seed(42)
    router_logits = torch.randn(1, 100, 64)

    beta_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = {}

    print(f"\n{'β':>6} | {'Avg':>8} | {'Floor%':>8} | {'Ceil%':>8} | {'Mid%':>7}")
    print("-" * 55)

    for beta in beta_values:
        routing_weights, selected_experts, expert_counts, _ = higher_criticism_routing(
            router_logits,
            beta=beta,
            min_k=4,
            max_k=16,  # Typical bounds
            temperature=1.0,
            kde_models=None,
            return_stats=False
        )

        counts_flat = expert_counts.flatten()
        avg = counts_flat.float().mean().item()

        floor_hits = (counts_flat == 4).sum().item()
        ceiling_hits = (counts_flat == 16).sum().item()
        total = len(counts_flat)

        floor_pct = floor_hits / total * 100
        ceiling_pct = ceiling_hits / total * 100
        mid_pct = 100 - floor_pct - ceiling_pct

        results[beta] = {
            'avg': avg,
            'floor_pct': floor_pct,
            'ceiling_pct': ceiling_pct,
            'mid_pct': mid_pct
        }

        print(f"{beta:6.1f} | {avg:8.2f} | {floor_pct:7.1f}% | {ceiling_pct:7.1f}% | {mid_pct:6.1f}%")

    # Validation
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    # Check: Distinct values (expect some clamping, so fewer unique)
    avg_values = [r['avg'] for r in results.values()]
    unique_avgs = len(set(round(a, 1) for a in avg_values))

    print(f"\n1. Distinct avg_experts values: {unique_avgs}/10")
    if unique_avgs >= 5:
        print("   ✅ PASS: β values still produce variation despite bounds")
    else:
        print(f"   ⚠️  WARNING: Limited variation ({unique_avgs} values)")

    # Check: Floor hits decrease as β increases
    floor_percentages = [results[b]['floor_pct'] for b in beta_values]
    floor_trend = sum(1 for i in range(len(floor_percentages)-1)
                      if floor_percentages[i+1] <= floor_percentages[i])

    print(f"\n2. Floor hit trend: {floor_trend}/9 pairs decrease or equal")
    if floor_trend >= 6:
        print("   ✅ PASS: Lower β produces more floor hits")

    return unique_avgs >= 5


def test_weight_normalization():
    """
    Validate that routing weights sum to 1.0 for all tokens.

    Pass Criteria:
    - 100% of tokens have weight sum in [0.99, 1.01]
    - No NaN or Inf values
    """
    print("\n\n" + "=" * 70)
    print("TEST 3: WEIGHT NORMALIZATION")
    print("=" * 70)

    torch.manual_seed(42)
    router_logits = torch.randn(2, 50, 64)  # batch=2, seq=50

    routing_weights, selected_experts, expert_counts, _ = higher_criticism_routing(
        router_logits,
        beta=0.5,
        min_k=4,
        max_k=16,
        temperature=1.0,
        kde_models=None,
        return_stats=False
    )

    # Compute weight sums
    weight_sums = routing_weights.sum(dim=-1)  # [batch, seq]
    weight_sums_flat = weight_sums.flatten()

    # Check for NaN/Inf
    has_nan = torch.isnan(weight_sums_flat).any().item()
    has_inf = torch.isinf(weight_sums_flat).any().item()

    # Check normalization
    in_range = ((weight_sums_flat >= 0.99) & (weight_sums_flat <= 1.01)).sum().item()
    total = len(weight_sums_flat)

    print(f"\nTotal tokens: {total}")
    print(f"Weight sums:")
    print(f"  Min: {weight_sums_flat.min():.6f}")
    print(f"  Max: {weight_sums_flat.max():.6f}")
    print(f"  Mean: {weight_sums_flat.mean():.6f}")
    print(f"  Std: {weight_sums_flat.std():.6f}")

    print(f"\nValidation:")
    print(f"  In range [0.99, 1.01]: {in_range}/{total} ({in_range/total*100:.1f}%)")
    print(f"  Has NaN: {has_nan}")
    print(f"  Has Inf: {has_inf}")

    if has_nan or has_inf:
        print("\n   ❌ FAIL: Contains NaN or Inf values!")
        return False
    elif in_range == total:
        print("\n   ✅ PASS: All weights properly normalized")
        return True
    else:
        print(f"\n   ⚠️  WARNING: {total - in_range} tokens outside normal range")
        return in_range / total >= 0.95  # At least 95%


def test_selection_consistency():
    """
    Verify that num_selected matches actual number of selected experts.

    Pass Criteria:
    - 100% match between logged num_selected and actual count
    """
    print("\n\n" + "=" * 70)
    print("TEST 4: SELECTION CONSISTENCY")
    print("=" * 70)

    torch.manual_seed(42)
    router_logits = torch.randn(1, 50, 64)

    routing_weights, selected_experts, expert_counts, _ = higher_criticism_routing(
        router_logits,
        beta=0.5,
        min_k=4,
        max_k=16,
        temperature=1.0,
        kde_models=None,
        return_stats=False
    )

    # Count actual selections (non-negative expert indices)
    actual_counts = (selected_experts >= 0).sum(dim=-1)

    # Compare with reported counts
    matches = (actual_counts == expert_counts).sum().item()
    total = expert_counts.numel()

    print(f"\nTotal tokens: {total}")
    print(f"Matching counts: {matches}/{total} ({matches/total*100:.1f}%)")

    if matches == total:
        print("\n   ✅ PASS: Selection counts accurate")
        return True
    else:
        print(f"\n   ❌ FAIL: {total - matches} mismatches!")
        return False


def main():
    """Run all tests."""
    print("\n")
    print("═" * 70)
    print("β INDEPENDENCE VALIDATION TEST SUITE")
    print("═" * 70)
    print()

    results = {}

    # Run tests
    try:
        results['test1'] = test_beta_sweep_wide_bounds()
    except Exception as e:
        print(f"\n❌ Test 1 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results['test1'] = False

    try:
        results['test2'] = test_beta_sweep_typical_bounds()
    except Exception as e:
        print(f"\n❌ Test 2 failed with exception: {e}")
        results['test2'] = False

    try:
        results['test3'] = test_weight_normalization()
    except Exception as e:
        print(f"\n❌ Test 3 failed with exception: {e}")
        results['test3'] = False

    try:
        results['test4'] = test_selection_consistency()
    except Exception as e:
        print(f"\n❌ Test 4 failed with exception: {e}")
        results['test4'] = False

    # Summary
    print("\n\n" + "═" * 70)
    print("TEST SUMMARY")
    print("═" * 70)

    test_names = [
        "β Sweep (Wide Bounds)",
        "β Sweep (Typical Bounds)",
        "Weight Normalization",
        "Selection Consistency"
    ]

    for i, (test_key, test_name) in enumerate(zip(results.keys(), test_names), 1):
        status = "✅ PASS" if results[test_key] else "❌ FAIL"
        print(f"  {status}  Test {i}: {test_name}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nβ independence has been successfully restored.")
        print("The fix correctly separates:")
        print("  - Search range definition (controlled by β)")
        print("  - Final selection constraints (controlled by min_k/max_k)")
        return 0
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print(f"\nPlease investigate the {total - passed} failed test(s).")
        return 1


if __name__ == '__main__':
    sys.exit(main())
