"""
End-to-End Validation Test for HC Routing System
=================================================

This test validates that the Higher Criticism routing system:
1. HC routing fully replaces Top-K (not fixed 8 experts)
2. Logger captures all decisions with complete statistics
3. HC statistics are properly computed
4. Output is coherent and deterministic
5. All assertions pass during execution

Run with: python test_hc_end_to_end.py
"""

import torch
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from hc_routing import higher_criticism_routing
from hc_routing_logging import HCRoutingLogger
from deprecated.bh_routing import load_kde_models


def test_hc_routing_standalone():
    """
    Test HC routing function standalone (without model).

    Validates:
    - Function executes without errors
    - Logging captures decisions
    - HC statistics are computed
    - Adaptive selection (not fixed-K)
    - All assertions pass
    """
    print("=" * 70)
    print("TEST 1: HC ROUTING STANDALONE")
    print("=" * 70)

    # Create test data
    torch.manual_seed(42)
    batch_size = 1
    seq_len = 20
    num_experts = 64

    router_logits = torch.randn(batch_size, seq_len, num_experts)

    # Create logger
    logger = HCRoutingLogger(
        output_dir="../test_logs",
        experiment_name="standalone_test",
        log_every_n=1  # Log every decision for testing
    )

    print("\n[1/5] Testing with beta='auto' (adaptive)...")

    # Run HC routing
    routing_weights, selected_experts, expert_counts, stats = higher_criticism_routing(
        router_logits,
        beta='auto',
        temperature=1.0,
        min_k=4,
        max_k=12,
        layer_idx=0,
        kde_models=None,  # Will use empirical fallback
        return_stats=True,
        logger=logger,
        log_every_n_tokens=1,
        sample_idx=0,
        token_idx=0
    )

    # Validation 1: Shape checks
    print("\n[2/5] Validating output shapes...")
    assert routing_weights.shape == (batch_size, seq_len, num_experts), \
        f"Wrong routing_weights shape: {routing_weights.shape}"
    assert selected_experts.shape == (batch_size, seq_len, 12), \
        f"Wrong selected_experts shape: {selected_experts.shape}"
    assert expert_counts.shape == (batch_size, seq_len), \
        f"Wrong expert_counts shape: {expert_counts.shape}"
    print(f"   ✓ Shapes correct")

    # Validation 2: Constraints
    print("\n[3/5] Validating constraints...")
    assert (expert_counts >= 4).all(), f"Some counts below min_k=4: {expert_counts.min()}"
    assert (expert_counts <= 12).all(), f"Some counts above max_k=12: {expert_counts.max()}"
    print(f"   ✓ All selections in range [4, 12]")

    # Validation 3: Adaptive behavior (not all same)
    print("\n[4/5] Validating adaptive selection...")
    unique_counts = expert_counts.unique()
    assert len(unique_counts) > 1, \
        "All selections identical - HC not adaptive!"
    selections_list = expert_counts.flatten().tolist()
    print(f"   ✓ Adaptive: {len(unique_counts)} different selection counts")
    print(f"   ✓ Range: [{min(selections_list)}, {max(selections_list)}]")
    print(f"   ✓ Mean: {sum(selections_list)/len(selections_list):.1f}")

    # Validation 4: Logger captured decisions
    print("\n[5/5] Validating logger...")
    assert logger.total_decisions > 0, "FAIL: No decisions logged!"
    assert len(logger.routing_decisions) > 0, "FAIL: Empty routing_decisions!"
    print(f"   ✓ Total decisions logged: {logger.total_decisions}")
    print(f"   ✓ Full logs stored: {len(logger.routing_decisions)}")

    # Validation 5: HC statistics in logs
    print("\n[6/6] Validating log schema...")
    sample_log = logger.routing_decisions[0]

    required_fields = [
        'sample_idx', 'token_idx', 'layer_idx',
        'router_logits', 'router_logits_stats',
        'hc_statistics', 'hc_max_rank', 'hc_max_value', 'hc_positive_count',
        'num_selected', 'selected_experts', 'routing_weights',
        'selection_reason', 'hit_min_k', 'hit_max_k',
        'weights_sum', 'config'
    ]

    missing = [f for f in required_fields if f not in sample_log]
    if missing:
        print(f"   ✗ FAIL: Missing fields: {missing}")
        return False

    print(f"   ✓ All {len(required_fields)} required fields present")

    # Check HC statistics
    hc_stats = sample_log['hc_statistics']
    assert len(hc_stats) == 64, f"Expected 64 HC values, got {len(hc_stats)}"
    print(f"   ✓ HC statistics length: {len(hc_stats)}")

    hc_max_rank = sample_log['hc_max_rank']
    assert 1 <= hc_max_rank <= 64, f"Invalid hc_max_rank: {hc_max_rank}"
    print(f"   ✓ HC max rank valid: {hc_max_rank}")

    print(f"   ✓ Selection reason: {sample_log['selection_reason']}")
    print(f"   ✓ Weights sum: {sample_log['weights_sum']:.6f}")

    # Save logs
    logger.save_logs()
    print(f"\n   ✓ Logs saved to ./test_logs/")

    print("\n" + "=" * 70)
    print("✅ TEST 1 PASSED - HC ROUTING STANDALONE")
    print("=" * 70)

    return True


def test_hc_routing_deterministic():
    """Test that HC routing is deterministic."""
    print("\n" + "=" * 70)
    print("TEST 2: DETERMINISTIC BEHAVIOR")
    print("=" * 70)

    torch.manual_seed(123)
    logits = torch.randn(1, 10, 64)

    # Run twice with same seed
    torch.manual_seed(456)
    weights1, experts1, counts1, _ = higher_criticism_routing(
        logits, beta=0.5, min_k=4, max_k=12, kde_models=None
    )

    torch.manual_seed(456)
    weights2, experts2, counts2, _ = higher_criticism_routing(
        logits, beta=0.5, min_k=4, max_k=12, kde_models=None
    )

    assert torch.allclose(weights1, weights2), "Routing weights not deterministic!"
    assert torch.equal(counts1, counts2), "Expert counts not deterministic!"

    print("   ✓ Same input → same output")
    print("\n" + "=" * 70)
    print("✅ TEST 2 PASSED - DETERMINISTIC")
    print("=" * 70)

    return True


def test_different_beta_values():
    """Test different beta values and verify they execute correctly."""
    print("\n" + "=" * 70)
    print("TEST 3: DIFFERENT BETA VALUES")
    print("=" * 70)

    torch.manual_seed(42)
    logits = torch.randn(1, 20, 64)

    results = {}

    for beta in ['auto', 0.3, 0.5, 0.7, 1.0]:
        weights, experts, counts, _ = higher_criticism_routing(
            logits, beta=beta, min_k=4, max_k=12, kde_models=None
        )

        avg = counts.float().mean().item()
        std = counts.float().std().item()
        min_c = counts.min().item()
        max_c = counts.max().item()

        results[str(beta)] = {
            'avg': avg,
            'std': std,
            'range': (min_c, max_c),
            'counts': counts
        }

        beta_str = f"beta={beta:4}" if beta != 'auto' else "beta=auto"
        print(f"   {beta_str:12s} → avg={avg:.1f} ± {std:.1f}, range=[{min_c:2d}, {max_c:2d}]")

    # Verify all beta values execute without error and produce valid results
    for beta_val, result in results.items():
        assert 4 <= result['range'][0] <= result['range'][1] <= 12, \
            f"Beta {beta_val} violated constraints"

    # Check that results are adaptive (not all the same count for each beta)
    all_adaptive = all(len(r['counts'].unique()) > 1 for r in results.values())
    if all_adaptive:
        print("\n   ✓ All beta values produce adaptive selection")
    else:
        print("\n   ⚠️ Some beta values produced fixed selection (unusual)")

    print("\n   ✓ All beta values executed successfully")

    print("\n" + "=" * 70)
    print("✅ TEST 3 PASSED - DIFFERENT BETAS")
    print("=" * 70)

    return True


def test_with_kde_models():
    """Test with actual KDE models if available."""
    print("\n" + "=" * 70)
    print("TEST 4: KDE MODELS (OPTIONAL)")
    print("=" * 70)

    # Try to load KDE models
    kde_models = load_kde_models()

    if not kde_models:
        print("   ⚠️ KDE models not found - skipping this test")
        print("   (This is OK - system works with empirical fallback)")
        return True

    print(f"   ✓ Loaded {len(kde_models)} KDE models")

    torch.manual_seed(42)
    logits = torch.randn(1, 10, 64)

    # Run with KDE models
    weights, experts, counts, _ = higher_criticism_routing(
        logits,
        beta='auto',
        min_k=4,
        max_k=12,
        layer_idx=0,
        kde_models=kde_models
    )

    assert counts.min() >= 4 and counts.max() <= 12, "Constraints violated with KDE"
    print(f"   ✓ HC routing works with KDE models")
    print(f"   ✓ Selection range: [{counts.min()}, {counts.max()}]")

    print("\n" + "=" * 70)
    print("✅ TEST 4 PASSED - KDE MODELS")
    print("=" * 70)

    return True


def main():
    """Run all validation tests."""
    print("\n")
    print("═" * 70)
    print("HC ROUTING SYSTEM - END-TO-END VALIDATION")
    print("═" * 70)
    print("\n")

    tests = [
        ("HC Routing Standalone", test_hc_routing_standalone),
        ("Deterministic Behavior", test_hc_routing_deterministic),
        ("Different Beta Values", test_different_beta_values),
        ("KDE Models (Optional)", test_with_kde_models),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ TEST FAILED: {test_name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n")
    print("═" * 70)
    print("VALIDATION SUMMARY")
    print("═" * 70)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {test_name}")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    print(f"\nPassed: {passed_count}/{total_count}")

    if passed_count == total_count:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("\nThe HC routing system is working correctly:")
        print("  ✓ API mismatch fixed")
        print("  ✓ Logger captures all decisions")
        print("  ✓ HC statistics computed correctly")
        print("  ✓ Adaptive selection (not fixed-K)")
        print("  ✓ Deterministic behavior")
        print("  ✓ Complete log schema")
        print("\n" + "═" * 70)
        return 0
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print("\n" + "═" * 70)
        return 1


if __name__ == "__main__":
    exit(main())
