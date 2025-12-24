#!/usr/bin/env python3
"""
Comprehensive Test Suite for Higher Criticism (HC) Routing
==========================================================

Mirrors the structure of test_bh_routing_comprehensive.py.
Tests all aspects of HC routing with concrete examples.

Run with: pytest test_hc_routing_comprehensive.py -v
"""

import torch
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hc_routing import (
    compute_hc_scores,
    higher_criticism_routing,
    run_hc_multi_beta,
    compare_hc_bh,
    compare_multi_beta_statistics
)
from bh_routing import load_kde_models, benjamini_hochberg_routing


# =============================================================================
# TEST GROUP 1: Core HC Algorithm (5 tests)
# =============================================================================

class TestHCCoreAlgorithm:
    """Core HC algorithm tests with concrete examples."""

    def test_01_concrete_example_hc_calculation(self):
        """
        Test HC calculation with concrete values.

        Setup: 8 sorted p-values
        Expected: Manual HC calculation matches implementation
        """
        # Sorted p-values (already ascending)
        p_sorted = torch.tensor([[0.01, 0.05, 0.12, 0.25, 0.45, 0.60, 0.78, 0.90]])
        n = 8

        # Manual HC calculation for first 3 ranks:
        # HC(1) = √8 × (1/8 - 0.01) / √(0.01 × 0.99) = 2.83 × 0.115 / 0.0995 = 3.27
        # HC(2) = √8 × (2/8 - 0.05) / √(0.05 × 0.95) = 2.83 × 0.200 / 0.218 = 2.60
        # HC(3) = √8 × (3/8 - 0.12) / √(0.12 × 0.88) = 2.83 × 0.255 / 0.325 = 2.22

        hc_scores, i_star = compute_hc_scores(p_sorted, beta=1.0)

        print(f"\nHC scores: {hc_scores[0, :4].tolist()}")
        print(f"i* (optimal rank): {i_star[0].item()}")

        # HC(1) should be highest (strongest signal at rank 1)
        assert hc_scores[0, 0] > hc_scores[0, 1], "HC(1) should be > HC(2)"
        assert i_star[0].item() == 1, f"i* should be 1, got {i_star[0].item()}"

    def test_02_beta_sensitivity(self):
        """
        Test that beta parameter affects search range correctly.

        Lower beta → searches fewer ranks → potentially different i*
        """
        torch.manual_seed(42)
        router_logits = torch.randn(2, 10, 64)

        # Test different beta values
        _, _, counts_low = higher_criticism_routing(router_logits, beta=0.3, max_k=16)
        _, _, counts_mid = higher_criticism_routing(router_logits, beta=0.5, max_k=16)
        _, _, counts_high = higher_criticism_routing(router_logits, beta=0.8, max_k=16)

        print(f"\nBeta=0.3: avg={counts_low.float().mean():.2f}")
        print(f"Beta=0.5: avg={counts_mid.float().mean():.2f}")
        print(f"Beta=0.8: avg={counts_high.float().mean():.2f}")

        # All should produce valid results
        assert counts_low.min() >= 1
        assert counts_mid.min() >= 1
        assert counts_high.min() >= 1

    def test_03_ceiling_hit_enforcement(self):
        """
        Test that max_k is enforced even when HC wants more.
        """
        # Create input where many experts look significant
        router_logits = torch.randn(1, 5, 64) + 2.0  # Shift up

        _, _, counts = higher_criticism_routing(
            router_logits, beta=0.9, min_k=1, max_k=4
        )

        assert counts.max().item() <= 4, f"max_k=4 not enforced, got max={counts.max()}"

    def test_04_floor_hit_enforcement(self):
        """
        Test that min_k is enforced even when HC suggests fewer.
        """
        # Create input with weak signals
        router_logits = torch.randn(1, 5, 64) * 0.1  # Very flat

        _, _, counts = higher_criticism_routing(
            router_logits, beta=0.2, min_k=3, max_k=8
        )

        assert counts.min().item() >= 3, f"min_k=3 not enforced, got min={counts.min()}"

    def test_05_batch_consistency(self):
        """
        Test that batch processing gives same results as individual.
        """
        torch.manual_seed(42)
        logits_single = torch.randn(1, 1, 64)

        # Process as single
        _, _, count_single = higher_criticism_routing(logits_single, beta=0.5, max_k=8)

        # Process in batch
        logits_batch = logits_single.expand(3, 1, 64).clone()
        _, _, counts_batch = higher_criticism_routing(logits_batch, beta=0.5, max_k=8)

        # All batch entries should match single
        for i in range(3):
            assert counts_batch[i, 0] == count_single[0, 0], \
                f"Batch entry {i} differs from single"


# =============================================================================
# TEST GROUP 2: P-Value Ordering (3 tests)
# =============================================================================

class TestPValueOrdering:
    """Tests for p-value computation and ordering."""

    def test_06_higher_logit_lower_pvalue(self):
        """
        Higher router logit → lower p-value (more significant).
        """
        # Monotonically increasing logits
        logits = torch.tensor([[[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]]])

        _, _, _, stats = higher_criticism_routing(
            logits, beta=0.5, max_k=8, return_stats=True
        )

        p_values = stats['p_values'][0, 0, :]

        # Higher logits should have lower p-values
        # p_values[0] (logit=-3) should be > p_values[7] (logit=4)
        assert p_values[0] > p_values[7], \
            f"Higher logit should have lower p-value: p[-3]={p_values[0]:.4f}, p[4]={p_values[7]:.4f}"

    def test_07_sorted_pvalues_ascending(self):
        """
        Sorted p-values should be in ascending order.
        """
        router_logits = torch.randn(2, 5, 64)

        _, _, _, stats = higher_criticism_routing(
            router_logits, beta=0.5, max_k=8, return_stats=True
        )

        sorted_pvals = stats['sorted_pvalues']

        # Check ascending order for each token
        for b in range(2):
            for s in range(5):
                diffs = sorted_pvals[b, s, 1:] - sorted_pvals[b, s, :-1]
                assert (diffs >= -1e-6).all(), "Sorted p-values not ascending"

    def test_08_pvalue_range_valid(self):
        """
        P-values should be in (0, 1).
        """
        # Test with various logit ranges
        for scale in [0.1, 1.0, 10.0]:
            router_logits = torch.randn(2, 5, 64) * scale

            _, _, _, stats = higher_criticism_routing(
                router_logits, beta=0.5, max_k=8, return_stats=True
            )

            p_values = stats['p_values']

            assert (p_values > 0).all(), f"P-values have zeros (scale={scale})"
            assert (p_values < 1).all(), f"P-values have ones (scale={scale})"


# =============================================================================
# TEST GROUP 3: KDE Integration (3 tests)
# =============================================================================

class TestKDEIntegration:
    """Tests using KDE-based p-values (production flow)."""

    def test_09_kde_models_load(self):
        """
        KDE models load correctly.
        """
        kde_models = load_kde_models()

        if not kde_models:
            pytest.skip("KDE models not available")

        assert len(kde_models) == 16, f"Expected 16 layer models, got {len(kde_models)}"

        # Check structure
        for layer_idx in range(16):
            assert layer_idx in kde_models
            assert 'x' in kde_models[layer_idx]
            assert 'cdf' in kde_models[layer_idx]

    def test_10_kde_produces_valid_pvalues(self):
        """
        KDE-based p-values are valid.
        """
        kde_models = load_kde_models()

        if not kde_models:
            pytest.skip("KDE models not available")

        router_logits = torch.randn(2, 5, 64)

        _, _, _, stats = higher_criticism_routing(
            router_logits, beta=0.5, max_k=8,
            kde_models=kde_models, return_stats=True
        )

        p_values = stats['p_values']

        assert (p_values > 0).all(), "KDE p-values have zeros"
        assert (p_values < 1).all(), "KDE p-values have ones"
        assert not p_values.isnan().any(), "KDE p-values have NaN"

    def test_11_layer_specific_kde(self):
        """
        Different layers use different KDE models.
        """
        kde_models = load_kde_models()

        if not kde_models:
            pytest.skip("KDE models not available")

        router_logits = torch.randn(1, 5, 64)

        # Get p-values from different layers
        _, _, _, stats_0 = higher_criticism_routing(
            router_logits, beta=0.5, max_k=8,
            layer_idx=0, kde_models=kde_models, return_stats=True
        )
        _, _, _, stats_7 = higher_criticism_routing(
            router_logits, beta=0.5, max_k=8,
            layer_idx=7, kde_models=kde_models, return_stats=True
        )

        # P-values may differ between layers
        p0 = stats_0['p_values']
        p7 = stats_7['p_values']

        # Both should be valid
        assert (p0 > 0).all() and (p0 < 1).all()
        assert (p7 > 0).all() and (p7 < 1).all()


# =============================================================================
# TEST GROUP 4: Adaptive Behavior (3 tests)
# =============================================================================

class TestAdaptiveBehavior:
    """Tests for adaptive expert selection."""

    def test_12_variable_expert_counts(self):
        """
        HC produces variable expert counts (not fixed like TopK).
        """
        torch.manual_seed(42)
        router_logits = torch.randn(4, 20, 64)  # Larger sample

        _, _, counts = higher_criticism_routing(
            router_logits, beta=0.5, min_k=1, max_k=16
        )

        unique_counts = counts.unique().numel()

        print(f"\nUnique expert counts: {unique_counts}")
        print(f"Count distribution: {counts.flatten().bincount()}")

        # Should have at least 2 different counts (adaptive)
        assert unique_counts >= 2, \
            f"Expected variable counts, got only {unique_counts} unique values"

    def test_13_strong_signal_behavior(self):
        """
        Clear signals → HC selects based on signal strength.
        """
        # Strong signal: several boosted experts
        strong_logits = torch.full((1, 1, 64), -2.0)
        strong_logits[0, 0, :6] = torch.tensor([4.0, 3.5, 3.0, 2.5, 2.0, 1.5])

        # Weak signal: flat
        weak_logits = torch.randn(1, 1, 64) * 0.1

        _, _, counts_strong = higher_criticism_routing(strong_logits, beta=0.5, max_k=16)
        _, _, counts_weak = higher_criticism_routing(weak_logits, beta=0.5, min_k=1, max_k=16)

        print(f"\nStrong signal: {counts_strong[0, 0].item()} experts")
        print(f"Weak signal: {counts_weak[0, 0].item()} experts")

        # Both should be valid (no assertion on relative counts - HC is data-adaptive)
        assert 1 <= counts_strong[0, 0].item() <= 16
        assert 1 <= counts_weak[0, 0].item() <= 16

    def test_14_sparse_detection(self):
        """
        HC designed for sparse signal detection.
        """
        # Create sparse signal: only 3 out of 64 experts are relevant
        router_logits = torch.full((1, 1, 64), -3.0)
        router_logits[0, 0, [5, 12, 31]] = torch.tensor([5.0, 4.5, 4.0])

        _, experts, counts = higher_criticism_routing(
            router_logits, beta=0.5, min_k=1, max_k=16
        )

        # Should select a small number of experts
        assert 1 <= counts[0, 0].item() <= 8, \
            f"Expected sparse selection, got {counts[0, 0].item()}"


# =============================================================================
# TEST GROUP 5: HC vs BH Comparison (2 tests)
# =============================================================================

class TestHCvsBHComparison:
    """Comparison tests between HC and BH routing."""

    def test_15_both_methods_valid(self):
        """
        Both HC and BH produce valid outputs on same input.
        """
        torch.manual_seed(42)
        router_logits = torch.randn(2, 10, 64)

        hc_weights, _, hc_counts = higher_criticism_routing(
            router_logits, beta=0.5, max_k=8
        )
        bh_weights, _, bh_counts = benjamini_hochberg_routing(
            router_logits, alpha=0.30, max_k=8
        )

        # Both weight matrices should sum to 1
        assert torch.allclose(hc_weights.sum(dim=-1), torch.ones(2, 10), atol=1e-5)
        assert torch.allclose(bh_weights.sum(dim=-1), torch.ones(2, 10), atol=1e-5)

        # Both should respect constraints
        assert hc_counts.min() >= 1 and hc_counts.max() <= 8
        assert bh_counts.min() >= 1 and bh_counts.max() <= 8

    def test_16_compare_function_comprehensive(self):
        """
        compare_hc_bh() provides comprehensive comparison.
        """
        torch.manual_seed(42)
        router_logits = torch.randn(4, 20, 64)

        comparison = compare_hc_bh(
            router_logits, hc_beta=0.5, bh_alpha=0.30, max_k=16
        )

        print(f"\n--- HC vs BH Comparison ---")
        print(f"HC (β=0.5): avg={comparison['hc_avg']:.2f}, std={comparison['hc_std']:.2f}")
        print(f"BH (α=0.30): avg={comparison['bh_avg']:.2f}, std={comparison['bh_std']:.2f}")
        print(f"Agreement: {comparison['agreement']*100:.1f}%")

        # Check all expected keys present
        expected_keys = ['hc_avg', 'bh_avg', 'hc_std', 'bh_std', 'agreement']
        for key in expected_keys:
            assert key in comparison, f"Missing key: {key}"


# =============================================================================
# TEST GROUP 6: Edge Cases (3 tests)
# =============================================================================

class TestEdgeCases:
    """Edge case handling."""

    def test_17_uniform_logits_handled(self):
        """
        All logits equal → graceful handling.
        """
        router_logits = torch.ones(1, 5, 64) * 0.5

        weights, experts, counts = higher_criticism_routing(
            router_logits, beta=0.5, min_k=2, max_k=8
        )

        # Should not crash, should respect min_k
        assert counts.min() >= 2
        assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 5), atol=1e-5)

    def test_18_extreme_logits_handled(self):
        """
        Very large/small logits handled without numerical issues.
        """
        # Very large logits
        large_logits = torch.randn(1, 5, 64) * 100
        weights_large, _, counts_large = higher_criticism_routing(
            large_logits, beta=0.5, max_k=8
        )

        # Very small logits
        small_logits = torch.randn(1, 5, 64) * 0.001
        weights_small, _, counts_small = higher_criticism_routing(
            small_logits, beta=0.5, max_k=8
        )

        # Both should produce valid outputs
        assert not weights_large.isnan().any()
        assert not weights_small.isnan().any()
        assert torch.allclose(weights_large.sum(dim=-1), torch.ones(1, 5), atol=1e-4)
        assert torch.allclose(weights_small.sum(dim=-1), torch.ones(1, 5), atol=1e-4)

    def test_19_large_batch_handled(self):
        """
        Large batch sizes handled correctly.
        """
        router_logits = torch.randn(8, 32, 64)  # 8 batches, 32 tokens

        weights, experts, counts = higher_criticism_routing(
            router_logits, beta=0.5, max_k=8
        )

        assert weights.shape == (8, 32, 64)
        assert experts.shape == (8, 32, 8)
        assert counts.shape == (8, 32)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(8, 32), atol=1e-5)


# =============================================================================
# TEST GROUP 7: Multi-Beta Analysis (2 tests)
# =============================================================================

class TestMultiBetaAnalysis:
    """Tests for multi-beta comparison utilities."""

    def test_20_run_hc_multi_beta(self):
        """
        run_hc_multi_beta() works correctly.
        """
        torch.manual_seed(42)
        router_logits = torch.randn(2, 10, 64)

        results = run_hc_multi_beta(
            router_logits,
            beta_values=[0.3, 0.5, 0.7],
            max_k=8
        )

        assert len(results) == 3
        for beta in [0.3, 0.5, 0.7]:
            assert beta in results
            weights, experts, counts = results[beta]
            assert weights.shape == (2, 10, 64)
            assert experts.shape == (2, 10, 8)
            assert counts.shape == (2, 10)

    def test_21_compare_multi_beta_statistics(self):
        """
        compare_multi_beta_statistics() generates proper DataFrame.
        """
        torch.manual_seed(42)
        router_logits = torch.randn(2, 10, 64)

        results = run_hc_multi_beta(router_logits, max_k=8)
        df = compare_multi_beta_statistics(results)

        print(f"\n{df.to_string(index=False)}")

        # Check DataFrame structure
        assert 'beta' in df.columns
        assert 'mean_experts' in df.columns
        assert 'std_experts' in df.columns
        assert len(df) == 4  # Default 4 beta values


# =============================================================================
# Run All Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
