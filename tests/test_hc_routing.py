"""
Basic Test Suite for Higher Criticism (HC) Routing
==================================================

Tests core functionality of HC routing implementation.
Run with: pytest test_hc_routing.py -v
"""

import torch
import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hc_routing import (
    compute_hc_scores,
    higher_criticism_routing,
    run_hc_multi_beta,
    compare_hc_bh,
)
from deprecated.bh_routing import load_kde_models, benjamini_hochberg_routing


class TestHCScoreComputation:
    """Tests for compute_hc_scores() function."""

    def test_hc_formula_correctness(self):
        """Verify HC formula: HC(i) = √n × (i/n - p₍ᵢ₎) / √(p₍ᵢ₎(1-p₍ᵢ₎))"""
        # Known sorted p-values
        p_sorted = torch.tensor([[0.01, 0.05, 0.15, 0.35, 0.50, 0.65, 0.75, 0.85]])
        n = 8

        # Manual calculation for rank 1
        # HC(1) = √8 × (1/8 - 0.01) / √(0.01 × 0.99)
        expected_rank = 1
        expected_p = 0.01
        expected_hc_1 = np.sqrt(n) * (expected_rank/n - expected_p) / np.sqrt(expected_p * (1 - expected_p))

        hc_scores, i_star = compute_hc_scores(p_sorted, beta=1.0)

        # Check HC at rank 1 matches manual calculation
        assert abs(hc_scores[0, 0].item() - expected_hc_1) < 0.1, \
            f"Expected HC(1)={expected_hc_1:.2f}, got {hc_scores[0, 0].item():.2f}"

    def test_i_star_is_argmax(self):
        """Verify i_star is the rank where HC is maximized."""
        p_sorted = torch.tensor([[0.001, 0.01, 0.05, 0.3, 0.5, 0.7, 0.8, 0.9]])

        hc_scores, i_star = compute_hc_scores(p_sorted, beta=1.0)

        # i_star should be argmax + 1 (convert to 1-indexed)
        manual_argmax = hc_scores[0].argmax().item() + 1
        assert i_star[0].item() == manual_argmax

    def test_beta_limits_search_range(self):
        """Verify beta parameter limits the search range."""
        p_sorted = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])

        # With beta=0.5, only ranks 1-4 should be searched (8 * 0.5 = 4)
        hc_scores, i_star = compute_hc_scores(p_sorted, beta=0.5)

        # Ranks 5-8 should have -inf
        assert hc_scores[0, 4:].max().item() == float('-inf')

        # i_star should be in [1, 4]
        assert 1 <= i_star[0].item() <= 4

    def test_batch_processing(self):
        """HC scores computed correctly for batched input."""
        p_sorted = torch.tensor([
            [0.01, 0.05, 0.15, 0.35, 0.50, 0.65, 0.75, 0.85],
            [0.02, 0.08, 0.20, 0.40, 0.55, 0.70, 0.80, 0.90]
        ])

        hc_scores, i_star = compute_hc_scores(p_sorted, beta=0.5)

        assert hc_scores.shape == (2, 8)
        assert i_star.shape == (2,)
        assert i_star.min().item() >= 1


class TestHCRouting:
    """Tests for higher_criticism_routing() function."""

    def test_output_shapes_3d(self):
        """Correct output shapes for 3D input."""
        router_logits = torch.randn(2, 10, 64)

        weights, experts, counts = higher_criticism_routing(
            router_logits, beta=0.5, max_k=8
        )

        assert weights.shape == (2, 10, 64)
        assert experts.shape == (2, 10, 8)
        assert counts.shape == (2, 10)

    def test_output_shapes_2d(self):
        """Correct output shapes for 2D input."""
        router_logits = torch.randn(10, 64)

        weights, experts, counts = higher_criticism_routing(
            router_logits, beta=0.5, max_k=8
        )

        assert weights.shape == (10, 64)
        assert experts.shape == (10, 8)
        assert counts.shape == (10,)

    def test_weights_sum_to_one(self):
        """Routing weights sum to 1 for each token."""
        router_logits = torch.randn(2, 10, 64)

        weights, _, _ = higher_criticism_routing(router_logits, beta=0.5, max_k=8)

        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_min_k_enforced(self):
        """Minimum experts constraint is enforced."""
        router_logits = torch.randn(2, 10, 64)

        _, _, counts = higher_criticism_routing(
            router_logits, beta=0.5, min_k=3, max_k=8
        )

        assert counts.min().item() >= 3

    def test_max_k_enforced(self):
        """Maximum experts constraint is enforced."""
        router_logits = torch.randn(2, 10, 64)

        _, _, counts = higher_criticism_routing(
            router_logits, beta=0.9, min_k=1, max_k=4
        )

        assert counts.max().item() <= 4

    def test_determinism(self):
        """Same input produces same output."""
        torch.manual_seed(42)
        router_logits = torch.randn(2, 5, 64)

        result1 = higher_criticism_routing(router_logits, beta=0.5)
        result2 = higher_criticism_routing(router_logits, beta=0.5)

        assert torch.equal(result1[0], result2[0])  # weights
        assert torch.equal(result1[2], result2[2])  # counts

    def test_sparse_signal_detection(self):
        """HC correctly identifies sparse signals."""
        # Create clear sparse signal: 3 experts with high logits
        router_logits = torch.full((1, 1, 64), -2.0)
        router_logits[0, 0, 0] = 3.0
        router_logits[0, 0, 1] = 2.5
        router_logits[0, 0, 2] = 2.0

        weights, experts, counts = higher_criticism_routing(
            router_logits, beta=0.5, max_k=16
        )

        # Should select approximately 3 experts (the sparse signals)
        assert 1 <= counts[0, 0].item() <= 6, \
            f"Expected 1-6 experts, got {counts[0, 0].item()}"

        # Top experts should include 0, 1, 2
        selected = experts[0, 0, :counts[0, 0]].tolist()
        assert 0 in selected or 1 in selected, \
            f"Expected expert 0 or 1 in selection, got {selected}"


class TestHCvsBH:
    """Comparison tests between HC and BH routing."""

    def test_both_produce_valid_weights(self):
        """Both HC and BH produce valid routing weights."""
        torch.manual_seed(42)
        router_logits = torch.randn(2, 5, 64)

        hc_weights, _, hc_counts = higher_criticism_routing(
            router_logits, beta=0.5, max_k=8
        )
        bh_weights, _, bh_counts = benjamini_hochberg_routing(
            router_logits, alpha=0.30, max_k=8
        )

        # Both should sum to 1
        assert torch.allclose(hc_weights.sum(dim=-1), torch.ones(2, 5), atol=1e-5)
        assert torch.allclose(bh_weights.sum(dim=-1), torch.ones(2, 5), atol=1e-5)

        # Both should have valid counts
        assert hc_counts.min() >= 1 and hc_counts.max() <= 8
        assert bh_counts.min() >= 1 and bh_counts.max() <= 8

    def test_compare_function_works(self):
        """compare_hc_bh() function works correctly."""
        router_logits = torch.randn(2, 5, 64)

        comparison = compare_hc_bh(router_logits, hc_beta=0.5, bh_alpha=0.30, max_k=8)

        assert 'hc_avg' in comparison
        assert 'bh_avg' in comparison
        assert 'agreement' in comparison
        assert 0.0 <= comparison['agreement'] <= 1.0


class TestWithKDE:
    """Tests using KDE-based p-values."""

    def test_kde_integration(self):
        """HC works with KDE models."""
        kde_models = load_kde_models()

        if not kde_models:
            pytest.skip("KDE models not available")

        router_logits = torch.randn(2, 5, 64)

        weights, experts, counts = higher_criticism_routing(
            router_logits,
            beta=0.5,
            max_k=8,
            layer_idx=0,
            kde_models=kde_models
        )

        assert weights.shape == (2, 5, 64)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 5), atol=1e-5)


class TestEdgeCases:
    """Edge case tests."""

    def test_uniform_logits(self):
        """All logits equal - should fall back to min_k."""
        router_logits = torch.ones(1, 1, 64) * 0.5

        weights, experts, counts = higher_criticism_routing(
            router_logits, beta=0.5, min_k=2, max_k=8
        )

        # With uniform logits, HC has no signal - should hit min_k
        assert counts[0, 0].item() >= 2

    def test_single_dominant_expert(self):
        """One expert much stronger than others."""
        router_logits = torch.full((1, 1, 64), -5.0)
        router_logits[0, 0, 0] = 5.0  # One dominant

        weights, experts, counts = higher_criticism_routing(
            router_logits, beta=0.5, max_k=16
        )

        # Should select very few experts
        assert counts[0, 0].item() <= 4

        # Expert 0 should be selected
        assert 0 in experts[0, 0, :counts[0, 0]].tolist()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
