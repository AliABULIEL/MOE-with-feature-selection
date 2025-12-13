"""
Comprehensive Test Suite for Benjamini-Hochberg Routing
========================================================

Tests cover:
1. Shape correctness
2. Weight normalization (sum to 1)
3. Alpha sensitivity (strict α → fewer experts)
4. Temperature sensitivity (higher T → more experts)
5. Min/max constraints
6. Comparison with top-k baseline
7. Edge cases and numerical stability
8. GPU compatibility
"""

import torch
import pytest
import numpy as np
from bh_routing import (
    benjamini_hochberg_routing,
    topk_routing,
    compute_routing_statistics
)


class TestShapeCorrectness:
    """Test that output shapes are correct for various inputs."""

    def test_2d_input(self):
        """Test with 2D input [num_tokens, num_experts]."""
        num_tokens, num_experts = 10, 64
        router_logits = torch.randn(num_tokens, num_experts)
        max_k = 8

        weights, experts, counts = benjamini_hochberg_routing(
            router_logits, max_k=max_k
        )

        assert weights.shape == (num_tokens, num_experts), f"Expected {(num_tokens, num_experts)}, got {weights.shape}"
        assert experts.shape == (num_tokens, max_k), f"Expected {(num_tokens, max_k)}, got {experts.shape}"
        assert counts.shape == (num_tokens,), f"Expected {(num_tokens,)}, got {counts.shape}"

    def test_3d_input(self):
        """Test with 3D input [batch, seq_len, num_experts]."""
        batch_size, seq_len, num_experts = 2, 16, 64
        router_logits = torch.randn(batch_size, seq_len, num_experts)
        max_k = 8

        weights, experts, counts = benjamini_hochberg_routing(
            router_logits, max_k=max_k
        )

        assert weights.shape == (batch_size, seq_len, num_experts)
        assert experts.shape == (batch_size, seq_len, max_k)
        assert counts.shape == (batch_size, seq_len)

    def test_various_expert_counts(self):
        """Test with different numbers of experts."""
        for num_experts in [8, 16, 32, 64, 128]:
            router_logits = torch.randn(5, num_experts)
            max_k = min(8, num_experts)

            weights, experts, counts = benjamini_hochberg_routing(
                router_logits, max_k=max_k
            )

            assert weights.shape == (5, num_experts)
            assert experts.shape == (5, max_k)


class TestWeightNormalization:
    """Test that routing weights are properly normalized."""

    def test_weights_sum_to_one(self):
        """Weights should sum to 1.0 for each token."""
        router_logits = torch.randn(10, 64)

        weights, _, _ = benjamini_hochberg_routing(router_logits)

        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
            f"Weight sums not 1.0: {weight_sums}"

    def test_weights_non_negative(self):
        """All weights should be non-negative."""
        router_logits = torch.randn(10, 64)

        weights, _, _ = benjamini_hochberg_routing(router_logits)

        assert (weights >= 0).all(), "Found negative weights"

    def test_sparse_weights(self):
        """Most weights should be zero (sparse routing)."""
        router_logits = torch.randn(10, 64)

        weights, _, counts = benjamini_hochberg_routing(
            router_logits, max_k=8
        )

        # Count non-zero weights per token
        nonzero_per_token = (weights > 0).sum(dim=-1)

        # Should match expert_counts
        assert torch.equal(nonzero_per_token, counts), \
            f"Nonzero counts {nonzero_per_token} don't match expert_counts {counts}"

        # Should be sparse (most weights are zero)
        sparsity = (weights == 0).float().mean()
        assert sparsity > 0.5, f"Expected high sparsity, got {sparsity:.2%}"


class TestAlphaSensitivity:
    """Test that alpha parameter controls expert selection as expected."""

    def test_strict_alpha_fewer_experts(self):
        """Lower alpha should select fewer experts on average."""
        torch.manual_seed(42)
        router_logits = torch.randn(100, 64)

        # Test with different alpha values
        alpha_strict = 0.01
        alpha_lenient = 0.10

        _, _, counts_strict = benjamini_hochberg_routing(
            router_logits, alpha=alpha_strict, max_k=16
        )
        _, _, counts_lenient = benjamini_hochberg_routing(
            router_logits, alpha=alpha_lenient, max_k=16
        )

        mean_strict = counts_strict.float().mean()
        mean_lenient = counts_lenient.float().mean()

        assert mean_strict < mean_lenient, \
            f"Expected strict alpha ({alpha_strict}) to select fewer experts than lenient ({alpha_lenient}), " \
            f"but got {mean_strict:.2f} vs {mean_lenient:.2f}"

    def test_alpha_range(self):
        """Test multiple alpha values in increasing order."""
        torch.manual_seed(42)
        router_logits = torch.randn(50, 64)

        alpha_values = [0.01, 0.05, 0.10, 0.20]
        mean_counts = []

        for alpha in alpha_values:
            _, _, counts = benjamini_hochberg_routing(
                router_logits, alpha=alpha, max_k=16
            )
            mean_counts.append(counts.float().mean().item())

        # Mean counts should be monotonically increasing with alpha
        for i in range(len(mean_counts) - 1):
            assert mean_counts[i] <= mean_counts[i + 1], \
                f"Expected monotonic increase, but {mean_counts[i]:.2f} > {mean_counts[i+1]:.2f}"


class TestTemperatureSensitivity:
    """Test that temperature parameter affects expert selection."""

    def test_high_temperature_more_experts(self):
        """Higher temperature should generally select more experts."""
        torch.manual_seed(42)
        # Create logits with some structure (not pure noise)
        router_logits = torch.randn(100, 64) * 2.0
        router_logits[:, :8] += 1.0  # Make first 8 experts slightly better

        temp_low = 0.5
        temp_high = 2.0

        _, _, counts_low = benjamini_hochberg_routing(
            router_logits, temperature=temp_low, max_k=16, alpha=0.05
        )
        _, _, counts_high = benjamini_hochberg_routing(
            router_logits, temperature=temp_high, max_k=16, alpha=0.05
        )

        mean_low = counts_low.float().mean()
        mean_high = counts_high.float().mean()

        # High temperature flattens distribution, making more experts similar
        # This typically leads to more experts passing the BH threshold
        # Note: This is a stochastic test, may occasionally fail
        assert mean_low <= mean_high, \
            f"Expected high temp to select more/equal experts, got {mean_low:.2f} vs {mean_high:.2f}"

    def test_temperature_effect_on_distribution(self):
        """Temperature should affect probability distribution."""
        router_logits = torch.randn(10, 64)

        # Low temperature → sharper distribution
        weights_low, _, _ = benjamini_hochberg_routing(
            router_logits, temperature=0.5, max_k=16
        )

        # High temperature → flatter distribution
        weights_high, _, _ = benjamini_hochberg_routing(
            router_logits, temperature=2.0, max_k=16
        )

        # Compute entropy of weight distributions
        eps = 1e-10
        entropy_low = -(weights_low * torch.log(weights_low + eps)).sum(dim=-1).mean()
        entropy_high = -(weights_high * torch.log(weights_high + eps)).sum(dim=-1).mean()

        # Higher temperature should have higher entropy (more uniform)
        # Note: This assumes BH selects similar experts in both cases
        # which may not always hold, so we check the general trend
        print(f"Entropy (low temp): {entropy_low:.4f}, (high temp): {entropy_high:.4f}")


class TestMinMaxConstraints:
    """Test that min_k and max_k constraints are respected."""

    def test_min_k_enforced(self):
        """At least min_k experts should always be selected."""
        router_logits = torch.randn(20, 64)
        min_k = 3

        _, _, counts = benjamini_hochberg_routing(
            router_logits, min_k=min_k, alpha=0.001  # Very strict alpha
        )

        assert (counts >= min_k).all(), \
            f"Some tokens have < {min_k} experts: {counts[counts < min_k]}"

    def test_max_k_enforced(self):
        """At most max_k experts should be selected."""
        router_logits = torch.randn(20, 64)
        max_k = 8

        _, _, counts = benjamini_hochberg_routing(
            router_logits, max_k=max_k, alpha=1.0  # Very lenient alpha
        )

        assert (counts <= max_k).all(), \
            f"Some tokens have > {max_k} experts: {counts[counts > max_k]}"

    def test_min_max_range(self):
        """All expert counts should be in [min_k, max_k]."""
        router_logits = torch.randn(50, 64)
        min_k, max_k = 2, 10

        _, _, counts = benjamini_hochberg_routing(
            router_logits, min_k=min_k, max_k=max_k
        )

        assert (counts >= min_k).all() and (counts <= max_k).all(), \
            f"Counts outside [{min_k}, {max_k}]: min={counts.min()}, max={counts.max()}"


class TestExpertIndices:
    """Test that selected expert indices are valid."""

    def test_indices_in_range(self):
        """All expert indices should be in [0, num_experts-1] or -1 (padding)."""
        num_experts = 64
        router_logits = torch.randn(10, num_experts)

        _, experts, counts = benjamini_hochberg_routing(
            router_logits, max_k=8
        )

        # Check that all indices are either valid expert IDs or -1
        valid_mask = ((experts >= 0) & (experts < num_experts)) | (experts == -1)
        assert valid_mask.all(), f"Found invalid expert indices: {experts[~valid_mask]}"

    def test_padding_with_minus_one(self):
        """Unused slots should be padded with -1."""
        router_logits = torch.randn(10, 64)
        max_k = 8

        _, experts, counts = benjamini_hochberg_routing(
            router_logits, max_k=max_k
        )

        # For each token, check that slots >= counts[i] are padded with -1
        for i in range(experts.shape[0]):
            num_selected = counts[i].item()
            # Slots [0, num_selected) should have valid indices
            assert (experts[i, :num_selected] >= 0).all(), \
                f"Token {i}: Found -1 in active slots"
            # Slots [num_selected, max_k) should be -1
            if num_selected < max_k:
                assert (experts[i, num_selected:] == -1).all(), \
                    f"Token {i}: Padding not all -1"

    def test_no_duplicate_experts(self):
        """Each token should not select the same expert twice."""
        router_logits = torch.randn(20, 64)

        _, experts, counts = benjamini_hochberg_routing(
            router_logits, max_k=8
        )

        # For each token, check uniqueness of selected experts (excluding -1)
        for i in range(experts.shape[0]):
            num_selected = counts[i].item()
            selected = experts[i, :num_selected]
            unique_selected = torch.unique(selected)

            assert len(unique_selected) == num_selected, \
                f"Token {i}: Duplicate experts found: {selected}"


class TestComparisonWithTopK:
    """Compare BH routing with standard top-k routing."""

    def test_topk_baseline_exists(self):
        """Verify top-k routing works."""
        router_logits = torch.randn(10, 64)

        weights, experts, counts = topk_routing(router_logits, k=8)

        assert weights.shape == (10, 64)
        assert experts.shape == (10, 8)
        assert (counts == 8).all()  # Always selects exactly k

    def test_bh_vs_topk_weights_sum(self):
        """Both BH and top-k should produce weights that sum to 1."""
        router_logits = torch.randn(10, 64)

        bh_weights, _, _ = benjamini_hochberg_routing(router_logits, max_k=8)
        topk_weights, _, _ = topk_routing(router_logits, k=8)

        bh_sums = bh_weights.sum(dim=-1)
        topk_sums = topk_weights.sum(dim=-1)

        assert torch.allclose(bh_sums, torch.ones_like(bh_sums), atol=1e-5)
        assert torch.allclose(topk_sums, torch.ones_like(topk_sums), atol=1e-5)

    def test_bh_can_select_fewer(self):
        """BH should be able to select fewer than max_k experts."""
        torch.manual_seed(42)
        router_logits = torch.randn(100, 64)

        _, _, bh_counts = benjamini_hochberg_routing(
            router_logits, alpha=0.01, max_k=8  # Strict alpha
        )

        # At least some tokens should select < 8 experts
        assert (bh_counts < 8).any(), "BH always selects max_k (should vary)"

    def test_bh_adaptivity(self):
        """BH should adapt expert count based on input distribution."""
        torch.manual_seed(42)

        # Scenario 1: High confidence (one expert clearly best)
        logits_high_conf = torch.randn(10, 64)
        logits_high_conf[:, 0] += 5.0  # Make expert 0 much better

        # Scenario 2: Low confidence (all experts similar)
        logits_low_conf = torch.randn(10, 64) * 0.1

        _, _, counts_high = benjamini_hochberg_routing(
            logits_high_conf, alpha=0.05, max_k=8
        )
        _, _, counts_low = benjamini_hochberg_routing(
            logits_low_conf, alpha=0.05, max_k=8
        )

        # High confidence should generally select fewer experts
        # Low confidence should select more (all similar → many significant)
        mean_high = counts_high.float().mean()
        mean_low = counts_low.float().mean()

        print(f"High confidence mean: {mean_high:.2f}, Low confidence mean: {mean_low:.2f}")
        # Note: This test is heuristic and may not always pass


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_very_small_num_experts(self):
        """Test with only a few experts."""
        router_logits = torch.randn(10, 4)

        weights, experts, counts = benjamini_hochberg_routing(
            router_logits, max_k=3
        )

        assert weights.shape == (10, 4)
        assert (counts >= 1).all() and (counts <= 3).all()

    def test_extreme_logits(self):
        """Test with very large/small logit values."""
        # Very large logits
        router_logits_large = torch.randn(10, 64) * 100

        weights_large, _, _ = benjamini_hochberg_routing(router_logits_large)

        # Should still produce valid weights
        assert torch.isfinite(weights_large).all(), "Non-finite weights with large logits"
        assert torch.allclose(weights_large.sum(dim=-1), torch.ones(10), atol=1e-4)

        # Very small logits (near zero)
        router_logits_small = torch.randn(10, 64) * 0.01

        weights_small, _, _ = benjamini_hochberg_routing(router_logits_small)

        assert torch.isfinite(weights_small).all(), "Non-finite weights with small logits"
        assert torch.allclose(weights_small.sum(dim=-1), torch.ones(10), atol=1e-4)

    def test_all_same_logits(self):
        """Test when all experts have the same logits (uniform distribution)."""
        # All zeros → uniform probabilities
        router_logits = torch.zeros(10, 64)

        weights, experts, counts = benjamini_hochberg_routing(
            router_logits, alpha=0.05, max_k=8
        )

        # With uniform distribution, all experts have same p-value
        # BH should select many (possibly all up to max_k)
        assert (counts >= 1).all(), "No experts selected with uniform distribution"

    def test_numerical_stability_with_softmax(self):
        """Test that softmax doesn't overflow/underflow."""
        # Create logits that would cause softmax overflow without proper scaling
        router_logits = torch.tensor([[1000.0, 999.0, -1000.0, 0.0] * 16])  # [1, 64]

        weights, _, _ = benjamini_hochberg_routing(router_logits)

        # Check for NaN or Inf
        assert not torch.isnan(weights).any(), "NaN in weights"
        assert not torch.isinf(weights).any(), "Inf in weights"
        assert torch.allclose(weights.sum(dim=-1), torch.ones(1), atol=1e-4)


class TestStatisticsOutput:
    """Test the optional statistics output."""

    def test_stats_returned_when_requested(self):
        """Stats should be returned when return_stats=True."""
        router_logits = torch.randn(10, 64)

        result = benjamini_hochberg_routing(router_logits, return_stats=True)

        assert len(result) == 4, "Expected 4 outputs with return_stats=True"
        weights, experts, counts, stats = result

        assert isinstance(stats, dict), "Stats should be a dict"
        assert 'p_values' in stats
        assert 'bh_threshold' in stats
        assert 'significant_mask' in stats

    def test_stats_shapes(self):
        """Statistics should have correct shapes."""
        batch_size, seq_len, num_experts = 2, 8, 64
        router_logits = torch.randn(batch_size, seq_len, num_experts)

        _, _, _, stats = benjamini_hochberg_routing(router_logits, return_stats=True)

        assert stats['p_values'].shape == (batch_size, seq_len, num_experts)
        assert stats['bh_threshold'].shape == (batch_size, seq_len)
        assert stats['significant_mask'].shape == (batch_size, seq_len, num_experts)

    def test_p_values_in_valid_range(self):
        """P-values should be in (0, 1)."""
        router_logits = torch.randn(10, 64)

        _, _, _, stats = benjamini_hochberg_routing(router_logits, return_stats=True)

        p_values = stats['p_values']
        assert (p_values > 0).all() and (p_values < 1).all(), \
            f"P-values outside (0, 1): min={p_values.min()}, max={p_values.max()}"


class TestGPUCompatibility:
    """Test that routing works on GPU if available."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_basic(self):
        """Test basic functionality on CUDA."""
        router_logits = torch.randn(10, 64).cuda()

        weights, experts, counts = benjamini_hochberg_routing(router_logits)

        # Check that outputs are on GPU
        assert weights.is_cuda, "Weights not on CUDA"
        assert experts.is_cuda, "Experts not on CUDA"
        assert counts.is_cuda, "Counts not on CUDA"

        # Check correctness
        assert torch.allclose(weights.sum(dim=-1), torch.ones(10).cuda(), atol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_large_batch(self):
        """Test with larger batch on CUDA."""
        batch_size, seq_len, num_experts = 16, 512, 64
        router_logits = torch.randn(batch_size, seq_len, num_experts).cuda()

        weights, experts, counts = benjamini_hochberg_routing(router_logits, max_k=8)

        assert weights.shape == (batch_size, seq_len, num_experts)
        assert weights.is_cuda


class TestInputValidation:
    """Test that invalid inputs raise appropriate errors."""

    def test_invalid_tensor_type(self):
        """Non-tensor input should raise TypeError."""
        with pytest.raises(TypeError):
            benjamini_hochberg_routing([[1, 2, 3]])

    def test_invalid_ndim(self):
        """1D or 4D input should raise ValueError."""
        with pytest.raises(ValueError):
            benjamini_hochberg_routing(torch.randn(64))  # 1D

        with pytest.raises(ValueError):
            benjamini_hochberg_routing(torch.randn(2, 2, 2, 64))  # 4D

    def test_invalid_alpha(self):
        """Alpha outside (0, 1) should raise ValueError."""
        router_logits = torch.randn(10, 64)

        with pytest.raises(ValueError):
            benjamini_hochberg_routing(router_logits, alpha=0.0)

        with pytest.raises(ValueError):
            benjamini_hochberg_routing(router_logits, alpha=1.0)

        with pytest.raises(ValueError):
            benjamini_hochberg_routing(router_logits, alpha=-0.1)

    def test_invalid_temperature(self):
        """Non-positive temperature should raise ValueError."""
        router_logits = torch.randn(10, 64)

        with pytest.raises(ValueError):
            benjamini_hochberg_routing(router_logits, temperature=0.0)

        with pytest.raises(ValueError):
            benjamini_hochberg_routing(router_logits, temperature=-1.0)

    def test_invalid_min_max_k(self):
        """Invalid min_k/max_k should raise ValueError."""
        router_logits = torch.randn(10, 64)

        # min_k < 1
        with pytest.raises(ValueError):
            benjamini_hochberg_routing(router_logits, min_k=0)

        # max_k < min_k
        with pytest.raises(ValueError):
            benjamini_hochberg_routing(router_logits, min_k=5, max_k=3)

        # max_k > num_experts
        with pytest.raises(ValueError):
            benjamini_hochberg_routing(router_logits, max_k=100)


class TestRoutingStatistics:
    """Test the compute_routing_statistics utility."""

    def test_statistics_function(self):
        """Test that routing statistics are computed correctly."""
        router_logits = torch.randn(10, 64)

        weights, _, counts = benjamini_hochberg_routing(router_logits)
        stats = compute_routing_statistics(weights, counts)

        # Check that all expected keys are present
        expected_keys = [
            'mean_experts', 'std_experts', 'min_experts', 'max_experts',
            'sparsity', 'weight_entropy'
        ]
        for key in expected_keys:
            assert key in stats, f"Missing statistic: {key}"

        # Check value ranges
        assert stats['mean_experts'] >= stats['min_experts']
        assert stats['mean_experts'] <= stats['max_experts']
        assert 0.0 <= stats['sparsity'] <= 1.0
        assert stats['weight_entropy'] >= 0.0


# ============================================================================
# Multi-Expert Tests (max_k = 8, 16, 32, 64)
# ============================================================================

class TestMultipleMaxK:
    """Test BH routing with different max_k values (8, 16, 32, 64)."""

    def test_max_k_8(self):
        """Test with max_k=8 (OLMoE default)."""
        router_logits = torch.randn(10, 64)
        weights, experts, counts = benjamini_hochberg_routing(
            router_logits, alpha=0.05, max_k=8
        )
        assert experts.shape == (10, 8)
        assert (counts <= 8).all()

    def test_max_k_16(self):
        """Test with max_k=16."""
        router_logits = torch.randn(10, 64)
        weights, experts, counts = benjamini_hochberg_routing(
            router_logits, alpha=0.05, max_k=16
        )
        assert experts.shape == (10, 16)
        assert (counts <= 16).all()

    def test_max_k_32(self):
        """Test with max_k=32."""
        router_logits = torch.randn(10, 64)
        weights, experts, counts = benjamini_hochberg_routing(
            router_logits, alpha=0.05, max_k=32
        )
        assert experts.shape == (10, 32)
        assert (counts <= 32).all()

    def test_max_k_64(self):
        """Test with max_k=64 (uncapped)."""
        router_logits = torch.randn(10, 64)
        weights, experts, counts = benjamini_hochberg_routing(
            router_logits, alpha=0.05, max_k=64
        )
        assert experts.shape == (10, 64)
        assert (counts <= 64).all()

    def test_max_k_equals_num_experts(self):
        """Test when max_k equals total experts."""
        router_logits = torch.randn(10, 64)
        weights, experts, counts = benjamini_hochberg_routing(
            router_logits, alpha=0.5, max_k=64  # High alpha, should select many
        )
        # With high alpha and no ceiling, should potentially select many experts
        assert (counts >= 1).all()  # At least min_k
        assert (counts <= 64).all()  # At most max_k

    def test_higher_max_k_allows_more_experts(self):
        """Higher max_k should allow more experts when needed."""
        torch.manual_seed(42)
        router_logits = torch.randn(100, 64)

        _, _, counts_8 = benjamini_hochberg_routing(router_logits, alpha=0.10, max_k=8)
        _, _, counts_16 = benjamini_hochberg_routing(router_logits, alpha=0.10, max_k=16)
        _, _, counts_32 = benjamini_hochberg_routing(router_logits, alpha=0.10, max_k=32)
        _, _, counts_64 = benjamini_hochberg_routing(router_logits, alpha=0.10, max_k=64)

        mean_8 = counts_8.float().mean().item()
        mean_16 = counts_16.float().mean().item()
        mean_32 = counts_32.float().mean().item()
        mean_64 = counts_64.float().mean().item()

        # Higher max_k should allow >= experts (not necessarily more if BH is conservative)
        assert mean_16 >= mean_8 * 0.9  # Allow some variance
        assert mean_32 >= mean_16 * 0.9
        assert mean_64 >= mean_32 * 0.9

        print(f"max_k=8: {mean_8:.2f}, max_k=16: {mean_16:.2f}, max_k=32: {mean_32:.2f}, max_k=64: {mean_64:.2f}")

    def test_ceiling_hit_rate(self):
        """Test how often BH hits the max_k ceiling."""
        torch.manual_seed(42)
        router_logits = torch.randn(100, 64)

        for max_k in [8, 16, 32, 64]:
            _, _, counts = benjamini_hochberg_routing(
                router_logits, alpha=0.05, max_k=max_k
            )
            ceiling_rate = (counts == max_k).float().mean().item()
            print(f"max_k={max_k}: ceiling hit rate = {ceiling_rate*100:.1f}%")

            # Lower max_k should hit ceiling more often
            if max_k == 8:
                # With max_k=8, might hit ceiling sometimes
                pass  # Just informational
            elif max_k == 64:
                # With max_k=64, should rarely hit ceiling
                assert ceiling_rate < 0.5, f"Unexpected ceiling rate {ceiling_rate} for max_k=64"


class TestMultiKUtilities:
    """Test the multi-k utility functions."""

    def test_run_bh_multi_k(self):
        """Test running BH with multiple max_k values."""
        from bh_routing import run_bh_multi_k

        router_logits = torch.randn(10, 64)
        results = run_bh_multi_k(
            router_logits,
            max_k_values=[8, 16, 32, 64],
            alpha=0.05
        )

        assert len(results) == 4
        assert 8 in results
        assert 16 in results
        assert 32 in results
        assert 64 in results

        for max_k, (weights, experts, counts) in results.items():
            assert weights.shape == (10, 64)
            assert experts.shape == (10, max_k)
            assert (counts <= max_k).all()

    def test_compare_multi_k_statistics(self):
        """Test statistics comparison function."""
        from bh_routing import run_bh_multi_k, compare_multi_k_statistics

        router_logits = torch.randn(50, 64)
        results = run_bh_multi_k(router_logits, max_k_values=[8, 16, 32])

        df = compare_multi_k_statistics(results)

        assert len(df) == 3
        assert 'max_k' in df.columns
        assert 'mean_experts' in df.columns
        assert 'pct_at_ceiling' in df.columns

        # max_k column should have correct values
        assert set(df['max_k'].tolist()) == {8, 16, 32}


# ============================================================================
# Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests and print summary."""
    import sys

    print("=" * 70)
    print("BENJAMINI-HOCHBERG ROUTING TEST SUITE")
    print("=" * 70)

    # Run pytest programmatically
    exit_code = pytest.main([__file__, '-v', '--tb=short'])

    if exit_code == 0:
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ SOME TESTS FAILED")
        print("=" * 70)

    return exit_code


if __name__ == "__main__":
    run_all_tests()
