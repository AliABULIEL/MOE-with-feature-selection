#!/usr/bin/env python3
"""
Comprehensive Tests for BH Routing Implementation

Tests the benjamini_hochberg_routing function with concrete examples
and verifies each step of the algorithm.

Run with:
    python test_bh_routing_comprehensive.py
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deprecated.bh_routing import benjamini_hochberg_routing, compute_pvalues_empirical


class BHRoutingTests:
    """Test suite for BH routing with concrete examples."""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.tolerance = 1e-3  # Tolerance for floating point comparisons
    
    def assert_close(self, actual, expected, name, tolerance=None):
        """Assert that actual is close to expected within tolerance."""
        tol = tolerance or self.tolerance
        
        if isinstance(expected, (list, tuple)):
            expected = torch.tensor(expected, dtype=torch.float32)
        if isinstance(actual, torch.Tensor):
            actual = actual.float()
        
        if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
            if actual.shape != expected.shape:
                print(f"  ❌ {name}: Shape mismatch - got {actual.shape}, expected {expected.shape}")
                self.tests_failed += 1
                return False
            
            max_diff = (actual - expected).abs().max().item()
            if max_diff <= tol:
                print(f"  ✓ {name}: PASS (max diff: {max_diff:.6f})")
                self.tests_passed += 1
                return True
            else:
                print(f"  ❌ {name}: FAIL (max diff: {max_diff:.6f})")
                print(f"      Expected: {expected.tolist()}")
                print(f"      Actual:   {actual.tolist()}")
                self.tests_failed += 1
                return False
        else:
            diff = abs(actual - expected)
            if diff <= tol:
                print(f"  ✓ {name}: PASS (diff: {diff:.6f})")
                self.tests_passed += 1
                return True
            else:
                print(f"  ❌ {name}: FAIL - expected {expected}, got {actual}")
                self.tests_failed += 1
                return False
    
    def assert_equal(self, actual, expected, name):
        """Assert exact equality for integers/lists."""
        if isinstance(actual, torch.Tensor):
            actual = actual.tolist()
        if isinstance(expected, torch.Tensor):
            expected = expected.tolist()
        
        if actual == expected:
            print(f"  ✓ {name}: PASS")
            self.tests_passed += 1
            return True
        else:
            print(f"  ❌ {name}: FAIL - expected {expected}, got {actual}")
            self.tests_failed += 1
            return False

    # =========================================================================
    # TEST 1: Basic Example with α = 0.25
    # =========================================================================
    def test_basic_example_alpha_025(self):
        """Test with α=0.25, expecting 1 expert selected."""
        print("\n" + "=" * 70)
        print("TEST 1: Basic Example (α = 0.25)")
        print("=" * 70)
        
        # Setup
        router_logits = torch.tensor([[2.5, 1.8, 0.5, -0.3, -0.8, -1.2, -1.5, -2.0]])
        alpha = 0.25
        max_k = 4
        min_k = 1
        num_experts = 8
        
        print(f"\nInput logits: {router_logits[0].tolist()}")
        print(f"Alpha: {alpha}, min_k: {min_k}, max_k: {max_k}")
        
        # Run BH routing (without KDE - will use empirical fallback)
        routing_weights, selected_experts, expert_counts, stats = benjamini_hochberg_routing(
            router_logits,
            alpha=alpha,
            temperature=1.0,
            min_k=min_k,
            max_k=max_k,
            kde_models={},  # Force empirical p-values
            return_stats=True
        )
        
        print(f"\nResults:")
        print(f"  Expert counts: {expert_counts.tolist()}")
        print(f"  Selected experts: {selected_experts.tolist()}")
        print(f"  Routing weights: {[f'{w:.4f}' for w in routing_weights[0].tolist()]}")
        
        # Expected: With empirical p-values and strict alpha, only top expert passes
        print(f"\nVerifying:")
        
        # Expert 0 should definitely be selected (highest logit)
        self.assert_equal(selected_experts[0, 0].item(), 0, "Expert 0 selected first")
        
        # Weights should sum to 1
        self.assert_close(routing_weights.sum(dim=-1), torch.tensor([1.0]), "Total weight sums to 1")
        
        # Expert count should be at least min_k
        self.assert_equal(expert_counts[0].item() >= min_k, True, f"Count >= min_k ({min_k})")
        
        # Expert count should be at most max_k
        self.assert_equal(expert_counts[0].item() <= max_k, True, f"Count <= max_k ({max_k})")

    # =========================================================================
    # TEST 2: Example with α = 0.50 (More Permissive)
    # =========================================================================
    def test_alpha_050(self):
        """Test with α=0.50, expecting more experts selected."""
        print("\n" + "=" * 70)
        print("TEST 2: Higher Alpha (α = 0.50)")
        print("=" * 70)
        
        router_logits = torch.tensor([[2.5, 1.8, 0.5, -0.3, -0.8, -1.2, -1.5, -2.0]])
        alpha = 0.50
        max_k = 4
        min_k = 1
        
        print(f"\nInput logits: {router_logits[0].tolist()}")
        print(f"Alpha: {alpha}")
        
        routing_weights, selected_experts, expert_counts, stats = benjamini_hochberg_routing(
            router_logits,
            alpha=alpha,
            temperature=1.0,
            min_k=min_k,
            max_k=max_k,
            kde_models={},
            return_stats=True
        )
        
        print(f"\nResults:")
        print(f"  Expert counts: {expert_counts.tolist()}")
        print(f"  Selected experts: {selected_experts[0].tolist()}")
        print(f"  Routing weights: {[f'{w:.4f}' for w in routing_weights[0].tolist()]}")
        
        print(f"\nVerifying:")
        
        # With higher alpha, should select more experts
        count_alpha_050 = expert_counts[0].item()
        
        # Compare with lower alpha
        _, _, expert_counts_low, _ = benjamini_hochberg_routing(
            router_logits, alpha=0.25, min_k=min_k, max_k=max_k, kde_models={}, return_stats=True
        )
        count_alpha_025 = expert_counts_low[0].item()
        
        self.assert_equal(count_alpha_050 >= count_alpha_025, True, 
                         f"α=0.50 ({count_alpha_050}) >= α=0.25 ({count_alpha_025})")
        
        # Check weights sum to 1
        self.assert_close(routing_weights.sum(dim=-1), torch.tensor([1.0]), "Total weight sums to 1")
        
        # Expert 0 should have highest weight (highest logit)
        max_weight_idx = routing_weights[0].argmax().item()
        self.assert_equal(max_weight_idx, 0, "Expert 0 has highest weight")

    # =========================================================================
    # TEST 3: Ceiling Hit (α = 0.90)
    # =========================================================================
    def test_ceiling_hit(self):
        """Test that max_k constraint is enforced when BH selects more."""
        print("\n" + "=" * 70)
        print("TEST 3: Ceiling Hit (α = 0.90, max_k = 4)")
        print("=" * 70)
        
        router_logits = torch.tensor([[2.5, 1.8, 0.5, -0.3, -0.8, -1.2, -1.5, -2.0]])
        alpha = 0.90
        max_k = 4
        min_k = 1
        
        print(f"\nInput logits: {router_logits[0].tolist()}")
        print(f"Alpha: {alpha}, max_k: {max_k}")
        
        routing_weights, selected_experts, expert_counts, stats = benjamini_hochberg_routing(
            router_logits,
            alpha=alpha,
            temperature=1.0,
            min_k=min_k,
            max_k=max_k,
            kde_models={},
            return_stats=True
        )
        
        print(f"\nResults:")
        print(f"  Expert counts: {expert_counts.tolist()}")
        print(f"  Selected experts: {selected_experts[0].tolist()}")
        
        print(f"\nVerifying:")
        
        # With high alpha, BH would want many experts, but max_k caps it
        self.assert_equal(expert_counts[0].item() <= max_k, True, f"Count <= max_k ({max_k})")
        
        # Should select top experts by p-value (= top by logit with empirical method)
        selected_set = set(selected_experts[0].tolist()) - {-1}
        print(f"  Selected expert set: {selected_set}")
        
        # Weights should sum to 1
        self.assert_close(routing_weights.sum(dim=-1), torch.tensor([1.0]), "Total weight sums to 1")
        
        # Non-selected experts should have 0 weight
        non_selected_weight = routing_weights[0, 4:].sum().item()
        self.assert_close(non_selected_weight, 0.0, "Non-selected weights are 0", tolerance=1e-6)

    # =========================================================================
    # TEST 4: Floor Hit (Very Low Alpha)
    # =========================================================================
    def test_floor_hit(self):
        """Test that min_k constraint is enforced when BH selects too few."""
        print("\n" + "=" * 70)
        print("TEST 4: Floor Hit (α = 0.001, min_k = 2)")
        print("=" * 70)
        
        router_logits = torch.tensor([[2.5, 1.8, 0.5, -0.3, -0.8, -1.2, -1.5, -2.0]])
        alpha = 0.001  # Very strict
        max_k = 4
        min_k = 2  # Force at least 2
        
        print(f"\nInput logits: {router_logits[0].tolist()}")
        print(f"Alpha: {alpha}, min_k: {min_k}")
        
        routing_weights, selected_experts, expert_counts, stats = benjamini_hochberg_routing(
            router_logits,
            alpha=alpha,
            temperature=1.0,
            min_k=min_k,
            max_k=max_k,
            kde_models={},
            return_stats=True
        )
        
        print(f"\nResults:")
        print(f"  Expert counts: {expert_counts.tolist()}")
        print(f"  Selected experts: {selected_experts[0].tolist()}")
        
        print(f"\nVerifying:")
        
        # With very strict alpha, few would pass, but min_k forces at least 2
        self.assert_equal(expert_counts[0].item() >= min_k, True, f"Count >= min_k ({min_k})")
        
        # Weights should sum to 1
        self.assert_close(routing_weights.sum(dim=-1), torch.tensor([1.0]), "Total weight sums to 1")

    # =========================================================================
    # TEST 5: Batch Processing (2D input)
    # =========================================================================
    def test_batch_processing(self):
        """Test that batch processing works correctly."""
        print("\n" + "=" * 70)
        print("TEST 5: Batch Processing (3 tokens)")
        print("=" * 70)
        
        # 3 tokens with different patterns
        router_logits = torch.tensor([
            [3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0],  # Token 0: clear winner
            [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2],    # Token 1: more uniform
            [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],  # Token 2: alternating
        ])
        
        alpha = 0.50
        max_k = 4
        
        print(f"\nInput logits shape: {router_logits.shape}")
        print(f"Alpha: {alpha}")
        
        routing_weights, selected_experts, expert_counts, stats = benjamini_hochberg_routing(
            router_logits,
            alpha=alpha,
            temperature=1.0,
            min_k=1,
            max_k=max_k,
            kde_models={},
            return_stats=True
        )
        
        print(f"\nResults:")
        print(f"  Expert counts: {expert_counts.tolist()}")
        print(f"  Shapes: weights={routing_weights.shape}, experts={selected_experts.shape}")
        
        print(f"\nVerifying:")
        
        # Check shapes
        self.assert_equal(list(routing_weights.shape), [3, 8], "Routing weights shape")
        self.assert_equal(list(selected_experts.shape), [3, max_k], "Selected experts shape")
        self.assert_equal(list(expert_counts.shape), [3], "Expert counts shape")
        
        # Check weights sum to 1 for each token
        weight_sums = routing_weights.sum(dim=-1)
        self.assert_close(weight_sums, torch.tensor([1.0, 1.0, 1.0]), "All weight sums are 1")
        
        # Expert counts should be within bounds
        for i in range(3):
            count = expert_counts[i].item()
            in_bounds = 1 <= count <= max_k
            self.assert_equal(in_bounds, True, f"Token {i} count ({count}) in bounds [1, {max_k}]")

    # =========================================================================
    # TEST 6: 3D Input (batch_size, seq_len, num_experts)
    # =========================================================================
    def test_3d_input(self):
        """Test with 3D input tensor."""
        print("\n" + "=" * 70)
        print("TEST 6: 3D Input (batch=2, seq=3, experts=8)")
        print("=" * 70)
        
        batch_size = 2
        seq_len = 3
        num_experts = 8
        
        torch.manual_seed(42)
        router_logits = torch.randn(batch_size, seq_len, num_experts)
        
        alpha = 0.40
        max_k = 4
        
        print(f"\nInput shape: {router_logits.shape}")
        
        routing_weights, selected_experts, expert_counts, stats = benjamini_hochberg_routing(
            router_logits,
            alpha=alpha,
            temperature=1.0,
            min_k=1,
            max_k=max_k,
            kde_models={},
            return_stats=True
        )
        
        print(f"\nResults:")
        print(f"  Routing weights shape: {routing_weights.shape}")
        print(f"  Selected experts shape: {selected_experts.shape}")
        print(f"  Expert counts shape: {expert_counts.shape}")
        
        print(f"\nVerifying:")
        
        self.assert_equal(list(routing_weights.shape), [batch_size, seq_len, num_experts], "Weights shape")
        self.assert_equal(list(selected_experts.shape), [batch_size, seq_len, max_k], "Experts shape")
        self.assert_equal(list(expert_counts.shape), [batch_size, seq_len], "Counts shape")
        
        # All weights should sum to 1
        weight_sums = routing_weights.sum(dim=-1)
        expected_sums = torch.ones(batch_size, seq_len)
        self.assert_close(weight_sums, expected_sums, "All weight sums are 1")

    # =========================================================================
    # TEST 7: P-Value Ordering (Single Token)
    # =========================================================================
    def test_pvalue_ordering_single_token(self):
        """Test that higher logits produce lower p-values with single token."""
        print("\n" + "=" * 70)
        print("TEST 7: P-Value Ordering (Single Token)")
        print("=" * 70)
        
        # Logits in strictly decreasing order - SINGLE TOKEN
        router_logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0]])
        
        # Compute empirical p-values directly
        p_values = compute_pvalues_empirical(router_logits)
        
        print(f"\nLogits: {router_logits[0].tolist()}")
        print(f"P-values: {[f'{p:.4f}' for p in p_values[0].tolist()]}")
        
        # Expected p-values for single token (across-expert ranking):
        # Logit 4.0 -> rank 8/8 -> CDF 1.0 -> p ≈ 0 (clamped to eps)
        # Logit 3.0 -> rank 7/8 -> CDF 0.875 -> p = 0.125
        # Logit -3.0 -> rank 1/8 -> CDF 0.125 -> p = 0.875
        expected_ordering = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
        
        print(f"Expected p-values (approx): {expected_ordering}")
        
        print(f"\nVerifying:")
        
        # P-values should be in strictly increasing order (opposite of logits)
        p_list = p_values[0].tolist()
        is_increasing = all(p_list[i] <= p_list[i+1] for i in range(len(p_list)-1))
        self.assert_equal(is_increasing, True, "P-values increase as logits decrease")
        
        # Highest logit should have lowest p-value
        self.assert_equal(p_values[0].argmin().item(), 0, "Highest logit has lowest p-value")
        
        # Lowest logit should have highest p-value
        self.assert_equal(p_values[0].argmax().item(), 7, "Lowest logit has highest p-value")
        
        # Check approximate values (with tolerance for clamping)
        self.assert_close(p_values[0, 1], 0.125, "P-value for 2nd highest logit", tolerance=0.01)
        self.assert_close(p_values[0, 7], 0.875, "P-value for lowest logit", tolerance=0.01)

    # =========================================================================
    # TEST 7b: P-Value Ordering (Multiple Tokens)
    # =========================================================================
    def test_pvalue_ordering_multi_token(self):
        """Test p-value behavior with multiple tokens."""
        print("\n" + "=" * 70)
        print("TEST 7b: P-Value Ordering (Multiple Tokens)")
        print("=" * 70)
        
        # Multiple tokens with varying patterns
        router_logits = torch.tensor([
            [4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0],
            [3.5, 2.5, 1.5, 0.5, -0.5, -1.5, -2.5, -3.5],
            [4.5, 3.5, 2.5, 1.5, 0.5, -0.5, -1.5, -2.5],
        ])
        
        p_values = compute_pvalues_empirical(router_logits)
        
        print(f"\nInput shape: {router_logits.shape}")
        print(f"P-values shape: {p_values.shape}")
        
        for i in range(3):
            print(f"Token {i} p-values: {[f'{p:.3f}' for p in p_values[i].tolist()]}")
        
        print(f"\nVerifying:")
        
        # For each token, higher logit experts should have lower p-values
        for token_idx in range(3):
            logits = router_logits[token_idx]
            pvals = p_values[token_idx]
            
            # Expert with highest logit should have lowest p-value
            highest_logit_expert = logits.argmax().item()
            lowest_pval_expert = pvals.argmin().item()
            self.assert_equal(
                highest_logit_expert, lowest_pval_expert,
                f"Token {token_idx}: highest logit expert = lowest p-value expert"
            )

    # =========================================================================
    # TEST 8: BH Threshold Calculation
    # =========================================================================
    def test_bh_thresholds(self):
        """Test that BH thresholds are computed correctly."""
        print("\n" + "=" * 70)
        print("TEST 8: BH Threshold Calculation")
        print("=" * 70)
        
        num_experts = 8
        alpha = 0.25
        
        # Manual calculation: c_k = (k/N) * alpha
        expected_thresholds = [(k / num_experts) * alpha for k in range(1, num_experts + 1)]
        # [0.03125, 0.0625, 0.09375, 0.125, 0.15625, 0.1875, 0.21875, 0.25]
        
        print(f"\nN={num_experts}, α={alpha}")
        print(f"Expected thresholds: {[f'{t:.5f}' for t in expected_thresholds]}")
        
        # Get thresholds from actual run
        router_logits = torch.tensor([[1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5]])
        
        _, _, _, stats = benjamini_hochberg_routing(
            router_logits,
            alpha=alpha,
            temperature=1.0,
            min_k=1,
            max_k=8,
            kde_models={},
            return_stats=True
        )
        
        actual_thresholds = stats['critical_values'].tolist()
        print(f"Actual thresholds:   {[f'{t:.5f}' for t in actual_thresholds]}")
        
        print(f"\nVerifying:")
        self.assert_close(
            torch.tensor(actual_thresholds), 
            torch.tensor(expected_thresholds), 
            "BH thresholds match formula"
        )

    # =========================================================================
    # TEST 9: Alpha Sensitivity
    # =========================================================================
    def test_alpha_sensitivity(self):
        """Test that higher alpha leads to more experts selected."""
        print("\n" + "=" * 70)
        print("TEST 9: Alpha Sensitivity")
        print("=" * 70)
        
        torch.manual_seed(123)
        router_logits = torch.randn(10, 64)  # 10 tokens, 64 experts
        router_logits[:, :8] += 2.0  # Boost top 8
        
        alphas = [0.10, 0.30, 0.50, 0.70, 0.90]
        avg_experts = []
        
        print(f"\nTesting alpha sensitivity:")
        for alpha in alphas:
            _, _, expert_counts = benjamini_hochberg_routing(
                router_logits,
                alpha=alpha,
                min_k=1,
                max_k=64,  # High ceiling to not constrain
                kde_models={}
            )
            avg = expert_counts.float().mean().item()
            avg_experts.append(avg)
            print(f"  α={alpha:.2f}: avg experts = {avg:.2f}")
        
        print(f"\nVerifying:")
        
        # Higher alpha should generally select more experts
        # Allow some tolerance for statistical variation
        increasing_trend = avg_experts[-1] > avg_experts[0]
        self.assert_equal(increasing_trend, True, 
                         f"α=0.90 ({avg_experts[-1]:.2f}) > α=0.10 ({avg_experts[0]:.2f})")

    # =========================================================================
    # TEST 10: Weight Renormalization
    # =========================================================================
    def test_weight_renormalization(self):
        """Test that weights are properly renormalized after selection."""
        print("\n" + "=" * 70)
        print("TEST 10: Weight Renormalization")
        print("=" * 70)
        
        # Create logits where top experts have clear softmax probs
        router_logits = torch.tensor([[5.0, 3.0, 1.0, -1.0, -3.0, -5.0, -7.0, -9.0]])
        
        actual_weights, selected_experts, expert_counts = benjamini_hochberg_routing(
            router_logits,
            alpha=0.50,
            min_k=1,
            max_k=8,
            kde_models={}
        )
        
        print(f"\nLogits: {router_logits[0].tolist()}")
        print(f"Expert counts: {expert_counts.tolist()}")
        print(f"Routing weights: {[f'{w:.4f}' for w in actual_weights[0].tolist()]}")
        print(f"Weight sum: {actual_weights.sum(dim=-1).item():.6f}")
        
        print(f"\nVerifying:")
        
        # Weights must sum to 1
        self.assert_close(actual_weights.sum(dim=-1), torch.tensor([1.0]), "Weights sum to 1")
        
        # Selected experts should have positive weights
        num_selected = expert_counts[0].item()
        positive_weights = (actual_weights[0] > 0).sum().item()
        self.assert_equal(positive_weights, num_selected, "Num positive weights = num selected")
        
        # Non-selected should be exactly 0
        non_selected_mask = actual_weights[0] == 0
        num_zero = non_selected_mask.sum().item()
        self.assert_equal(num_zero, 8 - num_selected, "Remaining weights are 0")

    # =========================================================================
    # TEST 11: Determinism
    # =========================================================================
    def test_determinism(self):
        """Test that routing is deterministic (same input → same output)."""
        print("\n" + "=" * 70)
        print("TEST 11: Determinism")
        print("=" * 70)
        
        torch.manual_seed(42)
        router_logits = torch.randn(5, 16)
        
        # Run twice
        w1, e1, c1 = benjamini_hochberg_routing(router_logits, alpha=0.40, kde_models={})
        w2, e2, c2 = benjamini_hochberg_routing(router_logits, alpha=0.40, kde_models={})
        
        print(f"\nRun 1 counts: {c1.tolist()}")
        print(f"Run 2 counts: {c2.tolist()}")
        
        print(f"\nVerifying:")
        self.assert_close(w1, w2, "Weights identical across runs")
        self.assert_equal(e1.tolist(), e2.tolist(), "Selected experts identical")
        self.assert_equal(c1.tolist(), c2.tolist(), "Counts identical")

    # =========================================================================
    # TEST 12: Edge Case - All Same Logits
    # =========================================================================
    def test_uniform_logits(self):
        """Test behavior when all logits are identical."""
        print("\n" + "=" * 70)
        print("TEST 12: Edge Case - Uniform Logits")
        print("=" * 70)
        
        # All experts have same logit
        router_logits = torch.ones(1, 8) * 0.5
        
        print(f"\nLogits: {router_logits[0].tolist()}")
        
        routing_weights, selected_experts, expert_counts = benjamini_hochberg_routing(
            router_logits,
            alpha=0.50,
            min_k=1,
            max_k=4,
            kde_models={}
        )
        
        print(f"\nResults:")
        print(f"  Expert counts: {expert_counts.tolist()}")
        print(f"  Routing weights: {[f'{w:.4f}' for w in routing_weights[0].tolist()]}")
        
        print(f"\nVerifying:")
        
        # Should still work and select min_k experts
        self.assert_equal(expert_counts[0].item() >= 1, True, "At least 1 expert selected")
        self.assert_close(routing_weights.sum(dim=-1), torch.tensor([1.0]), "Weights sum to 1")

    # =========================================================================
    # TEST 13: Large Scale (64 experts like OLMoE)
    # =========================================================================
    def test_large_scale(self):
        """Test with 64 experts like OLMoE."""
        print("\n" + "=" * 70)
        print("TEST 13: Large Scale (64 experts)")
        print("=" * 70)
        
        torch.manual_seed(42)
        num_tokens = 100
        num_experts = 64
        
        router_logits = torch.randn(num_tokens, num_experts)
        # Boost top 8 to simulate realistic distribution
        router_logits[:, :8] += 2.0
        
        print(f"\nInput shape: {router_logits.shape}")
        
        routing_weights, selected_experts, expert_counts = benjamini_hochberg_routing(
            router_logits,
            alpha=0.30,
            min_k=1,
            max_k=8,
            kde_models={}
        )
        
        avg_experts = expert_counts.float().mean().item()
        std_experts = expert_counts.float().std().item()
        
        print(f"\nResults:")
        print(f"  Avg experts: {avg_experts:.2f} ± {std_experts:.2f}")
        print(f"  Range: [{expert_counts.min().item()}, {expert_counts.max().item()}]")
        
        print(f"\nVerifying:")
        
        # Shape checks
        self.assert_equal(list(routing_weights.shape), [num_tokens, num_experts], "Weights shape")
        self.assert_equal(list(selected_experts.shape), [num_tokens, 8], "Experts shape")
        
        # All weights sum to 1
        weight_sums = routing_weights.sum(dim=-1)
        expected_sums = torch.ones(num_tokens)
        self.assert_close(weight_sums, expected_sums, "All weight sums are 1")
        
        # Avg experts should be reasonable (not all 1 or all 8)
        reasonable_avg = 1.5 < avg_experts < 7.5
        self.assert_equal(reasonable_avg, True, f"Reasonable avg experts ({avg_experts:.2f})")

    # =========================================================================
    # RUN ALL TESTS
    # =========================================================================
    def run_all(self):
        """Run all tests and print summary."""
        print("\n" + "=" * 70)
        print("BH ROUTING COMPREHENSIVE TEST SUITE")
        print("=" * 70)
        
        self.test_basic_example_alpha_025()
        self.test_alpha_050()
        self.test_ceiling_hit()
        self.test_floor_hit()
        self.test_batch_processing()
        self.test_3d_input()
        self.test_pvalue_ordering_single_token()
        self.test_pvalue_ordering_multi_token()
        self.test_bh_thresholds()
        self.test_alpha_sensitivity()
        self.test_weight_renormalization()
        self.test_determinism()
        self.test_uniform_logits()
        self.test_large_scale()
        
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"  ✓ Passed: {self.tests_passed}")
        print(f"  ✗ Failed: {self.tests_failed}")
        print(f"  Total:   {self.tests_passed + self.tests_failed}")
        
        if self.tests_failed == 0:
            print("\n🎉 ALL TESTS PASSED!")
            return True
        else:
            print(f"\n⚠️  {self.tests_failed} TESTS FAILED")
            return False


if __name__ == "__main__":
    tester = BHRoutingTests()
    success = tester.run_all()
    sys.exit(0 if success else 1)
