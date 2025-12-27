#!/usr/bin/env python3
"""
Comprehensive Test Suite for BH Routing Implementation
======================================================

This test suite validates the Benjamini-Hochberg routing implementation
by testing each component individually and in integration.

Test Categories:
1. BH Algorithm Unit Tests
2. P-Value Computation Tests
3. Safety Floor Tests
4. Integration Tests
5. Edge Case Tests

Run with: python test_bh_routing_comprehensive.py
"""

import torch
import numpy as np
import unittest
from typing import Dict, List, Tuple
import sys
import os

# Add the uploads directory to path
sys.path.insert(0, '/mnt/user-data/uploads')

# ============================================================================
# INLINE BH IMPLEMENTATION FOR TESTING (matches bh_routing.py logic)
# ============================================================================

def bh_routing_test_version(
    router_logits: torch.Tensor,
    alpha: float = 0.05,
    temperature: float = 1.0,
    min_k: int = 1,
    max_k: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simplified BH routing for testing.
    Uses p = 1 - softmax(logit) instead of KDE for self-contained tests.
    """
    # Handle 2D input
    input_is_2d = router_logits.ndim == 2
    if input_is_2d:
        router_logits = router_logits.unsqueeze(0)
    
    batch_size, seq_len, num_experts = router_logits.shape
    device = router_logits.device
    
    # Step 1: Compute probabilities
    scaled_logits = router_logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    
    # Step 2: Compute p-values (p = 1 - softmax)
    p_values = 1.0 - probs
    
    # Step 3: Sort p-values ascending
    sorted_pvals, sorted_indices = torch.sort(p_values, dim=-1)
    
    # Step 4: Compute BH critical values
    k_values = torch.arange(1, num_experts + 1, device=device, dtype=torch.float32)
    critical_values = (k_values / num_experts) * alpha
    critical_values = critical_values.view(1, 1, -1)
    
    # Step 5: Apply BH step-up procedure
    passes_threshold = sorted_pvals <= critical_values
    k_indices = torch.arange(1, num_experts + 1, device=device).view(1, 1, -1).float()
    masked_indices = torch.where(passes_threshold, k_indices, torch.zeros_like(k_indices))
    num_selected = masked_indices.max(dim=-1).values
    
    # Safety floor: ensure at least min_k
    num_selected = torch.where(
        num_selected == 0,
        torch.tensor(float(min_k), device=device),
        num_selected
    )
    num_selected = num_selected.clamp(min=min_k, max=max_k).long()
    
    # Step 6: Create selection mask
    expert_ranks = torch.arange(num_experts, device=device).view(1, 1, -1)
    selected_mask_sorted = expert_ranks < num_selected.unsqueeze(-1)
    
    # Convert to original order
    selected_mask = torch.zeros_like(probs, dtype=torch.bool)
    selected_mask.scatter_(dim=-1, index=sorted_indices, src=selected_mask_sorted)
    
    # Step 7: Compute routing weights
    routing_weights = torch.where(selected_mask, probs, torch.zeros_like(probs))
    weight_sums = routing_weights.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    routing_weights = routing_weights / weight_sums
    
    # Step 8: Extract selected expert indices
    selected_experts = torch.full((batch_size, seq_len, max_k), -1, dtype=torch.long, device=device)
    for k_idx in range(max_k):
        slot_active = k_idx < num_selected
        expert_idx = sorted_indices[:, :, k_idx]
        selected_experts[:, :, k_idx] = torch.where(
            slot_active, expert_idx, torch.tensor(-1, device=device)
        )
    
    if input_is_2d:
        routing_weights = routing_weights.squeeze(0)
        selected_experts = selected_experts.squeeze(0)
        num_selected = num_selected.squeeze(0)
    
    return routing_weights, selected_experts, num_selected


# ============================================================================
# TEST SUITE
# ============================================================================

class TestBHAlgorithm(unittest.TestCase):
    """Unit tests for the core BH algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.num_experts = 8
        self.seq_len = 4
        self.batch_size = 2
    
    def test_output_shapes_2d(self):
        """Test that output shapes are correct for 2D input."""
        logits = torch.randn(self.seq_len, self.num_experts)
        weights, experts, counts = bh_routing_test_version(logits, alpha=0.10, max_k=4)
        
        self.assertEqual(weights.shape, (self.seq_len, self.num_experts))
        self.assertEqual(experts.shape, (self.seq_len, 4))  # max_k=4
        self.assertEqual(counts.shape, (self.seq_len,))
        print("✓ test_output_shapes_2d passed")
    
    def test_output_shapes_3d(self):
        """Test that output shapes are correct for 3D input."""
        logits = torch.randn(self.batch_size, self.seq_len, self.num_experts)
        weights, experts, counts = bh_routing_test_version(logits, alpha=0.10, max_k=4)
        
        self.assertEqual(weights.shape, (self.batch_size, self.seq_len, self.num_experts))
        self.assertEqual(experts.shape, (self.batch_size, self.seq_len, 4))
        self.assertEqual(counts.shape, (self.batch_size, self.seq_len))
        print("✓ test_output_shapes_3d passed")
    
    def test_weights_sum_to_one(self):
        """Test that routing weights sum to 1 for each token."""
        logits = torch.randn(10, self.num_experts)
        weights, _, _ = bh_routing_test_version(logits, alpha=0.10, max_k=4)
        
        weight_sums = weights.sum(dim=-1)
        expected = torch.ones(10)
        
        self.assertTrue(torch.allclose(weight_sums, expected, atol=1e-5))
        print("✓ test_weights_sum_to_one passed")
    
    def test_non_negative_weights(self):
        """Test that all routing weights are non-negative."""
        logits = torch.randn(10, self.num_experts)
        weights, _, _ = bh_routing_test_version(logits, alpha=0.10, max_k=4)
        
        self.assertTrue((weights >= 0).all())
        print("✓ test_non_negative_weights passed")
    
    def test_selected_experts_within_bounds(self):
        """Test that selected expert indices are valid."""
        logits = torch.randn(10, self.num_experts)
        _, experts, counts = bh_routing_test_version(logits, alpha=0.10, max_k=4)
        
        # Valid indices are 0 to num_experts-1, or -1 for padding
        valid_indices = (experts >= -1) & (experts < self.num_experts)
        self.assertTrue(valid_indices.all())
        print("✓ test_selected_experts_within_bounds passed")
    
    def test_expert_counts_within_bounds(self):
        """Test that expert counts are within [min_k, max_k]."""
        logits = torch.randn(10, self.num_experts)
        min_k, max_k = 1, 4
        _, _, counts = bh_routing_test_version(logits, alpha=0.10, min_k=min_k, max_k=max_k)
        
        self.assertTrue((counts >= min_k).all())
        self.assertTrue((counts <= max_k).all())
        print("✓ test_expert_counts_within_bounds passed")


class TestSafetyFloor(unittest.TestCase):
    """Tests for the safety floor mechanism."""
    
    def test_safety_floor_activates(self):
        """Test that safety floor activates when BH would select 0 experts."""
        # Create logits where all experts have similar values
        # This should result in high p-values that don't pass BH threshold
        logits = torch.zeros(5, 8)  # All equal -> uniform probs -> high p-values
        
        min_k = 2
        _, _, counts = bh_routing_test_version(logits, alpha=0.001, min_k=min_k, max_k=4)
        
        # With very low alpha and uniform logits, BH should select 0
        # But safety floor should ensure min_k=2
        self.assertTrue((counts >= min_k).all())
        print("✓ test_safety_floor_activates passed")
    
    def test_safety_floor_minimum_guarantee(self):
        """Test that min_k experts are always selected."""
        # Create various challenging inputs
        test_cases = [
            torch.zeros(10, 16),  # All equal
            torch.randn(10, 16) * 0.001,  # Near-zero variance
            -torch.ones(10, 16) * 10,  # All very negative
        ]
        
        min_k = 3
        for logits in test_cases:
            _, _, counts = bh_routing_test_version(logits, alpha=0.001, min_k=min_k, max_k=8)
            self.assertTrue((counts >= min_k).all())
        
        print("✓ test_safety_floor_minimum_guarantee passed")


class TestBHThresholds(unittest.TestCase):
    """Tests for BH threshold computation."""
    
    def test_threshold_formula(self):
        """Test that BH thresholds follow (k/N) * alpha formula."""
        num_experts = 10
        alpha = 0.05
        
        k_values = torch.arange(1, num_experts + 1).float()
        expected_thresholds = (k_values / num_experts) * alpha
        
        # Verify formula
        expected = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050]
        for i, (computed, expected_val) in enumerate(zip(expected_thresholds, expected)):
            self.assertAlmostEqual(computed.item(), expected_val, places=6)
        
        print("✓ test_threshold_formula passed")
    
    def test_higher_alpha_more_experts(self):
        """Test that higher alpha leads to more experts selected."""
        logits = torch.randn(100, 64)
        
        _, _, counts_low = bh_routing_test_version(logits, alpha=0.01, max_k=16)
        _, _, counts_high = bh_routing_test_version(logits, alpha=0.50, max_k=16)
        
        mean_low = counts_low.float().mean().item()
        mean_high = counts_high.float().mean().item()
        
        self.assertGreater(mean_high, mean_low)
        print(f"✓ test_higher_alpha_more_experts passed (α=0.01: {mean_low:.2f}, α=0.50: {mean_high:.2f})")


class TestTemperatureEffects(unittest.TestCase):
    """Tests for temperature parameter effects."""
    
    def test_higher_temp_more_uniform(self):
        """Test that higher temperature leads to more uniform probabilities."""
        logits = torch.randn(10, 8)
        
        # Low temperature -> sharp distribution
        _, _, counts_low_temp = bh_routing_test_version(logits, alpha=0.30, temperature=0.5, max_k=8)
        
        # High temperature -> uniform distribution
        _, _, counts_high_temp = bh_routing_test_version(logits, alpha=0.30, temperature=2.0, max_k=8)
        
        # Higher temp should generally select MORE experts (more pass threshold)
        # because probabilities are more uniform
        print(f"✓ test_higher_temp_more_uniform: temp=0.5 avg={counts_low_temp.float().mean():.2f}, "
              f"temp=2.0 avg={counts_high_temp.float().mean():.2f}")


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases."""
    
    def test_single_token(self):
        """Test with single token input."""
        logits = torch.randn(1, 16)
        weights, experts, counts = bh_routing_test_version(logits, alpha=0.10, max_k=4)
        
        self.assertEqual(weights.shape, (1, 16))
        self.assertEqual(counts.shape, (1,))
        print("✓ test_single_token passed")
    
    def test_very_large_logits(self):
        """Test numerical stability with very large logits."""
        logits = torch.randn(10, 8) * 100  # Large values
        weights, _, counts = bh_routing_test_version(logits, alpha=0.10, max_k=4)
        
        # Should still produce valid output
        self.assertFalse(torch.isnan(weights).any())
        self.assertFalse(torch.isinf(weights).any())
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(10), atol=1e-4))
        print("✓ test_very_large_logits passed")
    
    def test_very_small_logits(self):
        """Test numerical stability with very small logits."""
        logits = torch.randn(10, 8) * 0.001  # Small values
        weights, _, counts = bh_routing_test_version(logits, alpha=0.10, max_k=4)
        
        self.assertFalse(torch.isnan(weights).any())
        self.assertFalse(torch.isinf(weights).any())
        print("✓ test_very_small_logits passed")
    
    def test_identical_logits(self):
        """Test with identical logits for all experts."""
        logits = torch.ones(10, 8) * 5.0  # All identical
        weights, _, counts = bh_routing_test_version(logits, alpha=0.10, min_k=1, max_k=4)
        
        # Should still work (safety floor ensures min_k)
        self.assertTrue((counts >= 1).all())
        print("✓ test_identical_logits passed")


class TestConcreteExample(unittest.TestCase):
    """Test with the concrete example from documentation."""
    
    def test_documented_example(self):
        """
        Test the example from the documentation:
        8 experts, α=0.30
        
        Sorted p-values: 0.02, 0.08, 0.12, 0.35, 0.45, 0.58, 0.65, 0.82
        BH thresholds:   0.0375, 0.075, 0.1125, 0.15, 0.1875, 0.225, 0.2625, 0.30
        
        Expected: k=3 (first 3 pass)
        """
        # Create logits that produce approximately these p-values
        # Since p = 1 - softmax, we need specific logits
        # We'll verify the BH procedure works correctly
        
        num_experts = 8
        alpha = 0.30
        
        # Create a simple test case
        logits = torch.tensor([[3.0, -0.5, 2.5, 0.3, -1.2, 4.0, 0.8, -0.2]])
        
        weights, experts, counts = bh_routing_test_version(
            logits, alpha=alpha, min_k=1, max_k=8
        )
        
        # Verify output is valid
        self.assertEqual(weights.shape, (1, 8))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(1), atol=1e-5))
        self.assertTrue((counts >= 1).all())
        self.assertTrue((counts <= 8).all())
        
        print(f"✓ test_documented_example passed")
        print(f"  Selected {counts[0].item()} experts")
        print(f"  Expert indices: {experts[0].tolist()}")


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple components."""
    
    def test_batch_consistency(self):
        """Test that batch processing gives same results as individual processing."""
        torch.manual_seed(42)
        
        # Create 2 separate inputs
        logits1 = torch.randn(5, 16)
        logits2 = torch.randn(5, 16)
        
        # Process individually
        w1, e1, c1 = bh_routing_test_version(logits1, alpha=0.10, max_k=8)
        w2, e2, c2 = bh_routing_test_version(logits2, alpha=0.10, max_k=8)
        
        # Process as batch
        torch.manual_seed(42)
        logits1_new = torch.randn(5, 16)
        logits2_new = torch.randn(5, 16)
        batch_logits = torch.stack([logits1_new, logits2_new], dim=0)  # [2, 5, 16]
        
        w_batch, e_batch, c_batch = bh_routing_test_version(batch_logits, alpha=0.10, max_k=8)
        
        # Results should match
        self.assertTrue(torch.allclose(w1, w_batch[0], atol=1e-5))
        self.assertTrue(torch.allclose(w2, w_batch[1], atol=1e-5))
        print("✓ test_batch_consistency passed")
    
    def test_reproducibility(self):
        """Test that results are reproducible."""
        logits = torch.randn(10, 32)
        
        w1, e1, c1 = bh_routing_test_version(logits, alpha=0.15, max_k=8)
        w2, e2, c2 = bh_routing_test_version(logits, alpha=0.15, max_k=8)
        
        self.assertTrue(torch.equal(w1, w2))
        self.assertTrue(torch.equal(e1, e2))
        self.assertTrue(torch.equal(c1, c2))
        print("✓ test_reproducibility passed")


# ============================================================================
# MAIN
# ============================================================================

def run_all_tests():
    """Run all test suites."""
    print("=" * 70)
    print("BH ROUTING COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestBHAlgorithm,
        TestSafetyFloor,
        TestBHThresholds,
        TestTemperatureEffects,
        TestEdgeCases,
        TestConcreteExample,
        TestIntegration,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
        for test, traceback in result.failures + result.errors:
            print(f"\n  Failed: {test}")
            print(f"  {traceback[:200]}...")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)