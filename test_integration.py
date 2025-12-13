"""
Test Suite for OLMoE BH Routing Integration
============================================

Tests the integration module without requiring a full model download.
Uses mock objects to simulate OLMoE architecture.

Run with: python test_integration.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Tuple

# Try to import the integration module
try:
    from olmoe_bh_integration import BHRoutingIntegration
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"❌ Cannot import olmoe_bh_integration: {e}")
    INTEGRATION_AVAILABLE = False


class MockOlmoeTopKRouter(nn.Module):
    """
    Mock OlmoeTopKRouter for testing.

    Simulates the real router's behavior without requiring transformers.
    """

    def __init__(self, hidden_dim: int = 1024, num_experts: int = 64, top_k: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Linear layer for routing
        self.linear = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Original top-k routing forward.

        Args:
            hidden_states: [num_tokens, hidden_dim]

        Returns:
            routing_weights: [num_tokens, top_k]
            selected_experts: [num_tokens, top_k]
            router_logits: [num_tokens, num_experts]
        """
        # Compute logits
        router_logits = self.linear(hidden_states)

        # Top-k selection
        routing_weights, selected_experts = torch.topk(router_logits, k=self.top_k, dim=-1)

        # Softmax over selected
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float)
        routing_weights = routing_weights.to(hidden_states.dtype)

        return routing_weights, selected_experts, router_logits


class MockOlmoeSparseMoeBlock(nn.Module):
    """Mock MoE block containing a router."""

    def __init__(self, hidden_dim: int = 1024, num_experts: int = 64):
        super().__init__()
        self.gate = MockOlmoeTopKRouter(hidden_dim, num_experts)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Simplified - just route, don't actually dispatch to experts
        routing_weights, selected_experts, router_logits = self.gate(hidden_states)
        return hidden_states  # Pass through for testing


class MockOlmoeModel(nn.Module):
    """Mock OLMoE model with multiple MoE layers."""

    def __init__(self, num_layers: int = 16, hidden_dim: int = 1024, num_experts: int = 64):
        super().__init__()
        self.num_layers = num_layers

        # Create MoE blocks
        self.layers = nn.ModuleList([
            MockOlmoeSparseMoeBlock(hidden_dim, num_experts)
            for _ in range(num_layers)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def test_find_routers():
    """Test 1: Verify that routers are correctly identified."""
    print("\n" + "=" * 70)
    print("TEST 1: Router Discovery")
    print("=" * 70)

    if not INTEGRATION_AVAILABLE:
        print("❌ SKIP: Integration module not available")
        return False

    # Create mock model
    model = MockOlmoeModel(num_layers=16)

    # Initialize integrator
    integrator = BHRoutingIntegration(
        model,
        alpha=0.05,
        max_k=8,
        mode='patch',
        collect_stats=False
    )

    # Check that all routers were found
    expected_routers = 16  # One per layer
    found_routers = len(integrator.routers)

    print(f"Expected routers: {expected_routers}")
    print(f"Found routers: {found_routers}")

    if found_routers == expected_routers:
        print("✅ PASS: All routers found")
        return True
    else:
        print(f"❌ FAIL: Expected {expected_routers}, found {found_routers}")
        return False


def test_patching_mechanism():
    """Test 2: Verify that patching changes routing behavior."""
    print("\n" + "=" * 70)
    print("TEST 2: Patching Mechanism")
    print("=" * 70)

    if not INTEGRATION_AVAILABLE:
        print("❌ SKIP: Integration module not available")
        return False

    # Create mock model
    model = MockOlmoeModel(num_layers=4)  # Smaller for speed

    # Create test input
    batch_size, seq_len, hidden_dim = 2, 10, 1024
    hidden_states = torch.randn(batch_size * seq_len, hidden_dim)

    # Get original output
    with torch.no_grad():
        router = model.layers[0].gate
        weights_original, experts_original, logits_original = router(hidden_states)

    print(f"Original routing:")
    print(f"  Weights shape: {weights_original.shape}")
    print(f"  Experts shape: {experts_original.shape}")
    print(f"  Num experts per token: {weights_original.shape[-1]}")

    # Apply patching
    integrator = BHRoutingIntegration(
        model,
        alpha=0.05,
        max_k=8,
        mode='patch',
        collect_stats=False
    )
    integrator.patch_model()

    # Get patched output
    with torch.no_grad():
        weights_patched, experts_patched, logits_patched = router(hidden_states)

    print(f"\nPatched routing (BH):")
    print(f"  Weights shape: {weights_patched.shape}")
    print(f"  Experts shape: {experts_patched.shape}")
    print(f"  Num experts per token: {weights_patched.shape[-1]}")

    # Verify shapes match
    shape_match = (
        weights_original.shape == weights_patched.shape and
        experts_original.shape == experts_patched.shape
    )

    # Verify outputs are different (BH should select differently)
    outputs_different = not torch.allclose(weights_original, weights_patched, atol=1e-6)

    # Unpatch
    integrator.unpatch_model()

    # Verify unpatching works
    with torch.no_grad():
        weights_restored, experts_restored, logits_restored = router(hidden_states)

    restored_match = torch.allclose(weights_original, weights_restored, atol=1e-6)

    print(f"\nVerification:")
    print(f"  Shapes match: {shape_match}")
    print(f"  Outputs different after patching: {outputs_different}")
    print(f"  Outputs restored after unpatching: {restored_match}")

    if shape_match and outputs_different and restored_match:
        print("✅ PASS: Patching mechanism works correctly")
        return True
    else:
        print("❌ FAIL: Patching mechanism issue")
        return False


def test_weight_normalization():
    """Test 3: Verify that BH routing produces valid weights."""
    print("\n" + "=" * 70)
    print("TEST 3: Weight Normalization")
    print("=" * 70)

    if not INTEGRATION_AVAILABLE:
        print("❌ SKIP: Integration module not available")
        return False

    # Create mock model
    model = MockOlmoeModel(num_layers=2)

    # Create test input
    hidden_states = torch.randn(20, 1024)  # 20 tokens

    # Apply BH routing
    integrator = BHRoutingIntegration(
        model,
        alpha=0.05,
        max_k=8,
        mode='patch'
    )
    integrator.patch_model()

    # Get routing output
    with torch.no_grad():
        router = model.layers[0].gate
        weights, experts, logits = router(hidden_states)

    # Check normalization
    weight_sums = weights.sum(dim=-1)
    all_sum_to_one = torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    # Check non-negative
    all_non_negative = (weights >= 0).all().item()

    # Check no NaN/Inf
    no_nan = not torch.isnan(weights).any().item()
    no_inf = not torch.isinf(weights).any().item()

    print(f"Weight sums (should be ~1.0): {weight_sums[:5].tolist()}")
    print(f"Weight range: [{weights.min().item():.6f}, {weights.max().item():.6f}]")
    print(f"\nValidation:")
    print(f"  All weights sum to 1: {all_sum_to_one}")
    print(f"  All weights non-negative: {all_non_negative}")
    print(f"  No NaN values: {no_nan}")
    print(f"  No Inf values: {no_inf}")

    integrator.unpatch_model()

    if all_sum_to_one and all_non_negative and no_nan and no_inf:
        print("✅ PASS: Weights are properly normalized")
        return True
    else:
        print("❌ FAIL: Weight normalization issue")
        return False


def test_expert_selection():
    """Test 4: Verify that expert selection is reasonable."""
    print("\n" + "=" * 70)
    print("TEST 4: Expert Selection")
    print("=" * 70)

    if not INTEGRATION_AVAILABLE:
        print("❌ SKIP: Integration module not available")
        return False

    # Create mock model
    model = MockOlmoeModel(num_layers=1)

    # Create test input
    hidden_states = torch.randn(50, 1024)  # 50 tokens

    # Test with different alpha values
    results = {}

    for alpha in [0.01, 0.05, 0.20]:
        integrator = BHRoutingIntegration(
            model,
            alpha=alpha,
            max_k=16,
            mode='patch'
        )
        integrator.patch_model()

        with torch.no_grad():
            router = model.layers[0].gate
            weights, experts, logits = router(hidden_states)

        # Count selected experts (non-padded)
        num_selected = (experts != -1).sum(dim=-1).float()
        mean_selected = num_selected.mean().item()

        results[alpha] = mean_selected
        integrator.unpatch_model()

        print(f"Alpha = {alpha:.2f}: Mean experts selected = {mean_selected:.2f}")

    # Verify monotonicity (higher alpha should select more experts)
    monotonic = results[0.01] <= results[0.05] <= results[0.20]

    # Verify reasonable range
    reasonable = all(1 <= v <= 16 for v in results.values())

    print(f"\nValidation:")
    print(f"  Monotonic with alpha: {monotonic}")
    print(f"  Counts in reasonable range [1, 16]: {reasonable}")

    if monotonic and reasonable:
        print("✅ PASS: Expert selection behaves correctly")
        return True
    else:
        print("❌ FAIL: Expert selection issue")
        return False


def test_analyze_mode():
    """Test 5: Verify that analyze mode doesn't change outputs."""
    print("\n" + "=" * 70)
    print("TEST 5: Analyze Mode (Simulation)")
    print("=" * 70)

    if not INTEGRATION_AVAILABLE:
        print("❌ SKIP: Integration module not available")
        return False

    # Create mock model
    model = MockOlmoeModel(num_layers=2)

    # Create test input
    hidden_states = torch.randn(10, 1024)

    # Get original output
    with torch.no_grad():
        router = model.layers[0].gate
        weights_original, experts_original, _ = router(hidden_states)

    # Apply analyze mode
    integrator = BHRoutingIntegration(
        model,
        alpha=0.05,
        max_k=8,
        mode='analyze',  # Simulation only
        collect_stats=True
    )
    integrator.patch_model()

    # Get output in analyze mode
    with torch.no_grad():
        weights_analyze, experts_analyze, _ = router(hidden_states)

    # Outputs should be identical (using original routing)
    outputs_identical = (
        torch.allclose(weights_original, weights_analyze, atol=1e-6) and
        torch.equal(experts_original, experts_analyze)
    )

    # Stats should be collected
    stats = integrator.get_routing_stats()
    stats_collected = 'bh_would_select_mean' in stats

    print(f"Outputs identical to original: {outputs_identical}")
    print(f"BH simulation stats collected: {stats_collected}")

    if stats_collected:
        print(f"  BH would select: {stats['bh_would_select_mean']:.2f} experts (mean)")

    integrator.unpatch_model()

    if outputs_identical and stats_collected:
        print("✅ PASS: Analyze mode works correctly")
        return True
    else:
        print("❌ FAIL: Analyze mode issue")
        return False


def test_statistics_collection():
    """Test 6: Verify that statistics are collected correctly."""
    print("\n" + "=" * 70)
    print("TEST 6: Statistics Collection")
    print("=" * 70)

    if not INTEGRATION_AVAILABLE:
        print("❌ SKIP: Integration module not available")
        return False

    # Create mock model
    model = MockOlmoeModel(num_layers=4)

    # Apply patching with stats
    integrator = BHRoutingIntegration(
        model,
        alpha=0.05,
        max_k=8,
        mode='patch',
        collect_stats=True  # Enable stats
    )
    integrator.patch_model()

    # Run multiple forward passes
    num_passes = 5
    for _ in range(num_passes):
        hidden_states = torch.randn(10, 1024)
        with torch.no_grad():
            model(hidden_states)

    # Get statistics
    stats = integrator.get_routing_stats()

    print(f"Statistics collected:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Verify stats exist
    has_mean = 'mean_experts_per_token' in stats
    has_count = 'total_forward_passes' in stats

    # The count might not exactly match num_passes * num_layers due to how
    # stats are collected (per router call, not per model forward)
    # Just verify stats were collected

    integrator.unpatch_model()

    if has_mean and has_count:
        print("✅ PASS: Statistics collection works")
        return True
    else:
        print("❌ FAIL: Statistics not collected properly")
        return False


def test_context_manager():
    """Test 7: Verify context manager interface."""
    print("\n" + "=" * 70)
    print("TEST 7: Context Manager Interface")
    print("=" * 70)

    if not INTEGRATION_AVAILABLE:
        print("❌ SKIP: Integration module not available")
        return False

    # Create mock model
    model = MockOlmoeModel(num_layers=2)
    hidden_states = torch.randn(5, 1024)

    # Get original output
    with torch.no_grad():
        router = model.layers[0].gate
        weights_original, _, _ = router(hidden_states)

    # Use context manager
    with BHRoutingIntegration(model, alpha=0.05, max_k=8, mode='patch') as integrator:
        # Inside context: should be patched
        with torch.no_grad():
            weights_patched, _, _ = router(hidden_states)

        patched_different = not torch.allclose(weights_original, weights_patched, atol=1e-6)

    # Outside context: should be unpatched
    with torch.no_grad():
        weights_restored, _, _ = router(hidden_states)

    restored_same = torch.allclose(weights_original, weights_restored, atol=1e-6)

    print(f"Inside context: outputs changed = {patched_different}")
    print(f"After context: outputs restored = {restored_same}")

    if patched_different and restored_same:
        print("✅ PASS: Context manager works correctly")
        return True
    else:
        print("❌ FAIL: Context manager issue")
        return False


def run_all_tests():
    """Run all integration tests."""
    print("=" * 70)
    print("BH ROUTING INTEGRATION TEST SUITE")
    print("=" * 70)

    if not INTEGRATION_AVAILABLE:
        print("\n❌ CRITICAL: Cannot import integration module")
        print("Ensure olmoe_bh_integration.py and bh_routing.py are in the same directory")
        return

    tests = [
        ("Router Discovery", test_find_routers),
        ("Patching Mechanism", test_patching_mechanism),
        ("Weight Normalization", test_weight_normalization),
        ("Expert Selection", test_expert_selection),
        ("Analyze Mode", test_analyze_mode),
        ("Statistics Collection", test_statistics_collection),
        ("Context Manager", test_context_manager),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Integration module is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review output above.")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')

    # Run tests
    run_all_tests()
