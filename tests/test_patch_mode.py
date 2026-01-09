#!/usr/bin/env python3
"""
Standalone test for patch_mode implementation.
Tests the helper functions without requiring the full notebook context.
"""

import re
from typing import List, Tuple, Optional

# Copy of helper functions from Cell 18
def _extract_layer_idx(module_name: str) -> int:
    """Extract transformer layer index from MoE block module name."""
    match = re.search(r'(?:^|\.)layers\.(\d+)(?:\.|$)', module_name)
    if match:
        return int(match.group(1))
    raise ValueError(
        f"Cannot extract layer index from module name: '{module_name}'. "
        f"Expected pattern containing 'layers.<idx>' e.g., 'model.layers.15.mlp'"
    )

def _select_last_n_moe_layers(
    unique_sorted_indices: List[int],
    patch_mode: Optional[int]
) -> Tuple[List[int], bool]:
    """Select which layer indices to patch based on patch_mode."""
    if patch_mode is None or patch_mode <= 0:
        return (unique_sorted_indices[:], False)

    if patch_mode > len(unique_sorted_indices):
        return (unique_sorted_indices[:], True)  # clamped

    return (unique_sorted_indices[-patch_mode:], False)

def test_extraction():
    """Test layer index extraction."""
    print("Testing _extract_layer_idx()...")

    test_cases = [
        ('model.layers.0.mlp', 0),
        ('model.layers.15.mlp.experts', 15),
        ('transformer.layers.7.moe_block', 7),
        ('layers.3.block', 3),
        ('model.layers.10.mlp', 10),
    ]

    for path, expected in test_cases:
        result = _extract_layer_idx(path)
        assert result == expected, f"Failed: {path} -> {result} (expected {expected})"
        print(f"  ✓ {path} -> {result}")

    # Test error case
    try:
        _extract_layer_idx('invalid.path.no.layers')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ ValueError raised correctly: {str(e)[:50]}...")

    print("  ✅ All extraction tests passed!\n")

def test_selection():
    """Test layer selection logic."""
    print("Testing _select_last_n_moe_layers()...")

    # Full 16-layer model
    all_16 = list(range(16))

    tests = [
        (all_16, None, (all_16, False), "None -> all"),
        (all_16, 0, (all_16, False), "0 -> all"),
        (all_16, -5, (all_16, False), "negative -> all"),
        (all_16, 1, ([15], False), "1 -> [15]"),
        (all_16, 4, ([12, 13, 14, 15], False), "4 -> last 4"),
        (all_16, 16, (all_16, False), "16 -> all (exact)"),
        (all_16, 100, (all_16, True), "100 -> clamp"),
    ]

    for indices, patch_mode, expected, desc in tests:
        result = _select_last_n_moe_layers(indices, patch_mode)
        assert result == expected, f"Failed: {desc} -> {result} (expected {expected})"
        print(f"  ✓ {desc}")

    # Sparse layers
    sparse = [0, 2, 7, 15]
    sparse_tests = [
        (sparse, 1, ([15], False), "sparse: 1 -> [15]"),
        (sparse, 2, ([7, 15], False), "sparse: 2 -> [7, 15]"),
        (sparse, 3, ([2, 7, 15], False), "sparse: 3 -> [2, 7, 15]"),
        (sparse, 4, (sparse, False), "sparse: 4 -> all"),
        (sparse, 10, (sparse, True), "sparse: 10 -> clamp"),
    ]

    for indices, patch_mode, expected, desc in sparse_tests:
        result = _select_last_n_moe_layers(indices, patch_mode)
        assert result == expected, f"Failed: {desc} -> {result} (expected {expected})"
        print(f"  ✓ {desc}")

    print("  ✅ All selection tests passed!\n")

if __name__ == "__main__":
    print("=" * 70)
    print("PATCH_MODE STANDALONE TESTS")
    print("=" * 70)
    print()

    test_extraction()
    test_selection()

    print("=" * 70)
    print("🎉 ALL STANDALONE TESTS PASSED!")
    print("=" * 70)
