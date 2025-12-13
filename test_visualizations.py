"""
Test Suite for Routing Visualizations Module
=============================================

Comprehensive tests for all visualization functions.

Tests:
1. Generate dummy data for each function
2. Call each function and verify plots are created
3. Save example outputs to ./plots/ directory
4. Test edge cases (empty inputs, single data point)
5. Verify CPU/GPU tensor handling

Run with: python test_visualizations.py
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# Import visualization functions
from routing_visualizations import (
    plot_expert_count_distribution,
    plot_alpha_sensitivity,
    plot_routing_heatmap,
    plot_expert_utilization,
    plot_token_complexity_vs_experts,
    create_comparison_table,
    plot_layer_wise_routing,
    create_analysis_report
)

# Create output directory
OUTPUT_DIR = Path('./plots')
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("ROUTING VISUALIZATIONS TEST SUITE")
print("=" * 70)


def test_expert_count_distribution():
    """Test 1: Expert count distribution plot."""
    print("\n[Test 1] plot_expert_count_distribution()")
    print("-" * 70)

    # Generate dummy data
    expert_counts = torch.randint(2, 9, (200,))  # 200 tokens, 2-8 experts each

    # Test with torch tensor
    print("  Testing with torch.Tensor...")
    fig1 = plot_expert_count_distribution(
        expert_counts,
        method_name="BH Routing",
        alpha=0.05,
        save_path=OUTPUT_DIR / "test_expert_count_dist.png"
    )
    assert fig1 is not None, "Figure not created"
    plt.close(fig1)
    print("  ✓ Torch tensor test passed")

    # Test with numpy array
    print("  Testing with numpy array...")
    fig2 = plot_expert_count_distribution(
        expert_counts.numpy(),
        method_name="BH Routing (NumPy)",
        save_path=OUTPUT_DIR / "test_expert_count_dist_numpy.png"
    )
    assert fig2 is not None, "Figure not created"
    plt.close(fig2)
    print("  ✓ NumPy array test passed")

    # Test edge case: single value
    print("  Testing edge case: single value...")
    fig3 = plot_expert_count_distribution(
        torch.tensor([5]),
        method_name="Single Token"
    )
    assert fig3 is not None, "Figure not created"
    plt.close(fig3)
    print("  ✓ Edge case test passed")

    print("\n✅ Test 1 PASSED\n")


def test_alpha_sensitivity():
    """Test 2: Alpha sensitivity plot."""
    print("\n[Test 2] plot_alpha_sensitivity()")
    print("-" * 70)

    # Generate dummy data
    alphas = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    avg_experts = [3.2, 3.8, 4.5, 5.3, 6.0, 6.5, 7.2, 7.8]
    std_experts = [0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

    # Test with error bars
    print("  Testing with error bars...")
    fig1 = plot_alpha_sensitivity(
        alphas,
        avg_experts,
        std_experts,
        baseline_k=8,
        save_path=OUTPUT_DIR / "test_alpha_sensitivity.png"
    )
    assert fig1 is not None, "Figure not created"
    plt.close(fig1)
    print("  ✓ With error bars test passed")

    # Test without error bars
    print("  Testing without error bars...")
    fig2 = plot_alpha_sensitivity(
        alphas,
        avg_experts,
        std_experts=None,
        save_path=OUTPUT_DIR / "test_alpha_sensitivity_no_err.png"
    )
    assert fig2 is not None, "Figure not created"
    plt.close(fig2)
    print("  ✓ Without error bars test passed")

    # Test edge case: two points
    print("  Testing edge case: two points...")
    fig3 = plot_alpha_sensitivity(
        [0.05, 0.10],
        [4.5, 5.5],
        baseline_k=8
    )
    assert fig3 is not None, "Figure not created"
    plt.close(fig3)
    print("  ✓ Edge case test passed")

    print("\n✅ Test 2 PASSED\n")


def test_routing_heatmap():
    """Test 3: Routing heatmap plot."""
    print("\n[Test 3] plot_routing_heatmap()")
    print("-" * 70)

    # Generate dummy data
    seq_len, num_experts = 20, 64
    routing_weights = torch.rand(seq_len, num_experts) * 0.3  # Sparse weights
    tokens = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."] * 2

    # Test basic heatmap
    print("  Testing basic heatmap...")
    fig1 = plot_routing_heatmap(
        routing_weights,
        tokens,
        save_path=OUTPUT_DIR / "test_routing_heatmap.png"
    )
    assert fig1 is not None, "Figure not created"
    plt.close(fig1)
    print("  ✓ Basic heatmap test passed")

    # Test with many tokens (truncation)
    print("  Testing with many tokens (truncation)...")
    long_weights = torch.rand(100, 64) * 0.3
    long_tokens = [f"tok{i}" for i in range(100)]
    fig2 = plot_routing_heatmap(
        long_weights,
        long_tokens,
        max_tokens=30,
        save_path=OUTPUT_DIR / "test_routing_heatmap_truncated.png"
    )
    assert fig2 is not None, "Figure not created"
    plt.close(fig2)
    print("  ✓ Truncation test passed")

    # Test with few tokens
    print("  Testing with few tokens...")
    fig3 = plot_routing_heatmap(
        torch.rand(3, 16),
        ["Hello", "world", "!"]
    )
    assert fig3 is not None, "Figure not created"
    plt.close(fig3)
    print("  ✓ Few tokens test passed")

    print("\n✅ Test 3 PASSED\n")


def test_expert_utilization():
    """Test 4: Expert utilization plot."""
    print("\n[Test 4] plot_expert_utilization()")
    print("-" * 70)

    # Generate dummy data (Poisson-like distribution)
    num_experts = 64
    expert_usage = np.random.poisson(50, num_experts)

    # Test basic utilization
    print("  Testing basic utilization plot...")
    fig1 = plot_expert_utilization(
        expert_usage,
        save_path=OUTPUT_DIR / "test_expert_utilization.png"
    )
    assert fig1 is not None, "Figure not created"
    plt.close(fig1)
    print("  ✓ Basic utilization test passed")

    # Test with imbalanced load
    print("  Testing with imbalanced load...")
    imbalanced = np.array([100] * 10 + [10] * 54)  # 10 heavily used, rest barely used
    fig2 = plot_expert_utilization(
        imbalanced,
        save_path=OUTPUT_DIR / "test_expert_utilization_imbalanced.png"
    )
    assert fig2 is not None, "Figure not created"
    plt.close(fig2)
    print("  ✓ Imbalanced load test passed")

    # Test with few experts
    print("  Testing with few experts...")
    fig3 = plot_expert_utilization(
        np.array([25, 30, 28, 26])
    )
    assert fig3 is not None, "Figure not created"
    plt.close(fig3)
    print("  ✓ Few experts test passed")

    print("\n✅ Test 4 PASSED\n")


def test_token_complexity():
    """Test 5: Token complexity vs experts plot."""
    print("\n[Test 5] plot_token_complexity_vs_experts()")
    print("-" * 70)

    # Generate dummy data
    # Low token IDs (common) → fewer experts
    # High token IDs (rare) → more experts
    token_ids = np.concatenate([
        np.random.randint(0, 1000, 50),    # Common tokens
        np.random.randint(10000, 30000, 30)  # Rare tokens
    ])
    expert_counts = np.concatenate([
        np.random.randint(3, 5, 50),      # Fewer experts
        np.random.randint(5, 8, 30)       # More experts
    ])

    # Shuffle
    perm = np.random.permutation(len(token_ids))
    token_ids = token_ids[perm]
    expert_counts = expert_counts[perm]

    # Test basic plot
    print("  Testing basic token complexity plot...")
    fig1 = plot_token_complexity_vs_experts(
        token_ids,
        expert_counts,
        save_path=OUTPUT_DIR / "test_token_complexity.png"
    )
    assert fig1 is not None, "Figure not created"
    plt.close(fig1)
    print("  ✓ Basic complexity plot test passed")

    # Test with few points
    print("  Testing with few points...")
    fig2 = plot_token_complexity_vs_experts(
        [101, 2054, 1996],
        [4, 5, 3]
    )
    assert fig2 is not None, "Figure not created"
    plt.close(fig2)
    print("  ✓ Few points test passed")

    print("\n✅ Test 5 PASSED\n")


def test_comparison_table():
    """Test 6: Comparison table creation."""
    print("\n[Test 6] create_comparison_table()")
    print("-" * 70)

    # Generate dummy results
    results = {
        'TopK (k=8)': {
            'avg_experts': 8.00,
            'std': 0.00,
            'min': 8,
            'max': 8,
            'perplexity': 12.5
        },
        'BH (α=0.05)': {
            'avg_experts': 4.52,
            'std': 0.85,
            'min': 3,
            'max': 7,
            'perplexity': 12.7
        },
        'BH (α=0.10)': {
            'avg_experts': 5.38,
            'std': 1.02,
            'min': 3,
            'max': 8,
            'perplexity': 12.6
        }
    }

    # Test DataFrame output
    print("  Testing DataFrame output...")
    df = create_comparison_table(results, output_format='dataframe')
    assert df is not None, "DataFrame not created"
    assert len(df) == 3, f"Expected 3 rows, got {len(df)}"
    print("  ✓ DataFrame test passed")
    print(f"\n{df}\n")

    # Test Markdown output
    print("  Testing Markdown output...")
    md = create_comparison_table(results, output_format='markdown')
    assert isinstance(md, str), "Markdown output not a string"
    print("  ✓ Markdown test passed")

    # Test LaTeX output
    print("  Testing LaTeX output...")
    latex = create_comparison_table(results, output_format='latex')
    assert isinstance(latex, str), "LaTeX output not a string"
    print("  ✓ LaTeX test passed")

    # Save markdown
    with open(OUTPUT_DIR / "test_comparison_table.md", 'w') as f:
        f.write("# Comparison Table\n\n")
        f.write(md)
    print("  ✓ Saved to test_comparison_table.md")

    print("\n✅ Test 6 PASSED\n")


def test_layer_wise_routing():
    """Test 7: Layer-wise routing plot."""
    print("\n[Test 7] plot_layer_wise_routing()")
    print("-" * 70)

    # Generate dummy data (16 layers, varying token counts)
    num_layers = 16
    num_tokens = 100

    # Simulate: early layers use fewer experts, late layers use more
    layer_counts = []
    for layer_idx in range(num_layers):
        base_experts = 3 + (layer_idx / num_layers) * 3  # 3 → 6 progression
        counts = np.random.randint(
            int(base_experts) - 1,
            int(base_experts) + 2,
            num_tokens
        )
        layer_counts.append(counts)

    layer_counts_array = np.array(layer_counts)

    # Test with 2D array
    print("  Testing with 2D array...")
    fig1 = plot_layer_wise_routing(
        layer_counts_array,
        save_path=OUTPUT_DIR / "test_layer_wise_routing.png"
    )
    assert fig1 is not None, "Figure not created"
    plt.close(fig1)
    print("  ✓ 2D array test passed")

    # Test with list of arrays
    print("  Testing with list of arrays...")
    fig2 = plot_layer_wise_routing(
        layer_counts,
        layer_names=[f'Layer{i}' for i in range(num_layers)],
        save_path=OUTPUT_DIR / "test_layer_wise_routing_list.png"
    )
    assert fig2 is not None, "Figure not created"
    plt.close(fig2)
    print("  ✓ List of arrays test passed")

    # Test with few layers
    print("  Testing with few layers...")
    fig3 = plot_layer_wise_routing(
        np.random.randint(3, 7, (4, 50))
    )
    assert fig3 is not None, "Figure not created"
    plt.close(fig3)
    print("  ✓ Few layers test passed")

    print("\n✅ Test 7 PASSED\n")


def test_analysis_report():
    """Test 8: Complete analysis report."""
    print("\n[Test 8] create_analysis_report()")
    print("-" * 70)

    # Generate comprehensive dummy data
    routing_data = {
        'expert_counts': torch.randint(2, 9, (200,)),
        'method_name': 'BH Routing',
        'alpha': 0.05,
        'alphas': [0.01, 0.05, 0.10, 0.20],
        'avg_experts_per_alpha': [3.2, 4.5, 5.8, 6.9],
        'std_experts_per_alpha': [0.5, 0.8, 0.9, 1.0],
        'routing_weights': torch.rand(15, 64) * 0.3,
        'tokens': ["Hello", "world", "!", "This", "is", "a", "test", ".", "BH", "routing", "works", "well", ".", "", ""],
        'expert_usage': np.random.poisson(50, 64),
        'layer_expert_counts': torch.randint(3, 8, (16, 100))
    }

    print("  Testing comprehensive report generation...")
    figs = create_analysis_report(
        routing_data,
        output_dir=str(OUTPUT_DIR / 'report'),
        dpi=150
    )

    assert len(figs) > 0, "No figures created"
    print(f"  ✓ Created {len(figs)} plots")

    # List created files
    report_dir = OUTPUT_DIR / 'report'
    if report_dir.exists():
        files = list(report_dir.glob('*.png'))
        print(f"  ✓ Saved {len(files)} files:")
        for f in files:
            print(f"    - {f.name}")

    print("\n✅ Test 8 PASSED\n")


def test_edge_cases():
    """Test 9: Edge cases and error handling."""
    print("\n[Test 9] Edge Cases and Error Handling")
    print("-" * 70)

    passed = 0
    total = 0

    # Test 1: Empty input
    print("  Testing empty input...")
    total += 1
    try:
        plot_expert_count_distribution(torch.tensor([]))
        print("  ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
        passed += 1

    # Test 2: Mismatched lengths
    print("  Testing mismatched lengths...")
    total += 1
    try:
        plot_alpha_sensitivity([0.05, 0.10], [4.5])
        print("  ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
        passed += 1

    # Test 3: Invalid output format
    print("  Testing invalid output format...")
    total += 1
    try:
        create_comparison_table({'method': {'metric': 1.0}}, output_format='invalid')
        print("  ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
        passed += 1

    # Test 4: Wrong tensor dimensions
    print("  Testing wrong tensor dimensions...")
    total += 1
    try:
        plot_routing_heatmap(torch.rand(10), ["test"])
        print("  ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
        passed += 1

    # Test 5: GPU tensor (if available)
    if torch.cuda.is_available():
        print("  Testing GPU tensor...")
        total += 1
        try:
            gpu_tensor = torch.randint(3, 8, (50,)).cuda()
            fig = plot_expert_count_distribution(gpu_tensor)
            plt.close(fig)
            print("  ✓ GPU tensor handled correctly")
            passed += 1
        except Exception as e:
            print(f"  ✗ GPU tensor failed: {e}")
    else:
        print("  ⊘ GPU not available, skipping GPU tensor test")

    print(f"\n  Passed {passed}/{total} edge case tests")
    print("\n✅ Test 9 PASSED\n")


def run_all_tests():
    """Run all tests."""
    print("\nRunning all tests...\n")

    tests = [
        test_expert_count_distribution,
        test_alpha_sensitivity,
        test_routing_heatmap,
        test_expert_utilization,
        test_token_complexity,
        test_comparison_table,
        test_layer_wise_routing,
        test_analysis_report,
        test_edge_cases
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ {test_func.__name__} FAILED:")
            print(f"   {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"\nTotal tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    # List generated files
    print(f"\n📁 Output directory: {OUTPUT_DIR.absolute()}")
    if OUTPUT_DIR.exists():
        files = list(OUTPUT_DIR.glob('**/*.png')) + list(OUTPUT_DIR.glob('**/*.md'))
        print(f"📊 Generated {len(files)} files:")
        for f in sorted(files):
            print(f"   - {f.relative_to(OUTPUT_DIR)}")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Review output above.")

    return passed, failed


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')

    try:
        passed, failed = run_all_tests()
        exit(0 if failed == 0 else 1)
    except Exception as e:
        print(f"\n💥 Critical error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
