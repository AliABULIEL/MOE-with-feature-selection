"""
BH Routing Framework - Quick Start Guide
=========================================

This script demonstrates how to use the comprehensive BH routing evaluation
framework. Run this to execute a full two-phase experiment with all baselines
and BH configurations.

IMPORTANT: This script requires GPU and significant time to complete.
For a quick test, modify the parameters at the bottom.

Usage:
    # Full experiment (60 experiments, ~2-3 hours on A100)
    python BH_Routing_Quick_Start.py

    # Quick test (2 configs, ~5 minutes)
    python BH_Routing_Quick_Start.py --quick-test

Expected Output:
    - bh_routing_experiment/logs/*.json (120 files)
    - bh_routing_experiment/bh_routing_results.csv
    - bh_routing_experiment/bh_routing_results.json
    - bh_routing_experiment/visualizations/bh_comprehensive_comparison.png
"""

import argparse
from pathlib import Path

# Import the framework
from bh_routing_experiment_runner import BHRoutingExperimentRunner
from bh_routing_visualization import create_comprehensive_visualization


def run_quick_test():
    """
    Run a quick test with minimal configurations to verify setup.

    Tests:
    - 1 baseline (TopK-8)
    - 2 BH configs (alpha=0.40, 0.50 with max_k=8)
    - 1 dataset (WikiText, 50 samples)
    - Total: 3 experiments, ~5 minutes
    """
    print("=" * 70)
    print("QUICK TEST MODE")
    print("=" * 70)

    runner = BHRoutingExperimentRunner(
        model_name="allenai/OLMoE-1B-7B-0924",
        device="cuda",
        output_dir="./bh_routing_quick_test"
    )

    results_df = runner.run_two_phase_experiment(
        baseline_k_values=[8],  # Just one baseline
        bh_max_k_values=[8],  # Just one max_k
        bh_alpha_values=[0.40, 0.50],  # Two alpha values
        datasets=['wikitext'],  # Just WikiText
        max_samples=50  # Minimal samples
    )

    print("\n" + "=" * 70)
    print("QUICK TEST RESULTS")
    print("=" * 70)
    print(results_df[['config_name', 'dataset', 'perplexity', 'avg_experts', 'flops_reduction_pct']])

    # Generate visualization
    print("\nGenerating visualization...")
    fig = create_comprehensive_visualization(
        results_df,
        output_path='./bh_routing_quick_test/visualizations/test_comparison.png'
    )

    print("\n✅ Quick test complete!")
    print(f"   Results saved to: ./bh_routing_quick_test/")


def run_full_experiment():
    """
    Run the full two-phase experiment with all configurations.

    Configurations:
    - 4 baselines (TopK: K=8, 16, 32, 64)
    - 16 BH configs (max_k=[8,16,32,64] × alpha=[0.30,0.40,0.50,0.60])
    - 3 datasets (WikiText, LAMBADA, HellaSwag)
    - 200 samples each
    - Total: 60 experiments, ~2-3 hours on A100
    """
    print("=" * 70)
    print("FULL TWO-PHASE EXPERIMENT")
    print("=" * 70)
    print("\nThis will run 60 experiments (4 baseline + 16 BH × 3 datasets)")
    print("Estimated time: 2-3 hours on A100 GPU")
    print("Output: 120 JSON files + results CSV + visualizations")
    print("\nPress Ctrl+C to cancel, or wait 5 seconds to continue...")

    import time
    time.sleep(5)

    runner = BHRoutingExperimentRunner(
        model_name="allenai/OLMoE-1B-7B-0924",
        device="cuda",
        output_dir="./bh_routing_experiment"
    )

    results_df = runner.run_two_phase_experiment(
        baseline_k_values=[8, 16, 32, 64],
        bh_max_k_values=[8, 16, 32, 64],
        bh_alpha_values=[0.30, 0.40, 0.50, 0.60],
        datasets=['wikitext', 'lambada', 'hellaswag'],
        max_samples=200
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    # Show top configurations by efficiency
    print("\nTop 5 configurations by FLOPs reduction:")
    top_efficient = results_df.nlargest(5, 'flops_reduction_pct')[
        ['config_name', 'dataset', 'flops_reduction_pct', 'perplexity', 'avg_experts']
    ]
    print(top_efficient.to_string(index=False))

    # Show quality comparison
    print("\nPerplexity comparison (WikiText):")
    wiki_results = results_df[results_df['dataset'] == 'wikitext'].sort_values('perplexity')
    print(wiki_results[['config_name', 'perplexity', 'avg_experts']].head(10).to_string(index=False))

    # Generate comprehensive visualization
    print("\nGenerating comprehensive visualization...")
    fig = create_comprehensive_visualization(
        results_df,
        output_path='./bh_routing_experiment/visualizations/bh_comprehensive_comparison.png'
    )

    print("\n✅ Full experiment complete!")
    print(f"   Results saved to: ./bh_routing_experiment/")
    print(f"   Total files: {len(results_df) * 2} JSON files + CSV + visualization")


def run_custom_experiment(
    baseline_k=[8],
    bh_max_k=[8, 16],
    bh_alpha=[0.30, 0.50],
    datasets=['wikitext'],
    max_samples=100
):
    """
    Run a custom experiment with specified parameters.

    Args:
        baseline_k: List of K values for baseline TopK
        bh_max_k: List of max_k values for BH routing
        bh_alpha: List of alpha values for BH routing
        datasets: List of datasets to evaluate on
        max_samples: Samples per dataset

    Returns:
        DataFrame with results
    """
    print("=" * 70)
    print("CUSTOM EXPERIMENT")
    print("=" * 70)
    print(f"Baselines: {baseline_k}")
    print(f"BH max_k: {bh_max_k}")
    print(f"BH alpha: {bh_alpha}")
    print(f"Datasets: {datasets}")
    print(f"Samples: {max_samples}")

    total_experiments = (
        len(baseline_k) * len(datasets) +
        len(bh_max_k) * len(bh_alpha) * len(datasets)
    )
    print(f"\nTotal experiments: {total_experiments}")

    runner = BHRoutingExperimentRunner(
        model_name="allenai/OLMoE-1B-7B-0924",
        device="cuda",
        output_dir="./bh_routing_custom"
    )

    results_df = runner.run_two_phase_experiment(
        baseline_k_values=baseline_k,
        bh_max_k_values=bh_max_k,
        bh_alpha_values=bh_alpha,
        datasets=datasets,
        max_samples=max_samples
    )

    # Generate visualization
    print("\nGenerating visualization...")
    fig = create_comprehensive_visualization(
        results_df,
        output_path='./bh_routing_custom/visualizations/custom_comparison.png'
    )

    print("\n✅ Custom experiment complete!")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BH Routing Evaluation Framework")
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with minimal configurations (~5 min)'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full experiment with all configurations (~2-3 hours)'
    )

    args = parser.parse_args()

    if args.quick_test:
        run_quick_test()
    elif args.full:
        run_full_experiment()
    else:
        # Default: run custom moderate experiment
        print("No mode specified. Running moderate custom experiment...")
        print("Use --quick-test for fast verification or --full for complete evaluation")
        print()

        run_custom_experiment(
            baseline_k=[8, 16],
            bh_max_k=[8, 16],
            bh_alpha=[0.40, 0.50],
            datasets=['wikitext', 'lambada'],
            max_samples=100
        )
