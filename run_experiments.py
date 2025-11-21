#!/usr/bin/env python3
"""
OLMoE Routing Experiments - CLI Runner
=======================================

User-friendly command-line interface for running routing experiments.

Quick Mode:
    python run_experiments.py --quick

Standard Mode:
    python run_experiments.py

Custom Configuration:
    python run_experiments.py --experts 4 8 16 --strategies regular normalized

Usage Examples:
    # Quick test (5 minutes)
    python run_experiments.py --quick

    # Full experiment (60 minutes)
    python run_experiments.py

    # Custom configuration
    python run_experiments.py --experts 8 16 32 --strategies regular uniform --max-samples 200

    # Specific datasets
    python run_experiments.py --datasets wikitext --max-samples 1000
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from olmoe_routing_experiments import RoutingExperimentRunner


def print_banner():
    """Print welcome banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║            OLMoE ROUTING EXPERIMENTS FRAMEWORK                       ║
║                                                                      ║
║  Systematic evaluation of expert routing strategies for OLMoE       ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_config_summary(args, expert_counts, strategies, datasets):
    """Print experiment configuration summary."""
    print("\n" + "=" * 70)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 70)
    print(f"\nMode: {'QUICK' if args.quick else 'STANDARD'}")
    print(f"Expert Counts: {expert_counts}")
    print(f"Strategies: {strategies}")
    print(f"Datasets: {datasets}")
    print(f"Max Samples per Dataset: {args.max_samples}")
    print(f"\nTotal Experiments: {len(expert_counts) * len(strategies) * len(datasets)}")

    # Estimate time
    experiments_count = len(expert_counts) * len(strategies) * len(datasets)
    estimated_minutes = experiments_count * 2  # Rough estimate: 2 min per experiment
    print(f"Estimated Time: ~{estimated_minutes} minutes")
    print("\n" + "=" * 70)


def print_results_summary(df, output_dir):
    """Print results summary."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Best configurations
    best_ppl = df.loc[df['perplexity'].idxmin()]
    best_acc = df.loc[df['token_accuracy'].idxmax()]
    best_speed = df.loc[df['tokens_per_second'].idxmax()]

    print("\n🏆 BEST CONFIGURATIONS:")
    print(f"\n  Lowest Perplexity:")
    print(f"    Configuration: {best_ppl['config']}")
    print(f"    Perplexity: {best_ppl['perplexity']:.2f}")
    print(f"    Dataset: {best_ppl['dataset']}")

    print(f"\n  Highest Accuracy:")
    print(f"    Configuration: {best_acc['config']}")
    print(f"    Token Accuracy: {best_acc['token_accuracy']:.4f}")
    print(f"    Dataset: {best_acc['dataset']}")

    print(f"\n  Fastest Inference:")
    print(f"    Configuration: {best_speed['config']}")
    print(f"    Speed: {best_speed['tokens_per_second']:.1f} tokens/second")
    print(f"    Dataset: {best_speed['dataset']}")

    # Summary statistics by strategy
    print("\n📊 STRATEGY COMPARISON (Averaged across all configs):")
    print()
    strategy_summary = df.groupby('strategy').agg({
        'perplexity': 'mean',
        'token_accuracy': 'mean',
        'tokens_per_second': 'mean'
    }).round(4)

    for strategy in strategy_summary.index:
        row = strategy_summary.loc[strategy]
        print(f"  {strategy.upper()}:")
        print(f"    Perplexity: {row['perplexity']:.2f}")
        print(f"    Accuracy: {row['token_accuracy']:.4f}")
        print(f"    Speed: {row['tokens_per_second']:.1f} tok/s")
        print()

    # File locations
    print("=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    print(f"\n📁 Output Directory: {output_dir}")
    print(f"\n  📄 all_results.csv - Full results table")
    print(f"  📄 all_results.json - JSON format results")
    print(f"  📄 EXPERIMENT_REPORT.md - Detailed report")
    print(f"  📁 logs/ - Individual experiment logs")
    print(f"  📁 visualizations/ - Analysis plots")
    print()
    print("=" * 70)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run OLMoE routing experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run (~5 minutes)
  python run_experiments.py --quick

  # Full experiment with all configurations
  python run_experiments.py

  # Custom expert counts and strategies
  python run_experiments.py --experts 8 16 32 --strategies regular normalized

  # Evaluate on specific dataset with more samples
  python run_experiments.py --datasets wikitext --max-samples 1000
        """
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: 2 expert counts, 2 strategies, 100 samples (~5 min)'
    )

    parser.add_argument(
        '--experts',
        type=int,
        nargs='+',
        help='Expert counts to test (default: [4, 8, 16, 32, 64])'
    )

    parser.add_argument(
        '--strategies',
        type=str,
        nargs='+',
        choices=['regular', 'normalized', 'uniform', 'adaptive'],
        help='Routing strategies to test (default: regular, normalized, uniform)'
    )

    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        choices=['wikitext', 'lambada', 'piqa'],
        help='Datasets to evaluate on (default: wikitext, lambada)'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=500,
        help='Maximum samples per dataset (default: 500)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./routing_experiments',
        help='Output directory for results (default: ./routing_experiments)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='allenai/OLMoE-1B-7B-0924',
        help='Model to use (default: allenai/OLMoE-1B-7B-0924)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use (default: auto)'
    )

    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Skip generating visualizations'
    )

    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip generating markdown report'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()

    # Print banner
    print_banner()

    # Determine configurations
    if args.quick:
        expert_counts = args.experts or [8, 16]
        strategies = args.strategies or ['regular', 'normalized']
        datasets = args.datasets or ['wikitext']
        max_samples = 100
    else:
        expert_counts = args.experts or [4, 8, 16, 32, 64]
        strategies = args.strategies or ['regular', 'normalized', 'uniform']
        datasets = args.datasets or ['wikitext', 'lambada']
        max_samples = args.max_samples

    # Print configuration
    print_config_summary(args, expert_counts, strategies, datasets)

    # Confirm with user
    print("\n⚠️  This will run experiments with the above configuration.")
    response = input("\nContinue? [Y/n]: ").strip().lower()

    if response and response not in ['y', 'yes']:
        print("\n❌ Aborted by user")
        sys.exit(0)

    print("\n🚀 Starting experiments...\n")

    try:
        # Create experiment runner
        runner = RoutingExperimentRunner(
            model_name=args.model,
            device=args.device,
            output_dir=args.output_dir
        )

        # Run all experiments
        results_df = runner.run_all_experiments(
            expert_counts=expert_counts,
            strategies=strategies,
            datasets=datasets,
            max_samples=max_samples
        )

        # Generate visualizations
        if not args.no_visualize:
            print("\n📊 Generating visualizations...")
            runner.visualize_results(results_df)
            print("✅ Visualizations complete")

        # Generate report
        if not args.no_report:
            print("\n📝 Generating report...")
            runner.generate_report(results_df)
            print("✅ Report complete")

        # Print summary
        print_results_summary(results_df, runner.output_dir)

        print("\n✅ ALL EXPERIMENTS COMPLETE!")
        print(f"\n📁 Results saved to: {runner.output_dir}\n")

        return 0

    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        return 1

    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
