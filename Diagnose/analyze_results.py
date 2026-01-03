#!/usr/bin/env python3
"""
OLMoE Routing Experiments - Results Analyzer
=============================================

Advanced analysis and comparison tool for routing experiment results.

Usage:
    # Load and summarize results
    python analyze_results.py summary <results_dir>

    # Analyze specific strategy
    python analyze_results.py strategy regular <results_dir>

    # Compare all strategies
    python analyze_results.py compare <results_dir>

    # Find optimal configuration
    python analyze_results.py optimize --quality-weight 0.7 --speed-weight 0.3 <results_dir>

    # Generate comparison plots
    python analyze_results.py plot <results_dir>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ResultAnalyzer:
    """Analyzer for routing experiment results."""

    def __init__(self, results_dir: str):
        """
        Initialize analyzer with results directory.

        Args:
            results_dir: Path to experiment results directory
        """
        self.results_dir = Path(results_dir)
        self.csv_file = self.results_dir / "all_results.csv"
        self.json_file = self.results_dir / "all_results.json"
        self.logs_dir = self.results_dir / "logs"

        # Load data
        if not self.csv_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.csv_file}")

        self.df = pd.read_csv(self.csv_file)
        print(f"✅ Loaded {len(self.df)} experiment results from {results_dir}")

    def get_summary(self) -> Dict:
        """Get overall summary statistics."""
        summary = {
            'total_experiments': len(self.df),
            'expert_counts': sorted(self.df['num_experts'].unique().tolist()),
            'strategies': sorted(self.df['strategy'].unique().tolist()),
            'datasets': sorted(self.df['dataset'].unique().tolist()),
            'best_perplexity': {
                'value': self.df['perplexity'].min(),
                'config': self.df.loc[self.df['perplexity'].idxmin(), 'config'],
                'dataset': self.df.loc[self.df['perplexity'].idxmin(), 'dataset']
            },
            'best_accuracy': {
                'value': self.df['token_accuracy'].max(),
                'config': self.df.loc[self.df['token_accuracy'].idxmax(), 'config'],
                'dataset': self.df.loc[self.df['token_accuracy'].idxmax(), 'dataset']
            },
            'best_speed': {
                'value': self.df['tokens_per_second'].max(),
                'config': self.df.loc[self.df['tokens_per_second'].idxmax(), 'config'],
                'dataset': self.df.loc[self.df['tokens_per_second'].idxmax(), 'dataset']
            },
            'avg_perplexity': self.df['perplexity'].mean(),
            'avg_accuracy': self.df['token_accuracy'].mean(),
            'avg_speed': self.df['tokens_per_second'].mean()
        }
        return summary

    def print_summary(self):
        """Print overall summary."""
        summary = self.get_summary()

        print("\n" + "=" * 70)
        print("EXPERIMENT SUMMARY")
        print("=" * 70)

        print(f"\nTotal Experiments: {summary['total_experiments']}")
        print(f"Expert Counts Tested: {summary['expert_counts']}")
        print(f"Strategies Tested: {summary['strategies']}")
        print(f"Datasets Used: {summary['datasets']}")

        print("\n🏆 BEST RESULTS:")
        print(f"\n  Lowest Perplexity: {summary['best_perplexity']['value']:.2f}")
        print(f"    Config: {summary['best_perplexity']['config']}")
        print(f"    Dataset: {summary['best_perplexity']['dataset']}")

        print(f"\n  Highest Accuracy: {summary['best_accuracy']['value']:.4f}")
        print(f"    Config: {summary['best_accuracy']['config']}")
        print(f"    Dataset: {summary['best_accuracy']['dataset']}")

        print(f"\n  Fastest Speed: {summary['best_speed']['value']:.1f} tok/s")
        print(f"    Config: {summary['best_speed']['config']}")
        print(f"    Dataset: {summary['best_speed']['dataset']}")

        print("\n📊 AVERAGES:")
        print(f"  Perplexity: {summary['avg_perplexity']:.2f}")
        print(f"  Accuracy: {summary['avg_accuracy']:.4f}")
        print(f"  Speed: {summary['avg_speed']:.1f} tok/s")

        print("\n" + "=" * 70)

    def analyze_strategy(self, strategy_name: str) -> pd.DataFrame:
        """
        Analyze a specific routing strategy.

        Args:
            strategy_name: Name of strategy to analyze

        Returns:
            DataFrame with strategy breakdown
        """
        strategy_df = self.df[self.df['strategy'] == strategy_name].copy()

        if len(strategy_df) == 0:
            print(f"❌ No results found for strategy: {strategy_name}")
            return pd.DataFrame()

        print("\n" + "=" * 70)
        print(f"STRATEGY ANALYSIS: {strategy_name.upper()}")
        print("=" * 70)

        # Group by expert count
        by_experts = strategy_df.groupby('num_experts').agg({
            'perplexity': ['mean', 'std', 'min'],
            'token_accuracy': ['mean', 'std', 'max'],
            'tokens_per_second': ['mean', 'std', 'max']
        }).round(4)

        print("\nPerformance by Expert Count:")
        print(by_experts)

        # Best configuration
        best_idx = strategy_df['perplexity'].idxmin()
        best_row = strategy_df.loc[best_idx]

        print(f"\n🏆 Best Configuration:")
        print(f"  Expert Count: {best_row['num_experts']}")
        print(f"  Perplexity: {best_row['perplexity']:.2f}")
        print(f"  Accuracy: {best_row['token_accuracy']:.4f}")
        print(f"  Speed: {best_row['tokens_per_second']:.1f} tok/s")
        print(f"  Dataset: {best_row['dataset']}")

        print("\n" + "=" * 70)

        return strategy_df

    def analyze_expert_count(self, num_experts: int) -> pd.DataFrame:
        """
        Analyze a specific expert count across strategies.

        Args:
            num_experts: Number of experts to analyze

        Returns:
            DataFrame with expert count breakdown
        """
        expert_df = self.df[self.df['num_experts'] == num_experts].copy()

        if len(expert_df) == 0:
            print(f"❌ No results found for {num_experts} experts")
            return pd.DataFrame()

        print("\n" + "=" * 70)
        print(f"EXPERT COUNT ANALYSIS: {num_experts} EXPERTS")
        print("=" * 70)

        # Group by strategy
        by_strategy = expert_df.groupby('strategy').agg({
            'perplexity': ['mean', 'std', 'min'],
            'token_accuracy': ['mean', 'std', 'max'],
            'tokens_per_second': ['mean', 'std', 'max']
        }).round(4)

        print("\nPerformance by Strategy:")
        print(by_strategy)

        # Best strategy
        best_idx = expert_df['perplexity'].idxmin()
        best_row = expert_df.loc[best_idx]

        print(f"\n🏆 Best Strategy:")
        print(f"  Strategy: {best_row['strategy']}")
        print(f"  Perplexity: {best_row['perplexity']:.2f}")
        print(f"  Accuracy: {best_row['token_accuracy']:.4f}")
        print(f"  Speed: {best_row['tokens_per_second']:.1f} tok/s")
        print(f"  Dataset: {best_row['dataset']}")

        print("\n" + "=" * 70)

        return expert_df

    def find_optimal_config(
        self,
        quality_weight: float = 0.5,
        speed_weight: float = 0.5
    ) -> Dict:
        """
        Find optimal configuration based on weighted quality and speed.

        Args:
            quality_weight: Weight for quality metrics (0-1)
            speed_weight: Weight for speed metrics (0-1)

        Returns:
            Dictionary with optimal configuration
        """
        # Normalize weights
        total_weight = quality_weight + speed_weight
        quality_weight /= total_weight
        speed_weight /= total_weight

        # Normalize metrics to [0, 1]
        df = self.df.copy()

        # For perplexity, lower is better, so invert
        df['norm_ppl'] = 1 - (df['perplexity'] - df['perplexity'].min()) / (
            df['perplexity'].max() - df['perplexity'].min() + 1e-10
        )

        # For accuracy, higher is better
        df['norm_acc'] = (df['token_accuracy'] - df['token_accuracy'].min()) / (
            df['token_accuracy'].max() - df['token_accuracy'].min() + 1e-10
        )

        # For speed, higher is better
        df['norm_speed'] = (df['tokens_per_second'] - df['tokens_per_second'].min()) / (
            df['tokens_per_second'].max() - df['tokens_per_second'].min() + 1e-10
        )

        # Calculate quality score (average of perplexity and accuracy)
        df['quality_score'] = (df['norm_ppl'] + df['norm_acc']) / 2

        # Calculate combined score
        df['combined_score'] = (
            quality_weight * df['quality_score'] +
            speed_weight * df['norm_speed']
        )

        # Find best
        best_idx = df['combined_score'].idxmax()
        best_row = df.loc[best_idx]

        optimal = {
            'config': best_row['config'],
            'num_experts': int(best_row['num_experts']),
            'strategy': best_row['strategy'],
            'dataset': best_row['dataset'],
            'perplexity': float(best_row['perplexity']),
            'token_accuracy': float(best_row['token_accuracy']),
            'tokens_per_second': float(best_row['tokens_per_second']),
            'quality_score': float(best_row['quality_score']),
            'speed_score': float(best_row['norm_speed']),
            'combined_score': float(best_row['combined_score']),
            'weights': {
                'quality': quality_weight,
                'speed': speed_weight
            }
        }

        print("\n" + "=" * 70)
        print("OPTIMAL CONFIGURATION")
        print("=" * 70)
        print(f"\nWeights: Quality={quality_weight:.2f}, Speed={speed_weight:.2f}")
        print(f"\n🏆 Optimal Config: {optimal['config']}")
        print(f"  Strategy: {optimal['strategy']}")
        print(f"  Expert Count: {optimal['num_experts']}")
        print(f"  Dataset: {optimal['dataset']}")
        print(f"\nMetrics:")
        print(f"  Perplexity: {optimal['perplexity']:.2f}")
        print(f"  Accuracy: {optimal['token_accuracy']:.4f}")
        print(f"  Speed: {optimal['tokens_per_second']:.1f} tok/s")
        print(f"\nScores:")
        print(f"  Quality Score: {optimal['quality_score']:.4f}")
        print(f"  Speed Score: {optimal['speed_score']:.4f}")
        print(f"  Combined Score: {optimal['combined_score']:.4f}")
        print("\n" + "=" * 70)

        return optimal

    def compare_strategies(self) -> pd.DataFrame:
        """
        Create comparison table of all strategies.

        Returns:
            DataFrame with strategy comparison
        """
        comparison = self.df.groupby('strategy').agg({
            'perplexity': ['mean', 'std', 'min', 'max'],
            'token_accuracy': ['mean', 'std', 'min', 'max'],
            'tokens_per_second': ['mean', 'std', 'min', 'max'],
            'avg_entropy': 'mean',
            'weight_concentration': 'mean',
            'unique_experts_activated': 'mean'
        }).round(4)

        print("\n" + "=" * 70)
        print("STRATEGY COMPARISON")
        print("=" * 70)
        print("\n")
        print(comparison)
        print("\n" + "=" * 70)

        return comparison

    def create_comparison_plot(self, output_file: Optional[str] = None):
        """
        Create comprehensive comparison plots.

        Args:
            output_file: Optional output file path
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Strategy Comparison Analysis', fontsize=16, fontweight='bold')

        # 1. Perplexity by strategy (box plot)
        ax1 = axes[0, 0]
        self.df.boxplot(column='perplexity', by='strategy', ax=ax1)
        ax1.set_title('Perplexity Distribution by Strategy')
        ax1.set_xlabel('Strategy')
        ax1.set_ylabel('Perplexity (↓ better)')
        plt.sca(ax1)
        plt.xticks(rotation=45)

        # 2. Accuracy by strategy (box plot)
        ax2 = axes[0, 1]
        self.df.boxplot(column='token_accuracy', by='strategy', ax=ax2)
        ax2.set_title('Token Accuracy Distribution by Strategy')
        ax2.set_xlabel('Strategy')
        ax2.set_ylabel('Token Accuracy (↑ better)')
        plt.sca(ax2)
        plt.xticks(rotation=45)

        # 3. Speed-Quality scatter
        ax3 = axes[1, 0]
        for strategy in self.df['strategy'].unique():
            strategy_df = self.df[self.df['strategy'] == strategy]
            ax3.scatter(
                strategy_df['perplexity'],
                strategy_df['tokens_per_second'],
                label=strategy,
                s=100,
                alpha=0.6
            )
        ax3.set_xlabel('Perplexity (↓ better)')
        ax3.set_ylabel('Tokens/Second (↑ better)')
        ax3.set_title('Speed vs Quality Trade-off')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Expert count effect on perplexity
        ax4 = axes[1, 1]
        for strategy in self.df['strategy'].unique():
            strategy_df = self.df[self.df['strategy'] == strategy].groupby('num_experts')['perplexity'].mean()
            ax4.plot(strategy_df.index, strategy_df.values, marker='o', label=strategy, linewidth=2)
        ax4.set_xlabel('Number of Experts')
        ax4.set_ylabel('Perplexity (↓ better)')
        ax4.set_title('Perplexity vs Expert Count')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_file is None:
            output_file = self.results_dir / "comparison_analysis.png"

        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n📊 Comparison plot saved to: {output_file}")
        plt.close()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze OLMoE routing experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Analysis command')

    # Summary command
    summary_parser = subparsers.add_parser('summary', help='Show overall summary')
    summary_parser.add_argument('results_dir', help='Results directory')

    # Strategy command
    strategy_parser = subparsers.add_parser('strategy', help='Analyze specific strategy')
    strategy_parser.add_argument('strategy_name', help='Strategy to analyze')
    strategy_parser.add_argument('results_dir', help='Results directory')

    # Expert count command
    expert_parser = subparsers.add_parser('experts', help='Analyze specific expert count')
    expert_parser.add_argument('num_experts', type=int, help='Number of experts')
    expert_parser.add_argument('results_dir', help='Results directory')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare all strategies')
    compare_parser.add_argument('results_dir', help='Results directory')

    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Find optimal configuration')
    optimize_parser.add_argument('results_dir', help='Results directory')
    optimize_parser.add_argument('--quality-weight', type=float, default=0.5,
                                 help='Weight for quality (default: 0.5)')
    optimize_parser.add_argument('--speed-weight', type=float, default=0.5,
                                 help='Weight for speed (default: 0.5)')

    # Plot command
    plot_parser = subparsers.add_parser('plot', help='Generate comparison plots')
    plot_parser.add_argument('results_dir', help='Results directory')
    plot_parser.add_argument('--output', help='Output file path')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        # Create analyzer
        analyzer = ResultAnalyzer(args.results_dir)

        # Execute command
        if args.command == 'summary':
            analyzer.print_summary()

        elif args.command == 'strategy':
            analyzer.analyze_strategy(args.strategy_name)

        elif args.command == 'experts':
            analyzer.analyze_expert_count(args.num_experts)

        elif args.command == 'compare':
            analyzer.compare_strategies()

        elif args.command == 'optimize':
            analyzer.find_optimal_config(
                quality_weight=args.quality_weight,
                speed_weight=args.speed_weight
            )

        elif args.command == 'plot':
            analyzer.create_comparison_plot(args.output)

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
