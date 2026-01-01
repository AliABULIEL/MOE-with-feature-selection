"""
HC Routing Visualization Suite - ENHANCED
==========================================

This module implements comprehensive visualizations for HC routing
evaluation, including critical diagnostic plots for debugging.

ENHANCED PANELS (Added):
- M1: Per-Token Loss Distribution
- M2: Perplexity vs Expert Count Scatter
- M3: HC vs TopK Expert Overlap (CRITICAL)
- M4: Weight Distribution Box Plot
- M5: Weight Sum Distribution (diagnoses variable weight issue)

Original Panels:
1. Perplexity Comparison (baseline vs HC)
2. Task Accuracy Comparison
3. Expert Efficiency (avg experts selected)
4. HC Type Sensitivity (plus/standard/modified)
5. Pareto Frontier (efficiency vs quality)
6. Routing Behavior Summary (floor/mid/ceiling distribution)
7. HC Statistics Distribution (threshold ranks, max values)
8. Layer-wise Analysis
9. Speed-Quality Trade-off

Usage:
    from hc_routing_visualization import (
        create_comprehensive_visualization,
        create_diagnostic_plots,
        plot_weight_sum_distribution,
        plot_expert_overlap_analysis
    )

    # Standard visualization
    fig = create_comprehensive_visualization(results_df)
    
    # Diagnostic plots for debugging HC issues
    fig = create_diagnostic_plots(internal_logs, baseline_logs)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# NEW DIAGNOSTIC PLOTS (Critical for debugging HC routing issues)
# =============================================================================

def plot_weight_sum_distribution(
    hc_weight_sums: List[float],
    topk_weight_sums: Optional[List[float]] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    M5: Plot weight sum distribution - diagnoses variable weight sum issue.
    
    This is CRITICAL for understanding why HC has higher perplexity.
    OLMoE expects weight_sum ≈ 0.40 (from k=8 training).
    
    Args:
        hc_weight_sums: List of weight sums from HC routing
        topk_weight_sums: Optional list of weight sums from TopK (for comparison)
        output_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    
    Example:
        >>> # Get weight sums from integration
        >>> analysis = integration.get_weight_sum_analysis()
        >>> plot_weight_sum_distribution(hc_sums, topk_sums)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Histogram comparison
    ax1 = axes[0]
    ax1.hist(hc_weight_sums, bins=50, alpha=0.7, label='HC Routing', 
             color='blue', edgecolor='black', density=True)
    
    if topk_weight_sums:
        ax1.hist(topk_weight_sums, bins=50, alpha=0.7, label='TopK-8', 
                 color='red', edgecolor='black', density=True)
    
    # Mark expected value
    ax1.axvline(x=0.40, color='green', linestyle='--', linewidth=2, 
                label='Expected (k=8 training)')
    ax1.axvline(x=np.mean(hc_weight_sums), color='blue', linestyle=':', 
                linewidth=2, label=f'HC mean: {np.mean(hc_weight_sums):.3f}')
    
    ax1.set_xlabel('Weight Sum', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Weight Sum Distribution\n(Should be ~0.40 like k=8)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Right: Box plot by value range
    ax2 = axes[1]
    data_to_plot = [hc_weight_sums]
    labels = ['HC']
    
    if topk_weight_sums:
        data_to_plot.append(topk_weight_sums)
        labels.append('TopK-8')
    
    bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
        patch.set_facecolor(color)
    
    ax2.axhline(y=0.40, color='green', linestyle='--', linewidth=2, 
                label='Expected (k=8)')
    ax2.set_ylabel('Weight Sum', fontsize=12)
    ax2.set_title('Weight Sum Comparison', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistics annotation
    stats_text = f"HC Statistics:\n"
    stats_text += f"Mean: {np.mean(hc_weight_sums):.4f}\n"
    stats_text += f"Std: {np.std(hc_weight_sums):.4f}\n"
    stats_text += f"Min: {np.min(hc_weight_sums):.4f}\n"
    stats_text += f"Max: {np.max(hc_weight_sums):.4f}"
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
             verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
    
    return fig


def plot_per_token_loss_distribution(
    hc_losses: List[float],
    topk_losses: List[float],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    M1: Plot per-token loss distribution to identify if degradation is uniform or outlier-driven.
    
    Args:
        hc_losses: Per-token losses from HC routing
        topk_losses: Per-token losses from TopK routing
        output_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Left: Histogram overlay
    ax1 = axes[0]
    ax1.hist(topk_losses, bins=50, alpha=0.7, label=f'TopK-8 (mean={np.mean(topk_losses):.3f})', 
             color='green', edgecolor='black', density=True)
    ax1.hist(hc_losses, bins=50, alpha=0.7, label=f'HC (mean={np.mean(hc_losses):.3f})', 
             color='red', edgecolor='black', density=True)
    ax1.set_xlabel('Token Loss', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Per-Token Loss Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Middle: CDF comparison
    ax2 = axes[1]
    topk_sorted = np.sort(topk_losses)
    hc_sorted = np.sort(hc_losses)
    ax2.plot(topk_sorted, np.linspace(0, 1, len(topk_sorted)), 
             label='TopK-8', color='green', linewidth=2)
    ax2.plot(hc_sorted, np.linspace(0, 1, len(hc_sorted)), 
             label='HC', color='red', linewidth=2)
    ax2.set_xlabel('Token Loss', fontsize=12)
    ax2.set_ylabel('Cumulative Probability', fontsize=12)
    ax2.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Right: Box plot comparison
    ax3 = axes[2]
    bp = ax3.boxplot([topk_losses, hc_losses], labels=['TopK-8', 'HC'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax3.set_ylabel('Token Loss', fontsize=12)
    ax3.set_title('Loss Comparison', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add ratio annotation
    ratio = np.mean(hc_losses) / np.mean(topk_losses)
    ax3.text(0.5, 0.95, f'HC/TopK Ratio: {ratio:.2f}×', 
             transform=ax3.transAxes, ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
    
    return fig


def plot_perplexity_vs_expert_count(
    results: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    M2: Scatter plot of perplexity vs average expert count.
    
    Shows the relationship between expert selection and model quality.
    Reveals the non-monotonic pattern (TopK-16 worse than TopK-32).
    
    Args:
        results: List of dicts with 'avg_experts', 'perplexity', 'routing_type', 'config'
        output_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    
    Example result format:
        {
            'routing_type': 'topk' or 'hc',
            'config': 'TopK-8' or 'HC β=0.5',
            'avg_experts': 8.0,
            'perplexity': 84.34
        }
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Separate by routing type
    topk_results = [r for r in results if r.get('routing_type') == 'topk']
    hc_results = [r for r in results if r.get('routing_type') == 'hc']
    
    # Plot TopK results
    if topk_results:
        x_topk = [r['avg_experts'] for r in topk_results]
        y_topk = [r['perplexity'] for r in topk_results]
        labels_topk = [r.get('config', 'TopK') for r in topk_results]
        
        ax.scatter(x_topk, y_topk, s=200, marker='*', c='red', 
                   edgecolors='black', linewidths=2, label='TopK Baseline', zorder=10)
        
        # Add labels
        for x, y, label in zip(x_topk, y_topk, labels_topk):
            ax.annotate(label, (x, y), textcoords="offset points", 
                       xytext=(10, 5), fontsize=9)
    
    # Plot HC results
    if hc_results:
        x_hc = [r['avg_experts'] for r in hc_results]
        y_hc = [r['perplexity'] for r in hc_results]
        labels_hc = [r.get('config', 'HC') for r in hc_results]
        
        ax.scatter(x_hc, y_hc, s=150, marker='o', c='blue', 
                   edgecolors='black', linewidths=1, alpha=0.7, label='HC Routing')
        
        # Add labels
        for x, y, label in zip(x_hc, y_hc, labels_hc):
            ax.annotate(label, (x, y), textcoords="offset points", 
                       xytext=(10, -10), fontsize=9)
    
    # Add reference lines
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Target PPL < 100')
    
    ax.set_xlabel('Average Experts Selected', fontsize=12)
    ax.set_ylabel('Perplexity (lower is better)', fontsize=12)
    ax.set_title('Perplexity vs Expert Count\n(Reveals non-monotonic pattern)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Log scale for perplexity if range is large
    if max([r['perplexity'] for r in results]) > 300:
        ax.set_yscale('log')
        ax.set_ylabel('Perplexity (log scale, lower is better)', fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
    
    return fig


def plot_expert_overlap_analysis(
    hc_selections: Dict[int, List[List[int]]],  # layer_idx -> list of expert selections per token
    topk_selections: Dict[int, List[List[int]]],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    M3: Plot HC vs TopK expert overlap per layer - CRITICAL diagnostic.
    
    This verifies the "100% overlap" claim. If overlap is low, HC is 
    selecting DIFFERENT experts than TopK, explaining perplexity degradation.
    
    Args:
        hc_selections: Dict mapping layer_idx to list of expert index lists
        topk_selections: Dict mapping layer_idx to list of expert index lists
        output_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Compute overlap per layer
    layers = sorted(set(hc_selections.keys()) & set(topk_selections.keys()))
    
    jaccard_scores = []
    intersection_counts = []
    
    for layer_idx in layers:
        hc_lists = hc_selections[layer_idx]
        topk_lists = topk_selections[layer_idx]
        
        layer_jaccards = []
        layer_intersections = []
        
        for hc_exp, topk_exp in zip(hc_lists, topk_lists):
            hc_set = set(hc_exp) - {-1}  # Remove padding
            topk_set = set(topk_exp) - {-1}
            
            if hc_set and topk_set:
                intersection = len(hc_set & topk_set)
                union = len(hc_set | topk_set)
                jaccard = intersection / union if union > 0 else 0
                layer_jaccards.append(jaccard)
                layer_intersections.append(intersection)
        
        jaccard_scores.append(np.mean(layer_jaccards) if layer_jaccards else 0)
        intersection_counts.append(np.mean(layer_intersections) if layer_intersections else 0)
    
    # Left: Jaccard similarity per layer
    ax1 = axes[0]
    bars = ax1.bar(layers, jaccard_scores, color='steelblue', edgecolor='black', alpha=0.8)
    ax1.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Perfect Overlap')
    ax1.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='50% Overlap')
    
    # Color bars by quality
    for bar, score in zip(bars, jaccard_scores):
        if score > 0.8:
            bar.set_facecolor('green')
        elif score > 0.5:
            bar.set_facecolor('orange')
        else:
            bar.set_facecolor('red')
    
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('Jaccard Similarity', fontsize=12)
    ax1.set_title('Expert Overlap: HC vs TopK-8 per Layer\n(Jaccard Similarity)', 
                  fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Average intersection count
    ax2 = axes[1]
    ax2.bar(layers, intersection_counts, color='coral', edgecolor='black', alpha=0.8)
    ax2.axhline(y=8, color='green', linestyle='--', linewidth=2, label='TopK-8 count')
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('Avg Experts in Common', fontsize=12)
    ax2.set_title('Average Expert Intersection Count', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add overall summary
    avg_jaccard = np.mean(jaccard_scores)
    summary = f"Overall Jaccard: {avg_jaccard:.3f}\n"
    summary += "Green: >80% | Orange: 50-80% | Red: <50%"
    fig.text(0.5, -0.02, summary, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
    
    return fig


def plot_weight_distribution_boxplot(
    hc_weights: Dict[int, List[float]],  # layer_idx -> list of max weights per token
    topk_weights: Optional[Dict[int, List[float]]] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    M4: Plot weight distribution box plot per layer.
    
    Shows if HC is spreading weight too thinly or concentrating correctly.
    
    Args:
        hc_weights: Dict mapping layer_idx to list of max weight per token
        topk_weights: Optional dict for TopK comparison
        output_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    layers = sorted(hc_weights.keys())
    
    data = []
    positions = []
    labels = []
    
    for i, layer_idx in enumerate(layers):
        data.append(hc_weights[layer_idx])
        positions.append(i * 3)
        labels.append(f'L{layer_idx}')
        
        if topk_weights and layer_idx in topk_weights:
            data.append(topk_weights[layer_idx])
            positions.append(i * 3 + 1)
    
    bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.8)
    
    # Color HC blue, TopK red
    for i, (patch, pos) in enumerate(zip(bp['boxes'], positions)):
        if pos % 3 == 0:  # HC
            patch.set_facecolor('lightblue')
        else:  # TopK
            patch.set_facecolor('lightcoral')
    
    ax.set_xticks([i * 3 + 0.5 for i in range(len(layers))])
    ax.set_xticklabels(labels)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Max Routing Weight', fontsize=12)
    ax.set_title('Routing Weight Distribution per Layer\n(Blue=HC, Red=TopK)', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
    
    return fig


def create_diagnostic_plots(
    internal_logs: List[Dict[str, Any]],
    baseline_logs: Optional[List[Dict[str, Any]]] = None,
    output_dir: str = './diagnostic_plots',
    figsize: Tuple[int, int] = (14, 10)
) -> Dict[str, plt.Figure]:
    """
    Create all diagnostic plots from internal logs.
    
    This is the main entry point for debugging HC routing issues.
    
    Args:
        internal_logs: List of sample logs from HC routing
        baseline_logs: Optional list of sample logs from TopK routing
        output_dir: Directory to save plots
        figsize: Figure size for combined plot
    
    Returns:
        Dict mapping plot name to Figure
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    figures = {}
    
    # Extract data from logs
    hc_losses = []
    hc_weight_sums = []
    hc_expert_counts = []
    
    for log in internal_logs:
        if 'loss' in log and log['loss'] is not None:
            hc_losses.append(log['loss'])
        
        for layer_stats in log.get('layer_stats', {}).values():
            if 'avg_weight_sum' in layer_stats:
                hc_weight_sums.append(layer_stats['avg_weight_sum'])
            if 'avg_experts' in layer_stats:
                hc_expert_counts.append(layer_stats['avg_experts'])
    
    # Extract baseline data if available
    topk_losses = []
    topk_weight_sums = []
    
    if baseline_logs:
        for log in baseline_logs:
            if 'loss' in log and log['loss'] is not None:
                topk_losses.append(log['loss'])
            
            for layer_stats in log.get('layer_stats', {}).values():
                if 'avg_weight_sum' in layer_stats:
                    topk_weight_sums.append(layer_stats['avg_weight_sum'])
    
    # Plot M5: Weight Sum Distribution (CRITICAL)
    if hc_weight_sums:
        fig = plot_weight_sum_distribution(
            hc_weight_sums,
            topk_weight_sums if topk_weight_sums else None,
            output_path=os.path.join(output_dir, 'M5_weight_sum_distribution.png')
        )
        figures['weight_sum'] = fig
        plt.close(fig)
    
    # Plot M1: Per-Token Loss Distribution
    if hc_losses and topk_losses:
        fig = plot_per_token_loss_distribution(
            hc_losses,
            topk_losses,
            output_path=os.path.join(output_dir, 'M1_loss_distribution.png')
        )
        figures['loss_distribution'] = fig
        plt.close(fig)
    
    print(f"✅ Created {len(figures)} diagnostic plots in {output_dir}")
    
    return figures


# =============================================================================
# ORIGINAL VISUALIZATION FUNCTIONS (kept for backwards compatibility)
# =============================================================================

def create_comprehensive_visualization(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (24, 18),
    dpi: int = 300
) -> plt.Figure:
    """
    Create 9-panel comprehensive comparison visualization for HC routing.

    Args:
        results_df: DataFrame with all experiment results
        output_path: Path to save figure (optional)
        figsize: Figure size in inches (width, height)
        dpi: Resolution for saved figure

    Returns:
        matplotlib Figure object

    Examples:
        >>> fig = create_comprehensive_visualization(results_df)
        >>> fig = create_comprehensive_visualization(
        ...     results_df,
        ...     output_path='./visualizations/hc_comprehensive_comparison.png'
        ... )
    """
    # Set style
    sns.set_style('whitegrid')
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10

    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    fig.suptitle(
        'HC Routing Comprehensive Evaluation',
        fontsize=18,
        fontweight='bold',
        y=0.995
    )

    # Filter HC results for analysis
    hc_df = results_df[results_df['routing_type'] == 'hc'].copy()
    baseline_df = results_df[results_df['routing_type'] == 'topk'].copy()

    # Panel 1: Perplexity Comparison
    _plot_perplexity_comparison(axes[0, 0], results_df, baseline_df, hc_df)

    # Panel 2: Task Accuracy Comparison
    _plot_task_accuracy_comparison(axes[0, 1], results_df, baseline_df, hc_df)

    # Panel 3: Expert Efficiency
    _plot_expert_efficiency(axes[0, 2], results_df, baseline_df, hc_df)

    # Panel 4: HC Type Sensitivity
    _plot_hc_variant_sensitivity(axes[1, 0], hc_df)

    # Panel 5: Pareto Frontier
    _plot_pareto_frontier(axes[1, 1], results_df, baseline_df, hc_df)

    # Panel 6: Routing Behavior Summary
    _plot_routing_behavior(axes[1, 2], hc_df)

    # Panel 7: HC Statistics Distribution
    _plot_hc_statistics(axes[2, 0], hc_df)

    # Panel 8: Layer-wise Analysis
    _plot_layer_analysis(axes[2, 1], hc_df)

    # Panel 9: Speed-Quality Trade-off
    _plot_speed_quality_tradeoff(axes[2, 2], results_df, baseline_df, hc_df)

    # Adjust layout
    plt.tight_layout()

    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Saved visualization: {output_path}")

    return fig


# Helper functions for comprehensive visualization (unchanged)
def _plot_perplexity_comparison(ax, results_df, baseline_df, hc_df):
    """Panel 1: Perplexity Comparison."""
    ax.set_title('Perplexity Comparison (Lower is Better)', fontweight='bold')

    wiki_df = results_df[results_df['dataset'] == 'wikitext'].copy()

    if wiki_df.empty or 'perplexity' not in wiki_df.columns:
        ax.text(0.5, 0.5, 'No perplexity data available', ha='center', va='center')
        return

    for max_k in sorted(hc_df['k_or_max_k'].unique() if 'k_or_max_k' in hc_df.columns else []):
        baseline = wiki_df[
            (wiki_df['routing_type'] == 'topk') &
            (wiki_df['k_or_max_k'] == max_k)
        ]['perplexity'].values

        baseline_val = baseline[0] if len(baseline) > 0 else None

        if baseline_val:
            ax.axhline(y=baseline_val, color='red', linestyle='--', alpha=0.5,
                      label=f'TopK-{max_k}' if max_k == 8 else '')

        hc_subset = wiki_df[
            (wiki_df['routing_type'] == 'hc') &
            (wiki_df['k_or_max_k'] == max_k)
        ]

        if not hc_subset.empty and 'hc_variant' in hc_subset.columns:
            for hc_variant in hc_subset['hc_variant'].unique():
                type_data = hc_subset[hc_subset['hc_variant'] == hc_variant]
                ppls = type_data['perplexity'].values
                if len(ppls) > 0:
                    ax.scatter([hc_variant], ppls, s=100, label=f'HC-{hc_variant} max_k={max_k}')

    ax.set_xlabel('HC Type')
    ax.set_ylabel('Perplexity')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)


def _plot_task_accuracy_comparison(ax, results_df, baseline_df, hc_df):
    """Panel 2: Task Accuracy Comparison."""
    ax.set_title('Task Accuracy Comparison (Higher is Better)', fontweight='bold')

    acc_data = []

    for dataset in ['lambada', 'hellaswag']:
        col_name = f'{dataset}_accuracy'
        if col_name in results_df.columns:
            subset = results_df[results_df['dataset'] == dataset].copy()
            subset['task'] = dataset
            subset['accuracy'] = subset[col_name]
            acc_data.append(subset[['routing_type', 'k_or_max_k', 'hc_variant', 'task', 'accuracy']])

    if not acc_data:
        ax.text(0.5, 0.5, 'No task accuracy data available', ha='center', va='center')
        return

    combined = pd.concat(acc_data, ignore_index=True)

    for max_k in sorted(combined['k_or_max_k'].unique()):
        baseline = combined[
            (combined['routing_type'] == 'topk') &
            (combined['k_or_max_k'] == max_k)
        ]['accuracy'].mean()

        if not np.isnan(baseline):
            ax.axhline(y=baseline, color='red', linestyle='--', alpha=0.5,
                      label=f'TopK-{max_k}' if max_k == 8 else '')

        hc_subset = combined[
            (combined['routing_type'] == 'hc') &
            (combined['k_or_max_k'] == max_k)
        ]

        if not hc_subset.empty and 'hc_variant' in hc_subset.columns:
            for hc_variant in hc_subset['hc_variant'].unique():
                type_acc = hc_subset[hc_subset['hc_variant'] == hc_variant]['accuracy'].mean()
                ax.scatter([hc_variant], [type_acc], s=100, marker='s',
                          label=f'HC-{hc_variant} max_k={max_k}')

    ax.set_xlabel('HC Type')
    ax.set_ylabel('Average Accuracy')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)


def _plot_expert_efficiency(ax, results_df, baseline_df, hc_df):
    """Panel 3: Expert Efficiency."""
    ax.set_title('Expert Efficiency (Lower = More Efficient)', fontweight='bold')

    if 'avg_experts' not in results_df.columns:
        ax.text(0.5, 0.5, 'No expert count data available', ha='center', va='center')
        return

    for max_k in sorted(hc_df['k_or_max_k'].unique() if 'k_or_max_k' in hc_df.columns else []):
        ax.axhline(y=max_k, color='red', linestyle='--', alpha=0.5,
                  label=f'TopK-{max_k}' if max_k == 8 else '')

        hc_subset = hc_df[hc_df['k_or_max_k'] == max_k]

        if not hc_subset.empty and 'hc_variant' in hc_subset.columns:
            for hc_variant in hc_subset['hc_variant'].unique():
                type_data = hc_subset[hc_subset['hc_variant'] == hc_variant]
                avg_experts = type_data['avg_experts'].mean()
                reduction = (max_k - avg_experts) / max_k * 100
                ax.scatter([hc_variant], [avg_experts], s=100, marker='^',
                          label=f'HC-{hc_variant} max_k={max_k} ({reduction:.1f}% reduction)')

    ax.set_xlabel('HC Type')
    ax.set_ylabel('Average Experts Selected')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)


def _plot_hc_variant_sensitivity(ax, hc_df):
    """Panel 4: HC Type Sensitivity."""
    ax.set_title('HC Type Sensitivity (Avg Experts)', fontweight='bold')

    if hc_df.empty or 'avg_experts' not in hc_df.columns:
        ax.text(0.5, 0.5, 'No HC data available', ha='center', va='center')
        return

    if 'hc_variant' not in hc_df.columns:
        ax.text(0.5, 0.5, 'No hc_variant data available', ha='center', va='center')
        return

    grouped = hc_df.groupby(['hc_variant', 'k_or_max_k'])['avg_experts'].mean().unstack()

    if grouped.empty:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        return

    grouped.plot(kind='bar', ax=ax, alpha=0.7)
    ax.set_xlabel('HC Type')
    ax.set_ylabel('Average Experts Selected')
    ax.legend(title='max_k', fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    plt.sca(ax)
    plt.xticks(rotation=0)


def _plot_pareto_frontier(ax, results_df, baseline_df, hc_df):
    """Panel 5: Pareto Frontier."""
    ax.set_title('Pareto Frontier: Efficiency vs Quality', fontweight='bold')

    plot_data = results_df[
        (results_df['dataset'] == 'wikitext') &
        ('perplexity' in results_df.columns) &
        ('avg_experts' in results_df.columns)
    ].copy()

    if plot_data.empty:
        ax.text(0.5, 0.5, 'Insufficient data for Pareto frontier', ha='center', va='center')
        return

    baseline_data = plot_data[plot_data['routing_type'] == 'topk']
    ax.scatter(baseline_data['avg_experts'], baseline_data['perplexity'],
              s=200, marker='*', color='red', label='Baseline TopK',
              edgecolors='black', linewidths=2, zorder=10)

    hc_data = plot_data[plot_data['routing_type'] == 'hc']
    if not hc_data.empty and 'hc_variant' in hc_data.columns:
        for hc_variant in hc_data['hc_variant'].unique():
            type_data = hc_data[hc_data['hc_variant'] == hc_variant]
            ax.scatter(type_data['avg_experts'], type_data['perplexity'],
                      s=100, alpha=0.7, label=f'HC-{hc_variant}',
                      edgecolors='black', linewidths=0.5)

    ax.set_xlabel('Average Experts (Lower = More Efficient)')
    ax.set_ylabel('Perplexity (Lower = Better Quality)')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)


def _plot_routing_behavior(ax, hc_df):
    """Panel 6: Routing Behavior Summary."""
    ax.set_title('Routing Behavior: Constraint Distribution', fontweight='bold')

    if hc_df.empty:
        ax.text(0.5, 0.5, 'No HC data available', ha='center', va='center')
        return

    required_cols = ['ceiling_hit_rate', 'floor_hit_rate', 'mid_range_rate']
    if not all(col in hc_df.columns for col in required_cols):
        ax.text(0.5, 0.5, 'Missing routing behavior metrics', ha='center', va='center')
        return

    if 'hc_variant' in hc_df.columns and 'k_or_max_k' in hc_df.columns:
        grouped = hc_df.groupby(['k_or_max_k', 'hc_variant'])[required_cols].mean()
    else:
        grouped = hc_df[required_cols].mean().to_frame().T

    if grouped.empty:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        return

    if isinstance(grouped.index, pd.MultiIndex):
        configs = [f"k{row[0]}_{row[1]}" for row in grouped.index]
    else:
        configs = ['HC']

    floor = grouped['floor_hit_rate'].values
    mid = grouped['mid_range_rate'].values
    ceiling = grouped['ceiling_hit_rate'].values

    x = np.arange(len(configs))
    ax.bar(x, floor, label='Floor Hits (min_k)', color='red', alpha=0.7)
    ax.bar(x, mid, bottom=floor, label='Mid-Range (Adaptive)', color='green', alpha=0.7)
    ax.bar(x, ceiling, bottom=floor + mid, label='Ceiling Hits (max_k)', color='blue', alpha=0.7)

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Percentage (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, axis='y', alpha=0.3)


def _plot_hc_statistics(ax, hc_df):
    """Panel 7: HC Statistics Distribution."""
    ax.set_title('HC Statistics: Threshold Ranks & Max Values', fontweight='bold')

    if hc_df.empty:
        ax.text(0.5, 0.5, 'No HC data available', ha='center', va='center')
        return

    has_threshold = 'avg_hc_threshold' in hc_df.columns
    has_max_value = 'avg_hc_max_value' in hc_df.columns

    if not has_threshold and not has_max_value:
        ax.text(0.5, 0.5, 'No HC statistics available', ha='center', va='center')
        return

    ax2 = ax.twinx()

    x_labels = []
    threshold_vals = []
    max_vals = []

    if 'hc_variant' in hc_df.columns and 'k_or_max_k' in hc_df.columns:
        grouped = hc_df.groupby(['k_or_max_k', 'hc_variant'])
        for (max_k, hc_variant), group in grouped:
            x_labels.append(f"k{max_k}_{hc_variant}")
            if has_threshold:
                threshold_vals.append(group['avg_hc_threshold'].mean())
            if has_max_value:
                max_vals.append(group['avg_hc_max_value'].mean())
    else:
        x_labels = ['HC']
        if has_threshold:
            threshold_vals = [hc_df['avg_hc_threshold'].mean()]
        if has_max_value:
            max_vals = [hc_df['avg_hc_max_value'].mean()]

    x = np.arange(len(x_labels))

    if threshold_vals:
        ax.bar(x - 0.2, threshold_vals, width=0.4, label='Avg HC Threshold Rank',
               color='steelblue', alpha=0.7)
        ax.set_ylabel('HC Threshold Rank', color='steelblue')

    if max_vals:
        ax2.bar(x + 0.2, max_vals, width=0.4, label='Avg HC Max Value',
                color='orange', alpha=0.7)
        ax2.set_ylabel('HC Max Value', color='orange')

    ax.set_xlabel('Configuration')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='best')


def _plot_layer_analysis(ax, hc_df):
    """Panel 8: Layer-wise Analysis."""
    ax.set_title('Layer-wise Expert Selection Variance', fontweight='bold')

    if 'layer_expert_variance' not in hc_df.columns:
        ax.text(0.5, 0.5, 'No layer-wise data available', ha='center', va='center')
        return

    if 'hc_variant' in hc_df.columns:
        hc_df.boxplot(column='layer_expert_variance', by='hc_variant', ax=ax)
        ax.set_xlabel('HC Type')
    elif 'k_or_max_k' in hc_df.columns:
        hc_df.boxplot(column='layer_expert_variance', by='k_or_max_k', ax=ax)
        ax.set_xlabel('max_k')
    else:
        ax.hist(hc_df['layer_expert_variance'], bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Layer Expert Variance')

    ax.set_ylabel('Layer Expert Variance')
    ax.set_title('Layer-wise Expert Variance', fontsize=10)
    plt.sca(ax)
    plt.xticks(rotation=0)


def _plot_speed_quality_tradeoff(ax, results_df, baseline_df, hc_df):
    """Panel 9: Speed-Quality Trade-off."""
    ax.set_title('Speed vs Quality Trade-off', fontweight='bold')

    plot_data = results_df[
        ('tokens_per_second' in results_df.columns) &
        ('perplexity' in results_df.columns) &
        (results_df['dataset'] == 'wikitext')
    ].copy()

    if plot_data.empty:
        ax.text(0.5, 0.5, 'Insufficient speed/quality data', ha='center', va='center')
        return

    baseline_data = plot_data[plot_data['routing_type'] == 'topk']
    if not baseline_data.empty:
        sizes = baseline_data['k_or_max_k'] * 20 if 'k_or_max_k' in baseline_data.columns else 100
        ax.scatter(baseline_data['tokens_per_second'], baseline_data['perplexity'],
                  s=sizes, marker='*', color='red', label='Baseline TopK',
                  edgecolors='black', linewidths=2, alpha=0.8, zorder=10)

    hc_data = plot_data[plot_data['routing_type'] == 'hc']
    if not hc_data.empty:
        sizes = hc_data['avg_experts'] * 20 if 'avg_experts' in hc_data.columns else 100

        if 'hc_variant' in hc_data.columns:
            for hc_variant in hc_data['hc_variant'].unique():
                type_data = hc_data[hc_data['hc_variant'] == hc_variant]
                ax.scatter(type_data['tokens_per_second'], type_data['perplexity'],
                          s=sizes, alpha=0.6, label=f'HC-{hc_variant}',
                          edgecolors='black', linewidths=0.5)
        else:
            ax.scatter(hc_data['tokens_per_second'], hc_data['perplexity'],
                      s=sizes, alpha=0.6, label='HC',
                      edgecolors='black', linewidths=0.5)

    ax.set_xlabel('Tokens/Second (Higher = Faster)')
    ax.set_ylabel('Perplexity (Lower = Better)')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)


def plot_hc_statistic_vs_rank(
    p_values_sorted: np.ndarray,
    hc_stats: np.ndarray,
    threshold_rank: int,
    num_selected: int,
    output_path: Optional[str] = None,
    title: str = "HC Statistic vs Rank"
) -> plt.Figure:
    """
    Plot HC statistic curve showing where threshold is found.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    n = len(p_values_sorted)
    ranks = np.arange(1, n + 1)
    expected = ranks / n

    ax1.plot(ranks, expected, 'k--', label='Expected (Uniform)', linewidth=2, alpha=0.7)
    ax1.plot(ranks, p_values_sorted, 'b-', label='Actual P-values', linewidth=1.5)
    ax1.axvline(x=threshold_rank+1, color='red', linestyle='--', linewidth=2,
                label=f'Threshold (rank {threshold_rank+1})')
    ax1.fill_between(ranks[:num_selected], 0, 1, alpha=0.2, color='green',
                     label=f'Selected ({num_selected} experts)')
    ax1.set_ylabel('P-value')
    ax1.set_title(title, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, min(n, 30))

    ax2.plot(ranks, hc_stats, 'g-', linewidth=2, label='HC Statistic')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.axvline(x=threshold_rank+1, color='red', linestyle='--', linewidth=2,
                label=f'Max HC at rank {threshold_rank+1}')
    ax2.scatter([threshold_rank+1], [hc_stats[threshold_rank]], s=200, c='red',
               marker='*', zorder=10, edgecolors='black', linewidths=2)
    ax2.set_xlabel('Rank')
    ax2.set_ylabel('HC Statistic')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, min(n, 30))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")

    return fig
