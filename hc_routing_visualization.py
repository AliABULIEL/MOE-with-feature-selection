"""
HC Routing Visualization Suite
===============================

This module implements comprehensive 9-panel visualizations for HC routing
evaluation, tailored for Higher Criticism analysis.

Panels:
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
    from hc_routing_visualization import create_comprehensive_visualization

    fig = create_comprehensive_visualization(
        results_df,
        output_path='hc_comprehensive_comparison.png'
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple, List
import warnings

warnings.filterwarnings('ignore')


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


def _plot_perplexity_comparison(ax, results_df, baseline_df, hc_df):
    """Panel 1: Perplexity Comparison."""
    ax.set_title('Perplexity Comparison (Lower is Better)', fontweight='bold')

    # Filter WikiText results only
    wiki_df = results_df[results_df['dataset'] == 'wikitext'].copy()

    if wiki_df.empty or 'perplexity' not in wiki_df.columns:
        ax.text(0.5, 0.5, 'No perplexity data available', ha='center', va='center')
        return

    # Group by max_k and hc_variant
    for max_k in sorted(hc_df['k_or_max_k'].unique() if 'k_or_max_k' in hc_df.columns else []):
        # Baseline for this max_k
        baseline = wiki_df[
            (wiki_df['routing_type'] == 'topk') &
            (wiki_df['k_or_max_k'] == max_k)
        ]['perplexity'].values

        baseline_val = baseline[0] if len(baseline) > 0 else None

        if baseline_val:
            ax.axhline(y=baseline_val, color='red', linestyle='--', alpha=0.5,
                      label=f'TopK-{max_k}' if max_k == 8 else '')

        # HC variants for this max_k
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

    # Aggregate LAMBADA and HellaSwag accuracies
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

    # Plot by max_k
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
    """Panel 3: Expert Efficiency (Avg Experts Selected)."""
    ax.set_title('Expert Efficiency (Lower = More Efficient)', fontweight='bold')

    if 'avg_experts' not in results_df.columns:
        ax.text(0.5, 0.5, 'No expert count data available', ha='center', va='center')
        return

    # Plot by max_k
    for max_k in sorted(hc_df['k_or_max_k'].unique() if 'k_or_max_k' in hc_df.columns else []):
        # Baseline
        ax.axhline(y=max_k, color='red', linestyle='--', alpha=0.5,
                  label=f'TopK-{max_k}' if max_k == 8 else '')

        # HC variants
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

    # Check if hc_variant column exists
    if 'hc_variant' not in hc_df.columns:
        ax.text(0.5, 0.5, 'No hc_variant data available', ha='center', va='center')
        return

    # Group by hc_variant and max_k
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
    """Panel 5: Pareto Frontier (Efficiency vs Quality)."""
    ax.set_title('Pareto Frontier: Efficiency vs Quality', fontweight='bold')

    # Use perplexity as quality metric
    plot_data = results_df[
        (results_df['dataset'] == 'wikitext') &
        ('perplexity' in results_df.columns) &
        ('avg_experts' in results_df.columns)
    ].copy()

    if plot_data.empty:
        ax.text(0.5, 0.5, 'Insufficient data for Pareto frontier', ha='center', va='center')
        return

    # Scatter baseline
    baseline_data = plot_data[plot_data['routing_type'] == 'topk']
    ax.scatter(baseline_data['avg_experts'], baseline_data['perplexity'],
              s=200, marker='*', color='red', label='Baseline TopK',
              edgecolors='black', linewidths=2, zorder=10)

    # Scatter HC by type
    hc_data = plot_data[plot_data['routing_type'] == 'hc']
    if not hc_data.empty and 'hc_variant' in hc_data.columns:
        for hc_variant in hc_data['hc_variant'].unique():
            type_data = hc_data[hc_data['hc_variant'] == hc_variant]
            ax.scatter(type_data['avg_experts'], type_data['perplexity'],
                      s=100, alpha=0.7, label=f'HC-{hc_variant}',
                      edgecolors='black', linewidths=0.5)

        # Identify Pareto-optimal points
        pareto_points = _find_pareto_frontier(
            hc_data['avg_experts'].values,
            hc_data['perplexity'].values
        )

        if len(pareto_points) > 0:
            pareto_experts = hc_data['avg_experts'].values[pareto_points]
            pareto_ppl = hc_data['perplexity'].values[pareto_points]
            sorted_idx = np.argsort(pareto_experts)
            ax.plot(pareto_experts[sorted_idx], pareto_ppl[sorted_idx],
                   'g--', linewidth=2, label='Pareto Frontier')

    ax.set_xlabel('Average Experts (Lower = More Efficient)')
    ax.set_ylabel('Perplexity (Lower = Better Quality)')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)


def _find_pareto_frontier(x: np.ndarray, y: np.ndarray) -> List[int]:
    """
    Find Pareto-optimal points where both x and y are minimized.

    Returns indices of Pareto-optimal points.
    """
    n = len(x)
    pareto_indices = []

    for i in range(n):
        is_pareto = True
        for j in range(n):
            if i != j:
                # Point j dominates point i if both x and y are better
                if x[j] <= x[i] and y[j] <= y[i] and (x[j] < x[i] or y[j] < y[i]):
                    is_pareto = False
                    break
        if is_pareto:
            pareto_indices.append(i)

    return pareto_indices


def _plot_routing_behavior(ax, hc_df):
    """Panel 6: Routing Behavior Summary (Floor/Mid/Ceiling)."""
    ax.set_title('Routing Behavior: Constraint Distribution', fontweight='bold')

    if hc_df.empty:
        ax.text(0.5, 0.5, 'No HC data available', ha='center', va='center')
        return

    # Check if we have the metrics
    required_cols = ['ceiling_hit_rate', 'floor_hit_rate', 'mid_range_rate']
    if not all(col in hc_df.columns for col in required_cols):
        ax.text(0.5, 0.5, 'Missing routing behavior metrics', ha='center', va='center')
        return

    # Group by config
    if 'hc_variant' in hc_df.columns and 'k_or_max_k' in hc_df.columns:
        grouped = hc_df.groupby(['k_or_max_k', 'hc_variant'])[required_cols].mean()
    else:
        grouped = hc_df[required_cols].mean().to_frame().T

    if grouped.empty:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        return

    # Create stacked bar chart
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

    # Check for HC-specific metrics
    has_threshold = 'avg_hc_threshold' in hc_df.columns
    has_max_value = 'avg_hc_max_value' in hc_df.columns

    if not has_threshold and not has_max_value:
        ax.text(0.5, 0.5, 'No HC statistics available', ha='center', va='center')
        return

    # Create dual y-axis plot
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

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='best')


def _plot_layer_analysis(ax, hc_df):
    """Panel 8: Layer-wise Analysis."""
    ax.set_title('Layer-wise Expert Selection Variance', fontweight='bold')

    if 'layer_expert_variance' not in hc_df.columns:
        ax.text(0.5, 0.5, 'No layer-wise data available', ha='center', va='center')
        return

    # Box plot by max_k or hc_variant
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

    # Scatter baseline
    baseline_data = plot_data[plot_data['routing_type'] == 'topk']
    if not baseline_data.empty:
        sizes = baseline_data['k_or_max_k'] * 20 if 'k_or_max_k' in baseline_data.columns else 100
        ax.scatter(baseline_data['tokens_per_second'], baseline_data['perplexity'],
                  s=sizes, marker='*', color='red', label='Baseline TopK',
                  edgecolors='black', linewidths=2, alpha=0.8, zorder=10)

    # Scatter HC
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


# Individual panel functions for custom use
def create_single_panel(
    results_df: pd.DataFrame,
    panel_name: str,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Create a single visualization panel.

    Args:
        results_df: DataFrame with experiment results
        panel_name: Panel to create (e.g., 'perplexity', 'pareto', 'hc_stats', etc.)
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object

    Examples:
        >>> fig = create_single_panel(results_df, 'pareto')
        >>> fig = create_single_panel(results_df, 'hc_stats', 'hc_statistics.png')
    """
    fig, ax = plt.subplots(figsize=figsize)

    hc_df = results_df[results_df['routing_type'] == 'hc']
    baseline_df = results_df[results_df['routing_type'] == 'topk']

    panel_functions = {
        'perplexity': _plot_perplexity_comparison,
        'accuracy': _plot_task_accuracy_comparison,
        'efficiency': _plot_expert_efficiency,
        'hc_variant_sensitivity': _plot_hc_variant_sensitivity,
        'pareto': _plot_pareto_frontier,
        'behavior': _plot_routing_behavior,
        'hc_stats': _plot_hc_statistics,
        'layer': _plot_layer_analysis,
        'speed_quality': _plot_speed_quality_tradeoff
    }

    if panel_name not in panel_functions:
        raise ValueError(f"Unknown panel: {panel_name}. Choose from {list(panel_functions.keys())}")

    # Call the appropriate panel function
    if panel_name in ['hc_variant_sensitivity', 'behavior', 'hc_stats', 'layer']:
        panel_functions[panel_name](ax, hc_df)
    else:
        panel_functions[panel_name](ax, results_df, baseline_df, hc_df)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")

    return fig


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

    This is the fundamental HC visualization showing:
    - HC statistic at each rank
    - Expected p-value line (uniform distribution)
    - Vertical line at threshold rank
    - Shaded region for selected experts

    Args:
        p_values_sorted: Sorted p-values [num_experts]
        hc_stats: HC statistics [num_experts]
        threshold_rank: Rank where HC peaks (0-indexed)
        num_selected: Number of experts selected
        output_path: Optional save path
        title: Plot title

    Returns:
        matplotlib Figure

    Examples:
        >>> fig = plot_hc_statistic_vs_rank(p_vals, hc_stats, 5, 6)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    n = len(p_values_sorted)
    ranks = np.arange(1, n + 1)
    expected = ranks / n

    # Top panel: P-values vs expected
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
    ax1.set_xlim(0, min(n, 30))  # Focus on first 30 ranks

    # Bottom panel: HC statistic
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
