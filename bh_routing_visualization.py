"""
BH Routing Visualization Suite
===============================

This module implements comprehensive 9-panel visualizations for BH routing
evaluation, aligned with the template's visualization patterns.

Panels:
1. Perplexity Comparison (baseline vs BH)
2. Task Accuracy Comparison
3. Expert Efficiency (avg experts selected)
4. Alpha Sensitivity Heatmap
5. Pareto Frontier (efficiency vs quality)
6. Routing Behavior Summary (floor/mid/ceiling distribution)
7. Expert Utilization
8. Layer-wise Analysis
9. Speed-Quality Trade-off

Usage:
    from bh_routing_visualization import create_comprehensive_visualization

    fig = create_comprehensive_visualization(
        results_df,
        output_path='bh_comprehensive_comparison.png'
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
    Create 9-panel comprehensive comparison visualization.

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
        ...     output_path='./visualizations/bh_comprehensive_comparison.png'
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
        'BH Routing Comprehensive Evaluation',
        fontsize=18,
        fontweight='bold',
        y=0.995
    )

    # Filter BH results for analysis
    bh_df = results_df[results_df['routing_type'] == 'bh'].copy()
    baseline_df = results_df[results_df['routing_type'] == 'topk'].copy()

    # Panel 1: Perplexity Comparison
    _plot_perplexity_comparison(axes[0, 0], results_df, baseline_df, bh_df)

    # Panel 2: Task Accuracy Comparison
    _plot_task_accuracy_comparison(axes[0, 1], results_df, baseline_df, bh_df)

    # Panel 3: Expert Efficiency
    _plot_expert_efficiency(axes[0, 2], results_df, baseline_df, bh_df)

    # Panel 4: Alpha Sensitivity Heatmap
    _plot_alpha_sensitivity(axes[1, 0], bh_df)

    # Panel 5: Pareto Frontier
    _plot_pareto_frontier(axes[1, 1], results_df, baseline_df, bh_df)

    # Panel 6: Routing Behavior Summary
    _plot_routing_behavior(axes[1, 2], bh_df)

    # Panel 7: Expert Utilization
    _plot_expert_utilization(axes[2, 0], results_df, baseline_df, bh_df)

    # Panel 8: Layer-wise Analysis
    _plot_layer_analysis(axes[2, 1], bh_df)

    # Panel 9: Speed-Quality Trade-off
    _plot_speed_quality_tradeoff(axes[2, 2], results_df, baseline_df, bh_df)

    # Adjust layout
    plt.tight_layout()

    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Saved visualization: {output_path}")

    return fig


def _plot_perplexity_comparison(ax, results_df, baseline_df, bh_df):
    """Panel 1: Perplexity Comparison."""
    ax.set_title('Perplexity Comparison (Lower is Better)', fontweight='bold')

    # Filter WikiText results only
    wiki_df = results_df[results_df['dataset'] == 'wikitext'].copy()

    if wiki_df.empty or 'perplexity' not in wiki_df.columns:
        ax.text(0.5, 0.5, 'No perplexity data available', ha='center', va='center')
        return

    # Group by max_k
    for max_k in sorted(bh_df['k_or_max_k'].unique()):
        # Baseline for this max_k
        baseline = wiki_df[
            (wiki_df['routing_type'] == 'topk') &
            (wiki_df['k_or_max_k'] == max_k)
        ]['perplexity'].values

        baseline_val = baseline[0] if len(baseline) > 0 else None

        # BH variants for this max_k
        bh_subset = wiki_df[
            (wiki_df['routing_type'] == 'bh') &
            (wiki_df['k_or_max_k'] == max_k)
        ]

        if baseline_val:
            ax.axhline(y=baseline_val, color='red', linestyle='--', alpha=0.5,
                      label=f'TopK-{max_k}' if max_k == 8 else '')

        if not bh_subset.empty:
            alphas = bh_subset['alpha'].values
            ppls = bh_subset['perplexity'].values
            ax.plot(alphas, ppls, marker='o', label=f'BH max_k={max_k}')

    ax.set_xlabel('Alpha (FDR level)')
    ax.set_ylabel('Perplexity')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)


def _plot_task_accuracy_comparison(ax, results_df, baseline_df, bh_df):
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
            acc_data.append(subset[['routing_type', 'k_or_max_k', 'alpha', 'task', 'accuracy']])

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

        bh_subset = combined[
            (combined['routing_type'] == 'bh') &
            (combined['k_or_max_k'] == max_k)
        ].groupby('alpha')['accuracy'].mean()

        if not np.isnan(baseline):
            ax.axhline(y=baseline, color='red', linestyle='--', alpha=0.5,
                      label=f'TopK-{max_k}' if max_k == 8 else '')

        if not bh_subset.empty:
            ax.plot(bh_subset.index, bh_subset.values, marker='s',
                   label=f'BH max_k={max_k}')

    ax.set_xlabel('Alpha')
    ax.set_ylabel('Average Accuracy')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)


def _plot_expert_efficiency(ax, results_df, baseline_df, bh_df):
    """Panel 3: Expert Efficiency (Avg Experts Selected)."""
    ax.set_title('Expert Efficiency (Lower = More Efficient)', fontweight='bold')

    if 'avg_experts' not in results_df.columns:
        ax.text(0.5, 0.5, 'No expert count data available', ha='center', va='center')
        return

    # Plot by max_k
    for max_k in sorted(bh_df['k_or_max_k'].unique()):
        # Baseline
        ax.axhline(y=max_k, color='red', linestyle='--', alpha=0.5,
                  label=f'TopK-{max_k}' if max_k == 8 else '')

        # BH variants
        bh_subset = bh_df[bh_df['k_or_max_k'] == max_k].groupby('alpha')['avg_experts'].mean()

        if not bh_subset.empty:
            reduction = (max_k - bh_subset.values) / max_k * 100
            ax.plot(bh_subset.index, bh_subset.values, marker='^',
                   label=f'BH max_k={max_k} ({reduction.mean():.1f}% reduction)')

    ax.set_xlabel('Alpha')
    ax.set_ylabel('Average Experts Selected')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)


def _plot_alpha_sensitivity(ax, bh_df):
    """Panel 4: Alpha Sensitivity Heatmap."""
    ax.set_title('Alpha Sensitivity (Avg Experts)', fontweight='bold')

    if 'avg_experts' not in bh_df.columns or bh_df.empty:
        ax.text(0.5, 0.5, 'No BH data available', ha='center', va='center')
        return

    # Create pivot table
    pivot = bh_df.groupby(['k_or_max_k', 'alpha'])['avg_experts'].mean().unstack()

    if pivot.empty:
        ax.text(0.5, 0.5, 'Insufficient data for heatmap', ha='center', va='center')
        return

    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax,
                cbar_kws={'label': 'Avg Experts'}, vmin=1, vmax=pivot.max().max())
    ax.set_xlabel('Alpha')
    ax.set_ylabel('max_k')


def _plot_pareto_frontier(ax, results_df, baseline_df, bh_df):
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

    # Scatter BH
    bh_data = plot_data[plot_data['routing_type'] == 'bh']
    scatter = ax.scatter(bh_data['avg_experts'], bh_data['perplexity'],
                        c=bh_data['alpha'], cmap='viridis', s=100,
                        alpha=0.7, edgecolors='black', linewidths=0.5)

    # Identify Pareto-optimal points
    pareto_points = _find_pareto_frontier(
        bh_data['avg_experts'].values,
        bh_data['perplexity'].values
    )

    if len(pareto_points) > 0:
        pareto_experts = bh_data['avg_experts'].values[pareto_points]
        pareto_ppl = bh_data['perplexity'].values[pareto_points]
        sorted_idx = np.argsort(pareto_experts)
        ax.plot(pareto_experts[sorted_idx], pareto_ppl[sorted_idx],
               'g--', linewidth=2, label='Pareto Frontier')

    ax.set_xlabel('Average Experts (Lower = More Efficient)')
    ax.set_ylabel('Perplexity (Lower = Better Quality)')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Alpha', fontsize=8)


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


def _plot_routing_behavior(ax, bh_df):
    """Panel 6: Routing Behavior Summary (Floor/Mid/Ceiling)."""
    ax.set_title('Routing Behavior: Constraint Distribution', fontweight='bold')

    if bh_df.empty:
        ax.text(0.5, 0.5, 'No BH data available', ha='center', va='center')
        return

    # Check if we have the metrics
    required_cols = ['ceiling_hit_rate', 'floor_hit_rate', 'mid_range_rate']
    if not all(col in bh_df.columns for col in required_cols):
        ax.text(0.5, 0.5, 'Missing routing behavior metrics', ha='center', va='center')
        return

    # Group by config
    grouped = bh_df.groupby(['k_or_max_k', 'alpha'])[required_cols].mean()

    if grouped.empty:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        return

    # Create stacked bar chart
    configs = [f"k{row[0]}_a{row[1]:.2f}" for row in grouped.index]
    floor = grouped['floor_hit_rate'].values
    mid = grouped['mid_range_rate'].values
    ceiling = grouped['ceiling_hit_rate'].values

    x = np.arange(len(configs))
    ax.bar(x, floor, label='Floor Hits (min_k)', color='red', alpha=0.7)
    ax.bar(x, mid, bottom=floor, label='Mid-Range', color='gray', alpha=0.7)
    ax.bar(x, ceiling, bottom=floor + mid, label='Ceiling Hits (max_k)', color='blue', alpha=0.7)

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Percentage (%)')
    ax.set_xticks(x[::max(1, len(x)//10)])  # Show every 10th label
    ax.set_xticklabels(configs[::max(1, len(x)//10)], rotation=45, ha='right', fontsize=7)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, axis='y', alpha=0.3)


def _plot_expert_utilization(ax, results_df, baseline_df, bh_df):
    """Panel 7: Expert Utilization."""
    ax.set_title('Expert Utilization (Higher = Better Load Balance)', fontweight='bold')

    if 'expert_utilization' not in results_df.columns:
        ax.text(0.5, 0.5, 'No expert utilization data available', ha='center', va='center')
        return

    # Compare baseline vs BH
    width = 0.35
    labels = []
    baseline_vals = []
    bh_vals = []

    for max_k in sorted(bh_df['k_or_max_k'].unique()):
        baseline_util = baseline_df[
            baseline_df['k_or_max_k'] == max_k
        ]['expert_utilization'].mean()

        bh_util = bh_df[
            bh_df['k_or_max_k'] == max_k
        ]['expert_utilization'].mean()

        if not np.isnan(baseline_util):
            labels.append(f'max_k={max_k}')
            baseline_vals.append(baseline_util)
            bh_vals.append(bh_util)

    x = np.arange(len(labels))
    ax.bar(x - width/2, baseline_vals, width, label='TopK Baseline', color='red', alpha=0.7)
    ax.bar(x + width/2, bh_vals, width, label='BH (avg)', color='green', alpha=0.7)

    ax.set_ylabel('Expert Utilization')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, axis='y', alpha=0.3)


def _plot_layer_analysis(ax, bh_df):
    """Panel 8: Layer-wise Analysis."""
    ax.set_title('Layer-wise Expert Selection (BH Routing)', fontweight='bold')

    if 'layer_expert_variance' not in bh_df.columns:
        ax.text(0.5, 0.5, 'No layer-wise data available', ha='center', va='center')
        return

    # Plot variance across layers for different configs
    sample_configs = bh_df.groupby(['k_or_max_k', 'alpha'])['layer_expert_variance'].mean().head(5)

    if sample_configs.empty:
        ax.text(0.5, 0.5, 'Insufficient layer-wise data', ha='center', va='center')
        return

    for (max_k, alpha), variance in sample_configs.items():
        # Placeholder: would need actual per-layer data from internal logs
        # For now, show variance as bar chart
        pass

    # Simplified: show layer variance distribution
    bh_df.boxplot(column='layer_expert_variance', by=['k_or_max_k'], ax=ax)
    ax.set_xlabel('max_k')
    ax.set_ylabel('Layer Expert Variance')
    ax.set_title('Layer-wise Expert Variance by max_k', fontsize=10)
    plt.sca(ax)
    plt.xticks(rotation=0)


def _plot_speed_quality_tradeoff(ax, results_df, baseline_df, bh_df):
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
        sizes = baseline_data['k_or_max_k'] * 20
        ax.scatter(baseline_data['tokens_per_second'], baseline_data['perplexity'],
                  s=sizes, marker='*', color='red', label='Baseline TopK',
                  edgecolors='black', linewidths=2, alpha=0.8, zorder=10)

    # Scatter BH
    bh_data = plot_data[plot_data['routing_type'] == 'bh']
    if not bh_data.empty:
        sizes = bh_data['avg_experts'] * 20 if 'avg_experts' in bh_data.columns else 100
        scatter = ax.scatter(bh_data['tokens_per_second'], bh_data['perplexity'],
                            c=bh_data['alpha'], s=sizes, cmap='viridis',
                            alpha=0.6, edgecolors='black', linewidths=0.5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Alpha', fontsize=8)

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
        panel_name: Panel to create (e.g., 'perplexity', 'pareto', etc.)
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object

    Examples:
        >>> fig = create_single_panel(results_df, 'pareto')
        >>> fig = create_single_panel(results_df, 'perplexity', 'ppl_comparison.png')
    """
    fig, ax = plt.subplots(figsize=figsize)

    bh_df = results_df[results_df['routing_type'] == 'bh']
    baseline_df = results_df[results_df['routing_type'] == 'topk']

    panel_functions = {
        'perplexity': _plot_perplexity_comparison,
        'accuracy': _plot_task_accuracy_comparison,
        'efficiency': _plot_expert_efficiency,
        'alpha_sensitivity': _plot_alpha_sensitivity,
        'pareto': _plot_pareto_frontier,
        'behavior': _plot_routing_behavior,
        'utilization': _plot_expert_utilization,
        'layer': _plot_layer_analysis,
        'speed_quality': _plot_speed_quality_tradeoff
    }

    if panel_name not in panel_functions:
        raise ValueError(f"Unknown panel: {panel_name}. Choose from {list(panel_functions.keys())}")

    # Call the appropriate panel function
    if panel_name == 'alpha_sensitivity':
        panel_functions[panel_name](ax, bh_df)
    else:
        panel_functions[panel_name](ax, results_df, baseline_df, bh_df)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")

    return fig
