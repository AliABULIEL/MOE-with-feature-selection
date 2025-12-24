"""
Comprehensive Visualization for BH & HC Routing
================================================

Creates publication-quality visualizations for all metrics.
Handles missing data gracefully with informative placeholders.

Usage:
    from moe_visualization import (
        create_comprehensive_dashboard,
        plot_per_layer_routing,
        plot_expert_usage_heatmap,
        plot_bh_vs_hc_comparison
    )
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any


def create_comprehensive_dashboard(
    results_df: pd.DataFrame,
    output_path: str,
    title: str = "MoE Routing Analysis Dashboard",
    routing_method: str = "BH/HC"
) -> str:
    """
    Create comprehensive 16-panel dashboard visualization.

    Handles NaN values gracefully with informative messages.
    Works for both BH and HC routing results.

    Args:
        results_df: DataFrame with experiment results
        output_path: Path to save figure
        title: Dashboard title
        routing_method: 'BH', 'HC', or 'BH/HC'

    Returns:
        Path to saved figure
    """

    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(28, 24))
    fig.suptitle(f"{title} ({routing_method})", fontsize=18, fontweight='bold', y=0.98)

    df = results_df.copy()

    # Identify routing types
    if 'routing_type' in df.columns:
        adaptive_df = df[df['routing_type'].isin(['bh', 'hc'])].copy()
        baseline_df = df[df['routing_type'] == 'topk'].copy()
    else:
        adaptive_df = df.copy()
        baseline_df = pd.DataFrame()

    # Color scheme
    colors = {
        'bh': '#3498db',      # Blue
        'hc': '#9b59b6',      # Purple
        'topk': '#95a5a6',    # Gray
        'good': '#2ecc71',    # Green
        'bad': '#e74c3c',     # Red
        'neutral': '#f39c12'  # Orange
    }

    # =========================================================================
    # ROW 1: Quality Metrics (4 panels)
    # =========================================================================

    # 1.1 Perplexity by Dataset
    ax1 = plt.subplot(4, 4, 1)
    _plot_metric_by_dataset(ax1, df, 'perplexity', 'Perplexity (↓ better)', colors)

    # 1.2 Token Accuracy by Dataset
    ax2 = plt.subplot(4, 4, 2)
    _plot_metric_by_dataset(ax2, df, 'token_accuracy', 'Token Accuracy (↑ better)', colors, percentage=True)

    # 1.3 LAMBADA Accuracy
    ax3 = plt.subplot(4, 4, 3)
    _plot_metric_by_dataset(ax3, df, 'lambada_accuracy', 'LAMBADA Accuracy (↑ better)', colors, percentage=True)

    # 1.4 HellaSwag Accuracy
    ax4 = plt.subplot(4, 4, 4)
    _plot_metric_by_dataset(ax4, df, 'hellaswag_accuracy', 'HellaSwag Accuracy (↑ better)', colors, percentage=True)

    # =========================================================================
    # ROW 2: Routing Efficiency (4 panels)
    # =========================================================================

    # 2.1 Average Experts Used
    ax5 = plt.subplot(4, 4, 5)
    if 'avg_experts' in df.columns and len(adaptive_df) > 0:
        valid = adaptive_df.dropna(subset=['avg_experts']).head(15)
        if len(valid) > 0:
            bars = ax5.barh(range(len(valid)), valid['avg_experts'], color=colors['bh'], alpha=0.8)
            ax5.axvline(x=8, color=colors['bad'], linestyle='--', linewidth=2, label='TopK=8')
            ax5.set_yticks(range(len(valid)))
            ax5.set_yticklabels(valid['config_name'], fontsize=7)
            ax5.set_xlabel('Average Experts')
            ax5.legend()
        else:
            _empty_plot(ax5, 'No avg_experts data')
    else:
        _empty_plot(ax5, 'avg_experts not computed')
    ax5.set_title('Average Experts Used')

    # 2.2 Expert Reduction vs Baseline
    ax6 = plt.subplot(4, 4, 6)
    if 'reduction_vs_baseline' in adaptive_df.columns:
        valid = adaptive_df.dropna(subset=['reduction_vs_baseline'])
        valid = valid.sort_values('reduction_vs_baseline', ascending=False).head(10)
        if len(valid) > 0:
            colors_bars = [colors['good'] if r > 0 else colors['bad'] for r in valid['reduction_vs_baseline']]
            ax6.barh(range(len(valid)), valid['reduction_vs_baseline'], color=colors_bars, alpha=0.8)
            ax6.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax6.set_yticks(range(len(valid)))
            ax6.set_yticklabels(valid['config_name'], fontsize=7)
            ax6.set_xlabel('Reduction (%)')
        else:
            _empty_plot(ax6, 'No reduction data')
    else:
        _empty_plot(ax6, 'reduction_vs_baseline not computed')
    ax6.set_title('Expert Reduction vs TopK=8')

    # 2.3 Ceiling/Floor Hit Rates
    ax7 = plt.subplot(4, 4, 7)
    if all(col in adaptive_df.columns for col in ['ceiling_hit_rate', 'floor_hit_rate', 'mid_range_rate']):
        valid = adaptive_df.dropna(subset=['ceiling_hit_rate', 'floor_hit_rate']).head(10)
        if len(valid) > 0:
            x = np.arange(len(valid))
            width = 0.25
            ax7.bar(x - width, valid['floor_hit_rate'], width, label='Floor (min_k)', color=colors['bh'], alpha=0.7)
            ax7.bar(x, valid.get('mid_range_rate', 0), width, label='Mid-range', color=colors['good'], alpha=0.7)
            ax7.bar(x + width, valid['ceiling_hit_rate'], width, label='Ceiling (max_k)', color=colors['bad'], alpha=0.7)
            ax7.set_xticks(x)
            ax7.set_xticklabels([f"{i}" for i in range(len(valid))], fontsize=7)
            ax7.set_ylabel('Hit Rate (%)')
            ax7.legend(fontsize=7)
        else:
            _empty_plot(ax7, 'No hit rate data')
    else:
        _empty_plot(ax7, 'Hit rates not computed')
    ax7.set_title('Ceiling/Floor Hit Rates')

    # 2.4 Adaptive Range
    ax8 = plt.subplot(4, 4, 8)
    if 'adaptive_range' in adaptive_df.columns:
        valid = adaptive_df.dropna(subset=['adaptive_range'])
        valid = valid.sort_values('adaptive_range', ascending=False).head(10)
        if len(valid) > 0:
            ax8.barh(range(len(valid)), valid['adaptive_range'], color=colors['neutral'], alpha=0.8)
            ax8.set_yticks(range(len(valid)))
            ax8.set_yticklabels(valid['config_name'], fontsize=7)
            ax8.set_xlabel('Range (max - min experts)')
        else:
            _empty_plot(ax8, 'No adaptive_range data')
    else:
        _empty_plot(ax8, 'adaptive_range not computed')
    ax8.set_title('Adaptive Range (Dynamic Selection)')

    # =========================================================================
    # ROW 3: Routing Statistics (4 panels)
    # =========================================================================

    # 3.1 Expert Count Distribution
    ax9 = plt.subplot(4, 4, 9)
    if 'avg_experts' in adaptive_df.columns and 'std_experts' in adaptive_df.columns:
        valid = adaptive_df.dropna(subset=['avg_experts', 'std_experts']).head(15)
        if len(valid) > 0:
            ax9.errorbar(
                range(len(valid)),
                valid['avg_experts'],
                yerr=valid['std_experts'],
                fmt='o',
                capsize=4,
                color=colors['bh'],
                markersize=6
            )
            ax9.axhline(y=8, color=colors['bad'], linestyle='--', label='TopK=8')
            ax9.axhline(y=1, color=colors['good'], linestyle=':', label='min_k=1')
            ax9.set_ylabel('Experts (mean ± std)')
            ax9.legend(fontsize=7)
        else:
            _empty_plot(ax9, 'No distribution data')
    else:
        _empty_plot(ax9, 'Expert distribution not computed')
    ax9.set_title('Expert Count Distribution')

    # 3.2 Routing Entropy Distribution
    ax10 = plt.subplot(4, 4, 10)
    if 'avg_entropy' in df.columns:
        valid = df['avg_entropy'].dropna()
        if len(valid) > 0:
            ax10.hist(valid, bins=20, color=colors['hc'], alpha=0.7, edgecolor='black')
            ax10.axvline(x=valid.mean(), color=colors['bad'], linestyle='--',
                        label=f'Mean: {valid.mean():.2f}')
            ax10.set_xlabel('Routing Entropy')
            ax10.set_ylabel('Count')
            ax10.legend(fontsize=7)
        else:
            _empty_plot(ax10, 'No entropy data')
    else:
        _empty_plot(ax10, 'avg_entropy not computed')
    ax10.set_title('Routing Entropy Distribution')

    # 3.3 Expert Utilization
    ax11 = plt.subplot(4, 4, 11)
    if 'expert_utilization' in df.columns or 'unique_experts' in df.columns:
        if 'expert_utilization' in df.columns:
            valid = df.dropna(subset=['expert_utilization']).head(15)
            values = valid['expert_utilization'] * 100
        else:
            valid = df.dropna(subset=['unique_experts']).head(15)
            values = valid['unique_experts'] / 64 * 100

        if len(valid) > 0:
            ax11.bar(range(len(valid)), values, color=colors['good'], alpha=0.8)
            ax11.axhline(y=50, color=colors['neutral'], linestyle='--', label='50%')
            ax11.set_ylabel('Utilization (%)')
            ax11.set_ylim(0, 100)
            ax11.legend(fontsize=7)
        else:
            _empty_plot(ax11, 'No utilization data')
    else:
        _empty_plot(ax11, 'Expert utilization not computed')
    ax11.set_title('Expert Utilization Rate')

    # 3.4 Weight Concentration
    ax12 = plt.subplot(4, 4, 12)
    if 'avg_max_weight' in df.columns or 'avg_concentration' in df.columns:
        col = 'avg_max_weight' if 'avg_max_weight' in df.columns else 'avg_concentration'
        valid = df.dropna(subset=[col]).head(15)
        if len(valid) > 0:
            ax12.bar(range(len(valid)), valid[col], color=colors['neutral'], alpha=0.8)
            ax12.set_ylabel('Concentration')
            ax12.set_ylim(0, 1)
        else:
            _empty_plot(ax12, 'No concentration data')
    else:
        _empty_plot(ax12, 'Weight concentration not computed')
    ax12.set_title('Weight Concentration')

    # =========================================================================
    # ROW 4: Comparisons & Summary (4 panels)
    # =========================================================================

    # 4.1 Perplexity: Adaptive vs Baseline
    ax13 = plt.subplot(4, 4, 13)
    if len(baseline_df) > 0 and len(adaptive_df) > 0 and 'perplexity' in df.columns:
        for dataset in df['dataset'].unique():
            base = baseline_df[baseline_df['dataset'] == dataset]['perplexity'].dropna()
            adap = adaptive_df[adaptive_df['dataset'] == dataset]['perplexity'].dropna()
            if len(base) > 0 and len(adap) > 0:
                base_mean = base.mean()
                ax13.scatter([base_mean] * len(adap), adap, label=dataset, s=60, alpha=0.7)

        lim = ax13.get_xlim()
        ax13.plot(lim, lim, 'k--', alpha=0.3, label='y=x')
        ax13.set_xlabel('Baseline Perplexity')
        ax13.set_ylabel('BH/HC Perplexity')
        ax13.legend(fontsize=7)
    else:
        _empty_plot(ax13, 'Cannot compare (missing baseline or perplexity)')
    ax13.set_title('Adaptive vs Baseline Perplexity')

    # 4.2 Speed vs Quality Trade-off
    ax14 = plt.subplot(4, 4, 14)
    if 'tokens_per_second' in df.columns and 'perplexity' in df.columns:
        valid = df.dropna(subset=['tokens_per_second', 'perplexity'])
        if len(valid) > 0:
            for rt in valid.get('routing_type', pd.Series(['bh'] * len(valid))).unique():
                subset = valid[valid.get('routing_type', 'bh') == rt] if 'routing_type' in valid.columns else valid
                color = colors.get(rt, colors['bh'])
                ax14.scatter(subset['perplexity'], subset['tokens_per_second'],
                            c=color, s=80, alpha=0.7, label=rt.upper())
            ax14.set_xlabel('Perplexity (↓ better)')
            ax14.set_ylabel('Tokens/Second (↑ better)')
            ax14.legend(fontsize=7)
        else:
            _empty_plot(ax14, 'No speed/quality data')
    else:
        _empty_plot(ax14, 'Speed metrics not computed')
    ax14.set_title('Speed vs Quality Trade-off')

    # 4.3 Parameter Sensitivity (Alpha/Beta)
    ax15 = plt.subplot(4, 4, 15)
    param_col = 'alpha' if 'alpha' in adaptive_df.columns else ('beta' if 'beta' in adaptive_df.columns else None)
    if param_col and 'avg_experts' in adaptive_df.columns:
        valid = adaptive_df.dropna(subset=[param_col, 'avg_experts'])
        if len(valid) > 0:
            for max_k in valid.get('max_k', valid.get('k_or_max_k', pd.Series([8]))).unique():
                subset = valid[(valid.get('max_k', valid.get('k_or_max_k', 8)) == max_k)]
                if len(subset) > 0:
                    ax15.plot(subset[param_col], subset['avg_experts'], 'o-',
                             label=f'max_k={int(max_k)}', markersize=8)
            ax15.set_xlabel(f'{param_col.capitalize()} Parameter')
            ax15.set_ylabel('Average Experts')
            ax15.legend(fontsize=7)
        else:
            _empty_plot(ax15, 'No parameter sensitivity data')
    else:
        _empty_plot(ax15, 'Alpha/Beta parameter not found')
    ax15.set_title('Parameter Sensitivity')

    # 4.4 Summary Statistics Table
    ax16 = plt.subplot(4, 4, 16)
    ax16.axis('off')

    summary = _generate_summary_text(df, adaptive_df, baseline_df, routing_method)
    ax16.text(0.05, 0.95, summary, transform=ax16.transAxes, fontsize=9,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax16.set_title('Summary Statistics', fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved dashboard: {output_path}")
    return output_path


def _plot_metric_by_dataset(ax, df, metric, ylabel, colors, percentage=False):
    """Helper to plot a metric by dataset."""
    if metric in df.columns:
        valid = df.dropna(subset=[metric])
        if len(valid) > 0:
            datasets = valid['dataset'].unique() if 'dataset' in valid.columns else ['default']
            x = np.arange(len(valid))
            width = 0.8 / len(datasets)

            for i, dataset in enumerate(datasets):
                subset = valid[valid['dataset'] == dataset] if 'dataset' in valid.columns else valid
                values = subset[metric] * 100 if percentage else subset[metric]
                ax.bar(np.arange(len(subset)) + i * width, values, width,
                      label=dataset, alpha=0.8)

            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7)
            if percentage:
                ax.set_ylim(0, 100)
        else:
            _empty_plot(ax, f'No valid {metric} data')
    else:
        _empty_plot(ax, f'{metric} not computed')
    ax.set_title(ylabel.split('(')[0].strip())


def _empty_plot(ax, message):
    """Create placeholder for empty/missing data."""
    ax.text(0.5, 0.5, message, ha='center', va='center',
            transform=ax.transAxes, fontsize=10, color='gray',
            style='italic')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _generate_summary_text(df, adaptive_df, baseline_df, routing_method):
    """Generate summary statistics text."""
    lines = [
        f"{'='*40}",
        f"  EXPERIMENT SUMMARY ({routing_method})",
        f"{'='*40}",
        f"",
        f"Configurations: {len(df)}",
        f"  • Adaptive (BH/HC): {len(adaptive_df)}",
        f"  • Baseline (TopK): {len(baseline_df)}",
        f""
    ]

    if 'dataset' in df.columns:
        lines.append(f"Datasets: {df['dataset'].nunique()}")
        for ds in df['dataset'].unique():
            lines.append(f"  • {ds}")
        lines.append("")

    # Quality metrics
    lines.append("Quality Metrics:")
    for metric, label in [('perplexity', 'Perplexity'),
                          ('token_accuracy', 'Token Acc'),
                          ('lambada_accuracy', 'LAMBADA Acc'),
                          ('hellaswag_accuracy', 'HellaSwag Acc')]:
        if metric in df.columns:
            valid = df[metric].dropna()
            if len(valid) > 0:
                lines.append(f"  • {label}: {valid.mean():.3f} (n={len(valid)})")
            else:
                lines.append(f"  • {label}: No data")
        else:
            lines.append(f"  • {label}: ❌ Not computed")

    lines.append("")

    # Routing metrics
    if len(adaptive_df) > 0 and 'avg_experts' in adaptive_df.columns:
        valid = adaptive_df['avg_experts'].dropna()
        if len(valid) > 0:
            lines.append("Routing Efficiency:")
            lines.append(f"  • Avg Experts: {valid.mean():.2f}")
            lines.append(f"  • vs TopK=8: {(8 - valid.mean()) / 8 * 100:.1f}% reduction")

    return '\n'.join(lines)


def plot_per_layer_routing(
    per_layer_stats: Dict[int, Dict],
    output_path: str,
    title: str = "Per-Layer Routing Statistics"
):
    """Plot routing statistics per layer."""

    layers = sorted(per_layer_stats.keys())
    entropies = [per_layer_stats[l].get('avg_entropy', per_layer_stats[l].get('entropy', 0)) for l in layers]
    max_probs = [per_layer_stats[l].get('avg_max_prob', per_layer_stats[l].get('max_prob', 0)) for l in layers]
    concentrations = [per_layer_stats[l].get('avg_concentration', per_layer_stats[l].get('concentration', 0)) for l in layers]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    axes[0].bar(layers, entropies, color='#3498db', alpha=0.8, edgecolor='black')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Entropy')
    axes[0].set_title('Routing Entropy per Layer')

    axes[1].bar(layers, max_probs, color='#2ecc71', alpha=0.8, edgecolor='black')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Max Probability')
    axes[1].set_title('Max Routing Probability per Layer')

    axes[2].bar(layers, concentrations, color='#e74c3c', alpha=0.8, edgecolor='black')
    axes[2].set_xlabel('Layer')
    axes[2].set_ylabel('Concentration')
    axes[2].set_title('Weight Concentration per Layer')

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved per-layer plot: {output_path}")


def plot_expert_usage_heatmap(
    expert_usage: Dict[int, int],
    num_experts: int,
    output_path: str,
    title: str = "Expert Usage Heatmap"
):
    """Plot expert usage as 8x8 heatmap."""

    usage_array = np.zeros(num_experts)
    for expert_id, count in expert_usage.items():
        if expert_id < num_experts:
            usage_array[expert_id] = count

    # Normalize
    total = usage_array.sum()
    if total > 0:
        usage_array = usage_array / total * 100

    # Reshape to 8x8
    usage_matrix = usage_array.reshape(8, 8)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(usage_matrix, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Usage %'})
    ax.set_title(title)
    ax.set_xlabel('Expert Index (mod 8)')
    ax.set_ylabel('Expert Index (// 8)')

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved expert heatmap: {output_path}")


def plot_bh_vs_hc_comparison(
    bh_results: pd.DataFrame,
    hc_results: pd.DataFrame,
    output_path: str
):
    """Compare BH and HC routing side by side."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('BH vs HC Routing Comparison', fontsize=16, fontweight='bold')

    # 1. Perplexity comparison
    ax = axes[0, 0]
    if 'perplexity' in bh_results.columns and 'perplexity' in hc_results.columns:
        bh_ppl = bh_results['perplexity'].dropna()
        hc_ppl = hc_results['perplexity'].dropna()
        ax.boxplot([bh_ppl, hc_ppl], labels=['BH', 'HC'])
        ax.set_ylabel('Perplexity')
    ax.set_title('Perplexity Distribution')

    # 2. Average experts
    ax = axes[0, 1]
    if 'avg_experts' in bh_results.columns and 'avg_experts' in hc_results.columns:
        bh_exp = bh_results['avg_experts'].dropna()
        hc_exp = hc_results['avg_experts'].dropna()
        ax.boxplot([bh_exp, hc_exp], labels=['BH', 'HC'])
        ax.axhline(y=8, color='red', linestyle='--', label='TopK=8')
        ax.set_ylabel('Average Experts')
        ax.legend()
    ax.set_title('Expert Count Distribution')

    # 3. Reduction comparison
    ax = axes[0, 2]
    if 'reduction_vs_baseline' in bh_results.columns and 'reduction_vs_baseline' in hc_results.columns:
        bh_red = bh_results['reduction_vs_baseline'].dropna()
        hc_red = hc_results['reduction_vs_baseline'].dropna()
        ax.bar([0, 1], [bh_red.mean(), hc_red.mean()], color=['#3498db', '#9b59b6'], alpha=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['BH', 'HC'])
        ax.set_ylabel('Avg Reduction (%)')
    ax.set_title('Average Expert Reduction')

    # 4. Entropy comparison
    ax = axes[1, 0]
    if 'avg_entropy' in bh_results.columns and 'avg_entropy' in hc_results.columns:
        bh_ent = bh_results['avg_entropy'].dropna()
        hc_ent = hc_results['avg_entropy'].dropna()
        ax.boxplot([bh_ent, hc_ent], labels=['BH', 'HC'])
        ax.set_ylabel('Routing Entropy')
    ax.set_title('Routing Entropy')

    # 5. Adaptive range
    ax = axes[1, 1]
    if 'adaptive_range' in bh_results.columns and 'adaptive_range' in hc_results.columns:
        bh_range = bh_results['adaptive_range'].dropna()
        hc_range = hc_results['adaptive_range'].dropna()
        ax.boxplot([bh_range, hc_range], labels=['BH', 'HC'])
        ax.set_ylabel('Adaptive Range')
    ax.set_title('Selection Dynamism')

    # 6. Summary
    ax = axes[1, 2]
    ax.axis('off')

    summary = []
    summary.append("COMPARISON SUMMARY")
    summary.append("=" * 30)
    summary.append(f"BH Configs: {len(bh_results)}")
    summary.append(f"HC Configs: {len(hc_results)}")
    summary.append("")

    if 'avg_experts' in bh_results.columns:
        summary.append(f"BH Avg Experts: {bh_results['avg_experts'].mean():.2f}")
    if 'avg_experts' in hc_results.columns:
        summary.append(f"HC Avg Experts: {hc_results['avg_experts'].mean():.2f}")

    ax.text(0.1, 0.9, '\n'.join(summary), transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved comparison plot: {output_path}")
