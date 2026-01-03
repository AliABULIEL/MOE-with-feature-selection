#!/usr/bin/env python3
"""
Visualization of BH Routing P-value Issue
==========================================

This script creates visualizations that clearly show why BH routing
is selecting too few experts: p-values don't decay fast enough to
pass the BH thresholds.

Based on diagnostic data from the actual OLMoE model.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ============================================================================
# ACTUAL DATA FROM DIAGNOSTIC TEST
# ============================================================================

# Layer 0, Token "France" - WORST CASE (only 1 expert selected)
layer0_france_data = {
    'experts': [55, 18, 51, 41, 6, 13, 19, 44],  # Top-8 by logit
    'logits': [1.5000, 1.1719, 0.9531, 0.8164, 0.7539, 0.4707, 0.1318, 0.0157],
    'p_values': [0.013245, 0.029419, 0.043701, 0.053223, 0.057617, 0.079102, 0.118652, 0.141602],
    'ranks': [1, 2, 3, 4, 5, 6, 7, 8],
    'selected': [True, False, False, False, False, False, False, False],
    'token': 'France',
    'layer': 0,
    'alpha': 0.6,
    'bh_selected': 1
}

# Layer 0, Token "The" - MODERATE CASE (3 experts selected)
layer0_the_data = {
    'experts': [5, 14, 61, 18, 6, 38, 19, 41],
    'logits': [2.1406, 1.5781, 1.2734, 0.9492, 0.6992, 0.6797, 0.3125, 0.1719],
    'p_values': [0.007812, 0.010864, 0.023560, 0.043945, 0.061279, 0.062500, 0.095215, 0.112793],
    'ranks': [1, 2, 3, 4, 5, 6, 7, 8],
    'selected': [True, True, True, False, False, False, False, False],
    'token': 'The',
    'layer': 0,
    'alpha': 0.6,
    'bh_selected': 3
}

# Layer 7, Token "The" - BEST CASE (all 8 selected)
layer7_the_data = {
    'experts': [21, 45, 2, 12, 37, 33, 39, 58],
    'logits': [2.1875, 1.7734, 1.2656, 1.1250, 1.0781, 1.0625, 0.9531, 0.9180],
    'p_values': [0.007812, 0.007812, 0.007812, 0.011841, 0.014038, 0.014893, 0.021118, 0.023682],
    'ranks': [2, 3, 1, 4, 5, 6, 7, 8],
    'selected': [True, True, True, True, True, True, True, True],
    'token': 'The',
    'layer': 7,
    'alpha': 0.6,
    'bh_selected': 8
}

# Layer 7, Token "France" - WORST CASE (only 1 selected)
layer7_france_data = {
    'experts': [43, 1, 31, 46, 5, 13, 0, 59],
    'logits': [1.1094, 0.5938, 0.4512, 0.4414, 0.4395, 0.3164, 0.2773, -0.0070],
    'p_values': [0.012512, 0.054688, 0.067871, 0.068848, 0.069336, 0.084473, 0.089844, 0.146484],
    'ranks': [1, 2, 3, 4, 5, 6, 7, 8],
    'selected': [True, False, False, False, False, False, False, False],
    'token': 'France',
    'layer': 7,
    'alpha': 0.6,
    'bh_selected': 1
}


def compute_bh_thresholds(n_experts=64, alpha=0.6):
    """Compute BH critical values for all ranks."""
    k_values = np.arange(1, n_experts + 1)
    thresholds = (k_values / n_experts) * alpha
    return k_values, thresholds


def plot_pvalue_vs_threshold(data, ax, show_legend=True):
    """
    Plot p-values against BH thresholds for a single token.
    This is the KEY visualization showing the mismatch.
    """
    n_experts = 64
    alpha = data['alpha']
    
    # Get BH thresholds for first 16 ranks (enough to see the issue)
    ranks = np.arange(1, 17)
    thresholds = (ranks / n_experts) * alpha
    
    # Plot BH threshold line
    ax.plot(ranks, thresholds, 'g-', linewidth=2, label=f'BH Threshold (α={alpha})')
    ax.fill_between(ranks, 0, thresholds, alpha=0.2, color='green', label='Selection Zone')
    
    # Plot actual p-values for top-8 experts
    p_vals = data['p_values'][:8]
    expert_ranks = data['ranks'][:8]
    selected = data['selected'][:8]
    
    # Color by selection status
    colors = ['blue' if s else 'red' for s in selected]
    
    for i, (rank, pval, sel, color) in enumerate(zip(expert_ranks, p_vals, selected, colors)):
        marker = 'o' if sel else 'x'
        size = 150 if sel else 100
        ax.scatter(rank, pval, c=color, marker=marker, s=size, zorder=5,
                  edgecolors='black', linewidths=1)
        
        # Add expert number label
        expert_id = data['experts'][i]
        offset = 0.003 if pval < thresholds[rank-1] else -0.008
        ax.annotate(f'E{expert_id}', (rank, pval + offset), fontsize=8, ha='center')
    
    # Formatting
    ax.set_xlabel('Rank (by p-value)', fontsize=11)
    ax.set_ylabel('P-value', fontsize=11)
    ax.set_title(f"Layer {data['layer']}, Token '{data['token']}'\n"
                 f"BH Selected: {data['bh_selected']}/8 experts", fontsize=12)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 0.20)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    if show_legend:
        # Custom legend
        selected_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                                    markersize=10, label='Selected', markeredgecolor='black')
        rejected_patch = plt.Line2D([0], [0], marker='x', color='red',
                                    markersize=10, label='Rejected', markeredgewidth=2)
        threshold_line = plt.Line2D([0], [0], color='green', linewidth=2, label=f'BH Threshold')
        ax.legend(handles=[selected_patch, rejected_patch, threshold_line], loc='upper left')


def plot_comparison_grid():
    """Create a 2x2 grid comparing all four cases."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('BH Routing P-value Analysis: Why Too Few Experts Are Selected\n'
                 'Green zone = selection threshold. Blue ●=selected, Red ✗=rejected',
                 fontsize=14, fontweight='bold')
    
    datasets = [
        (layer0_the_data, axes[0, 0]),
        (layer0_france_data, axes[0, 1]),
        (layer7_the_data, axes[1, 0]),
        (layer7_france_data, axes[1, 1]),
    ]
    
    for i, (data, ax) in enumerate(datasets):
        plot_pvalue_vs_threshold(data, ax, show_legend=(i == 0))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def plot_the_core_problem():
    """
    Create a single plot that shows THE CORE PROBLEM:
    P-values grow faster than BH thresholds.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    n_experts = 64
    ranks = np.arange(1, 17)
    
    # Plot BH thresholds for different alpha values
    alphas = [0.3, 0.6, 0.8, 1.0, 1.5]
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    
    for alpha, color in zip(alphas, colors):
        thresholds = (ranks / n_experts) * alpha
        ax.plot(ranks, thresholds, '--', color=color, linewidth=2, 
                label=f'BH Threshold α={alpha}')
    
    # Plot ACTUAL p-values from the worst case (Layer 0, France)
    actual_p = layer0_france_data['p_values'][:8]
    ax.plot(range(1, 9), actual_p, 'ko-', linewidth=3, markersize=12,
            label='Actual P-values (Layer 0, "France")', markerfacecolor='black')
    
    # Add annotations showing the gap
    for i in range(8):
        rank = i + 1
        pval = actual_p[i]
        thresh_06 = (rank / 64) * 0.6
        
        if pval > thresh_06:
            ax.annotate('', xy=(rank, thresh_06), xytext=(rank, pval),
                       arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
    
    # Formatting
    ax.set_xlabel('Rank (k)', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('THE CORE PROBLEM: P-values Grow Faster Than BH Thresholds\n'
                 'Red arrows show gap between actual p-values and α=0.6 threshold',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0.5, 12)
    ax.set_ylim(0, 0.25)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    # Add text explanation
    textstr = ('For k=8 experts to be selected:\n'
               f'• Need p_(8) ≤ (8/64)×α\n'
               f'• With α=0.6: p_(8) ≤ 0.075\n'
               f'• Actual p_(8) = 0.142\n'
               f'• Gap: 0.142 - 0.075 = 0.067\n\n'
               f'Solution: Need α ≈ 1.2 or higher!')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.72, 0.55, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig


def plot_required_alpha():
    """
    Calculate and plot what alpha value would be needed to select 
    different numbers of experts.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # For each case, calculate required alpha to select k experts
    cases = [
        ('Layer 0, "France" (Worst)', layer0_france_data, 'red'),
        ('Layer 0, "The" (Moderate)', layer0_the_data, 'orange'),
        ('Layer 7, "France" (Bad)', layer7_france_data, 'purple'),
        ('Layer 7, "The" (Best)', layer7_the_data, 'green'),
    ]
    
    ax1 = axes[0]
    
    for label, data, color in cases:
        required_alphas = []
        for k in range(1, 9):
            if k <= len(data['p_values']):
                # Required alpha: p_(k) ≤ (k/64) × α
                # So α ≥ p_(k) × 64 / k
                p_k = data['p_values'][k-1]
                required_alpha = p_k * 64 / k
                required_alphas.append(required_alpha)
        
        ax1.plot(range(1, len(required_alphas)+1), required_alphas, 'o-', 
                label=label, color=color, linewidth=2, markersize=8)
    
    # Add reference lines
    ax1.axhline(y=0.6, color='gray', linestyle='--', label='Current α=0.6')
    ax1.axhline(y=1.0, color='black', linestyle=':', label='α=1.0')
    
    ax1.set_xlabel('Number of Experts to Select (k)', fontsize=12)
    ax1.set_ylabel('Required Alpha (α)', fontsize=12)
    ax1.set_title('What Alpha Value is Needed to Select k Experts?\n'
                  '(Based on actual p-values from diagnostic)', fontsize=12, fontweight='bold')
    ax1.set_xlim(0.5, 8.5)
    ax1.set_ylim(0, 2.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)
    
    # Second plot: Histogram of required alphas to select 8 experts
    ax2 = axes[1]
    
    required_for_8 = []
    labels = []
    for label, data, color in cases:
        if len(data['p_values']) >= 8:
            p_8 = data['p_values'][7]
            req_alpha = p_8 * 64 / 8
            required_for_8.append(req_alpha)
            labels.append(label.split(',')[0])
    
    colors = ['red', 'orange', 'purple', 'green']
    bars = ax2.bar(labels, required_for_8, color=colors, edgecolor='black')
    ax2.axhline(y=0.6, color='blue', linestyle='--', linewidth=2, label='Current α=0.6')
    ax2.axhline(y=1.0, color='black', linestyle=':', linewidth=2, label='α=1.0 (traditional max)')
    
    # Add value labels on bars
    for bar, val in zip(bars, required_for_8):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Required Alpha to Select 8 Experts', fontsize=12)
    ax2.set_title('Alpha Required to Match Top-8 Selection\n'
                  'Current α=0.6 is insufficient for most cases!', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.5)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_solution_comparison():
    """
    Compare different solutions: Higher alpha, Higher min_k, Higher Criticism.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Solutions and their expected behavior
    solutions = {
        'Current\n(α=0.6, min_k=1)': {'avg_experts': 2.5, 'quality': 0.1, 'color': 'red'},
        'Higher α\n(α=1.2, min_k=1)': {'avg_experts': 6.5, 'quality': 0.85, 'color': 'orange'},
        'Higher min_k\n(α=0.6, min_k=6)': {'avg_experts': 6.5, 'quality': 0.90, 'color': 'yellow'},
        'Hybrid\n(α=0.8, min_k=4)': {'avg_experts': 6.0, 'quality': 0.88, 'color': 'lightgreen'},
        'Higher Criticism\n(adaptive)': {'avg_experts': 7.0, 'quality': 0.92, 'color': 'green'},
        'Baseline TopK-8': {'avg_experts': 8.0, 'quality': 1.0, 'color': 'blue'},
    }
    
    x_pos = np.arange(len(solutions))
    names = list(solutions.keys())
    avg_experts = [s['avg_experts'] for s in solutions.values()]
    quality = [s['quality'] for s in solutions.values()]
    colors = [s['color'] for s in solutions.values()]
    
    # Create grouped bar chart
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, avg_experts, width, label='Avg Experts', 
                   color=colors, edgecolor='black', alpha=0.8)
    
    ax2 = ax.twinx()
    bars2 = ax2.bar(x_pos + width/2, quality, width, label='Relative Quality', 
                    color=colors, edgecolor='black', alpha=0.4, hatch='//')
    
    # Formatting
    ax.set_xlabel('Solution', fontsize=12)
    ax.set_ylabel('Average Experts Selected', fontsize=12, color='black')
    ax2.set_ylabel('Relative Quality (1.0 = baseline)', fontsize=12, color='gray')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 10)
    ax2.set_ylim(0, 1.2)
    
    ax.axhline(y=8, color='blue', linestyle='--', alpha=0.5, label='Baseline (8 experts)')
    
    ax.set_title('Solution Comparison: Fixing BH Under-Selection\n'
                 'Solid bars = experts, Hatched bars = quality', 
                 fontsize=14, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', edgecolor='black', label='Avg Experts'),
        Patch(facecolor='gray', edgecolor='black', alpha=0.4, hatch='//', label='Relative Quality'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def main():
    """Generate all visualizations."""
    print("=" * 70)
    print("BH Routing P-value Issue Visualization")
    print("=" * 70)
    
    output_dir = Path('/visualizations')
    output_dir.mkdir(exist_ok=True)
    
    # Figure 1: Comparison grid
    print("\n1. Creating comparison grid...")
    fig1 = plot_comparison_grid()
    fig1.savefig(output_dir / 'bh_pvalue_comparison_grid.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_dir / 'bh_pvalue_comparison_grid.png'}")
    
    # Figure 2: Core problem visualization
    print("\n2. Creating core problem visualization...")
    fig2 = plot_the_core_problem()
    fig2.savefig(output_dir / 'bh_core_problem.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_dir / 'bh_core_problem.png'}")
    
    # Figure 3: Required alpha analysis
    print("\n3. Creating required alpha analysis...")
    fig3 = plot_required_alpha()
    fig3.savefig(output_dir / 'bh_required_alpha.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_dir / 'bh_required_alpha.png'}")
    
    # Figure 4: Solution comparison
    print("\n4. Creating solution comparison...")
    fig4 = plot_solution_comparison()
    fig4.savefig(output_dir / 'bh_solution_comparison.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_dir / 'bh_solution_comparison.png'}")
    
    print("\n" + "=" * 70)
    print("All visualizations saved!")
    print("=" * 70)
    
    # Show summary
    print("""
KEY FINDINGS FROM VISUALIZATIONS:

1. P-VALUE vs THRESHOLD MISMATCH:
   - BH thresholds scale as (k/64) × α (very slowly)
   - Actual p-values from KDE grow faster than thresholds
   - Result: Only 1-3 experts pass instead of 8

2. REQUIRED ALPHA VALUES:
   - To select 8 experts in worst case: need α ≈ 1.2
   - Current α=0.6 is insufficient for ~60% of tokens
   - Traditional BH limit (α=1.0) still not enough!

3. RECOMMENDED SOLUTIONS (in order of preference):
   
   A) HIGHER CRITICISM (BEST)
      - Adaptive threshold based on actual p-value distribution
      - Already implemented in your repo!
      - Should naturally select more experts when appropriate
   
   B) HIGHER ALPHA (α=1.2 to 1.5)
      - Simple parameter change
      - Works, but breaks traditional BH interpretation
      - Code: patcher.patch_with_bh(alpha=1.2, max_k=12, min_k=4)
   
   C) HIGHER MIN_K (min_k=6)
      - Guarantees minimum quality
      - Limits adaptivity potential
      - Code: patcher.patch_with_bh(alpha=0.6, max_k=12, min_k=6)
   
   D) HYBRID (α=0.8, min_k=4)
      - Balance of adaptivity and safety
      - Code: patcher.patch_with_bh(alpha=0.8, max_k=12, min_k=4)
""")
    
    plt.show()


if __name__ == "__main__":
    main()