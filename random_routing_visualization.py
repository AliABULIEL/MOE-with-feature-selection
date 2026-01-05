"""
Random Routing Visualization Suite
==========================================

This module implements visualizations for random routing evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple

def create_comprehensive_visualization(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 150
) -> plt.Figure:
    """
    Create a 2-panel visualization for random routing.

    Args:
        results_df: DataFrame with experiment results.
        output_path: Path to save the figure (optional).
        figsize: Figure size in inches.
        dpi: Resolution for the saved figure.
    """
    sns.set_style('whitegrid')
    plt.rcParams['font.size'] = 10
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Random Routing Evaluation', fontsize=16, fontweight='bold')

    # Panel 1: Distribution of experts selected
    ax1 = axes[0]
    if 'avg_experts' in results_df.columns:
        ax1.hist(results_df['avg_experts'], bins=10, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Average Experts Selected')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Expert Selection Distribution')
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No expert count data', ha='center')

    # Panel 2: Distribution of weight sums
    ax2 = axes[1]
    if 'avg_weight_sum' in results_df.columns:
        ax2.hist(results_df['avg_weight_sum'], bins=20, edgecolor='black', alpha=0.7, color='coral')
        ax2.set_xlabel('Average Sum of Weights')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Weight Sum Distribution')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No weight sum data', ha='center')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Saved visualization: {output_path}")

    return fig
