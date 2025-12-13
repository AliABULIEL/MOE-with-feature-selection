"""
Routing Visualizations Module
==============================

Dedicated visualization functions for analyzing MoE routing decisions.

This module provides publication-quality visualizations for:
- Expert selection distributions
- Alpha/temperature sensitivity analysis
- Per-token routing heatmaps
- Expert utilization patterns
- Token complexity analysis
- Cross-method comparisons
- Layer-wise routing patterns

All functions:
- Support both CPU and GPU tensors (auto-convert)
- Include comprehensive docstrings with examples
- Handle edge cases (empty inputs, single data point)
- Provide save_path option for exporting
- Return figure objects for further customization
- Use consistent styling (seaborn default, 12pt labels)

Author: BH Routing Analysis Project
Date: 2025-12-13
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Union, Any
from pathlib import Path
import warnings

# Set default style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def _to_numpy(tensor: Union[torch.Tensor, np.ndarray, List]) -> np.ndarray:
    """
    Convert tensor to numpy array, handling CPU/GPU tensors.

    Args:
        tensor: Input tensor, array, or list

    Returns:
        NumPy array
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, list):
        return np.array(tensor)
    else:
        raise TypeError(f"Unsupported type: {type(tensor)}")


def plot_expert_count_distribution(
    expert_counts: Union[torch.Tensor, np.ndarray],
    method_name: str = "BH Routing",
    alpha: Optional[float] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 100
) -> plt.Figure:
    """
    Plot distribution of expert counts with histogram and KDE overlay.

    Shows how many experts are selected per token, with statistical annotations.

    Args:
        expert_counts: Expert counts per token [num_tokens]
        method_name: Name of routing method for title
        alpha: FDR control level (for annotation)
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
        dpi: Resolution for display/saving

    Returns:
        matplotlib Figure object

    Example:
        >>> expert_counts = torch.tensor([4, 5, 3, 6, 4, 5, 4, 3])
        >>> fig = plot_expert_count_distribution(expert_counts, "BH Routing", alpha=0.05)
        >>> plt.show()

    Raises:
        ValueError: If expert_counts is empty
    """
    # Convert to numpy
    counts = _to_numpy(expert_counts).flatten()

    if len(counts) == 0:
        raise ValueError("expert_counts is empty")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Compute statistics
    mean_val = counts.mean()
    median_val = np.median(counts)
    std_val = counts.std()
    min_val = counts.min()
    max_val = counts.max()

    # Plot histogram
    n, bins, patches = ax.hist(
        counts,
        bins=max(int(max_val - min_val + 1), 10),
        alpha=0.6,
        color='steelblue',
        edgecolor='black',
        density=True,
        label='Distribution'
    )

    # Plot KDE if we have enough data
    if len(counts) > 5:
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(counts)
            x_range = np.linspace(min_val - 0.5, max_val + 0.5, 200)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        except:
            # KDE might fail for very small datasets
            pass

    # Add mean and median lines
    ax.axvline(mean_val, color='darkred', linestyle='--', linewidth=2,
               label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='darkgreen', linestyle='-.', linewidth=2,
               label=f'Median: {median_val:.2f}')

    # Annotations
    title = f'Expert Count Distribution - {method_name}'
    if alpha is not None:
        title += f' (α={alpha:.3f})'
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Number of Experts Selected', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')

    # Add statistics text box
    stats_text = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}\nRange: [{min_val:.0f}, {max_val:.0f}]'
    ax.text(0.98, 0.97, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi if dpi > 100 else 300, bbox_inches='tight')

    return fig


def plot_alpha_sensitivity(
    alphas: List[float],
    avg_experts: List[float],
    std_experts: Optional[List[float]] = None,
    baseline_k: int = 8,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 100
) -> plt.Figure:
    """
    Plot alpha sensitivity analysis with error bars.

    Shows how FDR control level (alpha) affects expert selection.

    Args:
        alphas: List of alpha values tested
        avg_experts: Average experts selected for each alpha
        std_experts: Standard deviation for each alpha (optional)
        baseline_k: Baseline top-k value for reference line
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
        dpi: Resolution for display/saving

    Returns:
        matplotlib Figure object

    Example:
        >>> alphas = [0.01, 0.05, 0.10, 0.20]
        >>> avg_experts = [3.2, 4.5, 5.8, 6.9]
        >>> std_experts = [0.5, 0.7, 0.8, 0.9]
        >>> fig = plot_alpha_sensitivity(alphas, avg_experts, std_experts)
        >>> plt.show()

    Raises:
        ValueError: If alphas and avg_experts have different lengths
    """
    if len(alphas) != len(avg_experts):
        raise ValueError(f"alphas ({len(alphas)}) and avg_experts ({len(avg_experts)}) must have same length")

    if len(alphas) == 0:
        raise ValueError("alphas is empty")

    # Convert to numpy
    alphas_arr = np.array(alphas)
    avg_arr = np.array(avg_experts)
    std_arr = np.array(std_experts) if std_experts is not None else None

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot line with error bars
    if std_arr is not None:
        ax.errorbar(alphas_arr, avg_arr, yerr=std_arr,
                   marker='o', markersize=8, linewidth=2, capsize=5,
                   label='BH Routing', color='steelblue')
    else:
        ax.plot(alphas_arr, avg_arr, 'o-', markersize=8, linewidth=2,
               label='BH Routing', color='steelblue')

    # Add baseline reference
    ax.axhline(y=baseline_k, color='red', linestyle='--', linewidth=2,
               label=f'Baseline (Top-{baseline_k})', alpha=0.7)

    # Annotations
    ax.set_title('Alpha Sensitivity Analysis', fontweight='bold')
    ax.set_xlabel('Alpha (FDR Control Level)', fontweight='bold')
    ax.set_ylabel('Mean Experts Selected', fontweight='bold')

    # Format x-axis
    ax.set_xticks(alphas_arr)
    ax.set_xticklabels([f'{a:.2f}' for a in alphas_arr])

    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Add shaded region showing reduction
    if avg_arr.max() < baseline_k:
        ax.fill_between(alphas_arr, avg_arr, baseline_k, alpha=0.2, color='green',
                        label='Reduction vs Baseline')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi if dpi > 100 else 300, bbox_inches='tight')

    return fig


def plot_routing_heatmap(
    routing_weights: Union[torch.Tensor, np.ndarray],
    token_strs: List[str],
    expert_indices: Optional[Union[torch.Tensor, np.ndarray]] = None,
    max_tokens: int = 50,
    max_experts: int = 32,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    dpi: int = 100
) -> plt.Figure:
    """
    Plot routing weights as a heatmap showing expert selection per token.

    Visualizes which experts are selected for each token with color intensity
    representing routing weight.

    Args:
        routing_weights: Routing weights [seq_len, num_experts]
        token_strs: Token strings for y-axis labels
        expert_indices: Optional expert indices to show only selected experts
        max_tokens: Maximum tokens to display (truncate if exceeded)
        max_experts: Maximum experts to display
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
        dpi: Resolution for display/saving

    Returns:
        matplotlib Figure object

    Example:
        >>> routing_weights = torch.rand(20, 64) * 0.3  # 20 tokens, 64 experts
        >>> tokens = ["The", "cat", "sat", "on", "the", "mat"] * 3 + [".", "."]
        >>> fig = plot_routing_heatmap(routing_weights, tokens)
        >>> plt.show()

    Raises:
        ValueError: If routing_weights is empty or shape mismatch
    """
    # Convert to numpy
    weights = _to_numpy(routing_weights)

    if weights.ndim != 2:
        raise ValueError(f"routing_weights must be 2D, got shape {weights.shape}")

    seq_len, num_experts = weights.shape

    if len(token_strs) != seq_len:
        warnings.warn(f"token_strs length ({len(token_strs)}) != seq_len ({seq_len}). Truncating/padding.")
        if len(token_strs) < seq_len:
            token_strs = token_strs + [''] * (seq_len - len(token_strs))
        else:
            token_strs = token_strs[:seq_len]

    # Truncate if needed
    if seq_len > max_tokens:
        weights = weights[:max_tokens, :]
        token_strs = token_strs[:max_tokens]
        seq_len = max_tokens

    if num_experts > max_experts:
        # Show first max_experts
        weights = weights[:, :max_experts]
        num_experts = max_experts

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot heatmap
    im = ax.imshow(weights, aspect='auto', cmap='YlOrRd',
                   interpolation='nearest', vmin=0, vmax=weights.max())

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Routing Weight', rotation=270, labelpad=20, fontweight='bold')

    # Set ticks
    ax.set_yticks(np.arange(seq_len))
    ax.set_yticklabels(token_strs, fontsize=9)

    if num_experts <= 32:
        ax.set_xticks(np.arange(num_experts))
        ax.set_xticklabels(range(num_experts), fontsize=8)
    else:
        # Show fewer ticks for many experts
        tick_positions = np.linspace(0, num_experts - 1, 8, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_positions, fontsize=8)

    # Labels
    ax.set_xlabel('Expert ID', fontweight='bold')
    ax.set_ylabel('Tokens', fontweight='bold')
    ax.set_title('Routing Weights Heatmap', fontweight='bold')

    # Grid
    ax.set_xticks(np.arange(num_experts) - 0.5, minor=True)
    ax.set_yticks(np.arange(seq_len) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.2)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi if dpi > 100 else 300, bbox_inches='tight')

    return fig


def plot_expert_utilization(
    expert_counts_by_expert: Union[torch.Tensor, np.ndarray, List],
    num_experts: Optional[int] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
    dpi: int = 100
) -> plt.Figure:
    """
    Plot expert utilization showing load distribution across experts.

    Bar chart showing how often each expert is used, highlighting load imbalance.

    Args:
        expert_counts_by_expert: Usage count for each expert [num_experts]
        num_experts: Total number of experts (for padding if needed)
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
        dpi: Resolution for display/saving

    Returns:
        matplotlib Figure object

    Example:
        >>> expert_usage = np.random.poisson(50, 64)  # 64 experts
        >>> fig = plot_expert_utilization(expert_usage)
        >>> plt.show()

    Raises:
        ValueError: If expert_counts_by_expert is empty
    """
    # Convert to numpy
    counts = _to_numpy(expert_counts_by_expert).flatten()

    if len(counts) == 0:
        raise ValueError("expert_counts_by_expert is empty")

    # Pad if num_experts specified
    if num_experts is not None and len(counts) < num_experts:
        padded = np.zeros(num_experts)
        padded[:len(counts)] = counts
        counts = padded

    num_exp = len(counts)

    # Compute statistics
    mean_usage = counts.mean()
    std_usage = counts.std()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Color bars based on usage (relative to mean)
    colors = ['red' if c < mean_usage * 0.5 else
              'orange' if c < mean_usage * 0.8 else
              'lightgreen' if c < mean_usage * 1.2 else
              'green' for c in counts]

    # Plot bars
    bars = ax.bar(range(num_exp), counts, color=colors, alpha=0.7, edgecolor='black')

    # Add mean line
    ax.axhline(y=mean_usage, color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {mean_usage:.1f}')

    # Add ±1 std bands
    ax.axhline(y=mean_usage + std_usage, color='blue', linestyle=':',
               linewidth=1, alpha=0.5)
    ax.axhline(y=mean_usage - std_usage, color='blue', linestyle=':',
               linewidth=1, alpha=0.5)
    ax.fill_between(range(num_exp), mean_usage - std_usage, mean_usage + std_usage,
                     alpha=0.1, color='blue', label='±1 Std Dev')

    # Labels
    ax.set_xlabel('Expert ID', fontweight='bold')
    ax.set_ylabel('Usage Count', fontweight='bold')
    ax.set_title('Expert Utilization Distribution', fontweight='bold')

    # Compute load balance metrics
    cv = std_usage / mean_usage if mean_usage > 0 else 0  # Coefficient of variation
    max_min_ratio = counts.max() / counts.min() if counts.min() > 0 else float('inf')

    # Add statistics text box
    stats_text = f'Mean: {mean_usage:.1f}\nStd: {std_usage:.1f}\nCV: {cv:.3f}\nMax/Min: {max_min_ratio:.2f}'
    ax.text(0.98, 0.97, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

    # Format x-axis
    if num_exp <= 32:
        ax.set_xticks(range(num_exp))
        ax.set_xticklabels(range(num_exp), fontsize=8)
    else:
        # Show fewer ticks
        tick_positions = np.linspace(0, num_exp - 1, 16, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_positions, fontsize=8)

    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi if dpi > 100 else 300, bbox_inches='tight')

    return fig


def plot_token_complexity_vs_experts(
    token_ids: Union[torch.Tensor, np.ndarray, List],
    expert_counts: Union[torch.Tensor, np.ndarray, List],
    tokenizer: Optional[Any] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 100
) -> plt.Figure:
    """
    Plot token complexity (rarity) vs number of experts selected.

    Scatter plot showing relationship between token frequency and expert usage.
    Hypothesis: Rare/complex tokens use more experts.

    Args:
        token_ids: Token IDs [seq_len]
        expert_counts: Expert counts per token [seq_len]
        tokenizer: Optional tokenizer to get token strings
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
        dpi: Resolution for display/saving

    Returns:
        matplotlib Figure object

    Example:
        >>> token_ids = [101, 2054, 2003, 1996, 3007]  # [CLS], what, is, the, capital
        >>> expert_counts = [4, 5, 3, 3, 6]
        >>> fig = plot_token_complexity_vs_experts(token_ids, expert_counts)
        >>> plt.show()

    Raises:
        ValueError: If token_ids and expert_counts have different lengths
    """
    # Convert to numpy
    toks = _to_numpy(token_ids).flatten()
    counts = _to_numpy(expert_counts).flatten()

    if len(toks) != len(counts):
        raise ValueError(f"token_ids ({len(toks)}) and expert_counts ({len(counts)}) must have same length")

    if len(toks) == 0:
        raise ValueError("token_ids is empty")

    # Estimate token frequency (inverse of token ID as proxy)
    # Lower token IDs are typically more common
    # This is a heuristic; ideally use actual corpus frequency
    token_frequency = 1.0 / (toks + 1)  # Add 1 to avoid division by zero

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Scatter plot with alpha for overlapping points
    scatter = ax.scatter(token_frequency, counts, alpha=0.5, s=50,
                        c=counts, cmap='viridis', edgecolors='black', linewidth=0.5)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Experts Selected', rotation=270, labelpad=20, fontweight='bold')

    # Fit trend line
    if len(toks) > 2:
        z = np.polyfit(token_frequency, counts, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(token_frequency.min(), token_frequency.max(), 100)
        ax.plot(x_trend, p(x_trend), 'r--', linewidth=2,
               label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')

    # Labels
    ax.set_xlabel('Token Frequency (Proxy: 1/(ID+1))', fontweight='bold')
    ax.set_ylabel('Experts Selected', fontweight='bold')
    ax.set_title('Token Complexity vs Expert Selection', fontweight='bold')

    # Add annotation
    if tokenizer is not None:
        # Annotate a few interesting points (highest/lowest expert counts)
        top_indices = np.argsort(counts)[-3:]  # Top 3 expert counts
        for idx in top_indices:
            if idx < len(toks):
                try:
                    token_str = tokenizer.decode([int(toks[idx])])
                    ax.annotate(token_str,
                              (token_frequency[idx], counts[idx]),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, alpha=0.7)
                except:
                    pass

    # Compute correlation
    if len(toks) > 1:
        corr = np.corrcoef(token_frequency, counts)[0, 1]
        ax.text(0.02, 0.98, f'Correlation: {corr:.3f}',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)

    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi if dpi > 100 else 300, bbox_inches='tight')

    return fig


def create_comparison_table(
    results_dict: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    output_format: str = 'dataframe'
) -> Union[pd.DataFrame, str]:
    """
    Create comparison table from multiple routing methods.

    Formats results into a pandas DataFrame or markdown/LaTeX table.

    Args:
        results_dict: Dict mapping method name to metrics dict
            Example: {
                'TopK': {'avg_experts': 8.0, 'std': 0.0, 'min': 8, 'max': 8},
                'BH': {'avg_experts': 4.5, 'std': 0.8, 'min': 3, 'max': 7}
            }
        metrics: List of metric names to include (None = all)
        output_format: 'dataframe', 'markdown', or 'latex'

    Returns:
        pandas DataFrame or formatted string

    Example:
        >>> results = {
        ...     'TopK': {'avg_experts': 8.0, 'std': 0.0, 'perplexity': 12.5},
        ...     'BH (α=0.05)': {'avg_experts': 4.5, 'std': 0.8, 'perplexity': 12.7}
        ... }
        >>> df = create_comparison_table(results)
        >>> print(df)

    Raises:
        ValueError: If results_dict is empty or invalid format
    """
    if not results_dict:
        raise ValueError("results_dict is empty")

    # Create DataFrame
    df = pd.DataFrame(results_dict).T

    # Select metrics if specified
    if metrics is not None:
        available_metrics = [m for m in metrics if m in df.columns]
        if not available_metrics:
            warnings.warn(f"None of requested metrics {metrics} found in data. Using all.")
        else:
            df = df[available_metrics]

    # Sort columns for consistency
    df = df.reindex(sorted(df.columns), axis=1)

    # Format numeric values
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32]:
            df[col] = df[col].apply(lambda x: f'{x:.2f}')

    # Return based on format
    if output_format == 'dataframe':
        return df
    elif output_format == 'markdown':
        return df.to_markdown()
    elif output_format == 'latex':
        return df.to_latex()
    else:
        raise ValueError(f"Unknown output_format: {output_format}. Use 'dataframe', 'markdown', or 'latex'.")


def plot_layer_wise_routing(
    layer_expert_counts: Union[torch.Tensor, np.ndarray, List[np.ndarray]],
    layer_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
    dpi: int = 100
) -> plt.Figure:
    """
    Plot layer-wise routing patterns showing distribution per layer.

    Box plot showing expert count distribution across different layers,
    revealing early vs middle vs late layer routing patterns.

    Args:
        layer_expert_counts: Expert counts per layer
            - Tensor/array of shape [num_layers, num_tokens], OR
            - List of arrays with varying token counts per layer
        layer_names: Optional layer names for x-axis
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
        dpi: Resolution for display/saving

    Returns:
        matplotlib Figure object

    Example:
        >>> # 16 layers, 100 tokens each
        >>> layer_counts = np.random.randint(3, 8, (16, 100))
        >>> fig = plot_layer_wise_routing(layer_counts)
        >>> plt.show()

    Raises:
        ValueError: If layer_expert_counts is empty
    """
    # Handle different input formats
    if isinstance(layer_expert_counts, (torch.Tensor, np.ndarray)):
        counts = _to_numpy(layer_expert_counts)
        if counts.ndim == 1:
            # Single layer
            counts = counts.reshape(1, -1)
        num_layers = counts.shape[0]
        data_list = [counts[i, :] for i in range(num_layers)]
    elif isinstance(layer_expert_counts, list):
        data_list = [_to_numpy(arr).flatten() for arr in layer_expert_counts]
        num_layers = len(data_list)
    else:
        raise TypeError(f"Unsupported type: {type(layer_expert_counts)}")

    if num_layers == 0:
        raise ValueError("layer_expert_counts is empty")

    # Generate layer names if not provided
    if layer_names is None:
        layer_names = [f'L{i}' for i in range(num_layers)]
    elif len(layer_names) != num_layers:
        warnings.warn(f"layer_names length ({len(layer_names)}) != num_layers ({num_layers}). Using defaults.")
        layer_names = [f'L{i}' for i in range(num_layers)]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Create box plot
    bp = ax.boxplot(data_list, labels=layer_names, patch_artist=True,
                    notch=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=6))

    # Color boxes with gradient (early layers lighter, late layers darker)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, num_layers))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Compute mean per layer for trend line
    means = [np.mean(data) for data in data_list]
    ax.plot(range(1, num_layers + 1), means, 'r--', linewidth=2,
           marker='D', markersize=8, label='Mean Trend')

    # Labels
    ax.set_xlabel('Layer', fontweight='bold')
    ax.set_ylabel('Experts Selected', fontweight='bold')
    ax.set_title('Layer-wise Routing Patterns', fontweight='bold')

    # Rotate x-labels if many layers
    if num_layers > 10:
        plt.xticks(rotation=45, ha='right')

    # Compute statistics
    overall_mean = np.mean([np.mean(data) for data in data_list])
    early_mean = np.mean([np.mean(data) for data in data_list[:num_layers//3]])
    late_mean = np.mean([np.mean(data) for data in data_list[-num_layers//3:]])

    # Add statistics text box
    stats_text = f'Overall Mean: {overall_mean:.2f}\nEarly Layers: {early_mean:.2f}\nLate Layers: {late_mean:.2f}'
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi if dpi > 100 else 300, bbox_inches='tight')

    return fig


# Convenience function for batch plotting
def create_analysis_report(
    routing_data: Dict[str, Any],
    output_dir: str = './plots',
    dpi: int = 300
) -> Dict[str, plt.Figure]:
    """
    Create comprehensive analysis report with all visualizations.

    Convenience function that generates all relevant plots from routing data.

    Args:
        routing_data: Dictionary with routing analysis results
            Required keys vary by plot type. Example:
            {
                'expert_counts': [...],
                'method_name': 'BH Routing',
                'alpha': 0.05,
                'alphas': [...],
                'avg_experts_per_alpha': [...],
                # ... etc
            }
        output_dir: Directory to save plots
        dpi: Resolution for saved plots

    Returns:
        Dictionary mapping plot names to Figure objects

    Example:
        >>> data = {
        ...     'expert_counts': torch.randint(3, 8, (100,)),
        ...     'method_name': 'BH Routing',
        ...     'alpha': 0.05
        ... }
        >>> figs = create_analysis_report(data, './my_plots')
    """
    from pathlib import Path

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    figures = {}

    # 1. Expert count distribution
    if 'expert_counts' in routing_data:
        fig = plot_expert_count_distribution(
            routing_data['expert_counts'],
            method_name=routing_data.get('method_name', 'Unknown'),
            alpha=routing_data.get('alpha'),
            save_path=f"{output_dir}/expert_count_distribution.png",
            dpi=dpi
        )
        figures['expert_count_distribution'] = fig
        plt.close(fig)

    # 2. Alpha sensitivity
    if 'alphas' in routing_data and 'avg_experts_per_alpha' in routing_data:
        fig = plot_alpha_sensitivity(
            routing_data['alphas'],
            routing_data['avg_experts_per_alpha'],
            std_experts=routing_data.get('std_experts_per_alpha'),
            save_path=f"{output_dir}/alpha_sensitivity.png",
            dpi=dpi
        )
        figures['alpha_sensitivity'] = fig
        plt.close(fig)

    # 3. Routing heatmap
    if 'routing_weights' in routing_data and 'tokens' in routing_data:
        fig = plot_routing_heatmap(
            routing_data['routing_weights'],
            routing_data['tokens'],
            save_path=f"{output_dir}/routing_heatmap.png",
            dpi=dpi
        )
        figures['routing_heatmap'] = fig
        plt.close(fig)

    # 4. Expert utilization
    if 'expert_usage' in routing_data:
        fig = plot_expert_utilization(
            routing_data['expert_usage'],
            save_path=f"{output_dir}/expert_utilization.png",
            dpi=dpi
        )
        figures['expert_utilization'] = fig
        plt.close(fig)

    # 5. Layer-wise routing
    if 'layer_expert_counts' in routing_data:
        fig = plot_layer_wise_routing(
            routing_data['layer_expert_counts'],
            save_path=f"{output_dir}/layer_wise_routing.png",
            dpi=dpi
        )
        figures['layer_wise_routing'] = fig
        plt.close(fig)

    print(f"✅ Created {len(figures)} plots in {output_dir}/")

    return figures


if __name__ == "__main__":
    # Quick demonstration
    print("Routing Visualizations Module")
    print("=" * 70)
    print("\nThis module provides 7 visualization functions:")
    print("  1. plot_expert_count_distribution()")
    print("  2. plot_alpha_sensitivity()")
    print("  3. plot_routing_heatmap()")
    print("  4. plot_expert_utilization()")
    print("  5. plot_token_complexity_vs_experts()")
    print("  6. create_comparison_table()")
    print("  7. plot_layer_wise_routing()")
    print("\nPlus convenience function:")
    print("  - create_analysis_report()")
    print("\nRun test_visualizations.py for examples and tests.")
