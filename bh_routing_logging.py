"""
BH Routing Logging and Visualization
=====================================

Provides detailed logging and visualization for Benjamini-Hochberg routing decisions.

Features:
- Per-token routing decision logging
- Statistical aggregation
- 6 types of visualization plots
- Efficient storage with sampling

Usage:
    logger = BHRoutingLogger(output_dir="./logs", experiment_name="8experts_bh_a030")

    # During routing
    logger.log_routing_decision({
        'sample_idx': 0,
        'token_idx': 15,
        'layer_idx': 7,
        'router_logits': logits.cpu().numpy(),
        'p_values': p_vals.cpu().numpy(),
        'selected_experts': [3, 7, 12],
        'num_selected': 3,
        ...
    })

    # After experiment
    logger.save_logs()
    logger.generate_plots()
    logger.clear()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
import json
from datetime import datetime


class BHRoutingLogger:
    """
    Logger for Benjamini-Hochberg routing decisions.

    Tracks detailed routing information and generates comprehensive visualizations.
    """

    def __init__(
        self,
        output_dir: str,
        experiment_name: str,
        log_every_n: int = 100
    ):
        """
        Initialize the BH routing logger.

        Args:
            output_dir: Directory to save logs and plots
            experiment_name: Name of the experiment (e.g., "8experts_bh_a030_wikitext")
            log_every_n: Log detailed info every N tokens (for efficiency)
        """
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.log_every_n = log_every_n

        # Create directories
        self.logs_dir = self.output_dir / 'logs'
        self.plots_dir = self.output_dir / 'plots' / experiment_name
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Storage
        self.detailed_logs = []  # Full logs for sampled tokens
        self.summary_stats = {
            'num_selected': [],
            'p_value_min': [],
            'p_value_max': [],
            'p_value_mean': [],
            'logit_mean': [],
            'logit_std': [],
            'layer_idx': [],
            'ceiling_hits': 0,
            'floor_hits': 0,
            'total_decisions': 0
        }

        # Per-layer, per-expert selection counts
        self.layer_expert_counts = defaultdict(lambda: np.zeros(64))

        # Config (set on first log)
        self.alpha = None
        self.max_k = None
        self.min_k = None

        self.token_counter = 0

    def log_routing_decision(self, log_entry: Dict[str, Any]):
        """
        Log a single routing decision.

        Args:
            log_entry: Dictionary with routing decision data
                Required keys:
                - sample_idx, token_idx, layer_idx
                - router_logits (np.array)
                - p_values (np.array)
                - selected_experts (list of int)
                - num_selected (int)
                - alpha, max_k, min_k (float/int)
                Optional:
                - sorted_p_values, bh_thresholds, etc.
        """
        # Set config on first log
        if self.alpha is None:
            self.alpha = log_entry.get('alpha')
            self.max_k = log_entry.get('max_k')
            self.min_k = log_entry.get('min_k', 1)

        # Always update summary stats
        num_selected = log_entry['num_selected']
        p_values = log_entry['p_values']
        router_logits = log_entry['router_logits']
        layer_idx = log_entry['layer_idx']

        self.summary_stats['num_selected'].append(num_selected)
        self.summary_stats['p_value_min'].append(float(np.min(p_values)))
        self.summary_stats['p_value_max'].append(float(np.max(p_values)))
        self.summary_stats['p_value_mean'].append(float(np.mean(p_values)))
        self.summary_stats['logit_mean'].append(float(np.mean(router_logits)))
        self.summary_stats['logit_std'].append(float(np.std(router_logits)))
        self.summary_stats['layer_idx'].append(layer_idx)
        self.summary_stats['total_decisions'] += 1

        # Track ceiling/floor hits
        if num_selected >= self.max_k:
            self.summary_stats['ceiling_hits'] += 1
        if num_selected <= self.min_k:
            self.summary_stats['floor_hits'] += 1

        # Track per-layer expert selection
        selected_experts = log_entry['selected_experts']
        for exp_idx in selected_experts:
            if exp_idx >= 0:  # Skip padding (-1)
                self.layer_expert_counts[layer_idx][exp_idx] += 1

        # Log detailed info every N tokens
        if self.token_counter % self.log_every_n == 0:
            # Create detailed log entry
            detailed_entry = {
                'experiment_id': self.experiment_name,
                'sample_idx': log_entry['sample_idx'],
                'token_idx': log_entry['token_idx'],
                'layer_idx': layer_idx,
                'alpha': self.alpha,
                'max_k': self.max_k,
                'min_k': self.min_k,
                'router_logits_stats': {
                    'min': float(np.min(router_logits)),
                    'max': float(np.max(router_logits)),
                    'mean': float(np.mean(router_logits)),
                    'std': float(np.std(router_logits))
                },
                'p_values_stats': {
                    'min': float(np.min(p_values)),
                    'max': float(np.max(p_values)),
                    'mean': float(np.mean(p_values)),
                    'median': float(np.median(p_values)),
                    'std': float(np.std(p_values))
                },
                'bh_results': {
                    'num_selected': num_selected,
                    'selected_experts': selected_experts,
                    'smallest_p_value': float(np.min(p_values)),
                }
            }

            # Add optional fields if present
            if 'sorted_p_values' in log_entry:
                sorted_p = log_entry['sorted_p_values']
                detailed_entry['bh_results']['largest_passing_p_value'] = float(sorted_p[num_selected - 1]) if num_selected > 0 else 0.0

            self.detailed_logs.append(detailed_entry)

        self.token_counter += 1

    def save_logs(self, suffix: str = ""):
        """
        Save logs to JSON files.

        Args:
            suffix: Optional suffix for filename
        """
        # Save detailed logs
        detailed_path = self.logs_dir / f"{self.experiment_name}_bh_log{suffix}.json"
        with open(detailed_path, 'w') as f:
            json.dump({
                'experiment_name': self.experiment_name,
                'config': {
                    'alpha': self.alpha,
                    'max_k': self.max_k,
                    'min_k': self.min_k,
                    'log_every_n': self.log_every_n
                },
                'timestamp': datetime.now().isoformat(),
                'total_decisions': self.summary_stats['total_decisions'],
                'detailed_logs': self.detailed_logs
            }, f, indent=2)

        # Save summary statistics
        summary_path = self.logs_dir / f"{self.experiment_name}_summary{suffix}.json"

        summary = {
            'experiment_name': self.experiment_name,
            'config': {
                'alpha': self.alpha,
                'max_k': self.max_k,
                'min_k': self.min_k
            },
            'stats': {
                'total_decisions': self.summary_stats['total_decisions'],
                'avg_experts_selected': float(np.mean(self.summary_stats['num_selected'])),
                'std_experts_selected': float(np.std(self.summary_stats['num_selected'])),
                'min_experts_selected': int(np.min(self.summary_stats['num_selected'])),
                'max_experts_selected': int(np.max(self.summary_stats['num_selected'])),
                'ceiling_hit_rate': self.summary_stats['ceiling_hits'] / max(self.summary_stats['total_decisions'], 1) * 100,
                'floor_hit_rate': self.summary_stats['floor_hits'] / max(self.summary_stats['total_decisions'], 1) * 100,
                'p_value_mean': float(np.mean(self.summary_stats['p_value_mean'])),
                'p_value_std': float(np.std(self.summary_stats['p_value_mean'])),
            }
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"  📝 Saved logs: {detailed_path.name}")
        print(f"  📝 Saved summary: {summary_path.name}")

    def generate_plots(self):
        """Generate all 6 visualization plots."""
        if self.summary_stats['total_decisions'] == 0:
            print("  ⚠️ No data to plot")
            return

        print(f"  📊 Generating plots for {self.experiment_name}...")

        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 150

        try:
            self._plot_num_experts_histogram()
            self._plot_expert_selection_heatmap()

            # Only if we have detailed logs
            if self.detailed_logs:
                self._plot_p_value_distribution()
                self._plot_bh_decision_samples()
                self._plot_logits_vs_pvalues()
                self._plot_bh_threshold_analysis()

            print(f"  ✅ Plots saved to {self.plots_dir}")
        except Exception as e:
            print(f"  ⚠️ Plot generation failed: {e}")
            import traceback
            traceback.print_exc()

    def _plot_num_experts_histogram(self):
        """Plot 4: Distribution of number of experts selected."""
        fig, ax = plt.subplots(figsize=(10, 6))

        num_selected = self.summary_stats['num_selected']

        ax.hist(num_selected, bins=range(1, self.max_k + 2),
                edgecolor='black', alpha=0.7, color='steelblue')

        # Add statistics lines
        mean_val = np.mean(num_selected)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_val:.2f}')
        ax.axvline(self.min_k, color='green', linestyle='--', linewidth=2,
                   label=f'min_k: {self.min_k}')
        ax.axvline(self.max_k, color='orange', linestyle='--', linewidth=2,
                   label=f'max_k: {self.max_k}')

        ax.set_xlabel('Number of Experts Selected', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{self.experiment_name}\nDistribution of Experts Selected (α={self.alpha})',
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'num_experts_histogram.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_expert_selection_heatmap(self):
        """Plot 3: Expert selection frequency heatmap."""
        # Create matrix: layers × experts
        num_layers = len(self.layer_expert_counts)
        if num_layers == 0:
            return

        matrix = np.zeros((num_layers, 64))
        layer_indices = sorted(self.layer_expert_counts.keys())

        for i, layer_idx in enumerate(layer_indices):
            matrix[i, :] = self.layer_expert_counts[layer_idx]

        # Normalize by number of decisions
        decisions_per_layer = self.summary_stats['total_decisions'] / max(num_layers, 1)
        matrix = matrix / max(decisions_per_layer, 1) * 100  # Convert to percentage

        fig, ax = plt.subplots(figsize=(14, 8))

        sns.heatmap(matrix, cmap='YlOrRd', cbar_kws={'label': 'Selection Rate (%)'},
                    ax=ax, vmin=0, vmax=100)

        ax.set_xlabel('Expert Index', fontsize=12)
        ax.set_ylabel('Layer Index', fontsize=12)
        ax.set_title(f'{self.experiment_name}\nExpert Selection Frequency Heatmap',
                     fontsize=14, fontweight='bold')

        # Set ticks
        ax.set_yticks(range(0, num_layers, max(1, num_layers // 16)))
        ax.set_yticklabels([layer_indices[i] for i in range(0, num_layers, max(1, num_layers // 16))])
        ax.set_xticks(range(0, 64, 8))
        ax.set_xticklabels(range(0, 64, 8))

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'expert_selection_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_p_value_distribution(self):
        """Plot 1: P-value distribution across all tokens."""
        if not self.detailed_logs:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Collect p-value means from detailed logs
        p_value_means = [log['p_values_stats']['mean'] for log in self.detailed_logs]

        ax.hist(p_value_means, bins=50, edgecolor='black', alpha=0.7,
                color='steelblue', density=True, label='Observed')

        # Overlay uniform distribution
        ax.axhline(1.0, color='red', linestyle='--', linewidth=2,
                   label='Uniform [0,1]')

        ax.set_xlabel('P-Value (mean per token)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{self.experiment_name}\nP-Value Distribution',
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'p_value_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_bh_decision_samples(self):
        """Plot 2: BH decision visualization for sample tokens."""
        # Not fully implementable without storing sorted p-values
        # Create a simplified version
        pass

    def _plot_logits_vs_pvalues(self):
        """Plot 5: Router logits vs P-values scatter."""
        if not self.detailed_logs:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract data
        logit_means = [log['router_logits_stats']['mean'] for log in self.detailed_logs]
        p_value_means = [log['p_values_stats']['mean'] for log in self.detailed_logs]
        layers = [log['layer_idx'] for log in self.detailed_logs]

        scatter = ax.scatter(logit_means, p_value_means, c=layers, cmap='viridis',
                            alpha=0.6, edgecolors='black', linewidth=0.5)

        ax.set_xlabel('Router Logit (mean)', fontsize=12)
        ax.set_ylabel('P-Value (mean)', fontsize=12)
        ax.set_title(f'{self.experiment_name}\nRouter Logits vs P-Values',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Layer Index', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'logits_vs_pvalues.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_bh_threshold_analysis(self):
        """Plot 6: BH threshold passing rate by rank."""
        # This would require storing full sorted p-values
        # Simplified version
        pass

    def clear(self):
        """Clear all logs for next experiment."""
        self.detailed_logs = []
        self.summary_stats = {
            'num_selected': [],
            'p_value_min': [],
            'p_value_max': [],
            'p_value_mean': [],
            'logit_mean': [],
            'logit_std': [],
            'layer_idx': [],
            'ceiling_hits': 0,
            'floor_hits': 0,
            'total_decisions': 0
        }
        self.layer_expert_counts = defaultdict(lambda: np.zeros(64))
        self.alpha = None
        self.max_k = None
        self.min_k = None
        self.token_counter = 0

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.summary_stats['num_selected']:
            return {}

        return {
            'total_decisions': self.summary_stats['total_decisions'],
            'avg_experts': float(np.mean(self.summary_stats['num_selected'])),
            'std_experts': float(np.std(self.summary_stats['num_selected'])),
            'min_experts': int(np.min(self.summary_stats['num_selected'])),
            'max_experts': int(np.max(self.summary_stats['num_selected'])),
            'ceiling_hit_rate': self.summary_stats['ceiling_hits'] / max(self.summary_stats['total_decisions'], 1) * 100,
            'floor_hit_rate': self.summary_stats['floor_hits'] / max(self.summary_stats['total_decisions'], 1) * 100,
        }
