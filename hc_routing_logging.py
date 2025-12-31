"""
Logging System for Higher Criticism Routing
============================================

Provides comprehensive logging of HC routing decisions including:
- P-values and their sorted order
- HC statistics at each rank
- Threshold selection (where HC peaked)
- Expert selection decisions

Mirrors BHRoutingLogger structure for consistency.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import defaultdict
import numpy as np


class HCRoutingLogger:
    """
    Logger for Higher Criticism routing decisions.

    Captures detailed information about each routing decision for
    analysis and debugging.

    Example:
        logger = HCRoutingLogger('./logs', 'experiment_1')
        # ... run inference with logger ...
        logger.save_logs()
        logger.generate_plots()
        summary = logger.get_summary()
    """

    def __init__(
        self,
        output_dir: str,
        experiment_name: str,
        log_every_n: int = 100
    ):
        """
        Initialize HC routing logger.

        Args:
            output_dir: Directory to save logs and plots
            experiment_name: Name for this experiment
            log_every_n: Log every N routing decisions (for efficiency)
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.log_every_n = log_every_n

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Storage for logged data
        self.routing_decisions: List[Dict[str, Any]] = []

        # Aggregated statistics per layer
        self.layer_stats: Dict[int, Dict[str, List]] = defaultdict(
            lambda: {
                'num_selected': [],
                'hc_threshold_rank': [],
                'hc_max_value': [],
                'floor_hits': 0,
                'ceiling_hits': 0,
                'total': 0,
            }
        )

        # Global counters
        self.total_decisions = 0
        self.logged_decisions = 0

        # Timestamp
        self.start_time = datetime.now()

    def log_routing_decision(self, log_entry: Dict[str, Any]):
        """
        Log a single routing decision with complete HC schema.

        Args:
            log_entry: Dict containing COMPLETE HC routing information:
                === IDENTIFICATION ===
                - sample_idx, token_idx, layer_idx: int

                === ROUTER OUTPUTS ===
                - router_logits: List[float] - raw logits [64]
                - router_logits_stats: Dict[str, float] - min/max/mean/std

                === P-VALUE COMPUTATION ===
                - kde_model_id: str - layer identifier
                - p_values: List[float] - p-values [64]
                - p_values_sorted: List[float] - sorted ascending
                - sort_indices: List[int] - mapping to original indices

                === HC STATISTICS (CRITICAL) ===
                - hc_statistics: List[float] - HC(i) for all 64 ranks
                - hc_max_rank: int - rank where HC peaks (1-indexed)
                - hc_max_value: float - maximum HC value
                - hc_positive_count: int - how many ranks have HC > 0
                - search_range: Dict - beta, start_rank, end_rank

                === SELECTION DECISION ===
                - num_selected: int
                - selected_experts: List[int]
                - routing_weights: List[float]
                - selection_reason: str

                === CONSTRAINT FLAGS ===
                - hit_min_k, hit_max_k, fallback_triggered: bool

                === VALIDATION ===
                - weights_sum: float
                - config: Dict - min_k, max_k, temperature
        """
        self.total_decisions += 1

        # Extract key fields
        layer_idx = log_entry.get('layer_idx', 0)
        num_selected = log_entry.get('num_selected', 0)
        config = log_entry.get('config', {})
        min_k = config.get('min_k', 1)
        max_k = config.get('max_k', 16)

        # Update layer statistics (always, even if not logging full entry)
        layer_stats = self.layer_stats[layer_idx]
        layer_stats['num_selected'].append(num_selected)
        layer_stats['total'] += 1

        # Track floor/ceiling hits
        if num_selected == min_k or log_entry.get('hit_min_k', False):
            layer_stats['floor_hits'] += 1
        if num_selected == max_k or log_entry.get('hit_max_k', False):
            layer_stats['ceiling_hits'] += 1

        # Track HC-specific statistics
        if 'hc_max_rank' in log_entry:
            layer_stats['hc_threshold_rank'].append(log_entry['hc_max_rank'])
        if 'hc_max_value' in log_entry:
            layer_stats['hc_max_value'].append(log_entry['hc_max_value'])

        # Log full entry periodically
        if self.total_decisions % self.log_every_n == 0:
            # Validate required fields before logging
            required_fields = [
                'sample_idx', 'token_idx', 'layer_idx', 'num_selected',
                'hc_statistics', 'hc_max_rank', 'hc_max_value'
            ]
            missing = [f for f in required_fields if f not in log_entry]
            if missing:
                print(f"⚠️ Warning: Log entry missing fields: {missing}")

            self.routing_decisions.append(log_entry)
            self.logged_decisions += 1

    def save_logs(self, filename: Optional[str] = None):
        """
        Save all logged routing decisions to JSON.

        Args:
            filename: Optional custom filename
        """
        if filename is None:
            filename = f"hc_routing_log_{self.experiment_name}.json"

        filepath = os.path.join(self.output_dir, filename)

        # Prepare data for JSON serialization
        output = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_decisions': self.total_decisions,
            'logged_decisions': self.logged_decisions,
            'log_every_n': self.log_every_n,
            'routing_decisions': self.routing_decisions,
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"✅ Saved HC routing logs to {filepath}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics across all logged decisions.

        Returns:
            Dict with summary statistics
        """
        summary = {
            'experiment_name': self.experiment_name,
            'total_decisions': self.total_decisions,
            'logged_decisions': self.logged_decisions,
            'per_layer': {},
        }

        # Aggregate across layers
        all_num_selected = []
        all_hc_thresholds = []
        all_hc_max = []
        total_floor = 0
        total_ceiling = 0

        for layer_idx, stats in self.layer_stats.items():
            if stats['total'] == 0:
                continue

            layer_summary = {
                'total': stats['total'],
                'avg_experts': np.mean(stats['num_selected']) if stats['num_selected'] else 0,
                'std_experts': np.std(stats['num_selected']) if stats['num_selected'] else 0,
                'floor_hit_rate': stats['floor_hits'] / stats['total'] * 100,
                'ceiling_hit_rate': stats['ceiling_hits'] / stats['total'] * 100,
            }

            if stats['hc_threshold_rank']:
                layer_summary['avg_hc_threshold'] = np.mean(stats['hc_threshold_rank'])
                layer_summary['std_hc_threshold'] = np.std(stats['hc_threshold_rank'])

            if stats['hc_max_value']:
                layer_summary['avg_hc_max'] = np.mean(stats['hc_max_value'])
                layer_summary['std_hc_max'] = np.std(stats['hc_max_value'])

            summary['per_layer'][layer_idx] = layer_summary

            # Accumulate for global stats
            all_num_selected.extend(stats['num_selected'])
            all_hc_thresholds.extend(stats['hc_threshold_rank'])
            all_hc_max.extend(stats['hc_max_value'])
            total_floor += stats['floor_hits']
            total_ceiling += stats['ceiling_hits']

        # Global statistics
        if all_num_selected:
            summary['global'] = {
                'avg_experts': np.mean(all_num_selected),
                'std_experts': np.std(all_num_selected),
                'min_experts': min(all_num_selected),
                'max_experts': max(all_num_selected),
                'floor_hit_rate': total_floor / len(all_num_selected) * 100,
                'ceiling_hit_rate': total_ceiling / len(all_num_selected) * 100,
            }

            if all_hc_thresholds:
                summary['global']['avg_hc_threshold'] = np.mean(all_hc_thresholds)
            if all_hc_max:
                summary['global']['avg_hc_max'] = np.mean(all_hc_max)

        return summary

    def generate_plots(self, save: bool = True):
        """
        Generate HC-specific visualization plots.

        Plots:
        1. HC statistic vs rank (showing threshold selection)
        2. Expert count distribution across layers
        3. HC threshold rank distribution
        4. Floor/ceiling hit rates by layer
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️ matplotlib not available, skipping plots")
            return

        if not self.routing_decisions:
            print("⚠️ No logged decisions to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'HC Routing Analysis: {self.experiment_name}', fontsize=14)

        # Plot 1: HC statistic vs rank for sample decisions
        ax1 = axes[0, 0]
        sample_entries = self.routing_decisions[:5]  # First 5 logged entries
        for entry in sample_entries:
            if 'hc_statistics' in entry:
                hc_stats = entry['hc_statistics']
                threshold = entry.get('hc_threshold_rank', 0)
                ax1.plot(range(1, len(hc_stats)+1), hc_stats, alpha=0.6)
                ax1.axvline(x=threshold+1, linestyle='--', alpha=0.3)
        ax1.set_xlabel('Rank')
        ax1.set_ylabel('HC Statistic')
        ax1.set_title('HC Statistic vs Rank (Sample Decisions)')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_xlim(1, 20)

        # Plot 2: Expert count distribution
        ax2 = axes[0, 1]
        all_counts = []
        for stats in self.layer_stats.values():
            all_counts.extend(stats['num_selected'])
        if all_counts:
            ax2.hist(all_counts, bins=range(1, max(all_counts)+2), edgecolor='black', alpha=0.7)
            ax2.axvline(x=np.mean(all_counts), color='red', linestyle='--',
                       label=f'Mean: {np.mean(all_counts):.1f}')
            ax2.set_xlabel('Number of Experts Selected')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Expert Count Distribution')
            ax2.legend()

        # Plot 3: HC threshold rank distribution
        ax3 = axes[1, 0]
        all_thresholds = []
        for stats in self.layer_stats.values():
            all_thresholds.extend(stats['hc_threshold_rank'])
        if all_thresholds:
            ax3.hist(all_thresholds, bins=range(0, max(all_thresholds)+2),
                    edgecolor='black', alpha=0.7, color='orange')
            ax3.axvline(x=np.mean(all_thresholds), color='red', linestyle='--',
                       label=f'Mean: {np.mean(all_thresholds):.1f}')
            ax3.set_xlabel('HC Threshold Rank')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Where HC Peaked (Threshold Rank)')
            ax3.legend()

        # Plot 4: Floor/ceiling rates by layer
        ax4 = axes[1, 1]
        layers = sorted(self.layer_stats.keys())
        floor_rates = []
        ceiling_rates = []
        for layer in layers:
            stats = self.layer_stats[layer]
            if stats['total'] > 0:
                floor_rates.append(stats['floor_hits'] / stats['total'] * 100)
                ceiling_rates.append(stats['ceiling_hits'] / stats['total'] * 100)
            else:
                floor_rates.append(0)
                ceiling_rates.append(0)

        x = np.arange(len(layers))
        width = 0.35
        ax4.bar(x - width/2, floor_rates, width, label='Floor Hits', color='red', alpha=0.7)
        ax4.bar(x + width/2, ceiling_rates, width, label='Ceiling Hits', color='blue', alpha=0.7)
        ax4.set_xlabel('Layer')
        ax4.set_ylabel('Hit Rate (%)')
        ax4.set_title('Floor/Ceiling Hit Rates by Layer')
        ax4.set_xticks(x)
        ax4.set_xticklabels(layers)
        ax4.legend()

        plt.tight_layout()

        if save:
            plot_path = os.path.join(self.output_dir, f'hc_routing_plots_{self.experiment_name}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved HC routing plots to {plot_path}")

        plt.close()

    def clear(self):
        """Clear all logged data for reuse."""
        self.routing_decisions.clear()
        self.layer_stats.clear()
        self.total_decisions = 0
        self.logged_decisions = 0
        self.start_time = datetime.now()
