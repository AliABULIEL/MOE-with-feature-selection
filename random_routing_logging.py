"""
Logging System for Random Routing
============================================

Provides logging for random routing decisions, including:
- The number of experts selected (`experts_amount`).
- The target sum of weights (`sum_of_weights`).
- The indices of the randomly selected experts.
- The final routing weights.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import defaultdict
import numpy as np

class RandomRoutingLogger:
    """
    Logger for Random routing decisions.
    """

    def __init__(
        self,
        output_dir: str,
        experiment_name: str,
        log_every_n: int = 100
    ):
        """
        Initialize random routing logger.

        Args:
            output_dir: Directory to save logs.
            experiment_name: Name for this experiment.
            log_every_n: Log every N routing decisions.
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.log_every_n = log_every_n

        os.makedirs(output_dir, exist_ok=True)

        self.routing_decisions: List[Dict[str, Any]] = []
        self.layer_stats: Dict[int, Dict[str, List]] = defaultdict(
            lambda: {'num_selected': [], 'weights_sum': []}
        )
        self.total_decisions = 0
        self.logged_decisions = 0
        self.start_time = datetime.now()

    def log_routing_decision(self, log_entry: Dict[str, Any]):
        """
        Log a single routing decision.

        Args:
            log_entry: Dict containing random routing information.
        """
        self.total_decisions += 1
        
        layer_idx = log_entry.get('layer_idx', 0)
        self.layer_stats[layer_idx]['num_selected'].append(log_entry.get('experts_amount', 0))
        self.layer_stats[layer_idx]['weights_sum'].append(log_entry.get('weights_sum', 0))

        if self.total_decisions % self.log_every_n == 0:
            self.routing_decisions.append(log_entry)
            self.logged_decisions += 1

    def save_logs(self, filename: Optional[str] = None):
        """
        Save all logged routing decisions to JSON.
        """
        if filename is None:
            filename = f"random_routing_log_{self.experiment_name}.json"

        filepath = os.path.join(self.output_dir, filename)

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

        print(f"✅ Saved random routing logs to {filepath}")
        return filepath

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics across all logged decisions.
        """
        summary = {
            'experiment_name': self.experiment_name,
            'total_decisions': self.total_decisions,
            'logged_decisions': self.logged_decisions,
            'per_layer': {},
        }

        all_num_selected = []
        all_weights_sum = []

        for layer_idx, stats in self.layer_stats.items():
            if not stats['num_selected']:
                continue

            summary['per_layer'][layer_idx] = {
                'avg_experts': np.mean(stats['num_selected']),
                'avg_weights_sum': np.mean(stats['weights_sum']),
            }
            all_num_selected.extend(stats['num_selected'])
            all_weights_sum.extend(stats['weights_sum'])

        if all_num_selected:
            summary['global'] = {
                'avg_experts': np.mean(all_num_selected),
                'avg_weights_sum': np.mean(all_weights_sum),
            }

        return summary

    def clear(self):
        """Clear all logged data for reuse."""
        self.routing_decisions.clear()
        self.layer_stats.clear()
        self.total_decisions = 0
        self.logged_decisions = 0
        self.start_time = datetime.now()
