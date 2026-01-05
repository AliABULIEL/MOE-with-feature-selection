"""
Metrics Computation for Random Routing
=================================================

Computes quality and efficiency metrics for random routing.
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field

@dataclass
class RandomMetrics:
    """
    Container for Random routing metrics.
    """
    avg_experts: float = 0.0
    sum_of_weights: Optional[float] = None
    expert_utilization: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)

class RandomMetricsComputer:
    """
    Compute comprehensive metrics for Random routing.
    """

    def __init__(self, num_experts: int = 64):
        """
        Initialize metrics computer.
        Args:
            num_experts: Total number of experts (default: 64)
        """
        self.num_experts = num_experts

    def compute(
        self,
        expert_counts: torch.Tensor,
        routing_weights: Optional[torch.Tensor] = None,
        sum_of_weights: Optional[float] = None,
    ) -> RandomMetrics:
        """
        Compute all Random routing metrics.
        """
        metrics = RandomMetrics(sum_of_weights=sum_of_weights)
        counts = expert_counts.float().cpu()
        
        metrics.avg_experts = counts.mean().item()

        if routing_weights is not None:
            weights = routing_weights.float().cpu()
            ever_selected = (weights > 0.01).any(dim=0)
            metrics.expert_utilization = ever_selected.sum().item() / self.num_experts

        return metrics

@dataclass
class UnifiedEvaluationMetrics:
    """
    Unified metrics dataclass for comprehensive random routing evaluation.
    """
    dataset_name: str
    routing_type: str = 'random'
    config_name: str
    experts_amount: int
    sum_of_weights: Optional[float] = None
    
    perplexity: Optional[float] = None
    accuracy: Optional[float] = None
    
    avg_experts: float = 0.0
    avg_weight_sum: float = 0.0
    expert_utilization: float = 0.0
    
    evaluation_time_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_dataframe_row(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])

    def save_json(self, filepath: str):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✅ Saved metrics to: {filepath}")

    @classmethod
    def load_json(cls, filepath: str) -> 'UnifiedEvaluationMetrics':
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)

def save_metrics(
    metrics: UnifiedEvaluationMetrics,
    output_dir: str = './metrics',
    experiment_name: str = 'experiment',
    append_to_cumulative: bool = True
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = output_dir / f'{experiment_name}_metrics_{timestamp}.json'
    metrics.save_json(json_path)

    if append_to_cumulative:
        cumulative_csv = output_dir / 'all_experiments_metrics.csv'
        df = metrics.to_dataframe_row()
        if cumulative_csv.exists():
            existing_df = pd.read_csv(cumulative_csv)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_csv(cumulative_csv, index=False)
        else:
            df.to_csv(cumulative_csv, index=False)
        print(f"✅ Appended to cumulative CSV: {cumulative_csv}")

    print("\n" + "="*70)
    print(f"METRICS SUMMARY: {experiment_name}")
    print("="*70)
    summary_data = {
        'Perplexity': f"{metrics.perplexity:.2f}" if metrics.perplexity is not None else 'N/A',
        'Accuracy': f"{metrics.accuracy:.4f}" if metrics.accuracy is not None else 'N/A',
        'Avg Experts': f"{metrics.avg_experts:.2f}",
        'Avg Weight Sum': f"{metrics.avg_weight_sum:.4f}",
        'Expert Utilization': f"{metrics.expert_utilization:.2f}",
    }
    max_key_len = max(len(k) for k in summary_data.keys())
    for key, value in summary_data.items():
        print(f"  {key:<{max_key_len}} : {value}")
    print("="*70 + "\n")
