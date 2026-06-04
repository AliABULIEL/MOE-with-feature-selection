"""
MoE Internal Router Logging (Shared by BH & HC)
================================================

Captures internal router outputs using forward hooks.
Works with any routing strategy (BH, HC, TopK baseline).

Usage:
    from moe_internal_logging import RouterLogger, InternalRoutingLogger

    # Per-sample logging
    logger = RouterLogger(model)
    logger.register_hooks(top_k=8)
    outputs = model(input_ids)
    routing_data = logger.get_routing_data()
    logger.clear_data()
    logger.remove_hooks()

    # Accumulated logging across experiment
    internal_logger = InternalRoutingLogger(output_dir, experiment_name)
    internal_logger.log_sample(sample_id, routing_data, loss, num_tokens)
    internal_logger.save_logs()
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from typing import Dict, List, Any, Optional
from collections import defaultdict
from pathlib import Path
from datetime import datetime


class RouterLogger:
    def __init__(self, model):
        """
        Initialize router logger.

        Args:
            model: DeepSeek-V2-Lite model instance
        """
        self.model = model
        self.hooks = []
        self.routing_data = []
        self.num_experts = getattr(model.config, "n_routed_experts", 64)
        self.top_k = getattr(model.config, "num_experts_per_tok", 6)

    def register_hooks(self, top_k: int = 8):
        """
        Register forward hooks on router gates to capture routing decisions.

        Args:
            top_k: Number of top experts to track in statistics
        """
        self.remove_hooks()  # Clear existing hooks
        self.top_k = top_k

        def create_hook(layer_idx, norm_top_k=False):
            def hook_fn(module, input, output):
                """Capture router output and compute routing statistics."""
                hidden_states = input[0]

                # Manually compute pre-softmax logits: hidden_states @ weight.T
                # We use the module's weight parameter
                try:
                    with torch.no_grad():
                        if hasattr(module, "weight"):
                            router_logits = F.linear(
                                hidden_states.to(torch.float32),
                                module.weight.to(torch.float32),
                            )
                        else:
                            print(f"Warning: {layer_idx} has no 'weight' attribute.")

                        # Check shape of router_logits, if it's [1, n, m], squeeze
                        if router_logits.dim() == 3 and router_logits.shape[0] == 1:
                            router_logits = router_logits.squeeze(0)
                        # Compute softmax probabilities
                        probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)

                        # Get top-k
                        topk_weights, topk_indices = torch.topk(
                            probs, min(self.top_k, probs.shape[-1]), dim=-1
                        )

                        # Normalize top-k weights if specified
                        if norm_top_k:
                            topk_weights = topk_weights / topk_weights.sum(
                                dim=-1, keepdim=True
                            )
                        
                        # Compute statistics
                        max_prob = probs.max(dim=-1).values.mean().item()
                        entropy = (
                            -(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item()
                        )
                        concentration = (
                            topk_weights[:, 0].mean().item()
                            if topk_weights.shape[1] > 0
                            else 0.0
                        )

                        # Track which experts are selected
                        unique_experts = topk_indices.unique().tolist()

                        self.routing_data.append(
                            {
                                "layer": layer_idx,
                                "router_logits": router_logits.detach().cpu(),
                                "expert_indices": topk_indices.detach().cpu(),
                                "expert_weights": topk_weights.detach().cpu(),
                                "probs": probs.detach().cpu(),
                                "stats": {
                                    "max_prob": max_prob,
                                    "entropy": entropy,
                                    "concentration": concentration,
                                    "num_tokens": router_logits.shape[0],
                                    "unique_experts_this_layer": len(unique_experts),
                                },
                            }
                        )
                except Exception as e:
                    print(f"Warning: Hook error at layer {layer_idx}: {e}")

            return hook_fn

        # Register hooks on each layer's gate
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
                norm_top_k = (
                    self.model.config.norm_topk_prob
                    if hasattr(self.model.config, "norm_topk_prob")
                    else False
                )
                hook = layer.mlp.gate.register_forward_hook(create_hook(i, norm_top_k))
                self.hooks.append(hook)
        print(f"✅ Registered {len(self.hooks)} router hooks (top_k={top_k})")

    def get_routing_data(self) -> List[Dict]:
        """Get captured routing data from all layers."""
        return self.routing_data

    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics across all layers."""
        if not self.routing_data:
            return {}

        all_max_probs = [d["stats"]["max_prob"] for d in self.routing_data]
        all_entropies = [d["stats"]["entropy"] for d in self.routing_data]
        all_concentrations = [d["stats"]["concentration"] for d in self.routing_data]

        # Collect all expert indices across all layers
        all_experts = set()
        for d in self.routing_data:
            indices = d["expert_indices"].numpy().flatten()
            all_experts.update(indices.tolist())

        return {
            "avg_max_prob": float(np.mean(all_max_probs)),
            "avg_entropy": float(np.mean(all_entropies)),
            "avg_concentration": float(np.mean(all_concentrations)),
            "unique_experts_used": len(all_experts),
            "expert_utilization": len(all_experts) / self.num_experts,
            "num_layers_captured": len(self.routing_data),
        }

    def get_per_layer_stats(self) -> Dict[int, Dict]:
        """Get statistics per layer."""
        per_layer = {}
        for d in self.routing_data:
            layer = d["layer"]
            per_layer[layer] = d["stats"]
        return per_layer

    def clear_data(self):
        """Clear captured routing data (call between samples)."""
        self.routing_data = []

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()


class InternalRoutingLogger:
    """
    Accumulated logging across multiple samples for an entire experiment.

    Collects routing statistics across evaluation run,
    then generates comprehensive logs and visualizations.

    Works for BH, HC, or any routing strategy.
    """

    def __init__(
        self, output_dir: str, experiment_name: str, routing_method: str = "unknown"
    ):
        """
        Initialize internal routing logger.

        Args:
            output_dir: Directory to save logs
            experiment_name: Name of experiment (e.g., 'bh_alpha030_maxk8')
            routing_method: 'bh', 'hc', or 'topk'
        """
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.routing_method = routing_method
        self.logs_dir = self.output_dir / "internal_logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Accumulated statistics
        self.all_samples = []
        self.per_layer_stats = defaultdict(list)
        self.expert_usage_counts = defaultdict(int)
        self.total_tokens = 0
        self.total_loss = 0.0

    def log_sample(
        self,
        sample_id: int,
        routing_data: List[Dict],
        loss: float,
        num_tokens: int,
        expert_counts: Optional[List[int]] = None,
    ):
        """
        Log routing data for a single sample.

        Args:
            sample_id: Sample identifier
            routing_data: Data from RouterLogger.get_routing_data()
            loss: Sample loss
            num_tokens: Number of tokens in sample
            expert_counts: Optional list of expert counts per token (from BH/HC)
        """
        self.total_tokens += num_tokens
        self.total_loss += loss * num_tokens

        sample_summary = {
            "sample_id": sample_id,
            "loss": loss,
            "num_tokens": num_tokens,
            "expert_counts": expert_counts,
            "layers": [],
        }

        for layer_data in routing_data:
            layer_idx = layer_data["layer"]
            stats = layer_data["stats"]

            # Accumulate per-layer stats
            self.per_layer_stats[layer_idx].append(stats)

            # Count expert usage
            indices = layer_data["expert_indices"].numpy().flatten()
            for idx in indices:
                self.expert_usage_counts[int(idx)] += 1

            sample_summary["layers"].append(
                {
                    "layer": layer_idx,
                    "entropy": stats["entropy"],
                    "max_prob": stats["max_prob"],
                    "concentration": stats["concentration"],
                    "unique_experts": stats.get("unique_experts_this_layer", 0),
                }
            )

        self.all_samples.append(sample_summary)

    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics across all samples."""
        if not self.all_samples:
            return {}

        # Average per-layer entropy
        layer_entropies = {}
        layer_concentrations = {}
        for layer_idx, stats_list in self.per_layer_stats.items():
            layer_entropies[layer_idx] = float(
                np.mean([s["entropy"] for s in stats_list])
            )
            layer_concentrations[layer_idx] = float(
                np.mean([s["concentration"] for s in stats_list])
            )

        # Expert utilization
        total_activations = sum(self.expert_usage_counts.values())
        expert_distribution = (
            {k: v / total_activations for k, v in self.expert_usage_counts.items()}
            if total_activations > 0
            else {}
        )

        # Expert count stats (from BH/HC routing)
        all_expert_counts = []
        for s in self.all_samples:
            if s.get("expert_counts"):
                all_expert_counts.extend(s["expert_counts"])

        expert_count_stats = {}
        if all_expert_counts:
            expert_count_stats = {
                "avg_experts": float(np.mean(all_expert_counts)),
                "std_experts": float(np.std(all_expert_counts)),
                "min_experts": int(np.min(all_expert_counts)),
                "max_experts": int(np.max(all_expert_counts)),
                "median_experts": float(np.median(all_expert_counts)),
            }

        return {
            "routing_method": self.routing_method,
            "num_samples": len(self.all_samples),
            "total_tokens": self.total_tokens,
            "avg_loss": (
                self.total_loss / self.total_tokens
                if self.total_tokens > 0
                else float("inf")
            ),
            "perplexity": (
                float(np.exp(self.total_loss / self.total_tokens))
                if self.total_tokens > 0
                else float("inf")
            ),
            "layer_entropies": layer_entropies,
            "layer_concentrations": layer_concentrations,
            "unique_experts": len(self.expert_usage_counts),
            "expert_utilization": len(self.expert_usage_counts) / 64,
            "most_used_experts": sorted(
                self.expert_usage_counts.items(), key=lambda x: x[1], reverse=True
            )[:10],
            "least_used_experts": sorted(
                self.expert_usage_counts.items(), key=lambda x: x[1]
            )[:10],
            **expert_count_stats,
        }

    def save_logs(self):
        """Save all accumulated logs to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"{self.experiment_name}_{timestamp}_internal.json"

        output = {
            "experiment": self.experiment_name,
            "routing_method": self.routing_method,
            "timestamp": timestamp,
            "aggregate_stats": self.get_aggregate_stats(),
            "per_layer_summary": {
                layer: {
                    "avg_entropy": float(np.mean([s["entropy"] for s in stats])),
                    "avg_max_prob": float(np.mean([s["max_prob"] for s in stats])),
                    "avg_concentration": float(
                        np.mean([s["concentration"] for s in stats])
                    ),
                }
                for layer, stats in self.per_layer_stats.items()
            },
            "expert_usage": dict(self.expert_usage_counts),
            "sample_details": self.all_samples[:50],  # First 50 samples
        }

        with open(log_file, "w") as f:
            json.dump(output, f, indent=2, default=str)

        print(f"✅ Saved internal routing logs: {log_file}")
        return log_file

    def clear(self):
        """Clear all accumulated data."""
        self.all_samples = []
        self.per_layer_stats = defaultdict(list)
        self.expert_usage_counts = defaultdict(int)
        self.total_tokens = 0
        self.total_loss = 0.0
