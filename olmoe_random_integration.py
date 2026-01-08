"""
OLMoE Integration for Random Routing
===============================================

Provides model patching to replace OLMoE's default TopK routing with
a random expert selection mechanism.

Usage:
    from olmoe_random_integration import RandomRoutingIntegration

    integration = RandomRoutingIntegration(model)
    integration.patch_model(experts_amount=8, sum_of_weights=None)

    # With internal logging (for evaluation)
    integration.start_sample_logging(0, "Sample text...")
    output = model(input_ids)
    integration.end_sample_logging(loss_value)

    logs = integration.get_internal_logs()

    integration.unpatch_model()
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, List, Callable, Union
from collections import defaultdict

from random_routing import random_routing


class RandomRoutingIntegration:
    """
    Integrates Random routing into OLMoE model.

    Replaces the forward method of OlmoeSparseMoeBlock to use Random
    instead of default TopK routing.

    Includes comprehensive internal logging that captures:
    - Router logits
    - Expert selection decisions
    - Per-sample and per-layer metrics
    """

    def __init__(self, model, device: str = 'cuda', enable_logging: bool = True):
        """
        Initialize integration.

        Args:
            model: OLMoE model instance
            device: Device for computation
            enable_logging: Whether to enable internal logging (default: True)
        """
        self.model = model
        self.device = device
        self.original_forwards: Dict[str, Callable] = {}
        self.is_patched = False
        self.stats = defaultdict(list)

        # =====================================================================
        # INTERNAL LOGGING SYSTEM (matches evaluation API)
        # =====================================================================
        self.enable_logging = enable_logging
        self._internal_logs: List[Dict[str, Any]] = []
        self._current_sample_log: Optional[Dict[str, Any]] = None
        self._sample_counter = 0
        self._token_counter = 0

        # Per-layer statistics for current sample
        self._layer_decisions: List[Dict[str, Any]] = []

        # Routing stats captured during forward pass
        self._current_routing_stats: Dict[int, List[Dict]] = defaultdict(list)

        # Find all MoE blocks
        self.moe_blocks = {}
        for name, module in model.named_modules():
            if module.__class__.__name__ == 'OlmoeSparseMoeBlock':
                self.moe_blocks[name] = module

        print(f"Found {len(self.moe_blocks)} MoE blocks")
        if enable_logging:
            print("✅ Internal logging enabled")

    # =========================================================================
    # LOGGING API
    # =========================================================================

    def start_sample_logging(self, sample_idx: int, text_preview: str = ""):
        if not self.enable_logging:
            return

        self._sample_counter = sample_idx
        self._token_counter = 0
        self._current_routing_stats.clear()

        self._current_sample_log = {
            'sample_idx': sample_idx,
            'text_preview': text_preview[:100] if text_preview else "",
            'layer_stats': {},
            'routing_decisions': [],
            'loss': None,
            'total_tokens': 0,
        }

    def end_sample_logging(self, loss: float = 0.0):
        if not self.enable_logging or self._current_sample_log is None:
            return

        self._current_sample_log['loss'] = loss
        self._current_sample_log['total_tokens'] = self._token_counter

        for layer_idx, decisions in self._current_routing_stats.items():
            if decisions:
                expert_counts = [d['num_selected'] for d in decisions]
                weight_sums = [d.get('weight_sum', 0) for d in decisions]

                self._current_sample_log['layer_stats'][layer_idx] = {
                    'avg_experts': float(np.mean(expert_counts)),
                    'std_experts': float(np.std(expert_counts)) if len(expert_counts) > 1 else 0.0,
                    'min_experts': int(min(expert_counts)),
                    'max_experts': int(max(expert_counts)),
                    'avg_weight_sum': float(np.mean(weight_sums)) if weight_sums else 0.0,
                    'num_tokens': len(expert_counts),
                }

        self._internal_logs.append(self._current_sample_log)
        self._current_sample_log = None

    def get_internal_logs(self) -> List[Dict[str, Any]]:
        return self._internal_logs.copy()

    def clear_internal_logs(self):
        self._internal_logs.clear()
        self._current_sample_log = None
        self._current_routing_stats.clear()
        self._sample_counter = 0
        self._token_counter = 0

    def _log_routing_decision(
        self,
        layer_idx: int,
        expert_counts: torch.Tensor,
        routing_weights: torch.Tensor,
        stats: Optional[Dict] = None
    ):
        if not self.enable_logging:
            return

        num_tokens = expert_counts.shape[0]

        for t in range(num_tokens):
            decision = {
                'token_idx': self._token_counter + t,
                'layer_idx': layer_idx,
                'num_selected': int(expert_counts[t].item()),
                'weight_sum': float(routing_weights[t].sum().item()),
            }

            self._current_routing_stats[layer_idx].append(decision)

            if self._current_sample_log is not None and (self._token_counter + t) % 50 == 0:
                self._current_sample_log['routing_decisions'].append(decision)

    # =========================================================================
    # MODEL PATCHING
    # =========================================================================

    def patch_model(
        self,
        experts_amount: int = 8,
        sum_of_weights: Optional[float] = None,
        temperature: float = 1.0,
        collect_stats: bool = True,
        layers_to_patch: Optional[List[int]] = None
    ):
        """
        Patch model to use Random routing.

        Args:
            experts_amount: Number of experts to select randomly.
            sum_of_weights: Target sum of weights for normalization. If None, it's calculated from top 8 weights.
            temperature: Softmax temperature for weight computation.
            collect_stats: Whether to collect routing statistics.
            layers_to_patch: A list of layer indices to patch. If None, all layers are patched.
        """
        if self.is_patched:
            print("Model already patched. Call unpatch_model() first.")
            return

        self._experts_amount = experts_amount
        self._sum_of_weights = sum_of_weights
        self._collect_stats = collect_stats

        def create_random_forward(layer_name: str, original_block):
            """Create Random routing-enabled forward function for a specific layer."""
            layer_idx = 0
            parts = layer_name.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        break
                    except ValueError:
                        pass

            def random_forward(hidden_states: torch.Tensor) -> tuple:
                nonlocal self

                batch_size, seq_len, hidden_dim = hidden_states.shape
                hidden_flat = hidden_states.view(-1, hidden_dim)
                router_logits = original_block.gate(hidden_flat)

                routing_weights, selected_experts, expert_counts, stats = random_routing(
                    router_logits=router_logits,
                    experts_amount=experts_amount,
                    sum_of_weights=sum_of_weights,
                    temperature=temperature
                )

                self._log_routing_decision(layer_idx, expert_counts, routing_weights, stats)
                self._token_counter += hidden_flat.shape[0]

                if self._collect_stats:
                    self.stats[f'layer_{layer_idx}_counts'].extend(
                        expert_counts.cpu().tolist()
                    )
                    weight_sums = routing_weights.sum(dim=-1).cpu().tolist()
                    self.stats[f'layer_{layer_idx}_weight_sums'].extend(weight_sums)

                num_experts = router_logits.shape[-1]
                final_hidden = torch.zeros_like(hidden_flat)

                for expert_idx in range(num_experts):
                    expert_layer = original_block.experts[expert_idx]
                    expert_mask = routing_weights[:, expert_idx] > 0

                    if expert_mask.any():
                        expert_input = hidden_flat[expert_mask]
                        expert_output = expert_layer(expert_input)
                        expert_weight = routing_weights[expert_mask, expert_idx].unsqueeze(-1)
                        final_hidden[expert_mask] += expert_weight * expert_output

                output = final_hidden.view(batch_size, seq_len, hidden_dim)

                return output, router_logits

            return random_forward

        blocks_to_patch = self.moe_blocks
        if layers_to_patch is not None:
            blocks_to_patch = {name: block for name, block in self.moe_blocks.items() if
                               int(name.split('.')[2]) in layers_to_patch}

        for name, block in blocks_to_patch.items():
            self.original_forwards[name] = block.forward
            block.forward = create_random_forward(name, block)

        self.is_patched = True
        assert self.is_patched, "Patching failed silently"
        assert len(blocks_to_patch) > 0, "No MoE blocks found to patch"

        print(f"✅ Patched {len(blocks_to_patch)} MoE blocks with Random routing")
        print(f"   Config: experts_amount={experts_amount}, sum_of_weights={sum_of_weights}")
        if layers_to_patch is not None:
            print(f"   Patched layers: {layers_to_patch}")

    def unpatch_model(self):
        if not self.is_patched:
            print("Model is not patched.")
            return

        for name, block in self.moe_blocks.items():
            if name in self.original_forwards:
                block.forward = self.original_forwards[name]

        self.original_forwards.clear()
        self.is_patched = False
        self._token_counter = 0
        self._sample_counter = 0
        print("✅ Restored original routing")

    # =========================================================================
    # STATISTICS AND ANALYSIS
    # =========================================================================

    def get_stats_summary(self) -> Dict[str, Any]:
        summary = {}
        for key, values in self.stats.items():
            if values:
                summary[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(min(values)),
                    'max': float(max(values)),
                    'count': len(values),
                }
        return summary

    def get_weight_sum_analysis(self) -> Dict[str, Any]:
        analysis = {}
        for layer_idx in range(len(self.moe_blocks)):
            key = f'layer_{layer_idx}_weight_sums'
            if key in self.stats and self.stats[key]:
                values = self.stats[key]
                analysis[f'layer_{layer_idx}'] = {
                    'avg_weight_sum': float(np.mean(values)),
                    'std_weight_sum': float(np.std(values)),
                    'min_weight_sum': float(min(values)),
                    'max_weight_sum': float(max(values)),
                    'variance': float(np.var(values)),
                }
        all_sums = []
        for layer_idx in range(len(self.moe_blocks)):
            key = f'layer_{layer_idx}_weight_sums'
            if key in self.stats:
                all_sums.extend(self.stats[key])
        if all_sums:
            analysis['global'] = {
                'avg_weight_sum': float(np.mean(all_sums)),
                'std_weight_sum': float(np.std(all_sums)),
                'min_weight_sum': float(min(all_sums)),
                'max_weight_sum': float(max(all_sums)),
                'total_tokens': len(all_sums),
            }
        return analysis

    def clear_stats(self):
        self.stats.clear()
        self._token_counter = 0
        self._sample_counter = 0

    def start_sample(self):
        self._sample_counter += 1
        self._token_counter = 0




# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_random_patched_model(
    model,
    experts_amount: int = 8,
    sum_of_weights: Optional[float] = None,
    device: str = 'cuda',
    layers_to_patch: Optional[List[int]] = None
) -> RandomRoutingIntegration:
    """
    Convenience function to create and patch a model with Random routing.

    Args:
        model: OLMoE model instance
        experts_amount: Number of experts to select randomly.
        sum_of_weights: Target sum of weights for normalization.
        device: Device for computation
        layers_to_patch: A list of layer indices to patch. If None, all layers are patched.

    Returns:
        RandomRoutingIntegration instance (already patched)

    Example:
        >>> integration = create_random_patched_model(model, experts_amount=4, layers_to_patch=[0, 1, 2])
        >>> output = model(input_ids)  # Uses Random routing on specified layers
        >>> integration.unpatch_model()
    """
    integration = RandomRoutingIntegration(model, device=device)
    integration.patch_model(
        experts_amount=experts_amount,
        sum_of_weights=sum_of_weights,
        layers_to_patch=layers_to_patch
    )
    return integration
