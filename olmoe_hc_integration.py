"""
OLMoE Integration for Higher Criticism Routing
===============================================

Provides model patching to replace OLMoE's default TopK routing with
Higher Criticism adaptive routing.

Usage:
    from olmoe_hc_integration import HCRoutingIntegration

    integration = HCRoutingIntegration(model)
    integration.patch_model(min_k=4, max_k=12, beta=0.5, hc_variant='plus')

    # ... run inference ...

    integration.unpatch_model()

References:
    Donoho, D., & Jin, J. (2004). Higher criticism for detecting sparse
    heterogeneous mixtures. The Annals of Statistics, 32(3), 962-994.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Callable, Union
from collections import defaultdict

from hc_routing import higher_criticism_routing
from bh_routing import load_kde_models
from hc_routing_logging import HCRoutingLogger


class HCRoutingIntegration:
    """
    Integrates Higher Criticism routing into OLMoE model.

    Replaces the forward method of OlmoeSparseMoeBlock to use HC
    instead of default TopK routing.
    """

    def __init__(self, model, device: str = 'cuda'):
        """
        Initialize integration.

        Args:
            model: OLMoE model instance
            device: Device for computation
        """
        self.model = model
        self.device = device
        self.original_forwards: Dict[str, Callable] = {}
        self.is_patched = False
        self.stats = defaultdict(list)

        # Find all MoE blocks
        self.moe_blocks = {}
        for name, module in model.named_modules():
            if module.__class__.__name__ == 'OlmoeSparseMoeBlock':
                self.moe_blocks[name] = module

        print(f"Found {len(self.moe_blocks)} MoE blocks")

    def patch_model(
        self,
        min_k: int = 1,
        max_k: int = 16,
        beta: Union[float, str] = 0.5,
        temperature: float = 1.0,
        logger: Optional[HCRoutingLogger] = None,
        log_every_n: int = 100,
        collect_stats: bool = True
    ):
        """
        Patch model to use HC routing (Donoho & Jin 2004).

        SIMPLIFIED API - Beta controls everything:
        - beta='auto'  → HC⁺ (adaptive search) - RECOMMENDED
        - beta=1.0     → HC (full search) - all ranks
        - beta=0.3-0.9 → HC* (partial search) - first β fraction

        Args:
            min_k: Minimum experts per token (safety floor)
            max_k: Maximum experts per token (ceiling)
            beta: THE MAIN TUNING PARAMETER
                - 'auto': Adaptive search (only where p < expected) [RECOMMENDED]
                - 1.0: Search all ranks
                - 0.3-0.9: Search first β fraction of ranks
            temperature: Softmax temperature for weight computation
            logger: Optional logger for detailed logging
            log_every_n: Logging frequency (every N tokens)
            collect_stats: Whether to collect routing statistics
        """
        if self.is_patched:
            print("Model already patched. Call unpatch_model() first.")
            return

        # Load KDE models once
        kde_models = load_kde_models()
        if not kde_models:
            print("⚠️ Warning: KDE models not found. Using empirical p-values.")

        # Token counter and sample tracking for logging
        self._token_counter = 0
        self._sample_counter = 0
        self._logger = logger
        self._collect_stats = collect_stats

        # Store config for assertions
        self._min_k = min_k
        self._max_k = max_k
        self._beta = beta

        def create_hc_forward(layer_name: str, original_block):
            """Create HC-enabled forward function for a specific layer."""

            # Extract layer index from name like "model.layers.5.mlp"
            layer_idx = 0
            parts = layer_name.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        break
                    except ValueError:
                        pass

            def hc_forward(hidden_states: torch.Tensor) -> tuple:
                """
                HC-enabled forward pass.

                Args:
                    hidden_states: [batch, seq_len, hidden_dim]

                Returns:
                    (output, router_logits)
                """
                nonlocal self

                batch_size, seq_len, hidden_dim = hidden_states.shape

                # Flatten for routing
                hidden_flat = hidden_states.view(-1, hidden_dim)

                # Get router logits
                router_logits = original_block.gate(hidden_flat)  # [B*S, num_experts]

                # Apply HC routing
                routing_weights, selected_experts, expert_counts, stats = higher_criticism_routing(
                    router_logits,
                    beta=beta,
                    temperature=temperature,
                    min_k=min_k,
                    max_k=max_k,
                    layer_idx=layer_idx,
                    kde_models=kde_models,
                    return_stats=True,
                    logger=self._logger,
                    log_every_n_tokens=log_every_n,
                    sample_idx=self._sample_counter,
                    token_idx=self._token_counter
                )

                # Update token counter
                self._token_counter += hidden_flat.shape[0]

                # Collect statistics
                if self._collect_stats:
                    self.stats[f'layer_{layer_idx}_counts'].extend(
                        expert_counts.cpu().tolist()
                    )

                # Compute expert outputs using routing weights
                # This mirrors the original OLMoE computation

                num_experts = router_logits.shape[-1]

                # Get expert weights and indices
                # routing_weights: [B*S, num_experts] - sparse, sums to 1

                # Compute output as weighted sum of expert outputs
                final_hidden = torch.zeros_like(hidden_flat)

                for expert_idx in range(num_experts):
                    expert_layer = original_block.experts[expert_idx]
                    expert_mask = routing_weights[:, expert_idx] > 0

                    if expert_mask.any():
                        expert_input = hidden_flat[expert_mask]
                        expert_output = expert_layer(expert_input)
                        expert_weight = routing_weights[expert_mask, expert_idx].unsqueeze(-1)
                        final_hidden[expert_mask] += expert_weight * expert_output

                # Reshape output
                output = final_hidden.view(batch_size, seq_len, hidden_dim)

                return output, router_logits

            return hc_forward

        # Patch each MoE block
        for name, block in self.moe_blocks.items():
            # Save original forward
            self.original_forwards[name] = block.forward

            # Create and set new forward
            block.forward = create_hc_forward(name, block)

        self.is_patched = True

        # Verify patching worked
        assert self.is_patched, "Patching failed silently"
        assert len(self.moe_blocks) == 16, f"Expected 16 MoE blocks, found {len(self.moe_blocks)}"

        print(f"✅ Patched {len(self.moe_blocks)} MoE blocks with HC routing")
        print(f"   Config: min_k={min_k}, max_k={max_k}, β={beta}")

    def unpatch_model(self):
        """Restore original routing."""
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

    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary of collected statistics."""
        summary = {}

        for key, values in self.stats.items():
            if values:
                summary[key] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values),
                }

        return summary

    def clear_stats(self):
        """Clear collected statistics."""
        self.stats.clear()
        self._token_counter = 0
        self._sample_counter = 0

    def start_sample(self):
        """Call this before processing each new sample for proper logging."""
        self._sample_counter += 1
        self._token_counter = 0
