"""
OLMoE Integration for Higher Criticism Routing
===============================================

Provides model patching to replace OLMoE's default TopK routing with
Higher Criticism adaptive routing.

UPDATED: Now includes comprehensive internal logging that matches
the evaluation API expectations.

Usage:
    from olmoe_hc_integration import HCRoutingIntegration

    integration = HCRoutingIntegration(model)
    integration.patch_model(min_k=4, max_k=12, beta=0.5)

    # With internal logging (for evaluation)
    integration.start_sample_logging(0, "Sample text...")
    output = model(input_ids)
    integration.end_sample_logging(loss_value)
    
    logs = integration.get_internal_logs()

    integration.unpatch_model()

References:
    Donoho, D., & Jin, J. (2004). Higher criticism for detecting sparse
    heterogeneous mixtures. The Annals of Statistics, 32(3), 962-994.
"""

import torch
import torch.nn.functional as F
import numpy as np
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
    
    Now includes comprehensive internal logging that captures:
    - Router logits and p-values
    - HC statistics at each rank
    - Expert selection decisions
    - Per-sample and per-layer metrics
    
    The logging API matches what hc_routing_evaluation.py expects:
    - start_sample_logging(sample_idx, text)
    - end_sample_logging(loss)
    - get_internal_logs()
    - clear_internal_logs()
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
    # LOGGING API (matches hc_routing_evaluation.py expectations)
    # =========================================================================
    
    def start_sample_logging(self, sample_idx: int, text_preview: str = ""):
        """
        Start logging for a new sample.
        
        Call this BEFORE running model forward pass on a sample.
        
        Args:
            sample_idx: Index of the sample being processed
            text_preview: First ~100 chars of the sample text (for debugging)
        """
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
        """
        End logging for the current sample.
        
        Call this AFTER running model forward pass on a sample.
        
        Args:
            loss: The loss value for this sample (e.g., from outputs.loss)
        """
        if not self.enable_logging or self._current_sample_log is None:
            return
            
        self._current_sample_log['loss'] = loss
        self._current_sample_log['total_tokens'] = self._token_counter
        
        # Aggregate layer statistics
        for layer_idx, decisions in self._current_routing_stats.items():
            if decisions:
                expert_counts = [d['num_selected'] for d in decisions]
                hc_values = [d.get('hc_max_value', 0) for d in decisions]
                weight_sums = [d.get('weight_sum', 0) for d in decisions]
                
                self._current_sample_log['layer_stats'][layer_idx] = {
                    'avg_experts': float(np.mean(expert_counts)),
                    'std_experts': float(np.std(expert_counts)) if len(expert_counts) > 1 else 0.0,
                    'min_experts': int(min(expert_counts)),
                    'max_experts': int(max(expert_counts)),
                    'avg_hc_value': float(np.mean(hc_values)) if hc_values else 0.0,
                    'avg_weight_sum': float(np.mean(weight_sums)) if weight_sums else 0.0,
                    'num_tokens': len(expert_counts),
                }
        
        # Store the completed log
        self._internal_logs.append(self._current_sample_log)
        self._current_sample_log = None
    
    def get_internal_logs(self) -> List[Dict[str, Any]]:
        """
        Get all internal logs collected so far.
        
        Returns:
            List of sample logs, each containing:
            - sample_idx: int
            - text_preview: str
            - layer_stats: Dict[int, Dict] - per-layer statistics
            - routing_decisions: List[Dict] - detailed routing info (sampled)
            - loss: float
            - total_tokens: int
        """
        return self._internal_logs.copy()
    
    def clear_internal_logs(self):
        """Clear all collected internal logs."""
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
        """
        Internal method to log a routing decision during forward pass.
        
        Called automatically by the patched forward function.
        """
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
            
            if stats is not None:
                if 'max_hc_values' in stats:
                    hc_vals = stats['max_hc_values']
                    if hc_vals.dim() == 0:
                        decision['hc_max_value'] = float(hc_vals.item())
                    elif t < hc_vals.shape[0]:
                        decision['hc_max_value'] = float(hc_vals[t].item())
                        
                if 'threshold_ranks' in stats:
                    ranks = stats['threshold_ranks']
                    if ranks.dim() == 0:
                        decision['hc_threshold_rank'] = int(ranks.item())
                    elif t < ranks.shape[0]:
                        decision['hc_threshold_rank'] = int(ranks[t].item())
            
            self._current_routing_stats[layer_idx].append(decision)
            
            # Also store sampled detailed decisions (every 50 tokens)
            if self._current_sample_log is not None and (self._token_counter + t) % 50 == 0:
                self._current_sample_log['routing_decisions'].append(decision)

    # =========================================================================
    # MODEL PATCHING
    # =========================================================================

    def patch_model(
        self,
        min_k: int = 1,
        max_k: int = 16,
        beta: Union[float, str] = 0.5,
        temperature: float = 1.0,
        target_weight_sum: Optional[float] = None,
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
            target_weight_sum: Target weight sum for scaling (None = no scaling)
                - Set to 0.40 to match OLMoE's k=8 training distribution
                - This fixes the variable weight sum problem!
            logger: Optional external logger for detailed logging
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

        # Store config
        self._min_k = min_k
        self._max_k = max_k
        self._beta = beta
        self._target_weight_sum = target_weight_sum
        self._logger = logger
        self._collect_stats = collect_stats

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

                # ============================================================
                # WEIGHT SCALING FIX (Optional but recommended!)
                # ============================================================
                # OLMoE expects weight_sum ≈ 0.40 (from k=8 training)
                # When HC picks more experts, weight_sum goes up → model confused
                # Fix: Scale weights to target sum
                if target_weight_sum is not None:
                    current_sum = routing_weights.sum(dim=-1, keepdim=True)
                    scale_factor = target_weight_sum / (current_sum + 1e-10)
                    routing_weights = routing_weights * scale_factor

                # Log routing decisions (internal logging)
                self._log_routing_decision(layer_idx, expert_counts, routing_weights, stats)

                # Update token counter
                self._token_counter += hidden_flat.shape[0]

                # Collect statistics
                if self._collect_stats:
                    self.stats[f'layer_{layer_idx}_counts'].extend(
                        expert_counts.cpu().tolist()
                    )
                    
                    # Also track weight sums for debugging
                    weight_sums = routing_weights.sum(dim=-1).cpu().tolist()
                    self.stats[f'layer_{layer_idx}_weight_sums'].extend(weight_sums)

                # Compute expert outputs using routing weights
                num_experts = router_logits.shape[-1]

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
        if target_weight_sum is not None:
            print(f"   Weight scaling: target_sum={target_weight_sum}")

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

    # =========================================================================
    # STATISTICS AND ANALYSIS
    # =========================================================================

    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary of collected statistics."""
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
        """
        Analyze weight sums across layers.
        
        This is critical for diagnosing the variable weight sum issue!
        
        Returns:
            Dict with per-layer weight sum statistics
        """
        analysis = {}
        
        for layer_idx in range(16):
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
        
        # Global summary
        all_sums = []
        for layer_idx in range(16):
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
        """Clear collected statistics."""
        self.stats.clear()
        self._token_counter = 0
        self._sample_counter = 0

    def start_sample(self):
        """
        Legacy method - use start_sample_logging() instead.

        Kept for backwards compatibility.
        """
        self._sample_counter += 1
        self._token_counter = 0

    def set_external_logger(self, logger: Optional[HCRoutingLogger]):
        """
        Set or update the external logger after patching.

        This allows evaluation functions to attach a detailed logger
        even if the model is already patched.

        Args:
            logger: HCRoutingLogger instance or None to disable

        Example:
            >>> integration = HCRoutingIntegration(model)
            >>> integration.patch_model(min_k=4, max_k=16, beta=0.5)
            >>> # Later, attach a logger
            >>> logger = HCRoutingLogger('./logs', 'experiment_1')
            >>> integration.set_external_logger(logger)
        """
        self._logger = logger
        if logger:
            print(f"✅ External logger attached: {logger.experiment_name}")
            print(f"   Logging to: {logger.output_dir}")
        else:
            print("ℹ️  External logger disabled")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_hc_patched_model(
    model,
    min_k: int = 4,
    max_k: int = 12,
    beta: Union[float, str] = 0.5,
    target_weight_sum: Optional[float] = 0.40,
    device: str = 'cuda'
) -> HCRoutingIntegration:
    """
    Convenience function to create and patch a model with HC routing.
    
    Args:
        model: OLMoE model instance
        min_k: Minimum experts per token
        max_k: Maximum experts per token
        beta: HC beta parameter ('auto', or float in (0, 1])
        target_weight_sum: Target weight sum (0.40 recommended, None to disable)
        device: Device for computation
    
    Returns:
        HCRoutingIntegration instance (already patched)
    
    Example:
        >>> integration = create_hc_patched_model(model, beta=0.5)
        >>> output = model(input_ids)  # Uses HC routing
        >>> integration.unpatch_model()
    """
    integration = HCRoutingIntegration(model, device=device)
    integration.patch_model(
        min_k=min_k,
        max_k=max_k,
        beta=beta,
        target_weight_sum=target_weight_sum
    )
    return integration
