"""
OLMoE BH Routing Integration
=============================

This module provides non-invasive integration of Benjamini-Hochberg (BH) routing
with pre-trained OLMoE models, compatible with Colab environments.

Approach: Method replacement (monkey-patching) that works without modifying
the transformers library source code.

Features:
- Two modes: 'patch' (changes routing) and 'analyze' (simulation only)
- Compatible with model.generate() and standard inference
- Statistics collection for routing analysis
- Handles all 16 MoE layers in OLMoE
- Reversible patching (can restore original behavior)

Usage:
    from olmoe_bh_integration import BHRoutingIntegration

    # Load model
    model = AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924")

    # Apply BH routing
    integrator = BHRoutingIntegration(model, alpha=0.05, max_k=8, mode='patch')
    integrator.patch_model()

    # Run inference
    outputs = model.generate(...)

    # Get statistics
    stats = integrator.get_routing_stats()

    # Restore original
    integrator.unpatch_model()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import warnings
from collections import defaultdict

# Import BH routing module
try:
    from deprecated.bh_routing import benjamini_hochberg_routing
except ImportError:
    raise ImportError(
        "Cannot import bh_routing module. Ensure bh_routing.py is in the same directory."
    )


class BHRoutingIntegration:
    """
    Integrates Benjamini-Hochberg routing with OLMoE models.

    This class patches OlmoeTopKRouter instances to use BH routing instead
    of fixed top-k selection. Works without modifying transformers source code.

    Attributes:
        model: The OLMoE model to patch
        alpha: FDR control level for BH procedure
        temperature: Softmax temperature for probability calibration
        min_k: Minimum experts to select per token
        max_k: Maximum experts to select per token
        mode: 'patch' (modify routing) or 'analyze' (simulate only)

    Example:
        >>> integrator = BHRoutingIntegration(model, alpha=0.05, max_k=8)
        >>> integrator.patch_model()
        >>> outputs = model.generate(input_ids)
        >>> stats = integrator.get_routing_stats()
        >>> integrator.unpatch_model()
    """

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.05,
        temperature: float = 1.0,
        min_k: int = 1,
        max_k: int = 8,
        mode: str = 'patch',
        collect_stats: bool = True
    ):
        """
        Initialize BH routing integration.

        Args:
            model: OLMoE model (from transformers)
            alpha: FDR control level (0.01-0.20)
            temperature: Softmax temperature (0.5-2.0)
            min_k: Minimum experts per token (>= 1)
            max_k: Maximum experts per token (<= 64 for OLMoE)
            mode: 'patch' to modify routing, 'analyze' to simulate only
            collect_stats: Whether to collect routing statistics

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If model doesn't have OLMoE architecture
        """
        self.model = model
        self.alpha = alpha
        self.temperature = temperature
        self.min_k = min_k
        self.max_k = max_k
        self.mode = mode
        self.collect_stats = collect_stats

        # Validate parameters
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if min_k < 1:
            raise ValueError(f"min_k must be >= 1, got {min_k}")
        if not 1 <= min_k <= max_k:
            raise ValueError(f"Must have 1 <= min_k <= max_k, got {min_k}, {max_k}")
        if mode not in ['patch', 'analyze']:
            raise ValueError(f"mode must be 'patch' or 'analyze', got {mode}")

        # Storage for patching
        self.routers: List[Tuple[str, nn.Module]] = []  # (name, router_module)
        self.original_forwards: Dict[str, callable] = {}  # name -> original forward
        self.is_patched = False

        # Statistics storage
        self.routing_stats: Dict[str, Any] = defaultdict(list)
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        # Find routers in model
        self._find_routers()

        if len(self.routers) == 0:
            raise RuntimeError(
                "No OlmoeTopKRouter modules found in model. "
                "Ensure you're using an OLMoE model from transformers."
            )

        print(f"[BH Integration] Found {len(self.routers)} router modules")
        print(f"[BH Integration] Mode: {mode}")
        print(f"[BH Integration] Alpha: {alpha}, Temperature: {temperature}, "
              f"min_k: {min_k}, max_k: {max_k}")

    def _find_routers(self):
        """Find all OlmoeTopKRouter instances in the model."""
        for name, module in self.model.named_modules():
            # Check if this is a router module
            # OLMoE uses OlmoeTopKRouter class
            if module.__class__.__name__ == 'OlmoeTopKRouter':
                self.routers.append((name, module))
                print(f"  Found router: {name}")

    def _bh_routing_compatible(
        self,
        router_logits: torch.Tensor,
        original_dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply BH routing and convert to topk-compatible format.

        The original OlmoeTopKRouter returns:
            routing_weights: [num_tokens, k] - dense weights
            selected_experts: [num_tokens, k] - expert indices

        BH routing returns sparse format, so we convert.

        Args:
            router_logits: [num_tokens, num_experts] or [batch, seq, num_experts]
            original_dtype: dtype to cast output to

        Returns:
            routing_weights: [num_tokens, max_k] - dense weights for selected experts
            selected_experts: [num_tokens, max_k] - expert indices (padded with -1)
        """
        # Handle 2D and 3D inputs
        input_shape = router_logits.shape
        if router_logits.ndim == 2:
            # [num_tokens, num_experts] - typical during forward
            pass
        elif router_logits.ndim == 3:
            # [batch, seq_len, num_experts] - may occur in some contexts
            # Reshape to 2D for processing
            batch_size, seq_len, num_experts = router_logits.shape
            router_logits = router_logits.view(-1, num_experts)
        else:
            raise ValueError(f"Unexpected router_logits shape: {input_shape}")

        # Apply BH routing (returns sparse format)
        sparse_weights, selected_experts, expert_counts = benjamini_hochberg_routing(
            router_logits,
            alpha=self.alpha,
            temperature=self.temperature,
            min_k=self.min_k,
            max_k=self.max_k,
            return_stats=False
        )

        # Convert sparse weights [num_tokens, num_experts] to dense [num_tokens, max_k]
        # sparse_weights has zeros for non-selected experts
        # selected_experts has -1 padding for unused slots

        num_tokens = router_logits.shape[0]
        device = router_logits.device

        # Gather weights for selected experts
        # Clamp indices to avoid gather error on -1 padding
        safe_indices = selected_experts.clamp(min=0)
        dense_weights = sparse_weights.gather(dim=-1, index=safe_indices)

        # Zero out weights for padded positions (-1 indices)
        padding_mask = selected_experts == -1
        dense_weights = dense_weights.masked_fill(padding_mask, 0.0)

        # Convert to original dtype
        dense_weights = dense_weights.to(original_dtype)

        # Restore original shape if needed
        if len(input_shape) == 3:
            dense_weights = dense_weights.view(batch_size, seq_len, self.max_k)
            selected_experts = selected_experts.view(batch_size, seq_len, self.max_k)

        return dense_weights, selected_experts

    def _create_patched_forward(
        self,
        router_module: nn.Module,
        router_name: str
    ) -> callable:
        """
        Create a patched forward method for a router.

        Args:
            router_module: The OlmoeTopKRouter instance
            router_name: Name of the router (for logging)

        Returns:
            Patched forward function
        """
        # Capture references in closure
        original_linear = router_module.linear
        alpha = self.alpha
        temperature = self.temperature
        mode = self.mode

        def patched_forward(hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Patched forward that uses BH routing.

            Original signature:
                def forward(hidden_states) -> (routing_weights, selected_experts, router_logits)

            We maintain the same signature for compatibility.
            """
            # Compute router logits using original linear layer
            router_logits = original_linear(hidden_states)
            # Shape: [num_tokens, num_experts] or [batch, seq, num_experts]

            if mode == 'analyze':
                # Simulation mode: use original topk but log BH decisions
                original_k = router_module.top_k
                routing_weights, selected_experts = torch.topk(
                    router_logits, k=original_k, dim=-1
                )
                routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float)
                routing_weights = routing_weights.to(hidden_states.dtype)

                # Log what BH would select (no grad needed)
                with torch.no_grad():
                    bh_weights, bh_experts = self._bh_routing_compatible(
                        router_logits.detach(),
                        hidden_states.dtype
                    )
                    # Store for analysis
                    if self.collect_stats:
                        self.routing_stats['bh_expert_counts'].append(
                            (bh_experts != -1).sum(dim=-1).float().mean().item()
                        )

                return routing_weights, selected_experts, router_logits

            else:  # mode == 'patch'
                # Actually use BH routing
                routing_weights, selected_experts = self._bh_routing_compatible(
                    router_logits,
                    hidden_states.dtype
                )

                # Collect statistics if enabled
                if self.collect_stats:
                    with torch.no_grad():
                        num_selected = (selected_experts != -1).sum(dim=-1).float()
                        self.routing_stats['expert_counts'].append(num_selected.mean().item())
                        self.routing_stats['expert_counts_std'].append(num_selected.std().item())

                return routing_weights, selected_experts, router_logits

        return patched_forward

    def patch_model(self):
        """
        Apply BH routing to the model.

        Replaces the forward method of all OlmoeTopKRouter instances.
        Can be undone with unpatch_model().

        Raises:
            RuntimeError: If model is already patched
        """
        if self.is_patched:
            raise RuntimeError("Model is already patched. Call unpatch_model() first.")

        print(f"\n[BH Integration] Patching {len(self.routers)} routers...")

        for router_name, router_module in self.routers:
            # Save original forward
            self.original_forwards[router_name] = router_module.forward

            # Create and apply patched forward
            patched_forward = self._create_patched_forward(router_module, router_name)
            router_module.forward = patched_forward

            print(f"  ✓ Patched {router_name}")

        self.is_patched = True
        print(f"[BH Integration] Patching complete. Mode: {self.mode}")

        if self.mode == 'patch':
            print("  Model now uses BH routing for expert selection.")
        else:
            print("  Model uses original routing, BH decisions logged for analysis.")

    def unpatch_model(self):
        """
        Restore original routing behavior.

        Reverts all router modules to their original forward methods.
        """
        if not self.is_patched:
            warnings.warn("Model is not currently patched.")
            return

        print(f"\n[BH Integration] Unpatching {len(self.routers)} routers...")

        for router_name, router_module in self.routers:
            # Restore original forward
            if router_name in self.original_forwards:
                router_module.forward = self.original_forwards[router_name]
                print(f"  ✓ Restored {router_name}")

        self.is_patched = False
        print("[BH Integration] Unpatching complete. Model restored to original state.")

    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get collected routing statistics.

        Returns:
            Dictionary with statistics:
                - expert_counts: list of mean expert counts per forward pass
                - expert_counts_std: list of std dev of expert counts
                - total_forward_passes: number of forward passes recorded

        Note:
            Statistics are only collected if collect_stats=True and model is patched.
        """
        if not self.routing_stats:
            warnings.warn("No statistics collected. Ensure model is patched and collect_stats=True.")
            return {}

        stats = {
            'mode': self.mode,
            'alpha': self.alpha,
            'temperature': self.temperature,
            'min_k': self.min_k,
            'max_k': self.max_k,
        }

        if self.mode == 'patch' and 'expert_counts' in self.routing_stats:
            counts = self.routing_stats['expert_counts']
            stats.update({
                'mean_experts_per_token': sum(counts) / len(counts) if counts else 0,
                'std_experts_per_token': sum(self.routing_stats['expert_counts_std']) / len(counts) if counts else 0,
                'total_forward_passes': len(counts),
            })

        if self.mode == 'analyze' and 'bh_expert_counts' in self.routing_stats:
            counts = self.routing_stats['bh_expert_counts']
            stats.update({
                'bh_would_select_mean': sum(counts) / len(counts) if counts else 0,
                'total_forward_passes': len(counts),
            })

        return stats

    def reset_stats(self):
        """Clear all collected statistics."""
        self.routing_stats.clear()
        print("[BH Integration] Statistics reset.")

    def __enter__(self):
        """Context manager support: patch on enter."""
        self.patch_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support: unpatch on exit."""
        self.unpatch_model()
        return False


def demonstrate_integration():
    """
    Demonstration of BH routing integration.

    Note: This requires transformers and a working OLMoE model.
    Run this only if you have the model available.
    """
    print("=" * 70)
    print("BH ROUTING INTEGRATION DEMONSTRATION")
    print("=" * 70)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("❌ transformers not available. Install with: pip install transformers")
        return

    # Try to load model
    model_name = "allenai/OLMoE-1B-7B-0924"
    print(f"\nLoading model: {model_name}")
    print("(This may take a while on first run...)")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"  # Use CPU for demo
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ Model loaded")
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        print("This is expected if running without internet or insufficient memory.")
        return

    # Test input
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Test 1: Original routing
    print("\n" + "-" * 70)
    print("Test 1: Original Top-K Routing")
    print("-" * 70)

    with torch.no_grad():
        outputs_original = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False
        )

    print(f"Output: {tokenizer.decode(outputs_original[0])}")

    # Test 2: BH routing (patch mode)
    print("\n" + "-" * 70)
    print("Test 2: BH Routing (Patch Mode)")
    print("-" * 70)

    integrator = BHRoutingIntegration(
        model,
        alpha=0.05,
        temperature=1.0,
        max_k=8,
        mode='patch',
        collect_stats=True
    )

    integrator.patch_model()

    with torch.no_grad():
        outputs_bh = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False
        )

    print(f"Output: {tokenizer.decode(outputs_bh[0])}")

    # Get statistics
    stats = integrator.get_routing_stats()
    print(f"\nRouting Statistics:")
    print(f"  Mean experts per token: {stats.get('mean_experts_per_token', 'N/A'):.2f}")
    print(f"  Forward passes: {stats.get('total_forward_passes', 'N/A')}")

    integrator.unpatch_model()

    # Test 3: Analyze mode
    print("\n" + "-" * 70)
    print("Test 3: Analysis Mode (Simulation)")
    print("-" * 70)

    integrator_analyze = BHRoutingIntegration(
        model,
        alpha=0.05,
        max_k=8,
        mode='analyze',
        collect_stats=True
    )

    integrator_analyze.patch_model()

    with torch.no_grad():
        outputs_analyze = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False
        )

    print(f"Output: {tokenizer.decode(outputs_analyze[0])}")

    stats_analyze = integrator_analyze.get_routing_stats()
    print(f"\nSimulated BH Statistics:")
    print(f"  BH would select (mean): {stats_analyze.get('bh_would_select_mean', 'N/A'):.2f} experts")

    integrator_analyze.unpatch_model()

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    # Run demonstration if executed directly
    print(__doc__)
    print("\nAttempting demonstration...")
    print("(Requires: torch, transformers, internet connection, ~3GB RAM)\n")

    demonstrate_integration()
