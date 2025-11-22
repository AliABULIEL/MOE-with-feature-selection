"""
OLMoE Routing Experiments Framework
====================================

This module provides a comprehensive framework for testing how different expert
routing strategies affect inference quality in OLMoE (Open Language Model with Experts).

Key Components:
- RoutingStrategy: Base class for implementing different routing strategies
- RegularRouting, NormalizedRouting, UniformRouting, AdaptiveRouting: Strategy implementations
- RoutingExperimentRunner: Main experiment orchestration class

Usage:
    runner = RoutingExperimentRunner()
    results_df = runner.run_all_experiments()
    runner.visualize_results(results_df)
    runner.generate_report(results_df)
"""

import torch
import torch.nn.functional as F
from transformers import OlmoeForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RoutingConfig:
    """Configuration for a routing experiment."""
    num_experts: int
    strategy: str
    description: str

    def get_name(self) -> str:
        """Get a unique identifier for this configuration."""
        return f"{self.num_experts}experts_{self.strategy}"


@dataclass
class ExperimentResults:
    """Results from a single routing experiment."""
    config: RoutingConfig
    dataset: str
    perplexity: float
    token_accuracy: float
    loss: float
    inference_time: float
    tokens_per_second: float
    avg_time_per_sample: float
    num_samples: int
    total_tokens: int
    avg_experts_used: float
    avg_max_weight: float
    avg_entropy: float
    weight_concentration: float
    unique_experts_activated: int

    def to_dict(self) -> Dict:
        """Convert results to dictionary with config name."""
        result_dict = asdict(self)
        result_dict['config'] = self.config.get_name()
        result_dict['num_experts'] = self.config.num_experts
        result_dict['strategy'] = self.config.strategy
        return result_dict


class RoutingStrategy:
    """Base class for expert routing strategies."""

    def __init__(self, num_experts: int):
        """
        Initialize routing strategy.

        Args:
            num_experts: Target number of experts to route to
        """
        self.num_experts = num_experts
        self.stats = {
            'max_weights': [],
            'entropies': [],
            'concentrations': [],
            'experts_used': set()
        }

    def route(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply routing strategy to router logits.

        Args:
            logits: Router logits tensor [batch_size, seq_len, num_total_experts]

        Returns:
            Tuple of (expert_indices, expert_weights)
        """
        raise NotImplementedError("Subclasses must implement route()")

    def update_stats(self, expert_indices: torch.Tensor, expert_weights: torch.Tensor):
        """
        Update routing statistics.

        Args:
            expert_indices: Selected expert indices
            expert_weights: Corresponding expert weights
        """
        # Convert to numpy and flatten
        # Note: Convert to supported dtypes first (bfloat16 not supported by numpy)
        indices_np = expert_indices.detach().cpu().to(torch.int64).numpy().flatten()
        weights_np = expert_weights.detach().cpu().to(torch.float32).numpy()

        # Reshape weights if needed for per-token statistics
        if weights_np.ndim > 1:
            # Flatten batch and sequence dimensions
            original_shape = weights_np.shape
            weights_np = weights_np.reshape(-1, original_shape[-1])

        # Track unique experts
        self.stats['experts_used'].update(indices_np.tolist())

        # Calculate per-token statistics
        for weight_vec in weights_np:
            # Max weight
            self.stats['max_weights'].append(float(np.max(weight_vec)))

            # Entropy: -sum(w * log(w + eps))
            weight_vec_safe = weight_vec + 1e-10
            entropy = -np.sum(weight_vec_safe * np.log(weight_vec_safe))
            self.stats['entropies'].append(float(entropy))

            # Concentration: max(w) / sum(w)
            concentration = np.max(weight_vec) / (np.sum(weight_vec) + 1e-10)
            self.stats['concentrations'].append(float(concentration))

    def get_summary_stats(self) -> Dict:
        """Get summary statistics for this routing strategy."""
        return {
            'avg_max_weight': np.mean(self.stats['max_weights']) if self.stats['max_weights'] else 0.0,
            'avg_entropy': np.mean(self.stats['entropies']) if self.stats['entropies'] else 0.0,
            'avg_concentration': np.mean(self.stats['concentrations']) if self.stats['concentrations'] else 0.0,
            'unique_experts': len(self.stats['experts_used'])
        }


class RegularRouting(RoutingStrategy):
    """Regular top-k routing with softmax probabilities."""

    def route(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply regular softmax-based top-k routing."""
        probs = F.softmax(logits, dim=-1)
        k = min(self.num_experts, logits.shape[-1])
        weights, indices = torch.topk(probs, k=k, dim=-1)
        self.update_stats(indices, weights)
        return indices, weights


class NormalizedRouting(RoutingStrategy):
    """Top-k routing with re-normalized weights."""

    def route(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply top-k routing with weight re-normalization."""
        probs = F.softmax(logits, dim=-1)
        k = min(self.num_experts, logits.shape[-1])
        weights, indices = torch.topk(probs, k=k, dim=-1)

        # KEY DIFFERENCE: Re-normalize selected weights to sum to 1
        weights = weights / weights.sum(dim=-1, keepdim=True)

        self.update_stats(indices, weights)
        return indices, weights


class UniformRouting(RoutingStrategy):
    """Top-k routing with uniform weights."""

    def route(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply top-k routing with equal weights for selected experts."""
        probs = F.softmax(logits, dim=-1)
        k = min(self.num_experts, logits.shape[-1])
        _, indices = torch.topk(probs, k=k, dim=-1)

        # KEY DIFFERENCE: Equal weights for all selected experts
        weights = torch.ones_like(indices, dtype=torch.float32) / k

        self.update_stats(indices, weights)
        return indices, weights


class AdaptiveRouting(RoutingStrategy):
    """Adaptive routing that adjusts expert count based on confidence."""

    def __init__(self, num_experts: int, thresholds: Optional[Dict] = None):
        """
        Initialize adaptive routing.

        Args:
            num_experts: Maximum number of experts
            thresholds: Optional dict of confidence thresholds
        """
        super().__init__(num_experts)
        self.thresholds = thresholds or {
            'high_concentration': {'max_prob': 0.7, 'min_experts': 4},
            'medium_concentration': {'max_prob': 0.5, 'min_experts': 8},
            'low_concentration': {'max_prob': 0.3, 'min_experts': 16},
            'default': {'min_experts': num_experts}
        }

    def determine_expert_count(self, logits: torch.Tensor) -> int:
        """
        Determine how many experts to use based on confidence.

        High confidence (max_prob > 0.7) → fewer experts (4)
        Medium confidence (max_prob > 0.5) → moderate experts (8)
        Low confidence (max_prob > 0.3) → more experts (16)
        Very low confidence → all experts
        """
        probs = F.softmax(logits, dim=-1)
        max_prob = torch.max(probs).item()

        if max_prob > 0.7:
            return min(4, self.num_experts)
        elif max_prob > 0.5:
            return min(8, self.num_experts)
        elif max_prob > 0.3:
            return min(16, self.num_experts)
        return self.num_experts

    def route(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply adaptive routing based on routing confidence."""
        k = min(self.determine_expert_count(logits), logits.shape[-1])
        probs = F.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, k=k, dim=-1)

        # Re-normalize weights
        weights = weights / weights.sum(dim=-1, keepdim=True)

        self.update_stats(indices, weights)
        return indices, weights


class ModelPatchingUtils:
    """Utilities for patching OLMoE model with custom routing logic."""

    @staticmethod
    def custom_select_experts(
        router_logits: torch.Tensor,
        top_k: int,
        num_experts: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Custom expert selection with uniform weights (internal logging version).

        This function is designed to be integrated directly into the model's forward pass,
        enabling internal logging of router_logits while applying custom routing logic.

        Args:
            router_logits: [tokens, num_experts] - Raw routing scores
            top_k: Number of experts to select
            num_experts: Total number of experts

        Returns:
            routing_weights: [tokens, top_k] - Uniform weights (1/top_k)
            selected_experts: [tokens, top_k] - Selected expert indices
        """
        # Convert logits to probabilities
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        # Select top-k experts based on probabilities
        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

        # Give each selected expert EQUAL probability (uniform routing)
        routing_weights = torch.ones_like(selected_experts, dtype=torch.float)
        routing_weights /= top_k

        return routing_weights.to(router_logits.dtype), selected_experts

    @staticmethod
    def create_patched_forward(top_k: int, num_experts: int):
        """
        Create a patched forward pass for OLMoE MLP layers.

        This replaces the standard forward pass with one that:
        1. Uses custom_select_experts for routing
        2. Returns router_logits for internal logging
        3. Enables analysis of routing decisions

        Args:
            top_k: Number of experts to select
            num_experts: Total number of experts available

        Returns:
            Patched forward function
        """
        def new_forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)

            # Get router logits: (batch * sequence_length, n_experts)
            router_logits = self.gate(hidden_states)

            # Use custom expert selection
            routing_weights, selected_experts = ModelPatchingUtils.custom_select_experts(
                router_logits,
                top_k=top_k,
                num_experts=num_experts
            )

            # Initialize output
            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )

            # Create expert mask for efficient indexing
            expert_mask = torch.nn.functional.one_hot(
                selected_experts,
                num_classes=num_experts
            ).permute(2, 1, 0)

            # Process each expert
            for expert_idx in range(num_experts):
                expert_layer = self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx])

                if top_x.numel() == 0:
                    continue

                # Compute expert output with routing weights
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

                # Accumulate results
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

            # Reshape output
            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

            # Return both output and router_logits for internal logging
            return final_hidden_states, router_logits

        return new_forward

    @staticmethod
    def patch_model(model, top_k: Optional[int] = None):
        """
        Patch all MoE layers in the model with custom routing.

        Args:
            model: OLMoE model instance
            top_k: Number of experts to use (None = use model default)

        Returns:
            Number of layers patched
        """
        if top_k is None:
            top_k = getattr(model.config, 'num_experts_per_tok', 8)

        num_experts = getattr(model.config, 'num_local_experts', 64)

        logger.info(f"Patching model with custom routing: top_k={top_k}, num_experts={num_experts}")

        patched_layers = 0

        for layer in model.model.layers:
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                # Create and apply patched forward
                layer.mlp.forward = ModelPatchingUtils.create_patched_forward(
                    top_k,
                    num_experts
                ).__get__(layer.mlp, layer.mlp.__class__)

                patched_layers += 1

        logger.info(f"✅ Patched {patched_layers} MoE layers with custom routing")
        return patched_layers

    @staticmethod
    def unpatch_model(model, original_forwards: Dict):
        """
        Restore original forward passes (for testing purposes).

        Args:
            model: OLMoE model instance
            original_forwards: Dictionary mapping layer indices to original forwards
        """
        for idx, layer in enumerate(model.model.layers):
            if idx in original_forwards:
                layer.mlp.forward = original_forwards[idx]

        logger.info("✅ Model unpatched - original forwards restored")


class RoutingExperimentRunner:
    """Main class for orchestrating routing experiments."""

    def __init__(
        self,
        model_name: str = "allenai/OLMoE-1B-7B-0924",
        device: str = "auto",
        output_dir: str = "./routing_experiments",
        use_model_patching: bool = False
    ):
        """
        Initialize experiment runner.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('auto', 'cuda', 'cpu')
            output_dir: Directory for saving results
            use_model_patching: If True, patch model with custom forward pass
        """
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"experiment_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.logs_dir = self.output_dir / "logs"
        self.viz_dir = self.output_dir / "visualizations"
        self.logs_dir.mkdir(exist_ok=True)
        self.viz_dir.mkdir(exist_ok=True)

        logger.info(f"Output directory: {self.output_dir}")

        # Load model and tokenizer
        logger.info(f"Loading model: {model_name}")
        self.model = OlmoeForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map=device
        )
        self.model.eval()

        # Hook management for capturing router outputs
        self.router_hooks = []
        self.logged_routing_data = []

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Store original configuration
        self.original_num_experts = getattr(
            self.model.config,
            'num_experts_per_tok',
            8
        )
        logger.info(f"Original num_experts_per_tok: {self.original_num_experts}")

        # Model patching support
        self.use_model_patching = use_model_patching
        if use_model_patching:
            logger.info("🔧 Model patching mode enabled")
            logger.info("   Custom forward pass will be used for routing")

        # Strategy factory
        self.strategy_factory = {
            'regular': RegularRouting,
            'normalized': NormalizedRouting,
            'uniform': UniformRouting,
            'adaptive': AdaptiveRouting,
            'baseline': RegularRouting  # Alias for regular routing (used in two-phase experiments)
        }

    def _set_expert_count(self, num_experts: int):
        """
        Update model configuration to use specified number of experts.

        Args:
            num_experts: Number of experts to route to
        """
        # Update config
        if hasattr(self.model.config, 'num_experts_per_tok'):
            self.model.config.num_experts_per_tok = num_experts

        # If using model patching, re-patch with new expert count
        if self.use_model_patching:
            ModelPatchingUtils.patch_model(self.model, top_k=num_experts)
        else:
            # Standard config update
            # Update all layers - try multiple possible attribute names
            for layer in self.model.model.layers:
                if hasattr(layer, 'mlp'):
                    mlp = layer.mlp

                    # Try different attribute names
                    if hasattr(mlp, 'num_experts_per_tok'):
                        mlp.num_experts_per_tok = num_experts
                    if hasattr(mlp, 'top_k'):
                        mlp.top_k = num_experts

                    # Try gate attributes
                    if hasattr(mlp, 'gate'):
                        if hasattr(mlp.gate, 'top_k'):
                            mlp.gate.top_k = num_experts
                        if hasattr(mlp.gate, 'num_experts_per_tok'):
                            mlp.gate.num_experts_per_tok = num_experts

        logger.info(f"Set expert count to: {num_experts}")

    def _register_router_hooks(self, top_k: int = 8):
        """
        Register forward hooks on layer.mlp.gate to capture router outputs.

        This is the reliable way to get routing logits - more robust than
        output_router_logits=True parameter which doesn't always work.

        Args:
            top_k: Number of top experts to select
        """
        # Clear any existing hooks first
        self._clear_router_hooks()

        # Clear logged data
        self.logged_routing_data = []

        def logging_hook_router(module, input, output, layer_index, k=8):
            """Hook to capture router outputs from gate module"""
            try:
                router_logits = output
                # Apply softmax to get probabilities
                softmax_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
                # Get top-k experts
                topk_weights, topk_indices = torch.topk(softmax_weights, k, dim=-1)

                self.logged_routing_data.append({
                    "layer": layer_index,
                    "router_logits": router_logits.detach(),
                    "expert_indices": topk_indices.detach(),
                    "softmax_weights": topk_weights.detach(),
                })
            except Exception as e:
                logger.warning(f"Error in hook for layer {layer_index}: {e}")

        # Register hooks on each layer's gate
        for i, layer in enumerate(self.model.model.layers):
            try:
                router_module = layer.mlp.gate
                hook_handle = router_module.register_forward_hook(
                    lambda m, inp, out, idx=i: logging_hook_router(m, inp, out, idx, k=top_k)
                )
                self.router_hooks.append(hook_handle)
            except AttributeError:
                logger.warning(f"Layer {i} has no 'gate' module. Skipping hook registration.")

        logger.debug(f"Registered {len(self.router_hooks)} router hooks")

    def _clear_router_hooks(self):
        """Remove all registered router hooks."""
        for hook in self.router_hooks:
            hook.remove()
        self.router_hooks = []
        logger.debug("Cleared all router hooks")

    def load_dataset(
        self,
        dataset_name: str,
        split: str = "test",
        max_samples: int = 500
    ) -> List[str]:
        """
        Load evaluation dataset.

        Args:
            dataset_name: Name of dataset ('wikitext', 'lambada', 'piqa')
            split: Dataset split to use
            max_samples: Maximum number of samples to load

        Returns:
            List of text strings
        """
        logger.info(f"Loading dataset: {dataset_name}")

        try:
            if dataset_name == 'wikitext':
                dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
                texts = [item['text'] for item in dataset if item['text'].strip()]

            elif dataset_name == 'lambada':
                dataset = load_dataset('lambada', split=split)
                texts = [item['text'] for item in dataset]

            elif dataset_name == 'piqa':
                dataset = load_dataset('piqa', split='validation')
                texts = [
                    f"{item['goal']} {item['sol1']}"
                    for item in dataset
                ]

            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")

            # Filter and limit
            texts = [t for t in texts if len(t.strip()) > 0]
            texts = texts[:max_samples]

            logger.info(f"Loaded {len(texts)} samples from {dataset_name}")
            return texts

        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            raise

    def evaluate_configuration(
        self,
        config: RoutingConfig,
        texts: List[str],
        dataset_name: str,
        max_length: int = 512,
        save_internal_logs: bool = True
    ) -> ExperimentResults:
        """
        Evaluate a single routing configuration.

        Args:
            config: Routing configuration to test
            texts: List of text samples
            dataset_name: Name of dataset being evaluated
            max_length: Maximum sequence length
            save_internal_logs: Whether to save detailed router_logits logs

        Returns:
            ExperimentResults with all metrics
        """
        logger.info(f"Evaluating: {config.get_name()} on {dataset_name}")

        # Set expert count
        self._set_expert_count(config.num_experts)

        # Register router hooks to capture routing data
        self._register_router_hooks(top_k=config.num_experts)

        # Create routing strategy
        strategy = self.strategy_factory[config.strategy](config.num_experts)

        # Initialize metrics
        total_loss = 0.0
        total_tokens = 0
        correct_predictions = 0
        num_samples = 0
        num_errors = 0

        # Track detailed errors (store first 10)
        error_details = []
        max_errors_to_track = 10

        # Internal routing logs storage
        internal_routing_logs = {
            'config': config.get_name(),
            'strategy': config.strategy,
            'num_experts': config.num_experts,
            'dataset': dataset_name,
            'samples': []
        }

        start_time = time.time()

        # Evaluate on all texts (no nested progress bar - using main progress bar)
        with torch.no_grad():
            for text in texts:
                try:
                    # Tokenize
                    inputs = self.tokenizer(
                        text,
                        return_tensors='pt',
                        truncation=True,
                        max_length=max_length
                    )

                    input_ids = inputs['input_ids'].to(self.model.device)

                    if input_ids.shape[1] < 2:
                        continue

                    # Clear routing data from previous sample
                    self.logged_routing_data = []

                    # Forward pass (hooks will capture routing data automatically)
                    outputs = self.model(
                        input_ids=input_ids,
                        labels=input_ids
                    )

                    # Validate outputs
                    if outputs.loss is None:
                        raise ValueError(f"outputs.loss is None for sample {num_samples}")
                    if outputs.logits is None:
                        raise ValueError(f"outputs.logits is None for sample {num_samples}")

                    # Accumulate loss
                    loss = outputs.loss.item()
                    total_loss += loss * input_ids.shape[1]
                    total_tokens += input_ids.shape[1]

                    # Calculate token accuracy
                    logits = outputs.logits
                    predictions = torch.argmax(logits[:, :-1, :], dim=-1)
                    targets = input_ids[:, 1:]
                    correct_predictions += (predictions == targets).sum().item()

                    # Process hook-captured routing data
                    if len(self.logged_routing_data) > 0:
                        sample_routing_data = {
                            'sample_id': num_samples,
                            'num_tokens': input_ids.shape[1],
                            'loss': loss,
                            'layers': []
                        }

                        for layer_data in self.logged_routing_data:
                            router_logit = layer_data['router_logits']
                            layer_idx = layer_data['layer']

                            # Apply routing strategy
                            expert_indices, expert_weights = strategy.route(router_logit)

                            # Save internal routing data if requested
                            if save_internal_logs:
                                # Convert to numpy for storage (sample first few tokens)
                                # Handle both 2D [tokens, experts] and 3D [batch, seq, experts] tensors
                                if router_logit.dim() == 2:
                                    # 2D: [tokens, num_experts] - flattened batch*seq
                                    max_tokens_to_log = min(10, router_logit.shape[0])
                                    layer_log_data = {
                                        'layer': layer_idx,
                                        'router_logits_shape': list(router_logit.shape),
                                        'selected_experts': expert_indices[:max_tokens_to_log, :].cpu().to(torch.int64).numpy().tolist(),
                                        'expert_weights': expert_weights[:max_tokens_to_log, :].cpu().to(torch.float32).numpy().tolist(),
                                        'router_logits_sample': router_logit[:max_tokens_to_log, :].cpu().to(torch.float32).numpy().tolist()
                                    }
                                else:
                                    # 3D: [batch, seq_len, num_experts]
                                    max_tokens_to_log = min(10, router_logit.shape[1])
                                    layer_log_data = {
                                        'layer': layer_idx,
                                        'router_logits_shape': list(router_logit.shape),
                                        'selected_experts': expert_indices[:, :max_tokens_to_log, :].cpu().to(torch.int64).numpy().tolist(),
                                        'expert_weights': expert_weights[:, :max_tokens_to_log, :].cpu().to(torch.float32).numpy().tolist(),
                                        'router_logits_sample': router_logit[:, :max_tokens_to_log, :].cpu().to(torch.float32).numpy().tolist()
                                    }
                                sample_routing_data['layers'].append(layer_log_data)

                        if save_internal_logs:
                            internal_routing_logs['samples'].append(sample_routing_data)
                    else:
                        logger.warning(f"Sample {num_samples}: No routing data captured by hooks")

                    num_samples += 1

                except Exception as e:
                    num_errors += 1

                    # Capture detailed error information for first few errors
                    if len(error_details) < max_errors_to_track:
                        import traceback
                        error_info = {
                            'sample_index': num_samples,
                            'error_type': type(e).__name__,
                            'error_message': str(e),
                            'text_preview': text[:100] if text else "Empty text",
                            'text_length': len(text) if text else 0,
                            'traceback': traceback.format_exc()
                        }
                        error_details.append(error_info)
                    continue

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Log detailed errors if any occurred
        if num_errors > 0:
            logger.error(f"=" * 70)
            logger.error(f"⚠️  EVALUATION ERRORS: Skipped {num_errors}/{len(texts)} samples")
            logger.error(f"=" * 70)

            if error_details:
                logger.error(f"\n📋 Detailed Error Information (first {len(error_details)} errors):\n")

                for i, error in enumerate(error_details, 1):
                    logger.error(f"\n{'='*70}")
                    logger.error(f"Error #{i}:")
                    logger.error(f"  Sample Index: {error['sample_index']}")
                    logger.error(f"  Error Type: {error['error_type']}")
                    logger.error(f"  Error Message: {error['error_message']}")
                    logger.error(f"  Text Length: {error['text_length']} characters")
                    logger.error(f"  Text Preview: {error['text_preview']}")
                    logger.error(f"\n  Traceback:")
                    for line in error['traceback'].split('\n'):
                        logger.error(f"    {line}")
                    logger.error(f"{'='*70}")

                # Log summary of error types
                error_types = {}
                for error in error_details:
                    error_type = error['error_type']
                    error_types[error_type] = error_types.get(error_type, 0) + 1

                logger.error(f"\n📊 Error Type Summary:")
                for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                    logger.error(f"  {error_type}: {count} occurrence(s)")

                logger.error(f"\n💡 Troubleshooting Tips:")
                logger.error(f"  1. Check if router hooks registered successfully on layer.mlp.gate")
                logger.error(f"  2. Verify device compatibility (CPU vs GPU)")
                logger.error(f"  3. Check for empty or malformed text samples")
                logger.error(f"  4. Verify model outputs contain expected fields (loss, logits)")
                logger.error(f"  5. Check if routing data is being captured by hooks")
                logger.error(f"  6. Run diagnose_errors.py for detailed testing")
                logger.error(f"=" * 70)
            else:
                logger.error(f"  No detailed error information captured")
                logger.error(f"=" * 70)

        # Calculate final metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss)
        token_accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0.0
        tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0.0
        avg_time_per_sample = elapsed_time / num_samples if num_samples > 0 else 0.0

        # Get routing statistics
        routing_stats = strategy.get_summary_stats()

        # Create results
        results = ExperimentResults(
            config=config,
            dataset=dataset_name,
            perplexity=float(perplexity),
            token_accuracy=float(token_accuracy),
            loss=float(avg_loss),
            inference_time=float(elapsed_time),
            tokens_per_second=float(tokens_per_second),
            avg_time_per_sample=float(avg_time_per_sample),
            num_samples=num_samples,
            total_tokens=total_tokens,
            avg_experts_used=float(config.num_experts),
            avg_max_weight=float(routing_stats['avg_max_weight']),
            avg_entropy=float(routing_stats['avg_entropy']),
            weight_concentration=float(routing_stats['avg_concentration']),
            unique_experts_activated=routing_stats['unique_experts']
        )

        # Save individual log file (summary)
        log_file = self.logs_dir / f"{config.get_name()}_{dataset_name}.json"
        with open(log_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)

        # Save detailed internal routing logs to unique file
        if save_internal_logs and internal_routing_logs['samples']:
            internal_log_file = self.logs_dir / f"{config.get_name()}_{dataset_name}_internal_routing.json"

            # Add summary statistics to internal logs
            internal_routing_logs['summary'] = {
                'total_samples': num_samples,
                'total_tokens': total_tokens,
                'perplexity': float(perplexity),
                'token_accuracy': float(token_accuracy),
                'avg_entropy': float(routing_stats['avg_entropy']),
                'unique_experts_activated': routing_stats['unique_experts'],
                'expert_utilization': routing_stats['unique_experts'] / 64
            }

            with open(internal_log_file, 'w') as f:
                json.dump(internal_routing_logs, f, indent=2)

            logger.info(f"Saved internal routing logs to: {internal_log_file}")

        logger.info(
            f"Results: PPL={perplexity:.2f}, Acc={token_accuracy:.4f}, "
            f"Speed={tokens_per_second:.1f} tok/s"
        )

        return results

    def run_all_experiments(
        self,
        expert_counts: List[int] = [4, 8, 16, 32, 64],
        strategies: List[str] = ['regular', 'normalized', 'uniform'],
        datasets: List[str] = ['wikitext', 'lambada'],
        max_samples: int = 500
    ) -> pd.DataFrame:
        """
        Run all experiment combinations.

        Args:
            expert_counts: List of expert counts to test
            strategies: List of strategy names to test
            datasets: List of datasets to evaluate on
            max_samples: Maximum samples per dataset

        Returns:
            DataFrame with all results
        """
        logger.info("=" * 70)
        logger.info("STARTING ROUTING EXPERIMENTS")
        logger.info("=" * 70)

        # Create all configurations
        configs = []
        for num_experts in expert_counts:
            for strategy in strategies:
                config = RoutingConfig(
                    num_experts=num_experts,
                    strategy=strategy,
                    description=f"{strategy.capitalize()} routing with {num_experts} experts"
                )
                configs.append(config)

        logger.info(f"Total experiments: {len(configs) * len(datasets)}")
        logger.info(f"Configurations: {len(configs)}")
        logger.info(f"Datasets: {len(datasets)}")

        # Run all experiments
        all_results = []

        # Create single progress bar for all experiments
        total_experiments = len(configs) * len(datasets)
        pbar = tqdm(total=total_experiments, desc="Running Experiments", unit="exp")

        for dataset_name in datasets:
            # Load dataset once
            texts = self.load_dataset(dataset_name, max_samples=max_samples)

            # Run all configs on this dataset
            for config in configs:
                try:
                    pbar.set_description(f"Running {config.get_name()} on {dataset_name}")
                    results = self.evaluate_configuration(
                        config, texts, dataset_name
                    )
                    all_results.append(results)
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Experiment failed: {config.get_name()} on {dataset_name}: {e}")
                    pbar.update(1)

        pbar.close()

        # Convert to DataFrame
        df = pd.DataFrame([r.to_dict() for r in all_results])

        # Save results
        csv_file = self.output_dir / "all_results.csv"
        json_file = self.output_dir / "all_results.json"

        df.to_csv(csv_file, index=False)
        df.to_json(json_file, orient='records', indent=2)

        logger.info(f"Results saved to: {csv_file}")
        logger.info("=" * 70)

        return df

    def visualize_results(self, df: pd.DataFrame):
        """
        Create comprehensive visualizations of results.

        Args:
            df: DataFrame with experiment results
        """
        logger.info("Generating visualizations...")

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (20, 24)

        # Create comprehensive figure with 12 subplots
        fig = plt.figure(figsize=(20, 24))

        # 1. Perplexity vs Expert Count (line per strategy)
        ax1 = plt.subplot(4, 3, 1)
        for strategy in df['strategy'].unique():
            strategy_df = df[df['strategy'] == strategy].groupby('num_experts')['perplexity'].mean()
            ax1.plot(strategy_df.index, strategy_df.values, marker='o', label=strategy, linewidth=2)
        ax1.set_xlabel('Number of Experts')
        ax1.set_ylabel('Perplexity (↓ better)')
        ax1.set_title('Perplexity vs Expert Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Accuracy vs Expert Count
        ax2 = plt.subplot(4, 3, 2)
        for strategy in df['strategy'].unique():
            strategy_df = df[df['strategy'] == strategy].groupby('num_experts')['token_accuracy'].mean()
            ax2.plot(strategy_df.index, strategy_df.values, marker='o', label=strategy, linewidth=2)
        ax2.set_xlabel('Number of Experts')
        ax2.set_ylabel('Token Accuracy (↑ better)')
        ax2.set_title('Token Accuracy vs Expert Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Speed vs Expert Count
        ax3 = plt.subplot(4, 3, 3)
        for strategy in df['strategy'].unique():
            strategy_df = df[df['strategy'] == strategy].groupby('num_experts')['tokens_per_second'].mean()
            ax3.plot(strategy_df.index, strategy_df.values, marker='o', label=strategy, linewidth=2)
        ax3.set_xlabel('Number of Experts')
        ax3.set_ylabel('Tokens/Second (↑ better)')
        ax3.set_title('Inference Speed vs Expert Count')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Perplexity Heatmap
        ax4 = plt.subplot(4, 3, 4)
        pivot_ppl = df.groupby(['num_experts', 'strategy'])['perplexity'].mean().reset_index().pivot(
            index='strategy', columns='num_experts', values='perplexity'
        )
        sns.heatmap(pivot_ppl, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax4)
        ax4.set_title('Perplexity Heatmap')

        # 5. Accuracy Heatmap
        ax5 = plt.subplot(4, 3, 5)
        pivot_acc = df.groupby(['num_experts', 'strategy'])['token_accuracy'].mean().reset_index().pivot(
            index='strategy', columns='num_experts', values='token_accuracy'
        )
        sns.heatmap(pivot_acc, annot=True, fmt='.4f', cmap='RdYlGn', ax=ax5)
        ax5.set_title('Token Accuracy Heatmap')

        # 6. Entropy by Strategy
        ax6 = plt.subplot(4, 3, 6)
        df.boxplot(column='avg_entropy', by='strategy', ax=ax6)
        ax6.set_xlabel('Strategy')
        ax6.set_ylabel('Average Entropy')
        ax6.set_title('Routing Entropy Distribution')
        plt.sca(ax6)
        plt.xticks(rotation=45)

        # 7. Expert Utilization
        ax7 = plt.subplot(4, 3, 7)
        df.boxplot(column='unique_experts_activated', by='strategy', ax=ax7)
        ax7.set_xlabel('Strategy')
        ax7.set_ylabel('Unique Experts Used')
        ax7.set_title('Expert Utilization')
        plt.sca(ax7)
        plt.xticks(rotation=45)

        # 8. Speed-Quality Scatter
        ax8 = plt.subplot(4, 3, 8)
        for strategy in df['strategy'].unique():
            strategy_df = df[df['strategy'] == strategy]
            ax8.scatter(
                strategy_df['perplexity'],
                strategy_df['tokens_per_second'],
                label=strategy,
                s=100,
                alpha=0.6
            )
        ax8.set_xlabel('Perplexity (↓ better)')
        ax8.set_ylabel('Tokens/Second (↑ better)')
        ax8.set_title('Speed vs Quality Trade-off')
        ax8.legend()
        ax8.grid(True, alpha=0.3)

        # 9. Relative Improvement
        ax9 = plt.subplot(4, 3, 9)
        baseline_ppl = df[df['strategy'] == 'regular']['perplexity'].mean()
        improvements = []
        for strategy in df['strategy'].unique():
            strategy_ppl = df[df['strategy'] == strategy]['perplexity'].mean()
            improvement = ((baseline_ppl - strategy_ppl) / baseline_ppl) * 100
            improvements.append({'strategy': strategy, 'improvement': improvement})
        imp_df = pd.DataFrame(improvements)
        ax9.bar(imp_df['strategy'], imp_df['improvement'])
        ax9.set_xlabel('Strategy')
        ax9.set_ylabel('% Improvement vs Regular')
        ax9.set_title('Relative Perplexity Improvement')
        ax9.axhline(y=0, color='r', linestyle='--')
        plt.sca(ax9)
        plt.xticks(rotation=45)

        # 10. Per-Dataset Perplexity Comparison
        ax10 = plt.subplot(4, 3, 10)
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset].groupby('strategy')['perplexity'].mean()
            ax10.bar(
                np.arange(len(dataset_df)) + 0.35 * list(df['dataset'].unique()).index(dataset),
                dataset_df.values,
                width=0.35,
                label=dataset
            )
        ax10.set_xlabel('Strategy')
        ax10.set_ylabel('Perplexity')
        ax10.set_title('Perplexity by Dataset')
        ax10.set_xticks(np.arange(len(df['strategy'].unique())))
        ax10.set_xticklabels(df['strategy'].unique())
        ax10.legend()
        plt.sca(ax10)
        plt.xticks(rotation=45)

        # 11. Weight Concentration
        ax11 = plt.subplot(4, 3, 11)
        for strategy in df['strategy'].unique():
            strategy_df = df[df['strategy'] == strategy].groupby('num_experts')['weight_concentration'].mean()
            ax11.plot(strategy_df.index, strategy_df.values, marker='o', label=strategy, linewidth=2)
        ax11.set_xlabel('Number of Experts')
        ax11.set_ylabel('Weight Concentration')
        ax11.set_title('Weight Concentration vs Expert Count')
        ax11.legend()
        ax11.grid(True, alpha=0.3)

        # 12. Inference Time Distribution
        ax12 = plt.subplot(4, 3, 12)
        df.boxplot(column='avg_time_per_sample', by='num_experts', ax=ax12)
        ax12.set_xlabel('Number of Experts')
        ax12.set_ylabel('Time per Sample (s)')
        ax12.set_title('Inference Time Distribution')

        plt.tight_layout()

        # Save figures
        png_file = self.viz_dir / "comprehensive_analysis.png"
        pdf_file = self.viz_dir / "comprehensive_analysis.pdf"

        plt.savefig(png_file, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_file, bbox_inches='tight')
        plt.close()

        logger.info(f"Visualizations saved to: {self.viz_dir}")

    def generate_report(self, df: pd.DataFrame):
        """
        Generate markdown report with experiment results.

        Args:
            df: DataFrame with experiment results
        """
        logger.info("Generating report...")

        report_file = self.output_dir / "EXPERIMENT_REPORT.md"

        with open(report_file, 'w') as f:
            f.write("# OLMoE Routing Experiments Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")

            best_ppl = df.loc[df['perplexity'].idxmin()]
            best_acc = df.loc[df['token_accuracy'].idxmax()]
            best_speed = df.loc[df['tokens_per_second'].idxmax()]

            f.write("### Best Configurations\n\n")
            f.write(f"- **Best Perplexity:** {best_ppl['config']} (PPL: {best_ppl['perplexity']:.2f})\n")
            f.write(f"- **Best Accuracy:** {best_acc['config']} (Acc: {best_acc['token_accuracy']:.4f})\n")
            f.write(f"- **Best Speed:** {best_speed['config']} (Speed: {best_speed['tokens_per_second']:.1f} tok/s)\n\n")

            # Results by Dataset
            f.write("## Results by Dataset\n\n")
            for dataset in df['dataset'].unique():
                f.write(f"### {dataset.upper()}\n\n")
                dataset_df = df[df['dataset'] == dataset].sort_values('perplexity')
                f.write(dataset_df[['config', 'perplexity', 'token_accuracy', 'tokens_per_second']].to_markdown(index=False))
                f.write("\n\n")

            # Strategy Analysis
            f.write("## Strategy Analysis\n\n")
            for strategy in df['strategy'].unique():
                f.write(f"### {strategy.upper()} Strategy\n\n")
                strategy_df = df[df['strategy'] == strategy].groupby('num_experts').agg({
                    'perplexity': 'mean',
                    'token_accuracy': 'mean',
                    'tokens_per_second': 'mean'
                }).reset_index()
                f.write(strategy_df.to_markdown(index=False))
                f.write("\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("Based on the experimental results:\n\n")
            f.write("1. **For Quality:** Use the configuration with lowest perplexity\n")
            f.write("2. **For Speed:** Consider configurations with higher expert counts if speed is critical\n")
            f.write("3. **For Balance:** Evaluate the speed-quality trade-off plot\n\n")

            # Files
            f.write("## Output Files\n\n")
            f.write(f"- Results CSV: `all_results.csv`\n")
            f.write(f"- Results JSON: `all_results.json`\n")
            f.write(f"- Visualizations: `visualizations/comprehensive_analysis.png`\n")
            f.write(f"- Individual logs: `logs/` directory\n\n")

        logger.info(f"Report saved to: {report_file}")

    def run_two_phase_experiment(
        self,
        expert_counts: List[int] = [8, 16, 32],
        datasets: List[str] = ['wikitext'],
        max_samples: int = 100,
        routing_modifications: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Run two-phase routing experiment:
        Phase 1: Analyze baseline (model's natural routing)
        Phase 2: Apply modified routing based on insights
        Phase 3: Compare and generate report

        Args:
            expert_counts: List of expert counts to test
            datasets: List of datasets to evaluate on
            max_samples: Maximum samples per dataset
            routing_modifications: List of modifications to test (default: ['uniform', 'normalized'])

        Returns:
            Tuple of (results_df, routing_insights)
        """
        if routing_modifications is None:
            routing_modifications = ['uniform', 'normalized']

        logger.info("=" * 70)
        logger.info("TWO-PHASE ROUTING EXPERIMENT")
        logger.info("=" * 70)

        # ==========================================
        # PHASE 1: BASELINE ANALYSIS
        # ==========================================
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 1: BASELINE ROUTING ANALYSIS")
        logger.info("=" * 70)
        logger.info("Analyzing model's natural routing behavior...\n")

        baseline_results = []
        routing_insights = {}

        # Ensure model patching is disabled for baseline
        original_patching_mode = self.use_model_patching
        self.use_model_patching = False

        total_baseline = len(expert_counts) * len(datasets)
        baseline_pbar = tqdm(total=total_baseline, desc="Phase 1: Baseline Analysis", unit="exp")

        for num_experts in expert_counts:
            self._set_expert_count(num_experts)

            for dataset_name in datasets:
                baseline_pbar.set_description(f"Baseline: {num_experts} experts on {dataset_name}")

                texts = self.load_dataset(dataset_name, max_samples=max_samples)

                # Create config for baseline
                config = RoutingConfig(
                    num_experts=num_experts,
                    strategy='baseline',
                    description=f"Baseline routing with {num_experts} experts"
                )

                # Run evaluation
                results = self.evaluate_configuration(config, texts, dataset_name)
                baseline_results.append(results)

                # Collect detailed insights
                key = f"{num_experts}_{dataset_name}"
                routing_insights[key] = {
                    'num_experts': num_experts,
                    'dataset': dataset_name,
                    'baseline_perplexity': results.perplexity,
                    'baseline_accuracy': results.token_accuracy,
                    'baseline_speed': results.tokens_per_second,
                    'avg_entropy': results.avg_entropy,
                    'avg_max_weight': results.avg_max_weight,
                    'weight_concentration': results.weight_concentration,
                    'unique_experts_used': results.unique_experts_activated,
                    'expert_utilization': results.unique_experts_activated / 64  # OLMoE has 64 experts
                }

                logger.info(
                    f"Baseline ({num_experts} experts, {dataset_name}): "
                    f"PPL={results.perplexity:.2f}, "
                    f"Acc={results.token_accuracy:.4f}, "
                    f"Experts Used={results.unique_experts_activated}/64"
                )

                baseline_pbar.update(1)

        baseline_pbar.close()

        logger.info("\n✅ Phase 1 Complete!")
        logger.info("\n📊 Baseline Insights:")
        for key, insights in routing_insights.items():
            logger.info(
                f"  {insights['num_experts']} experts ({insights['dataset']}): "
                f"Utilization={insights['expert_utilization']:.1%}, "
                f"Entropy={insights['avg_entropy']:.3f}"
            )

        # ==========================================
        # PHASE 2: MODIFIED ROUTING EXPERIMENTS
        # ==========================================
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 2: MODIFIED ROUTING EXPERIMENTS")
        logger.info("=" * 70)
        logger.info("Applying routing modifications based on Phase 1 insights...\n")

        modified_results = []

        total_modified = len(expert_counts) * len(datasets) * len(routing_modifications)
        modified_pbar = tqdm(total=total_modified, desc="Phase 2: Modified Routing", unit="exp")

        for num_experts in expert_counts:
            for dataset_name in datasets:
                texts = self.load_dataset(dataset_name, max_samples=max_samples)

                for modification in routing_modifications:
                    modified_pbar.set_description(
                        f"Modified: {modification} with {num_experts} experts on {dataset_name}"
                    )

                    # Apply modification
                    if modification == 'uniform':
                        # Use model patching for uniform routing
                        self.use_model_patching = True
                        ModelPatchingUtils.patch_model(self.model, top_k=num_experts)
                    else:
                        # Use strategy-based routing (no patching)
                        self.use_model_patching = False
                        self._set_expert_count(num_experts)

                    config = RoutingConfig(
                        num_experts=num_experts,
                        strategy=modification,
                        description=f"{modification.capitalize()} routing with {num_experts} experts"
                    )

                    # Run evaluation
                    results = self.evaluate_configuration(config, texts, dataset_name)
                    modified_results.append(results)

                    # Compare to baseline
                    key = f"{num_experts}_{dataset_name}"
                    baseline_ppl = routing_insights[key]['baseline_perplexity']
                    delta_ppl = results.perplexity - baseline_ppl

                    logger.info(
                        f"Modified ({modification}, {num_experts} experts, {dataset_name}): "
                        f"PPL={results.perplexity:.2f} (Δ{delta_ppl:+.2f}), "
                        f"Acc={results.token_accuracy:.4f}"
                    )

                    modified_pbar.update(1)

        modified_pbar.close()

        # Restore original patching mode
        self.use_model_patching = original_patching_mode

        logger.info("\n✅ Phase 2 Complete!")

        # ==========================================
        # PHASE 3: COMPARATIVE ANALYSIS
        # ==========================================
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 3: COMPARATIVE ANALYSIS")
        logger.info("=" * 70)

        # Combine all results
        all_results = baseline_results + modified_results
        df = pd.DataFrame([r.to_dict() for r in all_results])

        # Save results
        csv_file = self.output_dir / "two_phase_results.csv"
        json_file = self.output_dir / "two_phase_results.json"

        df.to_csv(csv_file, index=False)
        df.to_json(json_file, orient='records', indent=2)

        # Generate comparison report
        self._generate_two_phase_report(df, routing_insights)

        # Generate comparison visualizations
        self._visualize_two_phase_results(df, routing_insights)

        logger.info(f"\n✅ Two-phase experiment complete!")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("=" * 70)

        return df, routing_insights

    def _generate_two_phase_report(self, df: pd.DataFrame, insights: Dict):
        """Generate detailed comparison report for two-phase experiment."""

        report_file = self.output_dir / "two_phase_report.md"

        with open(report_file, 'w') as f:
            f.write("# Two-Phase Routing Experiment Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Experiment Overview\n\n")
            f.write("This experiment uses a two-phase approach:\n\n")
            f.write("1. **Phase 1 (Baseline)**: Analyze model's natural routing behavior\n")
            f.write("2. **Phase 2 (Modified)**: Apply custom routing strategies\n")
            f.write("3. **Phase 3 (Analysis)**: Compare baseline vs modified routing\n\n")

            # Phase 1 Summary
            f.write("## Phase 1: Baseline Analysis\n\n")
            f.write("| Expert Count | Dataset | Perplexity | Accuracy | Experts Used | Utilization |\n")
            f.write("|--------------|---------|------------|----------|--------------|-------------|\n")

            for key, insight in insights.items():
                f.write(
                    f"| {insight['num_experts']} | {insight['dataset']} | "
                    f"{insight['baseline_perplexity']:.2f} | "
                    f"{insight['baseline_accuracy']:.4f} | "
                    f"{insight['unique_experts_used']}/64 | "
                    f"{insight['expert_utilization']:.1%} |\n"
                )

            f.write("\n### Key Observations (Phase 1)\n\n")

            # Analyze patterns
            avg_utilization = np.mean([i['expert_utilization'] for i in insights.values()])
            avg_entropy = np.mean([i['avg_entropy'] for i in insights.values()])

            f.write(f"- **Average Expert Utilization**: {avg_utilization:.1%}\n")
            f.write(f"- **Average Routing Entropy**: {avg_entropy:.3f}\n")

            if avg_utilization < 0.5:
                f.write("- ⚠️ **Low utilization detected** - many experts are underutilized\n")
            if avg_entropy < 1.0:
                f.write("- ⚠️ **Low entropy detected** - routing is highly concentrated\n")

            f.write("\n")

            # Phase 2 Results
            f.write("## Phase 2: Modified Routing Results\n\n")

            for num_experts in df['num_experts'].unique():
                for dataset in df['dataset'].unique():
                    f.write(f"### {num_experts} Experts - {dataset}\n\n")

                    subset = df[(df['num_experts'] == num_experts) & (df['dataset'] == dataset)]

                    f.write("| Strategy | Perplexity | Δ vs Baseline | Accuracy | Δ vs Baseline | Speed (tok/s) |\n")
                    f.write("|----------|------------|---------------|----------|---------------|---------------|\n")

                    baseline_row = subset[subset['strategy'] == 'baseline'].iloc[0]

                    for _, row in subset.iterrows():
                        delta_ppl = row['perplexity'] - baseline_row['perplexity']
                        delta_acc = row['token_accuracy'] - baseline_row['token_accuracy']

                        f.write(
                            f"| {row['strategy']} | {row['perplexity']:.2f} | "
                            f"{delta_ppl:+.2f} | {row['token_accuracy']:.4f} | "
                            f"{delta_acc:+.4f} | {row['tokens_per_second']:.1f} |\n"
                        )

                    f.write("\n")

            # Recommendations
            f.write("## Recommendations\n\n")

            # Find best configuration
            baseline_df = df[df['strategy'] == 'baseline']
            modified_df = df[df['strategy'] != 'baseline']

            best_modified = modified_df.loc[modified_df['perplexity'].idxmin()]
            best_baseline = baseline_df.loc[baseline_df['perplexity'].idxmin()]

            if best_modified['perplexity'] < best_baseline['perplexity']:
                improvement = best_baseline['perplexity'] - best_modified['perplexity']
                f.write(
                    f"✅ **Modified routing improved quality!**\n\n"
                    f"- Best modified: {best_modified['strategy']} with {best_modified['num_experts']} experts\n"
                    f"- Perplexity: {best_modified['perplexity']:.2f} (improved by {improvement:.2f})\n\n"
                )
            else:
                f.write(
                    f"ℹ️ **Baseline routing performs best**\n\n"
                    f"- Modified routing did not improve quality in this experiment\n"
                    f"- Model's natural routing may already be optimal\n\n"
                )

            f.write("## Files Generated\n\n")
            f.write("- `two_phase_results.csv` - All experiment results\n")
            f.write("- `two_phase_results.json` - Results in JSON format\n")
            f.write("- `two_phase_comparison.png` - Visualization comparing phases\n")
            f.write("- `two_phase_report.md` - This report\n\n")

        logger.info(f"Two-phase report saved to: {report_file}")

    def _visualize_two_phase_results(self, df: pd.DataFrame, insights: Dict):
        """Generate visualizations comparing baseline and modified routing."""

        logger.info("Generating two-phase comparison visualizations...")

        fig = plt.figure(figsize=(20, 12))

        # 1. Perplexity Comparison (Baseline vs Modified)
        ax1 = plt.subplot(2, 3, 1)
        strategies = df['strategy'].unique()
        x = np.arange(len(df['num_experts'].unique()))
        width = 0.25

        for i, strategy in enumerate(strategies):
            strategy_data = df[df['strategy'] == strategy].groupby('num_experts')['perplexity'].mean()
            ax1.bar(x + i*width, strategy_data.values, width, label=strategy)

        ax1.set_xlabel('Number of Experts')
        ax1.set_ylabel('Perplexity (↓ better)')
        ax1.set_title('Perplexity: Baseline vs Modified Routing')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(df['num_experts'].unique())
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Accuracy Comparison
        ax2 = plt.subplot(2, 3, 2)
        for i, strategy in enumerate(strategies):
            strategy_data = df[df['strategy'] == strategy].groupby('num_experts')['token_accuracy'].mean()
            ax2.bar(x + i*width, strategy_data.values, width, label=strategy)

        ax2.set_xlabel('Number of Experts')
        ax2.set_ylabel('Token Accuracy (↑ better)')
        ax2.set_title('Accuracy: Baseline vs Modified Routing')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(df['num_experts'].unique())
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Delta Perplexity (vs Baseline)
        ax3 = plt.subplot(2, 3, 3)
        baseline_df = df[df['strategy'] == 'baseline']

        for strategy in [s for s in strategies if s != 'baseline']:
            deltas = []
            expert_counts = []

            for num_experts in df['num_experts'].unique():
                baseline_ppl = baseline_df[baseline_df['num_experts'] == num_experts]['perplexity'].mean()
                modified_ppl = df[(df['strategy'] == strategy) & (df['num_experts'] == num_experts)]['perplexity'].mean()
                deltas.append(modified_ppl - baseline_ppl)
                expert_counts.append(num_experts)

            ax3.plot(expert_counts, deltas, marker='o', label=strategy, linewidth=2)

        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Number of Experts')
        ax3.set_ylabel('Δ Perplexity (vs Baseline)')
        ax3.set_title('Perplexity Change from Baseline\n(negative = improvement)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Expert Utilization (Baseline)
        ax4 = plt.subplot(2, 3, 4)
        expert_counts = [insights[k]['num_experts'] for k in sorted(insights.keys())]
        utilizations = [insights[k]['expert_utilization'] * 100 for k in sorted(insights.keys())]

        ax4.bar(range(len(expert_counts)), utilizations, alpha=0.7, color='skyblue')
        ax4.set_xlabel('Configuration')
        ax4.set_ylabel('Expert Utilization (%)')
        ax4.set_title('Baseline: Expert Utilization Rate')
        ax4.set_xticks(range(len(expert_counts)))
        ax4.set_xticklabels([f"{ec}exp" for ec in expert_counts], rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. Routing Entropy Comparison
        ax5 = plt.subplot(2, 3, 5)
        for strategy in strategies:
            strategy_data = df[df['strategy'] == strategy].groupby('num_experts')['avg_entropy'].mean()
            ax5.plot(strategy_data.index, strategy_data.values, marker='s', label=strategy, linewidth=2)

        ax5.set_xlabel('Number of Experts')
        ax5.set_ylabel('Routing Entropy')
        ax5.set_title('Routing Entropy: Baseline vs Modified')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Speed-Quality Trade-off
        ax6 = plt.subplot(2, 3, 6)
        for strategy in strategies:
            strategy_df = df[df['strategy'] == strategy]
            ax6.scatter(
                strategy_df['perplexity'],
                strategy_df['tokens_per_second'],
                label=strategy,
                s=100,
                alpha=0.7
            )

        ax6.set_xlabel('Perplexity (↓ better)')
        ax6.set_ylabel('Tokens/Second (↑ better)')
        ax6.set_title('Speed vs Quality Trade-off')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        png_file = self.viz_dir / "two_phase_comparison.png"
        pdf_file = self.viz_dir / "two_phase_comparison.pdf"

        plt.savefig(png_file, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_file, bbox_inches='tight')
        plt.close()

        logger.info(f"Two-phase visualizations saved to: {self.viz_dir}")


if __name__ == "__main__":
    # Example usage
    runner = RoutingExperimentRunner()

    # Quick test
    results_df = runner.run_all_experiments(
        expert_counts=[8, 16],
        strategies=['regular', 'normalized'],
        datasets=['wikitext'],
        max_samples=50
    )

    runner.visualize_results(results_df)
    runner.generate_report(results_df)

    print(f"\n✅ Experiment complete! Results saved to: {runner.output_dir}")
