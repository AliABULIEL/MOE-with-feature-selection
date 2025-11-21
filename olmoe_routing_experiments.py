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
        indices_np = expert_indices.detach().cpu().numpy().flatten()
        weights_np = expert_weights.detach().cpu().numpy()

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
            torch_dtype=torch.bfloat16,
            device_map=device,
            output_router_logits=True
        )
        self.model.eval()

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
            'adaptive': AdaptiveRouting
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
        max_length: int = 512
    ) -> ExperimentResults:
        """
        Evaluate a single routing configuration.

        Args:
            config: Routing configuration to test
            texts: List of text samples
            dataset_name: Name of dataset being evaluated
            max_length: Maximum sequence length

        Returns:
            ExperimentResults with all metrics
        """
        logger.info(f"Evaluating: {config.get_name()} on {dataset_name}")

        # Set expert count
        self._set_expert_count(config.num_experts)

        # Create routing strategy
        strategy = self.strategy_factory[config.strategy](config.num_experts)

        # Initialize metrics
        total_loss = 0.0
        total_tokens = 0
        correct_predictions = 0
        num_samples = 0

        start_time = time.time()

        # Evaluate on all texts
        with torch.no_grad():
            for text in tqdm(texts, desc=f"{config.get_name()}", leave=False):
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

                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        labels=input_ids,
                        output_router_logits=True
                    )

                    # Accumulate loss
                    loss = outputs.loss.item()
                    total_loss += loss * input_ids.shape[1]
                    total_tokens += input_ids.shape[1]

                    # Calculate token accuracy
                    logits = outputs.logits
                    predictions = torch.argmax(logits[:, :-1, :], dim=-1)
                    targets = input_ids[:, 1:]
                    correct_predictions += (predictions == targets).sum().item()

                    # Apply routing strategy to router logits
                    if outputs.router_logits is not None:
                        for router_logit in outputs.router_logits:
                            if router_logit is not None:
                                strategy.route(router_logit)

                    num_samples += 1

                except Exception as e:
                    logger.warning(f"Error processing sample: {e}")
                    continue

        end_time = time.time()
        elapsed_time = end_time - start_time

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

        # Save individual log file
        log_file = self.logs_dir / f"{config.get_name()}_{dataset_name}.json"
        with open(log_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)

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

        for dataset_name in datasets:
            # Load dataset once
            texts = self.load_dataset(dataset_name, max_samples=max_samples)

            # Run all configs on this dataset
            for config in configs:
                try:
                    results = self.evaluate_configuration(
                        config, texts, dataset_name
                    )
                    all_results.append(results)
                except Exception as e:
                    logger.error(f"Experiment failed: {config.get_name()} on {dataset_name}: {e}")

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
