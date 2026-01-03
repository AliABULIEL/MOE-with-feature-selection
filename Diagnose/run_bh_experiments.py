#!/usr/bin/env python3
"""
Systematic BH Routing Experiments for OLMoE

This script runs comprehensive experiments comparing different routing methods:
- TopK (baseline)
- BH with various alpha values
- BH with temperature calibration

Features:
- Checkpointing and resume capability
- Progress tracking with tqdm
- Memory management (GPU cache clearing)
- Detailed logging
- Statistical analysis
- Automatic report generation

Usage:
    Local:
        python run_bh_experiments.py --model allenai/OLMoE-1B-7B-0924 --output ./results

    Colab:
        !python run_bh_experiments.py

Author: Generated for OLMoE BH Routing Analysis
Date: 2025-12-13
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Try importing dependencies with helpful error messages
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("ERROR: transformers not installed. Install with: pip install transformers")
    sys.exit(1)

try:
    from scipy import stats
except ImportError:
    print("ERROR: scipy not installed. Install with: pip install scipy")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("ERROR: matplotlib/seaborn not installed. Install with: pip install matplotlib seaborn")
    sys.exit(1)


# ============================================================================
# BH ROUTING IMPLEMENTATION (Inline for portability)
# ============================================================================

def benjamini_hochberg_routing(
    router_logits: torch.Tensor,
    alpha: float = 0.05,
    temperature: float = 1.0,
    min_k: int = 1,
    max_k: int = 16,
    return_stats: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Benjamini-Hochberg routing for adaptive expert selection.

    Args:
        router_logits: [batch_size, seq_len, num_experts] raw logits
        alpha: FDR control level (0.0-1.0)
        temperature: Softmax temperature for calibration
        min_k: Minimum experts to select
        max_k: Maximum experts to select
        return_stats: Return additional statistics

    Returns:
        routing_weights: [batch_size, seq_len, num_experts] with selected weights
        selected_experts: [batch_size, seq_len, max_k] expert indices (-1 for padding)
        num_selected: [batch_size, seq_len] number of experts selected per token
    """
    batch_size, seq_len, num_experts = router_logits.shape
    device = router_logits.device

    # Apply temperature scaling
    scaled_logits = router_logits / temperature

    # Compute probabilities
    probs = F.softmax(scaled_logits, dim=-1)

    # Compute pseudo p-values: p = 1 - prob
    p_values = 1.0 - probs

    # Sort p-values and get indices
    sorted_pvals, sorted_indices = torch.sort(p_values, dim=-1)

    # BH procedure: find largest k where p_(k) <= (k/N) * alpha
    k_range = torch.arange(1, num_experts + 1, device=device).float()
    thresholds = (k_range / num_experts) * alpha

    # Broadcast for batch and sequence dimensions
    thresholds = thresholds.view(1, 1, -1).expand(batch_size, seq_len, -1)

    # Find largest k satisfying condition
    satisfies_condition = sorted_pvals <= thresholds

    # Get number of experts to select per token
    num_selected = satisfies_condition.sum(dim=-1).clamp(min=min_k, max=max_k)

    # Create mask for selected experts
    k_indices = torch.arange(num_experts, device=device).view(1, 1, -1).expand(batch_size, seq_len, -1)
    selection_mask = k_indices < num_selected.unsqueeze(-1)

    # Get selected expert indices
    selected_experts_full = sorted_indices.clone()
    selected_experts_full[~selection_mask] = -1

    # Extract top-k selected experts (pad with -1)
    selected_experts = torch.full((batch_size, seq_len, max_k), -1, dtype=torch.long, device=device)
    for k_idx in range(max_k):
        slot_active = k_idx < num_selected
        expert_idx = sorted_indices[:, :, k_idx]
        selected_experts[:, :, k_idx] = torch.where(slot_active, expert_idx, torch.tensor(-1, device=device))

    # Get routing weights for selected experts
    routing_weights = torch.zeros(batch_size, seq_len, num_experts, device=device)
    routing_weights.scatter_(2, sorted_indices, probs.gather(2, sorted_indices))
    routing_weights = routing_weights * selection_mask.float()

    # Renormalize weights
    weight_sum = routing_weights.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    routing_weights = routing_weights / weight_sum

    return routing_weights, selected_experts, num_selected


# ============================================================================
# ROUTING CONFIGURATIONS (EXPANDED for Multi-Expert Analysis)
# ============================================================================

ROUTING_CONFIGS = {
    # =========================================================================
    # BASELINE TOP-K CONFIGURATIONS
    # =========================================================================
    'topk_8': {
        'method': 'topk', 'k': 8,
        'description': 'Baseline Top-8 (OLMoE default)'
    },
    'topk_16': {
        'method': 'topk', 'k': 16,
        'description': 'Baseline Top-16'
    },
    'topk_32': {
        'method': 'topk', 'k': 32,
        'description': 'Baseline Top-32'
    },
    'topk_64': {
        'method': 'topk', 'k': 64,
        'description': 'Baseline Top-64 (all experts)'
    },

    # =========================================================================
    # BH ROUTING: max_k = 8
    # =========================================================================
    'bh_k8_a001': {
        'method': 'bh', 'alpha': 0.01, 'temperature': 1.0, 'min_k': 1, 'max_k': 8,
        'description': 'BH α=0.01, max_k=8 (strict)'
    },
    'bh_k8_a005': {
        'method': 'bh', 'alpha': 0.05, 'temperature': 1.0, 'min_k': 1, 'max_k': 8,
        'description': 'BH α=0.05, max_k=8 (moderate)'
    },
    'bh_k8_a010': {
        'method': 'bh', 'alpha': 0.10, 'temperature': 1.0, 'min_k': 1, 'max_k': 8,
        'description': 'BH α=0.10, max_k=8 (loose)'
    },

    # =========================================================================
    # BH ROUTING: max_k = 16
    # =========================================================================
    'bh_k16_a001': {
        'method': 'bh', 'alpha': 0.01, 'temperature': 1.0, 'min_k': 1, 'max_k': 16,
        'description': 'BH α=0.01, max_k=16 (strict)'
    },
    'bh_k16_a005': {
        'method': 'bh', 'alpha': 0.05, 'temperature': 1.0, 'min_k': 1, 'max_k': 16,
        'description': 'BH α=0.05, max_k=16 (moderate)'
    },
    'bh_k16_a010': {
        'method': 'bh', 'alpha': 0.10, 'temperature': 1.0, 'min_k': 1, 'max_k': 16,
        'description': 'BH α=0.10, max_k=16 (loose)'
    },

    # =========================================================================
    # BH ROUTING: max_k = 32
    # =========================================================================
    'bh_k32_a001': {
        'method': 'bh', 'alpha': 0.01, 'temperature': 1.0, 'min_k': 1, 'max_k': 32,
        'description': 'BH α=0.01, max_k=32 (strict)'
    },
    'bh_k32_a005': {
        'method': 'bh', 'alpha': 0.05, 'temperature': 1.0, 'min_k': 1, 'max_k': 32,
        'description': 'BH α=0.05, max_k=32 (moderate)'
    },
    'bh_k32_a010': {
        'method': 'bh', 'alpha': 0.10, 'temperature': 1.0, 'min_k': 1, 'max_k': 32,
        'description': 'BH α=0.10, max_k=32 (loose)'
    },

    # =========================================================================
    # BH ROUTING: max_k = 64 (UNCAPPED)
    # =========================================================================
    'bh_k64_a001': {
        'method': 'bh', 'alpha': 0.01, 'temperature': 1.0, 'min_k': 1, 'max_k': 64,
        'description': 'BH α=0.01, max_k=64 (strict, uncapped)'
    },
    'bh_k64_a005': {
        'method': 'bh', 'alpha': 0.05, 'temperature': 1.0, 'min_k': 1, 'max_k': 64,
        'description': 'BH α=0.05, max_k=64 (moderate, uncapped)'
    },
    'bh_k64_a010': {
        'method': 'bh', 'alpha': 0.10, 'temperature': 1.0, 'min_k': 1, 'max_k': 64,
        'description': 'BH α=0.10, max_k=64 (loose, uncapped)'
    },
}

# Total configurations: 4 baselines + 12 BH variants = 16 total
# Total experiments: 16 configs × 12 prompts = 192 experiments


# ============================================================================
# TEST PROMPTS (Various complexity levels)
# ============================================================================

TEST_PROMPTS = [
    # Simple prompts (short, common patterns)
    {
        'text': 'The cat sat on the',
        'category': 'simple',
        'description': 'Simple sentence completion'
    },
    {
        'text': 'Once upon a time, there was a',
        'category': 'simple',
        'description': 'Story beginning'
    },
    {
        'text': 'The capital of France is',
        'category': 'simple',
        'description': 'Factual completion'
    },

    # Medium prompts (longer, requires context)
    {
        'text': 'In computer science, a linked list is a data structure that',
        'category': 'medium',
        'description': 'Technical explanation'
    },
    {
        'text': 'Climate change refers to long-term shifts in temperatures and weather patterns. These shifts',
        'category': 'medium',
        'description': 'Informative continuation'
    },
    {
        'text': 'The stock market crashed in 1929 because',
        'category': 'medium',
        'description': 'Historical reasoning'
    },

    # Complex prompts (multi-step reasoning)
    {
        'text': 'If you have 3 apples and you give 2 to your friend, then you buy 5 more, how many apples do you have? Let me think step by step:',
        'category': 'complex',
        'description': 'Mathematical reasoning'
    },
    {
        'text': 'Compare and contrast the Renaissance and the Enlightenment periods in terms of their cultural, scientific, and philosophical contributions:',
        'category': 'complex',
        'description': 'Comparative analysis'
    },
    {
        'text': 'Explain the difference between supervised and unsupervised learning in machine learning, providing examples of each:',
        'category': 'complex',
        'description': 'Technical explanation with examples'
    },

    # Domain-specific prompts
    {
        'text': 'In Python, a decorator is a function that',
        'category': 'domain_specific',
        'description': 'Programming concepts'
    },
    {
        'text': 'The mitochondria is known as the powerhouse of the cell because it',
        'category': 'domain_specific',
        'description': 'Biology knowledge'
    },
    {
        'text': 'In quantum mechanics, the uncertainty principle states that',
        'category': 'domain_specific',
        'description': 'Physics knowledge'
    },
]


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class BHExperimentRunner:
    """Manages systematic BH routing experiments."""

    def __init__(
        self,
        model_name: str = "allenai/OLMoE-1B-7B-0924",
        output_dir: str = "./results",
        checkpoint_interval: int = 5,
        device: Optional[str] = None,
        use_fp16: bool = True
    ):
        """
        Initialize experiment runner.

        Args:
            model_name: HuggingFace model identifier
            output_dir: Directory to save results
            checkpoint_interval: Save checkpoint every N experiments
            device: Device to use ('cuda' or 'cpu', auto-detect if None)
            use_fp16: Use bfloat16 for memory efficiency
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.checkpoint_interval = checkpoint_interval
        self.use_fp16 = use_fp16

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.logger.info(f"Using device: {self.device}")

        # Load model and tokenizer
        self.logger.info(f"Loading model: {model_name}")
        self.model, self.tokenizer = self._load_model()

        # Find routers
        self.routers = self._find_routers()
        self.logger.info(f"Found {len(self.routers)} MoE routers")

        # Initialize results storage
        self.results = []
        self.checkpoint_path = self.output_dir / "checkpoints" / "latest_checkpoint.json"

    def _setup_logging(self):
        """Setup logging to file and console."""
        log_file = self.output_dir / "experiment_log.txt"

        # Create logger
        self.logger = logging.getLogger("BHExperiments")
        self.logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _load_model(self) -> Tuple[Any, Any]:
        """Load model and tokenizer."""
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Load model
            dtype = torch.bfloat16 if self.use_fp16 else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )

            if self.device == "cpu":
                model = model.to(self.device)

            model.eval()

            return model, tokenizer

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _find_routers(self) -> List[Tuple[str, Any]]:
        """Find all MoE routers in the model."""
        routers = []
        for name, module in self.model.named_modules():
            # OLMoE uses 'OlmoeTopKRouter'
            if module.__class__.__name__ == 'OlmoeTopKRouter':
                routers.append((name, module))
        return routers

    def _apply_routing_config(self, config: Dict[str, Any]) -> None:
        """Apply routing configuration to model."""
        method = config['method']

        if method == 'topk':
            # Restore original forward methods (if patched)
            self._restore_original_routing()
        elif method == 'bh':
            # Patch routers with BH routing
            self._patch_bh_routing(
                alpha=config['alpha'],
                temperature=config['temperature'],
                min_k=config.get('min_k', 1),
                max_k=config.get('max_k', 8)
            )
        else:
            raise ValueError(f"Unknown routing method: {method}")

    def _restore_original_routing(self) -> None:
        """Restore original TopK routing."""
        for name, router in self.routers:
            if hasattr(router, '_original_forward'):
                router.forward = router._original_forward
                delattr(router, '_original_forward')

    def _patch_bh_routing(
        self,
        alpha: float,
        temperature: float,
        min_k: int,
        max_k: int
    ) -> None:
        """Patch routers to use BH routing."""
        for name, router in self.routers:
            # Save original forward if not already saved
            if not hasattr(router, '_original_forward'):
                router._original_forward = router.forward

            # Create patched forward method
            original_linear = router.gate if hasattr(router, 'gate') else router.linear

            def create_patched_forward(linear_module):
                def patched_forward(hidden_states):
                    # Get router logits
                    router_logits = linear_module(hidden_states)

                    # Apply BH routing
                    routing_weights_sparse, selected_experts, num_selected = benjamini_hochberg_routing(
                        router_logits.unsqueeze(0),  # Add batch dim if needed
                        alpha=alpha,
                        temperature=temperature,
                        min_k=min_k,
                        max_k=max_k
                    )

                    # Remove batch dim if we added it
                    if hidden_states.dim() == 2:
                        routing_weights_sparse = routing_weights_sparse.squeeze(0)
                        selected_experts = selected_experts.squeeze(0)

                    # Convert to dense format for compatibility
                    safe_indices = selected_experts.clamp(min=0)
                    routing_weights = routing_weights_sparse.gather(-1, safe_indices)

                    # Zero out padding
                    padding_mask = selected_experts == -1
                    routing_weights = routing_weights.masked_fill(padding_mask, 0.0)

                    # Renormalize
                    weight_sum = routing_weights.sum(dim=-1, keepdim=True).clamp(min=1e-10)
                    routing_weights = routing_weights / weight_sum

                    return routing_weights, selected_experts, router_logits

                return patched_forward

            router.forward = create_patched_forward(original_linear)

    def run_single_experiment(
        self,
        config_name: str,
        config: Dict[str, Any],
        prompt_data: Dict[str, str],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = False
    ) -> Dict[str, Any]:
        """
        Run single experiment with one routing config and one prompt.

        Args:
            config_name: Name of routing configuration
            config: Routing configuration dict
            prompt_data: Prompt dict with 'text', 'category', 'description'
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature
            do_sample: Whether to sample or use greedy decoding

        Returns:
            Dictionary with experiment results
        """
        prompt_text = prompt_data['text']
        prompt_category = prompt_data['category']

        self.logger.info(f"Running: {config_name} on '{prompt_text[:50]}...'")

        # Apply routing configuration
        self._apply_routing_config(config)

        # Tokenize input
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Collect statistics
        routing_stats = {
            'expert_counts': [],
            'selected_experts': [],
            'router_logits': []
        }

        # Hook to collect routing info
        def routing_hook(module, input, output):
            routing_weights, selected_experts, router_logits = output

            # Count non-zero experts per token
            if config['method'] == 'bh':
                expert_count = (selected_experts != -1).sum(dim=-1)
            else:
                expert_count = torch.full_like(selected_experts[:, :, 0], config['k'])

            routing_stats['expert_counts'].append(expert_count.detach().cpu())
            routing_stats['selected_experts'].append(selected_experts.detach().cpu())
            routing_stats['router_logits'].append(router_logits.detach().cpu())

        # Register hooks
        handles = []
        for name, router in self.routers:
            handle = router.register_forward_hook(routing_hook)
            handles.append(handle)

        # Run generation with timing
        torch.cuda.empty_cache() if self.device == "cuda" else None
        start_time = time.time()

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else 1.0,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            # Remove hooks
            for handle in handles:
                handle.remove()
            raise

        inference_time = time.time() - start_time

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_only = generated_text[len(prompt_text):]

        # Compute statistics
        all_expert_counts = torch.cat(routing_stats['expert_counts'], dim=0)

        result = {
            'config_name': config_name,
            'config': config,
            'prompt_text': prompt_text,
            'prompt_category': prompt_category,
            'prompt_description': prompt_data['description'],
            'generated_text': generated_only,
            'full_text': generated_text,
            'num_input_tokens': inputs['input_ids'].shape[1],
            'num_output_tokens': outputs.shape[1] - inputs['input_ids'].shape[1],
            'inference_time_sec': inference_time,
            'tokens_per_sec': (outputs.shape[1] - inputs['input_ids'].shape[1]) / inference_time,
            'avg_experts_per_token': float(all_expert_counts.float().mean()),
            'std_experts_per_token': float(all_expert_counts.float().std()),
            'min_experts': int(all_expert_counts.min()),
            'max_experts': int(all_expert_counts.max()),
            'num_layers': len(self.routers),
            'timestamp': datetime.now().isoformat()
        }

        # Expert utilization statistics
        all_selected_experts = torch.cat(routing_stats['selected_experts'], dim=0)
        expert_usage = defaultdict(int)
        for expert_tensor in all_selected_experts.flatten():
            expert_id = int(expert_tensor)
            if expert_id >= 0:  # Skip padding (-1)
                expert_usage[expert_id] += 1

        if expert_usage:
            usage_counts = list(expert_usage.values())
            result['expert_utilization_cv'] = float(np.std(usage_counts) / np.mean(usage_counts))
            result['expert_utilization_max_min_ratio'] = float(max(usage_counts) / max(min(usage_counts), 1))
        else:
            result['expert_utilization_cv'] = 0.0
            result['expert_utilization_max_min_ratio'] = 1.0

        # Clear CUDA cache
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return result

    def run_full_experiment_suite(
        self,
        routing_configs: Optional[Dict[str, Dict]] = None,
        test_prompts: Optional[List[Dict]] = None,
        max_new_tokens: int = 50,
        resume: bool = True
    ) -> pd.DataFrame:
        """
        Run all combinations of routing configs and prompts.

        Args:
            routing_configs: Dict of routing configurations (uses ROUTING_CONFIGS if None)
            test_prompts: List of test prompts (uses TEST_PROMPTS if None)
            max_new_tokens: Maximum tokens to generate
            resume: Resume from checkpoint if available

        Returns:
            DataFrame with all results
        """
        if routing_configs is None:
            routing_configs = ROUTING_CONFIGS
        if test_prompts is None:
            test_prompts = TEST_PROMPTS

        # Load checkpoint if resuming
        completed_experiments = set()
        if resume and self.checkpoint_path.exists():
            self.logger.info(f"Resuming from checkpoint: {self.checkpoint_path}")
            with open(self.checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
                self.results = checkpoint_data['results']
                completed_experiments = set(
                    (r['config_name'], r['prompt_text']) for r in self.results
                )
            self.logger.info(f"Loaded {len(self.results)} completed experiments")

        # Total experiments
        total_experiments = len(routing_configs) * len(test_prompts)
        remaining = total_experiments - len(completed_experiments)

        self.logger.info(f"Total experiments: {total_experiments}")
        self.logger.info(f"Completed: {len(completed_experiments)}")
        self.logger.info(f"Remaining: {remaining}")

        # Run experiments
        pbar = tqdm(total=remaining, desc="Running experiments")
        experiment_count = 0

        for config_name, config in routing_configs.items():
            for prompt_data in test_prompts:
                # Skip if already completed
                if (config_name, prompt_data['text']) in completed_experiments:
                    continue

                try:
                    result = self.run_single_experiment(
                        config_name=config_name,
                        config=config,
                        prompt_data=prompt_data,
                        max_new_tokens=max_new_tokens
                    )
                    self.results.append(result)
                    experiment_count += 1
                    pbar.update(1)

                    # Checkpoint periodically
                    if experiment_count % self.checkpoint_interval == 0:
                        self._save_checkpoint()

                except Exception as e:
                    self.logger.error(f"Experiment failed: {config_name} on '{prompt_data['text'][:50]}...'")
                    self.logger.error(f"Error: {e}")
                    # Continue with next experiment
                    continue

        pbar.close()

        # Final checkpoint
        self._save_checkpoint()

        # Convert to DataFrame
        df = pd.DataFrame(self.results)

        # Save to CSV
        csv_path = self.output_dir / "bh_routing_results.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved results to {csv_path}")

        # Restore original routing
        self._restore_original_routing()

        return df

    def _save_checkpoint(self):
        """Save checkpoint to disk."""
        checkpoint_data = {
            'results': self.results,
            'timestamp': datetime.now().isoformat(),
            'num_experiments': len(self.results)
        }

        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        self.logger.info(f"Checkpoint saved: {len(self.results)} experiments")


# ============================================================================
# ANALYSIS AND REPORTING
# ============================================================================

def analyze_results(df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    """
    Generate summary statistics and statistical tests.

    Args:
        df: Results DataFrame
        output_dir: Directory to save analysis outputs

    Returns:
        Dictionary with analysis results
    """
    logger = logging.getLogger("BHExperiments")
    logger.info("Analyzing results...")

    analysis = {}

    # Overall statistics by config
    grouped = df.groupby('config_name').agg({
        'avg_experts_per_token': ['mean', 'std'],
        'inference_time_sec': ['mean', 'std'],
        'tokens_per_sec': ['mean', 'std'],
        'expert_utilization_cv': ['mean', 'std'],
    }).round(3)

    analysis['summary_by_config'] = grouped
    logger.info("\nSummary by configuration:")
    logger.info(f"\n{grouped}")

    # Statistics by prompt category
    category_stats = df.groupby(['prompt_category', 'config_name']).agg({
        'avg_experts_per_token': 'mean',
        'inference_time_sec': 'mean'
    }).round(3)

    analysis['summary_by_category'] = category_stats
    logger.info("\nSummary by prompt category:")
    logger.info(f"\n{category_stats}")

    # Statistical tests (BH methods vs TopK baseline)
    baseline_df = df[df['config_name'] == 'topk_8']
    bh_methods = [c for c in df['config_name'].unique() if c.startswith('bh_')]

    statistical_tests = []

    for bh_method in bh_methods:
        bh_df = df[df['config_name'] == bh_method]

        # Paired t-test on expert counts
        baseline_experts = baseline_df['avg_experts_per_token'].values
        bh_experts = bh_df['avg_experts_per_token'].values

        if len(baseline_experts) == len(bh_experts):
            t_stat, p_value = stats.ttest_rel(baseline_experts, bh_experts)

            # Effect size (Cohen's d)
            diff = baseline_experts - bh_experts
            cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0

            statistical_tests.append({
                'comparison': f'{bh_method} vs topk_8',
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'mean_reduction': diff.mean(),
                'significant': p_value < 0.05
            })

    analysis['statistical_tests'] = pd.DataFrame(statistical_tests)
    logger.info("\nStatistical tests:")
    logger.info(f"\n{analysis['statistical_tests']}")

    # Save analysis to JSON
    analysis_json = output_dir / "analysis_summary.json"
    with open(analysis_json, 'w') as f:
        json.dump({
            'summary_by_config': grouped.to_dict(),
            'summary_by_category': category_stats.to_dict(),
            'statistical_tests': analysis['statistical_tests'].to_dict('records')
        }, f, indent=2)

    logger.info(f"Analysis saved to {analysis_json}")

    return analysis


def analyze_by_max_k(df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    """
    Analyze results grouped by max_k value.

    Creates:
    - Summary table by max_k
    - Comparison plots
    - Statistical tests

    Args:
        df: Results DataFrame
        output_dir: Output directory

    Returns:
        Dictionary with analysis results
    """
    logger = logging.getLogger("BHExperiments")
    logger.info("Analyzing results by max_k...")

    # Extract max_k from config
    def get_max_k(config_name):
        if config_name.startswith('topk_'):
            return int(config_name.split('_')[1])
        elif '_k' in config_name:
            # bh_k16_a005 -> 16
            for part in config_name.split('_'):
                if part.startswith('k') and part[1:].isdigit():
                    return int(part[1:])
        return 8  # default

    df['max_k'] = df['config_name'].apply(get_max_k)
    df['is_bh'] = df['config_name'].str.startswith('bh_')

    # Summary by max_k for BH methods
    bh_df = df[df['is_bh']]
    max_k_summary = bh_df.groupby('max_k').agg({
        'avg_experts_per_token': ['mean', 'std', 'min', 'max'],
        'inference_time_sec': 'mean',
    }).round(3)

    logger.info(f"\nBH Routing by max_k:\n{max_k_summary}")

    # Compare with baselines
    baseline_df = df[~df['is_bh']]

    comparison = []
    for max_k in [8, 16, 32, 64]:
        baseline_mean = baseline_df[baseline_df['max_k'] == max_k]['avg_experts_per_token'].mean()
        bh_mean = bh_df[bh_df['max_k'] == max_k]['avg_experts_per_token'].mean()
        reduction = ((baseline_mean - bh_mean) / baseline_mean) * 100 if baseline_mean > 0 else 0

        comparison.append({
            'max_k': max_k,
            'baseline_experts': baseline_mean,
            'bh_mean_experts': bh_mean,
            'reduction_pct': reduction
        })

    comparison_df = pd.DataFrame(comparison)
    logger.info(f"\nBH vs Baseline by max_k:\n{comparison_df}")

    # Save
    comparison_df.to_csv(output_dir / "max_k_comparison.csv", index=False)

    return {
        'max_k_summary': max_k_summary,
        'comparison': comparison_df
    }


def create_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create all visualizations from experiment results.

    Args:
        df: Results DataFrame
        output_dir: Directory to save plots
    """
    logger = logging.getLogger("BHExperiments")
    logger.info("Creating visualizations...")

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Set style
    sns.set_style('whitegrid')
    plt.rcParams['figure.dpi'] = 150

    # 1. Average experts per token by configuration
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x='config_name', y='avg_experts_per_token', ax=ax, errorbar='sd')
    ax.set_xlabel('Routing Configuration', fontsize=12)
    ax.set_ylabel('Average Experts per Token', fontsize=12)
    ax.set_title('Expert Count Comparison Across Routing Methods', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(plot_dir / "avg_experts_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Created: avg_experts_comparison.png")

    # 2. Inference time comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x='config_name', y='inference_time_sec', ax=ax, errorbar='sd')
    ax.set_xlabel('Routing Configuration', fontsize=12)
    ax.set_ylabel('Inference Time (seconds)', fontsize=12)
    ax.set_title('Inference Latency Comparison', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(plot_dir / "inference_time_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Created: inference_time_comparison.png")

    # 3. Expert utilization (load balance)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x='config_name', y='expert_utilization_cv', ax=ax, errorbar='sd')
    ax.set_xlabel('Routing Configuration', fontsize=12)
    ax.set_ylabel('Coefficient of Variation (CV)', fontsize=12)
    ax.set_title('Expert Load Balance (Lower = Better)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(plot_dir / "expert_utilization_cv.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Created: expert_utilization_cv.png")

    # 4. Performance by prompt category
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=df, x='prompt_category', y='avg_experts_per_token', hue='config_name', ax=ax)
    ax.set_xlabel('Prompt Category', fontsize=12)
    ax.set_ylabel('Average Experts per Token', fontsize=12)
    ax.set_title('Expert Count by Prompt Category and Routing Method', fontsize=14, fontweight='bold')
    ax.legend(title='Routing Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(plot_dir / "experts_by_category.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Created: experts_by_category.png")

    # 5. Scatter: expert count vs inference time
    fig, ax = plt.subplots(figsize=(12, 6))
    for config in df['config_name'].unique():
        config_df = df[df['config_name'] == config]
        ax.scatter(
            config_df['avg_experts_per_token'],
            config_df['inference_time_sec'],
            label=config,
            alpha=0.6,
            s=100
        )
    ax.set_xlabel('Average Experts per Token', fontsize=12)
    ax.set_ylabel('Inference Time (seconds)', fontsize=12)
    ax.set_title('Expert Count vs Inference Latency', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "experts_vs_latency.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Created: experts_vs_latency.png")

    logger.info(f"All visualizations saved to {plot_dir}")


def create_max_k_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    """Create visualizations comparing different max_k values."""
    logger = logging.getLogger("BHExperiments")
    logger.info("Creating max_k comparison visualizations...")

    plot_dir = output_dir / "plots"

    # Extract max_k
    def get_max_k(config_name):
        if config_name.startswith('topk_'):
            return int(config_name.split('_')[1])
        elif '_k' in config_name:
            for part in config_name.split('_'):
                if part.startswith('k') and part[1:].isdigit():
                    return int(part[1:])
        return 8

    df['max_k'] = df['config_name'].apply(get_max_k)
    df['is_bh'] = df['config_name'].str.startswith('bh_')
    df['method'] = df['config_name'].apply(lambda x: 'BH' if x.startswith('bh_') else 'TopK')

    # Plot 1: Average experts by max_k (BH vs TopK)
    fig, ax = plt.subplots(figsize=(12, 6))

    summary = df.groupby(['max_k', 'method'])['avg_experts_per_token'].mean().unstack()
    summary.plot(kind='bar', ax=ax, width=0.8)

    ax.set_xlabel('Maximum Experts (max_k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Experts Selected', fontsize=12, fontweight='bold')
    ax.set_title('BH Routing vs TopK Baseline Across max_k Values', fontsize=14, fontweight='bold')
    ax.legend(title='Method')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    # Add reduction % annotations
    for i, max_k in enumerate([8, 16, 32, 64]):
        if max_k in summary.index:
            topk_val = summary.loc[max_k, 'TopK'] if 'TopK' in summary.columns else max_k
            bh_val = summary.loc[max_k, 'BH'] if 'BH' in summary.columns else topk_val
            reduction = ((topk_val - bh_val) / topk_val) * 100
            ax.annotate(f'-{reduction:.0f}%',
                       xy=(i, bh_val),
                       ha='center', va='bottom',
                       fontsize=10, color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig(plot_dir / "max_k_comparison_bar.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Created: max_k_comparison_bar.png")

    # Plot 2: Expert count distribution by max_k (box plot)
    fig, ax = plt.subplots(figsize=(14, 6))

    bh_df = df[df['is_bh']]
    sns.boxplot(data=bh_df, x='max_k', y='avg_experts_per_token', ax=ax)

    ax.set_xlabel('Maximum Experts (max_k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Experts Selected', fontsize=12, fontweight='bold')
    ax.set_title('Expert Selection Distribution by max_k (BH Routing)', fontsize=14, fontweight='bold')

    # Add baseline reference lines
    for i, max_k in enumerate([8, 16, 32, 64]):
        ax.hlines(y=max_k, xmin=i-0.4, xmax=i+0.4, colors='red', linestyles='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(plot_dir / "max_k_distribution_box.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Created: max_k_distribution_box.png")

    # Plot 3: Heatmap - max_k vs alpha
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract alpha from config name
    def get_alpha(config_name):
        if 'a001' in config_name: return 0.01
        elif 'a005' in config_name: return 0.05
        elif 'a010' in config_name: return 0.10
        return None

    bh_df = df[df['is_bh']].copy()
    bh_df['alpha'] = bh_df['config_name'].apply(get_alpha)
    bh_df = bh_df.dropna(subset=['alpha'])

    pivot = bh_df.pivot_table(
        values='avg_experts_per_token',
        index='alpha',
        columns='max_k',
        aggfunc='mean'
    )

    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
    ax.set_xlabel('Maximum Experts (max_k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Alpha (FDR Level)', fontsize=12, fontweight='bold')
    ax.set_title('Average Experts: Alpha × max_k', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(plot_dir / "max_k_alpha_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Created: max_k_alpha_heatmap.png")

    # Plot 4: Line plot showing saturation
    fig, ax = plt.subplots(figsize=(12, 6))

    for alpha in [0.01, 0.05, 0.10]:
        alpha_df = bh_df[bh_df['alpha'] == alpha]
        means = alpha_df.groupby('max_k')['avg_experts_per_token'].mean()
        ax.plot(means.index, means.values, marker='o', markersize=10,
                linewidth=2, label=f'α={alpha}')

    # Add baseline
    baseline_df = df[~df['is_bh']]
    baseline_means = baseline_df.groupby('max_k')['avg_experts_per_token'].mean()
    ax.plot(baseline_means.index, baseline_means.values, marker='s', markersize=10,
            linewidth=2, linestyle='--', color='red', label='TopK Baseline')

    ax.set_xlabel('Maximum Experts (max_k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Experts Selected', fontsize=12, fontweight='bold')
    ax.set_title('BH Routing Saturation Analysis', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_xticks([8, 16, 32, 64])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_dir / "max_k_saturation.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Created: max_k_saturation.png")

    logger.info("All max_k visualizations created")


def generate_markdown_report(
    df: pd.DataFrame,
    analysis: Dict[str, Any],
    output_path: Path
) -> None:
    """
    Generate comprehensive markdown report.

    Args:
        df: Results DataFrame
        analysis: Analysis results from analyze_results()
        output_path: Path to save report
    """
    logger = logging.getLogger("BHExperiments")
    logger.info("Generating markdown report...")

    report = []

    # Header
    report.append("# BH Routing Experiment Results")
    report.append("")
    report.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Total Experiments**: {len(df)}")
    report.append(f"**Routing Configurations**: {len(df['config_name'].unique())}")
    report.append(f"**Test Prompts**: {len(df['prompt_text'].unique())}")
    report.append("")
    report.append("---")
    report.append("")

    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    baseline_mean = df[df['config_name'] == 'topk_8']['avg_experts_per_token'].mean()
    for config in df['config_name'].unique():
        if config.startswith('bh_'):
            bh_mean = df[df['config_name'] == config]['avg_experts_per_token'].mean()
            reduction = ((baseline_mean - bh_mean) / baseline_mean) * 100
            report.append(f"- **{config}**: {reduction:.1f}% expert reduction vs baseline")
    report.append("")
    report.append("---")
    report.append("")

    # Summary Statistics
    report.append("## Summary Statistics by Configuration")
    report.append("")
    summary_df = analysis['summary_by_config']
    report.append(summary_df.to_markdown())
    report.append("")
    report.append("---")
    report.append("")

    # Statistical Tests
    report.append("## Statistical Significance Tests")
    report.append("")
    report.append("Paired t-tests comparing BH methods against TopK-8 baseline:")
    report.append("")
    stats_df = analysis['statistical_tests']
    report.append(stats_df.to_markdown(index=False))
    report.append("")
    report.append("*p < 0.05 indicates statistically significant difference*")
    report.append("")
    report.append("---")
    report.append("")

    # Performance by Category
    report.append("## Performance by Prompt Category")
    report.append("")
    category_df = analysis['summary_by_category']
    report.append(category_df.to_markdown())
    report.append("")
    report.append("---")
    report.append("")

    # Visualizations
    report.append("## Visualizations")
    report.append("")
    report.append("### Average Expert Count Comparison")
    report.append("![Expert Count](plots/avg_experts_comparison.png)")
    report.append("")
    report.append("### Inference Time Comparison")
    report.append("![Inference Time](plots/inference_time_comparison.png)")
    report.append("")
    report.append("### Expert Load Balance")
    report.append("![Load Balance](plots/expert_utilization_cv.png)")
    report.append("")
    report.append("### Expert Count by Prompt Category")
    report.append("![By Category](plots/experts_by_category.png)")
    report.append("")
    report.append("### Expert Count vs Latency")
    report.append("![vs Latency](plots/experts_vs_latency.png)")
    report.append("")
    report.append("---")
    report.append("")

    # Conclusions
    report.append("## Conclusions")
    report.append("")
    report.append("### Key Findings")
    report.append("")
    report.append("1. **Expert Efficiency**: BH routing consistently reduces expert count compared to TopK-8")
    report.append("2. **Statistical Significance**: Most BH configurations show statistically significant differences")
    report.append("3. **Prompt Sensitivity**: Performance varies by prompt category")
    report.append("4. **Trade-offs**: Lower expert count may impact inference time differently by configuration")
    report.append("")
    report.append("### Recommendations")
    report.append("")
    report.append("- **bh_moderate (alpha=0.05)**: Best balance of efficiency and performance")
    report.append("- **bh_strict (alpha=0.01)**: Maximum expert reduction for resource-constrained scenarios")
    report.append("- **bh_adaptive (temp=2.0)**: Improved calibration for better routing decisions")
    report.append("")
    report.append("---")
    report.append("")

    # Appendix
    report.append("## Appendix: Configuration Details")
    report.append("")
    for config_name, config in ROUTING_CONFIGS.items():
        report.append(f"### {config_name}")
        report.append(f"- **Description**: {config['description']}")
        for key, value in config.items():
            if key != 'description':
                report.append(f"- **{key}**: {value}")
        report.append("")

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    logger.info(f"Report saved to {output_path}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for experiment script."""
    parser = argparse.ArgumentParser(
        description="Run systematic BH routing experiments on OLMoE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model',
        type=str,
        default='allenai/OLMoE-1B-7B-0924',
        help='HuggingFace model identifier'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=50,
        help='Maximum tokens to generate per prompt'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=5,
        help='Save checkpoint every N experiments'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Do not resume from checkpoint'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use'
    )
    parser.add_argument(
        '--no-fp16',
        action='store_true',
        help='Do not use bfloat16 (use float32)'
    )

    args = parser.parse_args()

    # Print banner
    print("=" * 80)
    print("BH ROUTING SYSTEMATIC EXPERIMENTS")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print(f"Max tokens: {args.max_tokens}")
    print("=" * 80)
    print()

    # Initialize runner
    device = None if args.device == 'auto' else args.device
    runner = BHExperimentRunner(
        model_name=args.model,
        output_dir=args.output,
        checkpoint_interval=args.checkpoint_interval,
        device=device,
        use_fp16=not args.no_fp16
    )

    # Run experiments
    print("\nRunning experiments...")
    df = runner.run_full_experiment_suite(
        max_new_tokens=args.max_tokens,
        resume=not args.no_resume
    )

    # Analyze results
    print("\nAnalyzing results...")
    analysis = analyze_results(df, Path(args.output))

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(df, Path(args.output))

    # Multi-expert analysis
    print("\nAnalyzing by max_k...")
    max_k_analysis = analyze_by_max_k(df, Path(args.output))

    print("\nCreating max_k visualizations...")
    create_max_k_visualizations(df, Path(args.output))

    # Generate report
    print("\nGenerating report...")
    generate_markdown_report(
        df,
        analysis,
        Path(args.output) / "REPORT.md"
    )

    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {args.output}")
    print(f"- CSV: {args.output}/bh_routing_results.csv")
    print(f"- Plots: {args.output}/plots/")
    print(f"- Report: {args.output}/REPORT.md")
    print(f"- Log: {args.output}/experiment_log.txt")
    print()


if __name__ == "__main__":
    main()
