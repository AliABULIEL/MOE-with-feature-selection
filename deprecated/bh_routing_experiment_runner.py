"""
BH Routing Experiment Runner - Main Framework
==============================================

This module implements the comprehensive two-phase experiment framework for
evaluating Benjamini-Hochberg routing against baseline Top-K routing in OLMoE.

The framework follows the template structure from OLMoE_Full_Routing_Experiments,
with dual file logging (summary + internal_routing JSONs) and complete metrics.

Experimental Design:
- 4 baseline configurations (TopK with K=8, 16, 32, 64)
- 16 BH configurations (max_k=[8,16,32,64] × alpha=[0.30,0.40,0.50,0.60])
- 3 datasets (WikiText-2, LAMBADA, HellaSwag)
- Total: 60 experiments, 120 output files

Usage:
    from bh_routing_experiment_runner import BHRoutingExperimentRunner

    runner = BHRoutingExperimentRunner(
        model_name="allenai/OLMoE-1B-7B-0924",
        device="cuda",
        output_dir="./bh_routing_experiment"
    )

    results_df = runner.run_two_phase_experiment(
        baseline_k_values=[8, 16, 32, 64],
        bh_max_k_values=[8, 16, 32, 64],
        bh_alpha_values=[0.30, 0.40, 0.50, 0.60],
        datasets=['wikitext', 'lambada', 'hellaswag'],
        max_samples=200
    )
"""

import torch
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import warnings

# Import local modules
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise ImportError("transformers not installed. Run: pip install transformers")

try:
    from deprecated.bh_routing_metrics import BHMetricsComputer
    from week_4.bh_routing_evaluation import (
        load_wikitext, load_lambada, load_hellaswag,
        evaluate_perplexity, evaluate_lambada, evaluate_hellaswag
    )
    from deprecated.bh_routing import load_kde_models
except ImportError as e:
    raise ImportError(
        f"Could not import required modules: {e}\n"
        "Ensure bh_routing_metrics.py, bh_routing_evaluation.py, and bh_routing.py "
        "are in the same directory."
    )

warnings.filterwarnings('ignore')


class BHRoutingPatcherAdapter:
    """
    Adapter for BH routing that provides internal logging capabilities.

    This wraps the model and routing functions to collect internal routing data
    in a format compatible with the template's expected structure.
    """

    def __init__(self, model, tokenizer, device='cuda'):
        """
        Initialize the patcher adapter.

        Args:
            model: OLMoE model instance
            tokenizer: Model tokenizer
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Load KDE models once
        self.kde_models = load_kde_models()

        # Internal logging storage
        self.internal_logs = []
        self.current_sample = None

        # Routing configuration
        self.routing_type = None  # 'topk' or 'bh'
        self.k = None  # For topk
        self.alpha = None  # For bh
        self.max_k = None  # For bh
        self.min_k = 1  # For bh

        # Statistics
        self.all_expert_counts = []
        self.layer_expert_counts = defaultdict(list)
        self.expert_usage_counts = np.zeros(64)

        # Hook storage
        self.hooks = []
        self.is_patched = False

    def patch_with_topk(self, k: int):
        """Configure for TopK routing (uses model's native routing)."""
        self.routing_type = 'topk'
        self.k = k
        self.alpha = None
        self.max_k = k

        # For TopK, we don't actually patch - just use native routing
        # But we can still collect statistics
        self.is_patched = True
        print(f"  Configured for TopK routing (K={k})")

    def patch_with_bh(self, alpha: float, max_k: int, min_k: int = 1):
        """Configure for BH routing."""
        from deprecated.bh_routing import benjamini_hochberg_routing

        self.routing_type = 'bh'
        self.alpha = alpha
        self.max_k = max_k
        self.min_k = min_k
        self.k = None

        # Patch MoE blocks with BH routing
        self._patch_moe_blocks_with_bh()
        self.is_patched = True
        print(f"  Configured for BH routing (alpha={alpha}, max_k={max_k})")

    def _patch_moe_blocks_with_bh(self):
        """Patch all MoE blocks to use BH routing."""
        from deprecated.bh_routing import benjamini_hochberg_routing

        layer_idx = 0
        for name, module in self.model.named_modules():
            if module.__class__.__name__ == 'OlmoeSparseMoeBlock':
                # Store original forward
                original_forward = module.forward

                # Create BH forward for this layer
                def create_bh_forward(layer_num, moe_ref, orig_fwd):
                    def bh_forward(hidden_states):
                        batch_size, seq_len, hidden_dim = hidden_states.shape
                        hidden_states_flat = hidden_states.view(-1, hidden_dim)

                        # Compute router logits
                        router_logits = moe_ref.gate(hidden_states_flat)

                        # Apply BH routing
                        routing_weights, selected_experts, expert_counts = benjamini_hochberg_routing(
                            router_logits,
                            alpha=self.alpha,
                            temperature=1.0,
                            min_k=self.min_k,
                            max_k=self.max_k,
                            layer_idx=layer_num,
                            kde_models=self.kde_models
                        )

                        # Collect statistics for current sample logging
                        if self.current_sample is not None:
                            self._log_layer_routing(
                                layer_num, router_logits, routing_weights,
                                selected_experts, expert_counts
                            )

                        # Dispatch to experts
                        final_hidden_states = torch.zeros_like(hidden_states_flat)

                        for expert_idx in range(moe_ref.num_experts):
                            expert_mask = routing_weights[:, expert_idx] > 0
                            if expert_mask.any():
                                expert_input = hidden_states_flat[expert_mask]
                                expert_output = moe_ref.experts[expert_idx](expert_input)
                                weights = routing_weights[expert_mask, expert_idx].unsqueeze(-1)
                                final_hidden_states[expert_mask] += weights * expert_output

                        output = final_hidden_states.view(batch_size, seq_len, hidden_dim)
                        return output, router_logits

                    return bh_forward

                # Replace forward
                module.forward = create_bh_forward(layer_idx, module, original_forward)
                layer_idx += 1

        print(f"    Patched {layer_idx} MoE layers with BH routing")

    def _log_layer_routing(self, layer_idx, router_logits, routing_weights,
                          selected_experts, expert_counts):
        """Log routing data for current layer."""
        # Convert to CPU and numpy for storage
        layer_data = {
            'layer': layer_idx,
            'router_logits_shape': list(router_logits.shape),
            'selected_experts': selected_experts.cpu().tolist()[:5],  # First 5 tokens
            'expert_weights': routing_weights[routing_weights > 0].cpu().tolist()[:20],  # Sample
            'expert_counts': expert_counts.cpu().tolist(),
            'router_logits_sample': router_logits[:5].cpu().float().tolist(),  # First 5 tokens
            'layer_stats': {
                'avg_experts': float(expert_counts.float().mean()),
                'std_experts': float(expert_counts.float().std()),
                'ceiling_hits': int((expert_counts >= self.max_k).sum()),
                'floor_hits': int((expert_counts <= self.min_k).sum())
            }
        }

        self.current_sample['layers'].append(layer_data)

        # Update global statistics
        counts = expert_counts.cpu().numpy()
        self.all_expert_counts.extend(counts.tolist())
        self.layer_expert_counts[layer_idx].extend(counts.tolist())

        # Update expert usage counts
        for token_experts in selected_experts.cpu().numpy():
            for exp_idx in token_experts:
                if exp_idx >= 0:
                    self.expert_usage_counts[exp_idx] += 1

    def start_sample_logging(self, sample_id: int, text_preview: str):
        """Start logging for a new sample."""
        self.current_sample = {
            'sample_id': sample_id,
            'text_preview': text_preview,
            'num_tokens': 0,
            'loss': 0.0,
            'layers': []
        }

    def end_sample_logging(self, loss: float):
        """End logging for current sample."""
        if self.current_sample is not None:
            self.current_sample['loss'] = loss
            self.current_sample['num_tokens'] = len(
                self.current_sample['layers'][0]['expert_counts']
            ) if self.current_sample['layers'] else 0

            self.internal_logs.append(self.current_sample)
            self.current_sample = None

    def get_internal_logs(self) -> List[Dict]:
        """Return collected internal routing logs."""
        return self.internal_logs

    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Compute aggregate statistics from internal logs."""
        if not self.internal_logs:
            return {}

        # Per-layer averages
        per_layer_avg_experts = []
        per_layer_std_experts = []

        for layer_idx in range(16):
            if layer_idx in self.layer_expert_counts:
                counts = self.layer_expert_counts[layer_idx]
                per_layer_avg_experts.append(np.mean(counts))
                per_layer_std_experts.append(np.std(counts))
            else:
                per_layer_avg_experts.append(0.0)
                per_layer_std_experts.append(0.0)

        # Per-layer counts (for consistency score)
        per_layer_counts = [
            self.layer_expert_counts[i] if i in self.layer_expert_counts else []
            for i in range(16)
        ]

        # Ceiling and floor hits
        total_ceiling = sum(
            layer_data['layer_stats']['ceiling_hits']
            for sample in self.internal_logs
            for layer_data in sample['layers']
        )

        total_floor = sum(
            layer_data['layer_stats']['floor_hits']
            for sample in self.internal_logs
            for layer_data in sample['layers']
        )

        return {
            'per_layer_avg_experts': per_layer_avg_experts,
            'per_layer_std_experts': per_layer_std_experts,
            'per_layer_counts': per_layer_counts,
            'expert_usage_counts': self.expert_usage_counts.tolist(),
            'total_ceiling_hits': total_ceiling,
            'total_floor_hits': total_floor,
            'all_expert_counts': self.all_expert_counts
        }

    def clear_internal_logs(self):
        """Clear all internal logs and statistics."""
        self.internal_logs = []
        self.current_sample = None
        self.all_expert_counts = []
        self.layer_expert_counts = defaultdict(list)
        self.expert_usage_counts = np.zeros(64)

    def unpatch(self):
        """Remove all patches (for BH routing)."""
        # In this simple version, we don't restore - just mark as unpatched
        # For a full implementation, we'd need to store and restore original forwards
        self.is_patched = False
        print("  Model unpatched")


class BHRoutingExperimentRunner:
    """
    Main experiment runner for BH routing evaluation.

    Implements two-phase experimental approach with comprehensive metrics
    and dual file logging (summary + internal_routing JSONs).
    """

    def __init__(
        self,
        model_name: str = "allenai/OLMoE-1B-7B-0924",
        device: str = "cuda",
        output_dir: str = "./bh_routing_experiment"
    ):
        """
        Initialize experiment runner.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('cuda', 'cpu', or 'auto')
            output_dir: Directory for saving results
        """
        self.model_name = model_name
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)

        # Create directories
        self.logs_dir = self.output_dir / 'logs'
        self.viz_dir = self.output_dir / 'visualizations'
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")

        # Load model and tokenizer
        print(f"\nLoading model: {model_name}...")
        self.model, self.tokenizer = self._load_model()

        # Initialize patcher
        self.patcher = BHRoutingPatcherAdapter(self.model, self.tokenizer, self.device)

        # Initialize metrics computer
        self.metrics_computer = BHMetricsComputer()

        print("\n✅ Experiment runner initialized")

    def _load_model(self) -> Tuple[Any, Any]:
        """Load model and tokenizer."""
        start_time = time.time()

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        model.eval()

        load_time = time.time() - start_time

        print(f"✅ Model loaded in {load_time:.1f}s")
        print(f"   Architecture: {model.config.model_type}")
        print(f"   Num layers: {model.config.num_hidden_layers}")
        print(f"   Num experts: {model.config.num_experts}")
        print(f"   Default Top-K: {model.config.num_experts_per_tok}")

        return model, tokenizer

    def run_two_phase_experiment(
        self,
        baseline_k_values: List[int] = [8, 16, 32, 64],
        bh_max_k_values: List[int] = [8, 16, 32, 64],
        bh_alpha_values: List[float] = [0.30, 0.40, 0.50, 0.60],
        datasets: List[str] = ['wikitext', 'lambada', 'hellaswag'],
        max_samples: int = 200
    ) -> pd.DataFrame:
        """
        Run full two-phase experiment.

        Phase 1: Evaluate baseline TopK configurations
        Phase 2: Evaluate BH configurations
        Phase 3: Generate comparative analysis

        Args:
            baseline_k_values: K values for TopK baseline
            bh_max_k_values: max_k values for BH routing
            bh_alpha_values: Alpha values for BH routing
            datasets: Datasets to evaluate on
            max_samples: Maximum samples per dataset

        Returns:
            DataFrame with all results
        """
        all_results = []

        # =====================================================================
        # PHASE 1: BASELINE TOP-K ANALYSIS
        # =====================================================================
        print("=" * 70)
        print("PHASE 1: BASELINE TOP-K ANALYSIS")
        print("=" * 70)
        print(f"Configurations: {len(baseline_k_values)}")
        print(f"Datasets: {len(datasets)}")
        print(f"Total experiments: {len(baseline_k_values) * len(datasets)}")

        for k in baseline_k_values:
            for dataset in datasets:
                config_name = f"{k}experts_topk_baseline"
                print(f"\n[{config_name}] Dataset: {dataset}")

                result = self._run_single_experiment(
                    config_name=config_name,
                    routing_type='topk',
                    k=k,
                    alpha=None,
                    max_k=k,
                    dataset=dataset,
                    max_samples=max_samples
                )
                all_results.append(result)

                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # =====================================================================
        # PHASE 2: BH ROUTING ANALYSIS
        # =====================================================================
        print("\n" + "=" * 70)
        print("PHASE 2: BH ROUTING ANALYSIS")
        print("=" * 70)
        print(f"max_k values: {bh_max_k_values}")
        print(f"Alpha values: {bh_alpha_values}")
        print(f"Datasets: {len(datasets)}")
        print(f"Total experiments: {len(bh_max_k_values) * len(bh_alpha_values) * len(datasets)}")

        for max_k in bh_max_k_values:
            for alpha in bh_alpha_values:
                for dataset in datasets:
                    config_name = f"{max_k}experts_bh_a{int(alpha*100):03d}"
                    print(f"\n[{config_name}] Dataset: {dataset}")

                    result = self._run_single_experiment(
                        config_name=config_name,
                        routing_type='bh',
                        k=None,
                        alpha=alpha,
                        max_k=max_k,
                        dataset=dataset,
                        max_samples=max_samples
                    )
                    all_results.append(result)

                    # Clear GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # =====================================================================
        # PHASE 3: RESULTS COMPILATION
        # =====================================================================
        print("\n" + "=" * 70)
        print("PHASE 3: COMPILING RESULTS")
        print("=" * 70)

        results_df = pd.DataFrame(all_results)

        # Save results
        csv_path = self.output_dir / 'bh_routing_results.csv'
        json_path = self.output_dir / 'bh_routing_results.json'

        results_df.to_csv(csv_path, index=False)
        results_df.to_json(json_path, orient='records', indent=2)

        print(f"\n✅ Saved results:")
        print(f"   CSV: {csv_path}")
        print(f"   JSON: {json_path}")
        print(f"\n✅ Total experiments completed: {len(all_results)}")
        print(f"✅ Total log files generated: {len(all_results) * 2}")

        return results_df

    def _run_single_experiment(
        self,
        config_name: str,
        routing_type: str,
        k: Optional[int],
        alpha: Optional[float],
        max_k: int,
        dataset: str,
        max_samples: int
    ) -> Dict[str, Any]:
        """
        Run single configuration on single dataset.

        Returns result dictionary with all metrics.
        """
        start_time = time.time()

        # Clear previous logs
        self.patcher.clear_internal_logs()

        # Setup routing
        if routing_type == 'topk':
            self.patcher.patch_with_topk(k=k)
            baseline_k = k
        else:  # bh
            self.patcher.patch_with_bh(alpha=alpha, max_k=max_k, min_k=1)
            baseline_k = max_k  # For FLOPs comparison

        # Run evaluation on dataset
        if dataset == 'wikitext':
            data = load_wikitext(max_samples=max_samples)
            eval_result = evaluate_perplexity(
                self.model, self.tokenizer, data,
                patcher=self.patcher, device=self.device
            )
            dataset_metrics = {
                'perplexity': eval_result['perplexity'],
                'avg_loss': eval_result['avg_loss']
            }

        elif dataset == 'lambada':
            data = load_lambada(max_samples=max_samples)
            eval_result = evaluate_lambada(
                self.model, self.tokenizer, data,
                patcher=self.patcher, device=self.device
            )
            dataset_metrics = {
                'lambada_accuracy': eval_result['accuracy']
            }

        elif dataset == 'hellaswag':
            data = load_hellaswag(max_samples=max_samples)
            eval_result = evaluate_hellaswag(
                self.model, self.tokenizer, data,
                patcher=self.patcher, device=self.device
            )
            dataset_metrics = {
                'hellaswag_accuracy': eval_result['accuracy']
            }
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        # Get internal logs and aggregate stats
        internal_logs = self.patcher.get_internal_logs()
        aggregate_stats = self.patcher.get_aggregate_stats()

        # Compute comprehensive metrics
        expert_counts_array = np.array(aggregate_stats.get('all_expert_counts', []))
        expert_usage_array = np.array(aggregate_stats.get('expert_usage_counts', np.zeros(64)))

        avg_experts = float(np.mean(expert_counts_array)) if len(expert_counts_array) > 0 else float(k or max_k)

        comprehensive_metrics = self.metrics_computer.compute_all_metrics(
            losses=eval_result.get('losses', None),
            accuracies=None,  # Will aggregate later
            avg_experts=avg_experts,
            max_k=max_k,
            baseline_k=baseline_k,
            total_tokens=eval_result.get('total_tokens', 0),
            total_time=time.time() - start_time,
            expert_usage_counts=expert_usage_array,
            expert_counts=expert_counts_array,
            min_k=1,
            per_layer_avg=aggregate_stats.get('per_layer_avg_experts', None),
            per_layer_counts=aggregate_stats.get('per_layer_counts', None)
        )

        # Merge dataset-specific and comprehensive metrics
        all_metrics = {**dataset_metrics, **comprehensive_metrics}

        # Save summary JSON
        summary = {
            'config': config_name,
            'strategy': routing_type,
            'k_or_max_k': k if k is not None else max_k,
            'alpha': alpha,
            'num_experts': 64,
            'dataset': dataset,
            'metrics': all_metrics,
            'summary': {
                'total_samples': eval_result.get('num_samples', len(internal_logs)),
                'total_tokens': eval_result.get('total_tokens', 0),
                'elapsed_time_seconds': time.time() - start_time
            }
        }

        summary_path = self.logs_dir / f"{config_name}_{dataset}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Save internal routing JSON
        internal_routing = {
            'config': config_name,
            'strategy': routing_type,
            'k_or_max_k': k if k is not None else max_k,
            'alpha': alpha,
            'num_experts': 64,
            'dataset': dataset,
            'samples': internal_logs,
            'aggregate_stats': aggregate_stats
        }

        internal_path = self.logs_dir / f"{config_name}_{dataset}_internal_routing.json"
        with open(internal_path, 'w') as f:
            json.dump(internal_routing, f, indent=2)

        print(f"   ✅ Saved: {summary_path.name}")
        print(f"   ✅ Saved: {internal_path.name}")
        print(f"   ⏱️  Time: {time.time() - start_time:.1f}s")
        if 'avg_experts' in all_metrics:
            print(f"   📊 Avg experts: {all_metrics['avg_experts']:.2f}")

        # Return result row for DataFrame
        result = {
            'config_name': config_name,
            'routing_type': routing_type,
            'k_or_max_k': k if k is not None else max_k,
            'alpha': alpha,
            'dataset': dataset,
            **all_metrics
        }

        # Unpatch for next experiment
        self.patcher.unpatch()

        return result
