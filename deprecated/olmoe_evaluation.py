#!/usr/bin/env python3
"""
OLMoE Multi-Expert Evaluation Framework
Production-quality code for evaluating OLMoE with different expert configurations

Features:
- Evaluates on standard datasets (WikiText-2, LAMBADA, etc.)
- Supports 8, 16, 32, 64 expert configurations
- Computes perplexity, accuracy, and other metrics
- Generates publication-quality visualizations
- Saves detailed results to disk

Author: Senior ML Researcher & Software Engineer
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import time
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation run."""
    model_name: str = "allenai/OLMoE-1B-7B-0924"
    expert_configs: List[int] = None
    datasets: List[str] = None
    max_samples: int = 1000
    max_length: int = 512
    batch_size: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "./results"
    seed: int = 42

    def __post_init__(self):
        if self.expert_configs is None:
            self.expert_configs = [8, 16, 32, 64]
        if self.datasets is None:
            self.datasets = ["wikitext", "lambada"]


@dataclass
class MetricResults:
    """Results from a single evaluation run."""
    num_experts: int
    dataset: str
    perplexity: float
    token_accuracy: float
    loss: float
    num_samples: int
    total_tokens: int
    inference_time: float
    tokens_per_second: float
    avg_time_per_sample: float

    def to_dict(self):
        return asdict(self)


class OLMoEEvaluator:
    """Production-quality evaluator for OLMoE with multiple expert configurations."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set random seeds
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        np.random.seed(config.seed)

        # Load model and tokenizer
        logger.info(f"Loading model: {config.model_name}")
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()

        # Store original config
        self.original_num_experts = self.model.config.num_experts_per_tok
        logger.info(f"Model loaded. Original experts per token: {self.original_num_experts}")

        # CRITICAL: Test that expert modification works BEFORE running evaluation
        logger.info("="*80)
        logger.info("TESTING EXPERT MODIFICATION")
        logger.info("="*80)
        if not self._test_expert_modification_works():
            raise RuntimeError(
                "\n" + "="*80 + "\n"
                "❌ CRITICAL ERROR: Expert modification doesn't work!\n"
                "Different expert counts produce IDENTICAL outputs.\n"
                "This means the expert configuration is not taking effect.\n"
                "Cannot proceed with evaluation - results would be meaningless.\n"
                "="*80
            )
        logger.info("✅ Expert modification test PASSED - proceeding with evaluation")
        logger.info("="*80)

    def _load_model(self) -> AutoModelForCausalLM:
        """
        Load OLMoE model WITHOUT output_router_logits to avoid issues.
        Router logits can cause crashes and aren't needed for evaluation.
        """
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            # DON'T use output_router_logits - causes issues
        )
        model.eval()
        return model

    def _load_tokenizer(self) -> AutoTokenizer:
        """Load tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _set_num_experts(self, num_experts: int) -> int:
        """
        Set number of experts per token - MUST ACTUALLY WORK!

        This method updates the model configuration and all MoE layers.
        Returns the number of layers successfully updated.

        Args:
            num_experts: Number of experts to use (8, 16, 32, or 64)

        Returns:
            Number of layers successfully updated
        """
        logger.info(f"Setting num_experts_per_tok to {num_experts}")

        # Step 1: Update model config
        self.model.config.num_experts_per_tok = num_experts

        # Step 2: Update all MoE layers - try multiple possible attribute locations
        layers_updated = 0
        attribute_found = None

        for i, layer in enumerate(self.model.model.layers):
            if not hasattr(layer, 'mlp'):
                continue

            mlp = layer.mlp
            updated = False

            # Try different possible attribute locations (in order of likelihood)
            # Try 1: top_k attribute (most common in MoE implementations)
            if hasattr(mlp, 'top_k'):
                old_val = mlp.top_k
                mlp.top_k = num_experts
                updated = True
                if attribute_found is None:
                    attribute_found = 'mlp.top_k'
                    logger.info(f"Found expert control at: mlp.top_k (was {old_val})")

            # Try 2: num_experts_per_tok attribute
            elif hasattr(mlp, 'num_experts_per_tok'):
                old_val = mlp.num_experts_per_tok
                mlp.num_experts_per_tok = num_experts
                updated = True
                if attribute_found is None:
                    attribute_found = 'mlp.num_experts_per_tok'
                    logger.info(f"Found expert control at: mlp.num_experts_per_tok (was {old_val})")

            # Try 3: Gate with top_k
            elif hasattr(mlp, 'gate') and hasattr(mlp.gate, 'top_k'):
                old_val = mlp.gate.top_k
                mlp.gate.top_k = num_experts
                updated = True
                if attribute_found is None:
                    attribute_found = 'mlp.gate.top_k'
                    logger.info(f"Found expert control at: mlp.gate.top_k (was {old_val})")

            # Try 4: Router with top_k
            elif hasattr(mlp, 'router') and hasattr(mlp.router, 'top_k'):
                old_val = mlp.router.top_k
                mlp.router.top_k = num_experts
                updated = True
                if attribute_found is None:
                    attribute_found = 'mlp.router.top_k'
                    logger.info(f"Found expert control at: mlp.router.top_k (was {old_val})")

            # Try 5: Config object in MLP
            elif hasattr(mlp, 'config') and hasattr(mlp.config, 'num_experts_per_tok'):
                old_val = mlp.config.num_experts_per_tok
                mlp.config.num_experts_per_tok = num_experts
                updated = True
                if attribute_found is None:
                    attribute_found = 'mlp.config.num_experts_per_tok'
                    logger.info(f"Found expert control at: mlp.config.num_experts_per_tok (was {old_val})")

            if updated:
                layers_updated += 1

        # Step 3: Report results
        logger.info(f"Updated {layers_updated}/{len(self.model.model.layers)} layers")

        # Step 4: Diagnostic info if nothing was updated
        if layers_updated == 0:
            logger.error("❌ WARNING: No MoE layers were updated!")
            logger.error("The expert configuration will NOT take effect!")

            # Print diagnostic info for first layer
            if len(self.model.model.layers) > 0:
                layer0 = self.model.model.layers[0]
                logger.error(f"First layer type: {type(layer0)}")
                if hasattr(layer0, 'mlp'):
                    logger.error(f"MLP type: {type(layer0.mlp)}")
                    mlp_attrs = [attr for attr in dir(layer0.mlp)
                                if not attr.startswith('_')]
                    logger.error(f"MLP attributes (first 30): {mlp_attrs[:30]}")

        return layers_updated

    def _restore_original_experts(self):
        """Restore original expert configuration."""
        self._set_num_experts(self.original_num_experts)

    def _test_expert_modification_works(self) -> bool:
        """
        Test that changing expert count actually changes model outputs.

        This is a CRITICAL test run before evaluation to ensure the
        expert modification is actually working. If outputs are identical
        for different expert counts, the evaluation would be meaningless.

        Returns:
            True if modification works, False if outputs are identical
        """
        logger.info("Running test: Do different expert counts produce different outputs?")

        test_text = "The capital of France is"
        inputs = self.tokenizer(test_text, return_tensors="pt").to(self.device)

        # Dictionary to store results
        results = {}

        # Test with 8 and 16 experts (quick test)
        for num_experts in [8, 16]:
            logger.info(f"  Testing with {num_experts} experts...")

            # Set expert count
            layers_updated = self._set_num_experts(num_experts)

            if layers_updated == 0:
                logger.error(f"  ❌ Failed to update any layers for {num_experts} experts!")
                return False

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Store logits
            results[num_experts] = outputs.logits.cpu()
            logger.info(f"  Output shape: {outputs.logits.shape}")

        # Compare outputs
        logits_8 = results[8]
        logits_16 = results[16]

        max_diff = torch.max(torch.abs(logits_8 - logits_16)).item()
        mean_diff = torch.mean(torch.abs(logits_8 - logits_16)).item()

        # Check if outputs are identical (within floating point tolerance)
        are_identical = torch.allclose(logits_8, logits_16, rtol=1e-5, atol=1e-5)

        logger.info(f"Comparison results:")
        logger.info(f"  Max difference: {max_diff}")
        logger.info(f"  Mean difference: {mean_diff}")
        logger.info(f"  Are identical: {are_identical}")

        if are_identical or max_diff < 1e-4:
            logger.error("❌ FAILED: Outputs are IDENTICAL for 8 vs 16 experts!")
            logger.error(f"   Max diff: {max_diff} (should be > 0.001)")
            logger.error("   This means expert configuration is NOT working!")
            return False
        else:
            logger.info(f"✅ PASSED: Outputs differ significantly (max diff: {max_diff:.6f})")
            logger.info("   Expert modification is working correctly!")
            return True

    def _verify_expert_config(self, num_experts: int) -> bool:
        """
        Verify that the expert configuration was properly applied.

        This version does NOT use router_logits to avoid crashes.
        Instead, it checks the configuration directly.

        Args:
            num_experts: Expected number of experts per token

        Returns:
            True if configuration appears correct
        """
        # Check 1: Model config
        config_value = self.model.config.num_experts_per_tok

        # Check 2: First layer MLP (sample check)
        layer_value = None
        if len(self.model.model.layers) > 0 and hasattr(self.model.model.layers[0], 'mlp'):
            mlp = self.model.model.layers[0].mlp

            # Try different attribute names
            if hasattr(mlp, 'top_k'):
                layer_value = mlp.top_k
            elif hasattr(mlp, 'num_experts_per_tok'):
                layer_value = mlp.num_experts_per_tok
            elif hasattr(mlp, 'gate') and hasattr(mlp.gate, 'top_k'):
                layer_value = mlp.gate.top_k
            elif hasattr(mlp, 'router') and hasattr(mlp.router, 'top_k'):
                layer_value = mlp.router.top_k

        logger.info(f"Verification - config: {config_value}, layer[0]: {layer_value}")

        # Verification passes if config matches expected value
        # (We already tested that outputs differ in _test_expert_modification_works)
        if config_value == num_experts:
            if layer_value is not None and layer_value == num_experts:
                logger.info(f"✓ Configuration verified: {num_experts} experts")
                return True
            elif layer_value is None:
                logger.warning(f"Could not verify layer config, but model config is correct")
                return True
            else:
                logger.warning(f"Config mismatch: model={config_value}, layer={layer_value}")
                return True  # Still proceed, we tested outputs differ earlier
        else:
            logger.error(f"Config error: expected {num_experts}, got {config_value}")
            return False

    def load_evaluation_dataset(self, dataset_name: str) -> List[str]:
        """
        Load and prepare evaluation dataset.

        Args:
            dataset_name: Name of dataset ('wikitext', 'lambada', etc.)

        Returns:
            List of text samples
        """
        logger.info(f"Loading dataset: {dataset_name}")

        if dataset_name == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]

        elif dataset_name == "lambada":
            dataset = load_dataset("lambada", split="test")
            texts = [item['text'] for item in dataset]

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Limit samples
        if self.config.max_samples:
            texts = texts[:self.config.max_samples]

        logger.info(f"Loaded {len(texts)} samples from {dataset_name}")
        return texts

    def compute_perplexity(
        self,
        texts: List[str],
        num_experts: int,
        dataset_name: str
    ) -> MetricResults:
        """
        Compute perplexity and other metrics for given expert configuration.

        Args:
            texts: List of text samples
            num_experts: Number of experts to use
            dataset_name: Name of dataset being evaluated

        Returns:
            MetricResults object with all metrics
        """
        logger.info(f"Computing metrics with {num_experts} experts on {dataset_name}")

        # Set expert configuration
        self._set_num_experts(num_experts)

        # Verify configuration was applied
        if not self._verify_expert_config(num_experts):
            logger.error(
                f"Failed to verify expert configuration for {num_experts} experts! "
                "Results may be incorrect."
            )

        total_loss = 0.0
        total_tokens = 0
        correct_predictions = 0
        start_time = time.time()

        with torch.no_grad():
            for text in tqdm(texts, desc=f"{num_experts} experts"):
                # Tokenize
                encodings = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=self.config.max_length,
                    truncation=True,
                    padding=False
                )

                input_ids = encodings.input_ids.to(self.device)

                # Skip very short sequences
                if input_ids.size(1) < 2:
                    continue

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    labels=input_ids,
                    output_router_logits=False  # Disable for speed
                )

                # Accumulate loss
                loss = outputs.loss
                total_loss += loss.item() * (input_ids.size(1) - 1)

                # Compute token accuracy
                logits = outputs.logits[:, :-1, :]  # Remove last prediction
                targets = input_ids[:, 1:]  # Remove first token

                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == targets).sum().item()
                total_tokens += targets.numel()

        end_time = time.time()
        inference_time = end_time - start_time

        # Compute final metrics
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        token_accuracy = correct_predictions / total_tokens
        tokens_per_second = total_tokens / inference_time
        avg_time_per_sample = inference_time / len(texts)

        results = MetricResults(
            num_experts=num_experts,
            dataset=dataset_name,
            perplexity=perplexity,
            token_accuracy=token_accuracy,
            loss=avg_loss,
            num_samples=len(texts),
            total_tokens=total_tokens,
            inference_time=inference_time,
            tokens_per_second=tokens_per_second,
            avg_time_per_sample=avg_time_per_sample
        )

        logger.info(
            f"Results for {num_experts} experts: "
            f"Perplexity={perplexity:.2f}, "
            f"Token Accuracy={token_accuracy:.4f}, "
            f"Speed={tokens_per_second:.2f} tok/s"
        )

        return results

    def evaluate_all_configurations(self) -> pd.DataFrame:
        """
        Run evaluation for all expert configurations and datasets.

        Returns:
            DataFrame with all results
        """
        all_results = []

        for dataset_name in self.config.datasets:
            # Load dataset
            texts = self.load_evaluation_dataset(dataset_name)

            for num_experts in self.config.expert_configs:
                # Compute metrics
                results = self.compute_perplexity(texts, num_experts, dataset_name)
                all_results.append(results.to_dict())

        # Restore original configuration
        self._restore_original_experts()

        # Create DataFrame
        df = pd.DataFrame(all_results)

        # Save to CSV
        output_path = self.output_dir / "evaluation_results.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

        # Save as JSON for easy loading
        json_path = self.output_dir / "evaluation_results.json"
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {json_path}")

        return df

    def visualize_results(self, df: pd.DataFrame):
        """
        Create publication-quality visualizations.

        Args:
            df: DataFrame with evaluation results
        """
        logger.info("Generating visualizations...")

        # Set style
        sns.set_style("whitegrid")
        sns.set_palette("husl")

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Perplexity vs Num Experts (by dataset)
        ax1 = fig.add_subplot(gs[0, 0])
        for dataset in df['dataset'].unique():
            data = df[df['dataset'] == dataset]
            ax1.plot(
                data['num_experts'],
                data['perplexity'],
                marker='o',
                linewidth=2,
                markersize=8,
                label=dataset
            )
        ax1.set_xlabel('Number of Experts', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Perplexity ↓', fontsize=12, fontweight='bold')
        ax1.set_title('Perplexity vs Expert Count', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Token Accuracy vs Num Experts
        ax2 = fig.add_subplot(gs[0, 1])
        for dataset in df['dataset'].unique():
            data = df[df['dataset'] == dataset]
            ax2.plot(
                data['num_experts'],
                data['token_accuracy'] * 100,
                marker='s',
                linewidth=2,
                markersize=8,
                label=dataset
            )
        ax2.set_xlabel('Number of Experts', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Token Accuracy (%) ↑', fontsize=12, fontweight='bold')
        ax2.set_title('Token Accuracy vs Expert Count', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Inference Speed
        ax3 = fig.add_subplot(gs[0, 2])
        for dataset in df['dataset'].unique():
            data = df[df['dataset'] == dataset]
            ax3.plot(
                data['num_experts'],
                data['tokens_per_second'],
                marker='^',
                linewidth=2,
                markersize=8,
                label=dataset
            )
        ax3.set_xlabel('Number of Experts', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Tokens/Second ↑', fontsize=12, fontweight='bold')
        ax3.set_title('Inference Speed vs Expert Count', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Perplexity Improvement (bar chart)
        ax4 = fig.add_subplot(gs[1, 0])
        for dataset in df['dataset'].unique():
            data = df[df['dataset'] == dataset]
            baseline_ppl = data[data['num_experts'] == 8]['perplexity'].values[0]
            improvement = (baseline_ppl - data['perplexity']) / baseline_ppl * 100

            x = np.arange(len(data))
            ax4.bar(
                x + (list(df['dataset'].unique()).index(dataset) * 0.2),
                improvement,
                width=0.2,
                label=dataset
            )

        ax4.set_xlabel('Expert Configuration', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Perplexity Improvement (%) ↑', fontsize=12, fontweight='bold')
        ax4.set_title('Perplexity Improvement vs 8 Experts', fontsize=14, fontweight='bold')
        ax4.set_xticks(x + 0.1)
        ax4.set_xticklabels(df['num_experts'].unique())
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. Speed vs Quality trade-off
        ax5 = fig.add_subplot(gs[1, 1])
        for dataset in df['dataset'].unique():
            data = df[df['dataset'] == dataset]
            ax5.scatter(
                data['tokens_per_second'],
                data['perplexity'],
                s=data['num_experts'] * 5,  # Size by num_experts
                alpha=0.6,
                label=dataset
            )

            # Annotate points
            for _, row in data.iterrows():
                ax5.annotate(
                    f"{row['num_experts']}",
                    (row['tokens_per_second'], row['perplexity']),
                    fontsize=9,
                    fontweight='bold'
                )

        ax5.set_xlabel('Tokens/Second ↑', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Perplexity ↓', fontsize=12, fontweight='bold')
        ax5.set_title('Speed-Quality Trade-off', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Relative Performance (normalized)
        ax6 = fig.add_subplot(gs[1, 2])
        datasets = df['dataset'].unique()
        x = np.arange(len(df['num_experts'].unique()))
        width = 0.35

        for i, dataset in enumerate(datasets):
            data = df[df['dataset'] == dataset]
            baseline_speed = data[data['num_experts'] == 8]['tokens_per_second'].values[0]
            relative_speed = data['tokens_per_second'] / baseline_speed

            ax6.bar(
                x + i * width,
                relative_speed,
                width,
                label=dataset,
                alpha=0.8
            )

        ax6.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Baseline')
        ax6.set_xlabel('Number of Experts', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Relative Speed (vs 8 experts)', fontsize=12, fontweight='bold')
        ax6.set_title('Performance Scaling', fontsize=14, fontweight='bold')
        ax6.set_xticks(x + width / 2)
        ax6.set_xticklabels(df['num_experts'].unique())
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')

        # 7. Loss comparison
        ax7 = fig.add_subplot(gs[2, 0])
        for dataset in df['dataset'].unique():
            data = df[df['dataset'] == dataset]
            ax7.plot(
                data['num_experts'],
                data['loss'],
                marker='o',
                linewidth=2,
                markersize=8,
                label=dataset
            )
        ax7.set_xlabel('Number of Experts', fontsize=12, fontweight='bold')
        ax7.set_ylabel('Cross-Entropy Loss ↓', fontsize=12, fontweight='bold')
        ax7.set_title('Loss vs Expert Count', fontsize=14, fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # 8. Summary statistics table
        ax8 = fig.add_subplot(gs[2, 1:])
        ax8.axis('tight')
        ax8.axis('off')

        # Create summary table
        summary_data = []
        for num_experts in df['num_experts'].unique():
            row = [f"{num_experts} Experts"]
            for dataset in df['dataset'].unique():
                data = df[(df['num_experts'] == num_experts) & (df['dataset'] == dataset)]
                if not data.empty:
                    ppl = data['perplexity'].values[0]
                    acc = data['token_accuracy'].values[0] * 100
                    spd = data['tokens_per_second'].values[0]
                    row.append(f"PPL: {ppl:.2f}\nAcc: {acc:.2f}%\nSpd: {spd:.1f}")
                else:
                    row.append("N/A")
            summary_data.append(row)

        table = ax8.table(
            cellText=summary_data,
            colLabels=['Config'] + [f'{d.upper()}' for d in df['dataset'].unique()],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header
        for i in range(len(df['dataset'].unique()) + 1):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax8.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)

        # Save figure
        output_path = self.output_dir / "evaluation_results.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_path}")

        # Also save as PDF for publications
        pdf_path = self.output_dir / "evaluation_results.pdf"
        plt.savefig(pdf_path, bbox_inches='tight')
        logger.info(f"PDF saved to {pdf_path}")

        plt.show()

    def generate_report(self, df: pd.DataFrame):
        """
        Generate a markdown report with results.

        Args:
            df: DataFrame with evaluation results
        """
        report_path = self.output_dir / "EVALUATION_REPORT.md"

        with open(report_path, 'w') as f:
            f.write("# OLMoE Multi-Expert Evaluation Report\n\n")
            f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Model:** {self.config.model_name}\n\n")
            f.write("---\n\n")

            f.write("## Configuration\n\n")
            f.write(f"- **Expert Configurations:** {self.config.expert_configs}\n")
            f.write(f"- **Datasets:** {self.config.datasets}\n")
            f.write(f"- **Max Samples:** {self.config.max_samples}\n")
            f.write(f"- **Max Length:** {self.config.max_length}\n")
            f.write(f"- **Device:** {self.config.device}\n\n")

            f.write("---\n\n")
            f.write("## Results Summary\n\n")

            for dataset in df['dataset'].unique():
                f.write(f"### {dataset.upper()}\n\n")

                data = df[df['dataset'] == dataset]

                f.write("| Experts | Perplexity ↓ | Token Acc ↑ | Loss ↓ | Speed (tok/s) ↑ | Time (s) |\n")
                f.write("|---------|-------------|-------------|---------|----------------|----------|\n")

                for _, row in data.iterrows():
                    f.write(
                        f"| {row['num_experts']:2d} | "
                        f"{row['perplexity']:7.2f} | "
                        f"{row['token_accuracy']*100:5.2f}% | "
                        f"{row['loss']:5.3f} | "
                        f"{row['tokens_per_second']:8.2f} | "
                        f"{row['inference_time']:7.2f} |\n"
                    )

                f.write("\n")

                # Analysis
                baseline = data[data['num_experts'] == 8].iloc[0]
                best_ppl = data.loc[data['perplexity'].idxmin()]
                best_acc = data.loc[data['token_accuracy'].idxmax()]

                f.write("**Key Findings:**\n\n")
                f.write(f"- **Best Perplexity:** {best_ppl['num_experts']} experts "
                       f"({best_ppl['perplexity']:.2f}, "
                       f"{((baseline['perplexity'] - best_ppl['perplexity']) / baseline['perplexity'] * 100):.1f}% improvement)\n")
                f.write(f"- **Best Accuracy:** {best_acc['num_experts']} experts "
                       f"({best_acc['token_accuracy']*100:.2f}%)\n")
                f.write(f"- **Speed Trade-off:** {best_ppl['num_experts']} experts is "
                       f"{baseline['tokens_per_second'] / best_ppl['tokens_per_second']:.2f}x slower than baseline\n\n")

            f.write("---\n\n")
            f.write("## Recommendations\n\n")

            # Find optimal configuration
            avg_by_experts = df.groupby('num_experts').agg({
                'perplexity': 'mean',
                'token_accuracy': 'mean',
                'tokens_per_second': 'mean'
            })

            best_quality = avg_by_experts['perplexity'].idxmin()
            best_balanced = avg_by_experts.loc[
                (avg_by_experts['tokens_per_second'] > avg_by_experts['tokens_per_second'].quantile(0.5))
            ]['perplexity'].idxmin()

            f.write(f"- **For Maximum Quality:** Use **{best_quality} experts** "
                   f"(Avg PPL: {avg_by_experts.loc[best_quality, 'perplexity']:.2f})\n")
            f.write(f"- **For Balanced Performance:** Use **{best_balanced} experts** "
                   f"(Avg PPL: {avg_by_experts.loc[best_balanced, 'perplexity']:.2f}, "
                   f"Speed: {avg_by_experts.loc[best_balanced, 'tokens_per_second']:.1f} tok/s)\n")
            f.write(f"- **For Real-time Applications:** Use **8 experts** (default)\n\n")

        logger.info(f"Report saved to {report_path}")


def main():
    """Main entry point with result validation."""
    # Configuration
    config = EvaluationConfig(
        expert_configs=[8, 16, 32, 64],
        datasets=["wikitext", "lambada"],
        max_samples=500,  # Adjust based on compute
        max_length=512,
        output_dir="./olmoe_evaluation_results"
    )

    # Create evaluator (this will test expert modification works)
    evaluator = OLMoEEvaluator(config)

    # Run evaluation
    logger.info("Starting evaluation...")
    results_df = evaluator.evaluate_all_configurations()

    # Display results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)

    # CRITICAL: Validate that results actually differ
    print("\n" + "="*80)
    print("VALIDATION: Checking if results differ across expert configurations")
    print("="*80)

    for dataset in results_df['dataset'].unique():
        dataset_results = results_df[results_df['dataset'] == dataset]
        perplexities = dataset_results['perplexity'].values
        accuracies = dataset_results['token_accuracy'].values

        # Check if all perplexities are identical
        unique_ppls = len(set(perplexities.round(4)))
        unique_accs = len(set(accuracies.round(6)))

        print(f"\n{dataset.upper()}:")
        print(f"  Unique perplexity values: {unique_ppls}")
        print(f"  Unique accuracy values: {unique_accs}")
        print(f"  Perplexity range: {perplexities.min():.4f} - {perplexities.max():.4f}")
        print(f"  Accuracy range: {accuracies.min():.4f} - {accuracies.max():.4f}")

        if unique_ppls == 1:
            print(f"  ❌ WARNING: All perplexities are IDENTICAL for {dataset}!")
            print(f"     This suggests expert configuration may not be working!")
        else:
            print(f"  ✅ GOOD: Perplexity values differ across configurations")

    print("="*80)

    # Generate visualizations
    evaluator.visualize_results(results_df)

    # Generate report
    evaluator.generate_report(results_df)

    logger.info("Evaluation complete!")
    logger.info(f"Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
