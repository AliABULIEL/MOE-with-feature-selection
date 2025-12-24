"""
Comprehensive Metrics for BH & HC Routing Evaluation
=====================================================

Provides:
- Perplexity (properly computed, handles edge cases)
- Token accuracy
- LAMBADA last-word prediction accuracy
- HellaSwag multiple-choice accuracy
- F1 score and exact match (for QA tasks)
- Routing efficiency metrics

Usage:
    from moe_metrics import MetricsComputer, ComprehensiveMetrics

    metrics = ComprehensiveMetrics()
    metrics.perplexity = MetricsComputer.compute_perplexity(total_loss, total_tokens)
"""

import torch
import numpy as np
import re
from typing import Dict, Tuple, List, Any, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class ComprehensiveMetrics:
    """All evaluation metrics in one container."""

    # === Quality Metrics ===
    perplexity: float = float('inf')
    token_accuracy: float = 0.0
    loss: float = float('inf')

    # === Task-Specific Metrics ===
    lambada_accuracy: float = 0.0      # Last word prediction
    hellaswag_accuracy: float = 0.0    # Multiple choice
    f1_score: float = 0.0              # Token-level F1
    exact_match: float = 0.0           # Exact string match

    # === Efficiency Metrics ===
    tokens_per_second: float = 0.0
    inference_time: float = 0.0

    # === Routing Metrics (BH/HC specific) ===
    avg_experts: float = 0.0
    std_experts: float = 0.0
    min_experts: int = 0
    max_experts: int = 0
    median_experts: float = 0.0

    # === Routing Behavior ===
    ceiling_hit_rate: float = 0.0      # % at max_k
    floor_hit_rate: float = 0.0        # % at min_k (usually 1)
    mid_range_rate: float = 0.0        # % in between
    adaptive_range: int = 0            # max - min experts used

    # === Expert Utilization ===
    avg_entropy: float = 0.0
    avg_max_weight: float = 0.0
    avg_concentration: float = 0.0
    unique_experts: int = 0
    expert_utilization: float = 0.0    # unique_experts / 64

    # === Comparison Metrics ===
    reduction_vs_baseline: float = 0.0  # % reduction vs TopK=8

    # === Counts ===
    num_samples: int = 0
    total_tokens: int = 0
    num_errors: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    def is_valid(self) -> bool:
        """Check if metrics are valid (no NaN/Inf in key fields)."""
        return (
            np.isfinite(self.perplexity) and
            np.isfinite(self.loss) and
            self.num_samples > 0
        )

    def compute_reduction(self, baseline_experts: float = 8.0):
        """Compute reduction vs baseline."""
        if baseline_experts > 0 and self.avg_experts > 0:
            self.reduction_vs_baseline = (baseline_experts - self.avg_experts) / baseline_experts * 100


class MetricsComputer:
    """Static methods for computing various metrics."""

    # =========================================================================
    # CORE QUALITY METRICS
    # =========================================================================

    @staticmethod
    def compute_perplexity(total_loss: float, total_tokens: int) -> float:
        """
        Compute perplexity from accumulated cross-entropy loss.

        Perplexity = exp(average_loss)
        Lower is better. Typical values: 10-100 for language models.
        """
        if total_tokens == 0:
            return float('inf')
        avg_loss = total_loss / total_tokens

        # Prevent overflow
        if avg_loss > 100:
            return float('inf')
        if not np.isfinite(avg_loss):
            return float('inf')

        return float(np.exp(avg_loss))

    @staticmethod
    def compute_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[int, int]:
        """
        Compute next-token prediction accuracy.

        Args:
            logits: [batch, seq_len, vocab_size]
            labels: [batch, seq_len]

        Returns:
            (correct_count, total_count)
        """
        # Shift for next-token prediction
        predictions = torch.argmax(logits[:, :-1, :], dim=-1)
        targets = labels[:, 1:]

        # Ignore padding tokens (typically -100)
        mask = targets != -100
        correct = ((predictions == targets) & mask).sum().item()
        total = mask.sum().item()

        return correct, total

    # =========================================================================
    # TASK-SPECIFIC METRICS
    # =========================================================================

    @staticmethod
    def compute_lambada_accuracy(
        model,
        tokenizer,
        context: str,
        target_word: str,
        device: str = 'cuda'
    ) -> Tuple[float, str]:
        """
        Compute LAMBADA last-word prediction accuracy.

        Args:
            model: Language model
            tokenizer: Tokenizer
            context: Text without last word
            target_word: Expected last word
            device: Device to run on

        Returns:
            (accuracy, predicted_word) - accuracy is 1.0 or 0.0
        """
        inputs = tokenizer(context, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode generated tokens
        generated_ids = outputs[0, inputs['input_ids'].shape[1]:]
        generated = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Extract first word
        predicted_word = generated.strip().split()[0] if generated.strip() else ""

        # Normalize for comparison
        pred_norm = predicted_word.lower().strip('.,!?;:\'"')
        target_norm = target_word.lower().strip('.,!?;:\'"')

        accuracy = 1.0 if pred_norm == target_norm else 0.0
        return accuracy, predicted_word

    @staticmethod
    def compute_hellaswag_accuracy(
        model,
        tokenizer,
        context: str,
        choices: List[str],
        label: int,
        device: str = 'cuda'
    ) -> Tuple[float, int]:
        """
        Compute HellaSwag multiple-choice accuracy.

        Scores each choice by perplexity, selects lowest (most likely).

        Args:
            model: Language model
            tokenizer: Tokenizer
            context: Context/prompt
            choices: List of 4 possible endings
            label: Correct choice index (0-3)
            device: Device

        Returns:
            (accuracy, predicted_label) - accuracy is 1.0 or 0.0
        """
        choice_losses = []

        for choice in choices:
            full_text = context + " " + choice
            inputs = tokenizer(
                full_text,
                return_tensors='pt',
                truncation=True,
                max_length=512
            ).to(device)

            with torch.no_grad():
                outputs = model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
                loss = outputs.loss.item() if outputs.loss is not None else float('inf')

            choice_losses.append(loss)

        # Select choice with lowest loss (highest likelihood)
        predicted_label = int(np.argmin(choice_losses))
        accuracy = 1.0 if predicted_label == label else 0.0

        return accuracy, predicted_label

    # =========================================================================
    # F1 AND EXACT MATCH
    # =========================================================================

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for F1/EM computation."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text

    @staticmethod
    def compute_f1(prediction: str, reference: str) -> float:
        """
        Compute token-level F1 score.

        F1 = 2 * (precision * recall) / (precision + recall)
        """
        pred_tokens = MetricsComputer.normalize_text(prediction).split()
        ref_tokens = MetricsComputer.normalize_text(reference).split()

        if not pred_tokens or not ref_tokens:
            return 0.0

        common = set(pred_tokens) & set(ref_tokens)
        num_common = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in common)

        if num_common == 0:
            return 0.0

        precision = num_common / len(pred_tokens)
        recall = num_common / len(ref_tokens)

        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def compute_exact_match(prediction: str, reference: str) -> float:
        """Compute exact match (1.0 if match, 0.0 otherwise)."""
        pred_norm = MetricsComputer.normalize_text(prediction)
        ref_norm = MetricsComputer.normalize_text(reference)
        return 1.0 if pred_norm == ref_norm else 0.0

    # =========================================================================
    # ROUTING METRICS
    # =========================================================================

    @staticmethod
    def compute_routing_stats(
        expert_counts: np.ndarray,
        max_k: int,
        min_k: int = 1
    ) -> Dict[str, float]:
        """
        Compute routing statistics from expert count array.

        Args:
            expert_counts: Array of expert counts per token
            max_k: Maximum allowed experts
            min_k: Minimum allowed experts

        Returns:
            Dictionary of routing statistics
        """
        if len(expert_counts) == 0:
            return {}

        counts = np.array(expert_counts).flatten()

        return {
            'avg_experts': float(np.mean(counts)),
            'std_experts': float(np.std(counts)),
            'min_experts': int(np.min(counts)),
            'max_experts': int(np.max(counts)),
            'median_experts': float(np.median(counts)),
            'ceiling_hit_rate': float(np.sum(counts >= max_k) / len(counts) * 100),
            'floor_hit_rate': float(np.sum(counts <= min_k) / len(counts) * 100),
            'mid_range_rate': float(np.sum((counts > min_k) & (counts < max_k)) / len(counts) * 100),
            'adaptive_range': int(np.max(counts) - np.min(counts)),
            'reduction_vs_baseline': float((8.0 - np.mean(counts)) / 8.0 * 100)
        }
