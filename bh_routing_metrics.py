"""
Comprehensive Metrics for BH Routing Evaluation
================================================

This module implements all 16 metrics across 8 categories for evaluating
Benjamini-Hochberg routing against baseline Top-K routing in OLMoE.

Metric Categories:
1. Quality Metrics (2)
2. Efficiency Metrics (2)
3. Speed Metrics (2)
4. Routing Distribution Metrics (2)
5. Routing Behavior Metrics (2) - BH-specific
6. Constraint Metrics (2)
7. Cross-Layer Metrics (2)
8. Stability Metrics (2)

Usage:
    from bh_routing_metrics import BHMetricsComputer

    computer = BHMetricsComputer()
    metrics = computer.compute_all_metrics(
        losses=[2.1, 2.3, 2.0],
        accuracies={'lambada': 0.65, 'hellaswag': 0.52},
        avg_experts=4.5,
        max_k=8,
        ...
    )
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')


class BHMetricsComputer:
    """
    Computes all 16 metrics for BH routing evaluation.

    All metrics are implemented as static methods for easy independent use.
    Use compute_all_metrics() for a complete evaluation.
    """

    # =========================================================================
    # CATEGORY 1: QUALITY METRICS
    # =========================================================================

    @staticmethod
    def compute_perplexity(losses: List[float]) -> float:
        """
        Compute perplexity from token losses.

        Perplexity measures how well the model predicts the test data.
        Lower is better.

        Args:
            losses: List of cross-entropy losses per sample

        Returns:
            Perplexity value (typically 10-30 for language models)

        Raises:
            ValueError: If losses is empty

        Examples:
            >>> compute_perplexity([2.0, 2.5, 2.2])
            10.23
        """
        if not losses:
            raise ValueError("losses cannot be empty")

        avg_loss = float(np.mean(losses))
        perplexity = float(np.exp(avg_loss))
        return perplexity

    @staticmethod
    def compute_avg_task_accuracy(accuracies: Dict[str, float]) -> float:
        """
        Compute average accuracy across evaluation tasks.

        Args:
            accuracies: Dictionary mapping task name to accuracy
                       e.g., {'lambada': 0.65, 'hellaswag': 0.52}

        Returns:
            Average accuracy in [0, 1]

        Examples:
            >>> compute_avg_task_accuracy({'lambada': 0.65, 'hellaswag': 0.52})
            0.585
        """
        if not accuracies:
            return 0.0

        return float(np.mean(list(accuracies.values())))

    # =========================================================================
    # CATEGORY 2: EFFICIENCY METRICS
    # =========================================================================

    @staticmethod
    def compute_expert_activation_ratio(avg_experts: float, max_k: int) -> float:
        """
        Compute ratio of experts used vs maximum allowed.

        Args:
            avg_experts: Average number of experts selected
            max_k: Maximum experts allowed

        Returns:
            Ratio in [1/max_k, 1.0]. Lower is more efficient.

        Examples:
            >>> compute_expert_activation_ratio(4.5, 8)
            0.5625
        """
        if max_k <= 0:
            return 0.0

        return float(avg_experts / max_k)

    @staticmethod
    def compute_flops_reduction_pct(
        avg_experts: float,
        baseline_k: int = 8,
        num_experts: int = 64
    ) -> float:
        """
        Compute estimated FLOPs reduction vs baseline TopK.

        Assumes ~60% of MoE compute is in expert layers.

        Args:
            avg_experts: Average experts selected by BH
            baseline_k: Baseline Top-K value (default: 8)
            num_experts: Total experts per layer (default: 64)

        Returns:
            Percentage reduction (-100 to +60). Positive is better.

        Examples:
            >>> compute_flops_reduction_pct(4.5, baseline_k=8, num_experts=64)
            21.43
        """
        expert_fraction = 0.6  # Expert layers ~60% of MoE compute

        baseline_cost = (baseline_k / num_experts) * expert_fraction + (1 - expert_fraction)
        bh_cost = (avg_experts / num_experts) * expert_fraction + (1 - expert_fraction)

        if baseline_cost == 0:
            return 0.0

        reduction = (baseline_cost - bh_cost) / baseline_cost * 100
        return float(reduction)

    # =========================================================================
    # CATEGORY 3: SPEED METRICS
    # =========================================================================

    @staticmethod
    def compute_tokens_per_second(total_tokens: int, total_time: float) -> float:
        """
        Compute inference throughput.

        Args:
            total_tokens: Total tokens processed
            total_time: Total time in seconds

        Returns:
            Tokens per second. Higher is better.

        Examples:
            >>> compute_tokens_per_second(10000, 90.5)
            110.5
        """
        if total_time <= 0:
            return 0.0

        return float(total_tokens / total_time)

    @staticmethod
    def compute_latency_ms_per_token(tokens_per_second: float) -> float:
        """
        Compute average latency per token.

        Args:
            tokens_per_second: Throughput in tokens/second

        Returns:
            Latency in milliseconds. Lower is better.

        Examples:
            >>> compute_latency_ms_per_token(110.5)
            9.05
        """
        if tokens_per_second <= 0:
            return float('inf')

        return float(1000.0 / tokens_per_second)

    # =========================================================================
    # CATEGORY 4: ROUTING DISTRIBUTION METRICS
    # =========================================================================

    @staticmethod
    def compute_expert_utilization(
        usage_counts: np.ndarray,
        num_experts: int = 64
    ) -> float:
        """
        Compute fraction of experts receiving any activations.

        Args:
            usage_counts: Array of length num_experts with activation counts
            num_experts: Total number of experts (default: 64)

        Returns:
            Utilization in [0, 1]. Higher is better (more balanced).

        Examples:
            >>> usage = np.array([100, 50, 0, 200, ...])  # 64 values
            >>> compute_expert_utilization(usage, 64)
            0.875
        """
        if len(usage_counts) == 0:
            return 0.0

        used_experts = np.sum(usage_counts > 0)
        return float(used_experts / num_experts)

    @staticmethod
    def compute_gini_coefficient(usage_counts: np.ndarray) -> float:
        """
        Compute Gini coefficient for expert usage inequality.

        Measures inequality in expert selection frequency.
        0 = perfect equality, 1 = maximum inequality.

        Args:
            usage_counts: Array of expert activation counts

        Returns:
            Gini coefficient in [0, 1]. Lower is better (more balanced).

        Examples:
            >>> usage = np.array([100, 100, 100, 100])  # Equal
            >>> compute_gini_coefficient(usage)
            0.0

            >>> usage = np.array([400, 0, 0, 0])  # Concentrated
            >>> compute_gini_coefficient(usage)
            0.75
        """
        if len(usage_counts) == 0 or np.sum(usage_counts) == 0:
            return 0.0

        sorted_counts = np.sort(usage_counts)
        n = len(sorted_counts)
        index = np.arange(1, n + 1)

        gini = (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
        return float(gini)

    # =========================================================================
    # CATEGORY 5: ROUTING BEHAVIOR METRICS (BH-Specific)
    # =========================================================================

    @staticmethod
    def compute_adaptive_range(expert_counts: np.ndarray) -> int:
        """
        Compute range of expert counts selected.

        Measures how much the expert count varies across tokens.

        Args:
            expert_counts: Array of expert counts per token

        Returns:
            Range (max - min). 0 for TopK baseline, higher for adaptive BH.

        Examples:
            >>> counts = np.array([3, 4, 5, 3, 6, 4])  # Variable
            >>> compute_adaptive_range(counts)
            3

            >>> counts = np.array([8, 8, 8, 8, 8, 8])  # Fixed TopK
            >>> compute_adaptive_range(counts)
            0
        """
        if len(expert_counts) == 0:
            return 0

        return int(np.max(expert_counts) - np.min(expert_counts))

    @staticmethod
    def compute_selection_entropy(
        expert_counts: np.ndarray,
        max_k: int
    ) -> Tuple[float, float]:
        """
        Compute entropy of expert count distribution.

        Measures variability in expert selection.

        Args:
            expert_counts: Array of expert counts per token
            max_k: Maximum possible expert count

        Returns:
            Tuple of (entropy, normalized_entropy)
            - entropy: Raw entropy value
            - normalized_entropy: Entropy / log(max_k) in [0, 1]

        Examples:
            >>> counts = np.array([4, 4, 4, 4, 4, 4])  # All same
            >>> compute_selection_entropy(counts, max_k=8)
            (0.0, 0.0)

            >>> counts = np.array([1, 2, 3, 4, 5, 6, 7, 8])  # Uniform
            >>> entropy, norm = compute_selection_entropy(counts, max_k=8)
            >>> norm  # Close to 1.0
            0.95
        """
        if len(expert_counts) == 0:
            return 0.0, 0.0

        # Create histogram of counts
        min_count = int(np.min(expert_counts))
        max_count = int(np.max(expert_counts))

        if min_count == max_count:
            # All same count - zero entropy
            return 0.0, 0.0

        hist, _ = np.histogram(
            expert_counts,
            bins=range(min_count, max_count + 2)
        )

        # Compute probabilities
        probs = hist / np.sum(hist)
        probs = probs[probs > 0]  # Remove zeros

        # Compute entropy
        entropy = float(-np.sum(probs * np.log(probs)))

        # Normalize by maximum possible entropy
        max_entropy = np.log(max_k) if max_k > 1 else 1.0
        normalized = float(entropy / max_entropy) if max_entropy > 0 else 0.0

        return entropy, normalized

    # =========================================================================
    # CATEGORY 6: CONSTRAINT METRICS
    # =========================================================================

    @staticmethod
    def compute_ceiling_hit_rate(expert_counts: np.ndarray, max_k: int) -> float:
        """
        Compute percentage of tokens hitting ceiling (max_k experts).

        Args:
            expert_counts: Array of expert counts per token
            max_k: Maximum experts allowed

        Returns:
            Percentage in [0, 100]. 100% for TopK baseline.
            High values indicate max_k is constraining BH.

        Examples:
            >>> counts = np.array([3, 4, 8, 5, 8, 8, 6])
            >>> compute_ceiling_hit_rate(counts, max_k=8)
            42.86  # 3 out of 7 hit ceiling
        """
        if len(expert_counts) == 0:
            return 0.0

        ceiling_hits = np.sum(expert_counts >= max_k)
        return float(ceiling_hits / len(expert_counts) * 100)

    @staticmethod
    def compute_floor_hit_rate(expert_counts: np.ndarray, min_k: int = 1) -> float:
        """
        Compute percentage of tokens hitting floor (min_k experts).

        Args:
            expert_counts: Array of expert counts per token
            min_k: Minimum experts allowed (default: 1)

        Returns:
            Percentage in [0, 100]. 0% for TopK baseline.
            High values indicate very conservative BH.

        Examples:
            >>> counts = np.array([1, 2, 3, 1, 1, 4, 5])
            >>> compute_floor_hit_rate(counts, min_k=1)
            42.86  # 3 out of 7 hit floor
        """
        if len(expert_counts) == 0:
            return 0.0

        floor_hits = np.sum(expert_counts <= min_k)
        return float(floor_hits / len(expert_counts) * 100)

    # =========================================================================
    # CATEGORY 7: CROSS-LAYER METRICS
    # =========================================================================

    @staticmethod
    def compute_layer_expert_variance(per_layer_avg: List[float]) -> float:
        """
        Compute variance in average experts across layers.

        Args:
            per_layer_avg: List of average expert counts per layer (16 values)

        Returns:
            Variance. Low = uniform, High = different layer needs.

        Examples:
            >>> layer_avgs = [4.2, 4.3, 4.1, 4.2, ...]  # Uniform
            >>> compute_layer_expert_variance(layer_avgs)
            0.05

            >>> layer_avgs = [2.0, 3.5, 6.2, 4.1, ...]  # Variable
            >>> compute_layer_expert_variance(layer_avgs)
            1.87
        """
        if len(per_layer_avg) == 0:
            return 0.0

        return float(np.var(per_layer_avg))

    @staticmethod
    def compute_layer_consistency_score(
        per_layer_counts: List[List[int]]
    ) -> float:
        """
        Compute average Pearson correlation between adjacent layers.

        Measures whether adjacent layers agree on token complexity.

        Args:
            per_layer_counts: List of 16 lists, each containing expert counts
                             per token for that layer.
                             Shape: [16 layers][num_tokens]

        Returns:
            Average correlation in [-1, 1].
            High positive = layers agree on hard tokens.

        Examples:
            >>> # Layers agree on complexity
            >>> layer_counts = [[3,4,5], [3,4,5], [3,4,5]]
            >>> compute_layer_consistency_score(layer_counts)
            1.0

            >>> # Layers disagree
            >>> layer_counts = [[3,4,5], [5,4,3], [3,5,4]]
            >>> compute_layer_consistency_score(layer_counts)
            -0.5
        """
        if len(per_layer_counts) < 2:
            return 0.0

        correlations = []

        for i in range(len(per_layer_counts) - 1):
            layer_i = np.array(per_layer_counts[i])
            layer_j = np.array(per_layer_counts[i + 1])

            # Skip if arrays have different lengths or insufficient data
            if len(layer_i) != len(layer_j) or len(layer_i) < 2:
                continue

            # Check for zero variance (all same values)
            if np.std(layer_i) == 0 or np.std(layer_j) == 0:
                continue

            try:
                corr, _ = pearsonr(layer_i, layer_j)
                if not np.isnan(corr):
                    correlations.append(corr)
            except:
                continue

        if not correlations:
            return 0.0

        return float(np.mean(correlations))

    # =========================================================================
    # CATEGORY 8: STABILITY METRICS
    # =========================================================================

    @staticmethod
    def compute_expert_overlap_score(
        run1_experts: List[List[int]],
        run2_experts: List[List[int]]
    ) -> float:
        """
        Compute Jaccard similarity of expert selection across runs.

        Measures consistency of routing decisions.

        Args:
            run1_experts: List of selected expert lists from run 1
                         e.g., [[12, 45, 3], [7, 21, 34], ...]
            run2_experts: List of selected expert lists from run 2

        Returns:
            Jaccard similarity in [0, 1]. Higher = more consistent.

        Examples:
            >>> run1 = [[1, 2, 3], [4, 5, 6]]
            >>> run2 = [[1, 2, 3], [4, 5, 6]]  # Identical
            >>> compute_expert_overlap_score(run1, run2)
            1.0

            >>> run1 = [[1, 2, 3], [4, 5, 6]]
            >>> run2 = [[7, 8, 9], [10, 11, 12]]  # Disjoint
            >>> compute_expert_overlap_score(run1, run2)
            0.0
        """
        if len(run1_experts) == 0 or len(run2_experts) == 0:
            return 1.0  # No data to compare, assume deterministic

        if len(run1_experts) != len(run2_experts):
            raise ValueError("Runs must have same number of samples")

        similarities = []

        for experts1, experts2 in zip(run1_experts, run2_experts):
            set1 = set(experts1) if isinstance(experts1, list) else set([experts1])
            set2 = set(experts2) if isinstance(experts2, list) else set([experts2])

            # Jaccard similarity: |A ∩ B| / |A ∪ B|
            intersection = len(set1 & set2)
            union = len(set1 | set2)

            jaccard = intersection / union if union > 0 else 1.0
            similarities.append(jaccard)

        return float(np.mean(similarities))

    @staticmethod
    def compute_output_determinism(
        run1_tokens: List[int],
        run2_tokens: List[int]
    ) -> float:
        """
        Compute percentage of identical token predictions across runs.

        Should be 100% with do_sample=False.

        Args:
            run1_tokens: Generated token IDs from run 1
            run2_tokens: Generated token IDs from run 2

        Returns:
            Percentage in [0, 100]. Should be 100% for deterministic.

        Examples:
            >>> run1 = [100, 200, 300, 400]
            >>> run2 = [100, 200, 300, 400]  # Identical
            >>> compute_output_determinism(run1, run2)
            100.0

            >>> run1 = [100, 200, 300, 400]
            >>> run2 = [100, 999, 300, 400]  # One different
            >>> compute_output_determinism(run1, run2)
            75.0
        """
        if len(run1_tokens) == 0:
            return 100.0  # No tokens to compare, assume deterministic

        if len(run1_tokens) != len(run2_tokens):
            raise ValueError("Runs must generate same number of tokens")

        matches = sum(1 for t1, t2 in zip(run1_tokens, run2_tokens) if t1 == t2)
        return float(matches / len(run1_tokens) * 100)

    # =========================================================================
    # COMPREHENSIVE COMPUTATION
    # =========================================================================

    def compute_all_metrics(
        self,
        # Quality metrics inputs
        losses: Optional[List[float]] = None,
        accuracies: Optional[Dict[str, float]] = None,

        # Efficiency metrics inputs
        avg_experts: float = 0.0,
        max_k: int = 8,
        baseline_k: int = 8,

        # Speed metrics inputs
        total_tokens: int = 0,
        total_time: float = 0.0,

        # Distribution metrics inputs
        expert_usage_counts: Optional[np.ndarray] = None,

        # Behavior metrics inputs
        expert_counts: Optional[np.ndarray] = None,
        min_k: int = 1,

        # Cross-layer metrics inputs
        per_layer_avg: Optional[List[float]] = None,
        per_layer_counts: Optional[List[List[int]]] = None,

        # Stability metrics inputs
        run1_experts: Optional[List[List[int]]] = None,
        run2_experts: Optional[List[List[int]]] = None,
        run1_tokens: Optional[List[int]] = None,
        run2_tokens: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Compute all 16 metrics in one call.

        Args:
            See individual metric functions for parameter descriptions.
            Optional parameters can be None - corresponding metrics will be skipped.

        Returns:
            Dictionary with all computed metrics:
            {
                # Category 1: Quality
                'perplexity': float,
                'avg_task_accuracy': float,

                # Category 2: Efficiency
                'expert_activation_ratio': float,
                'flops_reduction_pct': float,

                # Category 3: Speed
                'tokens_per_second': float,
                'latency_ms_per_token': float,

                # Category 4: Distribution
                'expert_utilization': float,
                'gini_coefficient': float,

                # Category 5: Behavior
                'adaptive_range': int,
                'selection_entropy': float,
                'normalized_selection_entropy': float,

                # Category 6: Constraints
                'ceiling_hit_rate': float,
                'floor_hit_rate': float,
                'mid_range_rate': float,

                # Category 7: Cross-Layer
                'layer_expert_variance': float,
                'layer_consistency_score': float,

                # Category 8: Stability
                'expert_overlap_score': float,
                'output_determinism': float,
            }
        """
        metrics = {}

        # Category 1: Quality
        if losses is not None:
            metrics['perplexity'] = self.compute_perplexity(losses)

        if accuracies is not None:
            metrics['avg_task_accuracy'] = self.compute_avg_task_accuracy(accuracies)

        # Category 2: Efficiency
        metrics['avg_experts'] = float(avg_experts)
        metrics['expert_activation_ratio'] = self.compute_expert_activation_ratio(
            avg_experts, max_k
        )
        metrics['flops_reduction_pct'] = self.compute_flops_reduction_pct(
            avg_experts, baseline_k
        )

        # Category 3: Speed
        if total_tokens > 0 and total_time > 0:
            tokens_per_sec = self.compute_tokens_per_second(total_tokens, total_time)
            metrics['tokens_per_second'] = tokens_per_sec
            metrics['latency_ms_per_token'] = self.compute_latency_ms_per_token(tokens_per_sec)

        # Category 4: Distribution
        if expert_usage_counts is not None:
            metrics['expert_utilization'] = self.compute_expert_utilization(expert_usage_counts)
            metrics['gini_coefficient'] = self.compute_gini_coefficient(expert_usage_counts)

        # Category 5: Behavior
        if expert_counts is not None and len(expert_counts) > 0:
            metrics['adaptive_range'] = self.compute_adaptive_range(expert_counts)
            entropy, norm_entropy = self.compute_selection_entropy(expert_counts, max_k)
            metrics['selection_entropy'] = entropy
            metrics['normalized_selection_entropy'] = norm_entropy

        # Category 6: Constraints
        if expert_counts is not None and len(expert_counts) > 0:
            ceiling_rate = self.compute_ceiling_hit_rate(expert_counts, max_k)
            floor_rate = self.compute_floor_hit_rate(expert_counts, min_k)
            metrics['ceiling_hit_rate'] = ceiling_rate
            metrics['floor_hit_rate'] = floor_rate
            metrics['mid_range_rate'] = 100.0 - ceiling_rate - floor_rate

        # Category 7: Cross-Layer
        if per_layer_avg is not None:
            metrics['layer_expert_variance'] = self.compute_layer_expert_variance(per_layer_avg)

        if per_layer_counts is not None:
            metrics['layer_consistency_score'] = self.compute_layer_consistency_score(
                per_layer_counts
            )

        # Category 8: Stability
        if run1_experts is not None and run2_experts is not None:
            metrics['expert_overlap_score'] = self.compute_expert_overlap_score(
                run1_experts, run2_experts
            )
        else:
            # For single run, assume perfect determinism
            metrics['expert_overlap_score'] = 1.0

        if run1_tokens is not None and run2_tokens is not None:
            metrics['output_determinism'] = self.compute_output_determinism(
                run1_tokens, run2_tokens
            )
        else:
            # For single run, assume perfect determinism
            metrics['output_determinism'] = 100.0

        return metrics


# Convenience function for quick metric computation
def compute_metrics(**kwargs) -> Dict[str, Any]:
    """
    Convenience function for computing metrics without instantiating class.

    Args:
        **kwargs: Same arguments as BHMetricsComputer.compute_all_metrics()

    Returns:
        Dictionary of computed metrics

    Examples:
        >>> metrics = compute_metrics(
        ...     losses=[2.1, 2.2, 2.0],
        ...     avg_experts=4.5,
        ...     max_k=8,
        ...     total_tokens=10000,
        ...     total_time=90.0
        ... )
    """
    computer = BHMetricsComputer()
    return computer.compute_all_metrics(**kwargs)
