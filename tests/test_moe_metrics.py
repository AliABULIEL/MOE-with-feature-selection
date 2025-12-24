"""
Unit Tests for moe_metrics.py
==============================

Tests for MetricsComputer and ComprehensiveMetrics classes.

Run with:
    pytest test_moe_metrics.py -v
"""

import pytest
import torch
import numpy as np

from moe_metrics import MetricsComputer, ComprehensiveMetrics


# =========================================================================
# Test ComprehensiveMetrics
# =========================================================================

class TestComprehensiveMetrics:
    """Tests for ComprehensiveMetrics dataclass."""

    def test_initialization(self):
        """Test default initialization."""
        metrics = ComprehensiveMetrics()

        # Quality metrics
        assert metrics.perplexity == float('inf')
        assert metrics.token_accuracy == 0.0
        assert metrics.loss == float('inf')

        # Task-specific metrics
        assert metrics.lambada_accuracy == 0.0
        assert metrics.hellaswag_accuracy == 0.0
        assert metrics.f1_score == 0.0
        assert metrics.exact_match == 0.0

        # Routing metrics
        assert metrics.avg_experts == 0.0
        assert metrics.std_experts == 0.0
        assert metrics.min_experts == 0
        assert metrics.max_experts == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = ComprehensiveMetrics()
        metrics.perplexity = 25.5
        metrics.avg_experts = 6.5

        d = metrics.to_dict()

        assert isinstance(d, dict)
        assert d['perplexity'] == 25.5
        assert d['avg_experts'] == 6.5

    def test_is_valid(self):
        """Test validity checking."""
        metrics = ComprehensiveMetrics()

        # Invalid: inf perplexity, no samples
        assert not metrics.is_valid()

        # Set valid values
        metrics.perplexity = 25.5
        metrics.loss = 3.2
        metrics.num_samples = 10

        assert metrics.is_valid()

    def test_compute_reduction(self):
        """Test reduction computation."""
        metrics = ComprehensiveMetrics()
        metrics.avg_experts = 6.0

        metrics.compute_reduction(baseline_experts=8.0)

        assert metrics.reduction_vs_baseline == 25.0  # (8-6)/8 * 100


# =========================================================================
# Test MetricsComputer - Core Quality Metrics
# =========================================================================

class TestMetricsComputerQuality:
    """Tests for core quality metrics."""

    def test_compute_perplexity_normal(self):
        """Test perplexity computation with normal values."""
        perplexity = MetricsComputer.compute_perplexity(
            total_loss=10.0,
            total_tokens=100
        )

        # perplexity = exp(10/100) = exp(0.1) ≈ 1.105
        assert 1.0 < perplexity < 1.2

    def test_compute_perplexity_zero_tokens(self):
        """Test perplexity with zero tokens."""
        perplexity = MetricsComputer.compute_perplexity(
            total_loss=10.0,
            total_tokens=0
        )

        assert perplexity == float('inf')

    def test_compute_perplexity_overflow(self):
        """Test perplexity with very large loss."""
        perplexity = MetricsComputer.compute_perplexity(
            total_loss=10000.0,
            total_tokens=100
        )

        # avg_loss = 100, should prevent overflow
        assert perplexity == float('inf')

    def test_compute_token_accuracy(self):
        """Test token accuracy computation."""
        # Create mock logits and labels
        batch_size, seq_len, vocab_size = 2, 5, 100

        # Perfect predictions
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits = torch.zeros(batch_size, seq_len, vocab_size)

        # Set logits to make labels the argmax
        for b in range(batch_size):
            for s in range(seq_len):
                logits[b, s, labels[b, s]] = 10.0

        correct, total = MetricsComputer.compute_token_accuracy(logits, labels)

        # Should have seq_len - 1 correct per batch (due to shifting)
        # Total = 2 * (5-1) = 8
        assert total == 8
        assert correct == 8

    def test_compute_token_accuracy_with_padding(self):
        """Test token accuracy with padding."""
        batch_size, seq_len, vocab_size = 2, 5, 100

        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels[:, -1] = -100  # Mark last token as padding

        logits = torch.zeros(batch_size, seq_len, vocab_size)

        # Set perfect predictions
        for b in range(batch_size):
            for s in range(seq_len):
                if labels[b, s] != -100:
                    logits[b, s, labels[b, s]] = 10.0

        correct, total = MetricsComputer.compute_token_accuracy(logits, labels)

        # Padding tokens should be ignored
        assert total < 8  # Less than without padding


# =========================================================================
# Test MetricsComputer - Text Normalization and F1
# =========================================================================

class TestMetricsComputerText:
    """Tests for text normalization and F1/EM metrics."""

    def test_normalize_text(self):
        """Test text normalization."""
        text = "Hello, World! This is a TEST."
        normalized = MetricsComputer.normalize_text(text)

        assert normalized == "hello world this is a test"

    def test_normalize_text_with_punctuation(self):
        """Test normalization removes punctuation."""
        text = "Hello... World??? Test!!!"
        normalized = MetricsComputer.normalize_text(text)

        assert normalized == "hello world test"

    def test_compute_f1_perfect_match(self):
        """Test F1 with perfect match."""
        prediction = "the quick brown fox"
        reference = "the quick brown fox"

        f1 = MetricsComputer.compute_f1(prediction, reference)

        assert f1 == 1.0

    def test_compute_f1_partial_match(self):
        """Test F1 with partial match."""
        prediction = "the quick brown fox"
        reference = "the slow brown fox"

        f1 = MetricsComputer.compute_f1(prediction, reference)

        # Common: the, brown, fox (3 out of 4)
        # precision = 3/4, recall = 3/4, f1 = 0.75
        assert 0.7 < f1 < 0.8

    def test_compute_f1_no_match(self):
        """Test F1 with no match."""
        prediction = "hello world"
        reference = "foo bar"

        f1 = MetricsComputer.compute_f1(prediction, reference)

        assert f1 == 0.0

    def test_compute_f1_empty(self):
        """Test F1 with empty strings."""
        f1 = MetricsComputer.compute_f1("", "test")
        assert f1 == 0.0

        f1 = MetricsComputer.compute_f1("test", "")
        assert f1 == 0.0

    def test_compute_exact_match_true(self):
        """Test exact match when strings match."""
        prediction = "The Quick Brown Fox"
        reference = "the quick brown fox"

        em = MetricsComputer.compute_exact_match(prediction, reference)

        assert em == 1.0

    def test_compute_exact_match_false(self):
        """Test exact match when strings differ."""
        prediction = "the quick brown fox"
        reference = "the slow brown fox"

        em = MetricsComputer.compute_exact_match(prediction, reference)

        assert em == 0.0


# =========================================================================
# Test MetricsComputer - Routing Metrics
# =========================================================================

class TestMetricsComputerRouting:
    """Tests for routing statistics computation."""

    def test_compute_routing_stats_normal(self):
        """Test routing stats with normal expert counts."""
        expert_counts = np.array([3, 4, 5, 6, 7, 8, 7, 6, 5, 4])

        stats = MetricsComputer.compute_routing_stats(
            expert_counts=expert_counts,
            max_k=8,
            min_k=1
        )

        assert stats['avg_experts'] == 5.5
        assert stats['min_experts'] == 3
        assert stats['max_experts'] == 8
        assert stats['median_experts'] == 5.5
        assert 0 <= stats['ceiling_hit_rate'] <= 100
        assert 0 <= stats['floor_hit_rate'] <= 100
        assert 0 <= stats['mid_range_rate'] <= 100

    def test_compute_routing_stats_all_at_ceiling(self):
        """Test routing stats when all at ceiling."""
        expert_counts = np.array([8, 8, 8, 8, 8])

        stats = MetricsComputer.compute_routing_stats(
            expert_counts=expert_counts,
            max_k=8,
            min_k=1
        )

        assert stats['avg_experts'] == 8
        assert stats['ceiling_hit_rate'] == 100.0
        assert stats['floor_hit_rate'] == 0.0

    def test_compute_routing_stats_all_at_floor(self):
        """Test routing stats when all at floor."""
        expert_counts = np.array([1, 1, 1, 1, 1])

        stats = MetricsComputer.compute_routing_stats(
            expert_counts=expert_counts,
            max_k=8,
            min_k=1
        )

        assert stats['avg_experts'] == 1
        assert stats['ceiling_hit_rate'] == 0.0
        assert stats['floor_hit_rate'] == 100.0

    def test_compute_routing_stats_empty(self):
        """Test routing stats with empty array."""
        stats = MetricsComputer.compute_routing_stats(
            expert_counts=np.array([]),
            max_k=8,
            min_k=1
        )

        assert stats == {}

    def test_compute_routing_stats_reduction(self):
        """Test reduction vs baseline computation."""
        expert_counts = np.array([6, 6, 6, 6, 6])

        stats = MetricsComputer.compute_routing_stats(
            expert_counts=expert_counts,
            max_k=8,
            min_k=1
        )

        # (8 - 6) / 8 * 100 = 25%
        assert stats['reduction_vs_baseline'] == 25.0


# =========================================================================
# Edge Cases and Error Handling
# =========================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_perplexity_nan_loss(self):
        """Test perplexity with NaN loss."""
        perplexity = MetricsComputer.compute_perplexity(
            total_loss=float('nan'),
            total_tokens=100
        )

        assert perplexity == float('inf')

    def test_perplexity_inf_loss(self):
        """Test perplexity with inf loss."""
        perplexity = MetricsComputer.compute_perplexity(
            total_loss=float('inf'),
            total_tokens=100
        )

        assert perplexity == float('inf')

    def test_routing_stats_with_outliers(self):
        """Test routing stats with outlier values."""
        expert_counts = np.array([1, 2, 3, 4, 5, 100])  # 100 is outlier

        stats = MetricsComputer.compute_routing_stats(
            expert_counts=expert_counts,
            max_k=8,
            min_k=1
        )

        # Should handle gracefully
        assert 'avg_experts' in stats
        assert stats['max_experts'] == 100
        assert stats['adaptive_range'] == 99  # 100 - 1

    def test_f1_with_special_characters(self):
        """Test F1 with special characters."""
        prediction = "hello@world.com"
        reference = "hello world com"

        f1 = MetricsComputer.compute_f1(prediction, reference)

        # Should normalize and match
        assert f1 > 0.9


# =========================================================================
# Integration Tests
# =========================================================================

class TestIntegration:
    """Integration tests combining multiple metrics."""

    def test_full_metrics_computation(self):
        """Test computing all metrics together."""
        metrics = ComprehensiveMetrics()

        # Quality metrics
        metrics.perplexity = MetricsComputer.compute_perplexity(10.0, 100)
        metrics.loss = 10.0 / 100

        # Routing metrics
        expert_counts = np.array([3, 4, 5, 6, 7, 8] * 10)
        routing_stats = MetricsComputer.compute_routing_stats(expert_counts, max_k=8)

        metrics.avg_experts = routing_stats['avg_experts']
        metrics.std_experts = routing_stats['std_experts']
        metrics.ceiling_hit_rate = routing_stats['ceiling_hit_rate']

        # Text metrics
        metrics.f1_score = MetricsComputer.compute_f1("the quick fox", "the quick brown fox")
        metrics.exact_match = MetricsComputer.compute_exact_match("hello", "world")

        # Verify all set
        assert metrics.perplexity > 1.0
        assert metrics.avg_experts > 0
        assert 0 <= metrics.f1_score <= 1.0
        assert metrics.exact_match in [0.0, 1.0]

        # Verify validity
        metrics.num_samples = 10
        assert metrics.is_valid()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
