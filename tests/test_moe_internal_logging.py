"""
Unit Tests for moe_internal_logging.py
=======================================

Tests for RouterLogger and InternalRoutingLogger classes.

Run with:
    pytest test_moe_internal_logging.py -v
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import tempfile
import shutil

from moe_internal_logging import RouterLogger, InternalRoutingLogger


# =========================================================================
# Mock Model for Testing
# =========================================================================

class MockGate(nn.Module):
    """Mock router gate module."""
    def __init__(self, num_experts=64):
        super().__init__()
        self.num_experts = num_experts
        self.linear = nn.Linear(512, num_experts)

    def forward(self, x):
        return self.linear(x)


class MockMLP(nn.Module):
    """Mock MLP with gate."""
    def __init__(self):
        super().__init__()
        self.gate = MockGate()


class MockLayer(nn.Module):
    """Mock transformer layer."""
    def __init__(self):
        super().__init__()
        self.mlp = MockMLP()


class MockLayers(nn.Module):
    """Mock layers container."""
    def __init__(self, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([MockLayer() for _ in range(num_layers)])


class MockModel(nn.Module):
    """Mock model matching OLMoE structure."""
    def __init__(self, num_layers=3):
        super().__init__()
        self.model = MockLayers(num_layers)
        self.config = type('obj', (object,), {'num_experts': 64})()

    def __call__(self, input_ids):
        # Simple forward pass for testing
        batch_size, seq_len = input_ids.shape
        hidden_states = torch.randn(batch_size, seq_len, 512)

        for layer in self.model.layers:
            router_logits = layer.mlp.gate(hidden_states.view(-1, 512))
            # Just return for testing hooks

        return type('obj', (object,), {
            'loss': torch.tensor(2.5),
            'logits': torch.randn(batch_size, seq_len, 50000)
        })()


# =========================================================================
# Test RouterLogger
# =========================================================================

class TestRouterLogger:
    """Tests for RouterLogger class."""

    @pytest.fixture
    def model(self):
        """Create a mock model."""
        return MockModel(num_layers=3)

    @pytest.fixture
    def logger(self, model):
        """Create a RouterLogger instance."""
        return RouterLogger(model)

    def test_initialization(self, logger, model):
        """Test RouterLogger initialization."""
        assert logger.model is model
        assert logger.num_experts == 64
        assert logger.top_k == 8
        assert logger.hooks == []
        assert logger.routing_data == []

    def test_register_hooks(self, logger):
        """Test hook registration."""
        logger.register_hooks(top_k=8)

        # Should have hooks for all 3 layers
        assert len(logger.hooks) == 3
        assert logger.top_k == 8

    def test_hook_capture(self, logger, model):
        """Test that hooks capture routing data."""
        logger.register_hooks(top_k=8)

        # Run forward pass
        input_ids = torch.randint(0, 1000, (2, 10))
        _ = model(input_ids)

        # Should have captured data from 3 layers
        routing_data = logger.get_routing_data()
        assert len(routing_data) == 3

        # Check data structure
        for data in routing_data:
            assert 'layer' in data
            assert 'router_logits' in data
            assert 'expert_indices' in data
            assert 'expert_weights' in data
            assert 'probs' in data
            assert 'stats' in data

    def test_summary_stats(self, logger, model):
        """Test summary statistics computation."""
        logger.register_hooks(top_k=8)

        # Run forward pass
        input_ids = torch.randint(0, 1000, (2, 10))
        _ = model(input_ids)

        # Get summary stats
        stats = logger.get_summary_stats()

        assert 'avg_max_prob' in stats
        assert 'avg_entropy' in stats
        assert 'avg_concentration' in stats
        assert 'unique_experts_used' in stats
        assert 'expert_utilization' in stats
        assert 'num_layers_captured' in stats

        assert stats['num_layers_captured'] == 3
        assert 0 <= stats['expert_utilization'] <= 1

    def test_clear_data(self, logger, model):
        """Test data clearing."""
        logger.register_hooks(top_k=8)

        # Run forward pass
        input_ids = torch.randint(0, 1000, (2, 10))
        _ = model(input_ids)

        # Clear data
        logger.clear_data()

        assert len(logger.routing_data) == 0

    def test_remove_hooks(self, logger):
        """Test hook removal."""
        logger.register_hooks(top_k=8)
        assert len(logger.hooks) == 3

        logger.remove_hooks()
        assert len(logger.hooks) == 0

    def test_per_layer_stats(self, logger, model):
        """Test per-layer statistics."""
        logger.register_hooks(top_k=8)

        # Run forward pass
        input_ids = torch.randint(0, 1000, (2, 10))
        _ = model(input_ids)

        # Get per-layer stats
        per_layer = logger.get_per_layer_stats()

        assert len(per_layer) == 3
        for layer_idx, stats in per_layer.items():
            assert 'entropy' in stats
            assert 'max_prob' in stats
            assert 'concentration' in stats


# =========================================================================
# Test InternalRoutingLogger
# =========================================================================

class TestInternalRoutingLogger:
    """Tests for InternalRoutingLogger class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for logs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def logger(self, temp_dir):
        """Create an InternalRoutingLogger instance."""
        return InternalRoutingLogger(
            output_dir=temp_dir,
            experiment_name='test_experiment',
            routing_method='bh'
        )

    def test_initialization(self, logger, temp_dir):
        """Test InternalRoutingLogger initialization."""
        assert logger.output_dir == Path(temp_dir)
        assert logger.experiment_name == 'test_experiment'
        assert logger.routing_method == 'bh'
        assert logger.logs_dir.exists()

    def test_log_sample(self, logger):
        """Test logging a sample."""
        # Create mock routing data
        routing_data = [
            {
                'layer': 0,
                'router_logits': torch.randn(20, 64),
                'expert_indices': torch.randint(0, 64, (20, 8)),
                'expert_weights': torch.rand(20, 8),
                'probs': torch.rand(20, 64),
                'stats': {
                    'max_prob': 0.5,
                    'entropy': 2.3,
                    'concentration': 0.4,
                    'num_tokens': 20,
                    'unique_experts_this_layer': 15
                }
            }
        ]

        logger.log_sample(
            sample_id=0,
            routing_data=routing_data,
            loss=2.5,
            num_tokens=20,
            expert_counts=[3, 4, 5, 6, 7, 8] * 3 + [3, 4]
        )

        assert len(logger.all_samples) == 1
        assert logger.total_tokens == 20
        assert logger.total_loss == 2.5 * 20

    def test_aggregate_stats(self, logger):
        """Test aggregate statistics computation."""
        # Log multiple samples
        for i in range(5):
            routing_data = [
                {
                    'layer': j,
                    'router_logits': torch.randn(20, 64),
                    'expert_indices': torch.randint(0, 64, (20, 8)),
                    'expert_weights': torch.rand(20, 8),
                    'probs': torch.rand(20, 64),
                    'stats': {
                        'max_prob': 0.5,
                        'entropy': 2.3,
                        'concentration': 0.4,
                        'num_tokens': 20,
                        'unique_experts_this_layer': 15
                    }
                }
                for j in range(3)
            ]

            logger.log_sample(
                sample_id=i,
                routing_data=routing_data,
                loss=2.5,
                num_tokens=20,
                expert_counts=[3, 4, 5, 6, 7, 8] * 3 + [3, 4]
            )

        stats = logger.get_aggregate_stats()

        assert stats['routing_method'] == 'bh'
        assert stats['num_samples'] == 5
        assert stats['total_tokens'] == 100
        assert 'avg_loss' in stats
        assert 'perplexity' in stats
        assert 'layer_entropies' in stats
        assert 'layer_concentrations' in stats
        assert 'unique_experts' in stats

    def test_save_logs(self, logger, temp_dir):
        """Test saving logs to JSON."""
        # Log a sample
        routing_data = [
            {
                'layer': 0,
                'router_logits': torch.randn(20, 64),
                'expert_indices': torch.randint(0, 64, (20, 8)),
                'expert_weights': torch.rand(20, 8),
                'probs': torch.rand(20, 64),
                'stats': {
                    'max_prob': 0.5,
                    'entropy': 2.3,
                    'concentration': 0.4,
                    'num_tokens': 20,
                    'unique_experts_this_layer': 15
                }
            }
        ]

        logger.log_sample(
            sample_id=0,
            routing_data=routing_data,
            loss=2.5,
            num_tokens=20,
            expert_counts=[3, 4, 5, 6, 7, 8]
        )

        # Save logs
        log_file = logger.save_logs()

        assert Path(log_file).exists()

        # Verify JSON structure
        with open(log_file, 'r') as f:
            data = json.load(f)

        assert data['experiment'] == 'test_experiment'
        assert data['routing_method'] == 'bh'
        assert 'aggregate_stats' in data
        assert 'per_layer_summary' in data
        assert 'expert_usage' in data

    def test_clear(self, logger):
        """Test clearing accumulated data."""
        # Log a sample
        routing_data = [{
            'layer': 0,
            'stats': {'entropy': 2.3}
        }]

        logger.log_sample(
            sample_id=0,
            routing_data=routing_data,
            loss=2.5,
            num_tokens=20
        )

        assert len(logger.all_samples) == 1

        # Clear
        logger.clear()

        assert len(logger.all_samples) == 0
        assert logger.total_tokens == 0
        assert logger.total_loss == 0.0


# =========================================================================
# Integration Tests
# =========================================================================

class TestIntegration:
    """Integration tests using both loggers together."""

    @pytest.fixture
    def model(self):
        """Create a mock model."""
        return MockModel(num_layers=3)

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for logs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_full_logging_pipeline(self, model, temp_dir):
        """Test complete logging pipeline."""
        # Create loggers
        router_logger = RouterLogger(model)
        internal_logger = InternalRoutingLogger(
            output_dir=temp_dir,
            experiment_name='integration_test',
            routing_method='bh'
        )

        # Register hooks
        router_logger.register_hooks(top_k=8)

        # Run multiple forward passes
        for i in range(3):
            router_logger.clear_data()

            input_ids = torch.randint(0, 1000, (2, 10))
            outputs = model(input_ids)

            internal_logger.log_sample(
                sample_id=i,
                routing_data=router_logger.get_routing_data(),
                loss=outputs.loss.item(),
                num_tokens=20,
                expert_counts=[3, 4, 5, 6] * 5
            )

        # Remove hooks
        router_logger.remove_hooks()

        # Get aggregate stats
        stats = internal_logger.get_aggregate_stats()

        assert stats['num_samples'] == 3
        assert stats['total_tokens'] == 60
        assert 'layer_entropies' in stats
        assert len(stats['layer_entropies']) == 3  # 3 layers

        # Save logs
        log_file = internal_logger.save_logs()
        assert Path(log_file).exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
