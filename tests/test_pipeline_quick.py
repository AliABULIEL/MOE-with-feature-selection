"""
Quick Pipeline Tests for DeepSeek and Qwen MoE Analysis Pipelines
==================================================================

Fast test suite that validates ALL pipeline components work correctly
WITHOUT loading actual large models (DeepSeek-V2-Lite, Qwen3-30B-A3B).

Tests should complete in under 60 seconds total.

Run with:
    pytest tests/test_pipeline_quick.py -v

Coverage:
    1. Import Tests - All modules import without errors
    2. RouterLogger Tests - Hook registration, data clearing, hook removal
    3. Utility Function Tests - Directory setup, dtype conversion, JSON parsing, trimming
    4. KDE Training Tests - KDE model training, pickle save/load
    5. End-to-End Mock Tests - Full pipeline with mocked model and tiny fake dataset
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import pickle
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from collections import defaultdict


# ==============================================================================
# SECTION 1: IMPORT TESTS
# ==============================================================================

class TestImports:
    """Test that all modules import without errors."""

    def test_import_deepseek_pipeline(self):
        """Test importing DeepSeek pipeline module."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "pipelines"))

        # Import utility functions from deepseek pipeline
        from run_deepseek_pipeline import (
            setup_directories,
            get_torch_dtype,
            extract_router_logits,
            trim_top_and_bottom_experts,
            get_ordinal,
            CONFIG
        )

        assert callable(setup_directories)
        assert callable(get_torch_dtype)
        assert callable(extract_router_logits)
        assert callable(trim_top_and_bottom_experts)
        assert isinstance(CONFIG, dict)

    def test_import_qwen_pipeline(self):
        """Test importing Qwen pipeline module."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "pipelines"))

        from run_qwen_pipeline import (
            setup_directories,
            get_torch_dtype,
            extract_router_logits,
            trim_top_and_bottom_experts,
            CONFIG
        )

        assert callable(setup_directories)
        assert callable(get_torch_dtype)
        assert callable(extract_router_logits)
        assert callable(trim_top_and_bottom_experts)
        assert isinstance(CONFIG, dict)

    def test_import_deepseek_logging(self):
        """Test importing DeepSeek logging module."""
        from moe_internal_logging_deepseek import RouterLogger, InternalRoutingLogger

        assert RouterLogger is not None
        assert InternalRoutingLogger is not None

    def test_import_qwen_logging(self):
        """Test importing Qwen logging module."""
        from moe_internal_logging_qwen import RouterLogger, InternalRoutingLogger

        assert RouterLogger is not None
        assert InternalRoutingLogger is not None

    def test_import_dependencies(self):
        """Test that all required dependencies are available."""
        import numpy as np
        import torch
        import matplotlib
        import seaborn
        from scipy.stats import gaussian_kde
        from sklearn.neighbors import KernelDensity
        from tqdm import tqdm

        assert np.__version__ is not None
        assert torch.__version__ is not None


# ==============================================================================
# SECTION 2: MOCK MODEL CLASSES
# ==============================================================================

class MockGateDeepSeek(nn.Module):
    """Mock router gate for DeepSeek (has weight attribute)."""
    def __init__(self, hidden_dim=512, num_experts=64):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_dim))

    def forward(self, x):
        return F.linear(x, self.weight)


class MockMLPDeepSeek(nn.Module):
    """Mock MLP with gate for DeepSeek."""
    def __init__(self, hidden_dim=512, num_experts=64):
        super().__init__()
        self.gate = MockGateDeepSeek(hidden_dim, num_experts)


class MockLayerDeepSeek(nn.Module):
    """Mock transformer layer for DeepSeek."""
    def __init__(self, hidden_dim=512, num_experts=64):
        super().__init__()
        self.mlp = MockMLPDeepSeek(hidden_dim, num_experts)


class MockModelDeepSeek(nn.Module):
    """Mock DeepSeek model."""
    def __init__(self, num_layers=3, hidden_dim=512, num_experts=64, top_k=6):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            MockLayerDeepSeek(hidden_dim, num_experts)
            for _ in range(num_layers)
        ])
        self.config = MagicMock()
        self.config.n_routed_experts = num_experts
        self.config.num_experts_per_tok = top_k
        self.config.norm_topk_prob = False
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        hidden_states = torch.randn(batch_size * seq_len, self.hidden_dim)

        for layer in self.model.layers:
            _ = layer.mlp.gate(hidden_states)

        outputs = MagicMock()
        outputs.loss = torch.tensor(2.5)
        outputs.logits = torch.randn(batch_size, seq_len, 50000)
        return outputs

    def eval(self):
        return self


class MockGateQwen(nn.Module):
    """Mock gate for Qwen (different structure - gate is inside mlp)."""
    def __init__(self, hidden_dim=512, num_experts=128):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts)

    def forward(self, x):
        if x.dim() == 3:
            batch_size, seq_len, hidden_dim = x.shape
            x = x.view(-1, hidden_dim)
        return self.gate(x)


class MockMLPQwen(nn.Module):
    """Mock MLP for Qwen - has internal gate attribute."""
    def __init__(self, hidden_dim=512, num_experts=128):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts)

    def forward(self, x):
        if x.dim() == 3:
            batch_size, seq_len, hidden_dim = x.shape
            x = x.view(-1, hidden_dim)
        return self.gate(x)


class MockLayerQwen(nn.Module):
    """Mock transformer layer for Qwen."""
    def __init__(self, hidden_dim=512, num_experts=128):
        super().__init__()
        self.mlp = MockMLPQwen(hidden_dim, num_experts)


class MockModelQwen(nn.Module):
    """Mock Qwen model."""
    def __init__(self, num_layers=3, hidden_dim=512, num_experts=128, top_k=8):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            MockLayerQwen(hidden_dim, num_experts)
            for _ in range(num_layers)
        ])
        self.config = MagicMock()
        self.config.num_experts = num_experts
        self.config.num_experts_per_tok = top_k
        self.config.norm_topk_prob = False
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        hidden_states = torch.randn(batch_size, seq_len, self.hidden_dim)

        for layer in self.model.layers:
            _ = layer.mlp(hidden_states)

        outputs = MagicMock()
        outputs.loss = torch.tensor(2.3)
        outputs.logits = torch.randn(batch_size, seq_len, 50000)
        return outputs

    def eval(self):
        return self


# ==============================================================================
# SECTION 3: ROUTER LOGGER TESTS
# ==============================================================================

class TestRouterLoggerDeepSeek:
    """Test RouterLogger for DeepSeek model."""

    @pytest.fixture
    def model(self):
        return MockModelDeepSeek(num_layers=3, num_experts=64, top_k=6)

    @pytest.fixture
    def logger(self, model):
        from moe_internal_logging_deepseek import RouterLogger
        return RouterLogger(model)

    def test_initialization(self, logger, model):
        """Test RouterLogger initialization."""
        assert logger.model is model
        assert logger.num_experts == 64
        assert logger.top_k == 6
        assert logger.hooks == []
        assert logger.routing_data == []

    def test_register_hooks(self, logger):
        """Test hook registration."""
        logger.register_hooks(top_k=6)
        assert len(logger.hooks) == 3  # 3 layers
        assert logger.top_k == 6

    def test_clear_data(self, logger, model):
        """Test data clearing works."""
        logger.register_hooks(top_k=6)

        input_ids = torch.randint(0, 1000, (2, 10))
        _ = model(input_ids)

        assert len(logger.routing_data) > 0

        logger.clear_data()
        assert len(logger.routing_data) == 0

    def test_remove_hooks(self, logger):
        """Test hook removal."""
        logger.register_hooks(top_k=6)
        assert len(logger.hooks) == 3

        logger.remove_hooks()
        # After removal, hooks list should still contain handle objects but they are removed
        # The remove_hooks function doesn't clear the list, just removes the hooks

    def test_get_routing_data_structure(self, logger, model):
        """Test routing data structure is correct."""
        logger.register_hooks(top_k=6)

        input_ids = torch.randint(0, 1000, (2, 10))
        _ = model(input_ids)

        routing_data = logger.get_routing_data()
        assert len(routing_data) == 3  # 3 layers

        for data in routing_data:
            assert 'layer' in data
            assert 'router_logits' in data
            assert 'expert_indices' in data
            assert 'expert_weights' in data
            assert 'probs' in data
            assert 'stats' in data

            stats = data['stats']
            assert 'max_prob' in stats
            assert 'entropy' in stats
            assert 'concentration' in stats
            assert 'num_tokens' in stats


class TestRouterLoggerQwen:
    """Test RouterLogger for Qwen model."""

    @pytest.fixture
    def model(self):
        return MockModelQwen(num_layers=3, num_experts=128, top_k=8)

    @pytest.fixture
    def logger(self, model):
        from moe_internal_logging_qwen import RouterLogger
        return RouterLogger(model)

    def test_initialization(self, logger, model):
        """Test RouterLogger initialization."""
        assert logger.model is model
        assert logger.num_experts == 128
        assert logger.top_k == 8

    def test_register_hooks(self, logger):
        """Test hook registration."""
        logger.register_hooks(top_k=8)
        assert len(logger.hooks) == 3  # 3 layers

    def test_clear_data(self, logger):
        """Test data clearing."""
        logger.routing_data = [{'test': 'data'}]
        logger.clear_data()
        assert len(logger.routing_data) == 0


# ==============================================================================
# SECTION 4: UTILITY FUNCTION TESTS
# ==============================================================================

class TestUtilityFunctions:
    """Test utility functions from both pipelines."""

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_setup_directories_deepseek(self, temp_dir):
        """Test setup_directories creates correct structure."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "pipelines"))
        from run_deepseek_pipeline import setup_directories

        config = {"output_dir": temp_dir}
        dirs = setup_directories(config)

        assert dirs["base"].exists()
        assert dirs["logs"].exists()
        assert dirs["kde_models"].exists()
        assert dirs["plots_basic"].exists()
        assert dirs["plots_per_expert"].exists()
        assert dirs["plots_kde"].exists()
        assert dirs["plots_pvalue"].exists()

    def test_setup_directories_qwen(self, temp_dir):
        """Test setup_directories for Qwen pipeline."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "pipelines"))
        from run_qwen_pipeline import setup_directories

        config = {"output_dir": temp_dir}
        dirs = setup_directories(config)

        assert all(d.exists() for d in dirs.values())

    def test_get_torch_dtype(self):
        """Test get_torch_dtype returns correct types."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "pipelines"))
        from run_deepseek_pipeline import get_torch_dtype

        assert get_torch_dtype("float32") == torch.float32
        assert get_torch_dtype("float16") == torch.float16
        assert get_torch_dtype("bfloat16") == torch.bfloat16
        assert get_torch_dtype("unknown") == torch.bfloat16  # default

    def test_get_ordinal(self):
        """Test get_ordinal returns correct ordinal strings."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "pipelines"))
        from run_deepseek_pipeline import get_ordinal

        assert get_ordinal(1) == "1st"
        assert get_ordinal(2) == "2nd"
        assert get_ordinal(3) == "3rd"
        assert get_ordinal(4) == "4th"
        assert get_ordinal(11) == "11th"
        assert get_ordinal(12) == "12th"
        assert get_ordinal(13) == "13th"
        assert get_ordinal(21) == "21st"
        assert get_ordinal(22) == "22nd"
        assert get_ordinal(23) == "23rd"

    def test_extract_router_logits(self):
        """Test extract_router_logits parses JSON correctly."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "pipelines"))
        from run_deepseek_pipeline import extract_router_logits

        # Create mock JSON data
        mock_data = {
            "num_layers": 2,
            "samples": [
                {
                    "sample_id": 0,
                    "layers": [
                        {
                            "layer": 0,
                            "router_logits_sample": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                            "expert_weights": [[0.3, 0.7], [0.4, 0.6]],
                            "selected_experts": [[0, 1], [2, 3]]
                        },
                        {
                            "layer": 1,
                            "router_logits_sample": [[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]],
                            "expert_weights": [[0.5, 0.5], [0.6, 0.4]],
                            "selected_experts": [[1, 2], [0, 3]]
                        }
                    ]
                }
            ]
        }

        logits, weights, choices = extract_router_logits(mock_data)

        # Check structure
        assert len(logits) == 2  # 2 layers
        assert len(weights) == 2
        assert len(choices) == 2

        # Check data
        assert logits[0].shape == (2, 4)  # 2 tokens, 4 experts
        assert logits[1].shape == (2, 4)

    def test_trim_top_and_bottom_experts(self):
        """Test trim_top_and_bottom_experts works correctly."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "pipelines"))
        from run_deepseek_pipeline import trim_top_and_bottom_experts

        # Create test array: (batch, tokens, experts)
        arr = np.array([[[1, 2, 3, 4, 5, 6, 7, 8]]])  # 1 batch, 1 token, 8 experts

        # No trimming
        result = trim_top_and_bottom_experts(arr, trim_amount=0)
        assert result.shape[-1] == 8

        # Trim 1 from each side
        result = trim_top_and_bottom_experts(arr, trim_amount=1)
        assert result.shape[-1] == 6  # 8 - 2 = 6

        # Trim 2 from each side
        result = trim_top_and_bottom_experts(arr, trim_amount=2)
        assert result.shape[-1] == 4  # 8 - 4 = 4

    def test_trim_top_and_bottom_2d_input(self):
        """Test trimming with 2D input."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "pipelines"))
        from run_deepseek_pipeline import trim_top_and_bottom_experts

        # 2D array (tokens, experts)
        arr = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])

        result = trim_top_and_bottom_experts(arr, trim_amount=2)
        assert result.shape[-1] == 4


# ==============================================================================
# SECTION 5: KDE TRAINING TESTS
# ==============================================================================

class TestKDETraining:
    """Test KDE model training functions."""

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_logits(self):
        """Create mock router logits."""
        np.random.seed(42)
        # Shape: (num_layers, num_tokens, num_experts)
        return np.random.randn(2, 100, 8).astype(np.float32)

    def test_kde_trains_on_mock_data(self, mock_logits):
        """Test KDE model trains on mock data."""
        from scipy.stats import gaussian_kde

        layer_data = mock_logits[0].flatten()

        # This should not raise
        kde = gaussian_kde(layer_data)

        # Test evaluation
        x_grid = np.linspace(layer_data.min(), layer_data.max(), 100)
        pdf = kde.evaluate(x_grid)

        assert len(pdf) == 100
        assert np.all(pdf >= 0)

    def test_kde_cdf_computation(self, mock_logits):
        """Test CDF computation from KDE."""
        from scipy.stats import gaussian_kde

        layer_data = mock_logits[0].flatten()
        kde = gaussian_kde(layer_data)

        x_grid = np.linspace(layer_data.min() - 0.2, layer_data.max() + 0.2, 1000)
        pdf_grid = kde.evaluate(x_grid)
        cdf_grid = np.cumsum(pdf_grid)
        cdf_grid /= cdf_grid[-1]  # Normalize

        # CDF should be monotonically increasing
        assert np.all(np.diff(cdf_grid) >= -1e-10)
        # CDF should end at 1.0
        assert np.isclose(cdf_grid[-1], 1.0)

    def test_kde_pickle_save_load(self, temp_dir, mock_logits):
        """Test KDE models save and load correctly via pickle."""
        from scipy.stats import gaussian_kde

        layer_data = mock_logits[0].flatten()
        kde = gaussian_kde(layer_data)

        x_grid = np.linspace(layer_data.min() - 0.2, layer_data.max() + 0.2, 1000)
        pdf_grid = kde.evaluate(x_grid)
        cdf_grid = np.cumsum(pdf_grid)
        cdf_grid /= cdf_grid[-1]

        model_data = {"x": x_grid, "cdf": cdf_grid}

        # Save
        model_path = temp_dir / "test_kde_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        assert model_path.exists()

        # Load
        with open(model_path, 'rb') as f:
            loaded_data = pickle.load(f)

        assert 'x' in loaded_data
        assert 'cdf' in loaded_data
        np.testing.assert_array_equal(loaded_data['x'], x_grid)
        np.testing.assert_array_equal(loaded_data['cdf'], cdf_grid)

    def test_sklearn_kde_kernels(self, mock_logits):
        """Test different sklearn KDE kernels work."""
        from sklearn.neighbors import KernelDensity

        layer_data = mock_logits[0].flatten()
        kernels = ["gaussian", "tophat", "epanechnikov", "linear"]

        for kernel in kernels:
            bandwidth = 1.06 * np.std(layer_data) * (len(layer_data) ** (-1/5))
            kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
            kde.fit(layer_data[:, np.newaxis])

            x_grid = np.linspace(layer_data.min(), layer_data.max(), 100)
            log_pdf = kde.score_samples(x_grid[:, np.newaxis])
            pdf = np.exp(log_pdf)

            assert len(pdf) == 100
            assert np.all(pdf >= 0), f"Kernel {kernel} produced negative PDF values"

    def test_probability_interpolation(self, mock_logits, temp_dir):
        """Test p-value computation via interpolation."""
        from scipy.stats import gaussian_kde

        # Train on first half
        train_data = mock_logits[0, :50, :].flatten()
        kde = gaussian_kde(train_data)

        x_grid = np.linspace(train_data.min() - 1, train_data.max() + 1, 1000)
        pdf_grid = kde.evaluate(x_grid)
        cdf_grid = np.cumsum(pdf_grid)
        cdf_grid /= cdf_grid[-1]

        # Test on second half
        test_data = mock_logits[0, 50:, :].flatten()
        probabilities = np.interp(test_data, x_grid, cdf_grid)

        # All probabilities should be in [0, 1]
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)

        # P-values
        p_values = 1 - probabilities
        assert np.all(p_values >= 0)
        assert np.all(p_values <= 1)


# ==============================================================================
# SECTION 6: END-TO-END MOCK TESTS
# ==============================================================================

class TestEndToEndMock:
    """End-to-end tests with mocked model and tiny fake dataset."""

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_deepseek_model(self):
        return MockModelDeepSeek(num_layers=2, num_experts=8, top_k=2)

    @pytest.fixture
    def mock_qwen_model(self):
        return MockModelQwen(num_layers=2, num_experts=8, top_k=2)

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "[EOS]"

        def mock_call(text, return_tensors="pt", truncation=True, max_length=512, padding=False):
            # Return random tokens
            result = MagicMock()
            result.__getitem__ = lambda self, key: torch.randint(0, 1000, (1, 10))
            return result

        tokenizer.__call__ = mock_call
        return tokenizer

    @pytest.fixture
    def tiny_dataset(self):
        """Create a tiny fake dataset."""
        return [
            {"text": "This is a test sentence for evaluation."},
            {"text": "Another example text for the pipeline."},
            {"text": "Third sample with some content here."},
        ]

    def test_deepseek_evaluation_mock(self, mock_deepseek_model, mock_tokenizer, tiny_dataset, temp_dir):
        """Test DeepSeek evaluation with mock model."""
        from moe_internal_logging_deepseek import RouterLogger

        router_logger = RouterLogger(mock_deepseek_model)
        router_logger.register_hooks(top_k=2)

        all_samples_data = []

        for i, sample in enumerate(tiny_dataset):
            router_logger.clear_data()

            # Simulate tokenization and forward pass
            input_ids = torch.randint(0, 1000, (1, 10))
            outputs = mock_deepseek_model(input_ids)

            routing_data = router_logger.get_routing_data()

            sample_data = {
                "sample_id": i,
                "num_tokens": 10,
                "loss": outputs.loss.item(),
                "layers": []
            }

            for layer_data in routing_data:
                sample_data["layers"].append({
                    "layer": layer_data["layer"],
                    "router_logits_shape": list(layer_data["router_logits"].shape),
                    "selected_experts": layer_data["expert_indices"].numpy().tolist(),
                    "expert_weights": layer_data["expert_weights"].numpy().tolist(),
                    "router_logits_sample": layer_data["router_logits"].numpy().tolist(),
                })

            all_samples_data.append(sample_data)

        router_logger.remove_hooks()

        # Verify structure
        assert len(all_samples_data) == 3
        for sample in all_samples_data:
            assert "layers" in sample
            assert len(sample["layers"]) == 2  # 2 layers

    def test_full_pipeline_mock_deepseek(self, mock_deepseek_model, temp_dir):
        """Test full pipeline stages with mock DeepSeek model."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "pipelines"))
        from run_deepseek_pipeline import setup_directories, extract_router_logits
        from moe_internal_logging_deepseek import RouterLogger
        from scipy.stats import gaussian_kde

        # Stage 1: Setup
        config = {
            "output_dir": str(temp_dir),
            "model_name": "test_deepseek",
            "num_experts": 8,
            "default_top_k": 2,
            "kde_trim_amounts": [0, 1],
            "kde_kernels": ["gaussian"],
            "plot_format": "png",
            "plot_dpi": 50,
            "show_plots": False,
        }

        dirs = setup_directories(config)
        assert dirs["base"].exists()

        # Stage 2: Mock evaluation and data collection
        router_logger = RouterLogger(mock_deepseek_model)
        router_logger.register_hooks(top_k=2)

        all_samples_data = []
        for i in range(5):  # 5 samples
            router_logger.clear_data()
            input_ids = torch.randint(0, 1000, (1, 8))
            outputs = mock_deepseek_model(input_ids)
            routing_data = router_logger.get_routing_data()

            sample_data = {
                "sample_id": i,
                "num_tokens": 8,
                "loss": 2.5,
                "layers": []
            }

            for layer_data in routing_data:
                sample_data["layers"].append({
                    "layer": layer_data["layer"],
                    "router_logits_shape": list(layer_data["router_logits"].shape),
                    "selected_experts": layer_data["expert_indices"].numpy().tolist(),
                    "expert_weights": layer_data["expert_weights"].numpy().tolist(),
                    "router_logits_sample": layer_data["router_logits"].numpy().tolist(),
                })

            all_samples_data.append(sample_data)

        router_logger.remove_hooks()

        # Save to JSON
        output_data = {
            "config": "test_model",
            "num_layers": 2,
            "num_experts": 8,
            "top_k": 2,
            "dataset": "test",
            "samples": all_samples_data
        }

        json_path = dirs["logs"] / "test_routing.json"
        with open(json_path, 'w') as f:
            json.dump(output_data, f)

        assert json_path.exists()

        # Stage 3: Load and extract
        with open(json_path, 'r') as f:
            loaded_data = json.load(f)

        logits, weights, choices = extract_router_logits(loaded_data)

        assert logits.shape[0] == 2  # 2 layers

        # Stage 4: Train KDE
        for layer_idx in range(2):
            layer_data = logits[layer_idx].flatten()
            kde = gaussian_kde(layer_data)

            x_grid = np.linspace(layer_data.min() - 0.5, layer_data.max() + 0.5, 100)
            pdf_grid = kde.evaluate(x_grid)
            cdf_grid = np.cumsum(pdf_grid)
            cdf_grid /= cdf_grid[-1]

            model_path = dirs["kde_models"] / f"test_kde_layer_{layer_idx}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump({"x": x_grid, "cdf": cdf_grid}, f)

            assert model_path.exists()

        # Stage 5: Verify KDE models load correctly
        for layer_idx in range(2):
            model_path = dirs["kde_models"] / f"test_kde_layer_{layer_idx}.pkl"
            with open(model_path, 'rb') as f:
                kde_data = pickle.load(f)

            assert 'x' in kde_data
            assert 'cdf' in kde_data

    def test_full_pipeline_mock_qwen(self, mock_qwen_model, temp_dir):
        """Test full pipeline stages with mock Qwen model."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "pipelines"))
        from run_qwen_pipeline import setup_directories, extract_router_logits
        from moe_internal_logging_qwen import RouterLogger
        from scipy.stats import gaussian_kde

        config = {
            "output_dir": str(temp_dir),
            "model_name": "test_qwen",
            "num_experts": 8,
            "default_top_k": 2,
            "kde_trim_amounts": [0],
            "kde_kernels": ["gaussian"],
            "plot_format": "png",
            "plot_dpi": 50,
            "show_plots": False,
        }

        dirs = setup_directories(config)

        # Mock evaluation
        router_logger = RouterLogger(mock_qwen_model)
        router_logger.register_hooks(top_k=2)

        all_samples_data = []
        for i in range(3):
            router_logger.clear_data()
            input_ids = torch.randint(0, 1000, (1, 6))
            outputs = mock_qwen_model(input_ids)
            routing_data = router_logger.get_routing_data()

            sample_data = {
                "sample_id": i,
                "num_tokens": 6,
                "loss": 2.3,
                "layers": []
            }

            for layer_data in routing_data:
                sample_data["layers"].append({
                    "layer": layer_data["layer"],
                    "router_logits_shape": list(layer_data["router_logits"].shape),
                    "selected_experts": layer_data["expert_indices"].numpy().tolist(),
                    "expert_weights": layer_data["expert_weights"].numpy().tolist(),
                    "router_logits_sample": layer_data["router_logits"].numpy().tolist(),
                })

            all_samples_data.append(sample_data)

        router_logger.remove_hooks()

        output_data = {
            "config": "test_qwen",
            "num_layers": 2,
            "num_experts": 8,
            "top_k": 2,
            "dataset": "test",
            "samples": all_samples_data
        }

        json_path = dirs["logs"] / "test_qwen_routing.json"
        with open(json_path, 'w') as f:
            json.dump(output_data, f)

        # Extract and verify
        with open(json_path, 'r') as f:
            loaded_data = json.load(f)

        logits, weights, choices = extract_router_logits(loaded_data)
        assert logits.shape[0] == 2


class TestInternalRoutingLogger:
    """Test InternalRoutingLogger for both models."""

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_deepseek_internal_logger(self, temp_dir):
        """Test InternalRoutingLogger from deepseek module."""
        from moe_internal_logging_deepseek import InternalRoutingLogger

        logger = InternalRoutingLogger(
            output_dir=temp_dir,
            experiment_name='test_deepseek',
            routing_method='topk'
        )

        # Log mock samples
        for i in range(3):
            routing_data = [
                {
                    'layer': j,
                    'router_logits': torch.randn(10, 64),
                    'expert_indices': torch.randint(0, 64, (10, 6)),
                    'expert_weights': torch.rand(10, 6),
                    'probs': torch.rand(10, 64),
                    'stats': {
                        'max_prob': 0.5,
                        'entropy': 2.3,
                        'concentration': 0.4,
                        'num_tokens': 10,
                        'unique_experts_this_layer': 12
                    }
                }
                for j in range(2)
            ]

            logger.log_sample(
                sample_id=i,
                routing_data=routing_data,
                loss=2.5,
                num_tokens=10
            )

        stats = logger.get_aggregate_stats()
        assert stats['num_samples'] == 3
        assert stats['total_tokens'] == 30

        log_file = logger.save_logs()
        assert Path(log_file).exists()

    def test_qwen_internal_logger(self, temp_dir):
        """Test InternalRoutingLogger from qwen module."""
        from moe_internal_logging_qwen import InternalRoutingLogger

        logger = InternalRoutingLogger(
            output_dir=temp_dir,
            experiment_name='test_qwen',
            routing_method='topk'
        )

        # Log mock samples
        for i in range(2):
            routing_data = [
                {
                    'layer': j,
                    'router_logits': torch.randn(8, 128),
                    'expert_indices': torch.randint(0, 128, (8, 8)),
                    'expert_weights': torch.rand(8, 8),
                    'probs': torch.rand(8, 128),
                    'stats': {
                        'max_prob': 0.4,
                        'entropy': 3.0,
                        'concentration': 0.3,
                        'num_tokens': 8,
                        'unique_experts_this_layer': 20
                    }
                }
                for j in range(2)
            ]

            logger.log_sample(
                sample_id=i,
                routing_data=routing_data,
                loss=2.3,
                num_tokens=8
            )

        stats = logger.get_aggregate_stats()
        assert stats['num_samples'] == 2

        logger.clear()
        assert len(logger.all_samples) == 0


# ==============================================================================
# SECTION 7: PLOTTING FUNCTION TESTS (Smoke Tests)
# ==============================================================================

class TestPlottingFunctions:
    """Smoke tests for plotting functions (don't actually display plots)."""

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_matplotlib_backend(self):
        """Test that matplotlib works with non-GUI backend."""
        import matplotlib
        matplotlib.use('Agg')  # Non-GUI backend
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        plt.close(fig)

    def test_seaborn_histplot(self):
        """Test seaborn histplot works."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        data = np.random.randn(100)
        fig, ax = plt.subplots()
        sns.histplot(data, bins=20, kde=True, ax=ax)
        plt.close(fig)

    def test_save_plot_function(self, temp_dir):
        """Test save_plot utility."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "pipelines"))
        from run_deepseek_pipeline import save_plot

        config = {
            "plot_format": "png",
            "plot_dpi": 50,
            "show_plots": False
        }

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        plot_path = temp_dir / "test_plot.png"
        save_plot(fig, plot_path, config)

        assert plot_path.exists()


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
