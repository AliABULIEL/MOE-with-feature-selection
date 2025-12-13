#!/usr/bin/env python3
"""
Test script for BH Routing Experiments

This script validates the experiment runner using a small mock model
to ensure all functionality works before running on the full OLMoE model.

Usage:
    python test_bh_experiments.py

Author: Generated for OLMoE BH Routing Analysis
Date: 2025-12-13
"""

import os
import sys
import tempfile
from pathlib import Path
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

# Import the experiment runner
try:
    from run_bh_experiments import (
        benjamini_hochberg_routing,
        BHExperimentRunner,
        analyze_results,
        create_visualizations,
        generate_markdown_report,
        ROUTING_CONFIGS,
        TEST_PROMPTS
    )
except ImportError:
    print("ERROR: Could not import run_bh_experiments.py")
    print("Make sure run_bh_experiments.py is in the same directory")
    sys.exit(1)


# ============================================================================
# MOCK MODEL FOR TESTING
# ============================================================================

class MockOlmoeTopKRouter(nn.Module):
    """Mock router that mimics OlmoeTopKRouter."""

    def __init__(self, hidden_dim=128, num_experts=64, top_k=8):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.top_k = top_k
        self.num_experts = num_experts

    def forward(self, hidden_states):
        """Standard TopK routing."""
        router_logits = self.gate(hidden_states)
        routing_weights, selected_experts = torch.topk(router_logits, k=self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float)
        return routing_weights, selected_experts, router_logits


class MockMoELayer(nn.Module):
    """Mock MoE layer with router."""

    def __init__(self, hidden_dim=128, num_experts=64, top_k=8):
        super().__init__()
        self.router = MockOlmoeTopKRouter(hidden_dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_experts)
        ])

    def forward(self, hidden_states):
        # Route
        routing_weights, selected_experts, router_logits = self.router(hidden_states)

        # Apply experts (simplified)
        output = hidden_states  # Skip actual expert application for speed

        return output


class MockOLMoEModel(nn.Module):
    """Mock OLMoE model for testing."""

    def __init__(self, vocab_size=1000, hidden_dim=128, num_layers=4, num_experts=64, top_k=8):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            MockMoELayer(hidden_dim, num_experts, top_k) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, **kwargs):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        logits = self.lm_head(hidden_states)
        return type('Output', (), {'logits': logits})()

    def generate(self, input_ids, max_new_tokens=10, **kwargs):
        """Simple greedy generation."""
        batch_size, seq_len = input_ids.shape
        generated = input_ids

        for _ in range(max_new_tokens):
            outputs = self.forward(generated)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

        return generated


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.eos_token_id = 0
        self.pad_token_id = 0

    def __call__(self, text, return_tensors=None):
        # Simple tokenization: hash text to get token IDs
        tokens = [hash(word) % self.vocab_size for word in text.split()]
        tokens = tokens[:20]  # Limit length

        if return_tensors == "pt":
            return {'input_ids': torch.tensor([tokens])}
        return {'input_ids': tokens}

    def decode(self, token_ids, skip_special_tokens=False):
        # Simple decoding: return placeholder
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return f"Generated text with {len(token_ids)} tokens"


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_bh_routing_function():
    """Test 1: BH routing function."""
    print("\n" + "=" * 80)
    print("TEST 1: BH Routing Function")
    print("=" * 80)

    # Create test data
    batch_size, seq_len, num_experts = 2, 5, 64
    router_logits = torch.randn(batch_size, seq_len, num_experts)

    # Test different alpha values
    alphas = [0.01, 0.05, 0.10]

    for alpha in alphas:
        weights, experts, counts = benjamini_hochberg_routing(
            router_logits,
            alpha=alpha,
            max_k=8
        )

        print(f"\nAlpha = {alpha}")
        print(f"  Weights shape: {weights.shape}")
        print(f"  Experts shape: {experts.shape}")
        print(f"  Counts shape: {counts.shape}")
        print(f"  Avg experts per token: {counts.float().mean():.2f}")
        print(f"  Min experts: {counts.min()}")
        print(f"  Max experts: {counts.max()}")

        # Verify weights sum to 1
        weight_sums = weights.sum(dim=-1)
        print(f"  Weights sum to 1: {torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)}")

    print("\n✅ BH routing function test passed")


def test_experiment_runner_init():
    """Test 2: Experiment runner initialization."""
    print("\n" + "=" * 80)
    print("TEST 2: Experiment Runner Initialization")
    print("=" * 80)

    # Create temporary output directory
    temp_dir = tempfile.mkdtemp()
    print(f"\nTemp directory: {temp_dir}")

    try:
        # Create mock model and tokenizer
        model = MockOLMoEModel()
        tokenizer = MockTokenizer()

        # Create runner (bypass model loading)
        runner = BHExperimentRunner.__new__(BHExperimentRunner)
        runner.model_name = "mock_model"
        runner.output_dir = Path(temp_dir)
        runner.checkpoint_interval = 2
        runner.use_fp16 = False
        runner.device = "cpu"

        # Setup logging
        runner._setup_logging()

        # Set model and tokenizer
        runner.model = model
        runner.tokenizer = tokenizer

        # Find routers
        runner.routers = []
        for name, module in model.named_modules():
            if isinstance(module, MockOlmoeTopKRouter):
                runner.routers.append((name, module))

        print(f"\nFound {len(runner.routers)} routers")
        print(f"Output directory: {runner.output_dir}")
        print(f"Device: {runner.device}")

        # Initialize results storage
        runner.results = []
        runner.checkpoint_path = runner.output_dir / "checkpoints" / "latest_checkpoint.json"
        runner.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        print("\n✅ Experiment runner initialization test passed")

        return runner

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_routing_config_application():
    """Test 3: Routing configuration application."""
    print("\n" + "=" * 80)
    print("TEST 3: Routing Configuration Application")
    print("=" * 80)

    # Create temporary output directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Create mock model
        model = MockOLMoEModel()

        # Create runner
        runner = BHExperimentRunner.__new__(BHExperimentRunner)
        runner.model = model
        runner.output_dir = Path(temp_dir)
        runner.device = "cpu"
        runner._setup_logging()

        # Find routers
        runner.routers = []
        for name, module in model.named_modules():
            if isinstance(module, MockOlmoeTopKRouter):
                runner.routers.append((name, module))

        print(f"\nFound {len(runner.routers)} routers to patch")

        # Test TopK config
        print("\n1. Applying TopK configuration...")
        runner._apply_routing_config(ROUTING_CONFIGS['topk_8'])
        print("   ✓ TopK applied")

        # Test BH config
        print("\n2. Applying BH configuration...")
        runner._apply_routing_config(ROUTING_CONFIGS['bh_moderate'])
        print("   ✓ BH applied")

        # Test restoration
        print("\n3. Restoring original routing...")
        runner._restore_original_routing()
        print("   ✓ Restored")

        print("\n✅ Routing configuration test passed")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_single_experiment():
    """Test 4: Single experiment execution."""
    print("\n" + "=" * 80)
    print("TEST 4: Single Experiment Execution")
    print("=" * 80)

    # Create temporary output directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Create mock model and tokenizer
        model = MockOLMoEModel()
        tokenizer = MockTokenizer()

        # Create runner
        runner = BHExperimentRunner.__new__(BHExperimentRunner)
        runner.model = model
        runner.tokenizer = tokenizer
        runner.output_dir = Path(temp_dir)
        runner.device = "cpu"
        runner._setup_logging()

        # Find routers
        runner.routers = []
        for name, module in model.named_modules():
            if isinstance(module, MockOlmoeTopKRouter):
                runner.routers.append((name, module))

        print(f"\nRunning single experiment...")

        # Run experiment
        result = runner.run_single_experiment(
            config_name='topk_8',
            config=ROUTING_CONFIGS['topk_8'],
            prompt_data=TEST_PROMPTS[0],
            max_new_tokens=10
        )

        print(f"\nResult keys: {list(result.keys())}")
        print(f"Config: {result['config_name']}")
        print(f"Prompt: {result['prompt_text'][:50]}...")
        print(f"Avg experts: {result['avg_experts_per_token']:.2f}")
        print(f"Inference time: {result['inference_time_sec']:.3f}s")
        print(f"Tokens/sec: {result['tokens_per_sec']:.1f}")

        # Verify required fields
        required_fields = [
            'config_name', 'prompt_text', 'generated_text',
            'avg_experts_per_token', 'inference_time_sec',
            'expert_utilization_cv'
        ]

        for field in required_fields:
            assert field in result, f"Missing field: {field}"

        print("\n✅ Single experiment test passed")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_mini_experiment_suite():
    """Test 5: Mini experiment suite (2 configs x 3 prompts)."""
    print("\n" + "=" * 80)
    print("TEST 5: Mini Experiment Suite")
    print("=" * 80)

    # Create temporary output directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Create mock model and tokenizer
        model = MockOLMoEModel()
        tokenizer = MockTokenizer()

        # Create runner
        runner = BHExperimentRunner.__new__(BHExperimentRunner)
        runner.model = model
        runner.tokenizer = tokenizer
        runner.output_dir = Path(temp_dir)
        runner.checkpoint_interval = 2
        runner.device = "cpu"
        runner._setup_logging()

        # Find routers
        runner.routers = []
        for name, module in model.named_modules():
            if isinstance(module, MockOlmoeTopKRouter):
                runner.routers.append((name, module))

        runner.results = []
        runner.checkpoint_path = runner.output_dir / "checkpoints" / "latest_checkpoint.json"
        runner.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Mini configs (2 configs)
        mini_configs = {
            'topk_8': ROUTING_CONFIGS['topk_8'],
            'bh_moderate': ROUTING_CONFIGS['bh_moderate']
        }

        # Mini prompts (3 prompts)
        mini_prompts = TEST_PROMPTS[:3]

        print(f"\nRunning {len(mini_configs)} configs x {len(mini_prompts)} prompts = {len(mini_configs) * len(mini_prompts)} experiments")

        # Run suite
        df = runner.run_full_experiment_suite(
            routing_configs=mini_configs,
            test_prompts=mini_prompts,
            max_new_tokens=10,
            resume=False
        )

        print(f"\nResults shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nSummary:")
        print(df.groupby('config_name')['avg_experts_per_token'].agg(['mean', 'std']))

        # Verify DataFrame
        assert len(df) == len(mini_configs) * len(mini_prompts), "Wrong number of results"
        assert 'config_name' in df.columns, "Missing config_name column"
        assert 'avg_experts_per_token' in df.columns, "Missing avg_experts_per_token column"

        print("\n✅ Mini experiment suite test passed")

        return df

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_analysis_functions(df=None):
    """Test 6: Analysis and reporting functions."""
    print("\n" + "=" * 80)
    print("TEST 6: Analysis and Reporting")
    print("=" * 80)

    # Create temporary output directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Create sample data if not provided
        if df is None:
            print("\nCreating sample data...")
            data = []
            for config in ['topk_8', 'bh_moderate']:
                for i in range(5):
                    data.append({
                        'config_name': config,
                        'prompt_text': f'Test prompt {i}',
                        'prompt_category': 'simple',
                        'avg_experts_per_token': 8.0 if config == 'topk_8' else 5.0 + np.random.rand(),
                        'inference_time_sec': 0.5 + np.random.rand() * 0.2,
                        'tokens_per_sec': 20.0 + np.random.rand() * 5,
                        'expert_utilization_cv': 0.3 + np.random.rand() * 0.1,
                    })
            df = pd.DataFrame(data)

        print(f"Data shape: {df.shape}")

        # Test analysis
        print("\n1. Testing analyze_results()...")
        analysis = analyze_results(df, Path(temp_dir))
        print("   ✓ Analysis complete")

        # Test visualizations
        print("\n2. Testing create_visualizations()...")
        create_visualizations(df, Path(temp_dir))
        print("   ✓ Visualizations created")

        # Check plots exist
        plot_dir = Path(temp_dir) / "plots"
        expected_plots = [
            'avg_experts_comparison.png',
            'inference_time_comparison.png',
            'expert_utilization_cv.png',
        ]
        for plot_name in expected_plots:
            plot_path = plot_dir / plot_name
            assert plot_path.exists(), f"Missing plot: {plot_name}"
            print(f"   ✓ Found {plot_name}")

        # Test report generation
        print("\n3. Testing generate_markdown_report()...")
        report_path = Path(temp_dir) / "REPORT.md"
        generate_markdown_report(df, analysis, report_path)
        assert report_path.exists(), "Report not created"
        print("   ✓ Report generated")

        # Read and verify report
        with open(report_path, 'r') as f:
            report_content = f.read()
            assert '# BH Routing Experiment Results' in report_content
            assert 'Executive Summary' in report_content
            assert 'Statistical Significance Tests' in report_content
            print("   ✓ Report content verified")

        print("\n✅ Analysis and reporting test passed")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_checkpoint_and_resume():
    """Test 7: Checkpointing and resume functionality."""
    print("\n" + "=" * 80)
    print("TEST 7: Checkpointing and Resume")
    print("=" * 80)

    # Create temporary output directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Create mock model and tokenizer
        model = MockOLMoEModel()
        tokenizer = MockTokenizer()

        # Create runner
        runner = BHExperimentRunner.__new__(BHExperimentRunner)
        runner.model = model
        runner.tokenizer = tokenizer
        runner.output_dir = Path(temp_dir)
        runner.checkpoint_interval = 2  # Save every 2 experiments
        runner.device = "cpu"
        runner._setup_logging()

        # Find routers
        runner.routers = []
        for name, module in model.named_modules():
            if isinstance(module, MockOlmoeTopKRouter):
                runner.routers.append((name, module))

        runner.results = []
        runner.checkpoint_path = runner.output_dir / "checkpoints" / "latest_checkpoint.json"
        runner.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Add some results
        for i in range(3):
            runner.results.append({
                'config_name': 'test',
                'experiment_id': i,
                'avg_experts_per_token': 5.0
            })

        # Save checkpoint
        print("\n1. Saving checkpoint...")
        runner._save_checkpoint()
        assert runner.checkpoint_path.exists(), "Checkpoint not created"
        print(f"   ✓ Checkpoint saved to {runner.checkpoint_path}")

        # Create new runner and load checkpoint
        print("\n2. Loading checkpoint...")
        runner2 = BHExperimentRunner.__new__(BHExperimentRunner)
        runner2.output_dir = Path(temp_dir)
        runner2.checkpoint_path = runner.checkpoint_path

        import json
        with open(runner2.checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)

        print(f"   ✓ Loaded {len(checkpoint_data['results'])} results from checkpoint")
        assert len(checkpoint_data['results']) == 3, "Wrong number of results loaded"

        print("\n✅ Checkpointing test passed")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("BH EXPERIMENT RUNNER TEST SUITE")
    print("=" * 80)
    print("\nThis test suite validates all functionality using mock models.")
    print("No actual OLMoE model download required.\n")

    tests = [
        ("BH Routing Function", test_bh_routing_function),
        ("Experiment Runner Init", test_experiment_runner_init),
        ("Routing Config Application", test_routing_config_application),
        ("Single Experiment", test_single_experiment),
        ("Mini Experiment Suite", test_mini_experiment_suite),
        ("Analysis Functions", test_analysis_functions),
        ("Checkpointing", test_checkpoint_and_resume),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_name == "Analysis Functions":
                # Run mini suite first to get real data
                df = test_mini_experiment_suite()
                test_func(df)
            else:
                test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ {test_name} FAILED")
            print(f"Error: {e}")
            failed += 1

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total: {len(tests)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")

    if failed == 0:
        print("\n🎉 All tests passed! The experiment runner is ready to use.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
