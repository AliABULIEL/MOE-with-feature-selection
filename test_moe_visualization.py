"""
Unit Tests for moe_visualization.py
====================================

Tests for visualization functions with mock data.

Run with:
    pytest test_moe_visualization.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from moe_visualization import (
    create_comprehensive_dashboard,
    plot_per_layer_routing,
    plot_expert_usage_heatmap,
    plot_bh_vs_hc_comparison,
    _plot_metric_by_dataset,
    _empty_plot,
    _generate_summary_text
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for output files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_results_df():
    """Create mock results DataFrame."""
    np.random.seed(42)

    data = []

    # Baseline configs
    for k in [8, 16]:
        data.append({
            'config_name': f'topk_{k}',
            'routing_type': 'topk',
            'dataset': 'wikitext',
            'k_or_max_k': k,
            'alpha': None,
            'beta': None,
            'perplexity': 20.0 + np.random.randn(),
            'token_accuracy': 0.5 + np.random.rand() * 0.1,
            'avg_experts': float(k),
            'std_experts': 0.0,
            'min_experts': k,
            'max_experts': k,
            'ceiling_hit_rate': 100.0,
            'floor_hit_rate': 0.0,
            'mid_range_rate': 0.0,
            'adaptive_range': 0,
            'reduction_vs_baseline': (8 - k) / 8 * 100,
            'avg_entropy': 2.0 + np.random.rand(),
            'avg_concentration': 0.5 + np.random.rand() * 0.2,
            'unique_experts': k * 4,
            'expert_utilization': k * 4 / 64
        })

    # BH configs
    for alpha in [0.30, 0.50]:
        for max_k in [8, 16]:
            data.append({
                'config_name': f'bh_a{int(alpha*100):03d}_maxk{max_k}',
                'routing_type': 'bh',
                'dataset': 'wikitext',
                'k_or_max_k': max_k,
                'alpha': alpha,
                'beta': None,
                'perplexity': 20.0 + np.random.randn(),
                'token_accuracy': 0.5 + np.random.rand() * 0.1,
                'avg_experts': max_k * (0.7 + alpha * 0.3),
                'std_experts': 1.5 + np.random.rand(),
                'min_experts': 1,
                'max_experts': max_k,
                'ceiling_hit_rate': 20.0 + np.random.rand() * 30,
                'floor_hit_rate': 5.0 + np.random.rand() * 10,
                'mid_range_rate': 50.0 + np.random.rand() * 20,
                'adaptive_range': max_k - 1,
                'reduction_vs_baseline': (8 - max_k * (0.7 + alpha * 0.3)) / 8 * 100,
                'avg_entropy': 2.5 + np.random.rand(),
                'avg_concentration': 0.4 + np.random.rand() * 0.2,
                'unique_experts': 40 + np.random.randint(-5, 5),
                'expert_utilization': (40 + np.random.randint(-5, 5)) / 64
            })

    # HC configs
    for beta in [0.30, 0.50]:
        for max_k in [8, 16]:
            data.append({
                'config_name': f'hc_b{int(beta*100):03d}_maxk{max_k}',
                'routing_type': 'hc',
                'dataset': 'wikitext',
                'k_or_max_k': max_k,
                'alpha': None,
                'beta': beta,
                'perplexity': 20.0 + np.random.randn(),
                'token_accuracy': 0.5 + np.random.rand() * 0.1,
                'avg_experts': max_k * (0.6 + beta * 0.4),
                'std_experts': 1.8 + np.random.rand(),
                'min_experts': 1,
                'max_experts': max_k,
                'ceiling_hit_rate': 15.0 + np.random.rand() * 25,
                'floor_hit_rate': 8.0 + np.random.rand() * 12,
                'mid_range_rate': 55.0 + np.random.rand() * 15,
                'adaptive_range': max_k - 1,
                'reduction_vs_baseline': (8 - max_k * (0.6 + beta * 0.4)) / 8 * 100,
                'avg_entropy': 2.7 + np.random.rand(),
                'avg_concentration': 0.35 + np.random.rand() * 0.2,
                'unique_experts': 45 + np.random.randint(-5, 5),
                'expert_utilization': (45 + np.random.randint(-5, 5)) / 64
            })

    return pd.DataFrame(data)


@pytest.fixture
def mock_per_layer_stats():
    """Create mock per-layer statistics."""
    return {
        i: {
            'avg_entropy': 2.0 + np.random.rand(),
            'avg_max_prob': 0.3 + np.random.rand() * 0.3,
            'avg_concentration': 0.4 + np.random.rand() * 0.2
        }
        for i in range(16)
    }


@pytest.fixture
def mock_expert_usage():
    """Create mock expert usage dictionary."""
    return {i: np.random.randint(100, 1000) for i in range(64)}


# =========================================================================
# Test Comprehensive Dashboard
# =========================================================================

class TestComprehensiveDashboard:
    """Tests for create_comprehensive_dashboard function."""

    def test_dashboard_creation(self, mock_results_df, temp_dir):
        """Test creating dashboard with valid data."""
        output_path = str(Path(temp_dir) / 'dashboard.png')

        result_path = create_comprehensive_dashboard(
            results_df=mock_results_df,
            output_path=output_path,
            title="Test Dashboard",
            routing_method="BH/HC"
        )

        assert Path(result_path).exists()
        assert Path(result_path).stat().st_size > 0  # File has content

    def test_dashboard_with_missing_columns(self, temp_dir):
        """Test dashboard creation with missing columns."""
        # Create minimal DataFrame
        df = pd.DataFrame({
            'config_name': ['test1', 'test2'],
            'avg_experts': [6.0, 7.0]
        })

        output_path = str(Path(temp_dir) / 'dashboard_minimal.png')

        # Should not crash, but create placeholders
        result_path = create_comprehensive_dashboard(
            results_df=df,
            output_path=output_path,
            title="Minimal Dashboard",
            routing_method="TEST"
        )

        assert Path(result_path).exists()

    def test_dashboard_with_nan_values(self, mock_results_df, temp_dir):
        """Test dashboard handles NaN values gracefully."""
        # Introduce NaN values
        df = mock_results_df.copy()
        df.loc[0, 'perplexity'] = np.nan
        df.loc[1, 'avg_experts'] = np.nan

        output_path = str(Path(temp_dir) / 'dashboard_nan.png')

        result_path = create_comprehensive_dashboard(
            results_df=df,
            output_path=output_path,
            title="Dashboard with NaN",
            routing_method="BH"
        )

        assert Path(result_path).exists()


# =========================================================================
# Test Per-Layer Routing Plot
# =========================================================================

class TestPerLayerRouting:
    """Tests for plot_per_layer_routing function."""

    def test_per_layer_plot_creation(self, mock_per_layer_stats, temp_dir):
        """Test creating per-layer routing plot."""
        output_path = str(Path(temp_dir) / 'per_layer.png')

        plot_per_layer_routing(
            per_layer_stats=mock_per_layer_stats,
            output_path=output_path,
            title="Test Per-Layer Stats"
        )

        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0

    def test_per_layer_with_missing_keys(self, temp_dir):
        """Test per-layer plot with missing stat keys."""
        # Create stats with only entropy
        stats = {i: {'entropy': 2.0} for i in range(5)}

        output_path = str(Path(temp_dir) / 'per_layer_minimal.png')

        plot_per_layer_routing(
            per_layer_stats=stats,
            output_path=output_path,
            title="Minimal Per-Layer"
        )

        assert Path(output_path).exists()


# =========================================================================
# Test Expert Usage Heatmap
# =========================================================================

class TestExpertUsageHeatmap:
    """Tests for plot_expert_usage_heatmap function."""

    def test_heatmap_creation(self, mock_expert_usage, temp_dir):
        """Test creating expert usage heatmap."""
        output_path = str(Path(temp_dir) / 'heatmap.png')

        plot_expert_usage_heatmap(
            expert_usage=mock_expert_usage,
            num_experts=64,
            output_path=output_path,
            title="Test Heatmap"
        )

        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0

    def test_heatmap_with_sparse_usage(self, temp_dir):
        """Test heatmap with sparse expert usage."""
        # Only a few experts used
        usage = {0: 1000, 5: 500, 10: 200}

        output_path = str(Path(temp_dir) / 'heatmap_sparse.png')

        plot_expert_usage_heatmap(
            expert_usage=usage,
            num_experts=64,
            output_path=output_path,
            title="Sparse Heatmap"
        )

        assert Path(output_path).exists()


# =========================================================================
# Test BH vs HC Comparison
# =========================================================================

class TestBHvsHCComparison:
    """Tests for plot_bh_vs_hc_comparison function."""

    def test_comparison_plot(self, mock_results_df, temp_dir):
        """Test creating BH vs HC comparison plot."""
        bh_df = mock_results_df[mock_results_df['routing_type'] == 'bh']
        hc_df = mock_results_df[mock_results_df['routing_type'] == 'hc']

        output_path = str(Path(temp_dir) / 'comparison.png')

        plot_bh_vs_hc_comparison(
            bh_results=bh_df,
            hc_results=hc_df,
            output_path=output_path
        )

        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0

    def test_comparison_with_empty_dfs(self, temp_dir):
        """Test comparison with empty DataFrames."""
        bh_df = pd.DataFrame()
        hc_df = pd.DataFrame()

        output_path = str(Path(temp_dir) / 'comparison_empty.png')

        plot_bh_vs_hc_comparison(
            bh_results=bh_df,
            hc_results=hc_df,
            output_path=output_path
        )

        assert Path(output_path).exists()


# =========================================================================
# Test Helper Functions
# =========================================================================

class TestHelperFunctions:
    """Tests for internal helper functions."""

    def test_empty_plot(self):
        """Test _empty_plot creates a valid plot."""
        fig, ax = plt.subplots()
        _empty_plot(ax, "No data available")

        # Should have text at center
        texts = [t for t in ax.texts]
        assert len(texts) > 0
        assert texts[0].get_text() == "No data available"

        plt.close(fig)

    def test_generate_summary_text(self, mock_results_df):
        """Test _generate_summary_text creates valid text."""
        df = mock_results_df
        adaptive_df = df[df['routing_type'].isin(['bh', 'hc'])]
        baseline_df = df[df['routing_type'] == 'topk']

        summary = _generate_summary_text(df, adaptive_df, baseline_df, "BH/HC")

        assert isinstance(summary, str)
        assert "EXPERIMENT SUMMARY" in summary
        assert "BH/HC" in summary
        assert "Configurations:" in summary


# =========================================================================
# Integration Tests
# =========================================================================

class TestVisualizationIntegration:
    """Integration tests for visualization pipeline."""

    def test_full_visualization_pipeline(self, mock_results_df, mock_per_layer_stats, mock_expert_usage, temp_dir):
        """Test generating all visualizations together."""
        # Dashboard
        dashboard_path = create_comprehensive_dashboard(
            results_df=mock_results_df,
            output_path=str(Path(temp_dir) / 'dashboard.png'),
            title="Full Pipeline Test",
            routing_method="BH/HC"
        )

        # Per-layer plot
        per_layer_path = str(Path(temp_dir) / 'per_layer.png')
        plot_per_layer_routing(
            per_layer_stats=mock_per_layer_stats,
            output_path=per_layer_path,
            title="Per-Layer Stats"
        )

        # Heatmap
        heatmap_path = str(Path(temp_dir) / 'heatmap.png')
        plot_expert_usage_heatmap(
            expert_usage=mock_expert_usage,
            num_experts=64,
            output_path=heatmap_path,
            title="Expert Usage"
        )

        # Comparison
        bh_df = mock_results_df[mock_results_df['routing_type'] == 'bh']
        hc_df = mock_results_df[mock_results_df['routing_type'] == 'hc']
        comparison_path = str(Path(temp_dir) / 'comparison.png')
        plot_bh_vs_hc_comparison(
            bh_results=bh_df,
            hc_results=hc_df,
            output_path=comparison_path
        )

        # All files should exist
        assert Path(dashboard_path).exists()
        assert Path(per_layer_path).exists()
        assert Path(heatmap_path).exists()
        assert Path(comparison_path).exists()

        # All should have content
        assert Path(dashboard_path).stat().st_size > 0
        assert Path(per_layer_path).stat().st_size > 0
        assert Path(heatmap_path).stat().st_size > 0
        assert Path(comparison_path).stat().st_size > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
