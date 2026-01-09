#!/usr/bin/env python3
"""
Verification Script for BH & HC Routing Enhancement
===================================================

This script verifies that all components are in place and working.

Run with:
    python verify_enhancement.py
"""

import sys
import os
from pathlib import Path
import importlib


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_check(passed, message):
    """Print check result."""
    symbol = "✅" if passed else "❌"
    print(f"{symbol} {message}")
    return passed


def verify_files_exist():
    """Verify all required files exist."""
    print_header("VERIFYING FILES EXIST")

    required_files = [
        # Shared modules
        'moe_internal_logging.py',
        'moe_metrics.py',
        'moe_visualization.py',

        # Routing implementations
        'bh_routing.py',
        'hc_routing.py',

        # Notebooks
        'OLMoE_BH_Routing_Experiments.ipynb',
        'OLMoE_HC_Routing_Experiments.ipynb',

        # Tests
        'test_moe_internal_logging.py',
        'test_moe_metrics.py',
        'test_moe_visualization.py',

        # Documentation
        'NOTEBOOK_ENHANCEMENT_IMPLEMENTATION.md',
        'ENHANCED_FRAMEWORK_INTEGRATION_GUIDE.md',
        'UNIFIED_ENHANCEMENT_IMPLEMENTATION_COMPLETE.md'
    ]

    all_passed = True
    for file in required_files:
        exists = Path(file).exists()
        all_passed &= print_check(exists, f"File exists: {file}")

    return all_passed


def verify_imports():
    """Verify all shared modules can be imported."""
    print_header("VERIFYING IMPORTS")

    all_passed = True

    # Test moe_internal_logging
    try:
        from moe_internal_logging import RouterLogger, InternalRoutingLogger
        all_passed &= print_check(True, "Import moe_internal_logging: RouterLogger, InternalRoutingLogger")
    except ImportError as e:
        all_passed &= print_check(False, f"Import moe_internal_logging failed: {e}")

    # Test moe_metrics
    try:
        from moe_metrics import MetricsComputer, ComprehensiveMetrics
        all_passed &= print_check(True, "Import moe_metrics: MetricsComputer, ComprehensiveMetrics")
    except ImportError as e:
        all_passed &= print_check(False, f"Import moe_metrics failed: {e}")

    # Test moe_visualization
    try:
        from moe_visualization import (
            create_comprehensive_dashboard,
            plot_per_layer_routing,
            plot_expert_usage_heatmap
        )
        all_passed &= print_check(True, "Import moe_visualization: dashboard, plots, heatmap")
    except ImportError as e:
        all_passed &= print_check(False, f"Import moe_visualization failed: {e}")

    # Test BH routing
    try:
        from deprecated.bh_routing import benjamini_hochberg_routing, load_kde_models
        all_passed &= print_check(True, "Import bh_routing: benjamini_hochberg_routing, load_kde_models")
    except ImportError as e:
        all_passed &= print_check(False, f"Import bh_routing failed: {e}")

    # Test HC routing
    try:
        from hc_routing import higher_criticism_routing
        all_passed &= print_check(True, "Import hc_routing: higher_criticism_routing")
    except ImportError as e:
        all_passed &= print_check(False, f"Import hc_routing failed: {e}")

    return all_passed


def verify_class_instantiation():
    """Verify classes can be instantiated."""
    print_header("VERIFYING CLASS INSTANTIATION")

    all_passed = True

    try:
        from moe_metrics import ComprehensiveMetrics
        metrics = ComprehensiveMetrics()
        all_passed &= print_check(True, "ComprehensiveMetrics instantiation")
        all_passed &= print_check(metrics.perplexity == float('inf'), "ComprehensiveMetrics default values")
    except Exception as e:
        all_passed &= print_check(False, f"ComprehensiveMetrics instantiation failed: {e}")

    return all_passed


def verify_function_calls():
    """Verify key functions work with mock data."""
    print_header("VERIFYING FUNCTION CALLS")

    all_passed = True

    # Test MetricsComputer
    try:
        from moe_metrics import MetricsComputer
        import numpy as np

        perplexity = MetricsComputer.compute_perplexity(10.0, 100)
        all_passed &= print_check(1.0 < perplexity < 2.0, f"compute_perplexity: {perplexity:.3f}")

        routing_stats = MetricsComputer.compute_routing_stats(
            np.array([3, 4, 5, 6, 7, 8]), max_k=8
        )
        all_passed &= print_check('avg_experts' in routing_stats, "compute_routing_stats")

        f1 = MetricsComputer.compute_f1("hello world", "hello world")
        all_passed &= print_check(f1 == 1.0, f"compute_f1: {f1}")

    except Exception as e:
        all_passed &= print_check(False, f"MetricsComputer functions failed: {e}")

    return all_passed


def verify_tests_runnable():
    """Verify tests can be discovered by pytest."""
    print_header("VERIFYING TESTS")

    all_passed = True

    try:
        import pytest
        all_passed &= print_check(True, "pytest is installed")

        # Check test files can be parsed
        test_files = [
            'test_moe_internal_logging.py',
            'test_moe_metrics.py',
            'test_moe_visualization.py'
        ]

        for test_file in test_files:
            if Path(test_file).exists():
                # Try to import the test module to check for syntax errors
                try:
                    module_name = test_file.replace('.py', '')
                    if module_name in sys.modules:
                        importlib.reload(sys.modules[module_name])
                    else:
                        __import__(module_name)
                    all_passed &= print_check(True, f"Test file parseable: {test_file}")
                except Exception as e:
                    all_passed &= print_check(False, f"Test file has errors: {test_file} - {e}")
            else:
                all_passed &= print_check(False, f"Test file missing: {test_file}")

    except ImportError:
        all_passed &= print_check(False, "pytest not installed (optional)")

    return all_passed


def verify_documentation():
    """Verify documentation is complete."""
    print_header("VERIFYING DOCUMENTATION")

    all_passed = True

    doc_files = {
        'NOTEBOOK_ENHANCEMENT_IMPLEMENTATION.md': 'Notebook enhancement guide',
        'ENHANCED_FRAMEWORK_INTEGRATION_GUIDE.md': 'Integration guide',
        'UNIFIED_ENHANCEMENT_IMPLEMENTATION_COMPLETE.md': 'Implementation summary'
    }

    for file, desc in doc_files.items():
        if Path(file).exists():
            size = Path(file).stat().st_size
            all_passed &= print_check(
                size > 1000,
                f"{desc}: {size:,} bytes"
            )
        else:
            all_passed &= print_check(False, f"Documentation missing: {file}")

    return all_passed


def verify_module_completeness():
    """Verify modules have all expected functions/classes."""
    print_header("VERIFYING MODULE COMPLETENESS")

    all_passed = True

    # Check moe_internal_logging
    try:
        import moe_internal_logging as mil
        expected = ['RouterLogger', 'InternalRoutingLogger']
        for name in expected:
            has_it = hasattr(mil, name)
            all_passed &= print_check(has_it, f"moe_internal_logging.{name}")
    except Exception as e:
        all_passed &= print_check(False, f"moe_internal_logging check failed: {e}")

    # Check moe_metrics
    try:
        import moe_metrics as mm
        expected = ['MetricsComputer', 'ComprehensiveMetrics']
        for name in expected:
            has_it = hasattr(mm, name)
            all_passed &= print_check(has_it, f"moe_metrics.{name}")

        # Check MetricsComputer has key methods
        methods = ['compute_perplexity', 'compute_token_accuracy', 'compute_routing_stats']
        for method in methods:
            has_it = hasattr(mm.MetricsComputer, method)
            all_passed &= print_check(has_it, f"MetricsComputer.{method}")

    except Exception as e:
        all_passed &= print_check(False, f"moe_metrics check failed: {e}")

    # Check moe_visualization
    try:
        import moe_visualization as mv
        expected = [
            'create_comprehensive_dashboard',
            'plot_per_layer_routing',
            'plot_expert_usage_heatmap',
            'plot_bh_vs_hc_comparison'
        ]
        for name in expected:
            has_it = hasattr(mv, name)
            all_passed &= print_check(has_it, f"moe_visualization.{name}")
    except Exception as e:
        all_passed &= print_check(False, f"moe_visualization check failed: {e}")

    return all_passed


def print_summary(results):
    """Print verification summary."""
    print_header("VERIFICATION SUMMARY")

    total = len(results)
    passed = sum(results.values())

    print(f"\nTotal Checks: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"\nSuccess Rate: {passed/total*100:.1f}%")

    if all(results.values()):
        print("\n🎉 ALL VERIFICATIONS PASSED!")
        print("\n✅ Framework is ready for use")
        print("\nNext steps:")
        print("  1. Apply notebook enhancements from NOTEBOOK_ENHANCEMENT_IMPLEMENTATION.md")
        print("  2. Run tests: pytest test_moe_*.py -v")
        print("  3. Execute notebooks with enhanced framework")
    else:
        print("\n⚠️ SOME VERIFICATIONS FAILED")
        print("\nFailed checks:")
        for check, passed in results.items():
            if not passed:
                print(f"  ❌ {check}")
        print("\nPlease review and fix issues before proceeding")

    print("\n" + "=" * 70)


def main():
    """Run all verifications."""
    print_header("BH & HC ROUTING ENHANCEMENT VERIFICATION")
    print(f"\nWorking directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")

    results = {}

    # Run verification sections
    results['Files Exist'] = verify_files_exist()
    results['Imports Work'] = verify_imports()
    results['Classes Instantiate'] = verify_class_instantiation()
    results['Functions Work'] = verify_function_calls()
    results['Tests Runnable'] = verify_tests_runnable()
    results['Documentation Complete'] = verify_documentation()
    results['Modules Complete'] = verify_module_completeness()

    # Print summary
    print_summary(results)

    # Exit code
    sys.exit(0 if all(results.values()) else 1)


if __name__ == '__main__':
    main()
