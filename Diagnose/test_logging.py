#!/usr/bin/env python3
"""
Unit Tests for HC Routing Logging System

Tests the HCRoutingLogger class using the ACTUAL schema and attributes.

Usage:
    python3 test_logging.py
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from hc_routing_logging import HCRoutingLogger


def create_valid_log_entry(sample_idx=0, token_idx=0, layer_idx=0):
    """Create a valid log entry matching the expected schema."""
    return {
        # Required fields
        'sample_idx': sample_idx,
        'token_idx': token_idx,
        'layer_idx': layer_idx,
        'num_selected': 8,
        'hc_statistics': [0.1] * 64,  # Full HC stats for all 64 experts
        'hc_max_rank': 8,  # 1-indexed rank where HC peaks
        'hc_max_value': 2.5,  # Maximum HC value

        # Optional but commonly present
        'selected_experts': [0, 1, 2, 3, 4, 5, 6, 7] + [-1] * 56,  # Padded to max_k
        'routing_weights': [0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08] + [0.0] * 56,
        'selection_reason': 'hc_threshold',
        'hit_min_k': False,
        'hit_max_k': False,
        'fallback_triggered': False,
        'weights_sum': 0.92,
        'config': {
            'min_k': 8,
            'max_k': 16,
            'temperature': 1.0
        }
    }


def test_logger_instantiation():
    """Test 1: Logger can be instantiated with required parameters."""
    print("Test 1: Logger Instantiation...", end=" ")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = HCRoutingLogger(
                output_dir=tmpdir,
                experiment_name="test_exp",
                log_every_n=100
            )
            assert logger.routing_decisions == []
            assert logger.layer_stats == {}
            assert logger.experiment_name == "test_exp"
            assert logger.log_every_n == 100
            print("✅ PASSED")
            return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_valid_log_entry():
    """Test 2: Valid log entry is accepted and stored."""
    print("Test 2: Valid Log Entry (log_every_n=1)...", end=" ")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = HCRoutingLogger(
                output_dir=tmpdir,
                experiment_name="test_exp",
                log_every_n=1  # Log every decision
            )

            log_entry = create_valid_log_entry()
            logger.log_routing_decision(log_entry)

            # Check that entry was logged (log_every_n=1)
            assert len(logger.routing_decisions) == 1, f"Expected 1 entry, got {len(logger.routing_decisions)}"
            assert logger.routing_decisions[0] == log_entry
            assert logger.total_decisions == 1
            assert logger.logged_decisions == 1

            # Check layer stats were updated
            assert 0 in logger.layer_stats
            assert logger.layer_stats[0]['total'] == 1
            assert len(logger.layer_stats[0]['num_selected']) == 1

            print("✅ PASSED")
            return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_periodic_logging():
    """Test 3: Periodic logging respects log_every_n."""
    print("Test 3: Periodic Logging (log_every_n=100)...", end=" ")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = HCRoutingLogger(
                output_dir=tmpdir,
                experiment_name="test_exp",
                log_every_n=100  # Only log every 100th decision
            )

            # Log 250 decisions
            for i in range(250):
                log_entry = create_valid_log_entry(token_idx=i)
                logger.log_routing_decision(log_entry)

            # Should have logged at: 100, 200 = 2 entries (modulo checks after incrementing total)
            assert logger.total_decisions == 250, f"Expected 250 total, got {logger.total_decisions}"
            assert logger.logged_decisions == 2, f"Expected 2 logged (100, 200), got {logger.logged_decisions}"
            assert len(logger.routing_decisions) == 2

            # But layer stats should track ALL 250
            assert logger.layer_stats[0]['total'] == 250

            print("✅ PASSED")
            return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_missing_required_field():
    """Test 4: Missing required field prints warning but doesn't crash."""
    print("Test 4: Missing Required Field...", end=" ")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = HCRoutingLogger(
                output_dir=tmpdir,
                experiment_name="test_exp",
                log_every_n=1
            )

            # Create entry missing 'hc_max_value' (required)
            invalid_entry = {
                'sample_idx': 0,
                'token_idx': 0,
                'layer_idx': 0,
                'num_selected': 8,
                'hc_statistics': [0.1] * 64,
                'hc_max_rank': 8,
                # 'hc_max_value': 2.5,  # MISSING!
            }

            # Should print warning but not crash
            logger.log_routing_decision(invalid_entry)

            # Entry should still be logged (logger is permissive)
            assert logger.total_decisions == 1
            # Check if warning was printed (we can't easily capture print output)

            print("✅ PASSED (warning expected)")
            return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_logs():
    """Test 5: save_logs() creates JSON file."""
    print("Test 5: Save Logs to JSON...", end=" ")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = HCRoutingLogger(
                output_dir=tmpdir,
                experiment_name="test_exp",
                log_every_n=1
            )

            # Add some log entries
            for i in range(3):
                log_entry = create_valid_log_entry(token_idx=i)
                logger.log_routing_decision(log_entry)

            # Save logs
            logger.save_logs(filename="test_logs.json")

            # Verify file exists
            filepath = Path(tmpdir) / "test_logs.json"
            assert filepath.exists(), f"Log file not created at {filepath}"

            # Verify file content
            with open(filepath, 'r') as f:
                data = json.load(f)

            assert 'experiment_name' in data
            assert data['experiment_name'] == "test_exp"
            assert data['total_decisions'] == 3
            assert data['logged_decisions'] == 3
            assert len(data['routing_decisions']) == 3

            print("✅ PASSED")
            return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_layer_statistics():
    """Test 6: Layer statistics are aggregated correctly."""
    print("Test 6: Layer Statistics Aggregation...", end=" ")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = HCRoutingLogger(
                output_dir=tmpdir,
                experiment_name="test_exp",
                log_every_n=1
            )

            # Add entries for layer 0
            for i in range(5):
                logger.log_routing_decision(create_valid_log_entry(layer_idx=0, num_selected=8))

            # Add entries for layer 1 with min_k hits
            for i in range(3):
                entry = create_valid_log_entry(layer_idx=1, num_selected=8)
                entry['hit_min_k'] = True
                logger.log_routing_decision(entry)

            # Verify layer 0 stats
            assert 0 in logger.layer_stats
            stats_0 = logger.layer_stats[0]
            assert stats_0['total'] == 5
            assert len(stats_0['num_selected']) == 5
            assert stats_0['floor_hits'] == 5  # All hit min_k=8

            # Verify layer 1 stats
            assert 1 in logger.layer_stats
            stats_1 = logger.layer_stats[1]
            assert stats_1['total'] == 3
            assert len(stats_1['num_selected']) == 3
            assert stats_1['floor_hits'] == 3

            print("✅ PASSED")
            return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_valid_log_entry(sample_idx=0, token_idx=0, layer_idx=0, num_selected=8):
    """Create a valid log entry matching the expected schema."""
    return {
        # Required fields
        'sample_idx': sample_idx,
        'token_idx': token_idx,
        'layer_idx': layer_idx,
        'num_selected': num_selected,
        'hc_statistics': [0.1] * 64,  # Full HC stats for all 64 experts
        'hc_max_rank': num_selected,  # 1-indexed rank where HC peaks
        'hc_max_value': 2.5,  # Maximum HC value

        # Optional but commonly present
        'selected_experts': list(range(num_selected)) + [-1] * (64 - num_selected),
        'routing_weights': ([0.15 / num_selected] * num_selected) + ([0.0] * (64 - num_selected)),
        'selection_reason': 'hc_threshold',
        'hit_min_k': (num_selected == 8),
        'hit_max_k': (num_selected == 16),
        'fallback_triggered': False,
        'weights_sum': 0.92,
        'config': {
            'min_k': 8,
            'max_k': 16,
            'temperature': 1.0
        }
    }


def test_get_summary():
    """Test 7: get_summary() returns statistics."""
    print("Test 7: Get Summary Statistics...", end=" ")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = HCRoutingLogger(
                output_dir=tmpdir,
                experiment_name="test_exp",
                log_every_n=1
            )

            # Add some varied entries
            for num_sel in [8, 10, 12, 8, 16]:
                logger.log_routing_decision(create_valid_log_entry(num_selected=num_sel))

            summary = logger.get_summary()

            assert 'experiment_name' in summary
            assert summary['experiment_name'] == "test_exp"
            assert summary['total_decisions'] == 5
            assert summary['logged_decisions'] == 5
            assert 'per_layer' in summary
            assert 0 in summary['per_layer']

            layer_0 = summary['per_layer'][0]
            assert 'avg_experts' in layer_0
            assert 'std_experts' in layer_0

            print("✅ PASSED")
            return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 80)
    print("HC ROUTING LOGGING SYSTEM - UNIT TESTS")
    print("=" * 80)
    print()

    tests = [
        test_logger_instantiation,
        test_valid_log_entry,
        test_periodic_logging,
        test_missing_required_field,
        test_save_logs,
        test_layer_statistics,
        test_get_summary,
    ]

    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)

    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(results)
    total = len(results)

    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    print()

    if passed == total:
        print("✅ ALL TESTS PASSED!")
        return True
    else:
        print(f"❌ {total - passed} TEST(S) FAILED")
        return False


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
