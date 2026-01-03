#!/usr/bin/env python3
"""
Verify Logging System Test

Minimal test to verify that HC routing logging works correctly.
This script will help identify why logging stopped working after recent fixes.

Usage:
    python3 verify_logging.py

Expected Output:
    ✅ Logger created
    ✅ Model loaded
    ✅ Model patched
    ✅ Forward pass completed
    ✅ Logs captured: X decisions
    ✅ Log file saved: <path>
    ✅ Schema validated
"""

import os
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from hc_routing_logging import HCRoutingLogger
from olmoe_hc_integration import HCRoutingIntegration


def verify_logging():
    """Run minimal logging verification test."""

    print("=" * 80)
    print("LOGGING VERIFICATION TEST")
    print("=" * 80)
    print()

    # Print environment info
    print("📍 Environment Info:")
    print(f"   Working Directory: {os.getcwd()}")
    print(f"   Script Directory: {Path(__file__).parent}")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    print()

    # Step 1: Create logger
    print("STEP 1: Creating HCRoutingLogger...")
    try:
        log_dir = "test_hc_logs"
        logger = HCRoutingLogger(
            output_dir=log_dir,
            experiment_name="verification_test",
            log_every_n=1  # Log every token for testing
        )
        log_dir_path = Path(log_dir).resolve()
        print(f"   ✅ Logger created successfully")
        print(f"   📁 Log directory: {log_dir_path}")
        print(f"   📝 Experiment: {logger.experiment_name}")
        print()
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False

    # Step 2: Load model and tokenizer
    print("STEP 2: Loading OLMoE model (this may take a minute)...")
    try:
        model_name = "allenai/OLMoE-1B-7B-0924"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"  # Use CPU for testing
        )
        print(f"   ✅ Model loaded: {model_name}")
        print(f"   🔧 Device: CPU (for testing)")
        print()
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        print(f"   💡 Make sure you have internet connection and HuggingFace access")
        return False

    # Step 3: Create HC integration
    print("STEP 3: Creating HC routing integration...")
    try:
        integration = HCRoutingIntegration(model)
        print(f"   ✅ Integration created")
        print()
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False

    # Step 4: Patch model with logger
    print("STEP 4: Patching model with HC routing and logger...")
    try:
        integration.patch_model(
            min_k=4,
            max_k=16,
            beta=0.5,
            logger=logger,  # ⚠️ CRITICAL: Pass logger here
            log_every_n=1   # ⚠️ CRITICAL: Must match logger.log_every_n!
        )
        print(f"   ✅ Model patched successfully")
        print(f"   📊 Config: beta=0.5, min_k=4, max_k=16")
        print(f"   📝 Logging enabled: log_every_n=1 (every token)")
        print()
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 5: Prepare test input
    print("STEP 5: Preparing test input...")
    try:
        test_text = "The quick brown fox jumps over the lazy dog."
        inputs = tokenizer(test_text, return_tensors="pt")
        num_tokens = inputs.input_ids.shape[1]
        print(f"   ✅ Test text tokenized")
        print(f"   📄 Text: \"{test_text}\"")
        print(f"   🔢 Tokens: {num_tokens}")
        print()
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False

    # Step 6: Run forward pass
    print("STEP 6: Running forward pass...")
    try:
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"   ✅ Forward pass completed")
        print(f"   📊 Output shape: {outputs.logits.shape}")
        print()
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 7: Check logs
    print("STEP 7: Checking captured logs...")
    try:
        num_logs = len(logger.routing_decisions)
        print(f"   📊 Total log entries: {num_logs}")

        if num_logs == 0:
            print(f"   ⚠️  WARNING: No logs captured!")
            print(f"   🔍 Possible reasons:")
            print(f"      - Logger not passed correctly")
            print(f"      - Conditional logging skipped all tokens")
            print(f"      - Exception occurred and was caught silently")
            print()
            return False
        else:
            print(f"   ✅ Logs captured successfully!")
            print()

            # Show first log entry
            print("   📋 First log entry sample:")
            first_log = logger.routing_decisions[0]
            print(f"      - sample_idx: {first_log.get('sample_idx')}")
            print(f"      - token_idx: {first_log.get('token_idx')}")
            print(f"      - layer_idx: {first_log.get('layer_idx')}")
            print(f"      - num_selected: {first_log.get('num_selected')}")
            print(f"      - hc_max_rank: {first_log.get('hc_max_rank')}")
            print(f"      - hc_max_value: {first_log.get('hc_max_value'):.6f}" if first_log.get('hc_max_value') is not None else "      - hc_max_value: None")
            print(f"      - selection_reason: {first_log.get('selection_reason')}")
            print()
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False

    # Step 8: Save logs
    print("STEP 8: Saving logs to file...")
    try:
        log_file = logger.save_logs(filename="verify_test_logs.json")
        if log_file:
            log_file_path = Path(log_file).resolve()
            print(f"   ✅ Logs saved successfully")
            print(f"   📁 File: {log_file_path}")

            # Check file size
            file_size = Path(log_file).stat().st_size
            print(f"   📊 File size: {file_size:,} bytes")
            print()
        else:
            print(f"   ⚠️  WARNING: No log file created")
            return False
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False

    # Step 9: Validate schema
    print("STEP 9: Validating log schema...")
    try:
        # These are the actual field names from hc_routing.py STEP 10
        required_fields = [
            'sample_idx', 'token_idx', 'layer_idx',
            'router_logits', 'router_logits_stats',
            'p_values', 'p_values_sorted', 'sort_indices',
            'hc_statistics', 'hc_max_rank', 'hc_max_value',
            'num_selected', 'selected_experts', 'routing_weights',
            'selection_reason', 'weights_sum', 'config'
        ]

        schema_valid = True
        for i, log_entry in enumerate(logger.routing_decisions):
            missing_fields = [f for f in required_fields if f not in log_entry]
            if missing_fields:
                print(f"   ❌ Log entry {i} missing fields: {missing_fields}")
                schema_valid = False

        if schema_valid:
            print(f"   ✅ All {num_logs} log entries have valid schema")
            print(f"   📋 Key fields present:")
            key_fields = ['sample_idx', 'layer_idx', 'num_selected', 'hc_statistics',
                         'routing_weights', 'selection_reason']
            for field in key_fields:
                print(f"      ✓ {field}")
            print()
        else:
            print(f"   ❌ Schema validation failed")
            return False
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False

    # Step 10: Layer statistics
    print("STEP 10: Checking layer statistics...")
    try:
        num_layers = len(logger.layer_stats)
        print(f"   📊 Layers logged: {num_layers}")

        for layer_idx in sorted(logger.layer_stats.keys()):
            stats = logger.layer_stats[layer_idx]
            print(f"      Layer {layer_idx}:")
            print(f"         - Selections: {stats['num_selections']}")
            print(f"         - Avg experts: {stats['total_experts_selected'] / stats['num_selections']:.2f}")
            print(f"         - Fallbacks: {stats['num_fallbacks']}")
        print()
    except Exception as e:
        print(f"   ⚠️  Warning: {e}")

    # Final summary
    print("=" * 80)
    print("✅ LOGGING VERIFICATION TEST PASSED!")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  ✅ Logger created and functional")
    print(f"  ✅ Model patched with HC routing")
    print(f"  ✅ Forward pass completed")
    print(f"  ✅ {num_logs} log entries captured")
    print(f"  ✅ Logs saved to: {log_file_path}")
    print(f"  ✅ Schema validated")
    print(f"  ✅ {num_layers} layers logged")
    print()
    print("🎉 Logging system is WORKING correctly!")
    print()

    return True


if __name__ == "__main__":
    try:
        success = verify_logging()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
