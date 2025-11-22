#!/usr/bin/env python3
"""
Test script to verify the hook-based routing fix works correctly.

This script runs a minimal experiment with:
- 4 experts
- Baseline strategy
- 10 samples from wikitext

Usage:
    python test_hook_fix.py
"""

import sys
import logging
from olmoe_routing_experiments import RoutingExperimentRunner

# Configure logging to see detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_hook_based_routing():
    """Test the hook-based routing approach."""
    print("=" * 70)
    print("Testing Hook-Based Routing Fix")
    print("=" * 70)

    try:
        # Create experiment runner
        print("\n1. Initializing RoutingExperimentRunner...")
        runner = RoutingExperimentRunner(
            device='auto',
            output_dir='./test_routing_experiments'
        )
        print("   ✅ Runner initialized successfully")

        # Run minimal experiment
        print("\n2. Running minimal experiment...")
        print("   Configuration:")
        print("     - Expert counts: [4]")
        print("     - Strategies: ['baseline']")
        print("     - Datasets: ['wikitext']")
        print("     - Max samples: 10")
        print()

        results = runner.run_all_experiments(
            expert_counts=[4],
            strategies=['baseline'],
            datasets=['wikitext'],
            max_samples=10
        )

        print("\n3. Experiment completed!")
        print("=" * 70)
        print("RESULTS:")
        print("=" * 70)
        print(results)
        print("=" * 70)

        # Check if samples were processed successfully
        if len(results) > 0:
            result = results.iloc[0]
            num_samples = result['num_samples']

            if num_samples > 0:
                print(f"\n✅ SUCCESS! Processed {num_samples} samples")
                print(f"   Perplexity: {result['perplexity']:.4f}")
                print(f"   Token Accuracy: {result['token_accuracy']:.4f}")
                print(f"   Tokens/sec: {result['tokens_per_second']:.2f}")
                return True
            else:
                print("\n❌ FAILED! Zero samples processed")
                print("   Check the error logs above for details")
                return False
        else:
            print("\n❌ FAILED! No results returned")
            return False

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {type(e).__name__}")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print(__doc__)
    success = test_hook_based_routing()
    sys.exit(0 if success else 1)
