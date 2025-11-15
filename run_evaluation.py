#!/usr/bin/env python3
"""
Simple runner script for testing the fixed OLMoE evaluation.

This script runs a quick test with fewer samples to verify the fix works.
"""

from olmoe_evaluation import OLMoEEvaluator, EvaluationConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("="*80)
    print("QUICK TEST - OLMoE Evaluation with Fixed Expert Configuration")
    print("="*80)
    print("\nThis will run a quick test with:")
    print("  - Expert configs: 8, 16")
    print("  - Dataset: wikitext only")
    print("  - Samples: 50 (very fast!)")
    print()

    # Quick test configuration
    config = EvaluationConfig(
        expert_configs=[8, 16],  # Just test two configs
        datasets=['wikitext'],    # Single dataset
        max_samples=50,           # Very fast
        max_length=256,           # Shorter sequences
        output_dir="./quick_test_results"
    )

    # Create evaluator (will test expert modification works)
    logger.info("Creating evaluator...")
    evaluator = OLMoEEvaluator(config)

    # Run evaluation
    logger.info("Running evaluation...")
    results_df = evaluator.evaluate_all_configurations()

    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(results_df[['num_experts', 'perplexity', 'token_accuracy', 'tokens_per_second']])
    print("="*80)

    # Validate
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)

    ppl_8 = results_df[results_df['num_experts'] == 8]['perplexity'].values[0]
    ppl_16 = results_df[results_df['num_experts'] == 16]['perplexity'].values[0]
    ppl_diff = abs(ppl_8 - ppl_16)

    print(f"Perplexity with 8 experts:  {ppl_8:.4f}")
    print(f"Perplexity with 16 experts: {ppl_16:.4f}")
    print(f"Difference: {ppl_diff:.4f}")

    if ppl_diff < 0.01:
        print("\n❌ FAILED: Perplexities are too similar!")
        print("   Expert configuration may not be working.")
        return False
    else:
        print("\n✅ PASSED: Perplexities differ significantly!")
        print("   Expert configuration is working correctly!")
        return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
