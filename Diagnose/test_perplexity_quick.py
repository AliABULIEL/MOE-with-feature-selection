#!/usr/bin/env python3
"""
Quick Perplexity Comparison Test
================================

Tests if β fix improved perplexity on a small sample.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hc_routing import higher_criticism_routing


def compute_simulated_perplexity(routing_weights, router_logits):
    """
    Simulate perplexity based on routing quality.

    Good routing should:
    - Select high-logit experts
    - Give them high weights
    - This correlates with low cross-entropy / perplexity
    """
    # Get indices of selected experts (where weight > 0)
    selected_mask = routing_weights > 0

    # Compute average logit of selected experts (weighted)
    selected_logits = router_logits * selected_mask.float()
    avg_selected_logit = (selected_logits * routing_weights).sum(dim=-1).mean()

    # Higher average logit → better routing → lower perplexity
    # Simplified metric for quick validation
    simulated_loss = -avg_selected_logit.item()
    simulated_ppl = torch.exp(torch.tensor(simulated_loss)).item()

    return simulated_ppl, avg_selected_logit.item()


def test_perplexity_proxy():
    """
    Test if different β values affect routing quality.

    We expect:
    - Some β values should produce better routing than others
    - The fix allows finding this optimal β
    - Before fix: all β were identical
    """
    print("=" * 70)
    print("PERPLEXITY PROXY TEST")
    print("=" * 70)
    print("\nSimulated test using routing quality metrics")
    print("(Not actual model perplexity - that requires full model)")

    torch.manual_seed(42)
    router_logits = torch.randn(1, 100, 64)

    # Add some structure: make certain experts consistently better
    # Experts 0-7 get higher logits (simulate being "good" experts)
    router_logits[:, :, :8] += 1.5

    beta_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    results = []

    print(f"\n{'β':>6} | {'Avg Logit':>12} | {'Sim PPL':>10} | {'Avg Experts':>12}")
    print("-" * 60)

    for beta in beta_values:
        routing_weights, _, expert_counts, _ = higher_criticism_routing(
            router_logits,
            beta=beta,
            min_k=4,
            max_k=16,
            temperature=1.0,
            kde_models=None,
            return_stats=False
        )

        sim_ppl, avg_logit = compute_simulated_perplexity(routing_weights, router_logits)
        avg_experts = expert_counts.float().mean().item()

        results.append({
            'beta': beta,
            'avg_logit': avg_logit,
            'sim_ppl': sim_ppl,
            'avg_experts': avg_experts
        })

        print(f"{beta:6.1f} | {avg_logit:12.4f} | {sim_ppl:10.2f} | {avg_experts:12.2f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Find best β (highest avg logit = best routing)
    best_result = max(results, key=lambda r: r['avg_logit'])
    worst_result = min(results, key=lambda r: r['avg_logit'])

    print(f"\nBest β: {best_result['beta']:.1f}")
    print(f"  Avg logit: {best_result['avg_logit']:.4f}")
    print(f"  Avg experts: {best_result['avg_experts']:.1f}")

    print(f"\nWorst β: {worst_result['beta']:.1f}")
    print(f"  Avg logit: {worst_result['avg_logit']:.4f}")
    print(f"  Avg experts: {worst_result['avg_experts']:.1f}")

    # Check variation
    logit_values = [r['avg_logit'] for r in results]
    variation = max(logit_values) - min(logit_values)

    print(f"\nVariation in avg_logit: {variation:.4f}")

    if variation > 0.1:
        print("✅ GOOD: β values produce different routing quality")
        print("   Different β find different optimal thresholds")
    else:
        print("⚠️  WARNING: Limited variation - may still have issues")

    # Expected pattern: optimal β somewhere in middle range
    # Too low β (0.1-0.2): may miss good experts
    # Too high β (0.8-1.0): may include noisy experts
    # Middle β (0.3-0.6): balanced search

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("\nThis test confirms β now controls routing behavior.")
    print("For actual perplexity, run full model evaluation with:")
    print("  - Real OLMoE model")
    print("  - WikiText or other dataset")
    print("  - Proper text generation")

    return variation > 0.1


if __name__ == '__main__':
    try:
        success = test_perplexity_proxy()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
