#!/usr/bin/env python3
"""
HC Routing Bug Diagnostic Script
=================================

Identifies why HC routing produces worse results than TopK
by comparing expert selections between the two methods.

Key Questions to Answer:
1. Are HC-selected experts the same as TopK-selected experts?
2. Are p-values correctly mapped (high logit → low p-value)?
3. Are weights correctly computed and normalized?
4. Is the selection mask correctly mapped from sorted to original order?
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


def analyze_expert_selection(
        router_logits: torch.Tensor,
        hc_selected: torch.Tensor,
        hc_weights: torch.Tensor,
        topk_k: int = 8
) -> Dict:
    """
    Compare HC selection with TopK selection.

    Args:
        router_logits: [num_tokens, 64] router outputs
        hc_selected: [num_tokens, max_k] HC-selected expert indices
        hc_weights: [num_tokens, 64] HC routing weights
        topk_k: K value for comparison

    Returns:
        Diagnostic report
    """
    num_tokens, num_experts = router_logits.shape

    # Get TopK selection
    topk_weights, topk_indices = torch.topk(router_logits, topk_k, dim=-1)

    results = {
        'overlap_analysis': [],
        'logit_analysis': [],
        'weight_analysis': [],
        'p_value_analysis': []
    }

    for t in range(min(num_tokens, 10)):  # Analyze first 10 tokens
        token_results = {}

        # 1. Get HC selected experts (non-negative indices)
        hc_experts = hc_selected[t][hc_selected[t] >= 0].tolist()
        topk_experts = topk_indices[t].tolist()

        # 2. Compute overlap
        hc_set = set(hc_experts)
        topk_set = set(topk_experts)
        overlap = hc_set & topk_set

        token_results['hc_count'] = len(hc_experts)
        token_results['topk_count'] = topk_k
        token_results['overlap_count'] = len(overlap)
        token_results['overlap_pct'] = len(overlap) / topk_k * 100 if topk_k > 0 else 0

        # 3. Compare logit ranks
        # Sort experts by logit (descending)
        sorted_indices = torch.argsort(router_logits[t], descending=True)
        expert_ranks = {idx.item(): rank for rank, idx in enumerate(sorted_indices)}

        # Get ranks of HC-selected experts
        hc_ranks = [expert_ranks[e] for e in hc_experts]
        topk_ranks = [expert_ranks[e] for e in topk_experts]

        token_results['hc_avg_rank'] = np.mean(hc_ranks) if hc_ranks else -1
        token_results['topk_avg_rank'] = np.mean(topk_ranks)  # Should be ~3.5 for k=8
        token_results['hc_ranks'] = hc_ranks[:5]  # First 5
        token_results['topk_ranks'] = topk_ranks[:5]

        # 4. Check if HC is selecting high-rank (bad) or low-rank (good) experts
        # If hc_avg_rank >> topk_avg_rank, HC is selecting WORSE experts!
        if token_results['hc_avg_rank'] > token_results['topk_avg_rank'] * 2:
            token_results['diagnosis'] = '🚨 HC selecting WORSE experts!'
        elif token_results['overlap_pct'] > 70:
            token_results['diagnosis'] = '✅ HC selecting similar experts'
        else:
            token_results['diagnosis'] = '⚠️ HC selecting different experts'

        # 5. Check weight normalization
        hc_weight_sum = hc_weights[t].sum().item()
        token_results['hc_weight_sum'] = hc_weight_sum

        results['overlap_analysis'].append(token_results)

    return results


def verify_pvalue_direction(router_logits: torch.Tensor) -> Dict:
    """
    Verify p-values are correctly computed.

    Correct behavior:
    - High logit → low p-value → likely selected
    - Low logit → high p-value → likely rejected

    Bug if inverted:
    - High logit → high p-value → rejected (WRONG!)
    """
    num_tokens, num_experts = router_logits.shape

    # Compute p-values using empirical ranking
    ranks = torch.argsort(torch.argsort(router_logits, dim=1, descending=True), dim=1)
    p_values = (ranks.float() + 1) / (num_experts + 1)

    results = []

    for t in range(min(num_tokens, 5)):
        logits = router_logits[t]
        pvals = p_values[t]

        # Get top-5 by logit
        top_logit_indices = torch.argsort(logits, descending=True)[:5]

        # Get top-5 by p-value (should be SAME as top logit if correct)
        top_pval_indices = torch.argsort(pvals, descending=False)[:5]  # Low p-value = good

        # Check correlation
        logit_order_pvals = pvals[top_logit_indices].tolist()

        result = {
            'top_logit_experts': top_logit_indices.tolist(),
            'top_pval_experts': top_pval_indices.tolist(),
            'pvals_of_top_logits': logit_order_pvals,
            'match': top_logit_indices.tolist() == top_pval_indices.tolist()
        }

        # Diagnosis
        if logit_order_pvals[0] < 0.1:  # Top logit has low p-value
            result['diagnosis'] = '✅ P-values correctly oriented'
        else:
            result['diagnosis'] = '🚨 P-values INVERTED - selecting wrong experts!'

        results.append(result)

    return results


def check_scatter_operation(num_experts: int = 64) -> Dict:
    """
    Verify scatter operation for mapping sorted → original indices.
    """
    # Simulate p-values and sorting
    torch.manual_seed(42)
    p_values = torch.rand(1, num_experts)

    # Sort ascending (lowest p-values first)
    p_sorted, sort_indices = torch.sort(p_values, dim=1)

    # Create mask in sorted order (select first 8)
    k = 8
    mask_sorted = torch.zeros(1, num_experts, dtype=torch.bool)
    mask_sorted[:, :k] = True

    # Method 1: scatter (current implementation)
    mask_original_scatter = torch.zeros_like(mask_sorted)
    mask_original_scatter.scatter_(1, sort_indices, mask_sorted)

    # Method 2: Direct indexing (correct)
    mask_original_direct = torch.zeros_like(mask_sorted)
    selected_indices = sort_indices[:, :k]
    for i in range(k):
        mask_original_direct[0, selected_indices[0, i]] = True

    # Check if they match
    match = torch.equal(mask_original_scatter, mask_original_direct)

    # Verify selected experts are the ones with lowest p-values
    selected_experts_scatter = torch.where(mask_original_scatter[0])[0].tolist()
    selected_experts_direct = torch.where(mask_original_direct[0])[0].tolist()

    # Check p-values of selected experts
    selected_pvals_scatter = p_values[0, selected_experts_scatter].tolist()
    selected_pvals_direct = p_values[0, selected_experts_direct].tolist()

    return {
        'scatter_matches_direct': match,
        'selected_by_scatter': selected_experts_scatter,
        'selected_by_direct': selected_experts_direct,
        'selected_pvals_scatter': selected_pvals_scatter,
        'selected_pvals_direct': selected_pvals_direct,
        'diagnosis': '✅ Scatter correct' if match else '🚨 Scatter bug!'
    }


def diagnose_hc_routing(
        router_logits: torch.Tensor,
        hc_weights: torch.Tensor,
        hc_selected: torch.Tensor,
        hc_counts: torch.Tensor
):
    """
    Full diagnostic report.
    """
    print("=" * 70)
    print("HC ROUTING BUG DIAGNOSTIC REPORT")
    print("=" * 70)

    # 1. P-value direction check
    print("\n1. P-VALUE DIRECTION CHECK")
    print("-" * 50)
    pval_results = verify_pvalue_direction(router_logits)
    for i, r in enumerate(pval_results[:3]):
        print(f"   Token {i}: {r['diagnosis']}")
        print(f"      Top logit experts: {r['top_logit_experts'][:5]}")
        print(f"      Top p-val experts:  {r['top_pval_experts'][:5]}")
        print(f"      Match: {r['match']}")

    # 2. Scatter operation check
    print("\n2. SCATTER OPERATION CHECK")
    print("-" * 50)
    scatter_result = check_scatter_operation()
    print(f"   {scatter_result['diagnosis']}")
    print(f"   Scatter == Direct: {scatter_result['scatter_matches_direct']}")

    # 3. Expert selection comparison
    print("\n3. HC vs TOPK EXPERT SELECTION")
    print("-" * 50)
    selection_results = analyze_expert_selection(
        router_logits, hc_selected, hc_weights, topk_k=8
    )

    for i, r in enumerate(selection_results['overlap_analysis'][:5]):
        print(f"   Token {i}: {r['diagnosis']}")
        print(f"      HC selected: {r['hc_count']} experts")
        print(f"      Overlap with TopK-8: {r['overlap_count']}/8 ({r['overlap_pct']:.0f}%)")
        print(f"      HC avg rank: {r['hc_avg_rank']:.1f} (lower=better, TopK-8 ≈ 3.5)")
        print(f"      Weight sum: {r['hc_weight_sum']:.4f}")

    # 4. Summary diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)

    avg_overlap = np.mean([r['overlap_pct'] for r in selection_results['overlap_analysis']])
    avg_hc_rank = np.mean([r['hc_avg_rank'] for r in selection_results['overlap_analysis']])

    print(f"   Average overlap with TopK-8: {avg_overlap:.1f}%")
    print(f"   Average HC expert rank: {avg_hc_rank:.1f}")

    if avg_hc_rank > 20:
        print("\n   🚨 CRITICAL: HC is selecting LOW-RANKED (BAD) experts!")
        print("      Root cause likely: P-value direction or selection mask inverted")
        print("      Fix: Check p-value computation or scatter operation")
    elif avg_overlap < 50:
        print("\n   ⚠️ WARNING: HC selecting different experts than TopK")
        print("      This may be expected, but quality suffers")
    else:
        print("\n   ✅ HC selection appears reasonable")
        print("      Perplexity issue may be due to OLMoE k=8 training")


# Standalone test
if __name__ == '__main__':
    print("Running HC routing diagnostics...")

    # Generate test data
    torch.manual_seed(42)
    router_logits = torch.randn(20, 64)

    # Simulate HC routing output
    # For testing, we'll create mock data
    from hc_routing import higher_criticism_routing

    hc_weights, hc_selected, hc_counts, _ = higher_criticism_routing(
        router_logits.unsqueeze(0),
        beta=0.5,
        min_k=4,
        max_k=16
    )

    diagnose_hc_routing(
        router_logits,
        hc_weights.squeeze(0),
        hc_selected.squeeze(0),
        hc_counts.squeeze(0)
    )