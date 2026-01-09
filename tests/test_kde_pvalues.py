#!/usr/bin/env python3
"""
Test KDE-based p-values to verify they're properly calibrated.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deprecated.bh_routing import benjamini_hochberg_routing, load_kde_models

print("=" * 70)
print("KDE P-VALUE DIAGNOSTIC")
print("=" * 70)

# Load KDE models
kde_models = load_kde_models()

if not kde_models:
    print("❌ No KDE models found!")
    sys.exit(1)

# Create test logits
torch.manual_seed(42)
num_tokens = 10
num_experts = 64

router_logits = torch.randn(num_tokens, num_experts)
router_logits[:, :8] += 2.0  # Boost top 8
router_logits[:, :3] += 1.0  # Boost top 3 even more

print(f"\nTest setup: {num_tokens} tokens, {num_experts} experts")
print("Top 8 experts have boosted logits\n")

# Test with alpha=0.05
alpha = 0.05
layer_idx = 0

routing_weights, selected_experts, expert_counts, stats = benjamini_hochberg_routing(
    router_logits,
    alpha=alpha,
    layer_idx=layer_idx,
    kde_models=kde_models,
    return_stats=True
)

p_values = stats['p_values']
critical_values = stats['critical_values']

print(f"Alpha: {alpha}")
print(f"Layer: {layer_idx}")
print(f"KDE available: {stats['kde_available']}")
print(f"\nAverage experts selected: {expert_counts.float().mean().item():.2f}")
print(f"Range: [{expert_counts.min().item()}, {expert_counts.max().item()}]")

# Analyze p-values for first token
print(f"\n{'-'*70}")
print("First Token Analysis")
print(f"{'-'*70}")

token_logits = router_logits[0]
token_pvals = p_values[0]

# Sort by p-value
sorted_pvals, sorted_idx = torch.sort(token_pvals)

print(f"\n{'Rank':<6} {'Expert':<8} {'Logit':<10} {'P-value':<12} {'BH Thresh':<12} {'Selected':<10}")
print("-" * 70)

for rank in range(15):
    expert_idx = sorted_idx[rank].item()
    logit = token_logits[expert_idx].item()
    p_val = sorted_pvals[rank].item()
    thresh = critical_values[rank].item()
    selected = "✅" if p_val <= thresh else "❌"

    print(f"{rank+1:<6} {expert_idx:<8} {logit:<10.4f} {p_val:<12.6f} {thresh:<12.6f} {selected}")

# Show which experts are in top 8 by logit
print(f"\n{'-'*70}")
print("Top 8 Experts by Logit (should have low p-values)")
print(f"{'-'*70}")

top_logits, top_indices = torch.topk(token_logits, k=8)
print(f"\n{'Rank':<6} {'Expert':<8} {'Logit':<10} {'P-value':<12}")
print("-" * 60)

for rank, (logit, expert_idx) in enumerate(zip(top_logits, top_indices)):
    p_val = token_pvals[expert_idx].item()
    print(f"{rank+1:<6} {expert_idx.item():<8} {logit.item():<10.4f} {p_val:<12.6f}")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

print(f"\n📊 P-value distribution:")
print(f"  Min: {token_pvals.min().item():.6f}")
print(f"  Median: {token_pvals.median().item():.6f}")
print(f"  Max: {token_pvals.max().item():.6f}")

# Count how many pass threshold
num_pass = (sorted_pvals <= critical_values).sum().item()
print(f"\n📊 BH procedure:")
print(f"  Experts passing threshold: {num_pass}/{num_experts}")
print(f"  With max_k=8 constraint: {min(num_pass, 8)}/8 selected")

# Check if top logit experts have low p-values
top8_pvals = [token_pvals[idx].item() for idx in top_indices]
avg_top8_pval = sum(top8_pvals) / len(top8_pvals)
print(f"\n📊 Top-8 by logit have avg p-value: {avg_top8_pval:.6f}")

if avg_top8_pval < 0.1:
    print("  ✅ Top experts have low p-values (good calibration)")
else:
    print("  ⚠️  Top experts have high p-values (calibration issue)")

print(f"\n{'='*70}")
