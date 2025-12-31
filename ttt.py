# BETA EFFECT TEST - With proper max_k
import torch

# Force reimport
import sys

for mod in list(sys.modules.keys()):
    if 'hc_routing' in mod or 'bh_routing' in mod:
        del sys.modules[mod]

from hc_routing import higher_criticism_routing, load_kde_models

kde_models = load_kde_models()
torch.manual_seed(42)
logits = torch.randn(100, 64)

print("=" * 70)
print("FULL BETA EFFECT TEST (max_k=64 - NO CEILING)")
print("=" * 70)
print("\nBeta  | Avg Experts | Std  | Min | Max | Ceiling Hit?")
print("-" * 70)

for beta in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    weights, experts, counts, _ = higher_criticism_routing(
        logits,
        beta=beta,
        min_k=4,
        max_k=64,  # ← NO CEILING - see true beta effect
        kde_models=kde_models
    )

    avg = counts.float().mean().item()
    std = counts.float().std().item()
    max_c = counts.max().item()
    search_range = int(64 * beta)
    ceiling_hit = "YES ⚠️" if max_c == 64 else "NO ✅"

    print(f" {beta:4} |   {avg:6.2f}    | {std:4.2f} |  {counts.min().item()}  |  {max_c:2}  | {ceiling_hit}")

print("=" * 70)
