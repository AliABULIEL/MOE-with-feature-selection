#!/usr/bin/env python3
"""
Diagnostic for log p-values in BH routing.
"""

import torch
import torch.nn.functional as F

# Create test logits
torch.manual_seed(42)
num_tokens = 1
num_experts = 64

router_logits = torch.randn(num_tokens, num_experts)
router_logits[:, :8] += 2.0  # Boost top 8
router_logits[:, :3] += 1.0  # Boost top 3 even more

print("=" * 70)
print("LOG P-VALUE DIAGNOSTIC")
print("=" * 70)

temperature = 1.0
alpha = 0.05

scaled_logits = router_logits / temperature

# Compute log p-values using log_softmax on NEGATIVE logits
log_p_values = F.log_softmax(-scaled_logits, dim=-1)

# Sort
sorted_log_p, sorted_indices = torch.sort(log_p_values, dim=-1)

# Compute BH thresholds
k_values = torch.arange(1, num_experts + 1, dtype=torch.float32)
log_N = torch.log(torch.tensor(num_experts, dtype=torch.float32))
log_alpha = torch.log(torch.tensor(alpha, dtype=torch.float32))
log_critical_values = torch.log(k_values) - log_N + log_alpha

# Convert to regular p-values for display
p_values = torch.exp(log_p_values[0])
sorted_p_values = torch.exp(sorted_log_p[0])
critical_values = torch.exp(log_critical_values)

print(f"\nAlpha: {alpha}")
print(f"log(alpha): {log_alpha.item():.6f}")
print(f"log(N): {log_N.item():.6f}")

print(f"\nTop 15 experts by log p-value:")
print(f"{'Rank':<6} {'Expert':<8} {'Logit':<10} {'log(p)':<12} {'p-value':<12} {'log(c_k)':<12} {'c_k':<12} {'Pass?':<6}")
print("-" * 90)

for k in range(15):
    expert_idx = sorted_indices[0, k].item()
    logit = scaled_logits[0, expert_idx].item()
    log_p = sorted_log_p[0, k].item()
    p_val = sorted_p_values[k].item()
    log_c = log_critical_values[k].item()
    c_val = critical_values[k].item()
    passes = "✅" if log_p <= log_c else "❌"

    print(f"{k+1:<6} {expert_idx:<8} {logit:<10.4f} {log_p:<12.6f} {p_val:<12.6e} {log_c:<12.6f} {c_val:<12.6e} {passes}")

# Count how many pass
passes_threshold = sorted_log_p[0] <= log_critical_values
num_pass = passes_threshold.sum().item()

print(f"\n{'='*70}")
print(f"Number of experts that pass threshold: {num_pass}/{num_experts}")
print(f"{'='*70}")

print(f"\nPROBLEM IDENTIFIED:")
print(f"When we use log_softmax(-logits), ALL p-values are very small")
print(f"because log_softmax normalizes to log-probabilities.")
print(f"\nThe best expert has p ≈ {sorted_p_values[0].item():.6e}")
print(f"The 30th expert has p ≈ {sorted_p_values[29].item():.6e}")
print(f"The 60th expert has p ≈ {sorted_p_values[59].item():.6e}")

print(f"\nBH thresholds are:")
print(f"c_1 = α/N = {critical_values[0].item():.6e}")
print(f"c_30 = 30α/N = {critical_values[29].item():.6e}")
print(f"c_60 = 60α/N = {critical_values[59].item():.6e}")

print(f"\nSince log_softmax creates a proper probability distribution,")
print(f"even the 'worst' experts get non-trivial probability mass,")
print(f"leading to small p-values that pass the BH threshold.")

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")
print("Log p-values from log_softmax(-logits) are TOO SMALL.")
print("Almost all experts pass the BH threshold.")
print("This is NOT the desired behavior for adaptive expert selection.")
