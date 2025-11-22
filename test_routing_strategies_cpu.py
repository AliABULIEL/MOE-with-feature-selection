#!/usr/bin/env python3
"""
CPU Test: Verify Routing Strategies Actually Modify Computation
================================================================

This test proves that uniform and normalized routing strategies
produce DIFFERENT results (not identical like the current bug).

Run on CPU, single sample, very fast (~1 minute).
"""

import torch
from transformers import OlmoeForCausalLM, AutoTokenizer
from olmoe_routing_experiments import ModelPatchingUtils
import sys

print("="*70)
print("ROUTING STRATEGIES CPU TEST")
print("="*70)

# Force CPU
device = 'cpu'
print(f"\n Using device: {device}")

# Load tiny model on CPU
print("\n1. Loading model on CPU...")
model = OlmoeForCausalLM.from_pretrained(
    "allenai/OLMoE-1B-7B-0924",
    torch_dtype=torch.float32,  # CPU doesn't support bfloat16
    device_map=device
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(" Model loaded")

# Test sample
test_text = "The quick brown fox jumps over the lazy dog. This is a test sentence."
print(f"\n2. Test text: '{test_text}'")

# Tokenize
inputs = tokenizer(test_text, return_tensors='pt', truncation=True, max_length=50)
input_ids = inputs['input_ids'].to(device)
print(f" Tokenized: {input_ids.shape[1]} tokens")

# Store original forwards for restoration
original_forwards = {}
for idx, layer in enumerate(model.model.layers):
    if hasattr(layer, 'mlp'):
        original_forwards[idx] = layer.mlp.forward

# Test configurations
num_experts = 8
results = {}

print(f"\n3. Testing with {num_experts} experts:")
print("-"*70)

# BASELINE: No patching
print("\n[1/3] Testing BASELINE routing...")
model.config.num_experts_per_tok = num_experts

# Restore original forwards first (in case previous run patched them)
ModelPatchingUtils.unpatch_model(model, original_forwards)

with torch.no_grad():
    outputs_baseline = model(input_ids=input_ids, labels=input_ids)
    loss_baseline = outputs_baseline.loss.item()
    ppl_baseline = torch.exp(outputs_baseline.loss).item()

results['baseline'] = {'loss': loss_baseline, 'perplexity': ppl_baseline}
print(f"  Loss: {loss_baseline:.4f}")
print(f"  Perplexity: {ppl_baseline:.4f}")

# UNIFORM: Patch with uniform weights
print("\n[2/3] Testing UNIFORM routing...")
ModelPatchingUtils.patch_model(model, top_k=num_experts, strategy='uniform')

with torch.no_grad():
    outputs_uniform = model(input_ids=input_ids, labels=input_ids)
    loss_uniform = outputs_uniform.loss.item()
    ppl_uniform = torch.exp(outputs_uniform.loss).item()

results['uniform'] = {'loss': loss_uniform, 'perplexity': ppl_uniform}
print(f"  Loss: {loss_uniform:.4f}")
print(f"  Perplexity: {ppl_uniform:.4f}")

# NORMALIZED: Patch with normalized weights
print("\n[3/3] Testing NORMALIZED routing...")
ModelPatchingUtils.patch_model(model, top_k=num_experts, strategy='normalized')

with torch.no_grad():
    outputs_normalized = model(input_ids=input_ids, labels=input_ids)
    loss_normalized = outputs_normalized.loss.item()
    ppl_normalized = torch.exp(outputs_normalized.loss).item()

results['normalized'] = {'loss': loss_normalized, 'perplexity': ppl_normalized}
print(f"  Loss: {loss_normalized:.4f}")
print(f"  Perplexity: {ppl_normalized:.4f}")

# Restore original
ModelPatchingUtils.unpatch_model(model, original_forwards)

# VERIFICATION
print("\n" + "="*70)
print("VERIFICATION")
print("="*70)

print("\nResults Summary:")
for strategy, metrics in results.items():
    print(f"  {strategy:12s}: Loss={metrics['loss']:.4f}, PPL={metrics['perplexity']:.4f}")

print("\nChecking if strategies produce DIFFERENT results...")

# Check all strategies are different
tolerance = 1e-4  # Allow tiny floating point differences

baseline_vs_uniform = abs(results['baseline']['loss'] - results['uniform']['loss'])
baseline_vs_normalized = abs(results['baseline']['loss'] - results['normalized']['loss'])
uniform_vs_normalized = abs(results['uniform']['loss'] - results['normalized']['loss'])

print(f"\n  Baseline vs Uniform difference:    {baseline_vs_uniform:.6f}")
print(f"  Baseline vs Normalized difference: {baseline_vs_normalized:.6f}")
print(f"  Uniform vs Normalized difference:  {uniform_vs_normalized:.6f}")

# Test assertions
tests_passed = True

if baseline_vs_uniform < tolerance:
    print("\n  FAIL: Baseline and Uniform are IDENTICAL (patching not working!)")
    tests_passed = False
else:
    print("\n  PASS: Baseline != Uniform")

if baseline_vs_normalized < tolerance:
    print("  FAIL: Baseline and Normalized are IDENTICAL (patching not working!)")
    tests_passed = False
else:
    print("  PASS: Baseline != Normalized")

if uniform_vs_normalized < tolerance:
    print("  FAIL: Uniform and Normalized are IDENTICAL (bug still exists!)")
    tests_passed = False
else:
    print("  PASS: Uniform != Normalized")

print("\n" + "="*70)

if tests_passed:
    print("ALL TESTS PASSED!")
    print("\nRouting strategies are working correctly.")
    print("Uniform and normalized produce DIFFERENT results.")
    print("You can now re-run the full experiment.")
    sys.exit(0)
else:
    print("TESTS FAILED!")
    print("\nRouting strategies are NOT modifying computation.")
    print("Fix the patching implementation before re-running experiments.")
    sys.exit(1)
