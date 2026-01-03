#!/usr/bin/env python3
"""
Check OLMoE Original Implementation
====================================

This script loads the original OLMoE model and inspects how it computes
routing weights to determine the correct implementation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import inspect


def inspect_olmoe_routing():
    """Inspect OLMoE's original routing implementation."""
    print("=" * 80)
    print("INSPECTING OLMoE ORIGINAL ROUTING IMPLEMENTATION")
    print("=" * 80)

    # Load a small sample to inspect
    model_name = "allenai/OLMoE-1B-7B-0924"
    print(f"\nLoading model: {model_name}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Find MoE blocks
    moe_blocks = []
    for name, module in model.named_modules():
        if 'SparseMoe' in module.__class__.__name__:
            moe_blocks.append((name, module))
            print(f"\n✅ Found MoE block: {name}")
            print(f"   Class: {module.__class__.__name__}")

    if not moe_blocks:
        print("\n❌ No MoE blocks found!")
        return

    # Inspect the first MoE block
    name, moe_block = moe_blocks[0]

    print(f"\n{'=' * 80}")
    print(f"INSPECTING: {name}")
    print(f"{'=' * 80}")

    # Check forward method
    print("\n1. Forward method signature:")
    print(f"   {inspect.signature(moe_block.forward)}")

    # Get source code of forward method
    print("\n2. Forward method source code:")
    try:
        source = inspect.getsource(moe_block.forward)
        print(source)
    except Exception as e:
        print(f"   ❌ Could not get source: {e}")
        print("   Trying to get from class...")
        try:
            source = inspect.getsource(moe_block.__class__.forward)
            print(source)
        except Exception as e2:
            print(f"   ❌ Could not get class source: {e2}")

    # Check for router/gate
    print("\n3. Router/Gate attributes:")
    for attr_name in ['gate', 'router', 'gating']:
        if hasattr(moe_block, attr_name):
            attr = getattr(moe_block, attr_name)
            print(f"   ✅ Found: {attr_name}")
            print(f"      Type: {type(attr)}")

    # Check for expert modules
    print("\n4. Expert modules:")
    if hasattr(moe_block, 'experts'):
        print(f"   ✅ Found 'experts' attribute")
        print(f"      Type: {type(moe_block.experts)}")
        print(f"      Num experts: {len(moe_block.experts)}")

    # Try to trace through with dummy input
    print("\n5. Tracing forward pass with dummy input:")
    try:
        dummy_input = torch.randn(1, 4, model.config.hidden_size)
        print(f"   Input shape: {dummy_input.shape}")

        # Add hooks to capture intermediate values
        routing_logits = []
        routing_weights = []

        def gate_hook(module, input, output):
            routing_logits.append(output.detach().clone())
            print(f"   📊 Gate output shape: {output.shape}")
            print(f"      Sample logits (first token): {output[0, :5]}")

        # Register hook if gate exists
        if hasattr(moe_block, 'gate'):
            hook = moe_block.gate.register_forward_hook(gate_hook)

        # Run forward
        output = moe_block(dummy_input)
        print(f"   ✅ Forward pass succeeded")
        print(f"   Output shape: {output[0].shape if isinstance(output, tuple) else output.shape}")

        # Analyze captured routing logits
        if routing_logits:
            logits = routing_logits[0]
            print(f"\n6. Analyzing routing computation:")
            print(f"   Logits shape: {logits.shape}")

            # Compute different weight patterns
            import torch.nn.functional as F

            # Pattern A: Full softmax (no selection)
            weights_full = F.softmax(logits, dim=-1)
            print(f"\n   Full softmax (all experts):")
            print(f"      Sum: {weights_full[0].sum():.6f}")
            print(f"      Top-5: {weights_full[0].topk(5).values}")

            # Pattern B: TopK then renorm (current buggy implementation)
            k = 8
            topk_weights, topk_idx = torch.topk(weights_full[0], k)
            weights_renorm = topk_weights / topk_weights.sum()
            print(f"\n   Top-{k} with renormalization (BUGGY):")
            print(f"      Sum: {topk_weights.sum():.6f} → 1.000000 after renorm")
            print(f"      Top expert: {topk_weights[0]:.6f} → {weights_renorm[0]:.6f}")

            # Pattern C: TopK softmax (softmax over selected logits only)
            topk_logits, _ = torch.topk(logits[0], k)
            weights_topk_softmax = F.softmax(topk_logits, dim=-1)
            print(f"\n   Top-{k} logits then softmax:")
            print(f"      Sum: {weights_topk_softmax.sum():.6f}")
            print(f"      Top expert: {weights_topk_softmax[0]:.6f}")

        # Clean up hook
        if hasattr(moe_block, 'gate'):
            hook.remove()

    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    inspect_olmoe_routing()
