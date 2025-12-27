"""
Diagnostic Script for BH Routing P-value Analysis
==================================================

This script analyzes the relationship between router logits, p-values, and
expert selection in the Benjamini-Hochberg routing implementation.

Purpose:
--------
Diagnose why BH routing is producing high perplexity by examining the
alignment between:
- Raw router logits (high = expert is relevant)
- Computed p-values (low = statistically significant)
- BH selection decisions (should select low p-value experts)

Test sentence: "The capital of France is Paris."
Analyzed tokens: "The" (first token) and "France" (semantic token)
Analyzed layers: Layer 0 (first MoE) and Layer 7 (middle MoE)
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from bh_routing import load_kde_models, compute_pvalues_kde, benjamini_hochberg_routing

# Configuration
MODEL_NAME = "allenai/OLMoE-1B-7B-0924"
TEST_SENTENCE = "The capital of France is Paris."
ALPHA = 0.6
MAX_K = 8
TEMPERATURE = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Layers to analyze
LAYERS_TO_ANALYZE = [0, 7]  # First and middle MoE layers

# Tokens to analyze (by text)
TOKENS_TO_ANALYZE = ["The", "France"]


def print_separator(title, char="="):
    """Print formatted separator."""
    print(f"\n{char * 80}")
    print(f"{title:^80}")
    print(f"{char * 80}\n")


def print_expert_analysis(expert_idx, logit, pvalue, rank, critical_val, selected, is_top8):
    """Print analysis for single expert."""
    marker = "***TOP-8***" if is_top8 else "           "
    selected_marker = "✓ SELECTED" if selected else "✗ REJECTED"

    print(f"{marker} Expert {expert_idx:2d}: "
          f"Logit={logit:+7.4f}  "
          f"P-val={pvalue:.6f}  "
          f"Rank={rank:2d}  "
          f"BH_crit={critical_val:.6f}  "
          f"{selected_marker}")


def analyze_token_routing(token_text, token_idx, router_logits, layer_idx, kde_models, alpha, max_k):
    """
    Analyze routing for a single token in detail.

    Args:
        token_text: Text of the token (for display)
        token_idx: Index of the token in the sequence
        router_logits: Router logits for all tokens [seq_len, num_experts]
        layer_idx: Which MoE layer
        kde_models: Pre-loaded KDE models
        alpha: BH alpha parameter
        max_k: Maximum experts to select
    """
    print_separator(f"Layer {layer_idx} - Token '{token_text}' (position {token_idx})", char="=")

    # Extract logits for this token
    token_logits = router_logits[token_idx]  # [num_experts]
    num_experts = len(token_logits)

    # Compute p-values using the same method as BH routing
    token_logits_2d = token_logits.unsqueeze(0)  # [1, num_experts]
    p_values = compute_pvalues_kde(token_logits_2d, layer_idx, kde_models)
    p_values = p_values.squeeze(0)  # [num_experts]

    # Sort p-values to get ranks
    sorted_pvals, sorted_indices = torch.sort(p_values)

    # Create rank array (1-indexed)
    ranks = torch.zeros(num_experts, dtype=torch.long)
    for rank_idx, expert_idx in enumerate(sorted_indices):
        ranks[expert_idx] = rank_idx + 1  # 1-indexed

    # Compute BH critical values
    k_values = torch.arange(1, num_experts + 1, device=p_values.device)
    critical_values = (k_values / num_experts) * alpha

    # Determine which experts pass BH threshold
    passes_threshold = sorted_pvals <= critical_values
    num_selected_bh = passes_threshold.sum().item()
    num_selected_bh = max(1, min(num_selected_bh, max_k))  # Clamp to [1, max_k]

    # Create selection mask
    selected_mask = ranks <= num_selected_bh

    # Identify top-8 experts by raw logit
    top8_logits, top8_indices = torch.topk(token_logits, k=8)
    top8_set = set(top8_indices.cpu().tolist())

    # Convert to numpy for easier printing (convert bfloat16 to float32 first)
    logits_np = token_logits.cpu().float().numpy()
    pvals_np = p_values.cpu().float().numpy()
    ranks_np = ranks.cpu().numpy()
    selected_np = selected_mask.cpu().numpy()

    # Print summary statistics
    print(f"Summary:")
    print(f"  Alpha: {alpha}")
    print(f"  Max K: {max_k}")
    print(f"  BH Selected: {num_selected_bh} experts")
    print(f"  Top-8 by logit: {top8_indices.cpu().tolist()[:8]}")
    print(f"  Selected by BH: {torch.where(selected_mask)[0].cpu().tolist()[:8]}...")

    # Check overlap between top-8 and BH-selected
    selected_set = set(torch.where(selected_mask)[0].cpu().tolist())
    overlap = top8_set & selected_set
    print(f"  Overlap (Top-8 ∩ BH-selected): {len(overlap)}/8 experts")
    print()

    # Print detailed analysis for all 64 experts
    print("Detailed Expert Analysis:")
    print("-" * 80)
    print("Column Legend:")
    print("  Logit   : Raw router logit (higher = more relevant)")
    print("  P-val   : Statistical p-value (lower = more significant)")
    print("  Rank    : Rank by p-value (1 = lowest p-value)")
    print("  BH_crit : BH critical value for this rank (k/64 * alpha)")
    print("  Status  : Whether expert was selected by BH procedure")
    print("-" * 80)
    print()

    # Sort experts by logit (descending) for display
    logit_order = np.argsort(logits_np)[::-1]

    print("--- TOP-8 EXPERTS (by raw logit) ---")
    for i in range(8):
        expert_idx = logit_order[i]
        logit = logits_np[expert_idx]
        pval = pvals_np[expert_idx]
        rank = ranks_np[expert_idx]
        critical_val = critical_values[rank - 1].item()  # rank is 1-indexed
        selected = selected_np[expert_idx]
        is_top8 = expert_idx in top8_set

        print_expert_analysis(expert_idx, logit, pval, rank, critical_val, selected, is_top8)

    print("\n--- REMAINING EXPERTS (sorted by logit descending) ---")
    for i in range(8, num_experts):
        expert_idx = logit_order[i]
        logit = logits_np[expert_idx]
        pval = pvals_np[expert_idx]
        rank = ranks_np[expert_idx]
        critical_val = critical_values[rank - 1].item()
        selected = selected_np[expert_idx]
        is_top8 = expert_idx in top8_set

        print_expert_analysis(expert_idx, logit, pval, rank, critical_val, selected, is_top8)

    print()
    print("-" * 80)
    print(f"DIAGNOSIS:")
    print(f"  - If high-logit experts have HIGH p-values → P-value calculation is inverted")
    print(f"  - If high-logit experts have LOW p-values → P-value calculation is correct")
    print(f"  - If BH selects random experts → Threshold logic may be wrong")
    print(f"  - Expected: BH should select most/all of the top-8 logit experts")
    print("-" * 80)


def main():
    print_separator("BH Routing Diagnostic Script", char="=")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Test sentence: '{TEST_SENTENCE}'")
    print(f"BH parameters: alpha={ALPHA}, max_k={MAX_K}")
    print()

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=DEVICE,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()
    print("✓ Model loaded\n")

    # Load KDE models
    print("Loading KDE models...")
    kde_models = load_kde_models()
    if not kde_models:
        print("WARNING: No KDE models found. Will use empirical fallback.")
    else:
        print(f"✓ Loaded {len(kde_models)} KDE models\n")

    # Tokenize input
    print(f"Tokenizing: '{TEST_SENTENCE}'")
    inputs = tokenizer(TEST_SENTENCE, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]
    tokens = [tokenizer.decode([token_id]) for token_id in input_ids[0]]

    print(f"Tokens: {tokens}")
    print(f"Token IDs: {input_ids[0].tolist()}")
    print()

    # Find token indices for analysis
    token_indices = {}
    for target_token in TOKENS_TO_ANALYZE:
        for idx, token in enumerate(tokens):
            if target_token.lower() in token.lower():
                token_indices[target_token] = idx
                print(f"Found '{target_token}' at position {idx}")
                break

    if len(token_indices) != len(TOKENS_TO_ANALYZE):
        print(f"WARNING: Could not find all target tokens!")
        print(f"Available tokens: {tokens}")
        print(f"Using first and last tokens instead...")
        token_indices = {
            "First": 0,
            "Last": len(tokens) - 1
        }

    print()

    # Storage for router logits at each layer
    layer_router_logits = {}

    # Hook function to capture router logits
    def create_hook(layer_idx):
        def hook_fn(module, input, output):
            # OLMoE MoE blocks return (hidden_states, router_logits)
            if isinstance(output, tuple) and len(output) == 2:
                hidden_states, router_logits = output
                # router_logits shape: [batch_size * seq_len, num_experts]
                # Reshape to [batch_size, seq_len, num_experts]
                batch_size = inputs["input_ids"].shape[0]
                seq_len = inputs["input_ids"].shape[1]
                router_logits_reshaped = router_logits.view(batch_size, seq_len, -1)
                layer_router_logits[layer_idx] = router_logits_reshaped[0]  # Take first batch
        return hook_fn

    # Register hooks on target MoE layers
    print("Registering hooks on MoE layers...")
    hooks = []
    layer_count = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'OlmoeSparseMoeBlock':
            if layer_count in LAYERS_TO_ANALYZE:
                hook = module.register_forward_hook(create_hook(layer_count))
                hooks.append(hook)
                print(f"  Hooked layer {layer_count}: {name}")
            layer_count += 1

    print()

    # Run forward pass
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(**inputs)
    print("✓ Forward pass complete\n")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Analyze each layer and token
    for layer_idx in LAYERS_TO_ANALYZE:
        if layer_idx not in layer_router_logits:
            print(f"WARNING: No logits captured for layer {layer_idx}")
            continue

        router_logits = layer_router_logits[layer_idx]

        for token_text, token_idx in token_indices.items():
            analyze_token_routing(
                token_text=token_text,
                token_idx=token_idx,
                router_logits=router_logits,
                layer_idx=layer_idx,
                kde_models=kde_models,
                alpha=ALPHA,
                max_k=MAX_K
            )

    print_separator("Diagnostic Complete", char="=")
    print("Next steps:")
    print("1. Check if high-logit experts have low p-values (expected)")
    print("2. Check if BH selects the top-8 logit experts (expected)")
    print("3. If not, examine the p-value computation or BH threshold logic")
    print()


if __name__ == "__main__":
    main()
