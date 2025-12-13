# OLMoE Routing Internals Documentation

**Date**: 2025-12-13
**Model**: allenai/OLMoE-1B-7B-0924
**Source**: HuggingFace Transformers library

---

## Overview

OLMoE (Open Language Model with Mixture-of-Experts) uses a **fine-grained sparse MoE architecture** with:
- **64 small experts per layer**
- **Top-8 routing** (fixed K=8 experts selected per token)
- **Dropless token-choice routing** (no tokens are dropped)

---

## Architecture Components

### 1. OlmoeTopKRouter

**Location**: `transformers/models/olmoe/modeling_olmoe.py`

**Purpose**: Selects top-K experts for each token based on router logits.

**Key Attributes**:
- `hidden_dim`: Input dimension (from model config)
- `num_experts`: Total experts available (64)
- `top_k`: Number of experts to select (8)
- `norm_topk_prob`: Whether to renormalize top-k probabilities
- `weight`: Linear projection matrix [hidden_dim, num_experts]

**Forward Method Signature**:
```python
def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Args:
        hidden_states: [batch*seq_len, hidden_dim]

    Returns:
        router_logits: [batch*seq_len, num_experts]  # All logits
        routing_weights: [batch*seq_len, top_k]      # Top-K weights (normalized)
        selected_experts: [batch*seq_len, top_k]     # Top-K indices
    """
```

**Implementation Flow**:
```python
def forward(self, hidden_states):
    # Step 1: Reshape input
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    # Shape: [batch*seq_len, hidden_dim]

    # Step 2: Compute router logits via linear projection
    router_logits = F.linear(hidden_states, self.weight)
    # Shape: [batch*seq_len, num_experts]

    # Step 3: Apply softmax to get probabilities
    router_logits = F.softmax(router_logits, dtype=torch.float, dim=-1)
    # Shape: [batch*seq_len, num_experts]

    # Step 4: Select top-K experts ← THIS IS THE KEY LINE TO REPLACE
    routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
    # routing_weights: [batch*seq_len, top_k]
    # selected_experts: [batch*seq_len, top_k]

    # Step 5: Optional renormalization
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # routing_weights: [batch*seq_len, top_k] (sums to 1.0)

    return router_logits, routing_weights, selected_experts
```

---

### 2. OlmoeSparseMoeBlock

**Purpose**: Combines router with expert networks to process tokens.

**Key Attributes**:
- `gate`: The OlmoeTopKRouter instance
- `experts`: OlmoeExperts module (contains all 64 expert networks)

**Forward Method**:
```python
def forward(self, hidden_states: Tensor) -> Tensor:
    """
    Args:
        hidden_states: [batch_size, seq_len, hidden_dim]

    Returns:
        final_hidden_states: [batch_size, seq_len, hidden_dim]
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape

    # Step 1: Flatten for routing
    hidden_states = hidden_states.view(-1, hidden_dim)
    # Shape: [batch*seq_len, hidden_dim]

    # Step 2: Route to experts
    _, top_k_weights, top_k_index = self.gate(hidden_states)
    # top_k_weights: [batch*seq_len, top_k]
    # top_k_index: [batch*seq_len, top_k]

    # Step 3: Dispatch and compute expert outputs
    final_hidden_states = self.experts(
        hidden_states, top_k_index, top_k_weights
    )
    # Shape: [batch*seq_len, hidden_dim]

    # Step 4: Reshape back
    final_hidden_states = final_hidden_states.reshape(
        batch_size, sequence_length, hidden_dim
    )

    return final_hidden_states
```

---

### 3. OlmoeExperts

**Purpose**: Dispatches tokens to selected experts and combines outputs.

**Dispatch Logic**:
1. Creates one-hot masks for expert selection
2. For each expert that was selected by at least one token:
   - Gathers tokens assigned to that expert
   - Applies gated linear transformations
   - Scales outputs by routing weights
3. Accumulates weighted expert outputs using `index_add_()`

**Key Operations**:
- Sparse dispatch (only computes selected experts)
- Weighted combination based on routing weights
- Efficient batched processing

---

## Patching Strategy for BH Routing

### Challenge: Tensor Shape Mismatch

**OLMoE Expects** (Top-K format):
```python
routing_weights: [batch*seq_len, 8]    # Exactly 8 weights per token
selected_experts: [batch*seq_len, 8]   # Exactly 8 indices per token
```

**BH Routing Produces** (Adaptive format):
```python
routing_weights: [batch, seq_len, 64]  # Sparse weights (zeros for unselected)
selected_experts: [batch, seq_len, 8]  # Padded with -1 for unused slots
expert_counts: [batch, seq_len]        # Actual count (1-8) per token
```

### Solution: Format Conversion

To make BH routing compatible with OLMoE, we need to convert BH's output to Top-K format:

```python
def bh_to_topk_format(sparse_weights, selected_experts, expert_counts, max_k=8):
    """
    Convert BH routing output to OLMoE's expected Top-K format.

    Args:
        sparse_weights: [B, S, num_experts] - sparse weights across all experts
        selected_experts: [B, S, max_k] - padded expert indices (-1 for unused)
        expert_counts: [B, S] - actual number of experts selected
        max_k: Maximum K (8 for OLMoE)

    Returns:
        dense_weights: [B*S, max_k] - weights for top-k positions
        dense_indices: [B*S, max_k] - indices for top-k positions
    """
    batch_size, seq_len, num_experts = sparse_weights.shape

    # Flatten batch and sequence dimensions
    sparse_weights_flat = sparse_weights.view(-1, num_experts)  # [B*S, num_experts]
    selected_experts_flat = selected_experts.view(-1, max_k)     # [B*S, max_k]

    # Extract weights for selected experts
    # Gather weights at selected indices
    safe_indices = selected_experts_flat.clamp(min=0)  # Replace -1 with 0 temporarily
    dense_weights = sparse_weights_flat.gather(dim=-1, index=safe_indices)

    # Mask out padding positions
    padding_mask = (selected_experts_flat == -1)
    dense_weights = dense_weights.masked_fill(padding_mask, 0.0)

    # Renormalize (since we may have fewer than max_k experts)
    weight_sums = dense_weights.sum(dim=-1, keepdim=True)
    dense_weights = dense_weights / torch.clamp(weight_sums, min=1e-10)

    # Dense indices (replace -1 padding with 0, but they'll have 0 weight)
    dense_indices = torch.where(
        selected_experts_flat == -1,
        torch.zeros_like(selected_experts_flat),
        selected_experts_flat
    )

    return dense_weights, dense_indices
```

---

## Implementation: Patching OlmoeTopKRouter

### Approach 1: Patch Router Forward (Recommended)

Patch the `OlmoeTopKRouter.forward()` method to use BH routing instead of topk:

```python
def create_bh_patched_router_forward(original_forward, bh_config):
    """
    Create a patched router forward that uses BH routing.

    Args:
        original_forward: The original forward method of OlmoeTopKRouter
        bh_config: Dict with 'alpha', 'temperature', 'min_k', 'max_k'
    """

    def patched_forward(self, hidden_states):
        # Step 1: Reshape (same as original)
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        # Shape: [batch*seq_len, hidden_dim]

        # Step 2: Compute router logits (same as original)
        router_logits = F.linear(hidden_states, self.weight)
        # Shape: [batch*seq_len, num_experts]

        # NOTE: OLMoE applies softmax to router_logits in-place for return
        # but torch.topk operates on the tensor before softmax
        # We need to check the actual implementation

        # Step 3: Apply BH routing INSTEAD of topk
        # BH routing expects [batch, seq, experts] or [tokens, experts]
        sparse_weights, selected_experts, expert_counts = benjamini_hochberg_routing(
            router_logits,  # [batch*seq_len, num_experts]
            alpha=bh_config['alpha'],
            temperature=bh_config['temperature'],
            min_k=bh_config.get('min_k', 1),
            max_k=bh_config.get('max_k', self.top_k)
        )
        # sparse_weights: [batch*seq_len, num_experts]
        # selected_experts: [batch*seq_len, max_k]
        # expert_counts: [batch*seq_len]

        # Step 4: Convert to Top-K format expected by OlmoeExperts
        routing_weights, selected_experts_dense = bh_to_topk_format(
            sparse_weights.unsqueeze(0),  # Add batch dim
            selected_experts.unsqueeze(0),  # Add batch dim
            expert_counts.unsqueeze(0),
            max_k=self.top_k
        )
        # routing_weights: [batch*seq_len, top_k]
        # selected_experts_dense: [batch*seq_len, top_k]

        # Step 5: Apply softmax to router_logits for return (if needed)
        router_logits_softmax = F.softmax(router_logits, dtype=torch.float, dim=-1)

        # Step 6: Return in expected format
        return router_logits_softmax, routing_weights, selected_experts_dense

    return patched_forward
```

### Approach 2: Patch SparseMoeBlock Forward (Alternative)

Alternatively, patch the entire `OlmoeSparseMoeBlock.forward()` method:

```python
def create_bh_patched_block_forward(original_forward, bh_config):
    """
    Create a patched MoE block forward that uses BH routing.
    """

    def patched_forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Compute router logits using the gate's linear layer
        router_logits = F.linear(hidden_states_flat, self.gate.weight)

        # Apply BH routing
        sparse_weights, selected_experts, expert_counts = benjamini_hochberg_routing(
            router_logits,
            alpha=bh_config['alpha'],
            temperature=bh_config['temperature'],
            min_k=bh_config.get('min_k', 1),
            max_k=bh_config.get('max_k', 8)
        )

        # Convert to Top-K format
        routing_weights, selected_experts_dense = bh_to_topk_format(
            sparse_weights.unsqueeze(0),
            selected_experts.unsqueeze(0),
            expert_counts.unsqueeze(0),
            max_k=self.gate.top_k
        )

        # Dispatch to experts (same as original)
        final_hidden_states = self.experts(
            hidden_states_flat,
            selected_experts_dense,
            routing_weights.to(hidden_states.dtype)
        )

        # Reshape back
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )

        return final_hidden_states

    return patched_forward
```

---

## Verification Checklist

After patching, verify that BH routing is actually working:

### ✅ Test 1: Expert counts should vary (not always 8)
```python
# Collect routing statistics
stats = analyzer.get_stats()
assert stats['min_experts'] < 8, "Should have tokens using fewer than 8 experts"
assert stats['mean_experts'] < 8, "Mean should be less than 8"
print(f"✓ Expert counts vary: min={stats['min_experts']}, mean={stats['mean_experts']:.2f}, max={stats['max_experts']}")
```

### ✅ Test 2: Simple prompts use fewer experts than complex prompts
```python
simple_prompt = "The cat sat"
complex_prompt = "The philosophical implications of quantum entanglement"

# Run both and compare
assert complex_stats['mean_experts'] > simple_stats['mean_experts'], \
    "Complex prompts should use more experts"
print(f"✓ Simple: {simple_stats['mean_experts']:.2f}, Complex: {complex_stats['mean_experts']:.2f}")
```

### ✅ Test 3: Output is still coherent
```python
prompt = "The capital of France is"
output = generate(model, prompt, max_tokens=10)
assert "Paris" in output or "paris" in output.lower(), \
    "Model should still generate coherent text"
print(f"✓ Output coherent: {output}")
```

### ✅ Test 4: Alpha parameter affects expert count
```python
# Lower alpha should select fewer experts
stats_alpha_01 = test_with_alpha(0.01)
stats_alpha_20 = test_with_alpha(0.20)
assert stats_alpha_01['mean_experts'] < stats_alpha_20['mean_experts'], \
    "Higher alpha should select more experts"
print(f"✓ Alpha sensitivity works: α=0.01 → {stats_alpha_01['mean_experts']:.2f}, α=0.20 → {stats_alpha_20['mean_experts']:.2f}")
```

---

## Critical Implementation Notes

### ⚠️ Note 1: Softmax Application Timing
The original router applies `F.softmax()` to `router_logits` **in-place** before returning. Check if this affects the topk operation.

### ⚠️ Note 2: Dtype Consistency
Ensure routing weights match the model's dtype (bfloat16 or float32):
```python
routing_weights = routing_weights.to(hidden_states.dtype)
```

### ⚠️ Note 3: Device Placement
All tensors must be on the same device as the model:
```python
device = hidden_states.device
```

### ⚠️ Note 4: Gradient Flow
For inference-only, use `torch.no_grad()` when collecting statistics:
```python
with torch.no_grad():
    stats = collect_routing_stats()
```

### ⚠️ Note 5: Padding Handling
BH routing uses `-1` for padding. Ensure this doesn't cause index errors:
```python
safe_indices = selected_experts.clamp(min=0)  # Replace -1 with 0
```

---

## Model Information

### Configuration
```python
model.config.num_experts = 64
model.config.num_experts_per_tok = 8
model.config.hidden_size = 1024 (example)
```

### MoE Layers
OLMoE-1B-7B typically has **16 transformer layers** with MoE blocks.

Each layer has a router at:
```
model.model.layers[i].moe.gate  # OlmoeTopKRouter instance
```

---

## Summary

| Component | Purpose | Key Method |
|-----------|---------|------------|
| OlmoeTopKRouter | Routes tokens to top-K experts | `forward(hidden_states)` |
| OlmoeSparseMoeBlock | Combines routing + expert computation | `forward(hidden_states)` |
| OlmoeExperts | Dispatches to experts and combines outputs | `forward(hidden_states, indices, weights)` |

**Patching Target**: `OlmoeTopKRouter.forward()` at line with `torch.topk()`

**Expected Output Format**: `(router_logits, routing_weights, selected_experts)` where weights and indices have shape `[batch*seq_len, top_k]`

---

## References

- **HuggingFace transformers**: https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmoe/modeling_olmoe.py
- **OLMoE Paper**: https://arxiv.org/abs/2409.02060
- **OLMoE GitHub**: https://github.com/allenai/OLMoE
- **OlmoCore Router Docs**: https://olmo-core.readthedocs.io/en/latest/_modules/olmo_core/nn/moe/router.html
