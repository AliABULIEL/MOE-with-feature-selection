# OLMoE Routing Code Analysis

**Purpose**: Extract and document the EXACT routing implementation in OLMoE for BH routing injection.

**Source**: HuggingFace Transformers Library
**File**: `transformers/src/transformers/models/olmoe/modeling_olmoe.py`
**Model**: `allenai/OLMoE-1B-7B-0924`

**Date**: 2025-12-13

---

## Table of Contents
1. [File Locations](#1-file-locations)
2. [OlmoeTopKRouter - Complete Code](#2-olmoetopkrouter---complete-code)
3. [OlmoeExperts - Complete Code](#3-olmoeexperts---complete-code)
4. [OlmoeSparseMoeBlock - Complete Code](#4-olmoesparsemoeblock---complete-code)
5. [Load Balancing Loss Function](#5-load-balancing-loss-function)
6. [Tensor Shape Reference](#6-tensor-shape-reference)
7. [BH Injection Point](#7-bh-injection-point)
8. [Data Flow Diagram](#8-data-flow-diagram)

---

## 1. File Locations

### 1.1 Source Code Repository
```
Repository: https://github.com/huggingface/transformers
Branch: main
File Path: src/transformers/models/olmoe/modeling_olmoe.py
```

**IMPORTANT NOTE**: This file is auto-generated from `modular_olmoe.py`. Do not edit `modeling_olmoe.py` directly.

### 1.2 Configuration File
```
File Path: src/transformers/models/olmoe/configuration_olmoe.py
```

**Key Configuration Parameters**:
```python
num_experts = 64                    # Total number of experts
num_experts_per_tok = 8             # Top-k routing parameter
num_local_experts = 64              # Same as num_experts
hidden_size = 2048                  # Hidden dimension
intermediate_size = 2048            # FFN intermediate dimension
norm_topk_prob = False              # Whether to renormalize top-k probabilities
router_aux_loss_coef = 0.01         # Load balancing loss coefficient
```

### 1.3 Local Repository

The local OLMoE repository at `/Users/aliab/Desktop/GitHub/MOE-with-feature-selection/OLMoE` is a **research artifacts repository**, not the model implementation. It contains:
- Training configurations (YAML files)
- Analysis scripts (Python)
- Visualization notebooks
- Evaluation tools

The **actual model code** is in the HuggingFace transformers library.

---

## 2. OlmoeTopKRouter - Complete Code

### 2.1 Source Code

```python
class OlmoeTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok           # 8
        self.num_experts = config.num_experts             # 64
        self.norm_topk_prob = config.norm_topk_prob       # False
        self.hidden_dim = config.hidden_size              # 2048

        # Router weight: [num_experts, hidden_dim] = [64, 2048]
        # No bias term
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        """
        Forward pass of the router.

        Args:
            hidden_states: Input token representations
                Shape: [batch_size, seq_len, hidden_dim] or [num_tokens, hidden_dim]

        Returns:
            router_logits: Softmax probabilities for all experts
                Shape: [num_tokens, num_experts] = [B*L, 64]
            router_scores: Selected expert weights (same as router_top_value)
                Shape: [num_tokens, top_k] = [B*L, 8]
            router_indices: Selected expert indices
                Shape: [num_tokens, top_k] = [B*L, 8]
        """
        # STEP 1: Flatten to 2D if needed
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        # Shape: [num_tokens, hidden_dim] = [B*L, 2048]

        # STEP 2: Compute router logits via linear projection
        # Formula: logits = hidden_states @ weight.T
        router_logits = F.linear(hidden_states, self.weight)
        # Shape: [num_tokens, num_experts] = [B*L, 64]

        # STEP 3: Apply softmax to convert logits to probabilities
        # Each row sums to 1.0
        router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        # Shape: [num_tokens, num_experts] = [B*L, 64]

        # STEP 4: Select top-k experts
        # ============================================================
        # ⚠️  CRITICAL: THIS IS THE LINE TO REPLACE FOR BH ROUTING ⚠️
        # ============================================================
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        # router_top_value: [num_tokens, top_k] = [B*L, 8]
        # router_indices: [num_tokens, top_k] = [B*L, 8]

        # STEP 5: Optional normalization (only if norm_topk_prob=True)
        # Re-normalize the top-k weights to sum to 1
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)

        # STEP 6: Convert back to original dtype (from float32 to model dtype)
        router_top_value = router_top_value.to(router_logits.dtype)

        # STEP 7: Prepare outputs
        router_scores = router_top_value

        return router_logits, router_scores, router_indices
```

### 2.2 Critical Variables

| Variable | Shape | Description |
|----------|-------|-------------|
| `self.weight` | `[64, 2048]` | Router weight matrix (no bias) |
| `hidden_states` | `[B*L, 2048]` | Flattened input tokens |
| `router_logits` | `[B*L, 64]` | Softmax probabilities over all experts |
| `router_top_value` | `[B*L, 8]` | Top-k expert weights |
| `router_indices` | `[B*L, 8]` | Top-k expert indices (values in [0, 63]) |
| `router_scores` | `[B*L, 8]` | Alias for `router_top_value` |

### 2.3 BH Injection Strategy

**Replace this line**:
```python
router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
```

**With BH routing**:
```python
router_top_value, router_indices = benjamini_hochberg_routing(
    router_logits,          # [B*L, 64] - softmax probabilities
    kde_model,              # Pre-trained KDE for this layer
    fdr_level=0.05,         # False discovery rate threshold
    min_experts=1,          # Minimum experts to select
    max_experts=self.top_k  # Maximum experts (8)
)
```

---

## 3. OlmoeExperts - Complete Code

### 3.1 Source Code

```python
class OlmoeExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config: OlmoeConfig):
        super().__init__()
        self.num_experts = config.num_local_experts      # 64
        self.hidden_dim = config.hidden_size             # 2048
        self.intermediate_dim = config.intermediate_size # 2048

        # Expert weights stored as 3D tensors for efficiency
        # gate_up_proj: [num_experts, 2*intermediate_dim, hidden_dim] = [64, 4096, 2048]
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )

        # down_proj: [num_experts, hidden_dim, intermediate_dim] = [64, 2048, 2048]
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )

        # SwiGLU activation function
        self.act_fn = ACT2FN[config.hidden_act]  # SiLU

    def forward(
        self,
        hidden_states: torch.Tensor,    # [num_tokens, hidden_dim] = [B*L, 2048]
        top_k_index: torch.Tensor,      # [num_tokens, top_k] = [B*L, 8]
        top_k_weights: torch.Tensor,    # [num_tokens, top_k] = [B*L, 8]
    ) -> torch.Tensor:
        """
        Process tokens through their selected experts.

        Returns:
            final_hidden_states: Weighted expert outputs
                Shape: [num_tokens, hidden_dim] = [B*L, 2048]
        """
        # Initialize output tensor
        final_hidden_states = torch.zeros_like(hidden_states)
        # Shape: [B*L, 2048]

        # Create expert assignment mask
        with torch.no_grad():
            # Convert top_k_index to one-hot encoding
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            # Shape: [num_tokens, top_k, num_experts] = [B*L, 8, 64]

            # Permute to [num_experts, top_k, num_tokens] = [64, 8, B*L]
            expert_mask = expert_mask.permute(2, 1, 0)

            # Find which experts have at least one token assigned
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            # expert_hit: indices of experts that are used

        # Loop over each expert that has tokens assigned
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]

            # Skip if index is out of bounds (shouldn't happen)
            if expert_idx == self.num_experts:
                continue

            # Find which tokens are assigned to this expert
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            # top_k_pos: position in top_k list (0-7)
            # token_idx: which tokens are assigned to this expert

            # Get tokens assigned to this expert
            current_state = hidden_states[token_idx]
            # Shape: [num_assigned_tokens, hidden_dim]

            # Apply expert FFN (SwiGLU)
            # gate_up: [num_assigned_tokens, 2*intermediate_dim]
            gate, up = nn.functional.linear(
                current_state,
                self.gate_up_proj[expert_idx]
            ).chunk(2, dim=-1)

            # SwiGLU: silu(gate) * up
            current_hidden_states = self.act_fn(gate) * up
            # Shape: [num_assigned_tokens, intermediate_dim]

            # Down projection
            current_hidden_states = nn.functional.linear(
                current_hidden_states,
                self.down_proj[expert_idx]
            )
            # Shape: [num_assigned_tokens, hidden_dim]

            # Weight by router scores
            # Multiply each hidden state by its routing weight
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]

            # Accumulate in output tensor
            # Uses index_add_ for proper accumulation when same token uses multiple experts
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

        return final_hidden_states
```

### 3.2 Key Operations

**Expert Assignment**:
```python
expert_mask[expert_idx, top_k_pos, token_idx] = 1
# expert_idx: which expert (0-63)
# top_k_pos: position in top-k list (0-7)
# token_idx: which token (0 to B*L-1)
```

**SwiGLU Activation**:
```python
output = silu(gate) * up
# gate: gating values
# up: upward projection
# silu: Sigmoid Linear Unit (Swish)
```

**Weight Application**:
```python
expert_output *= routing_weight
# Each expert's output is weighted by its routing score
```

**Accumulation**:
```python
final_hidden_states[token_idx] += weighted_expert_output
# Multiple experts' outputs are summed for each token
```

### 3.3 Variable-Length Expert Selection

**Current Assumption**: All tokens use exactly `top_k` experts

**For BH Routing**: Number of experts may vary per token

**Options**:
1. **Pad to max_k** (simpler, no code change needed)
   ```python
   # BH selects variable k, pad with zeros
   router_indices[token, num_selected:max_k] = 0  # Dummy expert
   router_weights[token, num_selected:max_k] = 0  # Zero weight
   ```

2. **Modify OlmoeExperts** (more efficient, requires changes)
   ```python
   # Support ragged tensors or list of lists
   # More complex implementation
   ```

**Recommendation**: Use padding approach initially for compatibility.

---

## 4. OlmoeSparseMoeBlock - Complete Code

### 4.1 Source Code

```python
class OlmoeSparseMoeBlock(nn.Module):
    """
    Complete MoE block that combines routing and expert computation.
    """

    def __init__(self, config):
        super().__init__()
        self.gate = OlmoeTopKRouter(config)      # Router
        self.experts = OlmoeExperts(config)      # Expert weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MoE block.

        Args:
            hidden_states: Input from previous layer
                Shape: [batch_size, seq_len, hidden_dim] = [B, L, 2048]

        Returns:
            final_hidden_states: MoE output
                Shape: [batch_size, seq_len, hidden_dim] = [B, L, 2048]
        """
        # Save original shape
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        # Flatten to 2D
        hidden_states = hidden_states.view(-1, hidden_dim)
        # Shape: [batch_size * seq_len, hidden_dim] = [B*L, 2048]

        # Route tokens to experts
        router_logits, top_k_weights, top_k_index = self.gate(hidden_states)
        # router_logits: [B*L, 64] - full probability distribution
        # top_k_weights: [B*L, 8] - selected expert weights
        # top_k_index: [B*L, 8] - selected expert indices

        # Process through experts
        final_hidden_states = self.experts(
            hidden_states,
            top_k_index,
            top_k_weights
        )
        # Shape: [B*L, 2048]

        # Reshape back to 3D
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        # Shape: [batch_size, seq_len, hidden_dim] = [B, L, 2048]

        return final_hidden_states
```

### 4.2 Integration with Transformer Layers

**Full Transformer Layer Structure**:
```python
class OlmoeDecoderLayer(nn.Module):
    def forward(self, hidden_states, ...):
        # Self-attention
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states, ...)
        hidden_states = residual + hidden_states

        # MoE block (replaces standard FFN)
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)  # ← OlmoeSparseMoeBlock
        hidden_states = residual + hidden_states

        return hidden_states
```

**Hook Location**: `layer.mlp.gate` is where we can intercept router logits

---

## 5. Load Balancing Loss Function

### 5.1 Complete Code

```python
def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    """
    Computes auxiliary load balancing loss as in Switch Transformer.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for details.

    Formula:
        loss = num_experts * sum(tokens_per_expert * router_prob_per_expert)

    Where:
        - tokens_per_expert[i] = fraction of tokens routed to expert i
        - router_prob_per_expert[i] = average routing probability for expert i

    This encourages uniform distribution of tokens across experts.

    Args:
        gate_logits: Tuple of router logits from all layers
            Each element: [batch_size * seq_len, num_experts]
        num_experts: Number of experts (64)
        top_k: Number of experts per token (8)
        attention_mask: Optional mask for padding tokens

    Returns:
        Load balancing loss (scalar)
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        # Concatenate logits from all layers
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat(
            [layer_gate.to(compute_device) for layer_gate in gate_logits],
            dim=0
        )
        # Shape: [num_layers * batch_size * seq_len, num_experts]

        # Compute routing probabilities
        routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

        # Select top-k experts
        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        # Shape: [num_layers * B * L, top_k]

        # Create expert mask (one-hot encoding)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
        # Shape: [num_layers * B * L, top_k, num_experts]

        if attention_mask is None:
            # Simple case: no padding
            # tokens_per_expert: [num_experts]
            # Average over all tokens and top_k positions
            tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

            # router_prob_per_expert: [num_experts]
            # Average routing probability to each expert
            router_prob_per_expert = torch.mean(routing_weights, dim=0)
        else:
            # Complex case: handle padding tokens
            batch_size, sequence_length = attention_mask.shape
            num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

            # Create attention mask for expert_mask
            expert_attention_mask = (
                attention_mask[None, :, :, None, None]
                .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
                .reshape(-1, top_k, num_experts)
                .to(compute_device)
            )

            # Compute tokens_per_expert with masking
            tokens_per_expert = torch.sum(
                expert_mask.float() * expert_attention_mask, dim=0
            ) / torch.sum(expert_attention_mask, dim=0)

            # Create attention mask for routing_weights
            router_per_expert_attention_mask = (
                attention_mask[None, :, :, None]
                .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
                .reshape(-1, num_experts)
                .to(compute_device)
            )

            # Compute router_prob_per_expert with masking
            router_prob_per_expert = torch.sum(
                routing_weights * router_per_expert_attention_mask, dim=0
            ) / torch.sum(router_per_expert_attention_mask, dim=0)

        # Final loss: dot product scaled by num_experts
        overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
        return overall_loss * num_experts
```

### 5.2 Auxiliary Losses

**Total Training Loss**:
```python
total_loss = language_model_loss + router_aux_loss_coef * load_balancing_loss
# router_aux_loss_coef = 0.01 (default)
```

**BH Routing Considerations**:
- Load balancing loss assumes fixed top-k
- With BH, different tokens use different numbers of experts
- May need to modify loss computation for BH
- Alternative: Use expert utilization penalty instead

---

## 6. Tensor Shape Reference

### 6.1 Complete Data Flow

```
Input to MoE Block:
  hidden_states: [batch_size, seq_len, hidden_dim]
               = [2, 512, 2048] (example)

Flatten for routing:
  hidden_states: [num_tokens, hidden_dim]
               = [1024, 2048]  (2*512 = 1024 tokens)

Router logits computation:
  raw_logits = hidden_states @ weight.T
  - hidden_states: [1024, 2048]
  - weight: [64, 2048]
  - raw_logits: [1024, 64]

Softmax normalization:
  router_logits = softmax(raw_logits, dim=-1)
  - router_logits: [1024, 64]  (each row sums to 1)

Top-K selection:
  router_top_value, router_indices = topk(router_logits, k=8, dim=-1)
  - router_top_value: [1024, 8]  (selected expert weights)
  - router_indices: [1024, 8]    (selected expert indices, values in [0, 63])

Expert computation:
  For each token (1024 total):
    - Uses 8 experts (indices in router_indices)
    - Weights outputs by router_top_value
    - Accumulates weighted outputs

  Output: [1024, 2048]

Reshape to original:
  final_output: [2, 512, 2048]
```

### 6.2 Per-Layer Breakdown

**Model**: OLMoE-1B-7B (16 layers, each with MoE)

| Component | Shape | Description |
|-----------|-------|-------------|
| Input embedding | `[B, L, 2048]` | Token embeddings |
| **Layer 0** | | |
| - Attention output | `[B, L, 2048]` | Self-attention |
| - Router logits | `[B*L, 64]` | Expert probabilities |
| - Selected experts | `[B*L, 8]` | Top-k indices |
| - Expert weights | `[B*L, 8]` | Top-k probabilities |
| - MoE output | `[B, L, 2048]` | Weighted expert outputs |
| **Layers 1-15** | | Same as Layer 0 |
| Output logits | `[B, L, vocab_size]` | Vocabulary predictions |

### 6.3 Memory Footprint (Batch=2, Seq=512)

**Router Logits Storage** (if logging all layers):
```
Per layer: [1024, 64] * 4 bytes (float32) = 262 KB
All layers: 262 KB * 16 = 4.2 MB per forward pass
```

**Expert Selection Storage**:
```
Indices: [1024, 8] * 8 bytes (long) = 65 KB per layer
Weights: [1024, 8] * 4 bytes (float32) = 32 KB per layer
Total per layer: 97 KB
All layers: 97 KB * 16 = 1.55 MB per forward pass
```

**For BH Routing** (additional storage):
```
P-values: [1024, 64] * 4 bytes = 262 KB per layer
All layers: 262 KB * 16 = 4.2 MB per forward pass
```

**Total logging overhead**: ~10 MB per forward pass (manageable)

---

## 7. BH Injection Point

### 7.1 Exact Location

**File**: `modeling_olmoe.py`
**Class**: `OlmoeTopKRouter`
**Method**: `forward()`
**Approximate Line**: ~117 (may vary between versions)

**Current code**:
```python
def forward(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states, self.weight)
    router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)

    # ⬇️ REPLACE THIS LINE ⬇️
    router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)

    if self.norm_topk_prob:
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
    router_top_value = router_top_value.to(router_logits.dtype)
    router_scores = router_top_value
    return router_logits, router_scores, router_indices
```

### 7.2 BH Replacement Code

**Option A: Direct Modification (Not Recommended)**
```python
# In modeling_olmoe.py (if you control the environment)
def forward(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states, self.weight)
    router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)

    # BH Routing
    if hasattr(self, 'use_bh_routing') and self.use_bh_routing:
        router_top_value, router_indices = benjamini_hochberg_select(
            router_logits,
            self.kde_model,
            self.fdr_level,
            self.top_k
        )
    else:
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)

    if self.norm_topk_prob:
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
    router_top_value = router_top_value.to(router_logits.dtype)
    router_scores = router_top_value
    return router_logits, router_scores, router_indices
```

**Option B: Model Patching (Recommended)**
```python
# In your experiment code (olmoe_routing_experiments.py)
# Monkey-patch the forward method without modifying transformers library

original_forward = layer.mlp.gate.forward

def bh_patched_forward(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states, self.weight)
    router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)

    # BH Routing
    router_top_value, router_indices = benjamini_hochberg_select(
        router_logits,
        kde_model,  # Passed via closure
        fdr_level,
        self.top_k
    )

    if self.norm_topk_prob:
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
    router_top_value = router_top_value.to(router_logits.dtype)
    router_scores = router_top_value
    return router_logits, router_scores, router_indices

# Apply patch
layer.mlp.gate.forward = bh_patched_forward.__get__(layer.mlp.gate, layer.mlp.gate.__class__)
```

### 7.3 Input/Output Specifications for BH Function

**Function Signature**:
```python
def benjamini_hochberg_select(
    router_logits: torch.Tensor,  # [num_tokens, num_experts] = [B*L, 64]
    kde_model: Dict,               # {'x': x_grid, 'cdf': cdf_grid}
    fdr_level: float,              # 0.05 (standard)
    max_experts: int,              # 8 (same as current top_k)
    min_experts: int = 1           # Minimum experts to select
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Benjamini-Hochberg procedure for expert selection.

    Returns:
        router_weights: [num_tokens, max_experts] - Selected expert weights
        router_indices: [num_tokens, max_experts] - Selected expert indices

    Note: If fewer than max_experts are selected, the rest are padded with zeros
    """
    pass
```

**Compatibility Requirements**:
1. **Output shapes must match**: `[num_tokens, max_experts]`
2. **Padding**: Unused slots should have `weight=0`, `index=0` (or dummy expert)
3. **Normalization**: Weights should sum to ≤1.0 per token
4. **Data type**: Match input dtype (bfloat16 or float32)
5. **Device**: Keep tensors on same device (CPU/GPU)

---

## 8. Data Flow Diagram

### 8.1 Text-Based Data Flow

```
┌────────────────────────────────────────────────────────────────┐
│                    OLMoE ROUTING DATA FLOW                     │
└────────────────────────────────────────────────────────────────┘

Input Tokens
    │
    ▼
┌─────────────────────────────────┐
│ Token Embedding                 │
│ Output: [B, L, 2048]            │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Layer 0: Self-Attention         │
│ Output: [B, L, 2048]            │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│ Layer 0: MoE Block (OlmoeSparseMoeBlock)                        │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ 1. Flatten: [B, L, 2048] → [B*L, 2048]                 │    │
│  └────────────────┬───────────────────────────────────────┘    │
│                   │                                             │
│                   ▼                                             │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ 2. Router (OlmoeTopKRouter)                            │    │
│  │                                                         │    │
│  │  a. Linear projection: [B*L, 2048] @ [64, 2048]^T      │    │
│  │     → router_logits_raw: [B*L, 64]                     │    │
│  │                                                         │    │
│  │  b. Softmax: router_logits_raw → router_logits         │    │
│  │     → router_logits: [B*L, 64] (sums to 1)             │    │
│  │                                                         │    │
│  │  ┌──────────────────────────────────────────────────┐  │    │
│  │  │ ⚠️  BH INJECTION POINT                           │  │    │
│  │  │                                                  │  │    │
│  │  │ CURRENT:                                         │  │    │
│  │  │   topk(router_logits, k=8)                       │  │    │
│  │  │   → weights: [B*L, 8]                            │  │    │
│  │  │   → indices: [B*L, 8]                            │  │    │
│  │  │                                                  │  │    │
│  │  │ BH ROUTING:                                      │  │    │
│  │  │   1. Compute p-values from router_logits        │  │    │
│  │  │      using KDE model                             │  │    │
│  │  │   2. Sort p-values ascending                     │  │    │
│  │  │   3. Apply BH procedure:                         │  │    │
│  │  │      For i=1 to 64:                              │  │    │
│  │  │        if p_i ≤ (i/64) * FDR:                    │  │    │
│  │  │          select expert i                         │  │    │
│  │  │   4. Return selected experts (variable k)        │  │    │
│  │  │   5. Pad to max_k=8 if needed                    │  │    │
│  │  └──────────────────────────────────────────────────┘  │    │
│  │                                                         │    │
│  └────────────────┬────────────────────────────────────────┘    │
│                   │                                             │
│                   │ top_k_weights: [B*L, 8]                     │
│                   │ top_k_indices: [B*L, 8]                     │
│                   │                                             │
│                   ▼                                             │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ 3. Experts (OlmoeExperts)                              │    │
│  │                                                         │    │
│  │  For each active expert (0-63):                        │    │
│  │    a. Find assigned tokens                             │    │
│  │    b. Apply expert FFN (SwiGLU):                       │    │
│  │       - gate_up projection                             │    │
│  │       - silu(gate) * up                                │    │
│  │       - down projection                                │    │
│  │    c. Weight by routing score                          │    │
│  │    d. Accumulate in output                             │    │
│  │                                                         │    │
│  │  Output: [B*L, 2048]                                   │    │
│  └────────────────┬───────────────────────────────────────┘    │
│                   │                                             │
│                   ▼                                             │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ 4. Reshape: [B*L, 2048] → [B, L, 2048]                 │    │
│  └────────────────┬───────────────────────────────────────┘    │
│                                                                  │
│  Output: [B, L, 2048]                                           │
└────────────────────┬─────────────────────────────────────────────┘
                     │
                     ▼
               ┌─────────────┐
               │ Layers 1-15 │ (Same structure as Layer 0)
               └─────────────┘
                     │
                     ▼
         ┌─────────────────────────┐
         │ LM Head                 │
         │ Output: [B, L, vocab]   │
         └─────────────────────────┘
                     │
                     ▼
              Final Logits
```

### 8.2 BH Procedure Detail

```
┌─────────────────────────────────────────────────────────────────┐
│         BENJAMINI-HOCHBERG EXPERT SELECTION PROCEDURE           │
└─────────────────────────────────────────────────────────────────┘

For each token t (processed independently):

Input: router_logits[t, :] = [prob_0, prob_1, ..., prob_63]  # Softmax probs

Step 1: Compute P-Values
├─ For each expert e:
│  ├─ expert_logit = raw_logit[t, e]  (before softmax)
│  ├─ CDF_value = KDE_model.estimate(expert_logit)
│  └─ p_value[e] = 1 - CDF_value
│
└─ Result: p_values[t, :] = [p_0, p_1, ..., p_63]

Step 2: Sort by P-Value (Ascending)
├─ sorted_pvalues, sorted_indices = sort(p_values[t, :])
└─ Result: sorted_pvalues = [0.001, 0.002, ..., 0.999]
           sorted_indices = [expert_42, expert_17, ..., expert_5]

Step 3: Apply BH Procedure
├─ For i = 1 to 64:
│  ├─ threshold = (i / 64) * FDR_level
│  ├─ if sorted_pvalues[i-1] ≤ threshold:
│  │  └─ num_selected = i
│  └─ else:
│     └─ break
│
└─ Result: num_selected = k (variable, typically 1-8)

Step 4: Enforce Constraints
├─ if num_selected < min_experts:
│  └─ num_selected = min_experts  (ensure at least 1 expert)
├─ if num_selected > max_experts:
│  └─ num_selected = max_experts  (cap at 8 for compatibility)
│
└─ selected_experts = sorted_indices[:num_selected]

Step 5: Compute Weights
├─ Option A: Use original softmax probabilities
│  ├─ weights = router_logits[t, selected_experts]
│  └─ weights = weights / weights.sum()  # Renormalize
│
├─ Option B: Inverse p-value weighting
│  ├─ weights = 1 / p_values[selected_experts]
│  └─ weights = weights / weights.sum()
│
└─ Option C: Uniform weighting
   └─ weights = [1/k, 1/k, ..., 1/k]

Step 6: Pad to max_k
├─ if num_selected < max_experts:
│  ├─ Pad weights with zeros: weights[num_selected:max_k] = 0
│  └─ Pad indices with dummy: indices[num_selected:max_k] = 0
│
└─ Result: router_weights[t, :] = [w_1, w_2, ..., w_k, 0, 0, ...]
           router_indices[t, :] = [e_1, e_2, ..., e_k, 0, 0, ...]

Output: router_weights[t, :max_k], router_indices[t, :max_k]
```

---

## Summary

### Critical Files for BH Implementation

1. **transformers/models/olmoe/modeling_olmoe.py**
   - `OlmoeTopKRouter.forward()` - Line ~117: `torch.topk()` call
   - This is the injection point

2. **Your repository: olmoe_routing_experiments.py**
   - `ModelPatchingUtils.custom_select_experts()` - Add BH strategy
   - `create_patched_forward()` - Modify to pass KDE models
   - `RoutingExperimentRunner` - Integrate BH routing

3. **Your repository: logs_eda.ipynb**
   - KDE training code (cells with `gaussian_kde`)
   - P-value computation code (cells with `np.interp`)
   - Extract into reusable functions

### Implementation Checklist

- [ ] Extract KDE training/inference code from notebook
- [ ] Implement `KDEModelManager` class
- [ ] Implement `benjamini_hochberg_select()` function
- [ ] Test BH function on synthetic data
- [ ] Modify `custom_select_experts()` to add BH strategy
- [ ] Modify `create_patched_forward()` for layer-aware patching
- [ ] Update `RoutingConfig` with BH parameters
- [ ] Add `BenjaminiHochbergRouting` strategy class
- [ ] Write unit tests for BH procedure
- [ ] Run integration test (CPU, single sample)
- [ ] Run full experiment (GPU, multiple datasets)
- [ ] Analyze results (perplexity, FDR, expert utilization)

### Expected Outcomes

**BH Routing Should**:
1. Select variable number of experts per token (1-8)
2. Control false discovery rate at specified level (e.g., FDR=0.05)
3. Achieve competitive perplexity vs. baseline
4. Potentially improve on:
   - Expert utilization balance
   - Computational efficiency (if k < 8 on average)
   - Robustness across datasets

**Metrics to Track**:
- Perplexity (quality)
- Tokens/second (speed)
- Average experts selected per token
- Expert utilization distribution
- Empirical FDR (if ground truth available)

---

**End of Routing Code Analysis**

This document provides everything needed to implement BH routing in OLMoE at the code level.
