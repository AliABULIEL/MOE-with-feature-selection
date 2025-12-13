"""
OLMOE ROUTING MECHANISM - DETAILED ANALYSIS
============================================

This file documents the exact routing implementation in the transformers library
for the OLMoE model, with precise variable names, shapes, and injection points
for implementing Benjamini-Hochberg (BH) statistical routing.

File Location: transformers/src/transformers/models/olmoe/modeling_olmoe.py
Model: allenai/OLMoE-1B-7B-0924 (1.3B active, 6.9B total parameters)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

class OlmoeRoutingConfig:
    """
    Key configuration parameters for OLMoE routing.

    Source: transformers/models/olmoe/configuration_olmoe.py (lines 109-170)
    """
    # MoE Architecture
    num_experts: int = 64                    # Total number of experts
    num_experts_per_tok: int = 8             # Number of experts selected per token (top-k)
    hidden_size: int = 2048                  # Hidden dimension
    intermediate_size: int = 2048            # FFN intermediate dimension

    # Routing Parameters
    norm_topk_prob: bool = False             # Whether to renormalize top-k probabilities
    output_router_logits: bool = False       # Whether to return router logits for analysis

    # Loss Parameters
    router_aux_loss_coef: float = 0.01       # Load balancing loss coefficient

    # Other
    hidden_act: str = "silu"                 # Activation function (SwiGLU)


# ============================================================================
# ROUTER IMPLEMENTATION
# ============================================================================

class OlmoeTopKRouter(nn.Module):
    """
    Top-K Router for OLMoE - THIS IS WHERE WE INJECT BH ROUTING

    Source: transformers/models/olmoe/modeling_olmoe.py (lines 340-358)

    Flow:
    1. Compute router logits via linear projection (no bias)
    2. Apply softmax to get probabilities
    3. Select top-k experts using torch.topk()  ← REPLACE THIS WITH BH ROUTING
    4. Optionally normalize top-k weights
    5. Return all routing information
    """

    def __init__(self, config):
        super().__init__()
        # Configuration
        self.top_k = config.num_experts_per_tok          # 8 for OLMoE
        self.num_experts = config.num_experts            # 64 for OLMoE
        self.norm_topk_prob = config.norm_topk_prob      # False by default
        self.hidden_dim = config.hidden_size             # 2048 for OLMoE

        # Router weight: [num_experts, hidden_dim] = [64, 2048]
        # No bias term
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        # Input:  [batch_size, seq_len, hidden_dim] or [num_tokens, hidden_dim]
        # Output: [num_tokens, hidden_dim]
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)

        # STEP 2: Compute router logits
        # Formula: logits = hidden_states @ weight.T
        # Input:  hidden_states [num_tokens, hidden_dim] = [B*L, 2048]
        #         self.weight [num_experts, hidden_dim] = [64, 2048]
        # Output: router_logits [num_tokens, num_experts] = [B*L, 64]
        router_logits = F.linear(hidden_states, self.weight)

        # STEP 3: Apply softmax to get probabilities
        # Converts raw logits to probabilities that sum to 1 over all experts
        # Shape: [num_tokens, num_experts] = [B*L, 64]
        # Each row sums to 1.0
        router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)

        # STEP 4: Select top-k experts
        # ============================================================
        # ⚠️  CRITICAL: THIS IS THE LINE TO REPLACE FOR BH ROUTING ⚠️
        # ============================================================
        # Current implementation: Standard top-k selection
        # Shape: router_top_value [num_tokens, top_k] = [B*L, 8]
        #        router_indices [num_tokens, top_k] = [B*L, 8]
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)

        # STEP 5: Optional normalization
        # If enabled, renormalize the top-k weights to sum to 1
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)

        # STEP 6: Convert back to original dtype
        router_top_value = router_top_value.to(router_logits.dtype)

        # STEP 7: Prepare outputs
        # router_scores is the same as router_top_value
        router_scores = router_top_value

        return router_logits, router_scores, router_indices


# ============================================================================
# EXPERT COMPUTATION
# ============================================================================

class OlmoeExperts(nn.Module):
    """
    Expert computation module - processes tokens through selected experts.

    Source: transformers/models/olmoe/modeling_olmoe.py (lines 301-337)

    Key points:
    - Uses 3D weight tensors for all experts
    - Loops over experts and processes their assigned tokens
    - Applies weighted combination using router_scores
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts           # 64
        self.hidden_dim = config.hidden_size            # 2048
        self.intermediate_dim = config.intermediate_size # 2048

        # Expert weights stored as 3D tensors
        # gate_up_proj: [num_experts, 2*intermediate_dim, hidden_dim] = [64, 4096, 2048]
        # down_proj: [num_experts, hidden_dim, intermediate_dim] = [64, 2048, 2048]
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )
        self.act_fn = nn.SiLU()  # SwiGLU activation

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process tokens through their selected experts.

        Args:
            hidden_states: Input tokens
                Shape: [num_tokens, hidden_dim] = [B*L, 2048]
            top_k_index: Selected expert indices from router
                Shape: [num_tokens, top_k] = [B*L, 8]
            top_k_weights: Expert weights from router
                Shape: [num_tokens, top_k] = [B*L, 8]

        Returns:
            final_hidden_states: Weighted expert outputs
                Shape: [num_tokens, hidden_dim] = [B*L, 2048]
        """
        # Initialize output tensor
        final_hidden_states = torch.zeros_like(hidden_states)

        # Create expert assignment mask
        # Convert top_k_index to one-hot encoding
        with torch.no_grad():
            # expert_mask: [num_tokens, top_k, num_experts] = [B*L, 8, 64]
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)

            # Permute to [num_experts, top_k, num_tokens] = [64, 8, B*L]
            expert_mask = expert_mask.permute(2, 1, 0)

            # Find which experts have at least one token assigned
            # expert_hit: indices of experts that are used
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        # Loop over each expert that has tokens assigned
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]

            # Skip if index is out of bounds
            if expert_idx == self.num_experts:
                continue

            # Find which tokens are assigned to this expert
            # top_k_pos: position in top_k list (0-7)
            # token_idx: which tokens are assigned to this expert
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])

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
            # Shape: [num_assigned_tokens, intermediate_dim]
            current_hidden_states = self.act_fn(gate) * up

            # Down projection
            # Shape: [num_assigned_tokens, hidden_dim]
            current_hidden_states = nn.functional.linear(
                current_hidden_states,
                self.down_proj[expert_idx]
            )

            # Weight by router scores
            # top_k_weights[token_idx, top_k_pos]: [num_assigned_tokens]
            # current_hidden_states: [num_assigned_tokens, hidden_dim]
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]

            # Accumulate in output tensor
            # Uses index_add_ for proper accumulation when same token uses multiple experts
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

        return final_hidden_states


# ============================================================================
# MOE BLOCK
# ============================================================================

class OlmoeSparseMoeBlock(nn.Module):
    """
    Complete MoE block that combines routing and expert computation.

    Source: transformers/models/olmoe/modeling_olmoe.py (lines 361-374)
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
        # Shape: [batch_size * seq_len, hidden_dim] = [B*L, 2048]
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Route tokens to experts
        # router_logits: [B*L, 64] - full probability distribution
        # top_k_weights: [B*L, 8] - selected expert weights
        # top_k_index: [B*L, 8] - selected expert indices
        router_logits, top_k_weights, top_k_index = self.gate(hidden_states)

        # Process through experts
        # Shape: [B*L, 2048]
        final_hidden_states = self.experts(
            hidden_states,
            top_k_index,
            top_k_weights
        )

        # Reshape back to 3D
        # Shape: [batch_size, seq_len, hidden_dim]
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )

        return final_hidden_states


# ============================================================================
# LOAD BALANCING LOSS
# ============================================================================

def load_balancing_loss_func(
    gate_logits: tuple,
    num_experts: int = 64,
    top_k: int = 8,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute auxiliary load balancing loss (Switch Transformer style).

    Source: transformers/models/olmoe/modeling_olmoe.py (lines 528-607)

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
        # Shape: [num_layers * B * L, top_k, num_experts]
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
        # Shape: [num_layers * B * L, num_experts]
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
    # Shape: scalar
    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


# ============================================================================
# SHAPE TRANSFORMATION SUMMARY
# ============================================================================

"""
COMPLETE DATA FLOW WITH SHAPES (batch_size=2, seq_len=512, hidden_dim=2048, num_experts=64, top_k=8)

1. Input to MoE Block:
   hidden_states: [2, 512, 2048]

2. Flatten for routing:
   hidden_states: [1024, 2048]  (2*512 = 1024 tokens)

3. Router logits computation:
   raw_logits = hidden_states @ weight.T
   - hidden_states: [1024, 2048]
   - weight: [64, 2048]
   - raw_logits: [1024, 64]

4. Softmax normalization:
   router_logits = softmax(raw_logits, dim=-1)
   - router_logits: [1024, 64]  (each row sums to 1)

5. Top-K selection:
   router_top_value, router_indices = topk(router_logits, k=8, dim=-1)
   - router_top_value: [1024, 8]  (selected expert weights)
   - router_indices: [1024, 8]    (selected expert indices, values in [0, 63])

6. Expert computation:
   For each token (1024 total):
     - Uses 8 experts (indices in router_indices)
     - Weights outputs by router_top_value
     - Accumulates weighted outputs

   Output: [1024, 2048]

7. Reshape to original:
   final_output: [2, 512, 2048]
"""


# ============================================================================
# INJECTION PLAN FOR BENJAMINI-HOCHBERG ROUTING
# ============================================================================

"""
BENJAMINI-HOCHBERG (BH) ROUTING IMPLEMENTATION PLAN
===================================================

Goal: Replace standard top-k routing with BH statistical routing that controls
      the False Discovery Rate (FDR) when selecting experts.

Location to Modify:
    OlmoeTopKRouter.forward() - Line 353 in modeling_olmoe.py

    Current code:
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)

    Replace with:
        router_top_value, router_indices = benjamini_hochberg_routing(
            router_logits,
            fdr_level=0.05,  # or adaptive
            min_experts=1,
            max_experts=self.top_k
        )

BH Routing Algorithm:
---------------------

For each token independently:

1. Get router probabilities: p = router_logits[token_idx]  # shape: [64]

2. Compute p-values:
   - Option A: Use probabilities as p-values directly
   - Option B: Compute via hypothesis test (token matches expert's specialty)

3. Sort p-values in ascending order:
   - sorted_pvalues, sorted_indices = sort(p)

4. Apply BH procedure:
   for i in range(num_experts):
       threshold = (i + 1) / num_experts * fdr_level
       if sorted_pvalues[i] <= threshold:
           selected_experts.append(sorted_indices[i])
       else:
           break

5. Ensure constraints:
   - If len(selected_experts) < min_experts: use top min_experts
   - If len(selected_experts) > max_experts: use top max_experts
   - Compute weights by normalizing p-values of selected experts

6. Return:
   - router_indices: [num_selected_experts]
   - router_weights: [num_selected_experts]
   - Pad to top_k if needed for compatibility

Key Considerations:
------------------

1. COMPATIBILITY: The rest of the code expects fixed-size outputs [num_tokens, top_k]
   - Solution: Pad router_indices and router_weights with zeros/dummy values
   - Or: Modify OlmoeExperts to handle variable-length expert lists

2. FDR LEVEL: How to set?
   - Fixed: fdr_level = 0.05 (standard)
   - Adaptive: fdr_level = f(layer_idx, token_importance, etc.)
   - Learned: Make fdr_level a learnable parameter

3. P-VALUE COMPUTATION:
   - Direct: p_value = 1 - router_prob (high prob → low p-value → more significant)
   - Statistical: Use token-expert compatibility score

4. WEIGHT COMPUTATION:
   - Option A: Use original router probabilities
   - Option B: Reweight based on BH selection (1/p-value)
   - Option C: Uniform weights for all selected experts

5. BATCHING:
   - Current: Process all tokens together with torch.topk
   - BH: May need per-token processing (slower but more principled)
   - Optimization: Vectorize BH procedure if possible

Implementation Pseudocode:
-------------------------

def benjamini_hochberg_routing(
    router_logits: torch.Tensor,  # [num_tokens, num_experts]
    fdr_level: float = 0.05,
    min_experts: int = 1,
    max_experts: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:

    num_tokens, num_experts = router_logits.shape
    device = router_logits.device

    # Initialize outputs
    router_indices = torch.zeros(num_tokens, max_experts, dtype=torch.long, device=device)
    router_weights = torch.zeros(num_tokens, max_experts, dtype=router_logits.dtype, device=device)

    # Process each token
    for token_idx in range(num_tokens):
        # Get probabilities for this token
        probs = router_logits[token_idx]  # [num_experts]

        # Convert to p-values (lower prob = higher p-value)
        p_values = 1.0 - probs

        # Sort p-values
        sorted_pvalues, sorted_indices = torch.sort(p_values)

        # Apply BH procedure
        num_selected = 0
        for i in range(num_experts):
            threshold = (i + 1) / num_experts * fdr_level
            if sorted_pvalues[i] <= threshold:
                num_selected = i + 1
            else:
                break

        # Enforce constraints
        num_selected = max(min_experts, min(num_selected, max_experts))

        # Get selected experts
        selected_indices = sorted_indices[:num_selected]
        selected_probs = probs[selected_indices]

        # Normalize weights
        selected_weights = selected_probs / selected_probs.sum()

        # Store (pad if needed)
        router_indices[token_idx, :num_selected] = selected_indices
        router_weights[token_idx, :num_selected] = selected_weights

    return router_weights, router_indices

Alternative: Vectorized BH
-------------------------

def vectorized_bh_routing(
    router_logits: torch.Tensor,  # [num_tokens, num_experts]
    fdr_level: float = 0.05,
    max_experts: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:

    num_tokens, num_experts = router_logits.shape

    # Convert to p-values
    p_values = 1.0 - router_logits

    # Sort
    sorted_pvalues, sorted_indices = torch.sort(p_values, dim=-1)

    # Compute BH thresholds
    ranks = torch.arange(1, num_experts + 1, device=p_values.device)
    thresholds = ranks / num_experts * fdr_level

    # Find cutoff for each token
    # significant[i,j] = True if expert j is significant for token i
    significant = sorted_pvalues <= thresholds.unsqueeze(0)

    # Find last significant expert for each token
    # This is tricky - need cumsum and masking
    # ... implementation details ...

    # Select top max_experts from significant ones
    # ... implementation details ...

    return router_weights, router_indices

Testing Plan:
------------

1. Unit tests:
   - Test BH procedure on synthetic data
   - Verify FDR control
   - Check edge cases (all/none significant)

2. Integration tests:
   - Replace router in small model
   - Verify output shapes match
   - Check gradients flow correctly

3. Evaluation:
   - Compare perplexity with standard routing
   - Analyze expert utilization
   - Measure FDR empirically
   - Check load balancing

4. Ablations:
   - Fixed vs adaptive FDR
   - Different p-value definitions
   - Different weight normalization schemes
"""


# ============================================================================
# VARIABLE NAMES QUICK REFERENCE
# ============================================================================

"""
CRITICAL VARIABLE NAMES FOR BH INJECTION
========================================

In OlmoeTopKRouter.forward():
-----------------------------
- self.weight: Router weight matrix [num_experts, hidden_dim] = [64, 2048]
- self.top_k: Number of experts per token = 8
- self.num_experts: Total experts = 64
- self.norm_topk_prob: Whether to renormalize (False)

- hidden_states: Input [num_tokens, hidden_dim] = [B*L, 2048]
- router_logits: Softmax probs [num_tokens, num_experts] = [B*L, 64]
- router_top_value: Selected weights [num_tokens, top_k] = [B*L, 8]
- router_indices: Selected expert IDs [num_tokens, top_k] = [B*L, 8]
- router_scores: Same as router_top_value

In OlmoeExperts.forward():
--------------------------
- top_k_index: Same as router_indices from router
- top_k_weights: Same as router_scores from router
- expert_mask: One-hot encoding [num_experts, top_k, num_tokens]

In load_balancing_loss_func():
------------------------------
- gate_logits: Tuple of router outputs from all layers
- routing_weights: Softmax of concatenated logits
- selected_experts: Top-k expert indices
- tokens_per_expert: Fraction of tokens per expert
- router_prob_per_expert: Average routing prob per expert
"""


if __name__ == "__main__":
    print("OLMoE Routing Mechanism Documentation")
    print("=" * 50)
    print("\nKey Classes:")
    print("  - OlmoeTopKRouter: Routing logic")
    print("  - OlmoeExperts: Expert computation")
    print("  - OlmoeSparseMoeBlock: Complete MoE block")
    print("\nBH Injection Point:")
    print("  - File: modeling_olmoe.py")
    print("  - Class: OlmoeTopKRouter")
    print("  - Method: forward()")
    print("  - Line: ~353 (torch.topk call)")
    print("\nNext Steps:")
    print("  1. Implement benjamini_hochberg_routing()")
    print("  2. Test on small examples")
    print("  3. Integrate into OlmoeTopKRouter")
    print("  4. Evaluate on real data")
