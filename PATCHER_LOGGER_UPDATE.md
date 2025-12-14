# OLMoERouterPatcher Logger Integration

## Issue

The current `OLMoERouterPatcher` class in the notebook doesn't support passing a logger to the `benjamini_hochberg_routing()` function.

## Solution

Modify the `OLMoERouterPatcher` class to:
1. Store logger as an instance variable
2. Accept logger parameter in `patch_with_bh()`
3. Pass logger to `benjamini_hochberg_routing()` in the custom forward method

## Required Changes to Section 6 (OLMoERouterPatcher class)

### 1. Add logger instance variable in `__init__`

```python
def __init__(self, model: OlmoeForCausalLM):
    self.model = model
    self.moe_blocks = []
    self.original_forwards = {}
    self.stats = defaultdict(list)
    self.patched = False
    self.logger = None  # NEW: Store logger instance

    self._find_moe_blocks()
```

### 2. Update `patch_with_bh()` signature

```python
def patch_with_bh(
    self,
    alpha: float = 0.05,
    temperature: float = 1.0,
    min_k: int = 1,
    max_k: int = 8,
    collect_stats: bool = True,
    logger: Optional['BHRoutingLogger'] = None  # NEW
):
    """
    Patch model to use BH routing with optional logging.

    Args:
        ...existing args...
        logger: Optional BHRoutingLogger instance for detailed logging
    """
    self.unpatch()
    self.stats.clear()
    self.logger = logger  # NEW: Store logger

    # ... rest of existing code ...
```

### 3. Modify `create_bh_forward()` to use logger

**Current code** (around line 216 in the notebook):
```python
# Step 3: Apply BH routing with CORRECT layer_idx and kde_models
routing_weights, selected_experts, expert_counts = benjamini_hochberg_routing(
    router_logits,
    alpha=alpha,
    temperature=temperature,
    min_k=min_k,
    max_k=max_k,
    layer_idx=layer_idx,
    kde_models=kde_models
)
```

**Updated code** (with logger support):
```python
# Step 3: Apply BH routing with CORRECT layer_idx, kde_models, and logger
routing_weights, selected_experts, expert_counts = benjamini_hochberg_routing(
    router_logits,
    alpha=alpha,
    temperature=temperature,
    min_k=min_k,
    max_k=max_k,
    layer_idx=layer_idx,
    kde_models=kde_models,
    logger=self.logger,        # NEW: Pass stored logger
    log_every_n_tokens=100,    # NEW: Sampling rate
    sample_idx=0,              # NEW: Default sample index
    token_idx=0                # NEW: Will be tracked internally
)
```

### 4. Optional: Add token tracking for better logging

To improve logging accuracy, add a token counter:

```python
def __init__(self, model: OlmoeForCausalLM):
    # ... existing code ...
    self.logger = None
    self.token_counter = 0  # NEW: Track tokens for logging
```

And increment it in the forward method:
```python
def bh_forward(hidden_states):
    # ... existing code ...

    # Increment token counter for logging
    if self.logger is not None:
        token_idx = self.token_counter
        self.token_counter += 1
    else:
        token_idx = 0

    # Step 3: Apply BH routing
    routing_weights, selected_experts, expert_counts = benjamini_hochberg_routing(
        router_logits,
        alpha=alpha,
        temperature=temperature,
        min_k=min_k,
        max_k=max_k,
        layer_idx=layer_idx,
        kde_models=kde_models,
        logger=self.logger,
        log_every_n_tokens=100,
        sample_idx=0,
        token_idx=token_idx  # Use tracked token index
    )
    # ... rest of code ...
```

## Complete Updated `patch_with_bh()` Method

Here's the complete updated method:

```python
def patch_with_bh(
    self,
    alpha: float = 0.05,
    temperature: float = 1.0,
    min_k: int = 1,
    max_k: int = 8,
    collect_stats: bool = True,
    logger: Optional['BHRoutingLogger'] = None
):
    """
    Patch model to use BH routing using DIRECT METHOD REPLACEMENT.

    Args:
        alpha: FDR control level
        temperature: Softmax temperature
        min_k: Minimum experts per token
        max_k: Maximum experts per token
        collect_stats: Whether to collect routing statistics
        logger: Optional BHRoutingLogger for detailed logging
    """
    from bh_routing import load_kde_models

    self.unpatch()
    self.stats.clear()
    self.logger = logger  # Store logger
    self.token_counter = 0  # Reset token counter

    # Load KDE models
    kde_models = load_kde_models()
    if kde_models:
        print(f"   📊 Loaded KDE models for {len(kde_models)} layers")
    else:
        print(f"   ⚠️  No KDE models found - using empirical fallback")

    if logger is not None:
        print(f"   📝 Logging enabled for experiment: {logger.experiment_name}")

    def create_bh_forward(layer_name, moe_block_ref):
        # Extract layer index
        layer_idx = 0
        if 'layers.' in layer_name:
            try:
                parts = layer_name.split('.')
                for i, part in enumerate(parts):
                    if part == 'layers' and i + 1 < len(parts):
                        layer_idx = int(parts[i + 1])
                        break
            except (ValueError, IndexError):
                layer_idx = 0

        def bh_forward(hidden_states):
            # Get token index for logging
            if self.logger is not None:
                token_idx = self.token_counter
                self.token_counter += 1
            else:
                token_idx = 0

            # Flatten input
            batch_size, seq_len, hidden_dim = hidden_states.shape
            hidden_states_flat = hidden_states.view(-1, hidden_dim)

            # Compute router logits
            router_logits = moe_block_ref.gate(hidden_states_flat)

            # Apply BH routing WITH LOGGER
            routing_weights, selected_experts, expert_counts = benjamini_hochberg_routing(
                router_logits,
                alpha=alpha,
                temperature=temperature,
                min_k=min_k,
                max_k=max_k,
                layer_idx=layer_idx,
                kde_models=kde_models,
                logger=self.logger,        # Pass logger
                log_every_n_tokens=100,    # Sampling rate
                sample_idx=0,
                token_idx=token_idx        # Tracked token index
            )

            # Collect statistics
            if collect_stats:
                self.stats['expert_counts'].extend(expert_counts.flatten().cpu().tolist())
                self.stats['layer_names'].extend([layer_name] * expert_counts.numel())

            # Dispatch to experts
            final_hidden_states = torch.zeros_like(hidden_states_flat)
            for expert_idx in range(moe_block_ref.num_experts):
                expert_mask = routing_weights[:, expert_idx] > 0
                if expert_mask.any():
                    expert_input = hidden_states_flat[expert_mask]
                    expert_output = moe_block_ref.experts[expert_idx](expert_input)
                    weights = routing_weights[expert_mask, expert_idx].unsqueeze(-1)
                    final_hidden_states[expert_mask] += weights * expert_output

            # Reshape and return
            output = final_hidden_states.view(batch_size, seq_len, hidden_dim)
            return output, router_logits

        return bh_forward

    # Replace forward methods
    for name, moe_block in self.moe_blocks:
        self.original_forwards[name] = moe_block.forward
        replacement_forward = create_bh_forward(name, moe_block)
        moe_block.forward = replacement_forward

    self.patched = True

    print(f"✅ Replaced forward() on {len(self.moe_blocks)} MoE blocks with BH routing")
    print(f"   🎯 DIRECT METHOD REPLACEMENT - Original TopK routing NEVER executes!")
    print(f"   Parameters: alpha={alpha}, temperature={temperature}, min_k={min_k}, max_k={max_k}")
```

## Implementation Steps

1. **Read the current notebook Section 6 cell**
2. **Locate the `OLMoERouterPatcher` class definition**
3. **Make the three changes:**
   - Add `self.logger = None` in `__init__`
   - Add `logger` parameter to `patch_with_bh()`
   - Modify the `benjamini_hochberg_routing()` call to pass logger

4. **Save the updated notebook**

## Automated Update Script

I can create a script to automatically make these changes if needed. The script would:
1. Parse the notebook JSON
2. Find the OLMoERouterPatcher cell
3. Modify the source code
4. Save the updated notebook

Would you like me to create this automated update script?

## Verification

After making these changes, verify with:

```python
# In notebook Section 9.5
logger = BHRoutingLogger(
    output_dir=str(OUTPUT_DIR),
    experiment_name="test_experiment",
    log_every_n=100
)

patcher.patch_with_bh(
    alpha=0.30,
    max_k=8,
    logger=logger  # Pass logger to patcher
)

# Run some inference
# Check that logs are created
logger.save_logs()
logger.generate_plots()
```

Expected output:
- Logs saved in `OUTPUT_DIR/logs/test_experiment_bh_log.json`
- Summary saved in `OUTPUT_DIR/logs/test_experiment_summary.json`
- Plots generated in `OUTPUT_DIR/plots/test_experiment/`

## Summary

The patcher modification enables:
- ✅ Seamless logger integration
- ✅ Per-token logging with sampling
- ✅ Automatic token tracking
- ✅ Zero overhead when logger=None
- ✅ Clean API: just pass logger to `patch_with_bh()`

This completes the comprehensive BH routing logging implementation!
