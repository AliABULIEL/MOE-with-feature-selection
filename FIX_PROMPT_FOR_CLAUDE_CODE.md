# Fix OLMoE Routing Experiments: All Samples Failing

## Problem
All 500 samples are being skipped with "WARNING: Skipped 500 samples due to errors" across ALL expert configurations (4, 8, 16, 32, 64). Zero samples are processing successfully.

## Root Cause (Found from Olmoe_tests.ipynb)

The current code relies on `output_router_logits=True` parameter:
```python
self.model = OlmoeForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map=device,
    output_router_logits=True  # ❌ This doesn't work reliably!
)
```

**However, the working notebook (Olmoe_tests.ipynb) shows the CORRECT approach:**
- Does NOT use `output_router_logits=True`
- Instead uses **forward hooks** to capture router outputs directly from `layer.mlp.gate`
- This is more reliable and gives direct access to routing logits

## Solution: Use the Notebook's Hook-Based Approach

### Key Code from Olmoe_tests.ipynb (Cell 7):

```python
# List to store the logged data
logged_routing_data = []

def logging_hook_router(module, input, output, layer_index, k=8):
    """Hook to capture router outputs from gate module"""
    try:
        router_logits = output
        softmax_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        topk_weights, topk_indices = torch.topk(softmax_weights, k, dim=-1)
        
        logged_routing_data.append({
            "layer": layer_index,
            "expert_indices": topk_indices.detach().cpu().numpy(),
            "softmax_weights": topk_weights.detach().cpu().numpy(),
        })
    except Exception as e:
        warnings.warn(f"Error in hook for layer {layer_index}: {e}")

# Register hooks on each layer's gate
for i, layer in enumerate(model.model.layers):
    try:
        router_module = layer.mlp.gate
        router_module.register_forward_hook(
            lambda m, input, output, idx=i: logging_hook_router(m, input, output, f"Layer_{idx}")
        )
    except AttributeError:
        print(f"Layer {i} has no 'gate'. Skipping")
```

### Model Loading from Notebook (Cell 5):

```python
model = OlmoeForCausalLM.from_pretrained(
    model_id, 
    dtype=torch.bfloat16  # Simple! No output_router_logits
).to(device)
model.eval()

# Then set expert count directly:
model.config.num_experts_per_tok = 20
```

## What Needs to Change in olmoe_routing_experiments.py

### 1. Fix Model Loading in `__init__`
- Remove `output_router_logits=True` parameter (it doesn't work)
- Use `torch_dtype` instead of `dtype`
- Simplify to match notebook's approach

### 2. Add Hook-Based Router Logging System
Create a new class method to register hooks that capture router_logits:
- Register forward hooks on `layer.mlp.gate` for each layer
- Store routing data in instance variable accessible during evaluation
- Clear hooks between experiments to avoid memory leaks

### 3. Modify `evaluate_configuration` Method
- Instead of relying on `outputs.router_logits`, use the hook-captured data
- The hooks will populate routing data automatically during forward pass
- Process the hook-captured data the same way current code processes `outputs.router_logits`

### 4. Fix Error Handling
Current code silently swallows errors:
```python
except Exception as e:
    num_errors += 1  # Just counts!
    continue
```

Should log first 5 errors with details:
```python
except Exception as e:
    num_errors += 1
    if num_errors <= 5:
        logger.error(f"Sample {idx} failed: {type(e).__name__}: {str(e)}")
        logger.error(f"  Text: {text[:100]}...")
    continue
```

## Implementation Steps

1. **Create a hook management system**:
   - `_register_router_hooks()` method
   - `_clear_router_hooks()` method  
   - Store hook handles to remove them later

2. **Update model initialization**:
   - Use simple loading like notebook
   - Register hooks after loading

3. **Modify evaluation loop**:
   - Clear routing data before each sample
   - Run forward pass (hooks capture routing data automatically)
   - Process captured routing data instead of `outputs.router_logits`

4. **Better error handling**:
   - Log actual errors
   - Add validation checks

## Expected Outcome

After fix:
- Samples should process successfully
- Routing data captured via hooks (more reliable than output_router_logits)
- Proper error messages if something still fails

## Test Command

```python
runner = RoutingExperimentRunner()
results = runner.run_all_experiments(
    expert_counts=[4],
    strategies=['baseline'],
    datasets=['wikitext'],
    max_samples=10
)
```

Should show: `Results: PPL=X.XX, Acc=0.XXXX, Speed=XXX tok/s` instead of "Skipped 500 samples"

## Key Insight from Notebook

The notebook successfully captures routing data WITHOUT using `output_router_logits=True`. This is the proven working approach - use hooks on `layer.mlp.gate` instead of relying on HF's output_router_logits parameter.
