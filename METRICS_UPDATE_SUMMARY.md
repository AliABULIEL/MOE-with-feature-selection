# Comprehensive Metrics Update Summary

**Date:** 2026-01-16
**Scope:** Enhanced evaluation framework to compute both perplexity AND accuracy for all three datasets
**Files Modified:** `hc_routing_evaluation.py`
**Files Created:** `test_all_metrics.py`, `METRICS_UPDATE_SUMMARY.md`

---

## Executive Summary

The evaluation framework has been successfully enhanced to compute **BOTH perplexity AND accuracy** for all three benchmark datasets (WikiText, LAMBADA, HellaSwag). Previously, each dataset computed only one metric; now all datasets compute both metrics, enabling more comprehensive quality assessment of routing strategies.

### What Changed

**Before:**
- WikiText: ✅ Perplexity only
- LAMBADA: ✅ Accuracy only
- HellaSwag: ✅ Accuracy only

**After:**
- WikiText: ✅ Perplexity + ✅ Token Accuracy
- LAMBADA: ✅ Accuracy + ✅ Perplexity
- HellaSwag: ✅ Accuracy + ✅ Perplexity

### Backward Compatibility

✅ **100% Backward Compatible** - All existing function signatures are preserved. New metrics are added to return dictionaries without removing any existing keys.

### Test Results

All tests passed successfully:
```
Total Tests: 10/10
✅ Passed: 10/10
❌ Failed: 0/10
```

---

## Section 1: Files Modified

| File | Changes | Lines Added | Lines Modified |
|------|---------|-------------|----------------|
| `hc_routing_evaluation.py` | Added helper functions, enhanced all 3 evaluation functions | ~280 new lines | ~40 modified lines |
| `test_all_metrics.py` | New comprehensive test suite | ~550 new lines | N/A (new file) |
| `METRICS_UPDATE_SUMMARY.md` | This summary document | N/A | N/A (new file) |

---

## Section 2: API Changes

### 2.1 Function Signature Changes

All function signatures remain **UNCHANGED** for backward compatibility. Only return dictionaries have been enhanced with additional keys.

#### evaluate_perplexity()

**BEFORE:**
```python
evaluate_perplexity(model, tokenizer, dataset, ...) -> {
    'perplexity': float,
    'avg_loss': float,
    'losses': List[float],
    'token_counts': List[int],
    'internal_logs': Optional[List[Dict]],
    'num_samples': int,
    'total_tokens': int
}
```

**AFTER:**
```python
evaluate_perplexity(model, tokenizer, dataset, ...) -> {
    # EXISTING KEYS (unchanged)
    'perplexity': float,
    'avg_loss': float,
    'losses': List[float],
    'token_counts': List[int],
    'internal_logs': Optional[List[Dict]],
    'num_samples': int,
    'total_tokens': int,

    # NEW KEYS
    'accuracy': float,  # ⭐ Next-token prediction accuracy [0, 1]
    'correct_tokens': int,  # Total correct predictions
    'predictable_tokens': int  # Total predictable tokens
}
```

#### evaluate_lambada()

**BEFORE:**
```python
evaluate_lambada(model, tokenizer, dataset, ...) -> {
    'accuracy': float,
    'correct': int,
    'total': int,
    'predictions': List[Dict],
    'internal_logs': Optional[List[Dict]]
}
```

**AFTER:**
```python
evaluate_lambada(model, tokenizer, dataset, ...) -> {
    # EXISTING KEYS (unchanged)
    'accuracy': float,
    'correct': int,
    'total': int,
    'predictions': List[Dict],
    'internal_logs': Optional[List[Dict]],

    # NEW KEYS
    'perplexity': float,  # ⭐ Perplexity on full texts
    'avg_loss': float,  # Average cross-entropy loss
    'total_tokens': int,  # Tokens processed
    'num_samples': int  # Samples processed
}
```

#### evaluate_hellaswag()

**BEFORE:**
```python
evaluate_hellaswag(model, tokenizer, dataset, ...) -> {
    'accuracy': float,
    'correct': int,
    'total': int,
    'accuracy_raw': float,
    'accuracy_normalized': float,
    'correct_raw': int,
    'correct_normalized': int,
    'predictions': List[Dict],
    'internal_logs': Optional[List[Dict]]
}
```

**AFTER:**
```python
evaluate_hellaswag(model, tokenizer, dataset, ...) -> {
    # EXISTING KEYS (unchanged)
    'accuracy': float,
    'correct': int,
    'total': int,
    'accuracy_raw': float,
    'accuracy_normalized': float,
    'correct_raw': int,
    'correct_normalized': int,
    'predictions': List[Dict],
    'internal_logs': Optional[List[Dict]],

    # NEW KEYS
    'perplexity': float,  # ⭐ Perplexity on context + correct ending
    'avg_loss': float,  # Average cross-entropy loss
    'total_tokens': int,  # Tokens processed
    'num_samples': int  # Samples processed
}
```

### 2.2 New Helper Functions Added

Three new reusable helper functions were added to support metric computation:

| Function | Signature | Purpose |
|----------|-----------|---------|
| `compute_token_accuracy()` | `(model, tokenizer, texts, max_length, device) -> Dict` | Compute next-token prediction accuracy across all tokens |
| `compute_perplexity_from_texts()` | `(model, tokenizer, texts, max_length, device) -> Dict` | Compute perplexity on a list of text samples |
| `extract_texts_for_perplexity()` | `(dataset, dataset_type) -> List[str]` | Extract text samples from different dataset formats |

These functions are **reusable** and can be called independently for custom evaluations.

### 2.3 Standardized Return Format

All evaluation functions now return a **consistent set of keys** for easier comparison:

```python
{
    # Core metrics (ALL datasets now have both)
    'perplexity': float,  # exp(avg_loss)
    'accuracy': float,  # Task-specific accuracy [0, 1]
    'avg_loss': float,  # Cross-entropy loss (log of perplexity)

    # Sample statistics
    'total_tokens': int,  # Total tokens processed
    'num_samples': int,  # Number of samples evaluated

    # Dataset-specific details
    'correct': int,  # For accuracy tasks (LAMBADA, HellaSwag)
    'total': int,  # For accuracy tasks
    'correct_tokens': int,  # For WikiText token accuracy
    'predictable_tokens': int,  # For WikiText token accuracy

    # Other metadata
    'internal_logs': Optional[List[Dict]],  # Routing logs if patcher provided
    'detailed_logger': Optional[HCRoutingLogger]  # Detailed logger if enabled
}
```

---

## Section 3: Notebook Integration Guide

### 3.1 Current Notebook Code (Cell 29)

The notebook currently extracts metrics like this:

```python
# WikiText
if dataset_name == 'wikitext':
    eval_result = evaluate_perplexity(...)
    result['perplexity'] = eval_result['perplexity']
    if 'perplexity_token_weighted' in eval_result:
        result['perplexity_token_weighted'] = eval_result['perplexity_token_weighted']
    if 'perplexity_sample_weighted' in eval_result:
        result['perplexity_sample_weighted'] = eval_result['perplexity_sample_weighted']
    print(f"  ✅ Perplexity: {eval_result['perplexity']:.2f}")

# LAMBADA
elif dataset_name == 'lambada':
    eval_result = evaluate_lambada(...)
    result['lambada_accuracy'] = eval_result['accuracy']
    print(f"  ✅ LAMBADA Accuracy: {eval_result['accuracy']:.4f}")

# HellaSwag
elif dataset_name == 'hellaswag':
    eval_result = evaluate_hellaswag(...)
    result['hellaswag_accuracy'] = eval_result['accuracy']
    print(f"  ✅ HellaSwag Accuracy: {eval_result['accuracy']:.4f}")
```

### 3.2 Required Changes

The notebook needs to extract **both metrics** from each evaluation result. Here are the specific changes needed:

#### Change 1: WikiText Section

**Location:** After line extracting `perplexity`

**Add:**
```python
# NEW: Extract WikiText token accuracy
result['wikitext_accuracy'] = eval_result['accuracy']
```

**Updated print statement:**
```python
print(f"  ✅ Perplexity: {eval_result['perplexity']:.2f}, Accuracy: {eval_result['accuracy']:.4f}")
```

#### Change 2: LAMBADA Section

**Location:** After line extracting `lambada_accuracy`

**Add:**
```python
# NEW: Extract LAMBADA perplexity
result['lambada_perplexity'] = eval_result['perplexity']
```

**Updated print statement:**
```python
print(f"  ✅ LAMBADA Accuracy: {eval_result['accuracy']:.4f}, Perplexity: {eval_result['perplexity']:.2f}")
```

#### Change 3: HellaSwag Section

**Location:** After line extracting `hellaswag_accuracy`

**Add:**
```python
# NEW: Extract HellaSwag perplexity
result['hellaswag_perplexity'] = eval_result['perplexity']
```

**Updated print statement:**
```python
print(f"  ✅ HellaSwag Accuracy: {eval_result['accuracy']:.4f}, Perplexity: {eval_result['perplexity']:.2f}")
```

### 3.3 New Result Dictionary Keys

After these changes, the `result` dictionary will contain these new keys:

| Key | Type | Dataset | Description |
|-----|------|---------|-------------|
| `perplexity` | float | wikitext | Perplexity score (existing) |
| `wikitext_accuracy` | float | wikitext | **NEW:** Token prediction accuracy |
| `lambada_accuracy` | float | lambada | Last word accuracy (existing) |
| `lambada_perplexity` | float | lambada | **NEW:** Perplexity score |
| `hellaswag_accuracy` | float | hellaswag | Ending selection accuracy (existing) |
| `hellaswag_perplexity` | float | hellaswag | **NEW:** Perplexity score |

### 3.4 Example: Before vs After

#### BEFORE (current notebook code)

```python
if dataset_name == 'wikitext':
    eval_result = evaluate_perplexity(
        model=eval_model,
        tokenizer=eval_tokenizer,
        dataset=dataset_data,
        patcher=current_patcher,
        device=device,
        log_routing=LOG_ROUTING,
        output_dir=LOG_DIR,
        experiment_name=f"{config.name}_{dataset_name}",
        log_every_n=LOG_EVERY_N
    )
    result['perplexity'] = eval_result['perplexity']
    if 'perplexity_token_weighted' in eval_result:
        result['perplexity_token_weighted'] = eval_result['perplexity_token_weighted']
    if 'perplexity_sample_weighted' in eval_result:
        result['perplexity_sample_weighted'] = eval_result['perplexity_sample_weighted']
    print(f"  ✅ Perplexity: {eval_result['perplexity']:.2f}")

elif dataset_name == 'lambada':
    eval_result = evaluate_lambada(
        model=eval_model,
        tokenizer=eval_tokenizer,
        dataset=dataset_data,
        patcher=current_patcher,
        device=device,
        log_routing=LOG_ROUTING,
        output_dir=LOG_DIR,
        experiment_name=f"{config.name}_{dataset_name}",
        log_every_n=LOG_EVERY_N
    )
    result['lambada_accuracy'] = eval_result['accuracy']
    print(f"  ✅ LAMBADA Accuracy: {eval_result['accuracy']:.4f}")

elif dataset_name == 'hellaswag':
    eval_result = evaluate_hellaswag(
        model=eval_model,
        tokenizer=eval_tokenizer,
        dataset=dataset_data,
        patcher=current_patcher,
        device=device,
        log_routing=LOG_ROUTING,
        output_dir=LOG_DIR,
        experiment_name=f"{config.name}_{dataset_name}",
        log_every_n=LOG_EVERY_N
    )
    result['hellaswag_accuracy'] = eval_result['accuracy']
    print(f"  ✅ HellaSwag Accuracy: {eval_result['accuracy']:.4f}")
```

#### AFTER (what notebook should become)

```python
if dataset_name == 'wikitext':
    eval_result = evaluate_perplexity(
        model=eval_model,
        tokenizer=eval_tokenizer,
        dataset=dataset_data,
        patcher=current_patcher,
        device=device,
        log_routing=LOG_ROUTING,
        output_dir=LOG_DIR,
        experiment_name=f"{config.name}_{dataset_name}",
        log_every_n=LOG_EVERY_N
    )
    result['perplexity'] = eval_result['perplexity']
    result['wikitext_accuracy'] = eval_result['accuracy']  # ⭐ NEW
    if 'perplexity_token_weighted' in eval_result:
        result['perplexity_token_weighted'] = eval_result['perplexity_token_weighted']
    if 'perplexity_sample_weighted' in eval_result:
        result['perplexity_sample_weighted'] = eval_result['perplexity_sample_weighted']
    print(f"  ✅ Perplexity: {eval_result['perplexity']:.2f}, Accuracy: {eval_result['accuracy']:.4f}")  # ⭐ UPDATED

elif dataset_name == 'lambada':
    eval_result = evaluate_lambada(
        model=eval_model,
        tokenizer=eval_tokenizer,
        dataset=dataset_data,
        patcher=current_patcher,
        device=device,
        log_routing=LOG_ROUTING,
        output_dir=LOG_DIR,
        experiment_name=f"{config.name}_{dataset_name}",
        log_every_n=LOG_EVERY_N
    )
    result['lambada_accuracy'] = eval_result['accuracy']
    result['lambada_perplexity'] = eval_result['perplexity']  # ⭐ NEW
    print(f"  ✅ LAMBADA Accuracy: {eval_result['accuracy']:.4f}, Perplexity: {eval_result['perplexity']:.2f}")  # ⭐ UPDATED

elif dataset_name == 'hellaswag':
    eval_result = evaluate_hellaswag(
        model=eval_model,
        tokenizer=eval_tokenizer,
        dataset=dataset_data,
        patcher=current_patcher,
        device=device,
        log_routing=LOG_ROUTING,
        output_dir=LOG_DIR,
        experiment_name=f"{config.name}_{dataset_name}",
        log_every_n=LOG_EVERY_N
    )
    result['hellaswag_accuracy'] = eval_result['accuracy']
    result['hellaswag_perplexity'] = eval_result['perplexity']  # ⭐ NEW
    print(f"  ✅ HellaSwag Accuracy: {eval_result['accuracy']:.4f}, Perplexity: {eval_result['perplexity']:.2f}")  # ⭐ UPDATED
```

### 3.5 CSV Output Updates

After making these notebook changes, the CSV output will include **all these new columns**:

#### Expected CSV Columns (Full List)

| Column Name | Dataset | Metric Type | Example Value |
|-------------|---------|-------------|---------------|
| `dataset` | ALL | identifier | 'wikitext', 'lambada', 'hellaswag' |
| `routing_type` | ALL | identifier | 'topk', 'hc' |
| `perplexity` | wikitext | quality | 18.45 |
| `wikitext_accuracy` | wikitext | quality | 0.6234 |
| `lambada_accuracy` | lambada | quality | 0.4512 |
| `lambada_perplexity` | lambada | quality | 12.78 |
| `hellaswag_accuracy` | hellaswag | quality | 0.3821 |
| `hellaswag_perplexity` | hellaswag | quality | 15.92 |
| `avg_experts` | ALL | routing | 5.42 |
| `floor_hit_rate` | ALL | routing | 12.3 |
| `ceiling_hit_rate` | ALL | routing | 8.7 |

#### Sample CSV Row (After Updates)

```csv
dataset,routing_type,perplexity,wikitext_accuracy,lambada_accuracy,lambada_perplexity,hellaswag_accuracy,hellaswag_perplexity,avg_experts,...
wikitext,hc,18.45,0.6234,-,-,-,-,5.42,...
lambada,hc,-,-,0.4512,12.78,-,-,5.38,...
hellaswag,hc,-,-,-,-,0.3821,15.92,5.51,...
```

---

## Section 4: Testing Instructions

### 4.1 Run Automated Tests

The comprehensive test suite validates all changes:

```bash
cd /Users/aliab/Desktop/GitHub/MOE-with-feature-selection/
python3 test_all_metrics.py
```

**Expected Output:**
```
======================================================================
TEST SUMMARY
======================================================================
Total Tests: 10
✅ Passed: 10/10
❌ Failed: 0/10
======================================================================
```

### 4.2 Manual Validation (Mini Evaluation)

Test with a small number of samples to verify integration:

```python
from hc_routing_evaluation import (
    load_wikitext, load_lambada, load_hellaswag,
    evaluate_perplexity, evaluate_lambada, evaluate_hellaswag
)
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "allenai/OLMoE-1B-7B-0924"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Test WikiText (5 samples)
wikitext = load_wikitext(max_samples=5)
result = evaluate_perplexity(model, tokenizer, wikitext, device='cpu')
print(f"WikiText - Perplexity: {result['perplexity']:.2f}, Accuracy: {result['accuracy']:.4f}")
assert 'perplexity' in result and 'accuracy' in result, "Missing metrics!"

# Test LAMBADA (5 samples)
lambada = load_lambada(max_samples=5)
result = evaluate_lambada(model, tokenizer, lambada, device='cpu')
print(f"LAMBADA - Accuracy: {result['accuracy']:.4f}, Perplexity: {result['perplexity']:.2f}")
assert 'accuracy' in result and 'perplexity' in result, "Missing metrics!"

# Test HellaSwag (5 samples)
hellaswag = load_hellaswag(max_samples=5)
result = evaluate_hellaswag(model, tokenizer, hellaswag, device='cpu')
print(f"HellaSwag - Accuracy: {result['accuracy']:.4f}, Perplexity: {result['perplexity']:.2f}")
assert 'accuracy' in result and 'perplexity' in result, "Missing metrics!"

print("\n✅ All validations passed!")
```

### 4.3 Verify CSV Output

After running the full notebook evaluation:

```python
import pandas as pd

# Load the results CSV
df = pd.read_csv('./metrics/all_experiments_metrics.csv')

# Check new columns exist
required_columns = [
    'wikitext_accuracy',
    'lambada_perplexity',
    'hellaswag_perplexity'
]

for col in required_columns:
    assert col in df.columns, f"Missing column: {col}"

print("✅ CSV has all required columns!")
print(df[['dataset', 'perplexity', 'wikitext_accuracy', 'lambada_accuracy',
          'lambada_perplexity', 'hellaswag_accuracy', 'hellaswag_perplexity']].head())
```

---

## Section 5: Migration Checklist

Use this checklist to ensure smooth migration:

### Pre-Migration
- [ ] 1. Backup current `hc_routing_evaluation.py`
  ```bash
  cp hc_routing_evaluation.py hc_routing_evaluation.py.backup
  ```
- [ ] 2. Read this summary document thoroughly
- [ ] 3. Understand the new return format for each function

### Testing Phase
- [ ] 4. Run automated test suite
  ```bash
  python3 test_all_metrics.py
  ```
- [ ] 5. Verify all 10 tests pass
- [ ] 6. Run manual validation with 5 samples per dataset (see Section 4.2)

### Notebook Updates
- [ ] 7. Open `OLMoE_HC_WorkFlow.ipynb` Cell 29
- [ ] 8. Add `result['wikitext_accuracy'] = eval_result['accuracy']` to WikiText section
- [ ] 9. Add `result['lambada_perplexity'] = eval_result['perplexity']` to LAMBADA section
- [ ] 10. Add `result['hellaswag_perplexity'] = eval_result['perplexity']` to HellaSwag section
- [ ] 11. Update all print statements to show both metrics (see Section 3.4)

### Validation Phase
- [ ] 12. Run mini evaluation (5 samples per dataset)
- [ ] 13. Verify console output shows both metrics for each dataset
- [ ] 14. Check `result` dictionary has all new keys
- [ ] 15. Verify CSV has new columns: `wikitext_accuracy`, `lambada_perplexity`, `hellaswag_perplexity`

### Full Run
- [ ] 16. Run full evaluation (200 samples per dataset)
- [ ] 17. Verify metrics look reasonable (perplexity > 1.0, accuracy in [0, 1])
- [ ] 18. Check CSV for data quality
- [ ] 19. Compare with previous results to ensure consistency

### Documentation
- [ ] 20. Update any experiment logs or documentation to reflect new metrics
- [ ] 21. Archive this summary document for future reference

---

## Section 6: Technical Details

### 6.1 How Token Accuracy is Computed (WikiText)

Token accuracy measures the percentage of next-token predictions that are correct:

```python
# For each token in the sequence (except the first):
#   1. Use previous tokens as context
#   2. Predict next token (argmax of logits)
#   3. Compare with actual next token
#   4. Count if correct

# Example sequence: [A, B, C, D]
# Predictions:
#   Context []    → Predict A (no context, skip)
#   Context [A]   → Predict B (compare with actual B)
#   Context [A,B] → Predict C (compare with actual C)
#   Context [A,B,C] → Predict D (compare with actual D)

# Accuracy = (correct predictions) / (total predictions)
#          = (# correct) / 3  (for this 4-token sequence)
```

### 6.2 How Perplexity is Computed (LAMBADA & HellaSwag)

For LAMBADA and HellaSwag, perplexity is computed on the **full text** after the accuracy evaluation:

**LAMBADA:** Perplexity on complete sentences (context + target word)
```python
texts = [sample['text'] for sample in dataset]  # Full sentences
perplexity = compute_perplexity_from_texts(model, tokenizer, texts)
```

**HellaSwag:** Perplexity on context + **correct** ending
```python
texts = [f"{sample['ctx']} {sample['endings'][sample['label']]}" for sample in dataset]
perplexity = compute_perplexity_from_texts(model, tokenizer, texts)
```

This uses the same token-weighted perplexity formula as WikiText:
```python
perplexity = exp(sum(loss_i * tokens_i) / sum(tokens_i))
```

### 6.3 Edge Cases Handled

The implementation gracefully handles:

1. **Empty datasets:** Returns `perplexity=inf`, `accuracy=0.0`, `num_samples=0`
2. **Empty samples:** Skipped during iteration, not counted
3. **Single-token sequences:** Skipped for accuracy (need ≥2 tokens for next-token prediction)
4. **Malformed samples:** Checked for required fields, skipped if missing

### 6.4 Performance Considerations

**Memory Usage:**
- WikiText: Computes accuracy during perplexity pass (single forward pass)
- LAMBADA: Runs accuracy loop first, then separate perplexity pass
- HellaSwag: Runs accuracy loop first (4 endings × N samples), then separate perplexity pass

**Optimization Tip:** If memory is constrained, consider reducing `max_length` from 512 to 256.

---

## Section 7: Troubleshooting

### Issue 1: "KeyError: 'accuracy'" when running old notebook

**Cause:** Notebook hasn't been updated yet
**Solution:** Follow Section 3 to add the new result dictionary keys

### Issue 2: Perplexity values seem very high (>1000)

**Possible Causes:**
- Model not loaded correctly
- Wrong device (CPU vs GPU mismatch)
- Tokenizer mismatch

**Solution:**
```python
# Verify model and tokenizer match
assert model.config.vocab_size == tokenizer.vocab_size
# Check device
print(next(model.parameters()).device)
```

### Issue 3: Accuracy always 0.0 for LAMBADA/HellaSwag

**Possible Causes:**
- Mock model being used (normal for tests)
- Generation settings incorrect
- Model not properly loaded

**Solution:** For real evaluation, use actual OLMoE model, not mock model

### Issue 4: Tests pass but notebook fails

**Possible Causes:**
- Model size too large for available memory
- Dataset loading issues

**Solution:**
```python
# Test with smaller sample size first
result = evaluate_perplexity(model, tokenizer, dataset[:5], device='cpu')
```

---

## Section 8: Quick Reference

### What Metrics Mean

| Metric | Dataset | Interpretation | Good Value |
|--------|---------|----------------|------------|
| Perplexity (WikiText) | wikitext | Model's uncertainty per token | Lower is better (10-30) |
| Accuracy (WikiText) | wikitext | % of tokens predicted correctly | Higher is better (30-70%) |
| Accuracy (LAMBADA) | lambada | % of last words predicted correctly | Higher is better (40-70%) |
| Perplexity (LAMBADA) | lambada | Model's uncertainty on full text | Lower is better (5-20) |
| Accuracy (HellaSwag) | hellaswag | % of correct endings selected | Higher is better (30-60%) |
| Perplexity (HellaSwag) | hellaswag | Model's uncertainty on completions | Lower is better (8-25) |

### Common Commands

```bash
# Run tests
python3 test_all_metrics.py

# Quick validation (in Python)
from hc_routing_evaluation import evaluate_perplexity
result = evaluate_perplexity(model, tokenizer, texts[:5], device='cpu')
print(result.keys())  # Should include 'perplexity' and 'accuracy'

# Check CSV columns
import pandas as pd
df = pd.read_csv('metrics/all_experiments_metrics.csv')
print(df.columns.tolist())
```

---

## Appendix A: Complete List of Changes

### New Functions

1. `compute_token_accuracy()` - Lines 46-134
2. `compute_perplexity_from_texts()` - Lines 137-222
3. `extract_texts_for_perplexity()` - Lines 225-278

### Modified Functions

1. `evaluate_perplexity()` - Lines 391-595
   - Added accuracy computation in forward pass loop
   - Added 'accuracy', 'correct_tokens', 'predictable_tokens' to return dict

2. `evaluate_lambada()` - Lines 598-771
   - Added perplexity computation after accuracy loop
   - Added 'perplexity', 'avg_loss', 'total_tokens', 'num_samples' to return dict

3. `evaluate_hellaswag()` - Lines 774-981
   - Added perplexity computation after accuracy loop
   - Added 'perplexity', 'avg_loss', 'total_tokens', 'num_samples' to return dict

4. `evaluate_all_datasets()` - Lines 1026-1051
   - Updated print statements to show both metrics

---

## Appendix B: Testing Output

```
======================================================================
COMPREHENSIVE METRICS TEST SUITE
======================================================================
Testing perplexity + accuracy for all datasets
======================================================================

TEST: compute_token_accuracy()
✅ PASSED - Accuracy: 1.0000, Correct: 23/23

TEST: compute_perplexity_from_texts()
✅ PASSED - Perplexity: 3.00, Tokens: 26, Samples: 3

TEST: extract_texts_for_perplexity()
✅ PASSED - All format extractions work

TEST: evaluate_perplexity() - BOTH METRICS
✅ PASSED - Perplexity: 2.97, Accuracy: 1.0000

TEST: evaluate_lambada() - BOTH METRICS
✅ PASSED - Accuracy: 0.0000, Perplexity: 2.57

TEST: evaluate_hellaswag() - BOTH METRICS
✅ PASSED - Accuracy: 0.0000, Perplexity: 5.76

TEST: Return Format Consistency
✅ PASSED - All datasets return consistent format

TEST: Edge Case - Empty Dataset
✅ PASSED - Empty dataset handled gracefully

TEST: Edge Case - Single Sample
✅ PASSED - Single sample handled correctly

TEST: Mathematical Consistency
✅ PASSED - Metrics are mathematically consistent

======================================================================
TEST SUMMARY: 10/10 PASSED
======================================================================
```

---

**End of Summary Document**
