"""
Comprehensive Test Suite for All Metrics Evaluation
=====================================================

Tests both perplexity and accuracy computation across all three datasets:
- WikiText: perplexity + token accuracy
- LAMBADA: accuracy + perplexity
- HellaSwag: accuracy + perplexity

Usage:
    python test_all_metrics.py
"""

import torch
import numpy as np
from typing import Dict, List, Any
import sys
from unittest.mock import Mock, MagicMock, patch


# ==============================================================================
# MOCK DATA SAMPLES
# ==============================================================================

# WikiText samples (list of text strings)
WIKITEXT_SAMPLES = [
    'The quick brown fox jumps over the lazy dog.',
    'Machine learning is a subset of artificial intelligence.',
    'Python is a high-level programming language.',
    '',  # Edge case: empty
]

# LAMBADA samples (dict with 'text' and 'target' keys)
LAMBADA_SAMPLES = [
    {'text': 'The capital of France is Paris', 'target': 'Paris'},
    {'text': 'She went to the store to buy milk', 'target': 'milk'},
    {'text': 'The sun rises in the east', 'target': 'east'},
    {'text': '', 'target': ''},  # Edge case: empty
]

# HellaSwag samples (dict with 'ctx', 'endings', 'label' keys)
HELLASWAG_SAMPLES = [
    {
        'ctx': 'A man is sitting on a roof.',
        'endings': [
            'He is installing shingles.',
            'He is swimming.',
            'He is eating a sandwich.',
            'He is reading.'
        ],
        'label': 0
    },
    {
        'ctx': 'A woman is holding a tennis racket.',
        'endings': [
            'She hits the ball.',
            'She plays piano.',
            'She cooks dinner.',
            'She reads a book.'
        ],
        'label': 0
    },
]


# ==============================================================================
# MOCK MODEL AND TOKENIZER
# ==============================================================================

class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.eos_token_id = 0
        self.vocab_size = 50000

    def __call__(self, text, return_tensors=None, truncation=True, max_length=512, padding=False):
        """Tokenize text into mock tensor."""
        if isinstance(text, str):
            # Simple mock: split by whitespace and assign IDs
            tokens = text.strip().split() if text.strip() else []
            # Map to IDs (use hash for deterministic but varied IDs)
            input_ids = [hash(token) % self.vocab_size for token in tokens]
            # Add BOS token (ID=1)
            input_ids = [1] + input_ids[:max_length-1]

            if return_tensors == 'pt':
                return {'input_ids': torch.tensor([input_ids])}
            return {'input_ids': input_ids}
        return {'input_ids': torch.tensor([[1]])}

    def decode(self, token_ids, skip_special_tokens=True):
        """Mock decode."""
        if isinstance(token_ids, list):
            # Return a mock word based on token ID
            if len(token_ids) > 0:
                token_id = token_ids[0]
                # Map IDs to some common words for LAMBADA testing
                word_map = {
                    hash('Paris') % self.vocab_size: 'Paris',
                    hash('milk') % self.vocab_size: 'milk',
                    hash('east') % self.vocab_size: 'east',
                }
                return word_map.get(token_id % self.vocab_size, f'word_{token_id}')
        return ''


class MockModelOutput:
    """Mock output from model forward pass."""

    def __init__(self, input_ids, vocab_size=50000):
        batch_size, seq_len = input_ids.shape

        # Create mock logits [batch_size, seq_len, vocab_size]
        self.logits = torch.randn(batch_size, seq_len, vocab_size)

        # Make the correct next token have highest logit (for accuracy testing)
        # This simulates a perfect model for testing purposes
        for b in range(batch_size):
            for t in range(seq_len - 1):
                # Make the logit for the actual next token the highest
                true_next_token = input_ids[b, t + 1].item()
                self.logits[b, t, true_next_token] = self.logits[b, t].max() + 1.0

        # Compute cross-entropy loss
        # For mock: use small random loss
        self.loss = torch.tensor(np.random.uniform(0.5, 2.0))


class MockModel:
    """Mock OLMoE model for testing."""

    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.eval_mode = False

    def eval(self):
        """Set model to eval mode."""
        self.eval_mode = True
        return self

    def __call__(self, input_ids, labels=None):
        """Forward pass."""
        return MockModelOutput(input_ids, self.vocab_size)

    def generate(self, input_ids, max_new_tokens=1, do_sample=False, pad_token_id=0):
        """Mock generation."""
        # Return input_ids with one additional token
        batch_size, seq_len = input_ids.shape
        # Add a random token
        new_token = torch.randint(0, self.vocab_size, (batch_size, 1))
        return torch.cat([input_ids, new_token], dim=1)


# ==============================================================================
# TEST FUNCTIONS
# ==============================================================================

def test_helper_compute_token_accuracy():
    """Test compute_token_accuracy helper function."""
    from hc_routing_evaluation import compute_token_accuracy

    print("\n" + "="*70)
    print("TEST: compute_token_accuracy()")
    print("="*70)

    model = MockModel()
    tokenizer = MockTokenizer()
    texts = [s for s in WIKITEXT_SAMPLES if s.strip()]

    result = compute_token_accuracy(
        model, tokenizer, texts, max_length=512, device='cpu'
    )

    # Validate output structure
    assert 'accuracy' in result, "Missing 'accuracy' key"
    assert 'correct_tokens' in result, "Missing 'correct_tokens' key"
    assert 'total_tokens' in result, "Missing 'total_tokens' key"
    assert 'per_sample_accuracy' in result, "Missing 'per_sample_accuracy' key"

    # Validate ranges
    assert 0 <= result['accuracy'] <= 1, f"Accuracy out of range: {result['accuracy']}"
    assert result['correct_tokens'] >= 0, "Negative correct_tokens"
    assert result['total_tokens'] > 0, "No tokens processed"
    assert result['correct_tokens'] <= result['total_tokens'], "More correct than total"

    print(f"✅ PASSED")
    print(f"   Accuracy: {result['accuracy']:.4f}")
    print(f"   Correct: {result['correct_tokens']}/{result['total_tokens']}")

    return result


def test_helper_compute_perplexity_from_texts():
    """Test compute_perplexity_from_texts helper function."""
    from hc_routing_evaluation import compute_perplexity_from_texts

    print("\n" + "="*70)
    print("TEST: compute_perplexity_from_texts()")
    print("="*70)

    model = MockModel()
    tokenizer = MockTokenizer()
    texts = [s for s in WIKITEXT_SAMPLES if s.strip()]

    result = compute_perplexity_from_texts(
        model, tokenizer, texts, max_length=512, device='cpu'
    )

    # Validate output structure
    assert 'perplexity' in result, "Missing 'perplexity' key"
    assert 'avg_loss' in result, "Missing 'avg_loss' key"
    assert 'total_tokens' in result, "Missing 'total_tokens' key"
    assert 'num_samples' in result, "Missing 'num_samples' key"

    # Validate ranges
    assert result['perplexity'] > 0, f"Invalid perplexity: {result['perplexity']}"
    assert result['perplexity'] >= 1.0, f"Perplexity < 1.0: {result['perplexity']}"
    assert result['total_tokens'] > 0, "No tokens processed"
    assert result['num_samples'] > 0, "No samples processed"

    print(f"✅ PASSED")
    print(f"   Perplexity: {result['perplexity']:.2f}")
    print(f"   Tokens: {result['total_tokens']}, Samples: {result['num_samples']}")

    return result


def test_helper_extract_texts_for_perplexity():
    """Test extract_texts_for_perplexity helper function."""
    from hc_routing_evaluation import extract_texts_for_perplexity

    print("\n" + "="*70)
    print("TEST: extract_texts_for_perplexity()")
    print("="*70)

    # Test WikiText
    wikitext_texts = extract_texts_for_perplexity(WIKITEXT_SAMPLES, 'wikitext')
    assert len(wikitext_texts) == 3, f"Expected 3 non-empty texts, got {len(wikitext_texts)}"
    assert all(isinstance(t, str) for t in wikitext_texts), "Not all texts are strings"
    print(f"✅ WikiText extraction: {len(wikitext_texts)} texts")

    # Test LAMBADA
    lambada_texts = extract_texts_for_perplexity(LAMBADA_SAMPLES, 'lambada')
    assert len(lambada_texts) == 3, f"Expected 3 non-empty texts, got {len(lambada_texts)}"
    assert all(isinstance(t, str) for t in lambada_texts), "Not all texts are strings"
    print(f"✅ LAMBADA extraction: {len(lambada_texts)} texts")

    # Test HellaSwag
    hellaswag_texts = extract_texts_for_perplexity(HELLASWAG_SAMPLES, 'hellaswag')
    assert len(hellaswag_texts) == 2, f"Expected 2 texts, got {len(hellaswag_texts)}"
    assert all(isinstance(t, str) for t in hellaswag_texts), "Not all texts are strings"
    # Verify it's context + correct ending
    assert HELLASWAG_SAMPLES[0]['ctx'] in hellaswag_texts[0], "Context not in extracted text"
    assert HELLASWAG_SAMPLES[0]['endings'][0] in hellaswag_texts[0], "Correct ending not in text"
    print(f"✅ HellaSwag extraction: {len(hellaswag_texts)} texts")

    print(f"✅ PASSED")

    return True


def test_evaluate_perplexity_both_metrics():
    """Test evaluate_perplexity returns both perplexity and accuracy."""
    from hc_routing_evaluation import evaluate_perplexity

    print("\n" + "="*70)
    print("TEST: evaluate_perplexity() - BOTH METRICS")
    print("="*70)

    model = MockModel()
    tokenizer = MockTokenizer()
    texts = [s for s in WIKITEXT_SAMPLES if s.strip()]

    result = evaluate_perplexity(
        model, tokenizer, texts, patcher=None, device='cpu'
    )

    # Validate BOTH metrics are present
    assert 'perplexity' in result, "Missing 'perplexity' key"
    assert 'accuracy' in result, "Missing 'accuracy' key"
    assert 'avg_loss' in result, "Missing 'avg_loss' key"

    # Validate ranges
    assert result['perplexity'] >= 1.0, f"Invalid perplexity: {result['perplexity']}"
    assert 0 <= result['accuracy'] <= 1, f"Accuracy out of range: {result['accuracy']}"

    # Validate supporting fields
    assert 'correct_tokens' in result, "Missing 'correct_tokens' key"
    assert 'predictable_tokens' in result, "Missing 'predictable_tokens' key"
    assert 'total_tokens' in result, "Missing 'total_tokens' key"
    assert 'num_samples' in result, "Missing 'num_samples' key"

    print(f"✅ PASSED")
    print(f"   Perplexity: {result['perplexity']:.2f}")
    print(f"   Accuracy: {result['accuracy']:.4f}")
    print(f"   Tokens: {result['total_tokens']}, Samples: {result['num_samples']}")

    return result


def test_evaluate_lambada_both_metrics():
    """Test evaluate_lambada returns both accuracy and perplexity."""
    from hc_routing_evaluation import evaluate_lambada

    print("\n" + "="*70)
    print("TEST: evaluate_lambada() - BOTH METRICS")
    print("="*70)

    model = MockModel()
    tokenizer = MockTokenizer()
    samples = [s for s in LAMBADA_SAMPLES if s['text'].strip()]

    result = evaluate_lambada(
        model, tokenizer, samples, patcher=None, device='cpu'
    )

    # Validate BOTH metrics are present
    assert 'accuracy' in result, "Missing 'accuracy' key"
    assert 'perplexity' in result, "Missing 'perplexity' key"
    assert 'avg_loss' in result, "Missing 'avg_loss' key"

    # Validate ranges
    assert 0 <= result['accuracy'] <= 1, f"Accuracy out of range: {result['accuracy']}"
    assert result['perplexity'] >= 1.0, f"Invalid perplexity: {result['perplexity']}"

    # Validate supporting fields
    assert 'correct' in result, "Missing 'correct' key"
    assert 'total' in result, "Missing 'total' key"
    assert 'total_tokens' in result, "Missing 'total_tokens' key"
    assert 'num_samples' in result, "Missing 'num_samples' key"

    print(f"✅ PASSED")
    print(f"   Accuracy: {result['accuracy']:.4f}")
    print(f"   Perplexity: {result['perplexity']:.2f}")
    print(f"   Correct: {result['correct']}/{result['total']}")

    return result


def test_evaluate_hellaswag_both_metrics():
    """Test evaluate_hellaswag returns both accuracy and perplexity."""
    from hc_routing_evaluation import evaluate_hellaswag

    print("\n" + "="*70)
    print("TEST: evaluate_hellaswag() - BOTH METRICS")
    print("="*70)

    model = MockModel()
    tokenizer = MockTokenizer()
    samples = HELLASWAG_SAMPLES

    result = evaluate_hellaswag(
        model, tokenizer, samples, patcher=None, device='cpu'
    )

    # Validate BOTH metrics are present
    assert 'accuracy' in result, "Missing 'accuracy' key"
    assert 'perplexity' in result, "Missing 'perplexity' key"
    assert 'avg_loss' in result, "Missing 'avg_loss' key"

    # Validate ranges
    assert 0 <= result['accuracy'] <= 1, f"Accuracy out of range: {result['accuracy']}"
    assert result['perplexity'] >= 1.0, f"Invalid perplexity: {result['perplexity']}"

    # Validate supporting fields
    assert 'correct' in result, "Missing 'correct' key"
    assert 'total' in result, "Missing 'total' key"
    assert 'total_tokens' in result, "Missing 'total_tokens' key"
    assert 'num_samples' in result, "Missing 'num_samples' key"

    print(f"✅ PASSED")
    print(f"   Accuracy: {result['accuracy']:.4f}")
    print(f"   Perplexity: {result['perplexity']:.2f}")
    print(f"   Correct: {result['correct']}/{result['total']}")

    return result


def test_return_format_consistency():
    """Test all evaluation functions return consistent format."""
    from hc_routing_evaluation import evaluate_perplexity, evaluate_lambada, evaluate_hellaswag

    print("\n" + "="*70)
    print("TEST: Return Format Consistency")
    print("="*70)

    model = MockModel()
    tokenizer = MockTokenizer()

    # Get results from all three functions
    wikitext_result = evaluate_perplexity(
        model, tokenizer, [s for s in WIKITEXT_SAMPLES if s.strip()],
        patcher=None, device='cpu'
    )

    lambada_result = evaluate_lambada(
        model, tokenizer, [s for s in LAMBADA_SAMPLES if s['text'].strip()],
        patcher=None, device='cpu'
    )

    hellaswag_result = evaluate_hellaswag(
        model, tokenizer, HELLASWAG_SAMPLES,
        patcher=None, device='cpu'
    )

    # All should have these common keys
    required_keys = ['perplexity', 'accuracy', 'avg_loss', 'total_tokens', 'num_samples']

    for key in required_keys:
        assert key in wikitext_result, f"WikiText missing key: {key}"
        assert key in lambada_result, f"LAMBADA missing key: {key}"
        assert key in hellaswag_result, f"HellaSwag missing key: {key}"

    print(f"✅ PASSED - All datasets return consistent format")
    print(f"   Common keys: {required_keys}")

    return True


def test_edge_case_empty_dataset():
    """Test handling of empty dataset."""
    from hc_routing_evaluation import evaluate_perplexity

    print("\n" + "="*70)
    print("TEST: Edge Case - Empty Dataset")
    print("="*70)

    model = MockModel()
    tokenizer = MockTokenizer()

    result = evaluate_perplexity(
        model, tokenizer, [], patcher=None, device='cpu'
    )

    # Should handle gracefully
    assert 'perplexity' in result, "Missing perplexity"
    assert 'accuracy' in result, "Missing accuracy"
    assert result['num_samples'] == 0, "Should have 0 samples"

    print(f"✅ PASSED - Empty dataset handled gracefully")

    return result


def test_edge_case_single_sample():
    """Test handling of single sample."""
    from hc_routing_evaluation import evaluate_perplexity

    print("\n" + "="*70)
    print("TEST: Edge Case - Single Sample")
    print("="*70)

    model = MockModel()
    tokenizer = MockTokenizer()

    result = evaluate_perplexity(
        model, tokenizer, ['This is a single test sample.'],
        patcher=None, device='cpu'
    )

    # Should work with single sample
    assert result['num_samples'] == 1, f"Expected 1 sample, got {result['num_samples']}"
    assert result['perplexity'] >= 1.0, "Invalid perplexity"
    assert 0 <= result['accuracy'] <= 1, "Invalid accuracy"

    print(f"✅ PASSED - Single sample handled correctly")
    print(f"   Perplexity: {result['perplexity']:.2f}, Accuracy: {result['accuracy']:.4f}")

    return result


def test_metrics_mathematically_consistent():
    """Test that metrics are mathematically consistent."""
    from hc_routing_evaluation import evaluate_perplexity

    print("\n" + "="*70)
    print("TEST: Mathematical Consistency")
    print("="*70)

    model = MockModel()
    tokenizer = MockTokenizer()
    texts = [s for s in WIKITEXT_SAMPLES if s.strip()]

    result = evaluate_perplexity(
        model, tokenizer, texts, patcher=None, device='cpu'
    )

    # Perplexity = exp(avg_loss)
    expected_ppl = np.exp(result['avg_loss'])
    assert abs(result['perplexity'] - expected_ppl) < 0.01, \
        f"Perplexity != exp(avg_loss): {result['perplexity']} vs {expected_ppl}"

    # Accuracy = correct / total
    if result['predictable_tokens'] > 0:
        expected_acc = result['correct_tokens'] / result['predictable_tokens']
        assert abs(result['accuracy'] - expected_acc) < 1e-6, \
            f"Accuracy != correct/total: {result['accuracy']} vs {expected_acc}"

    print(f"✅ PASSED - Metrics are mathematically consistent")

    return True


# ==============================================================================
# MAIN TEST RUNNER
# ==============================================================================

def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("COMPREHENSIVE METRICS TEST SUITE")
    print("="*70)
    print("Testing perplexity + accuracy for all datasets")
    print("="*70)

    tests = [
        ("Helper: compute_token_accuracy", test_helper_compute_token_accuracy),
        ("Helper: compute_perplexity_from_texts", test_helper_compute_perplexity_from_texts),
        ("Helper: extract_texts_for_perplexity", test_helper_extract_texts_for_perplexity),
        ("WikiText: Both Metrics", test_evaluate_perplexity_both_metrics),
        ("LAMBADA: Both Metrics", test_evaluate_lambada_both_metrics),
        ("HellaSwag: Both Metrics", test_evaluate_hellaswag_both_metrics),
        ("Return Format Consistency", test_return_format_consistency),
        ("Edge Case: Empty Dataset", test_edge_case_empty_dataset),
        ("Edge Case: Single Sample", test_edge_case_single_sample),
        ("Mathematical Consistency", test_metrics_mathematically_consistent),
    ]

    passed = 0
    failed = 0
    errors = []

    for i, (name, test_func) in enumerate(tests, 1):
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"❌ FAILED: {e}")
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"❌ ERROR: {e}")

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total Tests: {len(tests)}")
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")

    if errors:
        print("\n" + "="*70)
        print("FAILED TESTS:")
        print("="*70)
        for name, error in errors:
            print(f"\n{name}:")
            print(f"  {error}")

    print("\n" + "="*70)

    return passed == len(tests)


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
