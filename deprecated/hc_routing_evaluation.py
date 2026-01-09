"""
Dataset Evaluation for HC Routing Framework
============================================

This module provides dataset loading and evaluation functions for measuring
the quality impact of HC routing on standard NLP benchmarks.

Supported Datasets:
- WikiText-2: Language modeling (perplexity)
- LAMBADA: Last word prediction (accuracy)
- HellaSwag: Commonsense sentence completion (accuracy)

All evaluation functions support internal routing logging for analysis.

Usage:
    from hc_routing_evaluation import evaluate_perplexity, evaluate_lambada

    # With internal logging
    result = evaluate_perplexity(
        model, tokenizer, dataset,
        patcher=patcher,  # HCRoutingIntegration instance
        max_samples=200
    )

    print(f"Perplexity: {result['perplexity']}")
    internal_logs = result['internal_logs']  # Full routing data
"""

import torch
import numpy as np
from datasets import load_dataset
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm
import warnings

# Import detailed logging system
from hc_routing_logging import HCRoutingLogger

warnings.filterwarnings('ignore')


# ==============================================================================
# DATASET LOADING
# ==============================================================================

def load_wikitext(
    split: str = 'test',
    max_samples: Optional[int] = None
) -> List[str]:
    """
    Load WikiText-2 dataset for perplexity evaluation.

    Args:
        split: Dataset split ('test', 'train', 'validation')
        max_samples: Maximum number of samples to load (None = all)

    Returns:
        List of text samples

    Examples:
        >>> texts = load_wikitext(split='test', max_samples=200)
        >>> len(texts)
        200
    """
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)

    # Filter out empty texts
    texts = [sample['text'] for sample in dataset if sample['text'].strip()]

    if max_samples is not None:
        texts = texts[:max_samples]

    return texts


def load_lambada(
    split: str = 'test',
    max_samples: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    Load LAMBADA dataset for last-word prediction evaluation.

    Args:
        split: Dataset split ('test', 'validation')
        max_samples: Maximum number of samples to load (None = all)

    Returns:
        List of dictionaries with 'text' and 'target' keys

    Examples:
        >>> samples = load_lambada(split='test', max_samples=200)
        >>> samples[0].keys()
        dict_keys(['text', 'target'])
    """
    try:
        dataset = load_dataset('lambada', split=split)
    except:
        # Fallback to different configuration if needed
        dataset = load_dataset('EleutherAI/lambada_openai', 'en', split=split)

    samples = []
    for item in dataset:
        text = item['text'] if 'text' in item else ''
        samples.append({'text': text, 'target': text.split()[-1] if text else ''})

    if max_samples is not None:
        samples = samples[:max_samples]

    return samples


def load_hellaswag(
    split: str = 'validation',
    max_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load HellaSwag dataset for sentence completion evaluation.

    Args:
        split: Dataset split ('validation', 'train')
        max_samples: Maximum number of samples to load (None = all)

    Returns:
        List of dictionaries with context and choices

    Examples:
        >>> samples = load_hellaswag(split='validation', max_samples=200)
        >>> sample = samples[0]
        >>> 'ctx' in sample and 'endings' in sample
        True
    """
    dataset = load_dataset('hellaswag', split=split)

    samples = []
    for item in dataset:
        samples.append({
            'ctx': item['ctx'],
            'endings': item['endings'],
            'label': int(item['label']) if 'label' in item else 0
        })

    if max_samples is not None:
        samples = samples[:max_samples]

    return samples


# ==============================================================================
# EVALUATION FUNCTIONS
# ==============================================================================

def evaluate_perplexity(
    model,
    tokenizer,
    dataset: List[str],
    patcher=None,
    max_length: int = 512,
    batch_size: int = 1,
    device: str = 'cuda',
    # Detailed logging parameters
    log_routing: bool = False,
    output_dir: str = './logs',
    experiment_name: str = 'eval_wikitext',
    log_every_n: int = 5
) -> Dict[str, Any]:
    """
    Evaluate perplexity on WikiText-2 with optional internal routing logging.

    Args:
        model: OLMoE model instance
        tokenizer: Tokenizer for the model
        dataset: List of text samples from WikiText
        patcher: Optional HCRoutingIntegration instance for logging
        max_length: Maximum sequence length (default: 512)
        batch_size: Batch size (default: 1 for stable logging)
        device: Device to run on ('cuda', 'cpu')

    Returns:
        Dictionary containing:
        - perplexity: float
        - avg_loss: float
        - losses: List[float] per sample
        - internal_logs: List[Dict] if patcher provided, else None
        - num_samples: int
        - total_tokens: int

    Examples:
        >>> result = evaluate_perplexity(model, tokenizer, texts, patcher=patcher)
        >>> print(f"PPL: {result['perplexity']:.2f}")
        >>> len(result['internal_logs'])  # Full routing data
        200
    """
    model.eval()
    losses = []
    token_counts = []  # Track tokens per sample for proper weighting
    total_tokens = 0
    internal_logs = [] if patcher else None

    # Create detailed logger if requested
    detailed_logger = None
    if log_routing and output_dir and patcher:
        detailed_logger = HCRoutingLogger(
            output_dir=output_dir,
            experiment_name=experiment_name,
            log_every_n=log_every_n
        )
        print(f"✅ Created HCRoutingLogger: {experiment_name}")
        print(f"   Logging to: {output_dir}")
        print(f"   Log frequency: every {log_every_n} tokens")

        # Attach logger to integration
        if hasattr(patcher, 'set_external_logger'):
            patcher.set_external_logger(detailed_logger)
        elif hasattr(patcher, '_logger'):
            patcher._logger = detailed_logger
            print("✅ Logger attached via _logger attribute")
        else:
            print("⚠️  Warning: Cannot attach logger to patcher")
            detailed_logger = None

    # Clear any existing logs
    if patcher and hasattr(patcher, 'clear_internal_logs'):
        patcher.clear_internal_logs()

    with torch.no_grad():
        for i, text in enumerate(tqdm(dataset, desc="Evaluating Perplexity")):
            if not text.strip():
                continue

            # Start logging for this sample
            if patcher and hasattr(patcher, 'start_sample_logging'):
                patcher.start_sample_logging(i, text[:100])

            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
                padding=False
            )

            # Move to device
            input_ids = inputs['input_ids'].to(device)

            if input_ids.shape[1] < 2:
                continue  # Skip very short sequences

            # Forward pass
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
            num_tokens = input_ids.shape[1]

            losses.append(loss)
            token_counts.append(num_tokens)
            total_tokens += num_tokens

            # End logging for this sample
            if patcher and hasattr(patcher, 'end_sample_logging'):
                patcher.end_sample_logging(loss)

    # Collect internal logs
    if patcher and hasattr(patcher, 'get_internal_logs'):
        internal_logs = patcher.get_internal_logs()

    # Save detailed logs if logger was created
    if detailed_logger:
        print("\n" + "="*70)
        print("SAVING DETAILED ROUTING LOGS")
        print("="*70)
        detailed_logger.save_logs()
        detailed_logger.generate_plots()
        summary = detailed_logger.get_summary()
        print(f"✅ Saved {detailed_logger.logged_decisions} detailed routing decisions")
        print(f"   Total decisions: {detailed_logger.total_decisions}")
        print(f"   Log directory: {detailed_logger.output_dir}")
        if 'global' in summary:
            print(f"   Avg experts: {summary['global']['avg_experts']:.2f}")
            print(f"   Floor hit rate: {summary['global']['floor_hit_rate']:.1f}%")
            print(f"   Ceiling hit rate: {summary['global']['ceiling_hit_rate']:.1f}%")
        print("="*70)

    # Compute perplexity (both token-weighted and sample-weighted)
    if losses:
        # CORRECT: Token-weighted average (standard practice in language modeling)
        # Each sample contributes proportionally to its token count
        total_weighted_loss = sum(loss * count for loss, count in zip(losses, token_counts))
        avg_loss_token_weighted = total_weighted_loss / total_tokens
        perplexity_token_weighted = float(np.exp(avg_loss_token_weighted))

        # COMPARISON: Sample-weighted average (treats all samples equally)
        # This is biased toward shorter sequences
        avg_loss_sample_weighted = float(np.mean(losses))
        perplexity_sample_weighted = float(np.exp(avg_loss_sample_weighted))

        # Use token-weighted as primary metric
        perplexity = perplexity_token_weighted
        avg_loss = avg_loss_token_weighted
    else:
        perplexity = float('inf')
        avg_loss = float('inf')
        perplexity_token_weighted = float('inf')
        perplexity_sample_weighted = float('inf')
        avg_loss_token_weighted = float('inf')
        avg_loss_sample_weighted = float('inf')

    return {
        # Primary metrics (token-weighted - CORRECT)
        'perplexity': perplexity,
        'avg_loss': avg_loss,

        # Token-weighted metrics (explicit)
        'perplexity_token_weighted': perplexity_token_weighted,
        'avg_loss_token_weighted': avg_loss_token_weighted,

        # Sample-weighted metrics (for comparison)
        'perplexity_sample_weighted': perplexity_sample_weighted,
        'avg_loss_sample_weighted': avg_loss_sample_weighted,

        # Raw data
        'losses': losses,
        'token_counts': token_counts,
        'internal_logs': internal_logs,
        'num_samples': len(losses),
        'total_tokens': total_tokens,
        'detailed_logger': detailed_logger
    }


def evaluate_lambada(
    model,
    tokenizer,
    dataset: List[Dict[str, str]],
    patcher=None,
    max_length: int = 512,
    device: str = 'cuda',
    # Detailed logging parameters
    log_routing: bool = False,
    output_dir: str = './logs',
    experiment_name: str = 'eval_lambada',
    log_every_n: int = 5
) -> Dict[str, Any]:
    """
    Evaluate accuracy on LAMBADA with optional internal routing logging.

    Args:
        model: OLMoE model instance
        tokenizer: Tokenizer for the model
        dataset: List of dicts with 'text' and 'target' keys
        patcher: Optional HCRoutingIntegration instance for logging
        max_length: Maximum sequence length (default: 512)
        device: Device to run on ('cuda', 'cpu')

    Returns:
        Dictionary containing:
        - accuracy: float in [0, 1]
        - correct: int
        - total: int
        - predictions: List[Dict] with prediction details
        - internal_logs: List[Dict] if patcher provided, else None

    Examples:
        >>> result = evaluate_lambada(model, tokenizer, samples, patcher=patcher)
        >>> print(f"Accuracy: {result['accuracy']:.4f}")
    """
    model.eval()
    correct = 0
    total = 0
    predictions = []
    internal_logs = [] if patcher else None

    # Create detailed logger if requested
    detailed_logger = None
    if log_routing and output_dir and patcher:
        detailed_logger = HCRoutingLogger(
            output_dir=output_dir,
            experiment_name=experiment_name,
            log_every_n=log_every_n
        )
        print(f"✅ Created HCRoutingLogger: {experiment_name}")
        print(f"   Logging to: {output_dir}")

        # Attach logger to integration
        if hasattr(patcher, 'set_external_logger'):
            patcher.set_external_logger(detailed_logger)
        elif hasattr(patcher, '_logger'):
            patcher._logger = detailed_logger
        else:
            print("⚠️  Warning: Cannot attach logger to patcher")
            detailed_logger = None

    # Clear any existing logs
    if patcher and hasattr(patcher, 'clear_internal_logs'):
        patcher.clear_internal_logs()

    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataset, desc="Evaluating LAMBADA")):
            text = sample['text']
            target_word = sample['target']

            if not text or not target_word:
                continue

            # Split into context and target
            words = text.strip().split()
            if len(words) < 2:
                continue

            context = ' '.join(words[:-1])

            # Start logging
            if patcher and hasattr(patcher, 'start_sample_logging'):
                patcher.start_sample_logging(i, context[:100])

            # Tokenize context only
            inputs = tokenizer(context, return_tensors='pt', truncation=True, max_length=max_length)
            input_ids = inputs['input_ids'].to(device)

            # Generate next token
            outputs = model.generate(
                input_ids,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

            # Get predicted word
            predicted_token_id = outputs[0][-1].item()
            predicted_word = tokenizer.decode([predicted_token_id], skip_special_tokens=True).strip()

            # Check if correct (case-insensitive)
            is_correct = predicted_word.lower() == target_word.lower()

            if is_correct:
                correct += 1
            total += 1

            predictions.append({
                'context': context,
                'target': target_word,
                'predicted': predicted_word,
                'correct': is_correct
            })

            # End logging
            if patcher and hasattr(patcher, 'end_sample_logging'):
                patcher.end_sample_logging(0.0)  # No loss for generation

    # Collect internal logs
    if patcher and hasattr(patcher, 'get_internal_logs'):
        internal_logs = patcher.get_internal_logs()

    # Save detailed logs if logger was created
    if detailed_logger:
        print("\n" + "="*70)
        print("SAVING DETAILED ROUTING LOGS")
        print("="*70)
        detailed_logger.save_logs()
        detailed_logger.generate_plots()
        summary = detailed_logger.get_summary()
        print(f"✅ Saved {detailed_logger.logged_decisions} detailed routing decisions")
        print(f"   Total decisions: {detailed_logger.total_decisions}")
        print(f"   Log directory: {detailed_logger.output_dir}")
        if 'global' in summary:
            print(f"   Avg experts: {summary['global']['avg_experts']:.2f}")
        print("="*70)

    accuracy = correct / total if total > 0 else 0.0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'predictions': predictions,
        'internal_logs': internal_logs,
        'detailed_logger': detailed_logger
    }


def evaluate_hellaswag(
    model,
    tokenizer,
    dataset: List[Dict[str, Any]],
    patcher=None,
    max_length: int = 512,
    device: str = 'cuda',
    # Detailed logging parameters
    log_routing: bool = False,
    output_dir: str = './logs',
    experiment_name: str = 'eval_hellaswag',
    log_every_n: int = 5
) -> Dict[str, Any]:
    """
    Evaluate accuracy on HellaSwag with optional internal routing logging.

    Args:
        model: OLMoE model instance
        tokenizer: Tokenizer for the model
        dataset: List of dicts with 'ctx', 'endings', 'label' keys
        patcher: Optional HCRoutingIntegration instance for logging
        max_length: Maximum sequence length (default: 512)
        device: Device to run on ('cuda', 'cpu')

    Returns:
        Dictionary containing:
        - accuracy: float in [0, 1]
        - correct: int
        - total: int
        - predictions: List[Dict] with prediction details
        - internal_logs: List[Dict] if patcher provided, else None

    Examples:
        >>> result = evaluate_hellaswag(model, tokenizer, samples, patcher=patcher)
        >>> print(f"Accuracy: {result['accuracy']:.4f}")
    """
    model.eval()
    correct = 0
    correct_normalized = 0  # Track length-normalized accuracy separately
    total = 0
    predictions = []
    internal_logs = [] if patcher else None

    # Create detailed logger if requested
    detailed_logger = None
    if log_routing and output_dir and patcher:
        detailed_logger = HCRoutingLogger(
            output_dir=output_dir,
            experiment_name=experiment_name,
            log_every_n=log_every_n
        )
        print(f"✅ Created HCRoutingLogger: {experiment_name}")
        print(f"   Logging to: {output_dir}")

        # Attach logger to integration
        if hasattr(patcher, 'set_external_logger'):
            patcher.set_external_logger(detailed_logger)
        elif hasattr(patcher, '_logger'):
            patcher._logger = detailed_logger
        else:
            print("⚠️  Warning: Cannot attach logger to patcher")
            detailed_logger = None

    # Clear any existing logs
    if patcher and hasattr(patcher, 'clear_internal_logs'):
        patcher.clear_internal_logs()

    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataset, desc="Evaluating HellaSwag")):
            ctx = sample['ctx']
            endings = sample['endings']
            label = sample['label']

            if not ctx or not endings:
                continue

            # Start logging
            if patcher and hasattr(patcher, 'start_sample_logging'):
                patcher.start_sample_logging(i, ctx[:100])

            # Score each ending (both raw and length-normalized)
            ending_scores_raw = []
            ending_scores_normalized = []
            ending_lengths = []

            for ending in endings:
                full_text = ctx + " " + ending

                # Tokenize
                inputs = tokenizer(
                    full_text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=max_length
                )
                input_ids = inputs['input_ids'].to(device)

                # Get log probability of this completion
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss.item()
                num_tokens = input_ids.shape[1]

                # Raw score: Lower loss = higher probability
                raw_score = -loss
                ending_scores_raw.append(raw_score)
                ending_lengths.append(num_tokens)

                # Normalized score: Divide by length to prevent bias toward shorter endings
                # This is the fairer comparison method
                normalized_score = -loss / num_tokens if num_tokens > 0 else -loss
                ending_scores_normalized.append(normalized_score)

            # Predict ending with highest score (raw method)
            predicted_idx_raw = int(np.argmax(ending_scores_raw))
            is_correct_raw = (predicted_idx_raw == label)

            # Predict ending with highest score (normalized method - recommended)
            predicted_idx_normalized = int(np.argmax(ending_scores_normalized))
            is_correct_normalized = (predicted_idx_normalized == label)

            # Track both methods
            if is_correct_raw:
                correct += 1
            if is_correct_normalized:
                correct_normalized += 1
            total += 1

            predictions.append({
                'context': ctx,
                'endings': endings,
                'ending_lengths': ending_lengths,
                'label': label,
                'predicted_raw': predicted_idx_raw,
                'predicted_normalized': predicted_idx_normalized,
                'correct_raw': is_correct_raw,
                'correct_normalized': is_correct_normalized,
                'scores_raw': ending_scores_raw,
                'scores_normalized': ending_scores_normalized
            })

            # End logging
            if patcher and hasattr(patcher, 'end_sample_logging'):
                patcher.end_sample_logging(ending_scores_raw[label])  # Use true ending's score as loss

    # Collect internal logs
    if patcher and hasattr(patcher, 'get_internal_logs'):
        internal_logs = patcher.get_internal_logs()

    # Save detailed logs if logger was created
    if detailed_logger:
        print("\n" + "="*70)
        print("SAVING DETAILED ROUTING LOGS")
        print("="*70)
        detailed_logger.save_logs()
        detailed_logger.generate_plots()
        summary = detailed_logger.get_summary()
        print(f"✅ Saved {detailed_logger.logged_decisions} detailed routing decisions")
        print(f"   Total decisions: {detailed_logger.total_decisions}")
        print(f"   Log directory: {detailed_logger.output_dir}")
        if 'global' in summary:
            print(f"   Avg experts: {summary['global']['avg_experts']:.2f}")
        print("="*70)

    # Compute both accuracies
    accuracy_raw = correct / total if total > 0 else 0.0
    accuracy_normalized = correct_normalized / total if total > 0 else 0.0

    return {
        # Primary accuracy (raw - for backward compatibility)
        'accuracy': accuracy_raw,
        'correct': correct,
        'total': total,

        # Raw scoring (default, may favor shorter endings)
        'accuracy_raw': accuracy_raw,
        'correct_raw': correct,

        # Length-normalized scoring (recommended, fairer comparison)
        'accuracy_normalized': accuracy_normalized,
        'correct_normalized': correct_normalized,

        # Data
        'predictions': predictions,
        'internal_logs': internal_logs,
        'detailed_logger': detailed_logger
    }


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def evaluate_all_datasets(
    model,
    tokenizer,
    patcher=None,
    max_samples: int = 200,
    device: str = 'cuda'
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate on all supported datasets.

    Args:
        model: OLMoE model instance
        tokenizer: Tokenizer for the model
        patcher: Optional HCRoutingIntegration instance for logging
        max_samples: Maximum samples per dataset
        device: Device to run on

    Returns:
        Dictionary mapping dataset name to evaluation results:
        {
            'wikitext': {...},
            'lambada': {...},
            'hellaswag': {...}
        }

    Examples:
        >>> results = evaluate_all_datasets(model, tokenizer, patcher, max_samples=200)
        >>> results['wikitext']['perplexity']
        15.82
        >>> results['lambada']['accuracy']
        0.668
    """
    results = {}

    print("=" * 70)
    print("EVALUATING ON ALL DATASETS")
    print("=" * 70)

    # WikiText-2
    print("\n[1/3] WikiText-2...")
    wikitext_data = load_wikitext(max_samples=max_samples)
    results['wikitext'] = evaluate_perplexity(
        model, tokenizer, wikitext_data, patcher, device=device
    )
    print(f"  ✓ Perplexity: {results['wikitext']['perplexity']:.2f}")

    # LAMBADA
    print("\n[2/3] LAMBADA...")
    lambada_data = load_lambada(max_samples=max_samples)
    results['lambada'] = evaluate_lambada(
        model, tokenizer, lambada_data, patcher, device=device
    )
    print(f"  ✓ Accuracy: {results['lambada']['accuracy']:.4f}")

    # HellaSwag
    print("\n[3/3] HellaSwag...")
    hellaswag_data = load_hellaswag(max_samples=max_samples)
    results['hellaswag'] = evaluate_hellaswag(
        model, tokenizer, hellaswag_data, patcher, device=device
    )
    print(f"  ✓ Accuracy: {results['hellaswag']['accuracy']:.4f}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    return results
