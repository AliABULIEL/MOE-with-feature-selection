"""
Dataset Evaluation for Random Routing Framework
============================================

This module provides dataset loading and evaluation functions for measuring
the quality impact of random routing on standard NLP benchmarks.
"""

import torch
import numpy as np
from datasets import load_dataset
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import warnings

from random_routing_logging import RandomRoutingLogger

warnings.filterwarnings('ignore')

# Dataset loading functions are generic and can be reused
from hc_routing_evaluation import load_wikitext, load_lambada, load_hellaswag

def evaluate_perplexity(
    model,
    tokenizer,
    dataset: List[str],
    patcher=None,
    max_length: int = 512,
    device: str = 'cuda',
    log_routing: bool = False,
    output_dir: str = './logs',
    experiment_name: str = 'eval_wikitext_random',
    log_every_n: int = 5
) -> Dict[str, Any]:
    """
    Evaluate perplexity on WikiText-2 with optional random routing logging.
    """
    model.eval()
    losses = []
    token_counts = []
    total_tokens = 0
    internal_logs = [] if patcher else None

    detailed_logger = None
    if log_routing and output_dir and patcher:
        detailed_logger = RandomRoutingLogger(
            output_dir=output_dir,
            experiment_name=experiment_name,
            log_every_n=log_every_n
        )
        if hasattr(patcher, 'set_external_logger'):
            patcher.set_external_logger(detailed_logger)

    if patcher and hasattr(patcher, 'clear_internal_logs'):
        patcher.clear_internal_logs()

    with torch.no_grad():
        for i, text in enumerate(tqdm(dataset, desc="Evaluating Perplexity (Random Routing)")):
            if not text.strip():
                continue

            if patcher and hasattr(patcher, 'start_sample_logging'):
                patcher.start_sample_logging(i)

            inputs = tokenizer(
                text, return_tensors='pt', truncation=True, max_length=max_length
            )
            input_ids = inputs['input_ids'].to(device)

            if input_ids.shape[1] < 2:
                continue

            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
            num_tokens = input_ids.shape[1]

            losses.append(loss)
            token_counts.append(num_tokens)
            total_tokens += num_tokens

            if patcher and hasattr(patcher, 'end_sample_logging'):
                patcher.end_sample_logging(loss)

    if patcher and hasattr(patcher, 'get_internal_logs'):
        internal_logs = patcher.get_internal_logs()

    if detailed_logger:
        detailed_logger.save_logs()

    if losses:
        total_weighted_loss = sum(loss * count for loss, count in zip(losses, token_counts))
        avg_loss = total_weighted_loss / total_tokens if total_tokens > 0 else 0
        perplexity = float(np.exp(avg_loss))
    else:
        perplexity = float('inf')
        avg_loss = float('inf')

    return {
        'perplexity': perplexity,
        'avg_loss': avg_loss,
        'internal_logs': internal_logs,
        'num_samples': len(losses),
        'total_tokens': total_tokens,
    }


def evaluate_lambada(
    model,
    tokenizer,
    dataset: List[Dict[str, str]],
    patcher=None,
    max_length: int = 512,
    device: str = 'cuda',
    log_routing: bool = False,
    output_dir: str = './logs',
    experiment_name: str = 'eval_lambada_random',
    log_every_n: int = 5
) -> Dict[str, Any]:
    """
    Evaluate accuracy on LAMBADA with optional random routing logging.
    """
    model.eval()
    correct = 0
    total = 0
    internal_logs = [] if patcher else None

    detailed_logger = None
    if log_routing and output_dir and patcher:
        detailed_logger = RandomRoutingLogger(
            output_dir=output_dir,
            experiment_name=experiment_name,
            log_every_n=log_every_n
        )
        if hasattr(patcher, 'set_external_logger'):
            patcher.set_external_logger(detailed_logger)

    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataset, desc="Evaluating LAMBADA (Random Routing)")):
            text = sample['text']
            target_word = sample['target']

            if not text or not target_word:
                continue
            
            context = ' '.join(text.strip().split()[:-1])
            if not context:
                continue

            if patcher and hasattr(patcher, 'start_sample_logging'):
                patcher.start_sample_logging(i)

            inputs = tokenizer(context, return_tensors='pt', truncation=True, max_length=max_length)
            input_ids = inputs['input_ids'].to(device)

            outputs = model.generate(
                input_ids, max_new_tokens=1, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
            
            predicted_token_id = outputs[0][-1].item()
            predicted_word = tokenizer.decode([predicted_token_id], skip_special_tokens=True).strip()

            if predicted_word.lower() == target_word.lower():
                correct += 1
            total += 1
            
            if patcher and hasattr(patcher, 'end_sample_logging'):
                patcher.end_sample_logging(0.0)

    if patcher and hasattr(patcher, 'get_internal_logs'):
        internal_logs = patcher.get_internal_logs()

    if detailed_logger:
        detailed_logger.save_logs()

    return {
        'accuracy': correct / total if total > 0 else 0.0,
        'correct': correct,
        'total': total,
        'internal_logs': internal_logs,
    }

def evaluate_hellaswag(
    model,
    tokenizer,
    dataset: List[Dict[str, Any]],
    patcher=None,
    max_length: int = 512,
    device: str = 'cuda',
    log_routing: bool = False,
    output_dir: str = './logs',
    experiment_name: str = 'eval_hellaswag_random',
    log_every_n: int = 5
) -> Dict[str, Any]:
    """
    Evaluate accuracy on HellaSwag with optional random routing logging.
    """
    model.eval()
    correct = 0
    total = 0
    internal_logs = [] if patcher else None

    detailed_logger = None
    if log_routing and output_dir and patcher:
        detailed_logger = RandomRoutingLogger(
            output_dir=output_dir,
            experiment_name=experiment_name,
            log_every_n=log_every_n
        )
        if hasattr(patcher, 'set_external_logger'):
            patcher.set_external_logger(detailed_logger)

    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataset, desc="Evaluating HellaSwag (Random Routing)")):
            ctx = sample['ctx']
            endings = sample['endings']
            label = sample['label']

            if not ctx or not endings:
                continue
            
            if patcher and hasattr(patcher, 'start_sample_logging'):
                patcher.start_sample_logging(i)

            ending_scores = []
            for ending in endings:
                full_text = ctx + " " + ending
                inputs = tokenizer(full_text, return_tensors='pt', truncation=True, max_length=max_length)
                input_ids = inputs['input_ids'].to(device)
                
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss.item()
                ending_scores.append(-loss)
            
            if int(np.argmax(ending_scores)) == label:
                correct += 1
            total += 1

            if patcher and hasattr(patcher, 'end_sample_logging'):
                patcher.end_sample_logging(ending_scores[label])

    if patcher and hasattr(patcher, 'get_internal_logs'):
        internal_logs = patcher.get_internal_logs()

    if detailed_logger:
        detailed_logger.save_logs()

    return {
        'accuracy': correct / total if total > 0 else 0.0,
        'correct': correct,
        'total': total,
        'internal_logs': internal_logs,
    }
