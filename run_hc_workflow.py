#!/usr/bin/env python3
"""
Multi-Model HC Routing Workflow
================================

Unified script for running HC routing experiments on DeepSeek-v2-Lite and Qwen3-30B-A3B.
Uses configurable estimators (KDE, Student-T, GMM, etc.) for p-value computation.

Usage:
    python run_hc_workflow.py --config configs/deepseek_kde.yaml
    python run_hc_workflow.py --config configs/qwen_student_t.yaml

Features:
- Model-agnostic HC routing
- Configurable estimator selection
- Cloud-ready (no interactive prompts)
- Automatic metrics collection
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
import json

# Import configuration schema
from config_schema import ExperimentConfig

# Import utilities
from model_loader import load_model_from_config
from estimator_factory import create_estimator, load_estimator_models

# Import HC routing
from hc_routing import higher_criticism_routing, compute_hc_routing_statistics

# Import existing evaluation utilities
from hc_routing_evaluation import load_wikitext, load_lambada, load_hellaswag
from hc_routing_metrics import HCMetricsComputer
from hc_routing_logging import HCRoutingLogger


def load_config(config_path: str) -> ExperimentConfig:
    """Load and validate configuration from YAML file."""
    print(f"Loading configuration from {config_path}...")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Validate using Pydantic
    config = ExperimentConfig(**config_dict)
    
    print(f"✅ Configuration loaded and validated")
    print(f"   Model: {config.model.model_id}")
    print(f"   Estimator: {config.estimator.type}")
    print(f"   HC beta: {config.hc_routing.beta}")
    print(f"   Datasets: {', '.join(config.evaluation.datasets)}")
    
    return config


def setup_output_directories(config: ExperimentConfig) -> Dict[str, Path]:
    """Create output directory structure."""
    base_dir = Path(config.output.results_dir)
    dirs = {
        "base": base_dir,
        "logs": base_dir / "logs",
        "metrics": base_dir / "metrics",
        "plots": base_dir / "plots",
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Output directories created at {base_dir}")
    return dirs


def load_dataset(dataset_name: str, max_samples: int) -> List[Dict]:
    """Load dataset samples."""
    dataset_loaders = {
        "wikitext": lambda: load_wikitext(max_samples),
        "lambada": lambda: load_lambada(max_samples),
        "hellaswag": lambda: load_hellaswag(max_samples),
    }
    
    if dataset_name not in dataset_loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"Loading {dataset_name} dataset ({max_samples} samples)...")
    samples = dataset_loaders[dataset_name]()
    print(f"✅ Loaded {len(samples)} samples")
    
    return samples


def prepare_estimators_for_layers(
    config: ExperimentConfig,
    num_layers: int,
    training_data: Dict[int, List[float]] = None
) -> Dict[int, Any]:
    """
    Prepare estimators for each layer.
    
    Either loads pre-trained estimators or trains new ones on provided data.
    
    Args:
        config: Experiment configuration
        num_layers: Number of layers in the model
        training_data: Optional dict mapping layer_idx to training logits
        
    Returns:
        Dict mapping layer_idx to fitted estimator instances
    """
    estimators = {}
    
    # Try to load pre-trained models first
    if config.estimator.model_dir:
        print(f"Attempting to load pre-trained estimators from {config.estimator.model_dir}...")
        # For now, we'll create fresh estimators and fit them
        # In the future, we could save/load fitted estimator objects
    
    # Create and fit estimators for each layer
    print(f"Creating {config.estimator.type} estimators for {num_layers} layers...")
    
    for layer_idx in tqdm(range(num_layers), desc="Preparing estimators"):
        estimator = create_estimator(config.estimator)
        
        # If training data is provided, fit the estimator
        if training_data and layer_idx in training_data:
            layer_data = np.array(training_data[layer_idx])
            estimator.fit(layer_data)
        
        estimators[layer_idx] = estimator
    
    print(f"✅ Prepared estimators for {num_layers} layers")
    return estimators


def patch_model_with_hc_routing(model, config: ExperimentConfig, estimators: Dict = None):
    """
    Patch model with HC routing using model-specific router logger.
    
    This is a simplified version - you'll need to adapt based on your
    actual router patching implementation for DeepSeek/Qwen.
    """
    # Import model-specific logging
    if config.model.name == "deepseek":
        from moe_internal_logging_deepseek import RouterLogger
    elif config.model.name == "qwen":
        from moe_internal_logging_qwen import RouterLogger
    else:
        raise ValueError(f"Unknown model: {config.model.name}")
    
    # For now, this is a placeholder
    # You'll need to implement actual router patching similar to OLMoE's HCRouterPatcher
    print(f"⚠️  Router patching not fully implemented yet")
    print(f"   This would patch {config.model.name} with HC routing")
    print(f"   HC params: beta={config.hc_routing.beta}, min_k={config.hc_routing.min_k}, max_k={config.hc_routing.max_k}")
    
    return None  # Return patcher instance when implemented


def run_evaluation(
    model,
    tokenizer,
    dataset_samples: List[Dict],
    dataset_name: str,
    config: ExperimentConfig,
    output_dirs: Dict[str, Path]
) -> Dict[str, Any]:
    """Run evaluation on dataset with HC routing."""
    
    print(f"\n{'='*60}")
    print(f"Evaluating on {dataset_name.upper()}")
    print(f"{'='*60}\n")
    
    device = config.device
    metrics_computer = HCMetricsComputer()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataset_samples, desc=f"Evaluating {dataset_name}")):
            text = sample.get("text", "")
            if not text.strip():
                continue
            
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=config.evaluation.max_length,
                padding=False,
            )
            input_ids = inputs["input_ids"].to(device)
            
            if input_ids.shape[1] < 2:
                continue
            
            # Forward pass
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
            num_tokens = input_ids.shape[1]
            
            total_loss += loss * num_tokens
            total_tokens += num_tokens
    
    # Compute metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = float(np.exp(avg_loss))
    
    results = {
        "dataset": dataset_name,
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
        "num_samples": len(dataset_samples),
        "timestamp": datetime.now().isoformat(),
    }
    
    print(f"\n✅ Evaluation complete:")
    print(f"   Perplexity: {perplexity:.2f}")
    print(f"   Avg Loss: {avg_loss:.4f}")
    print(f"   Total Tokens: {total_tokens}")
    
    # Save results
    results_file = output_dirs["metrics"] / f"{dataset_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Results saved to: {results_file}")
    
    return results


def main():
    """Main workflow execution."""
    parser = argparse.ArgumentParser(description="Run HC routing experiments")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup output directories
    output_dirs = setup_output_directories(config)
    
    # Save configuration to output directory
    config_file = output_dirs["base"] / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config.dict(), f, default_flow_style=False)
    print(f"✅ Configuration saved to {config_file}")
    
    # Load model and tokenizer
    print(f"\n{'='*60}")
    print("LOADING MODEL")
    print(f"{'='*60}\n")
    model, tokenizer = load_model_from_config(config)
    
    # Get number of layers (model-specific)
    # This is a placeholder - you'll need to extract this properly
    num_layers = 16  # Default, adjust based on model
    
    # Prepare estimators
    # For now, creating empty estimators - in full implementation,
    # you'd first collect training data, then fit estimators
    print(f"\n{'='*60}")
    print("PREPARING ESTIMATORS")
    print(f"{'='*60}\n")
    estimators = prepare_estimators_for_layers(config, num_layers)
    
    # Patch model with HC routing
    print(f"\n{'='*60}")
    print("PATCHING MODEL WITH HC ROUTING")
    print(f"{'='*60}\n")
    patcher = patch_model_with_hc_routing(model, config, estimators)
    
    # Run evaluation on each dataset
    all_results = {}
    
    for dataset_name in config.evaluation.datasets:
        # Load dataset
        dataset_samples = load_dataset(dataset_name, config.evaluation.max_samples)
        
        # Run evaluation
        results = run_evaluation(
            model,
            tokenizer,
            dataset_samples,
            dataset_name,
            config,
            output_dirs
        )
        
        all_results[dataset_name] = results
    
    # Save combined results
    combined_results = {
        "config": config.dict(),
        "results": all_results,
        "timestamp": datetime.now().isoformat(),
    }
    
    combined_file = output_dirs["base"] / "all_results.json"
    with open(combined_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("WORKFLOW COMPLETE")
    print(f"{'='*60}\n")
    print(f"Results saved to: {output_dirs['base']}")
    print("\nSummary:")
    for dataset_name, results in all_results.items():
        print(f"  {dataset_name}: Perplexity = {results['perplexity']:.2f}")


if __name__ == "__main__":
    main()
