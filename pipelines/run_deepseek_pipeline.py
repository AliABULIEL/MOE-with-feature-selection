"""
DeepSeek-V2-Lite MoE Analysis Pipeline
======================================

Complete end-to-end pipeline for DeepSeek-V2-Lite MoE router analysis.
Combines internal routing logging + KDE training + visualization.

This script uses the existing RouterLogger from moe_internal_logging_deepseek.py

Usage:
    python run_deepseek_pipeline.py

Stages:
    1. Model Loading & Evaluation with Internal Routing Logging
    2. KDE Model Training  
    3. Basic Plots (expert weights, choice counts, logit distributions)
    4. Per-Expert Plots (optional, generates many plots)
    5. KDE Analysis Plots
    6. P-Value Distribution Plots

All parameters are configurable via the CONFIG dictionary below.
"""

import os
import sys
import json
import pickle
import textwrap
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

# Add parent directory to path so we can import the logging modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the existing RouterLogger
from moe_internal_logging_deepseek import RouterLogger, InternalRoutingLogger

# ==============================================================================
# CONFIGURABLE PARAMETERS
# ==============================================================================

CONFIG = {
    # Model Configuration
    "model_id": "deepseek-ai/DeepSeek-V2-Lite",
    "model_name": "deepseek",
    "num_experts": 64,
    "default_top_k": 6,
    
    # Dataset Configuration
    "datasets": ["lambada", "hellaswag", "wikitext"],
    "max_samples": {
        "lambada": 500,
        "hellaswag": 500,
        "wikitext": 300,
    },
    "max_length": 512,
    
    # KDE Configuration
    "kde_train_dataset": "lambada",
    "kde_test_datasets": ["hellaswag", "wikitext"],
    "kde_trim_amounts": [0, 1, 2, 4],
    "kde_kernels": ["gaussian", "tophat", "epanechnikov", "linear"],
    
    # Output Configuration
    "output_dir": "./outputs/deepseek",
    
    # Pipeline Stage Toggles
    "run_evaluation": True,
    "run_kde_training": True,
    "run_basic_plots": True,
    "run_per_expert_plots": True,
    "run_kde_plots": True,
    "run_pvalue_plots": True,
    
    # Hardware Configuration
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": "bfloat16",
    
    # Plot Configuration
    "plot_format": "png",
    "plot_dpi": 100,
    "show_plots": False,
}


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def setup_directories(config: Dict) -> Dict[str, Path]:
    """Create output directory structure."""
    base_dir = Path(config["output_dir"])
    dirs = {
        "base": base_dir,
        "logs": base_dir / "logs",
        "kde_models": base_dir / "kde_models",
        "plots_basic": base_dir / "plots" / "basic",
        "plots_per_expert": base_dir / "plots" / "per_expert",
        "plots_kde": base_dir / "plots" / "kde",
        "plots_pvalue": base_dir / "plots" / "pvalue",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def save_plot(fig, path: Path, config: Dict, close: bool = True):
    """Save plot to file."""
    fig.savefig(path, format=config["plot_format"], dpi=config["plot_dpi"], 
                bbox_inches='tight')
    if config["show_plots"]:
        plt.show()
    if close:
        plt.close(fig)


def get_ordinal(n: int) -> str:
    """Get ordinal string for a number (1st, 2nd, 3rd, etc.)."""
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"


# ==============================================================================
# STAGE 1: MODEL LOADING
# ==============================================================================

def load_model(config: Dict) -> Tuple[Any, Any]:
    """Load DeepSeek-V2-Lite model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model: {config['model_id']}")
    print(f"Device: {config['device']}, Dtype: {config['dtype']}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        torch_dtype=get_torch_dtype(config["dtype"]),
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    print(f"✅ Model loaded successfully")
    return model, tokenizer


# ==============================================================================
# STAGE 1B: DATASET LOADING
# ==============================================================================

def load_dataset_samples(dataset_name: str, max_samples: int) -> List[Dict]:
    """Load dataset samples."""
    from datasets import load_dataset
    
    print(f"Loading {dataset_name} dataset (max {max_samples} samples)...")
    
    if dataset_name == "lambada":
        try:
            dataset = load_dataset("lambada", split="test")
        except:
            dataset = load_dataset("EleutherAI/lambada_openai", "en", split="test")
        
        samples = []
        for item in dataset:
            text = item.get('text', '')
            if text.strip():
                samples.append({"text": text, "target": text.split()[-1] if text else ""})
            if len(samples) >= max_samples:
                break
                
    elif dataset_name == "hellaswag":
        dataset = load_dataset("hellaswag", split="validation")
        
        samples = []
        for item in dataset:
            ctx = item.get('ctx', '')
            endings = item.get('endings', [])
            label = int(item.get('label', 0))
            
            if ctx and endings and 0 <= label < len(endings):
                samples.append({
                    "ctx": ctx,
                    "endings": endings,
                    "label": label,
                    "text": f"{ctx} {endings[label]}"
                })
            if len(samples) >= max_samples:
                break
                
    elif dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        
        samples = []
        for item in dataset:
            text = item.get('text', '')
            if text.strip() and len(text.split()) > 10:
                samples.append({"text": text})
            if len(samples) >= max_samples:
                break
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"✅ Loaded {len(samples)} samples from {dataset_name}")
    return samples


# ==============================================================================
# STAGE 1C: EVALUATION WITH LOGGING (USES YOUR RouterLogger)
# ==============================================================================

def run_evaluation(
    model, 
    tokenizer, 
    samples: List[Dict],
    dataset_name: str,
    config: Dict,
    output_dir: Path
) -> Tuple[Dict[str, Any], Path]:
    """
    Run evaluation on dataset with internal routing logging.
    
    Uses RouterLogger from moe_internal_logging_deepseek.py to capture
    router logits during inference.
    """
    
    print(f"\n{'='*60}")
    print(f"Evaluating on {dataset_name.upper()}")
    print(f"{'='*60}")
    
    device = config["device"]
    max_length = config["max_length"]
    
    # Use the imported RouterLogger from moe_internal_logging_deepseek.py
    router_logger = RouterLogger(model)
    router_logger.register_hooks(top_k=config["default_top_k"])
    
    # Prepare output data structure matching logs_eda.ipynb expected format
    all_samples_data = []
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, sample in enumerate(tqdm(samples, desc=f"Evaluating {dataset_name}")):
            text = sample.get("text", "")
            if not text.strip():
                continue
            
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False
            )
            input_ids = inputs["input_ids"].to(device)
            
            if input_ids.shape[1] < 2:
                continue
            
            # Clear previous routing data
            router_logger.clear_data()
            
            # Forward pass - hooks capture routing data
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
            num_tokens = input_ids.shape[1]
            
            # Get routing data captured by hooks
            routing_data = router_logger.get_routing_data()
            
            # Convert to logs_eda.ipynb format
            sample_data = {
                "sample_id": i,
                "num_tokens": num_tokens,
                "loss": loss,
                "layers": []
            }
            
            for layer_data in routing_data:
                layer_entry = {
                    "layer": layer_data["layer"],
                    "router_logits_shape": list(layer_data["router_logits"].shape),
                    "selected_experts": layer_data["expert_indices"].numpy().tolist(),
                    "expert_weights": layer_data["expert_weights"].numpy().tolist(),
                    "router_logits_sample": layer_data["router_logits"].numpy().tolist(),
                }
                sample_data["layers"].append(layer_entry)
            
            all_samples_data.append(sample_data)
            total_loss += loss * num_tokens
            total_tokens += num_tokens
    
    router_logger.remove_hooks()
    
    # Compute metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = float(np.exp(avg_loss))
    
    # Save in logs_eda.ipynb expected format
    output_data = {
        "config": config["model_id"],
        "strategy": "topk_baseline",
        "num_experts": config["num_experts"],
        "top_k": config["default_top_k"],
        "dataset": dataset_name,
        "max_samples": config["max_samples"].get(dataset_name, 0),
        "timestamp": datetime.now().isoformat(),
        "num_layers": len(all_samples_data[0]["layers"]) if all_samples_data else 0,
        "samples": all_samples_data
    }
    
    json_path = output_dir / f"{config['model_name']}_{dataset_name}_internal_routing.json"
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Evaluation complete")
    print(f"   Perplexity: {perplexity:.2f}")
    print(f"   Avg Loss: {avg_loss:.4f}")
    print(f"   Total Tokens: {total_tokens}")
    print(f"   Saved to: {json_path}")
    
    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
        "num_samples": len(all_samples_data)
    }, json_path


# ==============================================================================
# STAGE 2: KDE MODEL TRAINING (from logs_eda.ipynb)
# ==============================================================================

def load_routing_data(json_path: Path) -> Dict:
    """Load routing data from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_router_logits(data: Dict) -> Tuple[np.ndarray, List, List]:
    """
    Extract router logits from JSON data.
    
    This mirrors the extraction logic in logs_eda.ipynb:
    - layers_router_logits_raw[layer_idx].extend(layer['router_logits_sample'])
    
    Returns:
        layers_router_logits_raw: np.ndarray of shape (num_layers, num_tokens, num_experts)
        layers_expert_weights: List of expert weights per layer
        layers_expert_choices: List of selected experts per layer
    """
    num_layers = data["num_layers"]
    
    layers_router_logits_raw = [[] for _ in range(num_layers)]
    layers_expert_weights = [[] for _ in range(num_layers)]
    layers_expert_choices = [[] for _ in range(num_layers)]
    
    for sample in data["samples"]:
        for layer_data in sample["layers"]:
            layer_idx = layer_data["layer"]
            layers_router_logits_raw[layer_idx].extend(layer_data["router_logits_sample"])
            layers_expert_weights[layer_idx].extend(layer_data["expert_weights"])
            layers_expert_choices[layer_idx].extend(layer_data["selected_experts"])
    
    # Convert to numpy arrays
    layers_router_logits_raw = np.array([np.array(arr) for arr in layers_router_logits_raw])
    
    return layers_router_logits_raw, layers_expert_weights, layers_expert_choices


def trim_top_and_bottom_experts(arr: np.ndarray, trim_amount: int = 2) -> np.ndarray:
    """
    Trim top and bottom experts from router logits.
    
    From logs_eda.ipynb:
        sorted_indices = np.argsort(arr, axis=-1)
        kept_indices = sorted_indices[:, :, trim_amount:-trim_amount]
        trimmed_arr = np.take_along_axis(arr, kept_indices, axis=-1)
    """
    if trim_amount == 0:
        return arr
    
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    
    sorted_indices = np.argsort(arr, axis=-1)
    kept_indices = sorted_indices[:, :, trim_amount:-trim_amount]
    trimmed_arr = np.take_along_axis(arr, kept_indices, axis=-1)
    
    return trimmed_arr.squeeze()


def train_kde_models(
    layers_router_logits: np.ndarray,
    output_dir: Path,
    model_name: str,
    config: Dict
) -> Dict[int, Dict]:
    """
    Train KDE models for each layer.
    
    From logs_eda.ipynb:
        kde = gaussian_kde(train_data.T)
        pdf_grid = kde.evaluate(x_grid)
        cdf_grid = np.cumsum(pdf_grid)
        cdf_grid /= cdf_grid[-1]
        pickle.dump({'x': x_grid, 'cdf': cdf_grid}, f)
    """
    
    print(f"\n{'='*60}")
    print("TRAINING KDE MODELS")
    print(f"{'='*60}")
    
    kde_models = {}
    num_layers = layers_router_logits.shape[0]
    
    for layer_idx in tqdm(range(num_layers), desc="Training KDE models"):
        layer_data = layers_router_logits[layer_idx].flatten()
        
        # Train basic gaussian KDE (scipy)
        kde = gaussian_kde(layer_data)
        
        # Create evaluation grid
        data_min, data_max = layer_data.min(), layer_data.max()
        x_grid = np.linspace(
            data_min - 0.2 * abs(data_min),
            data_max + 0.2 * abs(data_max),
            10000
        )
        
        # Compute PDF and CDF
        pdf_grid = kde.evaluate(x_grid)
        cdf_grid = np.cumsum(pdf_grid)
        cdf_grid /= cdf_grid[-1]
        
        # Save model
        model_data = {"x": x_grid, "cdf": cdf_grid}
        model_path = output_dir / f"{model_name}_distribution_model_layer_{layer_idx}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        kde_models[layer_idx] = model_data
        
        # Train trimmed KDE variants
        for trim_amt in config["kde_trim_amounts"]:
            if trim_amt == 0:
                continue
            
            trimmed_data = trim_top_and_bottom_experts(
                layers_router_logits[layer_idx][np.newaxis, :, :], 
                trim_amt
            )
            trimmed_flat = trimmed_data.flatten()
            
            kde_trimmed = gaussian_kde(trimmed_flat)
            t_min, t_max = trimmed_flat.min(), trimmed_flat.max()
            x_grid_t = np.linspace(t_min - 0.2*abs(t_min), t_max + 0.2*abs(t_max), 10000)
            
            pdf_t = kde_trimmed.evaluate(x_grid_t)
            cdf_t = np.cumsum(pdf_t)
            cdf_t /= cdf_t[-1]
            
            model_path_t = output_dir / f"{model_name}_distribution_model_layer_{layer_idx}_trimmed_{trim_amt*2}.pkl"
            with open(model_path_t, 'wb') as f:
                pickle.dump({"x": x_grid_t, "cdf": cdf_t}, f)
        
        # Train with different kernels (sklearn)
        for kernel_type in config["kde_kernels"]:
            n = len(layer_data)
            std_dev = np.std(layer_data)
            bandwidth = 1.06 * std_dev * (n ** (-1/5)) if std_dev > 0 else 1.0
            
            kde_sklearn = KernelDensity(kernel=kernel_type, bandwidth=bandwidth)
            kde_sklearn.fit(layer_data[:, np.newaxis])
            
            log_pdf = kde_sklearn.score_samples(x_grid[:, np.newaxis])
            pdf_k = np.exp(log_pdf)
            cdf_k = np.cumsum(pdf_k)
            cdf_k /= cdf_k[-1]
            
            model_path_k = output_dir / f"{model_name}_distribution_model_layer_{layer_idx}_{kernel_type}.pkl"
            with open(model_path_k, 'wb') as f:
                pickle.dump({"x": x_grid, "cdf": cdf_k}, f)
    
    print(f"✅ Trained KDE models for {num_layers} layers")
    return kde_models


# ==============================================================================
# STAGE 3: BASIC PLOTS (from logs_eda.ipynb)
# ==============================================================================

def generate_basic_plots(
    data: Dict,
    layers_router_logits: np.ndarray,
    layers_expert_weights: List,
    layers_expert_choices: List,
    output_dir: Path,
    model_name: str,
    config: Dict
):
    """
    Generate basic analysis plots.
    
    From logs_eda.ipynb:
    - Expert weight sum distributions per layer
    - Expert choice counts per layer
    - Router softmax distributions
    - Raw router logit distributions
    """
    
    print(f"\n{'='*60}")
    print("GENERATING BASIC PLOTS")
    print(f"{'='*60}")
    
    num_layers = layers_router_logits.shape[0]
    num_experts = layers_router_logits.shape[2]
    
    # Convert expert weights to numpy
    layers_expert_weights_np = [np.array(weights) for weights in layers_expert_weights]
    
    # 1. Expert weight sum distributions per layer
    print("Plotting expert weight sum distributions...")
    weights_sum_dir = output_dir / "expert_weights_sum"
    weights_sum_dir.mkdir(exist_ok=True)
    
    layers_weights_sum = []
    for layer_idx in tqdm(range(num_layers), desc="Weight sum plots"):
        weights_sum = np.sum(layers_expert_weights_np[layer_idx], axis=1)
        layers_weights_sum.append(weights_sum)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(weights_sum, bins=50, kde=True, ax=ax)
        ax.set_xlim(0, 1)
        ax.set_title(f"Distribution of Sum of Expert Weights - Layer {layer_idx}")
        ax.set_xlabel("Sum of Expert Weights")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        
        save_plot(fig, weights_sum_dir / f"{model_name}_layer_{layer_idx}_weights_sum.png", config)
    
    # Combined weight sum distribution
    combined_weights_sum = np.concatenate(layers_weights_sum)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(combined_weights_sum, bins=50, kde=True, ax=ax)
    ax.set_xlim(0, 1)
    ax.set_title(f"{model_name.upper()} - Distribution of Sum of Expert Weights (All Layers)")
    ax.set_xlabel("Sum of Expert Weights")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    save_plot(fig, output_dir / f"{model_name}_combined_weights_sum.png", config)
    
    # 2. Expert choice counts per layer
    print("Plotting expert choice counts...")
    choice_counts_dir = output_dir / "expert_choice_counts"
    choice_counts_dir.mkdir(exist_ok=True)
    
    layers_choice_counts = []
    for layer_idx in tqdm(range(num_layers), desc="Choice count plots"):
        flat_choices = [exp for token_choices in layers_expert_choices[layer_idx] for exp in token_choices]
        unique, counts = np.unique(flat_choices, return_counts=True)
        choice_dict = dict(zip(unique, counts))
        layers_choice_counts.append(choice_dict)
        
        fig, ax = plt.subplots(figsize=(20, 6))
        experts = list(range(num_experts))
        counts_list = [choice_dict.get(e, 0) for e in experts]
        sns.barplot(x=experts, y=counts_list, ax=ax)
        ax.set_title(f"Expert Choice Counts - Layer {layer_idx}")
        ax.set_xlabel("Expert")
        ax.set_ylabel("Count")
        
        save_plot(fig, choice_counts_dir / f"{model_name}_layer_{layer_idx}_choice_counts.png", config)
    
    # 3. Router softmax distributions per layer
    print("Plotting router softmax distributions...")
    softmax_dir = output_dir / "router_softmax"
    softmax_dir.mkdir(exist_ok=True)
    
    for layer_idx in tqdm(range(num_layers), desc="Softmax plots"):
        layer_logits = layers_router_logits[layer_idx]
        softmax_logits = np.exp(layer_logits) / np.sum(np.exp(layer_logits), axis=1, keepdims=True)
        flat_softmax = softmax_logits.flatten()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(flat_softmax, bins=100, kde=True, ax=ax)
        ax.set_title(f"Distribution of Router Softmax Outputs - Layer {layer_idx}")
        ax.set_xlabel("Router Softmax Output")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        
        save_plot(fig, softmax_dir / f"{model_name}_layer_{layer_idx}_softmax.png", config)
    
    # 4. Raw router logit distributions per layer
    print("Plotting raw router logit distributions...")
    raw_logits_dir = output_dir / "router_raw_logits"
    raw_logits_dir.mkdir(exist_ok=True)
    
    for layer_idx in tqdm(range(num_layers), desc="Raw logit plots"):
        flat_logits = layers_router_logits[layer_idx].flatten()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(flat_logits, bins=100, kde=True, ax=ax)
        ax.set_title(f"Distribution of Raw Router Logits - Layer {layer_idx}")
        ax.set_xlabel("Raw Router Logit")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        
        save_plot(fig, raw_logits_dir / f"{model_name}_layer_{layer_idx}_raw_logits.png", config)
    
    print(f"✅ Generated basic plots")
    return layers_choice_counts


# ==============================================================================
# STAGE 4: PER-EXPERT PLOTS
# ==============================================================================

def generate_per_expert_plots(
    layers_router_logits: np.ndarray,
    output_dir: Path,
    model_name: str,
    config: Dict
):
    """Generate per-expert distribution plots (many plots!)."""
    
    print(f"\n{'='*60}")
    print("GENERATING PER-EXPERT PLOTS")
    print(f"{'='*60}")
    
    num_layers = layers_router_logits.shape[0]
    num_experts = layers_router_logits.shape[2]
    
    softmax_expert_dir = output_dir / "per_expert_softmax"
    softmax_expert_dir.mkdir(exist_ok=True)
    
    raw_expert_dir = output_dir / "per_expert_raw_logits"
    raw_expert_dir.mkdir(exist_ok=True)
    
    total_plots = num_layers * num_experts * 2
    print(f"Generating {total_plots} plots...")
    
    for layer_idx in tqdm(range(num_layers), desc="Per-expert plots"):
        layer_logits = layers_router_logits[layer_idx]
        softmax_logits = np.exp(layer_logits) / np.sum(np.exp(layer_logits), axis=1, keepdims=True)
        
        for expert_idx in range(num_experts):
            # Softmax distribution
            expert_softmax = softmax_logits[:, expert_idx]
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(expert_softmax, bins=100, kde=True, ax=ax)
            ax.set_title(f"Softmax Distribution - Layer {layer_idx}, Expert {expert_idx}")
            ax.set_xlabel("Softmax Output")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)
            save_plot(fig, softmax_expert_dir / f"{model_name}_layer_{layer_idx}_expert_{expert_idx}_softmax.png", config)
            
            # Raw logit distribution
            expert_raw = layer_logits[:, expert_idx]
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(expert_raw, bins=100, kde=True, ax=ax)
            ax.set_title(f"Raw Logit Distribution - Layer {layer_idx}, Expert {expert_idx}")
            ax.set_xlabel("Raw Logit")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)
            save_plot(fig, raw_expert_dir / f"{model_name}_layer_{layer_idx}_expert_{expert_idx}_raw.png", config)
    
    print(f"✅ Generated {total_plots} per-expert plots")


# ==============================================================================
# STAGE 5: KDE ANALYSIS PLOTS
# ==============================================================================

def generate_kde_plots(
    train_logits: np.ndarray,
    test_logits: np.ndarray,
    kde_models_dir: Path,
    output_dir: Path,
    model_name: str,
    train_dataset: str,
    test_dataset: str,
    config: Dict
):
    """Generate KDE analysis plots."""
    
    print(f"\n{'='*60}")
    print(f"GENERATING KDE PLOTS ({train_dataset} -> {test_dataset})")
    print(f"{'='*60}")
    
    num_layers = train_logits.shape[0]
    
    # 1. KDE cumulative density plots
    kde_density_dir = output_dir / "kde_cumulative_density"
    kde_density_dir.mkdir(exist_ok=True)
    
    print("Plotting KDE cumulative density...")
    for layer_idx in tqdm(range(num_layers), desc="KDE density plots"):
        model_path = kde_models_dir / f"{model_name}_distribution_model_layer_{layer_idx}.pkl"
        with open(model_path, 'rb') as f:
            kde_data = pickle.load(f)
        
        x_grid, cdf_grid = kde_data['x'], kde_data['cdf']
        test_flat = test_logits[layer_idx].flatten()
        probabilities = np.interp(test_flat, x_grid, cdf_grid)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(probabilities, bins=100, ax=ax)
        ax.set_xlim(0, 1)
        ax.set_title(f"KDE Cumulative Density - Layer {layer_idx} on {test_dataset}")
        ax.set_xlabel("Cumulative Density")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        save_plot(fig, kde_density_dir / f"{model_name}_kde_density_layer_{layer_idx}_{test_dataset}.png", config)
    
    # 2. Trimmed logit comparison plots
    print("Plotting trimmed logit comparisons...")
    trimmed_dir = output_dir / "trimmed_comparison"
    trimmed_dir.mkdir(exist_ok=True)
    
    trim_amounts = config["kde_trim_amounts"]
    
    for layer_idx in tqdm(range(num_layers), desc="Trimmed comparison plots"):
        rows = int(np.ceil(len(trim_amounts) / 2))
        fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))
        axes = axes.flatten()
        
        for i, trim_amt in enumerate(trim_amounts):
            if trim_amt == 0:
                train_trimmed = train_logits[layer_idx].flatten()
                test_trimmed = test_logits[layer_idx].flatten()
            else:
                train_trimmed = trim_top_and_bottom_experts(
                    train_logits[layer_idx][np.newaxis, :, :], trim_amt
                ).flatten()
                test_trimmed = trim_top_and_bottom_experts(
                    test_logits[layer_idx][np.newaxis, :, :], trim_amt
                ).flatten()
            
            ax = axes[i]
            sns.histplot(train_trimmed, bins=100, kde=True, color='blue', 
                        label=train_dataset, alpha=0.5, ax=ax)
            sns.histplot(test_trimmed, bins=100, kde=True, color='orange', 
                        label=test_dataset, alpha=0.5, ax=ax)
            
            ax.set_title(f"Layer {layer_idx} - Trimmed {trim_amt} experts each side")
            ax.set_xlabel("Router Logit")
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        for j in range(len(trim_amounts), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        save_plot(fig, trimmed_dir / f"{model_name}_trimmed_comparison_layer_{layer_idx}.png", config)
    
    # 3. Kernel comparison plots
    print("Plotting kernel comparisons...")
    kernel_dir = output_dir / "kernel_comparison"
    kernel_dir.mkdir(exist_ok=True)
    
    for layer_idx in tqdm(range(num_layers), desc="Kernel comparison plots"):
        fig, ax = plt.subplots(figsize=(12, 7))
        test_flat = test_logits[layer_idx].flatten()
        
        for kernel_type in config["kde_kernels"]:
            model_path = kde_models_dir / f"{model_name}_distribution_model_layer_{layer_idx}_{kernel_type}.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    kde_data = pickle.load(f)
                probs = np.interp(test_flat, kde_data['x'], kde_data['cdf'])
                sns.histplot(probs, bins=100, element="step", fill=False,
                           label=kernel_type.capitalize(), alpha=0.7, linewidth=1.5, ax=ax)
        
        ax.set_xlim(0, 1)
        ax.set_title(f"KDE Kernel Comparison - Layer {layer_idx}")
        ax.set_xlabel("Cumulative Density")
        ax.set_ylabel("Frequency")
        ax.legend(title="Kernel")
        ax.grid(True, alpha=0.3)
        save_plot(fig, kernel_dir / f"{model_name}_kernel_comparison_layer_{layer_idx}.png", config)
    
    print(f"✅ Generated KDE analysis plots")


# ==============================================================================
# STAGE 6: P-VALUE PLOTS
# ==============================================================================

def generate_pvalue_plots(
    train_logits: np.ndarray,
    test_logits: np.ndarray,
    layers_choice_counts: List[Dict],
    kde_models_dir: Path,
    output_dir: Path,
    model_name: str,
    test_dataset: str,
    config: Dict
):
    """Generate p-value distribution plots."""
    
    print(f"\n{'='*60}")
    print(f"GENERATING P-VALUE PLOTS")
    print(f"{'='*60}")
    
    num_layers = train_logits.shape[0]
    num_experts = train_logits.shape[2]
    rng = np.random.default_rng(seed=42)
    
    # 1. Basic p-value distributions
    pvalue_dir = output_dir / "pvalue_basic"
    pvalue_dir.mkdir(exist_ok=True)
    
    print("Plotting basic p-value distributions...")
    for layer_idx in tqdm(range(num_layers), desc="P-value plots"):
        model_path = kde_models_dir / f"{model_name}_distribution_model_layer_{layer_idx}.pkl"
        with open(model_path, 'rb') as f:
            kde_data = pickle.load(f)
        
        x_grid, cdf_grid = kde_data['x'], kde_data['cdf']
        test_flat = test_logits[layer_idx].flatten()
        probabilities = np.interp(test_flat, x_grid, cdf_grid)
        p_values = 1 - probabilities
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(p_values, bins=100, ax=ax)
        ax.set_xlim(0, 1)
        ax.set_title(f"P-Value Distribution - Layer {layer_idx} on {test_dataset}")
        ax.set_xlabel("P-Value")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        save_plot(fig, pvalue_dir / f"{model_name}_pvalue_layer_{layer_idx}_{test_dataset}.png", config)
    
    # 2. P-value vs uniform distribution
    pvalue_uniform_dir = output_dir / "pvalue_vs_uniform"
    pvalue_uniform_dir.mkdir(exist_ok=True)
    
    print("Plotting p-value vs uniform comparisons...")
    for layer_idx in tqdm(range(num_layers), desc="P-value uniform plots"):
        model_path = kde_models_dir / f"{model_name}_distribution_model_layer_{layer_idx}.pkl"
        with open(model_path, 'rb') as f:
            kde_data = pickle.load(f)
        
        x_grid, cdf_grid = kde_data['x'], kde_data['cdf']
        test_data = test_logits[layer_idx]
        uniform_samples = rng.uniform(0, 1, size=test_data.shape)
        
        for j in range(0, num_experts, 4):
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            axes = axes.flatten()
            
            for k_offset, ax in enumerate(axes):
                k = j + k_offset
                if k >= num_experts:
                    ax.set_visible(False)
                    continue
                
                kth_best_logits = np.sort(test_data, axis=1)[:, -(k+1)]
                probabilities = np.interp(kth_best_logits, x_grid, cdf_grid)
                p_values = 1 - probabilities
                kth_uniform = np.sort(uniform_samples, axis=1)[:, k]
                
                common_bins = np.linspace(0, 1, 101)
                sns.histplot(p_values, bins=common_bins, color='blue', 
                           label='KDE P-Value', alpha=0.7, ax=ax)
                sns.histplot(kth_uniform, bins=common_bins, color='orange',
                           label='Uniform', alpha=0.5, ax=ax)
                
                ax.set_title(f"{get_ordinal(k + 1)} Best Expert - Layer {layer_idx}")
                ax.set_xlabel("P-Value")
                ax.set_ylabel("Frequency")
                ax.set_xlim(0, 1)
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_plot(fig, pvalue_uniform_dir / 
                     f"{model_name}_pvalue_uniform_layer_{layer_idx}_experts_{j+1}_to_{min(j+4, num_experts)}.png", 
                     config)
    
    # 3. P-value by most chosen experts
    pvalue_chosen_dir = output_dir / "pvalue_by_popularity"
    pvalue_chosen_dir.mkdir(exist_ok=True)
    
    print("Plotting p-values by expert popularity...")
    for layer_idx in tqdm(range(num_layers), desc="P-value popularity plots"):
        model_path = kde_models_dir / f"{model_name}_distribution_model_layer_{layer_idx}.pkl"
        with open(model_path, 'rb') as f:
            kde_data = pickle.load(f)
        
        x_grid, cdf_grid = kde_data['x'], kde_data['cdf']
        test_data = test_logits[layer_idx]
        
        choice_dict = layers_choice_counts[layer_idx]
        sorted_experts = sorted(range(num_experts), 
                               key=lambda e: choice_dict.get(e, 0), reverse=True)
        
        for j in range(0, num_experts, 4):
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            axes = axes.flatten()
            
            for i, ax in enumerate(axes):
                rank = j + i
                if rank >= num_experts:
                    ax.set_visible(False)
                    continue
                
                expert_idx = sorted_experts[rank]
                expert_logits = test_data[:, expert_idx]
                probabilities = np.interp(expert_logits, x_grid, cdf_grid)
                p_values = 1 - probabilities
                
                total_samples = len(p_values)
                num_bins = 50
                edges = np.linspace(0, 1, num_bins + 1)
                samples_per_bin = total_samples // num_bins
                bin_centers = 0.5 * (edges[:-1] + edges[1:])
                uniform_data = np.repeat(bin_centers, samples_per_bin)
                
                sns.histplot(p_values, bins=50, color='blue', label='KDE P-Value', ax=ax)
                sns.histplot(uniform_data, bins=edges, color='orange', label='Uniform', alpha=0.5, ax=ax)
                
                ax.set_title(f"{rank+1}-th Most Chosen Expert ({expert_idx}) - Layer {layer_idx}")
                ax.set_xlabel("P-Value")
                ax.set_ylabel("Frequency")
                ax.set_xlim(0, 1)
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_plot(fig, pvalue_chosen_dir / 
                     f"{model_name}_pvalue_popularity_layer_{layer_idx}_rank_{j+1}_to_{min(j+4, num_experts)}.png", 
                     config)
    
    print(f"✅ Generated p-value plots")


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def run_pipeline(config: Dict):
    """Run the complete analysis pipeline."""
    
    print("="*70)
    print(f"DEEPSEEK-V2-LITE MOE ANALYSIS PIPELINE")
    print(f"Model: {config['model_id']}")
    print(f"Output: {config['output_dir']}")
    print("="*70)
    
    dirs = setup_directories(config)
    routing_data = {}
    
    # =========================================================================
    # STAGE 1: EVALUATION WITH INTERNAL LOGGING
    # Uses RouterLogger from moe_internal_logging_deepseek.py
    # =========================================================================
    if config["run_evaluation"]:
        print("\n" + "="*70)
        print("STAGE 1: MODEL EVALUATION WITH INTERNAL LOGGING")
        print("(Using RouterLogger from moe_internal_logging_deepseek.py)")
        print("="*70)
        
        model, tokenizer = load_model(config)
        
        for dataset_name in config["datasets"]:
            samples = load_dataset_samples(
                dataset_name, 
                config["max_samples"][dataset_name]
            )
            
            metrics, json_path = run_evaluation(
                model, tokenizer, samples, dataset_name, config, dirs["logs"]
            )
        
        del model, tokenizer
        torch.cuda.empty_cache()
    
    # =========================================================================
    # LOAD ROUTING DATA
    # =========================================================================
    print("\n" + "="*70)
    print("LOADING ROUTING DATA")
    print("="*70)
    
    for dataset_name in config["datasets"]:
        json_path = dirs["logs"] / f"{config['model_name']}_{dataset_name}_internal_routing.json"
        if json_path.exists():
            data = load_routing_data(json_path)
            logits, weights, choices = extract_router_logits(data)
            routing_data[dataset_name] = {
                "data": data,
                "logits": logits,
                "weights": weights,
                "choices": choices
            }
            print(f"✅ Loaded {dataset_name}: {logits.shape}")
        else:
            print(f"⚠️  {dataset_name} routing data not found at {json_path}")
    
    if not routing_data:
        print("❌ No routing data available. Run evaluation first.")
        return
    
    # =========================================================================
    # STAGE 2: KDE MODEL TRAINING
    # =========================================================================
    if config["run_kde_training"]:
        train_dataset = config["kde_train_dataset"]
        if train_dataset in routing_data:
            train_kde_models(
                routing_data[train_dataset]["logits"],
                dirs["kde_models"],
                config["model_name"],
                config
            )
    
    # =========================================================================
    # STAGE 3: BASIC PLOTS
    # =========================================================================
    layers_choice_counts = None
    if config["run_basic_plots"]:
        train_dataset = config["kde_train_dataset"]
        if train_dataset in routing_data:
            layers_choice_counts = generate_basic_plots(
                routing_data[train_dataset]["data"],
                routing_data[train_dataset]["logits"],
                routing_data[train_dataset]["weights"],
                routing_data[train_dataset]["choices"],
                dirs["plots_basic"],
                config["model_name"],
                config
            )
    
    # =========================================================================
    # STAGE 4: PER-EXPERT PLOTS
    # =========================================================================
    if config["run_per_expert_plots"]:
        train_dataset = config["kde_train_dataset"]
        if train_dataset in routing_data:
            generate_per_expert_plots(
                routing_data[train_dataset]["logits"],
                dirs["plots_per_expert"],
                config["model_name"],
                config
            )
    
    # =========================================================================
    # STAGE 5: KDE PLOTS
    # =========================================================================
    if config["run_kde_plots"]:
        train_dataset = config["kde_train_dataset"]
        if train_dataset in routing_data:
            for test_dataset in config["kde_test_datasets"]:
                if test_dataset in routing_data:
                    generate_kde_plots(
                        routing_data[train_dataset]["logits"],
                        routing_data[test_dataset]["logits"],
                        dirs["kde_models"],
                        dirs["plots_kde"],
                        config["model_name"],
                        train_dataset,
                        test_dataset,
                        config
                    )
    
    # =========================================================================
    # STAGE 6: P-VALUE PLOTS
    # =========================================================================
    if config["run_pvalue_plots"]:
        train_dataset = config["kde_train_dataset"]
        if train_dataset in routing_data and layers_choice_counts:
            for test_dataset in config["kde_test_datasets"]:
                if test_dataset in routing_data:
                    generate_pvalue_plots(
                        routing_data[train_dataset]["logits"],
                        routing_data[test_dataset]["logits"],
                        layers_choice_counts,
                        dirs["kde_models"],
                        dirs["plots_pvalue"],
                        config["model_name"],
                        test_dataset,
                        config
                    )
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print(f"Output directory: {config['output_dir']}")
    print("="*70)


def main():
    """Main entry point."""
    config = CONFIG.copy()
    
    # Override for quick testing:
    # config["max_samples"] = {"lambada": 50, "hellaswag": 50, "wikitext": 30}
    # config["run_per_expert_plots"] = False
    
    run_pipeline(config)


if __name__ == "__main__":
    main()
