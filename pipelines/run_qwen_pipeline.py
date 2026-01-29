"""
Qwen3-30B-A3B MoE Analysis Pipeline
====================================

Complete end-to-end pipeline for Qwen3-30B-A3B MoE router analysis.
Combines internal routing logging + KDE training + visualization.

This script uses the existing RouterLogger from moe_internal_logging_qwen.py

Usage:
    python run_qwen_pipeline.py

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
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

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

# Import the existing RouterLogger from moe_internal_logging_qwen.py
from moe_internal_logging_qwen import RouterLogger, InternalRoutingLogger

# ==============================================================================
# CONFIGURABLE PARAMETERS
# ==============================================================================

"""
CONFIG definition example (save as JSON file in configs/ directory):
{
    "model_id": "Qwen/Qwen3-30B-A3B",
    "model_name": "qwen",
    "num_experts": 128,
    "default_top_k": 8,
    
    "datasets": ["lambada", "hellaswag", "wikitext"],
    "max_samples": {
        "lambada": 500,
        "hellaswag": 500,
        "wikitext": 300
    },
    "max_length": 512,
    
    "kde_train_dataset": "lambada",
    "kde_test_datasets": ["hellaswag", "wikitext"],
    "kde_trim_amounts": [0, 1, 2, 4],
    "kde_kernels": ["gaussian", "tophat", "epanechnikov", "linear"],
    
    "output_dir": "./outputs/qwen",
    
    "run_evaluation": true,
    "run_kde_training": true,
    "run_basic_plots": true,
    "run_per_expert_plots": true,
    "run_kde_plots": true,
    "run_pvalue_plots": true,
    
    "device": "cuda",
    "dtype": "bfloat16",
    
    "plot_format": "png",
    "plot_dpi": 100,
    "show_plots": false
}
"""


def load_config_from_file(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    CONFIG_FILE = os.path.join(Path(__file__).parent, "configs", config_path)
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")
    with open(CONFIG_FILE, "r") as f:
        CONFIG = json.load(f)

    CONFIG["device"] = (
        "cuda" if CONFIG["device"] == "cuda" and torch.cuda.is_available() else "cpu"
    )
    return CONFIG


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
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    return dtype_map.get(dtype_str, torch.bfloat16)


def save_plot(fig, path: Path, config: Dict, close: bool = True):
    fig.savefig(path, format=config["plot_format"], dpi=config["plot_dpi"], bbox_inches='tight')
    if config["show_plots"]:
        plt.show()
    if close:
        plt.close(fig)


def get_ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"


# ==============================================================================
# STAGE 1: MODEL & DATASET LOADING
# ==============================================================================

def load_model(config: Dict) -> Tuple[Any, Any]:
    """Load Qwen3-30B-A3B model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {config['model_id']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], trust_remote_code=True)
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


def load_dataset_samples(dataset_name: str, max_samples: int) -> List[Dict]:
    """Load dataset samples."""
    from datasets import load_dataset

    print(f"Loading {dataset_name} dataset (max {max_samples} samples)...")

    if dataset_name == "lambada":
        try:
            dataset = load_dataset("lambada", split="test")
        except:
            dataset = load_dataset("EleutherAI/lambada_openai", "en", split="test")
        samples = [{"text": item.get('text', '')} for item in dataset if item.get('text', '').strip()]
    elif dataset_name == "hellaswag":
        dataset = load_dataset("hellaswag", split="validation")
        samples = []
        for item in dataset:
            ctx, endings, label = item.get('ctx', ''), item.get('endings', []), int(item.get('label', 0))
            if ctx and endings and 0 <= label < len(endings):
                samples.append({"text": f"{ctx} {endings[label]}"})
    elif dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        samples = [{"text": item.get('text', '')} for item in dataset if item.get('text', '').strip() and len(item.get('text', '').split()) > 10]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    samples = samples[:max_samples]
    print(f"✅ Loaded {len(samples)} samples from {dataset_name}")
    return samples


# ==============================================================================
# STAGE 1C: EVALUATION WITH LOGGING (USES YOUR RouterLogger)
# ==============================================================================

def run_evaluation(model, tokenizer, samples: List[Dict], dataset_name: str, config: Dict, output_dir: Path) -> Tuple[Dict, Path]:
    """Run evaluation using RouterLogger from moe_internal_logging_qwen.py."""

    print(f"\n{'='*60}\nEvaluating on {dataset_name.upper()}\n{'='*60}")

    # Use the imported RouterLogger from moe_internal_logging_qwen.py
    router_logger = RouterLogger(model)
    router_logger.register_hooks(top_k=config["default_top_k"])

    all_samples_data = []
    total_loss, total_tokens = 0.0, 0

    with torch.no_grad():
        for i, sample in enumerate(tqdm(samples, desc=f"Evaluating {dataset_name}")):
            text = sample.get("text", "")
            if not text.strip():
                continue

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=config["max_length"], padding=False)
            input_ids = inputs["input_ids"].to(config["device"])

            if input_ids.shape[1] < 2:
                continue

            router_logger.clear_data()
            outputs = model(input_ids, labels=input_ids)
            loss, num_tokens = outputs.loss.item(), input_ids.shape[1]

            routing_data = router_logger.get_routing_data()
            sample_data = {"sample_id": i, "num_tokens": num_tokens, "loss": loss, "layers": []}

            for layer_data in routing_data:
                sample_data["layers"].append({
                    "layer": layer_data["layer"],
                    "router_logits_shape": list(layer_data["router_logits"].shape),
                    "selected_experts": layer_data["expert_indices"].numpy().tolist(),
                    "expert_weights": layer_data["expert_weights"].numpy().tolist(),
                    "router_logits_sample": layer_data["router_logits"].numpy().tolist(),
                })

            all_samples_data.append(sample_data)
            total_loss += loss * num_tokens
            total_tokens += num_tokens

    router_logger.remove_hooks()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = float(np.exp(avg_loss))

    output_data = {
        "config": config["model_id"], "strategy": "topk_baseline", "num_experts": config["num_experts"],
        "top_k": config["default_top_k"], "dataset": dataset_name, "timestamp": datetime.now().isoformat(),
        "num_layers": len(all_samples_data[0]["layers"]) if all_samples_data else 0, "samples": all_samples_data
    }

    json_path = output_dir / f"{config['model_name']}_{dataset_name}_internal_routing.json"
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ Perplexity: {perplexity:.2f}, Saved to: {json_path}")
    return {"perplexity": perplexity, "avg_loss": avg_loss, "total_tokens": total_tokens}, json_path


# ==============================================================================
# STAGE 2: KDE MODEL TRAINING
# ==============================================================================

def load_routing_data(json_path: Path) -> Dict:
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_router_logits(data: Dict) -> Tuple[np.ndarray, List, List]:
    num_layers = data["num_layers"]
    layers_logits = [[] for _ in range(num_layers)]
    layers_weights = [[] for _ in range(num_layers)]
    layers_choices = [[] for _ in range(num_layers)]

    for sample in data["samples"]:
        for layer_data in sample["layers"]:
            idx = layer_data["layer"]
            layers_logits[idx].extend(layer_data["router_logits_sample"])
            layers_weights[idx].extend(layer_data["expert_weights"])
            layers_choices[idx].extend(layer_data["selected_experts"])

    return np.array([np.array(arr) for arr in layers_logits]), layers_weights, layers_choices


def trim_top_and_bottom_experts(arr: np.ndarray, trim_amount: int) -> np.ndarray:
    if trim_amount == 0:
        return arr
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    sorted_indices = np.argsort(arr, axis=-1)
    kept_indices = sorted_indices[:, :, trim_amount:-trim_amount]
    return np.take_along_axis(arr, kept_indices, axis=-1).squeeze()


def train_kde_models(layers_logits: np.ndarray, output_dir: Path, model_name: str, config: Dict):
    print(f"\n{'='*60}\nTRAINING KDE MODELS\n{'='*60}")

    for layer_idx in tqdm(range(layers_logits.shape[0]), desc="Training KDE"):
        layer_data = layers_logits[layer_idx].flatten()
        kde = gaussian_kde(layer_data)

        data_min, data_max = layer_data.min(), layer_data.max()
        x_grid = np.linspace(data_min - 0.2*abs(data_min), data_max + 0.2*abs(data_max), 10000)

        pdf_grid = kde.evaluate(x_grid)
        cdf_grid = np.cumsum(pdf_grid)
        cdf_grid /= cdf_grid[-1]

        with open(output_dir / f"{model_name}_distribution_model_layer_{layer_idx}.pkl", 'wb') as f:
            pickle.dump({"x": x_grid, "cdf": cdf_grid}, f)

        # Trimmed variants
        for trim_amt in config["kde_trim_amounts"]:
            if trim_amt == 0:
                continue
            trimmed = trim_top_and_bottom_experts(layers_logits[layer_idx][np.newaxis, :, :], trim_amt).flatten()
            kde_t = gaussian_kde(trimmed)
            x_t = np.linspace(trimmed.min() - 0.2*abs(trimmed.min()), trimmed.max() + 0.2*abs(trimmed.max()), 10000)
            pdf_t = kde_t.evaluate(x_t)
            cdf_t = np.cumsum(pdf_t)
            cdf_t /= cdf_t[-1]
            with open(output_dir / f"{model_name}_distribution_model_layer_{layer_idx}_trimmed_{trim_amt*2}.pkl", 'wb') as f:
                pickle.dump({"x": x_t, "cdf": cdf_t}, f)

        # Different kernels
        for kernel in config["kde_kernels"]:
            bw = 1.06 * np.std(layer_data) * (len(layer_data) ** (-1/5))
            kde_sk = KernelDensity(kernel=kernel, bandwidth=bw)
            kde_sk.fit(layer_data[:, np.newaxis])
            pdf_k = np.exp(kde_sk.score_samples(x_grid[:, np.newaxis]))
            cdf_k = np.cumsum(pdf_k)
            cdf_k /= cdf_k[-1]
            with open(output_dir / f"{model_name}_distribution_model_layer_{layer_idx}_{kernel}.pkl", 'wb') as f:
                pickle.dump({"x": x_grid, "cdf": cdf_k}, f)

    print(f"✅ Trained KDE models")


# ==============================================================================
# STAGE 3-6: PLOTTING FUNCTIONS
# ==============================================================================

def generate_basic_plots(data, layers_logits, layers_weights, layers_choices, output_dir, model_name, config):
    print(f"\n{'='*60}\nGENERATING BASIC PLOTS\n{'='*60}")

    num_layers = layers_logits.shape[0]
    # Note: layers_logits is an object array where each element is 2D (num_tokens, num_experts)
    num_experts = layers_logits[0].shape[1]
    layers_weights_np = [np.array(w) for w in layers_weights]

    # Weight sum distributions
    (output_dir / "expert_weights_sum").mkdir(exist_ok=True)
    layers_weights_sum = []
    for layer_idx in tqdm(range(num_layers), desc="Weight sum plots"):
        ws = np.sum(layers_weights_np[layer_idx], axis=1)
        layers_weights_sum.append(ws)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(ws, bins=50, kde=True, ax=ax)
        ax.set_xlim(0, 1)
        ax.set_title(f"Expert Weight Sum - Layer {layer_idx}")
        save_plot(fig, output_dir / "expert_weights_sum" / f"{model_name}_layer_{layer_idx}_weights_sum.png", config)

    # Choice counts
    (output_dir / "expert_choice_counts").mkdir(exist_ok=True)
    layers_choice_counts = []
    for layer_idx in tqdm(range(num_layers), desc="Choice count plots"):
        flat = [e for tc in layers_choices[layer_idx] for e in tc]
        unique, counts = np.unique(flat, return_counts=True)
        choice_dict = dict(zip(unique, counts))
        layers_choice_counts.append(choice_dict)
        fig, ax = plt.subplots(figsize=(24, 6))
        sns.barplot(x=list(range(num_experts)), y=[choice_dict.get(e, 0) for e in range(num_experts)], ax=ax)
        ax.set_xticks(range(0, num_experts, 8))
        ax.set_title(f"Expert Choice Counts - Layer {layer_idx}")
        save_plot(fig, output_dir / "expert_choice_counts" / f"{model_name}_layer_{layer_idx}_choice_counts.png", config)

    # Softmax & raw distributions
    for subdir in ["router_softmax", "router_raw_logits"]:
        (output_dir / subdir).mkdir(exist_ok=True)

    for layer_idx in tqdm(range(num_layers), desc="Distribution plots"):
        logits = layers_logits[layer_idx]
        softmax = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(softmax.flatten(), bins=100, kde=True, ax=ax)
        ax.set_title(f"Router Softmax - Layer {layer_idx}")
        save_plot(fig, output_dir / "router_softmax" / f"{model_name}_layer_{layer_idx}_softmax.png", config)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(logits.flatten(), bins=100, kde=True, ax=ax)
        ax.set_title(f"Raw Router Logits - Layer {layer_idx}")
        save_plot(fig, output_dir / "router_raw_logits" / f"{model_name}_layer_{layer_idx}_raw_logits.png", config)

    print(f"✅ Generated basic plots")
    return layers_choice_counts


def generate_per_expert_plots(layers_logits, output_dir, model_name, config):
    print(f"\n{'='*60}\nGENERATING PER-EXPERT PLOTS\n{'='*60}")

    num_layers = layers_logits.shape[0]
    num_experts = layers_logits[0].shape[1]
    (output_dir / "per_expert_softmax").mkdir(exist_ok=True)
    (output_dir / "per_expert_raw_logits").mkdir(exist_ok=True)

    for layer_idx in tqdm(range(num_layers), desc="Per-expert plots"):
        logits = layers_logits[layer_idx]
        softmax = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

        for expert_idx in range(num_experts):
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(softmax[:, expert_idx], bins=100, kde=True, ax=ax)
            ax.set_title(f"Softmax - Layer {layer_idx}, Expert {expert_idx}")
            save_plot(fig, output_dir / "per_expert_softmax" / f"{model_name}_layer_{layer_idx}_expert_{expert_idx}_softmax.png", config)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(logits[:, expert_idx], bins=100, kde=True, ax=ax)
            ax.set_title(f"Raw Logits - Layer {layer_idx}, Expert {expert_idx}")
            save_plot(fig, output_dir / "per_expert_raw_logits" / f"{model_name}_layer_{layer_idx}_expert_{expert_idx}_raw.png", config)

    print(f"✅ Generated per-expert plots")


def generate_kde_plots(train_logits, test_logits, kde_dir, output_dir, model_name, train_ds, test_ds, config):
    print(f"\n{'='*60}\nGENERATING KDE PLOTS ({train_ds} -> {test_ds})\n{'='*60}")

    num_layers = train_logits.shape[0]

    for subdir in ["kde_cumulative_density", "trimmed_comparison", "kernel_comparison"]:
        (output_dir / subdir).mkdir(exist_ok=True)

    for layer_idx in tqdm(range(num_layers), desc="KDE plots"):
        with open(kde_dir / f"{model_name}_distribution_model_layer_{layer_idx}.pkl", 'rb') as f:
            kde_data = pickle.load(f)

        probs = np.interp(test_logits[layer_idx].flatten(), kde_data['x'], kde_data['cdf'])

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(probs, bins=100, ax=ax)
        ax.set_xlim(0, 1)
        ax.set_title(f"KDE Cumulative Density - Layer {layer_idx} on {test_ds}")
        save_plot(fig, output_dir / "kde_cumulative_density" / f"{model_name}_kde_density_layer_{layer_idx}_{test_ds}.png", config)

        # Trimmed comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        for i, trim_amt in enumerate(config["kde_trim_amounts"]):
            ax = axes.flatten()[i]
            if trim_amt == 0:
                train_t, test_t = train_logits[layer_idx].flatten(), test_logits[layer_idx].flatten()
            else:
                train_t = trim_top_and_bottom_experts(train_logits[layer_idx][np.newaxis, :, :], trim_amt).flatten()
                test_t = trim_top_and_bottom_experts(test_logits[layer_idx][np.newaxis, :, :], trim_amt).flatten()
            sns.histplot(train_t, bins=100, kde=True, color='blue', label=train_ds, alpha=0.5, ax=ax)
            sns.histplot(test_t, bins=100, kde=True, color='orange', label=test_ds, alpha=0.5, ax=ax)
            ax.set_title(f"Layer {layer_idx} - Trim {trim_amt}")
            ax.legend()
        plt.tight_layout()
        save_plot(fig, output_dir / "trimmed_comparison" / f"{model_name}_trimmed_layer_{layer_idx}.png", config)

        # Kernel comparison
        fig, ax = plt.subplots(figsize=(12, 7))
        for kernel in config["kde_kernels"]:
            kpath = kde_dir / f"{model_name}_distribution_model_layer_{layer_idx}_{kernel}.pkl"
            if kpath.exists():
                with open(kpath, 'rb') as f:
                    kdata = pickle.load(f)
                kprobs = np.interp(test_logits[layer_idx].flatten(), kdata['x'], kdata['cdf'])
                sns.histplot(kprobs, bins=100, element="step", fill=False, label=kernel.capitalize(), ax=ax)
        ax.set_xlim(0, 1)
        ax.set_title(f"Kernel Comparison - Layer {layer_idx}")
        ax.legend()
        save_plot(fig, output_dir / "kernel_comparison" / f"{model_name}_kernel_layer_{layer_idx}.png", config)

    print(f"✅ Generated KDE plots")


def generate_pvalue_plots(train_logits, test_logits, layers_choice_counts, kde_dir, output_dir, model_name, test_ds, config):
    print(f"\n{'='*60}\nGENERATING P-VALUE PLOTS\n{'='*60}")

    num_layers = train_logits.shape[0]
    num_experts = train_logits[0].shape[1]
    rng = np.random.default_rng(seed=42)

    for subdir in ["pvalue_basic", "pvalue_vs_uniform", "pvalue_by_popularity"]:
        (output_dir / subdir).mkdir(exist_ok=True)

    for layer_idx in tqdm(range(num_layers), desc="P-value plots"):
        with open(kde_dir / f"{model_name}_distribution_model_layer_{layer_idx}.pkl", 'rb') as f:
            kde_data = pickle.load(f)

        test_flat = test_logits[layer_idx].flatten()
        probs = np.interp(test_flat, kde_data['x'], kde_data['cdf'])
        p_values = 1 - probs

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(p_values, bins=100, ax=ax)
        ax.set_xlim(0, 1)
        ax.set_title(f"P-Value Distribution - Layer {layer_idx} on {test_ds}")
        save_plot(fig, output_dir / "pvalue_basic" / f"{model_name}_pvalue_layer_{layer_idx}_{test_ds}.png", config)

        # P-value vs uniform (4 at a time)
        test_data = test_logits[layer_idx]
        uniform = rng.uniform(0, 1, size=test_data.shape)

        for j in range(0, num_experts, 4):
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            for k_off, ax in enumerate(axes.flatten()):
                k = j + k_off
                if k >= num_experts:
                    ax.set_visible(False)
                    continue
                kth_logits = np.sort(test_data, axis=1)[:, -(k+1)]
                kth_probs = np.interp(kth_logits, kde_data['x'], kde_data['cdf'])
                kth_pval = 1 - kth_probs
                kth_unif = np.sort(uniform, axis=1)[:, k]

                bins = np.linspace(0, 1, 101)
                sns.histplot(kth_pval, bins=bins, color='blue', label='KDE P-Value', alpha=0.7, ax=ax)
                sns.histplot(kth_unif, bins=bins, color='orange', label='Uniform', alpha=0.5, ax=ax)
                ax.set_title(f"{get_ordinal(k+1)} Best Expert - Layer {layer_idx}")
                ax.set_xlim(0, 1)
                ax.legend()
            plt.tight_layout()
            save_plot(fig, output_dir / "pvalue_vs_uniform" / f"{model_name}_pvalue_uniform_layer_{layer_idx}_exp_{j+1}_to_{min(j+4, num_experts)}.png", config)

        # P-value by popularity
        choice_dict = layers_choice_counts[layer_idx]
        sorted_experts = sorted(range(num_experts), key=lambda e: choice_dict.get(e, 0), reverse=True)

        for j in range(0, num_experts, 4):
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            for i, ax in enumerate(axes.flatten()):
                rank = j + i
                if rank >= num_experts:
                    ax.set_visible(False)
                    continue
                expert_idx = sorted_experts[rank]
                exp_logits = test_data[:, expert_idx]
                exp_probs = np.interp(exp_logits, kde_data['x'], kde_data['cdf'])
                exp_pval = 1 - exp_probs

                edges = np.linspace(0, 1, 51)
                sns.histplot(exp_pval, bins=50, color='blue', label='KDE P-Value', ax=ax)
                ax.set_title(f"{rank+1}-th Popular Expert ({expert_idx}) - Layer {layer_idx}")
                ax.set_xlim(0, 1)
            plt.tight_layout()
            save_plot(fig, output_dir / "pvalue_by_popularity" / f"{model_name}_pvalue_pop_layer_{layer_idx}_rank_{j+1}_to_{min(j+4, num_experts)}.png", config)

    print(f"✅ Generated p-value plots")


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def run_pipeline(config: Dict):
    print("="*70)
    print(f"QWEN3-30B-A3B MOE ANALYSIS PIPELINE")
    print(f"Model: {config['model_id']}")
    print("="*70)

    dirs = setup_directories(config)
    routing_data = {}

    # STAGE 1: EVALUATION
    if config["run_evaluation"]:
        print("\n" + "="*70 + "\nSTAGE 1: EVALUATION (using RouterLogger from moe_internal_logging_qwen.py)\n" + "="*70)
        model, tokenizer = load_model(config)
        for ds_name in config["datasets"]:
            samples = load_dataset_samples(ds_name, config["max_samples"][ds_name])
            run_evaluation(model, tokenizer, samples, ds_name, config, dirs["logs"])
        del model, tokenizer
        torch.cuda.empty_cache()

    # LOAD DATA
    print("\n" + "="*70 + "\nLOADING ROUTING DATA\n" + "="*70)
    for ds_name in config["datasets"]:
        json_path = dirs["logs"] / f"{config['model_name']}_{ds_name}_internal_routing.json"
        if json_path.exists():
            data = load_routing_data(json_path)
            logits, weights, choices = extract_router_logits(data)
            routing_data[ds_name] = {"data": data, "logits": logits, "weights": weights, "choices": choices}
            print(f"✅ {ds_name}: {logits.shape}")

    if not routing_data:
        print("❌ No data. Run evaluation first.")
        return

    train_ds = config["kde_train_dataset"]

    # STAGE 2: KDE
    if config["run_kde_training"] and train_ds in routing_data:
        train_kde_models(routing_data[train_ds]["logits"], dirs["kde_models"], config["model_name"], config)

    # STAGE 3: BASIC PLOTS
    layers_choice_counts = None
    if config["run_basic_plots"] and train_ds in routing_data:
        layers_choice_counts = generate_basic_plots(
            routing_data[train_ds]["data"], routing_data[train_ds]["logits"],
            routing_data[train_ds]["weights"], routing_data[train_ds]["choices"],
            dirs["plots_basic"], config["model_name"], config
        )

    # STAGE 4: PER-EXPERT PLOTS
    if config["run_per_expert_plots"] and train_ds in routing_data:
        generate_per_expert_plots(routing_data[train_ds]["logits"], dirs["plots_per_expert"], config["model_name"], config)

    # STAGE 5: KDE PLOTS
    if config["run_kde_plots"] and train_ds in routing_data:
        for test_ds in config["kde_test_datasets"]:
            if test_ds in routing_data:
                generate_kde_plots(routing_data[train_ds]["logits"], routing_data[test_ds]["logits"],
                                   dirs["kde_models"], dirs["plots_kde"], config["model_name"], train_ds, test_ds, config)

    # STAGE 6: P-VALUE PLOTS
    if config["run_pvalue_plots"] and train_ds in routing_data and layers_choice_counts:
        for test_ds in config["kde_test_datasets"]:
            if test_ds in routing_data:
                generate_pvalue_plots(routing_data[train_ds]["logits"], routing_data[test_ds]["logits"],
                                      layers_choice_counts, dirs["kde_models"], dirs["plots_pvalue"],
                                      config["model_name"], test_ds, config)

    print("\n" + "="*70 + f"\nPIPELINE COMPLETE\nOutput: {config['output_dir']}\n" + "="*70)


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python run_qwen_pipeline.py <config_file.json>")
        sys.exit(1)
    config = load_config_from_file(sys.argv[1])

    # Quick test override:
    # config["max_samples"] = {"lambada": 50, "hellaswag": 50, "wikitext": 30}
    # config["run_per_expert_plots"] = False

    run_pipeline(config)


if __name__ == "__main__":
    main()