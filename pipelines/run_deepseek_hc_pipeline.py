"""
DeepSeek-V2-Lite HC Pipeline
=========================
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from moe_internal_logging_deepseek import RouterLogger
from pipelines.utils import calculate_text_metrics, load_dataset_samples
from pipelines.hc_models.utils import load_hc_estimator
from pipelines.hc_models.patching import patch_model_with_hc

def load_config_from_file(config_path: str) -> Dict:
    CONFIG_FILE = os.path.join(Path(__file__).parent, "configs", config_path)
    if not os.path.exists(CONFIG_FILE) and CONFIG_FILE.endswith(".json"):
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")
    with open(CONFIG_FILE, "r") as f:
        CONFIG = json.load(f)

    CONFIG["device"] = "cuda" if CONFIG.get("device") == "cuda" and torch.cuda.is_available() else "cpu"
    return CONFIG

def setup_directories(config: Dict) -> Dict[str, Path]:
    base_dir = Path(config["output_dir"])
    dirs = {
        "base": base_dir,
        "logs": base_dir / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs

def get_torch_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)

def load_model(config: Dict) -> Tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading model: {config['model_id']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        torch_dtype=get_torch_dtype(config["dtype"]),
        device_map="auto",
    )
    

    patch_model_with_hc(model, config['model_name'])
    
    # Configure HC routing on the model
    model.config.hc_routing_enabled = config.get("hc_routing_enabled", True)
    model.config.hc_layer = config.get("hc_layer", model.config.num_hidden_layers - 1)
    model.config.hc_threshold = config.get("hc_threshold", 4.0)
    
    model.eval()
    print(f"\n✅ Model loaded and patched successfully")
    return model, tokenizer



def run_evaluation(
    model,
    tokenizer,
    samples: List[Dict],
    dataset_name: str,
    config: Dict,
    output_dir: Path,
    estimator_name: str,
) -> Tuple[Dict, Path]:
    print(f"\n\n{'='*60}\nEvaluating on {dataset_name.upper()}\n{'='*60}")

    router_logger = RouterLogger(model)
    router_logger.register_hooks(top_k=config["default_top_k"])

    all_samples_data = []
    total_loss, total_tokens = 0.0, 0

    for i, sample in enumerate(tqdm(samples, desc=f"Evaluating {dataset_name}")):
        text = sample.get("text", "")
        if not text.strip():
            continue

        router_logger.clear_data()

        total_nll, per_token_loss, num_tokens = calculate_text_metrics(
            model, tokenizer, text
        )

        if num_tokens < 2:
            continue

        routing_data = router_logger.get_routing_data()
        loss = total_nll / num_tokens if num_tokens > 0 else float("inf")

        sample_data = {
            "sample_id": i,
            "num_tokens": num_tokens,
            "loss": loss,
            "per_token_loss": per_token_loss,
            "layers": [],
        }

        for layer_data in routing_data:
            sample_data["layers"].append(
                {
                    "layer": layer_data["layer"],
                    "router_logits_shape": list(layer_data["router_logits"].shape),
                    "selected_experts": layer_data["expert_indices"].numpy().tolist(),
                    "expert_weights": layer_data["expert_weights"].numpy().tolist(),
                    "router_logits_sample": layer_data["router_logits"]
                    .float()
                    .numpy()
                    .tolist(),
                }
            )

        all_samples_data.append(sample_data)
        total_loss += total_nll
        total_tokens += num_tokens

    router_logger.remove_hooks()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = float(np.exp(avg_loss))

    import pandas as pd

    df = pd.DataFrame(all_samples_data)
    parquet_path = (
        output_dir / f"{config['model_name']}_{dataset_name}_{estimator_name}_hc_metrics_routing.parquet"
    )
    df.to_parquet(parquet_path)
    saved_paths = str(parquet_path)

    print(f"\n✅ Perplexity: {perplexity:.2f}, Saved to: {saved_paths}")
    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
        "saved_path": saved_paths,
    }, parquet_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run HC Pipeline")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config file"
    )
    args = parser.parse_args()

    config = load_config_from_file(args.config)
    dirs = setup_directories(config)

    from pipelines.hc_models.patching import set_hc_estimator
    model, tokenizer = load_model(config)

    results = {}
    for dataset_name in config["datasets"]:
        samples = load_dataset_samples(dataset_name, config)
        if not samples:
            continue

        estimator_base_dir = Path(config["hc_estimators_base_dir"]) / dataset_name / f"layer_{config['hc_layer']}"
        if not estimator_base_dir.exists():
            continue
            
        estimators_to_run = []
        for p in estimator_base_dir.iterdir():
            if p.name == ".DS_Store": continue
            if p.is_file() and p.name.endswith(".pkl"):
                estimators_to_run.append((p.name.replace(".pkl", ""), str(p)))
            elif p.is_dir():
                estimators_to_run.append((f"per_expert_{p.name}", str(p)))

        dataset_results = {}
        for est_name, est_path in estimators_to_run:
            print(f"\n--- Testing Estimator: {est_name} ---")
            estimator = load_hc_estimator(est_path)
            set_hc_estimator(model, config['model_name'], config['hc_layer'], estimator)
            
            metrics, _ = run_evaluation(
                model, tokenizer, samples, dataset_name, config, dirs["base"], est_name
            )
            dataset_results[est_name] = metrics
            
        results[dataset_name] = dataset_results

    results_file = dirs["base"] / f"{config['model_name']}_hc_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ All results saved to {results_file}")

if __name__ == "__main__":
    main()
