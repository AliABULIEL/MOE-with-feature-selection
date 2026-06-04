"""
Qwen3-30B-A3B MoE Metrics Pipeline
====================================
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from internal_logging.moe_internal_logging_qwen import RouterLogger
from utils import calculate_text_metrics, load_dataset_samples


def load_config_from_file(config_path: str) -> Dict:
    CONFIG_FILE = os.path.join(Path(__file__).parent, "configs", config_path)
    if not os.path.exists(CONFIG_FILE) and CONFIG_FILE.endswith(".json"):
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")
    with open(CONFIG_FILE, "r") as f:
        CONFIG = json.load(f)

    CONFIG["device"] = (
        "cuda" if CONFIG["device"] == "cuda" and torch.cuda.is_available() else "cpu"
    )
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

    print(f"Loading model: {config['model_id']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        torch_dtype=get_torch_dtype(config["dtype"]),
        device_map="auto",
    )
    model.eval()
    print(f"✅ Model loaded successfully")
    return model, tokenizer


def run_evaluation(
    model,
    tokenizer,
    samples: List[Dict],
    dataset_name: str,
    config: Dict,
    output_dir: Path,
) -> Tuple[Dict, Path]:
    print(f"\n{'='*60}\nEvaluating on {dataset_name.upper()}\n{'='*60}")

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
            model, tokenizer, text, max_length=config.get("max_length")
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
        output_dir / f"{config['model_name']}_{dataset_name}_metrics_routing.parquet"
    )
    df.to_parquet(parquet_path)
    saved_paths = str(parquet_path)

    print(f"✅ Perplexity: {perplexity:.2f}, Saved to: {saved_paths}")
    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
    }, parquet_path


def run_pipeline(config: Dict):
    print("=" * 70)
    print(f"QWEN MOE METRICS PIPELINE")
    print(f"Model: {config['model_id']}")
    print("=" * 70)

    dirs = setup_directories(config)

    if config.get("run_evaluation", True):
        print("\n" + "=" * 70 + "\nSTAGE 1: EVALUATION\n" + "=" * 70)
        model, tokenizer = load_model(config)
        for ds_name in config["datasets"]:
            samples = load_dataset_samples(ds_name, config["max_samples"][ds_name])
            run_evaluation(model, tokenizer, samples, ds_name, config, dirs["logs"])
        del model, tokenizer
        torch.cuda.empty_cache()

    print(
        "\n"
        + "=" * 70
        + f"\nPIPELINE COMPLETE\nOutput: {config['output_dir']}\n"
        + "=" * 70
    )


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_qwen_metrics_pipeline.py <config_file.json>")
        sys.exit(1)
    config = load_config_from_file(sys.argv[1])
    run_pipeline(config)


if __name__ == "__main__":
    main()
