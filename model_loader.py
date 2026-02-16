"""
Model Loader for Multi-Model HC Routing
========================================

Factory functions for loading DeepSeek-v2-Lite and Qwen3-30B-A3B models.
Follows the exact loading patterns from pipelines/run_{model}_pipeline.py.
"""

from typing import Tuple, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def load_deepseek_model(
    model_id: str = "deepseek-ai/DeepSeek-V2-Lite",
    dtype: str = "bfloat16",
    device_map: str = "auto"
) -> Tuple[Any, Any]:
    """
    Load DeepSeek-V2-Lite model and tokenizer.
    
    Follows exact loading pattern from pipelines/run_deepseek_pipeline.py.
    
    Args:
        model_id: HuggingFace model identifier
        dtype: Model dtype ("float32", "float16", "bfloat16")
        device_map: Device mapping strategy
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading DeepSeek model: {model_id}")
    print(f"Dtype: {dtype}, Device map: {device_map}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        # trust_remote_code=True  # Uncomment if needed
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=get_torch_dtype(dtype),
        device_map=device_map,
        # trust_remote_code=True  # Uncomment if needed
    )
    model.eval()
    
    print(f"✅ DeepSeek model loaded successfully")
    return model, tokenizer


def load_qwen_model(
    model_id: str = "Qwen/Qwen3-30B-A3B",
    dtype: str = "bfloat16",
    device_map: str = "auto"
) -> Tuple[Any, Any]:
    """
    Load Qwen3-30B-A3B model and tokenizer.
    
    Follows exact loading pattern from pipelines/run_qwen_pipeline.py.
    
    Args:
        model_id: HuggingFace model identifier
        dtype: Model dtype ("float32", "float16", "bfloat16")
        device_map: Device mapping strategy
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading Qwen model: {model_id}")
    print(f"Dtype: {dtype}, Device map: {device_map}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        # trust_remote_code=True  # Uncomment if needed
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=get_torch_dtype(dtype),
        device_map=device_map,
        # trust_remote_code=True  # Uncomment if needed
    )
    model.eval()
    
    print(f"✅ Qwen model loaded successfully")
    return model, tokenizer


def load_model_from_config(config) -> Tuple[Any, Any]:
    """
    Load model and tokenizer based on configuration.
    
    Args:
        config: ExperimentConfig or ModelConfig object
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Handle both ExperimentConfig and ModelConfig
    if hasattr(config, 'model'):
        model_config = config.model
    else:
        model_config = config
    
    model_loaders = {
        "deepseek": load_deepseek_model,
        "qwen": load_qwen_model,
    }
    
    if model_config.name not in model_loaders:
        raise ValueError(
            f"Unknown model name: {model_config.name}. "
            f"Supported: {list(model_loaders.keys())}"
        )
    
    loader = model_loaders[model_config.name]
    return loader(
        model_id=model_config.model_id,
        dtype=model_config.dtype,
        device_map=model_config.device_map
    )


# Model metadata for easier reference
MODEL_METADATA = {
    "deepseek": {
        "default_model_id": "deepseek-ai/DeepSeek-V2-Lite",
        "num_experts": 64,
        "default_top_k": 6,
        "architecture": "deepseek",
    },
    "qwen": {
        "default_model_id": "Qwen/Qwen3-30B-A3B",
        "num_experts": 128,
        "default_top_k": 8,
        "architecture": "qwen",
    },
}


if __name__ == "__main__":
    import sys
    
    # Test loading
    model_name = sys.argv[1] if len(sys.argv) > 1 else "deepseek"
    
    print(f"Testing {model_name} model loading...")
    
    if model_name == "deepseek":
        model, tokenizer = load_deepseek_model()
    elif model_name == "qwen":
        model, tokenizer = load_qwen_model()
    else:
        print(f"Unknown model: {model_name}")
        sys.exit(1)
    
    print(f"\nModel config:")
    print(f"  Model type: {model.config.model_type if hasattr(model.config, 'model_type') else 'N/A'}")
    print(f"  Vocab size: {model.config.vocab_size}")
    print(f"  Tokenizer vocab size: {len(tokenizer)}")
    print("\n✅ Model loading test passed!")
