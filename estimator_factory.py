"""
Estimator Factory for Multi-Model HC Routing
============================================

Factory pattern for creating and loading estimators from configuration.
Supports all estimator types from the estimators/ directory.
"""

import os
import pickle
from typing import Dict, Optional, Any
from pathlib import Path
import numpy as np

from estimators import (
    BaseEstimator,
    KDEEstimator,
    StudentTEstimator,
    StudentTSkewedEstimator,
    GMMNullEstimator,
    LindseyGLMEstimator,
    AdvancedEstimator,
)


def create_estimator(estimator_config) -> BaseEstimator:
    """
    Create an estimator instance based on configuration.
    
    Args:
        estimator_config: EstimatorConfig object
        
    Returns:
        Initialized estimator instance
    """
    estimator_type = estimator_config.type
    quantile_range = tuple(estimator_config.quantile_range)
    max_samples = estimator_config.max_samples
    
    estimator_map = {
        "kde": lambda: KDEEstimator(
            quantile_range=quantile_range,
            max_samples=max_samples
        ),
        "student_t": lambda: StudentTEstimator(
            bins=estimator_config.bins or 71,
            quantile_range=quantile_range
        ),
        "student_t_skewed": lambda: StudentTSkewedEstimator(
            bins=estimator_config.bins or 71,
            quantile_range=quantile_range
        ),
        "gmm": lambda: GMMNullEstimator(
            n_components=estimator_config.n_components or 3,
            quantile_range=quantile_range
        ),
        "lindsey_glm": lambda: LindseyGLMEstimator(
            quantile_range=quantile_range
        ),
        "advanced": lambda: AdvancedEstimator(
            quantile_range=quantile_range
        ),
    }
    
    if estimator_type not in estimator_map:
        raise ValueError(
            f"Unknown estimator type: {estimator_type}. "
            f"Supported: {list(estimator_map.keys())}"
        )
    
    return estimator_map[estimator_type]()


def load_estimator_models(
    model_dir: Optional[str] = None,
    model_name: str = "deepseek",
    num_layers: Optional[int] = None
) -> Optional[Dict[int, Dict]]:
    """
    Load pre-trained estimator models from directory.
    
    This generalizes the `load_kde_models()` function from hc_routing.py
    to support different estimator types.
    
    Args:
        model_dir: Directory containing estimator model files
        model_name: Model name prefix (e.g., "deepseek", "qwen")
        num_layers: Expected number of layers (optional, for validation)
        
    Returns:
        Dictionary mapping layer_idx to model data, or None if not found
    """
    if model_dir is None:
        model_dir = "kde_models"  # Default fallback
    
    model_dir_path = Path(model_dir)
    if not model_dir_path.exists():
        print(f"⚠️  Estimator model directory not found: {model_dir}")
        return None
    
    # Look for model files matching pattern: {model_name}_distribution_model_layer_{layer_idx}.pkl
    estimator_models = {}
    
    pattern = f"{model_name}_distribution_model_layer_"
    for file_path in model_dir_path.glob(f"{pattern}*.pkl"):
        # Extract layer index from filename
        filename = file_path.stem
        try:
            layer_idx_str = filename.split(pattern)[1].split("_")[0]
            layer_idx = int(layer_idx_str)
            
            # Load the model data
            with open(file_path, "rb") as f:
                model_data = pickle.load(f)
            
            estimator_models[layer_idx] = model_data
            
        except (IndexError, ValueError) as e:
            print(f"⚠️  Could not parse layer index from {filename}: {e}")
            continue
    
    if not estimator_models:
        print(f"⚠️  No estimator models found in {model_dir} for {model_name}")
        return None
    
    num_loaded = len(estimator_models)
    max_layer = max(estimator_models.keys())
    min_layer = min(estimator_models.keys())
    
    print(f"✅ Loaded {num_loaded} estimator models for {model_name}")
    print(f"   Layer range: [{min_layer}, {max_layer}]")
    
    if num_layers is not None and num_loaded != num_layers:
        print(f"⚠️  Warning: Expected {num_layers} layers, found {num_loaded}")
    
    return estimator_models


def compute_pvalues_with_estimator(
    logits: Any,  # torch.Tensor
    layer_idx: int,
    estimators: Optional[Dict[int, BaseEstimator]] = None
) -> Any:  # torch.Tensor
    """
    Compute p-values using fitted estimator objects.
    
    Uses the estimator's .cdf() method directly for cleaner, more general implementation.
    
    Args:
        logits: Router logits [num_tokens, num_experts]
        layer_idx: Layer index
        estimators: Dict mapping layer_idx to fitted estimator instances
        
    Returns:
        P-values [num_tokens, num_experts]
    """
    import torch
    
    if estimators is None or layer_idx not in estimators:
        # Fallback to empirical p-values
        return compute_pvalues_empirical(logits)
    
    estimator = estimators[layer_idx]
    
    # Convert to numpy for estimator computation
    logits_np = logits.cpu().numpy().flatten()
    
    # Use estimator's cdf() method
    cdf_values = estimator.cdf(logits_np)
    
    # P-values = 1 - CDF
    p_values = 1.0 - cdf_values
    
    # Reshape back to original shape
    p_values = p_values.reshape(logits.shape)
    
    # Convert back to torch tensor
    p_values_tensor = torch.from_numpy(p_values).to(logits.device).float()
    
    return p_values_tensor


def compute_pvalues_empirical(logits: Any) -> Any:
    """
    Compute empirical p-values from logits (fallback method).
    
    Args:
        logits: Router logits [num_tokens, num_experts]
        
    Returns:
        P-values [num_tokens, num_experts]
    """
    import torch
    import torch.nn.functional as F
    
    # Use softmax as proxy for probabilities
    probs = F.softmax(logits, dim=-1)
    
    # P-value = 1 - probability
    p_values = 1.0 - probs
    
    return p_values


if __name__ == "__main__":
    import sys
    from config_schema import EstimatorConfig
    
    # Test estimator creation
    print("Testing estimator factory...")
    
    estimator_config = EstimatorConfig(
        type="kde",
        quantile_range=(0.05, 0.95),
        max_samples=50000
    )
    
    estimator = create_estimator(estimator_config)
    print(f"✅ Created {estimator.name} estimator")
    
    # Test model loading
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
        models = load_estimator_models(model_dir=model_dir, model_name="deepseek")
        if models:
            print(f"✅ Loaded estimator models from {model_dir}")
    else:
        print("\nTo test model loading, provide model directory:")
        print("  python estimator_factory.py kde_models")
