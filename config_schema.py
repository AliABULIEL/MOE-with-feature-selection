"""
Configuration Schema for Multi-Model HC Routing
===============================================

Pydantic models for validating configuration files used in HC routing experiments.
Supports DeepSeek-v2-Lite and Qwen3-30B-A3B with configurable estimators.
"""

from typing import Optional, Union, Literal, List, Dict, Any
from pydantic import BaseModel, Field, validator


class ModelConfig(BaseModel):
    """Model configuration."""
    name: Literal["deepseek", "qwen"] = Field(
        ..., description="Model architecture type"
    )
    model_id: str = Field(
        ..., description="HuggingFace model identifier"
    )
    num_experts: int = Field(..., description="Number of experts in MoE layers")
    default_top_k: int = Field(..., description="Default top-k routing")
    dtype: Literal["float32", "float16", "bfloat16"] = Field(
        default="bfloat16", description="Model dtype"
    )
    device_map: str = Field(default="auto", description="Device mapping strategy")


class EstimatorConfig(BaseModel):
    """Estimator configuration for p-value computation."""
    type: Literal["kde", "student_t", "student_t_skewed", "gmm", "lindsey_glm", "advanced"] = Field(
        ..., description="Type of estimator to use"
    )
    quantile_range: tuple[float, float] = Field(
        default=(0.05, 0.95), description="Quantile range for training"
    )
    max_samples: int = Field(
        default=50000, description="Max samples for KDE training"
    )
    model_dir: Optional[str] = Field(
        default=None, description="Directory with pre-trained estimator models (e.g., 'kde_models')"
    )
    
    # Estimator-specific parameters
    bins: Optional[int] = Field(default=71, description="Bins for Student-T estimators")
    n_components: Optional[int] = Field(default=3, description="Components for GMM estimator")


class HCRoutingConfig(BaseModel):
    """Higher Criticism routing configuration."""
    beta: Union[float, Literal["auto"]] = Field(
        default=0.5, description="Search fraction (0.0-1.0) or 'auto' for adaptive"
    )
    temperature: float = Field(default=1.0, description="Softmax temperature")
    min_k: int = Field(default=2, description="Minimum experts per token")
    max_k: int = Field(default=8, description="Maximum experts per token")
    patch_mode: Optional[int] = Field(
        default=None, description="Number of layers to patch (None = all layers)"
    )
    
    @validator("beta")
    def validate_beta(cls, v):
        if v == "auto":
            return v
        if isinstance(v, (int, float)) and not (0 < v <= 1.0):
            raise ValueError("beta must be 'auto' or in range (0, 1]")
        return v
    
    @validator("min_k", "max_k")
    def validate_k(cls, v):
        if v < 1:
            raise ValueError("min_k and max_k must be >= 1")
        return v


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    datasets: List[Literal["wikitext", "lambada", "hellaswag", "tinystories", "fineweb-edu"]] = Field(
        default=["wikitext"], description="Datasets to evaluate"
    )
    max_samples: int = Field(default=200, description="Max samples per dataset")
    max_length: int = Field(default=512, description="Max sequence length")
    batch_size: int = Field(default=1, description="Batch size")
    log_every_n: int = Field(default=100, description="Logging frequency")


class OutputConfig(BaseModel):
    """Output configuration."""
    results_dir: str = Field(
        default="./results", description="Results output directory"
    )
    save_plots: bool = Field(default=True, description="Save visualization plots")
    save_logs: bool = Field(default=True, description="Save detailed logs")
    plot_format: Literal["png", "pdf", "svg"] = Field(default="png")
    plot_dpi: int = Field(default=100)


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""
    model: ModelConfig
    estimator: EstimatorConfig
    hc_routing: HCRoutingConfig
    evaluation: EvaluationConfig
    output: OutputConfig
    
    # Hardware configuration
    device: str = Field(default="cuda", description="Device to use (cuda/cpu)")
    
    @validator("device")
    def validate_device(cls, v):
        import torch
        if v == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return v


# Example configuration for validation
EXAMPLE_CONFIG = {
    "model": {
        "name": "deepseek",
        "model_id": "deepseek-ai/DeepSeek-V2-Lite",
        "num_experts": 64,
        "default_top_k": 6,
        "dtype": "bfloat16"
    },
    "estimator": {
        "type": "kde",
        "quantile_range": [0.05, 0.95],
        "max_samples": 50000,
        "model_dir": "kde_models"
    },
    "hc_routing": {
        "beta": 0.5,
        "temperature": 1.0,
        "min_k": 2,
        "max_k": 8,
        "patch_mode": None
    },
    "evaluation": {
        "datasets": ["wikitext", "lambada"],
        "max_samples": 200,
        "max_length": 512,
        "batch_size": 1,
        "log_every_n": 100
    },
    "output": {
        "results_dir": "./results/deepseek_kde",
        "save_plots": True,
        "save_logs": True
    },
    "device": "cuda"
}


if __name__ == "__main__":
    # Validate example configuration
    config = ExperimentConfig(**EXAMPLE_CONFIG)
    print("✅ Example configuration is valid!")
    print(f"Model: {config.model.model_id}")
    print(f"Estimator: {config.estimator.type}")
    print(f"HC beta: {config.hc_routing.beta}")
