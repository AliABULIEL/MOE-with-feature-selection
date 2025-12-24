"""
Benjamini-Hochberg Routing for Mixture-of-Experts Models
==========================================================

This module implements the Benjamini-Hochberg (BH) procedure for expert selection
in sparse MoE models, providing statistical control of the False Discovery Rate (FDR).

Key Features:
- KDE-based p-values from empirical router logit distributions
- Fully vectorized PyTorch implementation
- GPU-compatible (with CPU fallback for KDE interpolation)
- Configurable FDR level (alpha)
- Temperature-based calibration
- Min/max expert constraints
- Numerical stability guarantees

References:
    Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate:
    a practical and powerful approach to multiple testing. Journal of the Royal
    Statistical Society: Series B, 57(1), 289-300.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, TYPE_CHECKING
import warnings
import numpy as np
import os
import pickle

# Import BHRoutingLogger for type checking only (avoid circular imports)
if TYPE_CHECKING:
    from bh_routing_logging import BHRoutingLogger

# Global cache for KDE models
_kde_models_cache: Dict[int, Dict] = {}


def load_kde_models(kde_dir: str = None) -> Dict[int, Dict]:
    """
    Load pre-trained KDE models for each layer.

    KDE models contain empirical CDF of router logits, used to compute
    properly calibrated p-values: p = 1 - CDF(logit)

    Args:
        kde_dir: Directory containing KDE model files
                 Default: looks for kde_models/models/ relative to this file

    Returns:
        Dictionary mapping layer_idx -> {'x': x_grid, 'cdf': cdf_grid}
    """
    global _kde_models_cache

    if _kde_models_cache:
        return _kde_models_cache

    if kde_dir is None:
        # Try to find kde_models relative to this file or in common locations
        possible_dirs = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kde_models', 'models'),
            './kde_models/models',
            '../kde_models/models',
            '/content/drive/MyDrive/MOE-with-feature-selection/kde_models/models',
        ]
        for d in possible_dirs:
            if os.path.exists(d):
                kde_dir = d
                break

    if kde_dir is None or not os.path.exists(kde_dir):
        warnings.warn(
            f"KDE models directory not found. Will use empirical fallback method.",
            UserWarning
        )
        return {}

    # Load all layer models
    loaded_count = 0
    for layer_idx in range(16):  # OLMoE has 16 layers
        model_path = os.path.join(kde_dir, f"distribution_model_layer_{layer_idx}.pkl")
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                _kde_models_cache[layer_idx] = pickle.load(f)
                loaded_count += 1

    if loaded_count > 0:
        print(f"✅ Loaded {loaded_count} KDE models from {kde_dir}")
    else:
        warnings.warn(f"No KDE models found in {kde_dir}", UserWarning)

    return _kde_models_cache


def compute_pvalues_kde(
    router_logits: torch.Tensor,
    layer_idx: int,
    kde_models: Dict[int, Dict]
) -> torch.Tensor:
    """
    Compute p-values using pre-trained KDE model for this layer.

    P-value = 1 - CDF(logit)
    - Higher logit → higher CDF → lower p-value → more significant
    - Based on empirical distribution of router logits

    Args:
        router_logits: [num_tokens, num_experts] raw logits
        layer_idx: Which layer (0-15 for OLMoE)
        kde_models: Pre-loaded KDE models

    Returns:
        p_values: [num_tokens, num_experts] p-values in [0, 1]
    """
    device = router_logits.device
    dtype = router_logits.dtype

    if layer_idx not in kde_models or not kde_models[layer_idx]:
        # Fallback: use empirical CDF from current batch
        return compute_pvalues_empirical(router_logits)

    model = kde_models[layer_idx]
    x_grid = model['x']  # numpy array
    cdf_grid = model['cdf']  # numpy array

    # Convert to numpy for interpolation
    # Convert bfloat16 to float32 before numpy conversion (numpy doesn't support bfloat16)
    logits_np = router_logits.detach().cpu().float().numpy().flatten()

    # Interpolate to get CDF values
    # Uses linear interpolation: for logit values outside x_grid range,
    # extrapolates with boundary values (CDF=0 for low, CDF=1 for high)
    cdf_values = np.interp(logits_np, x_grid, cdf_grid)

    # P-value = 1 - CDF
    # High logit → high CDF (close to 1) → low p-value (close to 0) → significant
    p_values_np = 1.0 - cdf_values

    # Reshape and convert back to tensor
    p_values = torch.from_numpy(p_values_np).reshape(router_logits.shape)
    p_values = p_values.to(device=device, dtype=dtype)

    # Clamp to (0, 1) for numerical stability
    eps = torch.finfo(dtype).eps if dtype.is_floating_point else 1e-8
    p_values = torch.clamp(p_values, min=eps, max=1.0 - eps)

    return p_values


def compute_pvalues_empirical(router_logits: torch.Tensor) -> torch.Tensor:
    """
    Fallback: Compute p-values using empirical CDF from current batch.

    Two modes based on number of tokens:
    1. Multi-token (num_tokens > 1): For each expert, compute CDF across tokens
    2. Single-token (num_tokens == 1): Compute CDF across experts for that token

    This is less accurate than KDE but works without pre-trained models.

    Args:
        router_logits: [num_tokens, num_experts]

    Returns:
        p_values: [num_tokens, num_experts] empirical p-values
    """
    device = router_logits.device
    dtype = router_logits.dtype
    num_tokens, num_experts = router_logits.shape

    # Epsilon for numerical stability
    eps = torch.finfo(dtype).eps if dtype.is_floating_point else 1e-8

    if num_tokens == 1:
        # Single-token case: compute p-values across experts
        # This ranks experts by their logits for this one token
        # Higher logit → higher rank → higher CDF → lower p-value
        
        logits_flat = router_logits[0]  # [num_experts]
        sorted_logits, _ = torch.sort(logits_flat)
        
        # For each expert, find its rank in the sorted order
        # Use searchsorted with side='right' to handle ties correctly
        ranks = torch.searchsorted(sorted_logits.contiguous(), logits_flat.contiguous(), side='right')
        
        # CDF = rank / N (rank is 1-indexed after searchsorted with side='right')
        cdf = ranks.float() / num_experts
        
        # P-value = 1 - CDF
        p_values = (1.0 - cdf).unsqueeze(0)  # [1, num_experts]
        
    else:
        # Multi-token case: for each expert, compute CDF across tokens
        p_values = torch.zeros_like(router_logits)

        for expert_idx in range(num_experts):
            expert_logits = router_logits[:, expert_idx]  # [num_tokens]

            # Sort logits for this expert
            sorted_logits, _ = torch.sort(expert_logits)

            # For each token, find rank and compute CDF
            # Use searchsorted with side='right' for proper ranking
            ranks = torch.searchsorted(sorted_logits.contiguous(), expert_logits.contiguous(), side='right')

            # CDF = rank / N
            cdf = ranks.float() / num_tokens

            # P-value = 1 - CDF
            p_values[:, expert_idx] = 1.0 - cdf

    # Clamp to (eps, 1-eps) for numerical stability
    p_values = torch.clamp(p_values, min=eps, max=1.0 - eps)

    return p_values.to(dtype)


def benjamini_hochberg_routing(
    router_logits: torch.Tensor,
    alpha: float = 0.05,
    temperature: float = 1.0,
    min_k: int = 1,
    max_k: int = 16,
    layer_idx: int = 0,
    kde_models: Optional[Dict[int, Dict]] = None,
    return_stats: bool = False,
    logger: Optional['BHRoutingLogger'] = None,
    log_every_n_tokens: int = 100,
    sample_idx: int = 0,
    token_idx: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Implements Benjamini-Hochberg procedure for expert selection in MoE models.

    The BH procedure controls the False Discovery Rate (FDR) when selecting which
    experts to activate for each token. Unlike fixed top-k routing, BH adapts the
    number of experts based on statistical significance.

    Mathematical Formulation:
    -------------------------
    Given router logits r ∈ R^N for N experts:

    1. Compute empirical p-values using KDE:
       - Use pre-trained KDE model for this layer's router logit distribution
       - For each logit value: p_i = 1 - CDF(r_i)
       - Higher logit → higher CDF → lower p-value → more significant
       - Based on actual empirical distribution, properly calibrated

    2. Sort p-values ascending: p_(1) ≤ p_(2) ≤ ... ≤ p_(N)

    3. Compute BH critical values: c_k = (k/N) × α

    4. Apply BH step-up procedure:
       - Find largest k where: p_(k) ≤ c_k
       - Select all experts with rank ≤ k (k smallest p-values)

    5. Compute routing weights: π = softmax(r / τ)

    6. Renormalize selected expert weights to sum to 1

    Why KDE-based p-values work:
    - Based on empirical distribution of router logits from real data
    - P-values properly calibrated: reflect actual statistical significance
    - High-logit experts get low p-values (as expected)
    - Low-logit experts get high p-values (correctly rejected)
    - BH thresholds c_k = (k/N)α work as intended

    The BH step-up procedure controls the False Discovery Rate (FDR):
    - Lower α → more conservative → fewer experts selected
    - Higher α → more permissive → more experts selected
    - α represents the expected proportion of false positives

    NOTE: Previous p-value approaches failed:
    - p_i = 1 - softmax(r)_i: all high (~0.85), none pass threshold
    - log p_i = log_softmax(-r): all low, almost all pass threshold
    - p_i = rank/N: thresholds too small (α/N), only 1 expert selected
    KDE approach solves this by using empirical distributions.

    Args:
        router_logits: Router logits from MoE gating network
            Shape: [batch_size, seq_len, num_experts] or [num_tokens, num_experts]
        alpha: FDR control level (default: 0.05)
            Lower values → more conservative → fewer experts selected
            Typical values: 0.01 (strict), 0.05 (standard), 0.10 (permissive)
        temperature: Softmax temperature for probability calibration (default: 1.0)
            Higher values → more uniform distribution → more experts selected
            Lower values → sharper distribution → fewer experts selected
        min_k: Minimum number of experts to select per token (default: 1)
            Ensures at least min_k experts are always activated
        max_k: Maximum number of experts to select per token (default: 16)
            Caps selection for compatibility with fixed-size expert computation
        layer_idx: Layer index for KDE model lookup (default: 0)
            Used to select the appropriate pre-trained KDE model
        kde_models: Pre-loaded KDE models (optional)
            If None, will attempt to load from default locations
            If loading fails, falls back to empirical p-values from batch
        return_stats: If True, return additional statistics (default: False)
        logger: Optional BHRoutingLogger for detailed logging (default: None)
            If provided, logs routing decisions at sampling intervals
        log_every_n_tokens: Log every N tokens (default: 100)
            Controls sampling rate to manage logging overhead
        sample_idx: Sample index in the batch (default: 0)
            Used for identifying logged entries
        token_idx: Token index in the sequence (default: 0)
            Used for identifying logged entries and sampling control

    Returns:
        routing_weights: Sparse routing weights, summing to 1 per token
            Shape: [batch_size, seq_len, num_experts]
            Non-selected experts have weight 0
        selected_experts: Indices of selected experts, padded with -1
            Shape: [batch_size, seq_len, max_k]
            Padding value: -1 indicates unused slot
        expert_counts: Number of experts selected per token
            Shape: [batch_size, seq_len]

        If return_stats=True, also returns:
            stats: Dict with keys:
                - 'p_values': Computed pseudo p-values [B, S, N]
                - 'bh_threshold': BH threshold used [B, S]
                - 'significant_mask': Boolean mask of selected experts [B, S, N]

    Raises:
        ValueError: If input shapes are invalid or parameters out of range
        RuntimeError: If numerical instability is detected

    Examples:
        >>> import torch
        >>> router_logits = torch.randn(2, 10, 64)  # 2 samples, 10 tokens, 64 experts
        >>> weights, experts, counts = benjamini_hochberg_routing(
        ...     router_logits, alpha=0.05, temperature=1.0, max_k=8
        ... )
        >>> print(weights.shape)  # [2, 10, 64] - sparse weights
        >>> print(experts.shape)  # [2, 10, 8] - selected expert indices
        >>> print(counts.shape)   # [2, 10] - number selected per token
        >>> print(weights.sum(dim=-1))  # All sum to 1.0

    Notes:
        - The function is fully vectorized and GPU-compatible
        - No Python loops are used for batch/sequence dimensions
        - Numerical stability is ensured via epsilon constants
        - If BH procedure selects 0 experts, min_k experts with lowest p-values are used
        - If BH selects > max_k, only top max_k by probability are kept
    """
    # =========================================================================
    # Input Validation
    # =========================================================================
    if not isinstance(router_logits, torch.Tensor):
        raise TypeError(f"router_logits must be a torch.Tensor, got {type(router_logits)}")

    if router_logits.ndim not in [2, 3]:
        raise ValueError(
            f"router_logits must be 2D [num_tokens, num_experts] or "
            f"3D [batch, seq_len, num_experts], got shape {router_logits.shape}"
        )

    # Handle 2D input by adding batch dimension
    input_is_2d = router_logits.ndim == 2
    if input_is_2d:
        router_logits = router_logits.unsqueeze(0)  # [1, num_tokens, num_experts]

    batch_size, seq_len, num_experts = router_logits.shape

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    if not 1 <= min_k <= num_experts:
        raise ValueError(f"min_k must be in [1, {num_experts}], got {min_k}")

    if not min_k <= max_k <= num_experts:
        raise ValueError(f"max_k must be in [{min_k}, {num_experts}], got {max_k}")

    # Warn if temperature is far from 1.0
    if temperature > 2.0 or temperature < 0.5:
        warnings.warn(
            f"Unusual temperature value: {temperature}. "
            f"Typical range is [0.5, 2.0]. This may lead to unexpected behavior.",
            UserWarning
        )

    device = router_logits.device
    dtype = router_logits.dtype

    # Epsilon for numerical stability
    eps = torch.finfo(dtype).eps if dtype.is_floating_point else 1e-8

    # =========================================================================
    # Step 1: Compute Softmax Probabilities (for final weights)
    # =========================================================================
    # Apply temperature scaling and softmax
    # probs: [batch_size, seq_len, num_experts]
    scaled_logits = router_logits / temperature
    probs = F.softmax(scaled_logits, dim=-1, dtype=torch.float32)

    # Convert back to original dtype
    probs = probs.to(dtype)

    # Ensure probabilities are valid (sum to 1, no negatives)
    # Note: Softmax guarantees this, but we check for numerical errors
    prob_sums = probs.sum(dim=-1)
    if not torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5):
        warnings.warn(
            f"Probability sums deviate from 1.0. Max deviation: {(prob_sums - 1.0).abs().max().item():.2e}",
            RuntimeWarning
        )

    # =========================================================================
    # Step 2: Load KDE Models and Compute P-Values
    # =========================================================================
    # Load KDE models if not provided
    if kde_models is None:
        kde_models = load_kde_models()

    # Flatten to 2D for p-value computation
    router_logits_2d = router_logits.view(-1, num_experts)

    # Compute p-values using KDE: p = 1 - CDF(logit)
    # Higher logit → higher CDF → lower p-value → more significant
    p_values = compute_pvalues_kde(router_logits_2d, layer_idx, kde_models)
    # Shape: [batch_size * seq_len, num_experts]

    # Reshape back to 3D
    p_values = p_values.view(batch_size, seq_len, num_experts)
    # Shape: [batch_size, seq_len, num_experts]

    # =========================================================================
    # Step 3: Sort P-Values (Ascending = Most Significant First)
    # =========================================================================
    sorted_pvals, sorted_indices = torch.sort(p_values, dim=-1)
    # sorted_pvals[i, j, 0] is the smallest (most significant) p-value
    # sorted_indices[i, j, k] is the original expert index for rank k

    # =========================================================================
    # Step 4: Compute BH Critical Values
    # =========================================================================
    # BH threshold for rank k: (k / N) * alpha
    k_values = torch.arange(1, num_experts + 1, device=device, dtype=torch.float32)
    critical_values = (k_values / num_experts) * alpha
    # Shape: [num_experts]

    # Broadcast to batch dimensions: [1, 1, num_experts]
    critical_values = critical_values.view(1, 1, -1)

    # =========================================================================
    # Step 5: Apply BH Step-Up Procedure
    # =========================================================================
    # For each token, find the largest k where p_(k) <= critical_value_(k)
    # BH step-up: All experts with rank <= k are selected

    # Compare each sorted p-value to its threshold
    passes_threshold = sorted_pvals <= critical_values
    # Shape: [batch_size, seq_len, num_experts]

    # Find the largest k that passes for each token
    # Create indices [1, 2, 3, ..., N]
    k_indices = torch.arange(1, num_experts + 1, device=device).view(1, 1, -1)

    # Mask indices where threshold is NOT passed
    masked_indices = torch.where(
        passes_threshold,
        k_indices.float(),
        torch.zeros_like(k_indices).float()
    )

    # Take the maximum index that passed (largest k)
    num_selected = masked_indices.max(dim=-1).values  # [batch_size, seq_len]

    # Handle case where nothing passes (default to min_k)
    num_selected = torch.where(
        num_selected == 0,
        torch.tensor(min_k, device=device, dtype=num_selected.dtype),
        num_selected
    )

    # Clamp to [min_k, max_k]
    num_selected = num_selected.clamp(min=min_k, max=max_k).long()

    # =========================================================================
    # Step 6: Select Experts and Compute Weights
    # =========================================================================
    # Create a mask for selected experts in the SORTED order
    # selected_mask_sorted: [batch_size, seq_len, num_experts]
    expert_ranks = torch.arange(num_experts, device=device).view(1, 1, -1)
    expert_ranks = expert_ranks.expand(batch_size, seq_len, -1)

    # selected_mask_sorted[b, s, k] = True if k < num_selected[b, s]
    selected_mask_sorted = expert_ranks < num_selected.unsqueeze(-1)

    # Convert mask from sorted order to original order
    # We need to scatter the mask using sorted_indices
    # selected_mask: [batch_size, seq_len, num_experts]
    selected_mask = torch.zeros_like(probs, dtype=torch.bool)
    selected_mask.scatter_(
        dim=-1,
        index=sorted_indices,
        src=selected_mask_sorted
    )

    # Extract selected probabilities and renormalize
    # routing_weights: [batch_size, seq_len, num_experts]
    routing_weights = torch.where(
        selected_mask,
        probs,
        torch.zeros_like(probs)
    )

    # Renormalize weights to sum to 1 per token
    weight_sums = routing_weights.sum(dim=-1, keepdim=True)
    # Avoid division by zero (shouldn't happen since min_k >= 1, but be safe)
    weight_sums = torch.clamp(weight_sums, min=eps)
    routing_weights = routing_weights / weight_sums

    # =========================================================================
    # Step 8: Extract Selected Expert Indices (Padded to max_k)
    # =========================================================================
    # selected_experts: [batch_size, seq_len, max_k]
    # Padding value: -1 (indicates unused slot)

    selected_experts = torch.full(
        (batch_size, seq_len, max_k),
        fill_value=-1,
        dtype=torch.long,
        device=device
    )

    # For each token, fill in the selected expert indices
    # Method: Take the first max_k indices from sorted_indices where mask is True

    for k_idx in range(max_k):
        # Check if this slot should be filled (k_idx < num_selected)
        slot_active = k_idx < num_selected  # [B, S]

        # Get the k_idx-th selected expert (in sorted order)
        # This is sorted_indices[:, :, k_idx] where selected_mask_sorted[:, :, k_idx] is True
        # Since selected_mask_sorted[:, :, k_idx] is True iff k_idx < num_selected,
        # we can directly take sorted_indices[:, :, k_idx]

        expert_idx = sorted_indices[:, :, k_idx]  # [B, S]

        # Only assign if slot is active
        selected_experts[:, :, k_idx] = torch.where(
            slot_active,
            expert_idx,
            torch.full_like(expert_idx, fill_value=-1)
        )

    # =========================================================================
    # Step 9: Prepare Outputs
    # =========================================================================

    # =========================================================================
    # LOGGING: Log routing decision if logger is provided
    # =========================================================================
    if logger is not None and token_idx % log_every_n_tokens == 0:
        # Log for the specified (sample_idx, token_idx) position
        # Handle both 2D (single sample) and 3D (batched) inputs
        if input_is_2d:
            # For 2D input: sample_idx=0 (implicit), token_idx is the sequence position
            if token_idx < seq_len:
                log_entry = {
                    'sample_idx': sample_idx,
                    'token_idx': token_idx,
                    'layer_idx': layer_idx,
                    'router_logits': router_logits[0, token_idx, :].detach().cpu().numpy(),
                    'p_values': p_values[0, token_idx, :].detach().cpu().numpy(),
                    'selected_experts': selected_experts[0, token_idx, :].detach().cpu().tolist(),
                    'num_selected': int(num_selected[0, token_idx].item()),
                    'alpha': alpha,
                    'max_k': max_k,
                    'min_k': min_k,
                    'sorted_p_values': sorted_pvals[0, token_idx, :].detach().cpu().numpy(),
                }
                logger.log_routing_decision(log_entry)
        else:
            # For 3D input: use provided sample_idx and token_idx
            if sample_idx < batch_size and token_idx < seq_len:
                log_entry = {
                    'sample_idx': sample_idx,
                    'token_idx': token_idx,
                    'layer_idx': layer_idx,
                    'router_logits': router_logits[sample_idx, token_idx, :].detach().cpu().numpy(),
                    'p_values': p_values[sample_idx, token_idx, :].detach().cpu().numpy(),
                    'selected_experts': selected_experts[sample_idx, token_idx, :].detach().cpu().tolist(),
                    'num_selected': int(num_selected[sample_idx, token_idx].item()),
                    'alpha': alpha,
                    'max_k': max_k,
                    'min_k': min_k,
                    'sorted_p_values': sorted_pvals[sample_idx, token_idx, :].detach().cpu().numpy(),
                }
                logger.log_routing_decision(log_entry)

    # If input was 2D, remove batch dimension from outputs
    if input_is_2d:
        routing_weights = routing_weights.squeeze(0)  # [seq_len, num_experts]
        selected_experts = selected_experts.squeeze(0)  # [seq_len, max_k]
        num_selected = num_selected.squeeze(0)  # [seq_len]

    # Prepare statistics if requested
    if return_stats:
        # Return diagnostic information about BH procedure
        selected_mask_output = selected_mask
        p_values_output = p_values
        critical_values_output = critical_values

        if input_is_2d:
            selected_mask_output = selected_mask.squeeze(0)
            p_values_output = p_values.squeeze(0)

        stats = {
            'selected_mask': selected_mask_output,
            'num_selected': num_selected,
            'p_values': p_values_output,
            'critical_values': critical_values_output.squeeze(),
            'alpha': alpha,
            'temperature': temperature,
            'layer_idx': layer_idx,
            'kde_available': layer_idx in kde_models if kde_models else False
        }

        return routing_weights, selected_experts, num_selected, stats

    return routing_weights, selected_experts, num_selected


def topk_routing(
    router_logits: torch.Tensor,
    k: int = 8,  # Now explicitly parameterized with default
    temperature: float = 1.0,
    normalize: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standard top-k routing for comparison with BH routing.

    Args:
        router_logits: Router logits [batch, seq_len, num_experts] or [num_tokens, num_experts]
        k: Number of experts to select (default: 8)
        temperature: Softmax temperature (default: 1.0)
        normalize: Whether to renormalize selected weights (default: True)

    Returns:
        routing_weights: Sparse routing weights [batch, seq_len, num_experts]
        selected_experts: Selected expert indices [batch, seq_len, k]
        expert_counts: Number of experts per token [batch, seq_len] (always k)
    """
    input_is_2d = router_logits.ndim == 2
    if input_is_2d:
        router_logits = router_logits.unsqueeze(0)

    batch_size, seq_len, num_experts = router_logits.shape

    if k > num_experts:
        raise ValueError(f"k={k} cannot exceed num_experts={num_experts}")

    # Compute softmax probabilities
    scaled_logits = router_logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)

    # Select top-k
    topk_weights, topk_indices = torch.topk(probs, k, dim=-1)

    # Renormalize if requested
    if normalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    # Create sparse weight tensor
    routing_weights = torch.zeros_like(probs)
    routing_weights.scatter_(dim=-1, index=topk_indices, src=topk_weights)

    # Expert counts (constant k for all tokens)
    expert_counts = torch.full(
        (batch_size, seq_len),
        fill_value=k,
        dtype=torch.long,
        device=router_logits.device
    )

    if input_is_2d:
        routing_weights = routing_weights.squeeze(0)
        topk_indices = topk_indices.squeeze(0)
        expert_counts = expert_counts.squeeze(0)

    return routing_weights, topk_indices, expert_counts


# ============================================================================
# Utility Functions
# ============================================================================

def compute_routing_statistics(
    routing_weights: torch.Tensor,
    expert_counts: torch.Tensor
) -> Dict[str, float]:
    """
    Compute statistics about routing decisions.

    Args:
        routing_weights: Routing weights [batch, seq_len, num_experts]
        expert_counts: Number of experts selected per token [batch, seq_len]

    Returns:
        Dictionary with statistics:
            - mean_experts: Average experts per token
            - std_experts: Std dev of experts per token
            - min_experts: Minimum experts selected
            - max_experts: Maximum experts selected
            - sparsity: Fraction of zero weights
            - weight_entropy: Average entropy of weight distribution
    """
    stats = {}

    # Expert count statistics
    stats['mean_experts'] = expert_counts.float().mean().item()
    stats['std_experts'] = expert_counts.float().std().item()
    stats['min_experts'] = expert_counts.min().item()
    stats['max_experts'] = expert_counts.max().item()

    # Sparsity (fraction of zero weights)
    total_weights = routing_weights.numel()
    zero_weights = (routing_weights == 0).sum().item()
    stats['sparsity'] = zero_weights / total_weights

    # Weight entropy (measure of concentration)
    # H = -sum(p * log(p)) for p > 0
    eps = 1e-10
    nonzero_weights = routing_weights + eps
    entropy = -(routing_weights * torch.log(nonzero_weights)).sum(dim=-1)
    stats['weight_entropy'] = entropy.mean().item()

    return stats


def run_bh_multi_k(
    router_logits: torch.Tensor,
    max_k_values: list = None,
    alpha: float = 0.05,
    temperature: float = 1.0,
    min_k: int = 1
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Run BH routing with multiple max_k values for comparison.

    Args:
        router_logits: [batch, seq_len, num_experts] or [num_tokens, num_experts]
        max_k_values: List of maximum expert counts to test (default: [8, 16, 32, 64])
        alpha: FDR control level
        temperature: Softmax temperature
        min_k: Minimum experts to select

    Returns:
        Dict mapping max_k to (routing_weights, selected_experts, expert_counts)

    Example:
        >>> results = run_bh_multi_k(logits, max_k_values=[8, 16, 32, 64])
        >>> for k, (weights, experts, counts) in results.items():
        ...     print(f"max_k={k}: avg experts = {counts.float().mean():.2f}")
    """
    if max_k_values is None:
        max_k_values = [8, 16, 32, 64]

    results = {}
    for max_k in max_k_values:
        weights, experts, counts = benjamini_hochberg_routing(
            router_logits,
            alpha=alpha,
            temperature=temperature,
            min_k=min_k,
            max_k=max_k
        )
        results[max_k] = (weights, experts, counts)
    return results


def compare_multi_k_statistics(
    results: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> 'pd.DataFrame':
    """
    Generate comparison statistics for different max_k values.

    Args:
        results: Output from run_bh_multi_k()

    Returns:
        DataFrame with columns:
            - max_k
            - mean_experts
            - std_experts
            - min_experts
            - max_experts
            - pct_at_ceiling (% hitting max_k limit)
            - pct_at_floor (% at min_k)
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for compare_multi_k_statistics. Install with: pip install pandas")

    rows = []
    for max_k, (weights, experts, counts) in results.items():
        counts_flat = counts.flatten().float()
        rows.append({
            'max_k': max_k,
            'mean_experts': counts_flat.mean().item(),
            'std_experts': counts_flat.std().item(),
            'min_experts': counts_flat.min().item(),
            'max_experts': counts_flat.max().item(),
            'pct_at_ceiling': (counts_flat == max_k).float().mean().item() * 100,
            'pct_at_floor': (counts_flat == 1).float().mean().item() * 100,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Quick demonstration
    print("BH Routing Demonstration")
    print("=" * 70)

    # Create sample router logits
    torch.manual_seed(42)
    batch_size, seq_len, num_experts = 2, 4, 16
    router_logits = torch.randn(batch_size, seq_len, num_experts)

    print(f"\nInput shape: {router_logits.shape}")
    print(f"Testing with alpha=0.05, temperature=1.0, max_k=8\n")

    # Run BH routing
    weights, experts, counts, stats = benjamini_hochberg_routing(
        router_logits,
        alpha=0.05,
        temperature=1.0,
        max_k=8,
        return_stats=True
    )

    print(f"Output shapes:")
    print(f"  routing_weights: {weights.shape}")
    print(f"  selected_experts: {experts.shape}")
    print(f"  expert_counts: {counts.shape}")

    print(f"\nExpert counts per token:")
    print(counts)

    print(f"\nWeight sums (should all be ~1.0):")
    print(weights.sum(dim=-1))

    print(f"\nRouting statistics:")
    routing_stats = compute_routing_statistics(weights, counts)
    for key, value in routing_stats.items():
        print(f"  {key}: {value:.4f}")

    print(f"\nBH statistics:")
    print(f"  Mean p-value: {stats['p_values'].mean().item():.4f}")
    print(f"  Mean BH threshold: {stats['bh_threshold'].mean().item():.4f}")

    # Compare with top-k
    print(f"\n" + "=" * 70)
    print("Comparison with Top-K routing (k=8)")
    print("=" * 70)

    topk_weights, topk_experts, topk_counts = topk_routing(
        router_logits, k=8, temperature=1.0
    )

    topk_stats = compute_routing_statistics(topk_weights, topk_counts)
    print(f"\nTop-K statistics:")
    for key, value in topk_stats.items():
        print(f"  {key}: {value:.4f}")

    print(f"\nDifference in mean experts: {routing_stats['mean_experts'] - topk_stats['mean_experts']:.2f}")
    print("(Negative = BH uses fewer experts on average)")

    # =========================================================================
    # NEW: Multi-K Demonstration
    # =========================================================================
    print(f"\n" + "=" * 70)
    print("Multi-K BH Routing Comparison")
    print("=" * 70)

    results = run_bh_multi_k(
        router_logits,
        max_k_values=[8, 16, 32, 64],
        alpha=0.05
    )

    comparison_df = compare_multi_k_statistics(results)
    print("\nComparison across max_k values:")
    print(comparison_df.to_string(index=False))

    print("\nKey Insights:")
    print("  - Higher max_k allows more experts when needed")
    print("  - pct_at_ceiling shows how often BH hits the max_k limit")
    print("  - pct_at_floor shows how conservative BH is being")
