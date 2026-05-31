import torch
import numpy as np
import os
import pickle

def load_hc_estimator(estimator_path):
    """
    Loads HC estimators. 
    If path is a file (e.g. aggregated_KDE.pkl), returns a single estimator.
    If path is a directory, returns a list of estimators (one per expert) sorted by index.
    """
    if os.path.isfile(estimator_path):
        with open(estimator_path, "rb") as f:
            return pickle.load(f)
    elif os.path.isdir(estimator_path):
        # Find all logit_{i}.pkl files
        files = [f for f in os.listdir(estimator_path) if f.startswith("logit_") and f.endswith(".pkl")]
        # Sort by the integer index
        files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        estimators = []
        for f in files:
            with open(os.path.join(estimator_path, f), "rb") as fp:
                estimators.append(pickle.load(fp))
        return estimators
    else:
        raise ValueError(f"Estimator path not found: {estimator_path}")

def compute_hc_mask(router_logits: torch.Tensor, hc_estimator, hc_threshold: float, default_top_k: int) -> torch.Tensor:
    """
    Computes a boolean mask [batch_size * sequence_length, num_experts]
    Tokens with max_hc > hc_threshold will use all experts.
    Tokens with max_hc <= hc_threshold will use default_top_k experts.
    """
    logits_np = router_logits.detach().float().cpu().numpy()
    num_tokens, num_experts = logits_np.shape
    
    # 1. Compute p-values
    if isinstance(hc_estimator, list):
        if len(hc_estimator) != num_experts:
            raise ValueError(f"Number of estimators ({len(hc_estimator)}) does not match num_experts ({num_experts})")
        p_values_np = np.zeros_like(logits_np)
        for i, est in enumerate(hc_estimator):
            p_values_np[:, i] = 1.0 - est.cdf(logits_np[:, i])
    else:
        p_values_np = 1.0 - hc_estimator.cdf(logits_np.flatten()).reshape(logits_np.shape)
        
    p_values = torch.from_numpy(p_values_np).to(router_logits.device)
    
    # 2. Compute HC statistic
    p_sorted, _ = torch.sort(p_values, dim=-1)
    ranks = torch.arange(1, num_experts + 1, device=router_logits.device).float()
    expected = ranks / num_experts
    eps = 1e-10
    p_clamped = torch.clamp(p_sorted, min=eps, max=1.0 - eps)
    numerator = expected - p_clamped
    denominator = torch.sqrt(p_clamped * (1.0 - p_clamped) + eps)
    hc_stats = np.sqrt(num_experts) * numerator / denominator
    
    max_hc, _ = torch.max(hc_stats, dim=-1)
    
    # 3. Determine threshold usage
    use_all = max_hc > hc_threshold
    
    # 4. Build boolean mask
    expert_mask_bool = torch.zeros_like(router_logits, dtype=torch.bool)
    
    # Default routing: top_k
    routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)
    _, topk_idx = torch.topk(routing_weights, default_top_k, dim=-1)
    expert_mask_bool.scatter_(1, topk_idx, True)
    
    # Override for HC-triggered tokens
    expert_mask_bool[use_all, :] = True
    
    return expert_mask_bool
