import torch
import torch.nn.functional as F
from .utils import compute_hc_mask

def qwen_moe_forward_patch(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(hidden_states)

    hc_enabled = getattr(self.config, "hc_routing_enabled", False)
    hc_layer = getattr(self.config, "hc_layer", None)
    hc_estimator = getattr(self, "hc_estimator", None)
    hc_threshold = getattr(self.config, "hc_threshold", 4.0)

    if hc_enabled and getattr(self, "layer_idx", -1) == hc_layer and hc_estimator is not None:
        expert_mask_bool = compute_hc_mask(router_logits, hc_estimator, hc_threshold, self.top_k)
    else:
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        _, topk_idx = torch.topk(routing_weights, self.top_k, dim=-1)
        expert_mask_bool = torch.zeros_like(routing_weights, dtype=torch.bool)
        expert_mask_bool.scatter_(1, topk_idx, True)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights = routing_weights * expert_mask_bool.float()

    if self.norm_topk_prob:
        routing_weights /= (routing_weights.sum(dim=-1, keepdim=True) + 1e-9)
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    expert_mask_t = expert_mask_bool.T
    expert_hit = torch.nonzero(expert_mask_t.sum(dim=1))

    for expert_idx in expert_hit:
        expert_idx = expert_idx.item()
        expert_layer = self.experts[expert_idx]
        top_x = torch.where(expert_mask_t[expert_idx])[0]

        current_state = hidden_states[top_x]
        weights = routing_weights[top_x, expert_idx].unsqueeze(-1)

        current_hidden_states = expert_layer(current_state) * weights
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits

def deepseek_moe_forward_patch(self, hidden_states: torch.Tensor) -> torch.Tensor:
    residuals = hidden_states
    orig_shape = hidden_states.shape
    batch_size, seq_len, hidden_dim = hidden_states.shape

    hidden_states = hidden_states.view(-1, hidden_dim)

    hc_enabled = getattr(self.config, "hc_routing_enabled", False)
    hc_layer = getattr(self.config, "hc_layer", None)
    hc_estimator = getattr(self, "hc_estimator", None)
    hc_threshold = getattr(self.config, "hc_threshold", 4.0)

    if hc_enabled and getattr(self, "layer_idx", -1) == hc_layer and hc_estimator is not None:
        logits = F.linear(hidden_states.type(torch.float32), self.gate.weight.type(torch.float32), None)
        expert_mask_bool = compute_hc_mask(logits, hc_estimator, hc_threshold, self.gate.top_k)

        routing_weights = F.softmax(logits, dim=-1, dtype=torch.float32)
        routing_weights = routing_weights * expert_mask_bool.float()

        if self.gate.norm_topk_prob:
            routing_weights /= (routing_weights.sum(dim=-1, keepdim=True) + 1e-9)

        routing_weights = routing_weights * self.gate.routed_scaling_factor

        routed_states = torch.zeros_like(hidden_states)
        expert_mask_t = expert_mask_bool.T
        expert_hit = torch.nonzero(expert_mask_t.sum(dim=1))

        for expert_idx in expert_hit:
            expert_idx = expert_idx.item()
            expert_layer = self.experts[expert_idx + self.ep_rank * self.experts_per_rank]
            top_x = torch.where(expert_mask_t[expert_idx])[0]

            current_state = hidden_states[top_x]
            weights = routing_weights[top_x, expert_idx].unsqueeze(-1)

            current_hidden_states = expert_layer(current_state) * weights
            routed_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        
        hidden_states = routed_states.view(*orig_shape)
    else:
        topk_indices, topk_weights = self.gate(hidden_states.view(*orig_shape))
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        
    hidden_states = hidden_states + self.shared_experts(residuals)
    return hidden_states

def olmoe_moe_forward_patch(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(hidden_states)

    hc_enabled = getattr(self.config, "hc_routing_enabled", False)
    hc_layer = getattr(self.config, "hc_layer", None)
    hc_estimator = getattr(self, "hc_estimator", None)
    hc_threshold = getattr(self.config, "hc_threshold", 4.0)

    if hc_enabled and getattr(self, "layer_idx", -1) == hc_layer and hc_estimator is not None:
        expert_mask_bool = compute_hc_mask(router_logits, hc_estimator, hc_threshold, self.top_k)
    else:
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        _, topk_idx = torch.topk(routing_weights, self.top_k, dim=-1)
        expert_mask_bool = torch.zeros_like(routing_weights, dtype=torch.bool)
        expert_mask_bool.scatter_(1, topk_idx, True)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights = routing_weights * expert_mask_bool.float()

    if self.norm_topk_prob:
        routing_weights /= (routing_weights.sum(dim=-1, keepdim=True) + 1e-9)
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    expert_mask_t = expert_mask_bool.T
    expert_hit = torch.nonzero(expert_mask_t.sum(dim=1))

    for expert_idx in expert_hit:
        expert_idx = expert_idx.item()
        expert_layer = self.experts[expert_idx]
        top_x = torch.where(expert_mask_t[expert_idx])[0]

        current_state = hidden_states[top_x]
        weights = routing_weights[top_x, expert_idx].unsqueeze(-1)

        current_hidden_states = expert_layer(current_state) * weights
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits

def gemma_router_forward_patch(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_states = self.norm(hidden_states)
    hidden_states = hidden_states * self.scale * self.scalar_root_size

    expert_scores = self.proj(hidden_states)
    router_probabilities = torch.nn.functional.softmax(expert_scores, dim=-1)

    hc_enabled = getattr(self.config, "hc_routing_enabled", False)
    hc_layer = getattr(self.config, "hc_layer", None)
    hc_estimator = getattr(self, "hc_estimator", None)
    hc_threshold = getattr(self.config, "hc_threshold", 4.0)

    if hc_enabled and getattr(self, "layer_idx", -1) == hc_layer and hc_estimator is not None:
        expert_mask_bool = compute_hc_mask(expert_scores, hc_estimator, hc_threshold, self.config.top_k_experts)
        
        num_active_experts = expert_mask_bool.sum(dim=-1)
        max_active = num_active_experts.max().item()

        top_k_index = torch.zeros((expert_mask_bool.size(0), max_active), device=expert_scores.device, dtype=torch.long)
        top_k_weights = torch.zeros((expert_mask_bool.size(0), max_active), device=expert_scores.device, dtype=torch.float32)

        for i in range(expert_mask_bool.size(0)):
            active_idx = expert_mask_bool[i].nonzero().squeeze(-1)
            top_k_index[i, :len(active_idx)] = active_idx
            top_k_weights[i, :len(active_idx)] = router_probabilities[i, active_idx]
    else:
        top_k_weights, top_k_index = torch.topk(
            router_probabilities,
            k=self.config.top_k_experts,
            dim=-1,
        )

    top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)

    valid_mask = top_k_index < self.config.num_experts
    per_expert_scale_extended = torch.cat([self.per_expert_scale, torch.ones(1, device=self.per_expert_scale.device)])
    top_k_weights = top_k_weights * per_expert_scale_extended[top_k_index]
    top_k_weights = top_k_weights * valid_mask.float()

    return router_probabilities, top_k_weights, top_k_index

def patch_model_with_hc(model, model_type: str):
    """
    Patches the given transformer model in-place to support HC routing.
    """
    if model_type == "qwen":
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
                layer.mlp.layer_idx = i
                # bind new method
                layer.mlp.forward = qwen_moe_forward_patch.__get__(layer.mlp, type(layer.mlp))
    elif model_type == "deepseek":
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
                layer.mlp.layer_idx = i
                layer.mlp.forward = deepseek_moe_forward_patch.__get__(layer.mlp, type(layer.mlp))
    elif model_type == "olmoe":
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
                layer.mlp.layer_idx = i
                layer.mlp.forward = olmoe_moe_forward_patch.__get__(layer.mlp, type(layer.mlp))
    elif model_type == "gemma":
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, "router"):
                layer.router.layer_idx = i
                layer.router.forward = gemma_router_forward_patch.__get__(layer.router, type(layer.router))
    else:
        raise ValueError(f"Unknown model_type {model_type}")

def set_hc_estimator(model, model_type: str, layer_idx: int, estimator):
    """
    Injects the given estimator into the correct layer of the patched model.
    """
    if model_type in ["qwen", "deepseek", "olmoe"]:
        layer = model.model.layers[layer_idx]
        if hasattr(layer, "mlp"):
            layer.mlp.hc_estimator = estimator
    elif model_type == "gemma":
        layer = model.model.layers[layer_idx]
        if hasattr(layer, "router"):
            layer.router.hc_estimator = estimator
    else:
        raise ValueError(f"Unknown model_type {model_type}")
