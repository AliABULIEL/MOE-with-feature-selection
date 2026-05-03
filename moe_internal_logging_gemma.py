import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict

class RouterLogger:
    def __init__(self, model):
        """
        Initialize router logger for Gemma 4.
        """
        self.model = model
        self.hooks = []
        self.routing_data = []
        
        self.num_experts = getattr(model.config, "num_experts", 128)
        self.top_k = getattr(model.config, "top_k_experts", 8)

    def register_hooks(self, top_k: Optional[int] = None):
        self.remove_hooks()
        if top_k is not None:
            self.top_k = top_k

        # UPDATE: We now pass the router_module itself so we can access per_expert_scale
        def create_hook(layer_idx, router_module):
            def hook_fn(module, input, output):
                try:
                    with torch.no_grad():
                        router_logits = output

                        if router_logits.dim() == 3:
                            router_logits = router_logits.view(-1, router_logits.shape[-1])

                        probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)

                        topk_weights, topk_indices = torch.topk(
                            probs, min(self.top_k, probs.shape[-1]), dim=-1
                        )

                        # ====== GEMMA 4 SPECIFIC MATH ======
                        # 1. Normalize the top-k weights so they sum to 1 per token
                        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
                        
                        # 2. Apply per-expert scale directly to the weights
                        topk_weights = topk_weights * router_module.per_expert_scale[topk_indices]
                        # ===================================

                        max_prob = probs.max(dim=-1).values.mean().item()
                        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item()
                        concentration = topk_weights[:, 0].mean().item() if topk_weights.shape[1] > 0 else 0.0

                        unique_experts = topk_indices.unique().tolist()

                        self.routing_data.append(
                            {
                                "layer": layer_idx,
                                "router_logits": router_logits.detach().cpu(),
                                "expert_indices": topk_indices.detach().cpu(),
                                "expert_weights": topk_weights.detach().cpu(),
                                "probs": probs.detach().cpu(),
                                "stats": {
                                    "max_prob": max_prob,
                                    "entropy": entropy,
                                    "concentration": concentration,
                                    "num_tokens": router_logits.shape[0],
                                    "unique_experts_this_layer": len(unique_experts),
                                },
                            }
                        )
                except Exception as e:
                    print(f"Warning: Hook error at layer {layer_idx}: {e}")

            return hook_fn

        # Robust layer resolution to handle standard, ForCausalLM, and PEFT wrappers
        layers = None
        
        if hasattr(self.model, "language_model") and hasattr(self.model.language_model, "layers"):
            # Direct base multimodal model
            layers = self.model.language_model.layers
        elif hasattr(self.model, "model") and hasattr(self.model.model, "language_model") and hasattr(self.model.model.language_model, "layers"):
            # Wrapped in ForConditionalGeneration or similar
            layers = self.model.model.language_model.layers
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # Standard HF wrapper fallback
            layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            # Passed the base text model directly
            layers = self.model.layers
        elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "model"):
            # Wrapped in PEFT / LoRA
            if hasattr(self.model.base_model.model, "language_model"):
                layers = self.model.base_model.model.language_model.layers
            elif hasattr(self.model.base_model.model, "layers"):
                layers = self.model.base_model.model.layers

        if layers is None:
            # Print the top-level modules to help with debugging if it still fails
            module_names = [name for name, _ in self.model.named_children()]
            raise ValueError(f"Could not find layers. Top-level modules found: {module_names}")

        for i, layer in enumerate(layers):
            if hasattr(layer, "router") and hasattr(layer.router, "proj"):
                # UPDATE: Pass layer.router into the hook creation
                hook = layer.router.proj.register_forward_hook(create_hook(i, layer.router))
                self.hooks.append(hook)
                
        print(f"✅ Registered {len(self.hooks)} Gemma 4 router hooks (top_k={self.top_k})")

    def get_routing_data(self) -> List[Dict]:
        return self.routing_data

    def get_summary_stats(self) -> Dict[str, float]:
        if not self.routing_data:
            return {}

        all_max_probs = [d["stats"]["max_prob"] for d in self.routing_data]
        all_entropies = [d["stats"]["entropy"] for d in self.routing_data]
        all_concentrations = [d["stats"]["concentration"] for d in self.routing_data]

        all_experts = set()
        for d in self.routing_data:
            indices = d["expert_indices"].numpy().flatten()
            all_experts.update(indices.tolist())

        return {
            "avg_max_prob": float(np.mean(all_max_probs)),
            "avg_entropy": float(np.mean(all_entropies)),
            "avg_concentration": float(np.mean(all_concentrations)),
            "unique_experts_used": len(all_experts),
            "expert_utilization": len(all_experts) / self.num_experts,
            "num_layers_captured": len(self.routing_data),
        }

    def get_per_layer_stats(self) -> Dict[int, Dict]:
        per_layer = {}
        for d in self.routing_data:
            layer = d["layer"]
            per_layer[layer] = d["stats"]
        return per_layer

    def clear_data(self):
        self.routing_data = []

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []