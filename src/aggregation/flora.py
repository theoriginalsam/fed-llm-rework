"""
FLoRA Baseline (Wang et al., AAAI 2024).
"FLoRA: Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations."

Core idea: stack all client A matrices and B matrices (concatenate rather than average).
The global LoRA has rank = Σ r_k. Redistribute by truncating to each client's rank.

For heterogeneous ranks: we handle by stacking all clients' updates,
then redistributing the top-r_j rows/cols to each client.
"""

import torch
from typing import Dict, List, Optional, Tuple


class FLoRAAggregator:
    """
    FLoRA: Stack-then-truncate aggregation for heterogeneous LoRA.

    Upload: each client sends (A_k, B_k) of shape (r_k, d_in) and (d_out, r_k).
    Aggregate: server stacks A matrices horizontally, B matrices vertically.
      A_global = [A_1; A_2; ...; A_K]  shape: (Σr_k, d_in)
      B_global = [B_1, B_2, ..., B_K]  shape: (d_out, Σr_k)
    W_global = B_global @ A_global (full-rank reconstruction)
    Redistribute: use SVD on W_global, project to client rank (same as FlexLoRA from here).
    """

    def __init__(self, max_rank: int = 32):
        self.max_rank = max_rank
        self._a_stack: Dict[str, List[torch.Tensor]] = {}
        self._b_stack: Dict[str, List[torch.Tensor]] = {}
        self._weights: List[float] = []
        self._initialized = False

    def reset(self):
        self._a_stack = {}
        self._b_stack = {}
        self._weights = []
        self._initialized = False

    def update(
        self,
        client_lora: Dict[str, Dict[str, torch.Tensor]],
        weight: float,
    ):
        """
        client_lora: {layer_key: {"A": tensor(r,d_in), "B": tensor(d_out,r)}}
        weight: fractional weight for this client.
        """
        self._weights.append(weight)

        for layer_key, mats in client_lora.items():
            A = mats["A"].cpu().float()
            B = mats["B"].cpu().float()

            if layer_key not in self._a_stack:
                self._a_stack[layer_key] = []
                self._b_stack[layer_key] = []

            # Weight the matrices before stacking
            self._a_stack[layer_key].append(A * weight)
            self._b_stack[layer_key].append(B * weight)

    def get_global(self) -> Dict[str, torch.Tensor]:
        """Reconstruct W_global by stacking then multiplying."""
        global_w = {}
        for layer_key in self._a_stack:
            A_stack = torch.cat(self._a_stack[layer_key], dim=0)   # (Σr_k, d_in)
            B_stack = torch.cat(self._b_stack[layer_key], dim=1)   # (d_out, Σr_k)
            global_w[layer_key] = B_stack @ A_stack                # (d_out, d_in)
        return global_w

    def distribute(
        self,
        client_ranks: Dict[str, int],
        device: str = "cuda",
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """SVD-project W_global to each client's rank (same as FlexLoRA from this step)."""
        from .spa import SPAAggregator
        global_weights = self.get_global()
        result = {}

        for client_id, rank in client_ranks.items():
            client_lora = {}
            for layer_key, w_agg in global_weights.items():
                B, A = SPAAggregator.project_to_rank(w_agg, rank, tau=0.0, device=device)
                client_lora[layer_key] = {"A": A, "B": B}
            result[client_id] = client_lora

        return result
