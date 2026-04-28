"""
Homogeneous baselines: Homo-r4, Homo-r8, and Hetero-Pad.

HomoAggregator: all clients use the same fixed rank; standard FedAvg on full ΔW.
HeteroPadAggregator: clients use heterogeneous ranks; pad A/B to max_rank with zeros,
                     aggregate, then truncate back to each client's rank.
"""

import torch
from typing import Dict, Tuple


class HomoAggregator:
    """Standard FedAvg on full ΔW matrices (all clients same rank)."""

    def __init__(self, rank: int, max_rank: int = 32):
        self.rank = rank
        self.max_rank = max_rank
        self._accumulator: Dict[str, torch.Tensor] = {}
        self._initialized = False

    def reset(self):
        self._accumulator = {}
        self._initialized = False

    def update(self, client_weights: Dict[str, torch.Tensor], weight: float):
        """client_weights: {layer_key: ΔW tensor}"""
        if not self._initialized:
            self._accumulator = {k: torch.zeros_like(v, device="cpu", dtype=torch.float32)
                                 for k, v in client_weights.items()}
            self._initialized = True
        for k, v in client_weights.items():
            self._accumulator[k] += v.cpu().float() * weight

    def get_global(self) -> Dict[str, torch.Tensor]:
        """Return W_agg dict (weighted average of ΔW matrices)."""
        return self._accumulator

    def distribute(
        self,
        client_ranks: Dict[str, int],
        device: str = "cuda",
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        from .spa import SPAAggregator
        result = {}
        for client_id in client_ranks:
            client_lora = {}
            for layer_key, w_agg in self._accumulator.items():
                B, A = SPAAggregator.project_to_rank(w_agg, self.rank, tau=0.0, device=device)
                client_lora[layer_key] = {"A": A, "B": B}
            result[client_id] = client_lora
        return result


class HeteroPadAggregator:
    """
    Zero-padding baseline for heterogeneous LoRA.

    Each client's (A, B) matrices are zero-padded to max_rank before aggregation.
    After averaging, truncate back to each client's rank.
    This is the simplest way to handle rank mismatch and serves as the
    "minimal change to FedAvg" baseline.
    """

    def __init__(self, max_rank: int = 32):
        self.max_rank = max_rank
        self._a_accum: Dict[str, torch.Tensor] = {}
        self._b_accum: Dict[str, torch.Tensor] = {}
        self._initialized = False

    def reset(self):
        self._a_accum = {}
        self._b_accum = {}
        self._initialized = False

    def update(
        self,
        client_lora: Dict[str, Dict[str, torch.Tensor]],
        weight: float,
        d_out_map: Dict[str, int],
        d_in_map: Dict[str, int],
    ):
        """
        client_lora: {layer_key: {"A": (r, d_in), "B": (d_out, r)}}
        d_out_map, d_in_map: layer dimensions for padding initialization.
        """
        for layer_key, mats in client_lora.items():
            A = mats["A"].cpu().float()   # (r, d_in)
            B = mats["B"].cpu().float()   # (d_out, r)
            r = A.shape[0]
            d_in = A.shape[1]
            d_out = B.shape[0]

            # Pad to max_rank
            A_padded = torch.zeros(self.max_rank, d_in)
            B_padded = torch.zeros(d_out, self.max_rank)
            A_padded[:r, :] = A
            B_padded[:, :r] = B

            if not self._initialized:
                self._a_accum[layer_key] = torch.zeros_like(A_padded)
                self._b_accum[layer_key] = torch.zeros_like(B_padded)

            self._a_accum[layer_key] += A_padded * weight
            self._b_accum[layer_key] += B_padded * weight

        self._initialized = True

    def distribute(
        self,
        client_ranks: Dict[str, int],
        device: str = "cuda",
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Truncate aggregated padded matrices to each client's rank."""
        result = {}
        for client_id, rank in client_ranks.items():
            client_lora = {}
            for layer_key in self._a_accum:
                A = self._a_accum[layer_key][:rank, :].to(device)
                B = self._b_accum[layer_key][:, :rank].to(device)
                client_lora[layer_key] = {"A": A.cpu(), "B": B.cpu()}
            result[client_id] = client_lora
        return result
