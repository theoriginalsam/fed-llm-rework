"""
SPA: Subspace Projection Aggregation (Our Method).

Algorithm:
  1. Each client k uploads (A_k, B_k) for each LoRA layer.
  2. Server reconstructs full weight update: ΔW_k = B_k @ A_k.
  3. Server computes weighted average: W_agg = Σ_k (n_k/N) ΔW_k.
  4. Server applies SVD: W_agg = U Σ V^T.
  5. Server projects back to each client's rank r_j:
       A_j = sqrt(Σ_{1:r_j}) @ V^T_{1:r_j}
       B_j = U_{:,1:r_j} @ sqrt(Σ_{1:r_j})
  6. Optional: drop components where σ_i < tau * σ_1 (spectral denoising threshold).

Key difference from FlexLoRA: SPA adds an adaptive singular-value threshold (tau)
that drops noise components based on the spectral decay, not just the hardware rank.
This is provably the best rank-r approximation in Frobenius norm.
"""

import torch
from typing import Dict, List, Optional, Tuple


class SPAAggregator:
    """Streaming weighted aggregation with SVD-based projection."""

    def __init__(self, max_rank: int = 32, tau: float = 0.0):
        """
        Args:
            max_rank: maximum LoRA rank in the system.
            tau: spectral denoising threshold. Drop singular values where
                 sigma_i < tau * sigma_1. 0.0 = no denoising (equivalent to FlexLoRA).
                 Recommended: 0.01–0.05 for noisy non-IID settings.
        """
        self.max_rank = max_rank
        self.tau = tau
        self._accumulator: Dict[str, torch.Tensor] = {}
        self._initialized = False

    def reset(self):
        self._accumulator = {}
        self._initialized = False

    def update(self, client_weights: Dict[str, torch.Tensor], weight: float):
        """
        Accumulate one client's reconstructed ΔW matrices.

        client_weights: dict mapping layer_key -> ΔW tensor (d_out x d_in).
                        Caller is responsible for computing ΔW = B @ A before passing.
        weight: fractional dataset weight for this client (n_k / N).
        """
        if not self._initialized:
            self._accumulator = {k: torch.zeros_like(v, device="cpu", dtype=torch.float32)
                                 for k, v in client_weights.items()}
            self._initialized = True

        for k, delta_w in client_weights.items():
            self._accumulator[k] += delta_w.cpu().float() * weight

    def get_global(self) -> Dict[str, torch.Tensor]:
        """Return accumulated W_agg dict."""
        return self._accumulator

    @staticmethod
    def project_to_rank(
        w_agg: torch.Tensor,
        rank: int,
        tau: float = 0.0,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project W_agg to LoRA matrices (B, A) of given rank using truncated SVD.

        Returns (B, A) such that B @ A ≈ W_agg with rank-r Frobenius-optimal approximation.
        Applies spectral denoising: drops components where σ_i < tau * σ_1.
        """
        W = w_agg.float().to(device)
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)

        # Determine effective rank after threshold
        if tau > 0.0 and len(S) > 0:
            threshold = tau * S[0].item()
            effective_rank = max(1, int((S >= threshold).sum().item()))
            k = min(rank, effective_rank)
        else:
            k = min(rank, len(S))

        S_k = S[:k]
        U_k = U[:, :k]
        Vt_k = Vt[:k, :]

        # Distribute singular values symmetrically between B and A
        S_sqrt = torch.diag(torch.sqrt(S_k))
        A = S_sqrt @ Vt_k          # shape: (k, d_in)
        B = U_k @ S_sqrt            # shape: (d_out, k)

        # Zero-pad if k < rank (happens when W_agg has numerical rank < rank)
        if k < rank:
            pad = rank - k
            A = torch.cat([A, torch.zeros(pad, A.shape[1], device=device)], dim=0)
            B = torch.cat([B, torch.zeros(B.shape[0], pad, device=device)], dim=1)

        return B.cpu(), A.cpu()

    def distribute(
        self,
        client_ranks: Dict[str, int],
        device: str = "cuda",
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        For each client, compute its rank-projected (B, A) from W_agg.

        client_ranks: {client_id: rank}
        Returns: {client_id: {layer_key: {"A": tensor, "B": tensor}}}
        """
        global_weights = self.get_global()
        result = {}

        for client_id, rank in client_ranks.items():
            client_lora = {}
            for layer_key, w_agg in global_weights.items():
                B, A = self.project_to_rank(w_agg, rank, self.tau, device)
                client_lora[layer_key] = {"A": A, "B": B}
            result[client_id] = client_lora

        return result
