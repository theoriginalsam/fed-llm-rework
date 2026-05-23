"""
HetLoRA Baseline (Cho et al., EMNLP 2024).
"Heterogeneous LoRA for Federated Fine-tuning of On-Device Foundation Models."

Algorithm (3 steps each round):
  1. Distribution via Truncation:
       Server holds global (B̄, Ā) at r_max. Distributes to client k by
       truncating: B̄_{:, :r_k}, Ā_{:r_k, :}. No SVD.

  2. Local Training with Rank Self-Pruning (gamma < 1):
       Each client trains LoRA + regularization on last-rank components.
       If norms of last-rank layers shrink below initial, prune rank.
       We implement gamma=1.0 (no pruning) by default for a clean
       apples-to-apples comparison — pruning is orthogonal to the
       aggregation contribution.

  3. Sparsity-Weighted Aggregation:
       Zero-pad all (B_k, A_k) to r_max. Weight each client by
       p_k ∝ ||ΔW_k||_F = ||B_k A_k||_F, computed cheaply as
       sqrt(tr(B_k^T B_k · A_k A_k^T)) without forming the full matrix.
       Aggregate directly in (B, A) space — NOT in ΔW space.

Key distinction from FlexLoRA/SPA:
  - FlexLoRA/SPA reconstruct ΔW = BA (rotation-invariant), aggregate,
    then SVD back to rank-r (Frobenius-optimal projection).
  - HetLoRA aggregates (B, A) directly (non-identifiable: BA = (BQ)(Q^{-1}A)
    for any invertible Q). Distributes by truncation, not SVD.

Our counter-argument (for paper): HetLoRA's direct (B,A) aggregation depends
on each client's arbitrary factorization chosen by SGD. Our reconstruction-first
approach operates on ΔW — invariant to the rotation ambiguity — providing a
principled foundation for aggregation.
"""

import torch
from typing import Dict, List, Tuple


class HetLoRAAggregator:
    """
    Sparsity-weighted (B, A) aggregation with truncation-based distribution.
    Stores raw client uploads; computes Frobenius weights at aggregation time.
    """

    def __init__(self, max_rank: int = 32):
        self.max_rank = max_rank
        self._client_loras: List[Dict] = []   # [{layer: {"A": ..., "B": ...}}, ...]

    def reset(self):
        self._client_loras = []

    def update(self, client_lora: Dict[str, Dict[str, torch.Tensor]], weight: float = 1.0):
        """
        Collect one client's (A, B) upload. The server-side data-volume weight
        is intentionally ignored — HetLoRA uses Frobenius norms as weights.

        client_lora: {layer_key: {"A": (r_k, d_in), "B": (d_out, r_k)}}
        """
        self._client_loras.append(client_lora)

    @staticmethod
    def _frob_norm(B: torch.Tensor, A: torch.Tensor) -> float:
        """
        ||BA||_F = sqrt(tr(B^T B · A A^T)) — O(r^3), no full matrix needed.
        """
        BtB = (B.T @ B).float()   # (r, r)
        AAt = (A @ A.T).float()   # (r, r)
        return float(torch.trace(BtB @ AAt).clamp(min=0.0).sqrt())

    def get_global_ba(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute Frobenius-weighted average of (B, A) pairs at r_max.

        Returns: {layer_key: {"A": (max_rank, d_in), "B": (d_out, max_rank)}}
        """
        if not self._client_loras:
            return {}

        # Compute per-client Frobenius norm (sum across all layers)
        frob_norms = []
        for client_lora in self._client_loras:
            total = 0.0
            for mats in client_lora.values():
                B = mats["B"].cpu().float()
                A = mats["A"].cpu().float()
                total += self._frob_norm(B, A)
            frob_norms.append(total)

        Z = sum(frob_norms)
        if Z < 1e-12:
            # Fallback to uniform if all updates are zero
            n = len(frob_norms)
            weights = [1.0 / n] * n
        else:
            weights = [f / Z for f in frob_norms]

        b_agg: Dict[str, torch.Tensor] = {}
        a_agg: Dict[str, torch.Tensor] = {}

        for client_lora, w in zip(self._client_loras, weights):
            for layer_key, mats in client_lora.items():
                B = mats["B"].cpu().float()   # (d_out, r_k)
                A = mats["A"].cpu().float()   # (r_k, d_in)
                r_k = A.shape[0]
                d_out, d_in = B.shape[0], A.shape[1]

                # Zero-pad to max_rank
                B_pad = torch.zeros(d_out, self.max_rank)
                A_pad = torch.zeros(self.max_rank, d_in)
                B_pad[:, :r_k] = B
                A_pad[:r_k, :] = A

                if layer_key not in b_agg:
                    b_agg[layer_key] = torch.zeros_like(B_pad)
                    a_agg[layer_key] = torch.zeros_like(A_pad)

                b_agg[layer_key] += B_pad * w
                a_agg[layer_key] += A_pad * w

        return {k: {"A": a_agg[k], "B": b_agg[k]} for k in b_agg}

    def get_global(self) -> Dict[str, torch.Tensor]:
        """
        Return ΔW = B_agg @ A_agg for eval compatibility.
        Used only for computing global_wagg for eval — the (B, A) path
        is used for actual client distribution.
        """
        ba = self.get_global_ba()
        return {k: v["B"] @ v["A"] for k, v in ba.items()}

    @staticmethod
    def distribute_to_client(
        global_ba: Dict[str, Dict[str, torch.Tensor]],
        rank: int,
        device: str = "cuda",
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Distribute global (B̄, Ā) to a client of given rank via truncation.
        No SVD — just slice the top-rank columns/rows.

        Returns: {layer_key: {"A": (rank, d_in), "B": (d_out, rank)}}
        """
        result = {}
        for layer_key, mats in global_ba.items():
            B = mats["B"][:, :rank]   # (d_out, rank)
            A = mats["A"][:rank, :]   # (rank, d_in)
            result[layer_key] = {"A": A.cpu(), "B": B.cpu()}
        return result
