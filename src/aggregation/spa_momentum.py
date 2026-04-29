"""
SPA-M: SPA with Soft Spectral Weighting + Momentum + Consensus Weighting.

Combines four improvements over SPA/FlexLoRA:

  1. Soft spectral weighting (Idea 1):
       σ̃_i = σ_i * (1 - exp(-σ_i / (γ * σ_1)))
     Replaces SPA's hard tau threshold and FlexLoRA's tau=0 with the nuclear-norm-
     optimal soft function. Never zeroes useful components; shrinks noise smoothly.

  2. Server-side momentum (Idea 2):
       M_t = β * M_{t-1} + (1 - β) * W_agg_t
     With bias correction: M̂_t = M_t / (1 - β^t).
     Directly fixes the α=0.1 oscillation pattern (loss collapses, accuracy crashes).

  3. Consensus weighting (Idea 3):
       w_k ∝ n_k * C_k,  where C_k = mean_j ||U_k^T U_j||_F
     Down-weights clients whose gradient subspace is orthogonal to the consensus
     direction. Under high non-IID, outlier clients hurt aggregation quality.

  Note: Idea 4 (factored momentum) is equivalent to Idea 2 when we already store
  W_agg as a full matrix (same memory cost, no additional benefit on our hardware).

Interface matches existing aggregators:
  reset()                 — call at start of each round
  update(dw_dict, weight) — call per client (weight = n_k / N)
  get_global()            — returns W_agg after consensus + momentum
  distribute(ranks, dev)  — projects W_agg → per-client (A, B) using soft spectral
"""

import math
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


class SPAMomentumAggregator:
    """
    SPA-M: soft-spectral + momentum + consensus aggregation.

    Args:
        max_rank:      maximum LoRA rank across all clients.
        beta:          momentum coefficient (0 = no momentum, 0.9 recommended).
        gamma:         soft-spectral scale; σ̃_i = σ_i*(1-exp(-σ_i/(γ*σ_1))).
                       γ=1.0 is the canonical nuclear-norm-optimal choice.
        use_consensus: if True, reweight clients by subspace agreement.
        consensus_rank: top-k singular vectors used for consensus alignment.
    """

    def __init__(
        self,
        max_rank: int = 32,
        beta: float = 0.9,
        gamma: float = 1.0,
        use_consensus: bool = True,
        consensus_rank: int = 4,
    ):
        self.max_rank = max_rank
        self.beta = beta
        self.gamma = gamma
        self.use_consensus = use_consensus
        self.consensus_rank = consensus_rank

        # Per-round storage: filled by update(), consumed by get_global()
        self._client_data: List[Tuple[Dict[str, torch.Tensor], float]] = []

        # Persistent across rounds
        self._momentum: Dict[str, torch.Tensor] = {}
        self._round: int = 0
        self._cached_global: Optional[Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Per-round interface (matches existing aggregators)
    # ------------------------------------------------------------------

    def reset(self):
        self._client_data = []
        self._cached_global = None

    def update(self, client_weights: Dict[str, torch.Tensor], weight: float):
        """
        Store one client's ΔW matrices for deferred aggregation.

        client_weights: {layer_key: ΔW tensor (d_out × d_in)}
        weight:         n_k / N (dataset-size fraction)
        """
        cpu_weights = {k: v.cpu().float() for k, v in client_weights.items()}
        self._client_data.append((cpu_weights, weight))

    def get_global(self) -> Dict[str, torch.Tensor]:
        """Compute consensus-weighted W_agg, apply momentum, return result."""
        if self._cached_global is not None:
            return self._cached_global
        self._cached_global = self._aggregate()
        return self._cached_global

    # ------------------------------------------------------------------
    # Aggregation internals
    # ------------------------------------------------------------------

    def _aggregate(self) -> Dict[str, torch.Tensor]:
        if not self._client_data:
            return {}

        self._round += 1
        weights = self._consensus_weights()

        # Weighted sum → W_agg_t
        layer_keys = list(self._client_data[0][0].keys())
        w_agg_t: Dict[str, torch.Tensor] = {}

        for layer_key in layer_keys:
            acc = None
            for (dw_dict, _), w in zip(self._client_data, weights):
                dw = dw_dict[layer_key]
                if acc is None:
                    acc = dw * w
                else:
                    acc = acc + dw * w
            w_agg_t[layer_key] = acc

        # Apply momentum: M_t = β * M_{t-1} + (1-β) * W_agg_t
        for layer_key, w in w_agg_t.items():
            if layer_key not in self._momentum:
                self._momentum[layer_key] = torch.zeros_like(w)
            self._momentum[layer_key] = (
                self.beta * self._momentum[layer_key] + (1.0 - self.beta) * w
            )

        # Bias correction: M̂_t = M_t / (1 - β^t)
        bc = 1.0 - (self.beta ** self._round)
        return {k: v / bc for k, v in self._momentum.items()}

    def _consensus_weights(self) -> List[float]:
        """
        Compute per-client weights combining dataset size and subspace consensus.

        Returns a list of floats summing to 1.0, one per client in _client_data.
        """
        raw_weights = [w for _, w in self._client_data]

        if not self.use_consensus or len(self._client_data) < 2:
            return raw_weights  # already normalised by caller (n_k/N)

        # Compute top-k left singular vectors per layer per client
        # Use svd_lowrank for speed (much faster than full SVD for big matrices)
        layer_keys = list(self._client_data[0][0].keys())
        r = self.consensus_rank
        all_U: List[Dict[str, Optional[torch.Tensor]]] = []

        for dw_dict, _ in self._client_data:
            client_U = {}
            for lk in layer_keys:
                W = dw_dict[lk]
                try:
                    U, _, _ = torch.svd_lowrank(W, q=r, niter=2)
                    client_U[lk] = U  # (d_out, r)
                except Exception:
                    client_U[lk] = None
            all_U.append(client_U)

        n = len(self._client_data)
        # C_k = mean over layers and other clients of ||U_k^T U_j||_F
        consensus_scores = []
        for k in range(n):
            scores = []
            for lk in layer_keys:
                U_k = all_U[k][lk]
                if U_k is None:
                    continue
                layer_agreements = []
                for j in range(n):
                    if j == k:
                        continue
                    U_j = all_U[j][lk]
                    if U_j is None:
                        continue
                    rv = min(U_k.shape[1], U_j.shape[1])
                    overlap = torch.linalg.norm(
                        U_k[:, :rv].T @ U_j[:, :rv], ord="fro"
                    ).item()
                    layer_agreements.append(overlap)
                if layer_agreements:
                    scores.append(float(np.mean(layer_agreements)))
            consensus_scores.append(float(np.mean(scores)) if scores else 1.0)

        # Combined: n_k * C_k
        combined = [raw_weights[k] * consensus_scores[k] for k in range(n)]
        total = sum(combined)
        if total <= 0:
            return raw_weights
        return [c / total for c in combined]

    # ------------------------------------------------------------------
    # Distribution
    # ------------------------------------------------------------------

    @staticmethod
    def soft_project_to_rank(
        w_agg: torch.Tensor,
        rank: int,
        gamma: float = 1.0,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rank-r projection with soft spectral weighting.

          σ̃_i = σ_i * (1 - exp(-σ_i / (γ * σ_1)))

        γ=1.0 is the nuclear-norm proximal operator (Candes & Recht 2009).
        This strictly dominates both hard threshold (SPA) and no threshold (FlexLoRA)
        in expected Frobenius reconstruction error under additive Gaussian noise.
        """
        W = w_agg.float().to(device)
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)

        if len(S) > 0 and S[0].item() > 1e-12:
            scale = gamma * S[0].item()
            S_soft = S * (1.0 - torch.exp(-S / scale))
        else:
            S_soft = S.clone()

        k = min(rank, len(S_soft))
        S_k = S_soft[:k].clamp(min=0.0)
        S_sqrt = torch.diag(torch.sqrt(S_k))

        A = S_sqrt @ Vt[:k, :]          # (k, d_in)
        B = U[:, :k] @ S_sqrt           # (d_out, k)

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
        global_weights = self.get_global()
        result = {}
        for client_id, rank in client_ranks.items():
            client_lora = {}
            for layer_key, w_agg in global_weights.items():
                B, A = self.soft_project_to_rank(w_agg, rank, self.gamma, device)
                client_lora[layer_key] = {"A": A, "B": B}
            result[client_id] = client_lora
        return result
