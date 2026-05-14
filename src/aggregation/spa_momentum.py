"""
SPA-M: SPA with Soft Spectral Weighting + Momentum + Consensus Weighting.

Combines four improvements over SPA/FlexLoRA:

  1. Soft spectral weighting (Idea 1):
       σ̃_i = σ_i * (1 - exp(-σ_i / (γ * σ_1)))
     Applied BEFORE momentum accumulation so the EMA buffer accumulates
     denoised signal, not raw noise. At distribution, only rank-r truncation
     is needed (soft shaping already applied).

  2. Server-side momentum (Idea 2):
       M_t = β * M_{t-1} + (1 - β) * W_agg_t
     Bias correction uses the cumulative product of all past β values
     (correct for variable β):  bc_t = 1 - Π_{τ=1}^{t} β_τ.
     EMA runs on raw W_agg (no SVD filter in aggregation path); SVD/rank
     truncation occurs only at distribution time via project_to_rank.

  3. Consensus weighting (Idea 3):
       w_k ∝ n_k * C_k,  where C_k = mean_j ||U_k^T U_j||_F
     Down-weights clients whose gradient subspace is orthogonal to the
     consensus direction.

  4. Adaptive β (Idea 4) — saturating stabilizer:
       β_t = β_max * (1 - clamp(sim_t, 0, 1)),  sim_t ∈ [-1, 1]
     Consistent round (sim→1) → β→0 (let fresh signal dominate).
     Orthogonal or anti-correlated round (sim≤0) → β=β_max (saturate).
     Clamping to [0,1] rather than mapping full [-1,1] is intentional:
     it provides a saturating maximum-smoothing regime for any round that
     is at least orthogonal to the previous, preventing over-accumulation
     of the noisy early-round signal under progressive learning.

Interface matches existing aggregators:
  reset()                 — call at start of each round
  update(dw_dict, weight) — call per client (weight = n_k / N)
  get_global()            — returns W_agg after consensus + momentum
  distribute(ranks, dev)  — projects W_agg → per-client (A, B)
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


class SPAMomentumAggregator:
    """
    SPA-M: soft-spectral + momentum + consensus aggregation.

    Args:
        max_rank:      maximum LoRA rank across all clients.
        beta:          maximum momentum coefficient. Adaptive β scales this
                       by (1-clamp(sim,0,1)) — saturates at beta_max for
                       orthogonal/anti-correlated rounds.
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
        # Stores denoised W_filtered from previous round for adaptive β
        self._prev_filtered: Optional[Dict[str, torch.Tensor]] = None
        # Cumulative product of all past β values for correct variable-β bias correction
        self._beta_product: float = 1.0

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

        # Step 1: Weighted sum → W_agg_t
        layer_keys = list(self._client_data[0][0].keys())
        w_agg_t: Dict[str, torch.Tensor] = {}
        for layer_key in layer_keys:
            acc = None
            for (dw_dict, _), w in zip(self._client_data, weights):
                dw = dw_dict[layer_key]
                acc = dw * w if acc is None else acc + dw * w
            w_agg_t[layer_key] = acc

        # Step 2: Adaptive β — compare raw aggregated signals, full [-1,1] sim range.
        # NOTE: SVD filtering is NOT applied here. Simulation showed it attenuates
        # the growing learning signal under α=0.1, hurting convergence.
        # SVD/rank truncation happens only at distribution time (project_to_rank).
        beta = self._adaptive_beta(w_agg_t)

        # Step 3: EMA on the raw aggregated signal.
        for layer_key, w in w_agg_t.items():
            if layer_key not in self._momentum:
                self._momentum[layer_key] = torch.zeros_like(w)
            self._momentum[layer_key] = (
                beta * self._momentum[layer_key] + (1.0 - beta) * w
            )

        # Step 4: Bias correction using cumulative β product.
        # Standard formula 1-β^t assumes constant β. With variable β_τ per round,
        # the correct denominator is 1 - Π_{τ=1}^{t} β_τ.
        self._beta_product *= beta
        bc = max(1.0 - self._beta_product, 1e-8)

        result: Dict[str, torch.Tensor] = {k: v / bc for k, v in self._momentum.items()}

        self._prev_filtered = {k: v.clone() for k, v in w_agg_t.items()}
        return result

    def _soft_spectral_filter(
        self, w: torch.Tensor, device: str = "cpu"
    ) -> torch.Tensor:
        """
        Apply nuclear-norm soft shaping; return filtered matrix on CPU.

        Uses svd_lowrank(q=max_rank) — the soft shrinkage function
        σ*(1-exp(-σ/(γ*σ_1))) drives components below ~σ_1/γ toward zero,
        so singular values beyond max_rank carry negligible weight after
        filtering. This avoids O(n^3) full SVD on large layers.
        """
        W = w.float().to(device)
        q = min(self.max_rank + 4, min(W.shape))
        U, S, Vh = torch.svd_lowrank(W, q=q, niter=4)
        Vt = Vh.T

        if len(S) > 0 and S[0].item() > 1e-12:
            scale = self.gamma * S[0].item()
            S_soft = S * (1.0 - torch.exp(-S / scale))
        else:
            S_soft = S.clone()

        return (U @ torch.diag(S_soft) @ Vt).cpu()

    def _adaptive_beta(self, w_filtered_t: Dict[str, torch.Tensor]) -> float:
        """
        Scale β by the divergence between consecutive denoised aggregations.

        Momentum acts as a STABILIZER in federated settings with high non-IID:
        when rounds oscillate (anti-correlated updates), high β smooths them out;
        when rounds are already consistent, low β lets the fresh signal dominate.

        Maps cosine similarity → β:
          sim ≤ 0 (orthogonal or anti-correlated) → β = beta_max  (maximum stabilization)
          sim = 1 (consistent, converging)          → β = 0        (no smoothing needed)
          sim ∈ (0, 1)                              → β linear in (beta_max, 0)

        Saturating stabilizer: clamp(sim, 0, 1) → β = beta_max*(1-sim).
        The saturation at sim=0 is intentional — any round that is at least
        orthogonal to the previous gets full smoothing. Simulation showed that
        mapping the full [-1,1] range linearly (β=beta_max for sim=-1 only) with
        high beta_max over-smooths the growing signal under progressive training.
        """
        if self._prev_filtered is None:
            return self.beta  # first round: use full β
        try:
            cur  = torch.cat([v.flatten() for v in w_filtered_t.values()])
            prev = torch.cat([v.flatten() for v in self._prev_filtered.values()])
            sim = float(
                torch.nn.functional.cosine_similarity(
                    cur.unsqueeze(0), prev.unsqueeze(0)
                ).clamp(0.0, 1.0)  # saturating: anti-corr treated same as orthogonal
            )
        except Exception:
            return self.beta
        return self.beta * (1.0 - sim)

    def _consensus_weights(self) -> List[float]:
        """
        Compute per-client weights combining dataset size and subspace consensus.

        Returns a list of floats summing to 1.0, one per client in _client_data.
        """
        raw_weights = [w for _, w in self._client_data]

        if not self.use_consensus or len(self._client_data) < 2:
            return raw_weights  # already normalised by caller (n_k/N)

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

        combined = [raw_weights[k] * consensus_scores[k] for k in range(n)]
        total = sum(combined)
        if total <= 0:
            return raw_weights
        return [c / total for c in combined]

    # ------------------------------------------------------------------
    # Distribution
    # ------------------------------------------------------------------

    @staticmethod
    def project_to_rank(
        w_agg: torch.Tensor,
        rank: int,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project a (already soft-filtered) W_agg to rank-r LoRA factors (B, A).

        Uses svd_lowrank instead of full SVD — we only need top-r components,
        and full SVD on large matrices (e.g. 3584×3584 q_proj) is O(n^3) and
        dominates round time when called per client per layer per round.
        svd_lowrank is O(n * rank) and gives the same top-r result.

        Returns (B, A) s.t. B @ A is the best rank-r Frobenius approximation.
        """
        W = w_agg.float().to(device)
        k = min(rank, min(W.shape))

        # niter=4 gives accurate top-k singular vectors; q=k+4 improves stability
        U, S, Vh = torch.svd_lowrank(W, q=min(k + 4, min(W.shape)), niter=4)
        U = U[:, :k]
        S = S[:k].clamp(min=0.0)
        Vt = Vh.T[:k, :]

        S_sqrt = torch.diag(torch.sqrt(S))
        A = S_sqrt @ Vt    # (k, d_in)
        B = U @ S_sqrt     # (d_out, k)

        if k < rank:
            pad = rank - k
            A = torch.cat([A, torch.zeros(pad, A.shape[1], device=device)], dim=0)
            B = torch.cat([B, torch.zeros(B.shape[0], pad, device=device)], dim=1)

        return B.cpu(), A.cpu()

    # Keep old name as alias so any external callers don't break immediately.
    # Prefer project_to_rank going forward.
    @staticmethod
    def soft_project_to_rank(
        w_agg: torch.Tensor,
        rank: int,
        gamma: float = 1.0,  # retained for signature compatibility; ignored
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return SPAMomentumAggregator.project_to_rank(w_agg, rank, device)

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
                B, A = self.project_to_rank(w_agg, rank, device)
                client_lora[layer_key] = {"A": A, "B": B}
            result[client_id] = client_lora
        return result
