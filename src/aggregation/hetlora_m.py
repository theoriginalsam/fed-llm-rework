"""
HetLoRA-M: HetLoRA + EMA Momentum on (B, A) aggregation.

Extends HetLoRA (Cho et al., EMNLP 2024) with server-side EMA momentum
applied directly to the aggregated (B̄, Ā) matrices at max_rank.

Algorithm each round:
  1. Compute Frobenius-weighted average (same as HetLoRA):
       B_raw = Σ_k p_k · pad(B_k),  A_raw = Σ_k p_k · pad(A_k)
       where p_k ∝ ||B_k A_k||_F

  2. Apply EMA in (B, A) space:
       B̄^(t) = β · B̄^(t-1) + (1−β) · B_raw^(t)
       Ā^(t) = β · Ā^(t-1) + (1−β) · A_raw^(t)

  3. Bias-correct and distribute via truncation (same as HetLoRA):
       B̄_bc = B̄^(t) / (1 − β^t)
       Client k receives: B̄_bc[:, :r_k], Ā_bc[:r_k, :]

Why this avoids SPA-M's feedback loop:
  SPA-M applies momentum to ΔW, which is then fed back into client
  initialization. Under α=0.1, clients train *relative* to the momentum
  state → ΔW uploads are deviations, not absolute signals → EMA
  accumulates noise over near-orthogonal rounds.

  HetLoRA-M applies momentum to the aggregated (B̄, Ā) *after* the
  Frobenius-weighted sum. Clients initialize from truncated (B̄, Ā) the
  same way as vanilla HetLoRA — the momentum smooths the global matrices
  across rounds without introducing a relative-deviation feedback loop.
  Low-rank clients still see their natural subspace positions (slots 0:r_k)
  smoothed over time, not reorganized by SVD energy ordering.
"""

import torch
from typing import Dict, Optional

from src.aggregation.hetlora import HetLoRAAggregator


class HetLoRAMomentumAggregator(HetLoRAAggregator):
    """
    HetLoRA with EMA momentum on (B, A) aggregation buffers.
    Inherits Frobenius-weighted aggregation and truncation distribution.
    """

    def __init__(self, max_rank: int = 32, beta: float = 0.9):
        super().__init__(max_rank)
        self.beta = beta
        self._b_ema: Dict[str, torch.Tensor] = {}   # {layer: (d_out, max_rank)}
        self._a_ema: Dict[str, torch.Tensor] = {}   # {layer: (max_rank, d_in)}
        self._round: int = 0

    def get_global_ba(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute Frobenius-weighted raw (B, A), apply EMA, return bias-corrected result.
        """
        raw_ba = super().get_global_ba()
        if not raw_ba:
            return {}

        self._round += 1
        bc = 1.0 - self.beta ** self._round   # bias correction factor

        for layer_key, mats in raw_ba.items():
            B_raw = mats["B"]   # (d_out, max_rank), float
            A_raw = mats["A"]   # (max_rank, d_in), float

            if layer_key not in self._b_ema:
                self._b_ema[layer_key] = (1.0 - self.beta) * B_raw
                self._a_ema[layer_key] = (1.0 - self.beta) * A_raw
            else:
                self._b_ema[layer_key] = (
                    self.beta * self._b_ema[layer_key] + (1.0 - self.beta) * B_raw
                )
                self._a_ema[layer_key] = (
                    self.beta * self._a_ema[layer_key] + (1.0 - self.beta) * A_raw
                )

        return {
            k: {
                "B": self._b_ema[k] / bc,
                "A": self._a_ema[k] / bc,
            }
            for k in self._b_ema
        }
