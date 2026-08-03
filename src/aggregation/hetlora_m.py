"""
HetLoRA-M: HetLoRA + evaluation-side EMA in (B, A) space.

Extends HetLoRA (Cho et al., EMNLP 2024) with server-side EMA momentum
that is applied ONLY to the evaluated (deployed) model. Clients always
receive the raw Frobenius-weighted aggregate, not the EMA state.

Algorithm each round:
  1. Compute Frobenius-weighted average (same as HetLoRA):
       B_raw = Σ_k p_k · pad(B_k),  A_raw = Σ_k p_k · pad(A_k)
       where p_k ∝ ||B_k A_k||_F

  2. Apply EMA in (B, A) space (server-side only, never sent to clients):
       B̄^(t) = β · B̄^(t-1) + (1−β) · B_raw^(t)
       Ā^(t) = β · Ā^(t-1) + (1−β) · A_raw^(t)

  3. Client distribution uses the RAW aggregate (same as HetLoRA):
       Client k receives: B_raw[:, :r_k], A_raw[:r_k, :]

  4. Evaluation uses the bias-corrected EMA state:
       B̄_bc = B̄^(t) / (1 − β^t),  Ā_bc likewise

Why this avoids SPA-M's feedback loop:
  SPA-M distributes the momentum state to clients, who train relative
  to it. Their uploads are deviations from momentum → EMA accumulates
  deviation noise that grows as training converges.

  HetLoRA-M distributes only the raw aggregate. Client training dynamics
  are identical to HetLoRA — the momentum state never contaminates
  client initialization. The EMA smooths a convergent sequence of raw
  aggregates at the server and is used only at evaluation time.

API:
  get_global_ba() → bias-corrected EMA state  (for evaluation)
  get_raw_ba()    → last raw aggregate         (for client distribution)
  Call get_global_ba() first each round; get_raw_ba() returns the cached raw.
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
        Compute raw Frobenius-weighted (B, A), update EMA, return bias-corrected EMA.
        Raw aggregate is cached in self._last_raw_ba for get_raw_ba().
        Use get_raw_ba() for client distribution; this return value for evaluation.
        """
        raw_ba = super().get_global_ba()
        self._last_raw_ba = raw_ba   # cache for get_raw_ba()
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

    def get_raw_ba(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Return the raw Frobenius-weighted aggregate from the current round.
        Must be called after get_global_ba() — returns the cached raw (no EMA).
        This is what clients receive so their training trajectory is identical to HetLoRA.
        """
        return getattr(self, "_last_raw_ba", {})
