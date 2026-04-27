"""
FlexLoRA Baseline (Bai et al., 2024).
"FlexLoRA: Any-Dimension Low-Rank Adaptation for Federated Learning."

Algorithm is identical to SPA with tau=0.0 (no spectral threshold).
This class is an explicit re-implementation to match the paper's description
and to make the comparison fair and transparent.

The key algorithmic distinction vs SPA:
  - FlexLoRA: rank cutoff = hardware rank (fixed)
  - SPA: rank cutoff = min(hardware rank, components above tau*sigma_1)
"""

import torch
from typing import Dict, Tuple
from .spa import SPAAggregator


class FlexLoRAAggregator(SPAAggregator):
    """
    FlexLoRA = SPA with tau=0 (no adaptive spectral threshold).

    We inherit SPAAggregator and set tau=0 to make the relationship explicit.
    This ensures a fair head-to-head comparison where the only difference is
    the presence/absence of the spectral denoising threshold.
    """

    def __init__(self, max_rank: int = 32):
        super().__init__(max_rank=max_rank, tau=0.0)

    @staticmethod
    def project_to_rank(
        w_agg: torch.Tensor,
        rank: int,
        tau: float = 0.0,  # always 0 for FlexLoRA
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Identical to SPA.project_to_rank with tau forced to 0."""
        return SPAAggregator.project_to_rank(w_agg, rank, tau=0.0, device=device)
