"""Sealed-noise release for training runs: clip the whole client update, add Gaussian noise.

This is the arithmetic of the FedGT sealed-noise box (FedGT/fedgt/sealed_noise.py)
in torch, for measuring model accuracy at a given privacy level. The protocol
parts of the box (attestation, signatures, encryption, nonces) do not change
what the server receives, so they are left out here; pass impl="box" to route
every release through the full FedGT box instead (slower, CPU only).

Privacy accounting: each release is a Gaussian mechanism with replace-one
sensitivity 2C on the client's whole update (all modules clipped jointly).
mu-GDP composes exactly: k releases of mu give sqrt(k) mu (Dong, Roth and Su).
eps is reported per client from its number of releases, with no credit taken
for client sampling.
"""
import math
import os
from collections import defaultdict
from typing import Dict, Optional

import torch
from scipy.optimize import brentq
from scipy.stats import norm


def _delta(eps, mu):
    return norm.cdf(mu / 2 - eps / mu) - math.exp(eps) * norm.cdf(-mu / 2 - eps / mu)


def eps_of_mu(mu: float, delta: float = 1e-5) -> float:
    if mu <= 0:
        return 0.0
    if _delta(0.0, mu) <= delta:
        return 0.0
    hi = 1.0
    while _delta(hi, mu) > delta:
        hi *= 2.0
    return brentq(lambda e: _delta(e, mu) - delta, 0.0, hi)


def sigma_of_eps(eps: float, sensitivity: float, delta: float = 1e-5) -> float:
    f = lambda s: _delta(eps, sensitivity / s) - delta
    lo, hi = 1e-12 * sensitivity, sensitivity
    while f(hi) > 0:
        hi *= 2.0
    return brentq(f, lo, hi)


class SealedRelease:
    """Clip-and-noise every client release; one release per client per round."""

    def __init__(self, clip: float, eps_per_release: Optional[float] = None,
                 delta: float = 1e-5, impl: str = "torch", shapes=None):
        self.clip = float(clip)
        self.delta = delta
        self.eps_per_release = eps_per_release
        # eps_per_release=None means clipping only (no noise): separates the cost of clipping
        self.sigma = 0.0 if eps_per_release is None else sigma_of_eps(eps_per_release, 2 * self.clip, delta)
        self.mu = math.inf if self.sigma == 0 else 2 * self.clip / self.sigma
        self.impl = impl
        self.releases = defaultdict(int)
        self.last_round: Dict[int, int] = {}
        self.log = []
        self._box = None

    def release(self, cid: int, rnd: int, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if rnd <= self.last_round.get(cid, 0):
            raise RuntimeError(f"client {cid}: second release in round {rnd}")
        self.last_round[cid] = rnd
        self.releases[cid] += 1
        keys = sorted(weights)
        norm2 = sum(float(weights[k].double().pow(2).sum()) for k in keys)
        n = math.sqrt(norm2)
        scale = min(1.0, self.clip / n) if n > 0 else 1.0
        if self.impl == "box" and self.sigma > 0:
            out = self._box_release(cid, rnd, weights, keys)
        else:
            gen = torch.Generator().manual_seed(int.from_bytes(os.urandom(8), "big"))
            out = {}
            for k in keys:
                w = weights[k].float() * scale
                if self.sigma > 0:
                    w = w + torch.randn(w.shape, generator=gen, dtype=torch.float32) * self.sigma
                out[k] = w
        self.log.append({"client": cid, "round": rnd, "norm": n, "clipped": scale < 1.0,
                         "eps_client": self.epsilon(cid)})
        return out

    def epsilon(self, cid: int) -> float:
        if self.sigma == 0:
            return math.inf
        return eps_of_mu(math.sqrt(self.releases[cid]) * self.mu, self.delta)

    # ------------------------------------------------------------------ full FedGT box
    def _box_release(self, cid, rnd, weights, keys):
        import sys, pathlib
        fedgt_root = pathlib.Path(__file__).resolve().parents[3] / "FedGT"
        sys.path.insert(0, str(fedgt_root))
        from fedgt.sealed_noise import BoxConfig, SealedNoiseBox, Server, Vendor, measurement_of
        if self._box is None:
            shapes = tuple((k, tuple(weights[k].shape)) for k in keys)
            cfg = BoxConfig(clip=self.clip, sigma=self.sigma, delta=self.delta, eps_budget=1e9, shapes=shapes)
            vendor = Vendor()
            server = Server(vendor.public_key, measurement_of(SealedNoiseBox), cfg)
            self._box = {"cfg": cfg, "vendor": vendor, "server": server, "boxes": {}}
        st = self._box
        if cid not in st["boxes"]:
            b = SealedNoiseBox(st["cfg"], cid, st["vendor"], st["server"].sign_pk, st["server"].enc_pk)
            st["server"].enrol(cid, b.certificate)
            st["boxes"][cid] = b
        ct = st["boxes"][cid].release(st["server"].nonce(cid, rnd), {k: weights[k].float().numpy() for k in keys})
        out = st["server"].receive(cid, ct)
        return {k: torch.from_numpy(out[k].copy()) for k in keys}


def shrink_aggregate(wagg: Dict[str, torch.Tensor], sigma_agg: float, max_rank: int = 64,
                     device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Server-side denoising of a noisy aggregate (post-processing: no privacy cost).

    Each module of the aggregate is signal + i.i.d. Gaussian noise with known
    per-entry std sigma_agg. Singular values of pure noise end at the
    Marchenko-Pastur edge sigma_agg (sqrt(d1) + sqrt(d2)); the Gavish-Donoho
    Frobenius-optimal shrinker sets everything below the edge to zero and
    shrinks what is above it to undo the noise inflation.
    """
    out = {}
    for k, W in wagg.items():
        d1, d2 = W.shape
        n, m = max(d1, d2), min(d1, d2)
        beta = m / n
        scale = sigma_agg * math.sqrt(n)
        X = W.to(device=device, dtype=torch.float32)
        q = min(max_rank, m)
        U, S, V = torch.svd_lowrank(X, q=q, niter=4)
        y = S / scale
        edge = 1 + math.sqrt(beta)
        keep = y > edge
        eta = torch.zeros_like(y)
        yk = y[keep]
        eta[keep] = torch.sqrt(torch.clamp((yk ** 2 - beta - 1) ** 2 - 4 * beta, min=0.0)) / yk
        s_hat = eta * scale
        out[k] = ((U * s_hat) @ V.T).to(W.dtype).cpu()
    return out
