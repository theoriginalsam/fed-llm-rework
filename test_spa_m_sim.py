"""
SPA-M Simulation — α=0.1 dynamics with progressive learning signal.

Simulates 20 FL rounds. W_agg is NOT pure noise — it has a weak but
growing signal component toward a "true" direction, plus heavy client-bias
noise (α=0.1 non-IID). This matches real FL: the model learns slowly but
individual round aggregations are dominated by whichever 5 clients were sampled.

Compares three aggregator versions side by side.
Runtime: ~2 seconds on CPU. No model, no GPU.

Usage: python test_spa_m_sim.py
"""

import sys, os, types
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.aggregation.spa_momentum import SPAMomentumAggregator

torch.manual_seed(0)
np.random.seed(0)

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
D_OUT, D_IN = 64, 32
NUM_CLIENTS = 50
PER_ROUND   = 5
NUM_ROUNDS  = 20
NUM_CLASSES = 5
ALPHA       = 0.1

# True signal: the direction the global model should converge to
TRUE = torch.randn(D_OUT, D_IN)
TRUE = TRUE / TRUE.norm()

# Class-specific directions (each client is biased toward one class)
torch.manual_seed(1)
CLASS_DIRS = [torch.randn(D_OUT, D_IN) for _ in range(NUM_CLASSES)]
CLASS_DIRS = [d / d.norm() for d in CLASS_DIRS]

# Dirichlet partition — fixed across all simulations for fair comparison
DIRICHLET = np.random.dirichlet([ALPHA] * NUM_CLASSES, size=NUM_CLIENTS)

def client_wagg(selected: list, learning_progress: float) -> torch.Tensor:
    """
    Simulate server's W_agg for this round.

    learning_progress ∈ [0,1]: how far through training we are.
    As progress increases, client updates gain a weak component toward TRUE
    (the model is learning). The dominant noise is still the client-specific
    class bias (α=0.1 non-IID).
    """
    acc = None
    for cid in selected:
        dist = DIRICHLET[cid]
        dominant = int(np.argmax(dist))

        # Client-specific class gradient (the non-IID noise)
        g_class = sum(CLASS_DIRS[c] * dist[c] for c in range(NUM_CLASSES))

        # Weak global signal that grows as training progresses
        g_signal = TRUE * learning_progress * 0.4

        # Small iid noise
        g_noise = torch.randn(D_OUT, D_IN) * 0.1

        g = g_class + g_signal + g_noise
        acc = g if acc is None else acc + g

    w_agg = acc / len(selected)
    return w_agg


# ---------------------------------------------------------------------------
# Aggregator factories
# ---------------------------------------------------------------------------

def make_v1():
    """V1: clamp [0,1] + β^t bias correction (original bugs)."""
    agg = SPAMomentumAggregator(max_rank=8, beta=0.5, gamma=1.0, use_consensus=False)

    def _adaptive_beta(self, w_filtered_t):
        if self._prev_filtered is None:
            return self.beta
        cur  = torch.cat([v.flatten() for v in w_filtered_t.values()])
        prev = torch.cat([v.flatten() for v in self._prev_filtered.values()])
        sim  = float(F.cosine_similarity(cur.unsqueeze(0), prev.unsqueeze(0)).clamp(0., 1.))
        return self.beta * (1.0 - sim)

    def _aggregate(self):
        if not self._client_data:
            return {}
        self._round += 1
        w_agg_t = {}
        weights = [w for _, w in self._client_data]
        for lk in self._client_data[0][0]:
            acc = None
            for (d, _), w in zip(self._client_data, weights):
                acc = d[lk] * w if acc is None else acc + d[lk] * w
            w_agg_t[lk] = acc
        beta = self._adaptive_beta(w_agg_t)
        for lk, w in w_agg_t.items():
            if lk not in self._momentum:
                self._momentum[lk] = torch.zeros_like(w)
            self._momentum[lk] = beta * self._momentum[lk] + (1 - beta) * w
        bc = max(1.0 - beta ** self._round, 1e-8)          # BUG: β^t, variable β
        self._prev_filtered = {k: v.clone() for k, v in w_agg_t.items()}
        return {k: v / bc for k, v in self._momentum.items()}

    agg._adaptive_beta = types.MethodType(_adaptive_beta, agg)
    agg._aggregate     = types.MethodType(_aggregate, agg)
    return agg


def make_v2_broken():
    """V2-broken: accelerator β (sim+1)/2 — caused the regression."""
    agg = SPAMomentumAggregator(max_rank=8, beta=0.5, gamma=1.0, use_consensus=False)

    def _adaptive_beta(self, w_filtered_t):
        if self._prev_filtered is None:
            return self.beta
        cur  = torch.cat([v.flatten() for v in w_filtered_t.values()])
        prev = torch.cat([v.flatten() for v in self._prev_filtered.values()])
        sim  = float(F.cosine_similarity(cur.unsqueeze(0), prev.unsqueeze(0)).clamp(-1., 1.))
        return self.beta * (sim + 1.0) / 2.0   # BUG: accelerator, β→0 when divergent

    agg._adaptive_beta = types.MethodType(_adaptive_beta, agg)
    return agg


def make_v21():
    """V2.1 (superseded): stabilizer β, cumulative bc, SVD filter + mag-norm in _aggregate.
    Kept for historical comparison — V2.2 showed SVD filter attenuates learning signal.
    Monkey-patches _aggregate to restore the old V2.1 behaviour.
    """
    agg = SPAMomentumAggregator(max_rank=8, beta=0.9, gamma=1.0, use_consensus=False)

    def _aggregate(self):
        if not self._client_data:
            return {}
        self._round += 1
        weights = [w for _, w in self._client_data]
        layer_keys = list(self._client_data[0][0].keys())
        w_agg_t = {}
        for lk in layer_keys:
            acc = None
            for (d, _), w in zip(self._client_data, weights):
                acc = d[lk] * w if acc is None else acc + d[lk] * w
            w_agg_t[lk] = acc
        device = "cuda" if torch.cuda.is_available() else "cpu"
        w_filtered_t = {lk: self._soft_spectral_filter(w, device) for lk, w in w_agg_t.items()}
        beta = self._adaptive_beta(w_filtered_t)
        for lk, w in w_filtered_t.items():
            if lk not in self._momentum:
                self._momentum[lk] = torch.zeros_like(w)
            self._momentum[lk] = beta * self._momentum[lk] + (1.0 - beta) * w
        self._beta_product *= beta
        bc = max(1.0 - self._beta_product, 1e-8)
        result = {}
        for k, v in self._momentum.items():
            bc_v = v / bc
            w_norm = torch.linalg.norm(w_agg_t[k].float())
            m_norm = torch.linalg.norm(bc_v.float())
            if m_norm > 1e-8 and w_norm > 1e-8:
                bc_v = bc_v * (w_norm / m_norm)
            result[k] = bc_v
        self._prev_filtered = {k: v.clone() for k, v in w_filtered_t.items()}
        return result

    agg._aggregate = types.MethodType(_aggregate, agg)
    return agg


def make_v27():
    """V2.7 (CURRENT CODE): saturating stabilizer clamp[0,1], cumulative bc, beta=0.9.
    No SVD filter or mag-norm in _aggregate. No monkey-patching needed.
    """
    return SPAMomentumAggregator(max_rank=8, beta=0.9, gamma=1.0, use_consensus=False)


make_v22 = make_v27  # alias for backward compat in run() calls


def make_v23():
    """V2.3: raw W_agg + stabilizer β + fixed bias correction + magnitude normalization only.

    Isolates whether magnitude normalization alone causes the regression vs V1.
    """
    agg = SPAMomentumAggregator(max_rank=8, beta=0.9, gamma=1.0, use_consensus=False)

    def _aggregate(self):
        if not self._client_data:
            return {}
        self._round += 1
        weights = [w for _, w in self._client_data]

        w_agg_t = {}
        for lk in self._client_data[0][0]:
            acc = None
            for (d, _), w in zip(self._client_data, weights):
                acc = d[lk] * w if acc is None else acc + d[lk] * w
            w_agg_t[lk] = acc

        beta = self._adaptive_beta(w_agg_t)

        for lk, w in w_agg_t.items():
            if lk not in self._momentum:
                self._momentum[lk] = torch.zeros_like(w)
            self._momentum[lk] = beta * self._momentum[lk] + (1.0 - beta) * w

        self._beta_product *= beta
        bc = max(1.0 - self._beta_product, 1e-8)

        result = {}
        for k, v in self._momentum.items():
            bc_v = v / bc
            w_norm = torch.linalg.norm(w_agg_t[k].float())
            m_norm = torch.linalg.norm(bc_v.float())
            if m_norm > 1e-8 and w_norm > 1e-8:
                bc_v = bc_v * (w_norm / m_norm)
            result[k] = bc_v

        self._prev_filtered = {k: v.clone() for k, v in w_agg_t.items()}
        return result

    agg._aggregate = types.MethodType(_aggregate, agg)
    return agg


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(agg, label: str, seeds: list = [0, 1, 2, 3, 4]):
    """Run multiple seeds, return (mean_cos_per_round, std_cos_per_round)."""
    all_sims = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        # Reset aggregator state between seeds
        agg._momentum.clear()
        agg._round = 0
        agg._cached_global = None
        agg._prev_filtered = None
        if hasattr(agg, '_beta_product'):
            agg._beta_product = 1.0

        sims = []
        for rnd in range(1, NUM_ROUNDS + 1):
            selected = rng.choice(NUM_CLIENTS, PER_ROUND, replace=False).tolist()
            progress = rnd / NUM_ROUNDS          # 0→1 as training progresses
            w_agg = client_wagg(selected, progress)

            agg.reset()
            agg.update({"layer0": w_agg}, weight=1.0)
            output = agg.get_global()["layer0"]

            cos = float(F.cosine_similarity(
                output.flatten().unsqueeze(0),
                TRUE.flatten().unsqueeze(0),
            ))
            sims.append(cos)
        all_sims.append(sims)

    arr = np.array(all_sims)          # (n_seeds, n_rounds)
    return arr.mean(axis=0), arr.std(axis=0), arr


def print_results(label, mean_sims, std_sims, all_sims):
    l5    = float(np.mean(mean_sims[-5:]))
    final = float(mean_sims[-1])
    seed_final_std = float(np.std(all_sims[:, -1]))   # variance across seeds at round 20

    print(f"\n{label}")
    print(f"  {'Rnd':>4}  {'Mean CosSim':>12}  {'±std':>8}")
    for i, (m, s) in enumerate(zip(mean_sims, std_sims)):
        bar = "█" * max(0, int(m * 40 + 20))
        print(f"  {i+1:>4}  {m:>12.4f}  {s:>8.4f}  {bar}")
    print(f"  Mean-L5 : {l5:.4f}")
    print(f"  Final   : {final:.4f}  (seed variance ±{seed_final_std:.4f})")


def main():
    print("=" * 65)
    print("SPA-M Simulation — α=0.1 with progressive learning signal")
    print("5 seeds × 20 rounds. Metric: cosine sim with true direction.")
    print("=" * 65)

    configs = [
        ("V1   (broken: clamp[0,1] + β^t bc, β=0.5)",      make_v1()),
        ("V2   (accelerator β — regression)",               make_v2_broken()),
        ("V2.1 (old stab[-1,1] + cumul-bc + SVD + mag)",   make_v21()),
        ("V2.7 (clamp[0,1] + cumul-bc, β=0.9) ← CURRENT", make_v27()),
    ]

    results = {}
    for label, agg in configs:
        mean_s, std_s, all_s = run(agg, label)
        results[label] = (mean_s, std_s, all_s)
        print_results(label, mean_s, std_s, all_s)

    print("\n" + "=" * 65)
    print(f"  {'Version':<45}  {'Mean-L5':>8}  {'Final':>8}  {'SeedVar':>8}")
    for label, (mean_s, std_s, all_s) in results.items():
        l5  = float(np.mean(mean_s[-5:]))
        fin = float(mean_s[-1])
        sv  = float(np.std(all_s[:, -1]))
        print(f"  {label:<45}  {l5:>8.4f}  {fin:>8.4f}  {sv:>8.4f}")
    print("=" * 65)
    print("\nInterpretation:")
    print("  Higher Mean-L5/Final = better convergence toward true signal")
    print("  Lower SeedVar = more reproducible across seeds")


if __name__ == "__main__":
    main()
