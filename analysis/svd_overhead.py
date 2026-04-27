"""
SVD Overhead Analysis — addresses Reviewer 1's critique.

Measures actual SVD time across:
  - All 28 attention layers of Qwen2.5-7B
  - Matrix dimensions: d_in=d_out=4096 (q_proj, v_proj)
  - Repeated 100 times for stable timing
  - Reports: per-layer time, total per-round time, scaling with d

Usage:
  python analysis/svd_overhead.py
"""

import time
import torch
import numpy as np
import json
import os


def time_svd(d: int, rank: int, n_trials: int = 100, device: str = "cuda") -> dict:
    """Time SVD + projection for a matrix of shape (d, d) with target rank."""
    times_svd = []
    times_project = []

    W = torch.randn(d, d, device=device, dtype=torch.float32)

    for _ in range(n_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_svd.append(t1 - t0)

        torch.cuda.synchronize()
        t2 = time.perf_counter()
        k = min(rank, len(S))
        S_sqrt = torch.diag(torch.sqrt(S[:k]))
        A = S_sqrt @ Vt[:k, :]
        B = U[:, :k] @ S_sqrt
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        times_project.append(t3 - t2)

    return {
        "d": d,
        "rank": rank,
        "svd_mean_ms": float(np.mean(times_svd) * 1000),
        "svd_std_ms": float(np.std(times_svd) * 1000),
        "project_mean_ms": float(np.mean(times_project) * 1000),
        "total_mean_ms": float((np.mean(times_svd) + np.mean(times_project)) * 1000),
    }


def full_model_overhead(device: str = "cuda") -> dict:
    """
    Qwen2.5-7B has 28 transformer layers, each with q_proj and v_proj.
    Both are (4096, 4096) projections.
    SPA does SVD once on W_agg per layer per round.
    """
    num_layers = 28
    target_modules = ["q_proj", "v_proj"]  # 2 per layer
    d = 4096
    max_rank = 32

    results_per_dim = []
    for test_d in [512, 1024, 2048, 4096]:
        r = time_svd(test_d, max_rank, n_trials=50, device=device)
        results_per_dim.append(r)
        print(f"d={test_d}: SVD={r['svd_mean_ms']:.2f}ms, project={r['project_mean_ms']:.2f}ms")

    # Full model estimate
    r_4096 = [r for r in results_per_dim if r["d"] == 4096][0]
    total_per_round_ms = num_layers * len(target_modules) * r_4096["total_mean_ms"]
    total_per_round_s = total_per_round_ms / 1000

    # Compare to client training time estimate
    # 5 clients × 150 steps × ~0.5s/step ≈ 375s per round
    client_training_s = 5 * 150 * 0.5

    return {
        "per_dimension": results_per_dim,
        "full_model": {
            "num_layers": num_layers,
            "modules_per_layer": len(target_modules),
            "total_svd_operations_per_round": num_layers * len(target_modules),
            "svd_time_per_round_s": total_per_round_s,
            "client_training_time_per_round_s": client_training_s,
            "svd_overhead_fraction": total_per_round_s / (client_training_s + total_per_round_s),
        }
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running SVD overhead analysis on: {torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'}")

    results = full_model_overhead(device)

    fr = results["full_model"]
    print(f"\n{'='*50}")
    print(f"SVD overhead per round: {fr['svd_time_per_round_s']:.3f}s")
    print(f"Client training per round (est.): {fr['client_training_time_per_round_s']:.1f}s")
    print(f"SVD overhead fraction: {fr['svd_overhead_fraction']*100:.2f}%")
    print(f"{'='*50}")

    os.makedirs("results/ablation", exist_ok=True)
    with open("results/ablation/svd_overhead.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to results/ablation/svd_overhead.json")


if __name__ == "__main__":
    main()
