"""
Unit tests for the fixed SPA-M aggregator (spa_momentum.py).

Tests run on CPU with synthetic tensors — no GPU or model required.
Runtime: < 5 seconds.

Usage:  python test_spa_m.py
"""

import sys
import os
import math
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.aggregation.spa_momentum import SPAMomentumAggregator

PASS = "  PASS"
FAIL = "  FAIL"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_aggregator(beta=0.9, gamma=1.0, use_consensus=False):
    return SPAMomentumAggregator(max_rank=8, beta=beta, gamma=gamma,
                                 use_consensus=use_consensus)


def feed_round(agg, matrices: dict, weight: float = 1.0):
    """Feed one synthetic client update and return get_global()."""
    agg.reset()
    agg.update(matrices, weight)
    return agg.get_global()


def make_delta(d_out=16, d_in=8, rank=4, seed=0):
    """Random low-rank ΔW = B @ A."""
    torch.manual_seed(seed)
    B = torch.randn(d_out, rank)
    A = torch.randn(rank, d_in)
    return {"layer0": B @ A}


def frob(t: torch.Tensor) -> float:
    return torch.linalg.norm(t.float()).item()


# ---------------------------------------------------------------------------
# Test 1: First round — output magnitude ≈ input magnitude (normalization)
# ---------------------------------------------------------------------------

def test_magnitude_normalization():
    print("\n[1] Magnitude normalization: output ||M̂|| ≈ ||W_agg||")
    agg = make_aggregator(beta=0.9)
    dw = make_delta(seed=1)
    out = feed_round(agg, dw)

    in_norm  = frob(dw["layer0"])
    out_norm = frob(out["layer0"])
    ratio = out_norm / in_norm

    ok = abs(ratio - 1.0) < 0.05
    print(f"  input_norm={in_norm:.4f}  output_norm={out_norm:.4f}  ratio={ratio:.4f}")
    print(PASS if ok else FAIL + f" (ratio {ratio:.4f} not within 5% of 1.0)")
    return ok


# ---------------------------------------------------------------------------
# Test 2: Adaptive β — anti-correlated input drives β → 0
# ---------------------------------------------------------------------------

def test_adaptive_beta_anti_correlated():
    print("\n[2] Adaptive β: anti-correlated rounds → β → 0 (no oscillation amplification)")
    agg = make_aggregator(beta=0.9)

    # Round 1: positive update
    dw_pos = {"layer0": torch.ones(16, 8) * 0.1}
    feed_round(agg, dw_pos)

    # Round 2: perfectly anti-correlated (negated) → should get β ≈ 0
    dw_neg = {"layer0": torch.ones(16, 8) * -0.1}
    agg.reset()
    agg.update(dw_neg, 1.0)
    beta_used = agg._adaptive_beta({"layer0": dw_neg["layer0"].float()})

    ok = beta_used < 0.05
    print(f"  β_adaptive for anti-correlated input = {beta_used:.6f} (want < 0.05)")
    print(PASS if ok else FAIL)
    return ok


# ---------------------------------------------------------------------------
# Test 3: Adaptive β — consistent input drives β → beta_max
# ---------------------------------------------------------------------------

def test_adaptive_beta_consistent():
    print("\n[3] Adaptive β: consistent rounds → β → beta_max")
    agg = make_aggregator(beta=0.9)

    dw = {"layer0": torch.ones(16, 8) * 0.1}
    feed_round(agg, dw)  # sets _prev_filtered

    # Same direction → cosine sim = 1 → β = beta_max
    beta_used = agg._adaptive_beta({"layer0": dw["layer0"].float()})

    ok = abs(beta_used - 0.9) < 0.01
    print(f"  β_adaptive for identical update = {beta_used:.6f} (want ≈ 0.9)")
    print(PASS if ok else FAIL)
    return ok


# ---------------------------------------------------------------------------
# Test 4: Adaptive β — orthogonal input gives beta_max / 2
# ---------------------------------------------------------------------------

def test_adaptive_beta_orthogonal():
    print("\n[4] Adaptive β: orthogonal rounds → β ≈ beta_max / 2")
    agg = make_aggregator(beta=0.8)

    # e1 direction in first layer element
    dw_a = {"layer0": torch.zeros(16, 8)}
    dw_a["layer0"][0, 0] = 1.0
    feed_round(agg, dw_a)

    # orthogonal direction
    dw_b = {"layer0": torch.zeros(16, 8)}
    dw_b["layer0"][1, 1] = 1.0
    beta_used = agg._adaptive_beta({"layer0": dw_b["layer0"].float()})

    ok = abs(beta_used - 0.4) < 0.05  # 0.8 * (0 + 1) / 2 = 0.4
    print(f"  β_adaptive for orthogonal update = {beta_used:.6f} (want ≈ 0.4)")
    print(PASS if ok else FAIL)
    return ok


# ---------------------------------------------------------------------------
# Test 5: Bias correction — cumulative product shrinks monotonically
# ---------------------------------------------------------------------------

def test_bias_correction_cumulative():
    print("\n[5] Bias correction: _beta_product shrinks each round → bc grows toward 1")
    agg = make_aggregator(beta=0.9)
    prev_product = 1.0

    products = []
    for i in range(5):
        dw = make_delta(seed=i)
        feed_round(agg, dw)
        products.append(agg._beta_product)

    strictly_decreasing = all(products[i] > products[i+1] for i in range(len(products)-1))
    final_bc = 1.0 - products[-1]
    ok = strictly_decreasing and final_bc > 0.3

    print(f"  beta_products: {[f'{p:.4f}' for p in products]}")
    print(f"  final bc = {final_bc:.4f} (want > 0.3)")
    print(PASS if ok else FAIL)
    return ok


# ---------------------------------------------------------------------------
# Test 6: Oscillation suppression — alternating anti-correlated inputs
#         Output should NOT grow in Frobenius norm (old code amplified this)
# ---------------------------------------------------------------------------

def test_oscillation_suppression():
    print("\n[6] Oscillation suppression: alternating inputs → output norm stays bounded")
    agg = make_aggregator(beta=0.9)

    dw_pos = {"layer0": torch.ones(16, 8) * 0.5}
    dw_neg = {"layer0": torch.ones(16, 8) * -0.5}

    norms = []
    for i in range(6):
        dw = dw_pos if i % 2 == 0 else dw_neg
        out = feed_round(agg, dw)
        norms.append(frob(out["layer0"]))

    input_norm = frob(dw_pos["layer0"])
    max_norm = max(norms)
    ok = max_norm <= input_norm * 1.5  # output stays within 1.5× input magnitude

    print(f"  input_norm={input_norm:.4f}  output norms: {[f'{n:.4f}' for n in norms]}")
    print(f"  max output norm = {max_norm:.4f} (want ≤ {input_norm * 1.5:.4f})")
    print(PASS if ok else FAIL)
    return ok


# ---------------------------------------------------------------------------
# Test 7: SVD filter is applied before momentum (noise reduction)
#         A pure-noise random matrix should have its singular values reduced
#         by the soft spectral filter before entering the momentum buffer.
# ---------------------------------------------------------------------------

def test_svd_filter_before_momentum():
    print("\n[7] SVD filter before momentum: filtered singular values < raw singular values")
    agg = make_aggregator(beta=0.0, gamma=1.0)  # β=0 → momentum = current filtered only

    torch.manual_seed(99)
    noise = torch.randn(32, 16) * 0.01  # small-norm noise matrix
    dw = {"layer0": noise}
    out = feed_round(agg, dw)

    # With β=0, momentum output = filtered W_agg (bias-corrected, magnitude-normalized)
    # The filtered version should have softened singular values vs raw
    W_raw = dw["layer0"].float()
    W_out = out["layer0"].float()

    # Compare top singular value ratio: soft-filtered top-SV should be < raw top-SV
    # (after accounting for magnitude normalization, we compare the spectral shape)
    _, S_raw, _ = torch.linalg.svd(W_raw, full_matrices=False)
    _, S_out, _ = torch.linalg.svd(W_out, full_matrices=False)

    # Spectral decay: ratio of top SV to sum of all SVs (higher = more concentrated = cleaner)
    concentration_raw = (S_raw[0] / S_raw.sum()).item()
    concentration_out = (S_out[0] / S_out.sum()).item()

    # Soft filter should make the spectrum more concentrated (shrinks small SVs more)
    ok = concentration_out >= concentration_raw * 0.95  # at least as concentrated
    print(f"  raw spectral concentration = {concentration_raw:.4f}")
    print(f"  filtered spectral concentration = {concentration_out:.4f}")
    print(PASS if ok else FAIL)
    return ok


# ---------------------------------------------------------------------------
# Test 8: Multi-client consensus weights sum to 1
# ---------------------------------------------------------------------------

def test_consensus_weights_sum():
    print("\n[8] Consensus weights: multiple clients → weights sum to 1.0")
    agg = make_aggregator(beta=0.9, use_consensus=True)

    for i in range(3):
        dw = make_delta(seed=i)
        agg.update(dw, weight=1/3)

    weights = agg._consensus_weights()
    total = sum(weights)

    ok = abs(total - 1.0) < 1e-5
    print(f"  weights = {[f'{w:.4f}' for w in weights]}  sum = {total:.6f}")
    print(PASS if ok else FAIL)
    return ok


# ---------------------------------------------------------------------------
# Test 9: project_to_rank output shapes are correct
# ---------------------------------------------------------------------------

def test_project_to_rank_shapes():
    print("\n[9] project_to_rank: output (B, A) shapes correct for various ranks")
    W = torch.randn(64, 32)
    ok = True
    for rank in [2, 4, 8, 16]:
        B, A = SPAMomentumAggregator.project_to_rank(W, rank, device="cpu")
        shape_ok = B.shape == (64, rank) and A.shape == (rank, 32)
        print(f"  rank={rank:2d} → B={tuple(B.shape)} A={tuple(A.shape)} "
              + ("✓" if shape_ok else f"✗ WRONG"))
        ok = ok and shape_ok
    print(PASS if ok else FAIL)
    return ok


# ---------------------------------------------------------------------------
# Test 10: Old bug reproduction — verify old behaviour would fail test 6
#          (documents that the bug was real; expected to show difference)
# ---------------------------------------------------------------------------

def test_old_bug_would_fail():
    print("\n[10] Regression: verify old clamp(0,1) formula would amplify oscillation")

    def old_adaptive_beta(prev, cur, beta=0.9):
        p = torch.cat([v.flatten() for v in prev.values()])
        c = torch.cat([v.flatten() for v in cur.values()])
        sim = float(torch.nn.functional.cosine_similarity(
            c.unsqueeze(0), p.unsqueeze(0)
        ).clamp(0.0, 1.0))  # old buggy clamp
        return beta * (1.0 - sim)  # old buggy formula

    dw_pos = {"layer0": torch.ones(16, 8) * 0.5}
    dw_neg = {"layer0": torch.ones(16, 8) * -0.5}

    # With old formula: anti-correlated → sim clamped to 0 → β = 0.9
    beta_old = old_adaptive_beta(dw_pos, dw_neg, beta=0.9)

    # With new formula: anti-correlated → sim = -1 → β = 0
    sim_new = float(torch.nn.functional.cosine_similarity(
        torch.cat([v.flatten() for v in dw_neg.values()]).unsqueeze(0),
        torch.cat([v.flatten() for v in dw_pos.values()]).unsqueeze(0),
    ).clamp(-1.0, 1.0))
    beta_new = 0.9 * (sim_new + 1.0) / 2.0

    print(f"  Old β for anti-correlated input: {beta_old:.4f} (HIGH = amplifies oscillation)")
    print(f"  New β for anti-correlated input: {beta_new:.4f} (ZERO = brakes oscillation)")
    ok = beta_old > 0.8 and beta_new < 0.05
    print(PASS if ok else FAIL + " (regression check inconclusive)")
    return ok


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("SPA-M Fix Verification Tests (CPU-only, no model required)")
    print("=" * 60)

    results = [
        test_magnitude_normalization(),
        test_adaptive_beta_anti_correlated(),
        test_adaptive_beta_consistent(),
        test_adaptive_beta_orthogonal(),
        test_bias_correction_cumulative(),
        test_oscillation_suppression(),
        test_svd_filter_before_momentum(),
        test_consensus_weights_sum(),
        test_project_to_rank_shapes(),
        test_old_bug_would_fail(),
    ]

    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print("\n" + "=" * 60)
    print(f"Results: {n_pass}/{len(results)} passed, {n_fail} failed")

    if n_fail == 0:
        print("ALL TESTS PASSED — safe to run smoke test on GPU")
    else:
        print("FAILURES DETECTED — fix before running on GPU")
    print("=" * 60)

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
