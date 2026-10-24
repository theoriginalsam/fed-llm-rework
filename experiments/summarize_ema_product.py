"""
Summarize run_ema_product.sh: EMA of the adapter product (Delta W = B A) vs EMA of the factors.

For each Yelp setting prints, per model (mean over seeds):
  AUC, Mean-Last-5, worst round, drops > 5 pp, within-run std
and paired t-tests (with 95% CIs) for:
  product_svd vs factor_svd   where the average is taken (the reviewers' question)
  factor_svd  vs factor       SVD truncation vs slicing of the same factor EMA
  product_svd vs raw          gain of product smoothing over HetLoRA
  factor      vs raw          FedMoLoRA gain (should match results_evalside)
Sanity checks: round 1 of product_svd equals factor_svd (bias correction), and raw / factor
match the earlier results_evalside runs.

Usage:
  python experiments/summarize_ema_product.py [--dir results_emaprod] [--evalside results_evalside]
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

MODELS = {
    "raw": "acc_raw_eval",
    "factor": "acc_ema_eval",
    "factor_svd": "acc_ema_factor_svd_eval",
    "product_svd": "acc_ema_product_svd_eval",
}
LABEL = {"raw": "HetLoRA (raw)", "factor": "FedMoLoRA (factor EMA)",
         "factor_svd": "factor EMA -> SVD", "product_svd": "product EMA -> SVD"}
PAIRS = [("product_svd", "factor_svd"), ("factor_svd", "factor"),
         ("product_svd", "raw"), ("factor", "raw")]


def load(directory, alpha):
    runs = {}
    for f in sorted(glob.glob(os.path.join(directory, "yelp", "hetlora_seed*_alpha*.json"))):
        j = json.load(open(f))
        if j["alpha"] != alpha:
            continue
        rounds = j["rounds"]
        runs[j["seed"]] = {m: 100 * np.array([r.get(k, np.nan) for r in rounds], dtype=float)
                           for m, k in MODELS.items()}
    return runs


def ci(d):
    d = np.asarray(d)
    if len(d) < 2:
        return float("nan"), float("nan")
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean() - h, d.mean() + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_emaprod")
    ap.add_argument("--evalside", default="results_evalside")
    args = ap.parse_args()

    for alpha in (0.1, 0.01):
        runs = load(args.dir, alpha)
        if not runs:
            print(f"\n### Yelp alpha={alpha}: no runs yet")
            continue
        seeds = sorted(runs)
        print(f"\n### Yelp alpha={alpha}  (seeds {seeds})")
        print(f"  {'model':26s} {'AUC':>6s} {'ML5':>6s} {'worst':>6s} {'drops':>6s} {'std':>5s}")
        auc = {}
        for m in MODELS:
            arr = np.array([runs[s][m] for s in seeds])
            if np.isnan(arr).all():
                continue
            auc[m] = arr.mean(1)
            drops = (np.diff(arr, axis=1) < -5).sum(1)
            print(f"  {LABEL[m]:26s} {arr.mean():6.2f} {arr[:, -5:].mean():6.2f} {arr.min(1).mean():6.1f} "
                  f"{drops.mean():6.1f} {arr.std(1, ddof=1).mean():5.1f}")
        print("  paired AUC comparisons (gain, 95% CI, p):")
        for a, b in PAIRS:
            if a in auc and b in auc and len(seeds) > 1:
                d = auc[a] - auc[b]
                lo, hi = ci(d)
                p = stats.ttest_rel(auc[a], auc[b]).pvalue
                print(f"    {LABEL[a]:22s} - {LABEL[b]:22s} {d.mean():+6.2f}  [{lo:+.2f}, {hi:+.2f}]  p={p:.4f}"
                      f"  ({(d > 0).sum()}/{len(d)} seeds positive)")

        r1 = [abs(runs[s]["product_svd"][0] - runs[s]["factor_svd"][0]) for s in seeds]
        print(f"  check: round-1 |product_svd - factor_svd| max = {max(r1):.4f} (expected 0)")
        old = load(args.evalside, alpha)
        common = [s for s in seeds if s in old]
        if common:
            dr = max(abs(runs[s]["raw"].mean() - old[s]["raw"].mean()) for s in common)
            df = max(abs(runs[s]["factor"].mean() - old[s]["factor"].mean()) for s in common)
            print(f"  check: vs {args.evalside}: max AUC difference raw {dr:.2f} pp, factor {df:.2f} pp "
                  f"(0 if training reproduced exactly)")


if __name__ == "__main__":
    main()
