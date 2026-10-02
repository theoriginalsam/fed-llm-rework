"""
Summarize the eval-side FedMoLoRA runs produced by run_evalside.sh.

Each results_evalside/<dataset>/hetlora_seed*_alpha*.json run holds, per round,
  acc_raw_eval -> HetLoRA (raw aggregate, what clients train from)
  acc_ema_eval -> FedMoLoRA (bias-corrected EMA, beta=0.5, evaluation only)
  ema_b<beta>  -> full metrics for each extra beta (Yelp alpha=0.1 beta ablation)

Prints, per setting: AUC and Mean-Last-5 for FedMoLoRA vs HetLoRA (paired within
run), FedMoLoRA vs the other baselines (paired by seed, Holm-corrected), the beta
sweep, and a reproducibility check of the raw trajectory against earlier HetLoRA runs.

Usage:
  python experiments/summarize_evalside.py
  python experiments/summarize_evalside.py --baseline-dirs results_prof results_v2
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

METRIC = {"yelp": "accuracy", "gsm8k": "exact_match", "alpaca": "rouge_l"}
SETTINGS = [("yelp", 0.5), ("yelp", 0.1), ("yelp", 0.01),
            ("gsm8k", 0.5), ("gsm8k", 0.1), ("gsm8k", 0.01), ("alpaca", 0.1)]
BASELINES = ["homo_r8", "hetero_pad", "flexlora", "spa_m"]


def scale(ds, v):
    v = np.asarray(v, dtype=float)
    return v if ds == "alpaca" else v * 100


def auc(v):
    return float(np.mean(v))


def last5(v):
    return float(np.mean(v[-5:]))


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def load_evalside(root):
    runs = defaultdict(dict)  # (ds, alpha) -> seed -> record dict
    for ds in METRIC:
        for f in glob.glob(os.path.join(root, ds, "hetlora_seed*_alpha*.json")):
            j = json.load(open(f))
            rounds = j["rounds"]
            rec = {
                "raw": scale(ds, [r["acc_raw_eval"] for r in rounds]),
                "ema": scale(ds, [r["acc_ema_eval"] for r in rounds]),
                "loss": [r["avg_loss"] for r in rounds],
                "betas": {},
            }
            for key in rounds[0]:
                if key.startswith("ema_b"):
                    rec["betas"][float(key[5:])] = scale(ds, [r[key][METRIC[ds]] for r in rounds])
            runs[(ds, j["alpha"])][j["seed"]] = rec
    return runs


def load_baselines(dirs, n_rounds):
    data = defaultdict(dict)  # (ds, method, alpha) -> seed -> values
    for d in dirs:
        for ds in METRIC:
            for f in sorted(glob.glob(os.path.join(d, ds, "*.json"))):
                try:
                    j = json.load(open(f))
                except (json.JSONDecodeError, OSError):
                    continue
                if "rounds" not in j:
                    continue
                key = (ds, j["method"], j["alpha"])
                if len(j["rounds"]) != n_rounds.get((ds, j["alpha"])) or j["seed"] in data[key]:
                    continue
                vals = [r.get(METRIC[ds]) for r in j["rounds"]]
                data[key][j["seed"]] = {"vals": scale(ds, vals),
                                        "loss": [r["avg_loss"] for r in j["rounds"]]}
    return data


def paired(a, b):
    a, b = np.asarray(a), np.asarray(b)
    p = stats.ttest_rel(a, b).pvalue if len(a) > 1 else float("nan")
    return float(np.mean(a - b)), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evalside-dir", default="results_evalside")
    ap.add_argument("--baseline-dirs", nargs="+", default=["results_prof", "results_v2"])
    args = ap.parse_args()

    runs = load_evalside(args.evalside_dir)
    n_rounds = {k: len(next(iter(v.values()))["raw"]) for k, v in runs.items() if v}
    base = load_baselines(args.baseline_dirs, n_rounds)
    fmt = lambda ds, x: f"{x:.3f}" if ds == "alpaca" else f"{x:.2f}"

    for ds, alpha in SETTINGS:
        seeds = sorted(runs.get((ds, alpha), {}))
        if not seeds:
            print(f"\n### {ds} alpha={alpha}: no eval-side runs yet")
            continue
        R = runs[(ds, alpha)]
        print(f"\n### {ds.upper()} alpha={alpha}  ({n_rounds[(ds, alpha)]} rounds, seeds {seeds})")

        for name, fn in (("AUC", auc), ("Mean-L5", last5)):
            ema = [fn(R[s]["ema"]) for s in seeds]
            raw = [fn(R[s]["raw"]) for s in seeds]
            gain, p = paired(ema, raw)
            print(f"  {name:8s} FedMoLoRA {fmt(ds, np.mean(ema))} ± {np.std(ema, ddof=1) if len(ema) > 1 else 0:.2f}"
                  f" | HetLoRA {fmt(ds, np.mean(raw))} | gain {gain:+.3f}  p={p:.4f} (paired within run)")

            rows = []
            for b in BASELINES:
                bd = base.get((ds, b, alpha), {})
                common = [s for s in seeds if s in bd]
                if len(common) < 2:
                    continue
                g, pb = paired([fn(R[s]["ema"]) for s in common], [fn(bd[s]["vals"]) for s in common])
                rows.append((b, len(common), np.mean([fn(bd[s]["vals"]) for s in common]), g, pb))
            if rows:
                adj = holm([r[4] for r in rows])
                for (b, n, mean_b, g, pb), pa in zip(rows, adj):
                    mark = "*" if pa < 0.05 else ("." if pb < 0.05 else " ")
                    print(f"           vs {b:10s} n={n} base {fmt(ds, mean_b)}  gain {g:+.3f}"
                          f"  p={pb:.4f}  Holm={pa:.4f} {mark}")

        betas = sorted(R[seeds[0]]["betas"])
        if len(betas) > 1:
            cells = "  ".join(f"b={b:g}: {fmt(ds, np.mean([auc(R[s]['betas'][b]) for s in seeds]))}"
                              for b in betas)
            print(f"  beta sweep (AUC)  {cells}")

        old = base.get((ds, "hetlora", alpha), {})
        diffs = [max(abs(x - y) for x, y in zip(R[s]["loss"], old[s]["loss"])) for s in seeds if s in old]
        if diffs:
            same = sum(d < 1e-6 for d in diffs)
            print(f"  reproducibility: raw training loss identical to earlier HetLoRA run in "
                  f"{same}/{len(diffs)} seeds (max diff {max(diffs):.2e})")

    print("\n* = significant after Holm correction; . = significant uncorrected only")


if __name__ == "__main__":
    main()
