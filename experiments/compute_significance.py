"""
Compute paired t-tests: HetLoRA-M vs each baseline on Yelp AUC (alpha=0.1).

Pairs by seed — only seeds present for BOTH methods are used.
AUC = mean accuracy over all rounds in the JSON.

Usage:
  python experiments/compute_significance.py --results results/V2 --alpha 01
  python experiments/compute_significance.py --results results/V2 --alpha 01 --verbose
"""

import os
import json
import argparse
import numpy as np
from scipy import stats


METHOD_FILENAMES = {
    "HetLoRA-M":  "hetlora_m",
    "HetLoRA":    "hetlora",
    "SPA-M":      "spa_m",
    "FlexLoRA":   "flexlora",
    "Hetero-Pad": "hetero_pad",
    "Homo r=8":   "homo_r8",
}


def compute_auc(rounds):
    """Mean accuracy over all rounds (our primary metric)."""
    accs = [r["accuracy"] for r in rounds if "accuracy" in r]
    return float(np.mean(accs)) * 100  # percent


def load_seeds(results_dir, method_stem, alpha):
    """
    Load all JSON files for a given method and alpha.
    Handles both filename conventions:
      - {method}_seed{seed}_alpha{alpha}.json  (old)
      - {method}_alpha{alpha}_seed{seed}.json  (new, run_yelp.py)
    Returns dict: seed (int) -> AUC (float)
    """
    seed_aucs = {}
    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue
        if " (1)" in fname:  # skip duplicate copies
            continue
        stem = fname.replace(".json", "")
        if not stem.startswith(method_stem + "_"):
            continue
        if f"alpha{alpha}" not in stem:
            continue
        seed = None
        # old: {method}_seed{N}_alpha{A}
        if "_seed" in stem and stem.index("_seed") < stem.index("alpha"):
            try:
                seed = int(stem.split("_seed")[1].split("_")[0])
            except (IndexError, ValueError):
                pass
        # new: {method}_alpha{A}_seed{N}
        if seed is None and "_seed" in stem:
            try:
                seed = int(stem.split("_seed")[1])
            except (IndexError, ValueError):
                pass
        if seed is None:
            continue
        path = os.path.join(results_dir, fname)
        with open(path) as f:
            data = json.load(f)
        auc = compute_auc(data["rounds"])
        seed_aucs[seed] = auc
    return seed_aucs


def paired_ttest(a_vals, b_vals, paired_seeds):
    """Paired t-test: a - b per seed."""
    diffs = [a_vals[s] - b_vals[s] for s in paired_seeds]
    t_stat, p_val = stats.ttest_1samp(diffs, popmean=0)
    mean_diff = np.mean(diffs)
    return mean_diff, p_val, len(paired_seeds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/V2",
                        help="Directory containing flat JSON result files")
    parser.add_argument("--alpha", default="01",
                        help="Alpha suffix in filename, e.g. '01' for alpha=0.1")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    results_dir = args.results
    alpha = args.alpha

    # Load HetLoRA-M seeds
    hetlora_m = load_seeds(results_dir, "hetlora_m", alpha)
    if not hetlora_m:
        print(f"ERROR: No hetlora_m files found in {results_dir} for alpha={alpha}")
        return
    print(f"\nHetLoRA-M seeds found: {sorted(hetlora_m.keys())}")
    for s, auc in sorted(hetlora_m.items()):
        print(f"  seed={s}  AUC={auc:.4f}%")
    print(f"  Mean AUC = {np.mean(list(hetlora_m.values())):.4f}%  "
          f"Std = {np.std(list(hetlora_m.values()), ddof=1):.4f}%")

    print(f"\n{'Comparison':<20}  {'Seeds':>5}  {'Gain':>8}  {'Relative':>9}  {'p-value':>9}  {'Sig?':>6}")
    print("-" * 65)

    baselines = [k for k in METHOD_FILENAMES if k != "HetLoRA-M"]
    for name in baselines:
        stem = METHOD_FILENAMES[name]
        baseline = load_seeds(results_dir, stem, alpha)
        common = sorted(set(hetlora_m) & set(baseline))
        if len(common) < 2:
            print(f"  vs. {name:<16}  n={len(common):>1}  (insufficient matched seeds, need ≥2)")
            if args.verbose and baseline:
                print(f"    baseline seeds: {sorted(baseline.keys())}")
            continue

        gain, p_val, n = paired_ttest(hetlora_m, baseline, common)
        base_mean = np.mean([baseline[s] for s in common])
        relative = gain / base_mean * 100
        sig = "✓" if p_val < 0.05 else ("~" if p_val < 0.10 else "✗")

        print(f"  vs. {name:<16}  n={n:>1}  {gain:>+7.2f}pp  {relative:>+8.1f}%  {p_val:>9.4f}  {sig:>6}")

        if args.verbose:
            print(f"    paired seeds: {common}")
            diffs = [hetlora_m[s] - baseline[s] for s in common]
            for s, d in zip(common, diffs):
                print(f"      seed={s}: HetLoRA-M={hetlora_m[s]:.3f}  baseline={baseline[s]:.3f}  diff={d:+.3f}")

    print()
    print("Note: paired t-test, H0: mean(HetLoRA-M - baseline) = 0, two-tailed.")
    print("      Only seeds present for BOTH methods are paired.")
    print("      Minimum 5 seeds (df=4) recommended for reliable p-values.")


if __name__ == "__main__":
    main()
