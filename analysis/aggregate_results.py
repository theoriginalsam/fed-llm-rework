"""
Aggregate experiment results across seeds and print LaTeX tables.

Usage:
  python analysis/aggregate_results.py --dataset yelp
  python analysis/aggregate_results.py --dataset alpaca
  python analysis/aggregate_results.py --dataset gsm8k
  python analysis/aggregate_results.py --all

Output: LaTeX table rows ready to paste into paper.
"""

import argparse
import json
import os
import glob
import numpy as np
from typing import Dict, List, Optional, Tuple

METHODS = ["homo_r4", "homo_r8", "hetero_pad", "flexlora", "hetero_spa"]
METHOD_LABELS = {
    "homo_r4":    "FedAvg (r=4)",
    "homo_r8":    "FedAvg (r=8)",
    "hetero_pad": "Hetero-Pad",
    "flexlora":   "FlexLoRA",
    "hetero_spa": "SPA (Ours)",
}
ALPHAS = [0.5, 0.1]
SEEDS = [42, 43, 44, 45, 46]

# Metrics of interest per dataset
DATASET_METRICS = {
    "yelp":   ["accuracy", "f1_macro", "perplexity"],
    "alpaca": ["rouge_l", "bleu", "hallucination_rate", "perplexity"],
    "gsm8k":  ["exact_match", "perplexity"],
}

HIGHER_BETTER = {"accuracy", "f1_macro", "rouge_l", "bleu", "exact_match"}
LOWER_BETTER  = {"perplexity", "hallucination_rate"}


def load_run(path: str) -> Optional[Dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def final_metrics(run: Dict) -> Dict[str, float]:
    """Extract metrics from the last completed round."""
    rounds = run.get("rounds", [])
    if not rounds:
        return {}
    return {k: v for k, v in rounds[-1].items()
            if isinstance(v, float) and k not in ("avg_loss", "round_time_s")}


def convergence_round(run: Dict, metric: str, threshold: float) -> Optional[int]:
    """First round where metric exceeds threshold (for higher-is-better)."""
    for r in run.get("rounds", []):
        if r.get(metric, 0) >= threshold:
            return r["round"]
    return None


def aggregate_dataset(dataset: str, alpha: float, results_base: str) -> Dict:
    """
    For each method, collect final-round metrics across seeds.
    Returns dict: method → {metric: (mean, std, n_runs)}
    """
    results_dir = os.path.join(results_base, dataset)
    metrics_list = DATASET_METRICS[dataset]

    agg = {}
    for method in METHODS:
        runs_data = []
        for seed in SEEDS:
            alpha_str = str(alpha).replace(".", "")
            fname = f"{method}_alpha{alpha_str}_seed{seed}.json"
            path = os.path.join(results_dir, fname)
            if not os.path.exists(path):
                # Try seed-first naming
                fname2 = f"{method}_seed{seed}_alpha{alpha_str}.json"
                path = os.path.join(results_dir, fname2)
            run = load_run(path)
            if run:
                m = final_metrics(run)
                if m:
                    runs_data.append(m)

        if not runs_data:
            agg[method] = {metric: (float("nan"), float("nan"), 0) for metric in metrics_list}
            continue

        method_agg = {}
        for metric in metrics_list:
            vals = [r[metric] for r in runs_data if metric in r]
            if vals:
                method_agg[metric] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
            else:
                method_agg[metric] = (float("nan"), float("nan"), 0)
        agg[method] = method_agg

    return agg


def bold_best(values: List[Tuple[float, float, int]], metrics: List[str],
              metric: str) -> List[bool]:
    """Return boolean list — True if this method is best for this metric."""
    means = [v[0] for v in values]
    valid = [(i, m) for i, m in enumerate(means) if not np.isnan(m)]
    if not valid:
        return [False] * len(values)
    if metric in HIGHER_BETTER:
        best_idx = max(valid, key=lambda x: x[1])[0]
    else:
        best_idx = min(valid, key=lambda x: x[1])[0]
    return [i == best_idx for i in range(len(values))]


def format_cell(mean: float, std: float, n: int, is_bold: bool, scale: float = 1.0,
                pct: bool = False) -> str:
    if np.isnan(mean):
        return "—"
    m = mean * scale
    s = std * scale
    suffix = "\\%" if pct else ""
    cell = f"{m:.2f}$\\pm${s:.2f}{suffix}"
    if n < len(SEEDS):
        cell += f" ({n})"
    if is_bold:
        cell = f"\\textbf{{{cell}}}"
    return cell


def print_latex_table(dataset: str, alpha: float, results_base: str):
    agg = aggregate_dataset(dataset, alpha, results_base)
    metrics = DATASET_METRICS[dataset]

    alpha_str = f"\\alpha={alpha}"
    print(f"\n% ============================================================")
    print(f"% Table: {dataset.upper()} | {alpha_str}")
    print(f"% ============================================================")
    print("\\begin{table}[h]")
    print("\\centering")

    # Column spec: Method + metrics columns
    col_spec = "l" + "c" * len(metrics)
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\toprule")

    # Header row
    metric_headers = {
        "accuracy": "Acc.",
        "f1_macro": "F1",
        "perplexity": "PPL",
        "rouge_l": "ROUGE-L",
        "bleu": "BLEU",
        "hallucination_rate": "Hall.",
        "exact_match": "EM",
    }
    headers = " & ".join(["Method"] + [metric_headers.get(m, m) for m in metrics])
    print(f"{headers} \\\\")
    print("\\midrule")

    for method in METHODS:
        row_data = agg[method]
        # Gather all values for bold comparison
        all_vals = [agg[m2].get(metric, (float("nan"), float("nan"), 0))
                    for m2 in METHODS for metric in metrics]
        # Per metric, compare across methods
        cells = []
        for metric in metrics:
            vals_for_metric = [agg[m2].get(metric, (float("nan"), float("nan"), 0))
                               for m2 in METHODS]
            bolds = bold_best(vals_for_metric, metrics, metric)
            my_idx = METHODS.index(method)
            mean, std, n = row_data.get(metric, (float("nan"), float("nan"), 0))

            # Scale: accuracy/f1/rouge_l/bleu/exact_match × 100 → percentage
            scale = 100.0 if metric in ("accuracy", "f1_macro", "rouge_l", "bleu", "exact_match") else 1.0
            pct = metric in ("accuracy", "f1_macro", "rouge_l", "bleu", "exact_match",
                             "hallucination_rate")
            cells.append(format_cell(mean, std, n, bolds[my_idx], scale, pct))

        label = METHOD_LABELS.get(method, method)
        row = " & ".join([label] + cells) + " \\\\"
        if method == "hetero_spa":
            row = "\\rowcolor{gray!15} " + row
        print(row)

    print("\\bottomrule")
    print("\\end{tabular}")
    print(f"\\caption{{Results on {dataset.upper()} with non-IID severity ${alpha_str}$.")
    print(f"Mean $\\pm$ std over {len(SEEDS)} seeds. Bold = best. PPL = perplexity (lower is better).}}")
    print(f"\\label{{tab:{dataset}_alpha{str(alpha).replace('.', '')}}}")
    print("\\end{table}")


def print_summary_table(dataset: str, results_base: str):
    """Combined table: one block per alpha, side by side or stacked."""
    print(f"\n{'='*70}")
    print(f"SUMMARY: {dataset.upper()} — Final Round Metrics (mean ± std over seeds)")
    print(f"{'='*70}")

    metrics = DATASET_METRICS[dataset]
    header = f"{'Method':<20}" + "".join(f"{'α=0.5':<15}{'α=0.1':<15}")
    print(header)

    for alpha in ALPHAS:
        agg = aggregate_dataset(dataset, alpha, results_base)
        for method in METHODS:
            label = METHOD_LABELS.get(method, method)
            row = f"  {label:<20}"
            for metric in metrics:
                mean, std, n = agg[method].get(metric, (float("nan"), float("nan"), 0))
                scale = 100.0 if metric in ("accuracy", "f1_macro", "rouge_l", "bleu", "exact_match") else 1.0
                if np.isnan(mean):
                    row += f"  {'—':>12}"
                else:
                    row += f"  {mean*scale:>6.2f}±{std*scale:<5.2f}"
            print(row)
        print()


def check_completeness(results_base: str):
    """Report which runs are done vs pending."""
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETENESS CHECK")
    print("=" * 70)

    for dataset in ["yelp", "alpaca", "gsm8k"]:
        results_dir = os.path.join(results_base, dataset)
        if dataset == "yelp":
            methods = METHODS
            alphas = ALPHAS
            seeds = SEEDS
        elif dataset == "alpaca":
            methods = METHODS
            alphas = [0.5]
            seeds = [42, 43, 44]
        else:  # gsm8k
            methods = METHODS
            alphas = [0.5]
            seeds = [42, 43, 44]

        done, total = 0, 0
        missing = []
        for method in methods:
            for alpha in alphas:
                for seed in seeds:
                    total += 1
                    alpha_str = str(alpha).replace(".", "")
                    fname = f"{method}_alpha{alpha_str}_seed{seed}.json"
                    path = os.path.join(results_dir, fname)
                    fname2 = f"{method}_seed{seed}_alpha{alpha_str}.json"
                    path2 = os.path.join(results_dir, fname2)
                    if os.path.exists(path) or os.path.exists(path2):
                        done += 1
                    else:
                        missing.append(f"{dataset}/{method}_a{alpha}_s{seed}")

        pct = 100 * done / total if total > 0 else 0
        print(f"  {dataset.upper():<8}: {done:>3}/{total} ({pct:.0f}%)")
        if missing and len(missing) <= 10:
            for m in missing:
                print(f"    MISSING: {m}")
        elif missing:
            print(f"    ... and {len(missing)} more missing runs")

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["yelp", "alpaca", "gsm8k"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--check", action="store_true", help="Just check completeness")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Base results directory")
    args = parser.parse_args()

    results_base = args.results_dir

    check_completeness(results_base)

    if args.check:
        return

    datasets = ["yelp", "alpaca", "gsm8k"] if args.all else [args.dataset]

    for dataset in datasets:
        print(f"\n{'#'*70}")
        print(f"# DATASET: {dataset.upper()}")
        print(f"{'#'*70}")

        if dataset == "yelp":
            for alpha in ALPHAS:
                print_latex_table(dataset, alpha, results_base)
                print_summary_table(dataset, results_base)
                break  # summary covers both alphas
        else:
            print_latex_table(dataset, 0.5, results_base)

        print_summary_table(dataset, results_base)


if __name__ == "__main__":
    main()
