"""
Plot convergence curves across methods and datasets.

Usage:
  python analysis/plot_convergence.py --dataset yelp --alpha 0.5
  python analysis/plot_convergence.py --all

Produces figures/convergence_{dataset}_alpha{alpha}.pdf
"""

import argparse
import json
import os
import glob
import numpy as np

METHODS = ["homo_r4", "homo_r8", "hetero_pad", "flexlora", "hetero_spa", "spa_m"]
METHOD_LABELS = {
    "homo_r4":    "FedAvg (r=4)",
    "homo_r8":    "FedAvg (r=8)",
    "hetero_pad": "Hetero-Pad",
    "flexlora":   "FlexLoRA",
    "hetero_spa": "SPA (Ours)",
    "spa_m":      "SPA-M (Ours)",
}
COLORS = {
    "homo_r4":    "#9e9e9e",
    "homo_r8":    "#607d8b",
    "hetero_pad": "#ff7043",
    "flexlora":   "#42a5f5",
    "hetero_spa": "#e53935",
    "spa_m":      "#6a1b9a",
}
LINESTYLES = {
    "homo_r4":    ":",
    "homo_r8":    "--",
    "hetero_pad": "-.",
    "flexlora":   "--",
    "hetero_spa": "-",
    "spa_m":      "-",
}
LINEWIDTHS = {
    "homo_r4":    1.2,
    "homo_r8":    1.2,
    "hetero_pad": 1.5,
    "flexlora":   1.8,
    "hetero_spa": 2.5,
    "spa_m":      2.5,
}

PRIMARY_METRIC = {
    "yelp":   "accuracy",
    "alpaca": "rouge_l",
    "gsm8k":  "exact_match",
}

SEEDS = [42, 43, 44, 45, 46]


def load_runs(dataset: str, method: str, alpha: float, results_base: str):
    results_dir = os.path.join(results_base, dataset)
    alpha_str = str(alpha).replace(".", "")
    runs = []
    for seed in SEEDS:
        for fname_template in [
            f"{method}_alpha{alpha_str}_seed{seed}.json",
            f"{method}_seed{seed}_alpha{alpha_str}.json",
        ]:
            path = os.path.join(results_dir, fname_template)
            if os.path.exists(path):
                with open(path) as f:
                    runs.append(json.load(f))
                break
    return runs


def extract_curve(runs, metric: str):
    """Extract per-round mean and std across seeds."""
    if not runs:
        return [], [], []

    n_rounds = max(len(r["rounds"]) for r in runs)
    per_round = {rnd: [] for rnd in range(1, n_rounds + 1)}

    for run in runs:
        for r in run["rounds"]:
            if metric in r:
                per_round[r["round"]].append(r[metric])

    rounds, means, stds = [], [], []
    for rnd in sorted(per_round):
        vals = per_round[rnd]
        if vals:
            rounds.append(rnd)
            means.append(np.mean(vals))
            stds.append(np.std(vals))

    return rounds, means, stds


def plot_convergence(dataset: str, alpha: float, results_base: str, out_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plots.")
        return

    metric = PRIMARY_METRIC[dataset]
    scale = 100.0 if metric in ("accuracy", "f1_macro", "rouge_l", "exact_match") else 1.0
    ylabel_map = {
        "accuracy":    "Accuracy (%)",
        "rouge_l":     "ROUGE-L (%)",
        "exact_match": "Exact Match (%)",
    }
    ylabel = ylabel_map.get(metric, metric)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for method in METHODS:
        runs = load_runs(dataset, method, alpha, results_base)
        rounds, means, stds = extract_curve(runs, metric)
        if not rounds:
            continue

        means = np.array(means) * scale
        stds = np.array(stds) * scale

        ax.plot(rounds, means,
                label=METHOD_LABELS[method],
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                linewidth=LINEWIDTHS[method])

        if len(runs) > 1:
            ax.fill_between(rounds, means - stds, means + stds,
                            alpha=0.12, color=COLORS[method])

    ax.set_xlabel("Federated Round", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{dataset.upper()} | Non-IID α={alpha}", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(1, None)

    os.makedirs(out_dir, exist_ok=True)
    alpha_str = str(alpha).replace(".", "")
    out_path = os.path.join(out_dir, f"convergence_{dataset}_alpha{alpha_str}.pdf")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_train_loss(dataset: str, alpha: float, results_base: str, out_dir: str):
    """Plot training loss convergence."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    for method in METHODS:
        runs = load_runs(dataset, method, alpha, results_base)
        rounds, means, stds = extract_curve(runs, "avg_loss")
        if not rounds:
            continue

        means = np.array(means)
        stds = np.array(stds)

        ax.plot(rounds, means,
                label=METHOD_LABELS[method],
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                linewidth=LINEWIDTHS[method])
        if len(runs) > 1:
            ax.fill_between(rounds, means - stds, means + stds,
                            alpha=0.12, color=COLORS[method])

    ax.set_xlabel("Federated Round", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title(f"{dataset.upper()} — Training Loss | α={alpha}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    os.makedirs(out_dir, exist_ok=True)
    alpha_str = str(alpha).replace(".", "")
    out_path = os.path.join(out_dir, f"train_loss_{dataset}_alpha{alpha_str}.pdf")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["yelp", "alpaca", "gsm8k"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--out-dir", type=str, default="figures")
    args = parser.parse_args()

    if args.all:
        for dataset in ["yelp", "alpaca", "gsm8k"]:
            for alpha in ([0.5, 0.1] if dataset == "yelp" else [0.5]):
                plot_convergence(dataset, alpha, args.results_dir, args.out_dir)
                plot_train_loss(dataset, alpha, args.results_dir, args.out_dir)
    else:
        plot_convergence(args.dataset, args.alpha, args.results_dir, args.out_dir)
        plot_train_loss(args.dataset, args.alpha, args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()
