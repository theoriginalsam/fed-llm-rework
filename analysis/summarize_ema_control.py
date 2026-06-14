"""
Summarize EMA-eval control results (tab:ema_control).

Reads results_ema_control/yelp/{method}_alpha01_seed{seed}.json files,
extracts per-round acc_raw_eval and acc_ema_eval, computes AUC (mean over rounds)
per seed, then reports mean ± std and paired t-test vs raw.

Usage:
    python analysis/summarize_ema_control.py
    python analysis/summarize_ema_control.py --results_dir results_ema_control
    python analysis/summarize_ema_control.py --seeds 42 43 44
"""

import argparse
import json
from pathlib import Path
from scipy import stats
import numpy as np


METHODS = ["homo_r8", "hetero_pad", "flexlora"]
METHOD_LABELS = {
    "homo_r8": "Homo r=8",
    "hetero_pad": "Hetero-Pad",
    "flexlora": "FlexLoRA",
}
# These two rows come from main-table results, not this control run
HETLORA_RAW_AUC  = 41.5   # HetLoRA main-table AUC (Yelp alpha=0.1)
HETLORA_M_AUC    = 44.6   # HetLoRA-M main-table AUC (== HetLoRA + EMA-eval)


def load_per_seed(results_dir: Path, method: str, alpha: float, seeds: list):
    """Return (raw_aucs, ema_aucs) lists across seeds."""
    raw_aucs, ema_aucs = [], []
    alpha_tag = str(alpha).replace(".", "")
    for seed in seeds:
        fp = results_dir / f"{method}_seed{seed}_alpha{alpha_tag}.json"
        if not fp.exists():
            print(f"  WARNING: {fp} not found — skipping seed {seed}")
            continue
        data = json.loads(fp.read_text())
        rounds = data["rounds"]

        has_ema = "acc_ema_eval" in rounds[0]
        if not has_ema:
            print(f"  WARNING: {fp} has no acc_ema_eval — was --ema-eval passed?")

        raw_accs = [r.get("acc_raw_eval", r.get("accuracy")) for r in rounds]
        raw_aucs.append(np.mean(raw_accs) * 100)

        if has_ema:
            ema_accs = [r["acc_ema_eval"] for r in rounds]
            ema_aucs.append(np.mean(ema_accs) * 100)

    return raw_aucs, ema_aucs


def fmt(mean, std):
    return f"{mean:.1f}±{std:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results_ema_control/yelp")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    rows = []

    print(f"\nEMA-eval control: Yelp alpha={args.alpha}, seeds={args.seeds}")
    print(f"{'Method':12s}  {'Raw AUC':>14s}  {'+ EMA AUC':>14s}  {'Δ':>6s}  {'p-value':>10s}")
    print("-" * 65)

    for method in METHODS:
        raw_aucs, ema_aucs = load_per_seed(results_dir, method, args.alpha, args.seeds)
        if not raw_aucs:
            continue

        raw_mean, raw_std = np.mean(raw_aucs), np.std(raw_aucs, ddof=1)
        if ema_aucs and len(ema_aucs) == len(raw_aucs):
            ema_mean, ema_std = np.mean(ema_aucs), np.std(ema_aucs, ddof=1)
            delta = ema_mean - raw_mean
            t, p = stats.ttest_rel(ema_aucs, raw_aucs)
            p_str = f"p={p:.4f}"
        else:
            ema_mean = ema_std = delta = None
            p_str = "—"

        row = {
            "method": METHOD_LABELS[method],
            "raw_mean": raw_mean, "raw_std": raw_std,
            "ema_mean": ema_mean, "ema_std": ema_std,
            "delta": delta, "p": p_str,
            "n_seeds": len(raw_aucs),
        }
        rows.append(row)
        ema_fmt = fmt(ema_mean, ema_std) if ema_mean is not None else "—"
        delta_fmt = f"{delta:+.1f}" if delta is not None else "—"
        print(f"{METHOD_LABELS[method]:12s}  {fmt(raw_mean, raw_std):>14s}  {ema_fmt:>14s}  {delta_fmt:>6s}  {p_str:>10s}")

    # HetLoRA / HetLoRA-M rows from main-table (paired by construction — same seeds)
    print(f"{'HetLoRA':12s}  {HETLORA_RAW_AUC:>13.1f}  {'—':>14s}  {'—':>6s}  {'—':>10s}")
    print(f"{'HetLoRA-M':12s}  {HETLORA_M_AUC:>13.1f}  {'n/a (IS EMA)':>14s}  {'—':>6s}  {'—':>10s}")
    print()

    # LaTeX table snippet
    print("─── LaTeX (paste into paper) ───────────────────────────────────────")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{EMA-eval control: applying bias-corrected EMA ($\beta=0.5$) post-hoc")
    print(r"to each baseline's server-side state on Yelp $\alpha=0.1$.")
    print(r"Raw = standard eval; +EMA = eval on smoothed model.")
    n = rows[0]["n_seeds"] if rows else 5
    print(rf"All methods: {n} seeds. \textbf{{Bold}} = best per column.}}")
    print(r"\label{tab:ema_control}")
    print(r"\small")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"\textbf{Method} & \textbf{Raw AUC} & \textbf{+EMA AUC} & $\Delta$ & $p$-value \\")
    print(r"\midrule")
    for row in rows:
        em = fmt(row["ema_mean"], row["ema_std"]) if row["ema_mean"] else "—"
        d = f'{row["delta"]:+.1f}' if row["delta"] else "—"
        print(f'{row["method"]:12s} & {fmt(row["raw_mean"], row["raw_std"])} & {em} & {d} & {row["p"]} \\\\')
    print(r"\midrule")
    print(rf'HetLoRA    & {HETLORA_RAW_AUC:.1f}  & — & — & — \\')
    print(rf'\textbf{{HetLoRA-M}} & — & \textbf{{{HETLORA_M_AUC:.1f}}} & — & — \\')
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
