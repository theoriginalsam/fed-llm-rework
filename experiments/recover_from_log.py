"""
Reconstruct ablation JSON files from nohup log output.

The nohup log captures all stdout including per-round metrics.
Use this when JSON files were overwritten before being moved to subdirs.

Usage:
  # Beta ablation
  python experiments/recover_from_log.py --type beta --log logs/ablation_beta.log --out results_ablation/beta/yelp

  # K participation ablation
  python experiments/recover_from_log.py --type k --log logs/ablation_k.log --out results_ablation/k_participation/yelp

  # Add --dry-run to preview without writing
  python experiments/recover_from_log.py --type beta --log logs/ablation_beta.log --out results_ablation/beta/yelp --dry-run
"""

import re
import json
import os
import argparse
import ast


def parse_metrics(raw):
    try:
        return json.loads(raw.replace("'", '"'))
    except Exception:
        try:
            return ast.literal_eval(raw)
        except Exception:
            return None


def parse_beta_log(log_path):
    """Parse ablation_beta.log → list of {method, beta, seed, alpha, rounds}"""
    runs = []
    current = None
    header_re = re.compile(
        r"Beta ablation: method=(\w+) \| beta=([\d.]+) \| alpha=([\d.]+) \| seed=(\d+)"
    )
    round_re = re.compile(r"Round (\d+) \| (\{.*?\}) \| time=")

    with open(log_path) as f:
        for line in f:
            line = re.sub(r"^\[[\s\d.]+s\] ", "", line).strip()
            m = header_re.match(line)
            if m:
                if current is not None:
                    runs.append(current)
                current = {"method": m.group(1), "beta": float(m.group(2)),
                           "alpha": float(m.group(3)), "seed": int(m.group(4)), "rounds": []}
                continue
            if current is None:
                continue
            m = round_re.search(line)
            if m:
                metrics = parse_metrics(m.group(2))
                if metrics:
                    current["rounds"].append({"round": int(m.group(1)), **metrics})

    if current is not None:
        runs.append(current)
    return runs


def parse_k_log(log_path):
    """Parse ablation_k.log → list of {method, k, seed, alpha, rounds}"""
    runs = []
    current = None
    header_re = re.compile(
        r"K ablation: K=(\d+) \| method=(\w+) \| alpha=([\d.]+) \| seed=(\d+)"
    )
    round_re = re.compile(r"Round (\d+) \| (\{.*?\}) \| time=")

    with open(log_path) as f:
        for line in f:
            line = re.sub(r"^\[[\s\d.]+s\] ", "", line).strip()
            m = header_re.match(line)
            if m:
                if current is not None:
                    runs.append(current)
                current = {"method": m.group(2), "k": int(m.group(1)),
                           "alpha": float(m.group(3)), "seed": int(m.group(4)), "rounds": []}
                continue
            if current is None:
                continue
            m = round_re.search(line)
            if m:
                metrics = parse_metrics(m.group(2))
                if metrics:
                    current["rounds"].append({"round": int(m.group(1)), **metrics})

    if current is not None:
        runs.append(current)
    return runs


def save_runs(runs, out_base, key_fn, subdir_fn, min_rounds=20, dry_run=False):
    print(f"Found {len(runs)} runs")
    saved, skipped, partial = 0, 0, 0
    for run in runs:
        n = len(run['rounds'])
        subdir = os.path.join(out_base, subdir_fn(run))
        fname = f"{run['method']}_seed{run['seed']}_alpha{str(run['alpha']).replace('.','')}.json"
        out_path = os.path.join(subdir, fname)
        status = "✓" if n >= min_rounds else f"⚠ partial ({n} rounds)"
        print(f"  {key_fn(run)} → {n} rounds → {out_path}  {status}")
        if dry_run:
            continue
        if os.path.exists(out_path):
            print(f"    SKIP — already exists")
            skipped += 1
            continue
        if n < min_rounds:
            print(f"    SKIP — incomplete ({n}/{min_rounds} rounds), still running?")
            partial += 1
            continue
        os.makedirs(subdir, exist_ok=True)
        payload = {"method": run['method'], "seed": run['seed'],
                   "alpha": run['alpha'], "rounds": run['rounds']}
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"    Saved ✓")
        saved += 1
    if not dry_run:
        print(f"\nDone: {saved} saved, {skipped} skipped (exist), {partial} skipped (incomplete)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", required=True, choices=["beta", "k"],
                        help="Ablation type: beta or k")
    parser.add_argument("--log", required=True, help="Path to nohup log file")
    parser.add_argument("--out", required=True, help="Base output dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.type == "beta":
        runs = parse_beta_log(args.log)
        save_runs(runs, args.out,
                  key_fn=lambda r: f"method={r['method']} beta={r['beta']} seed={r['seed']}",
                  subdir_fn=lambda r: f"beta{str(r['beta']).replace('.', '')}",
                  dry_run=args.dry_run)
    else:
        runs = parse_k_log(args.log)
        save_runs(runs, args.out,
                  key_fn=lambda r: f"method={r['method']} K={r['k']} seed={r['seed']}",
                  subdir_fn=lambda r: f"k{r['k']}",
                  dry_run=args.dry_run)


if __name__ == "__main__":
    main()
