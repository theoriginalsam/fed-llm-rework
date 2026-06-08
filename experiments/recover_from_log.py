"""
Reconstruct ablation JSON files from nohup log output.

The nohup log captures all stdout including per-round metrics.
Use this when JSON files were overwritten before being moved to subdirs.

Usage:
  python experiments/recover_from_log.py --log logs/ablation_beta.log --out results_ablation/beta/yelp
"""

import re
import json
import os
import argparse


def parse_log(log_path):
    """
    Parse nohup log and reconstruct per-run round data.
    Returns list of dicts: {method, beta, seed, alpha, rounds}
    """
    runs = []
    current = None

    # Matches: "Beta ablation: method=hetlora_m | beta=0.5 | alpha=0.1 | seed=42"
    run_header = re.compile(
        r"Beta ablation: method=(\w+) \| beta=([\d.]+) \| alpha=([\d.]+) \| seed=(\d+)"
    )
    # Matches: "  Round 5 | {'accuracy': 0.4231, 'rouge1': 0.312} | time=42.3s"
    round_line = re.compile(r"Round (\d+) \| (\{.*?\}) \| time=")

    with open(log_path) as f:
        for line in f:
            # Strip log prefix "[elapsed s] "
            line = re.sub(r"^\[[\s\d.]+s\] ", "", line).strip()

            m = run_header.match(line)
            if m:
                if current is not None:
                    runs.append(current)
                current = {
                    "method": m.group(1),
                    "beta":   float(m.group(2)),
                    "seed":   int(m.group(4)),
                    "alpha":  float(m.group(3)),
                    "rounds": [],
                }
                continue

            if current is None:
                continue

            m = round_line.search(line)
            if m:
                round_num = int(m.group(1))
                try:
                    metrics = json.loads(m.group(2).replace("'", '"'))
                except Exception:
                    # Try ast.literal_eval for non-JSON dicts
                    import ast
                    try:
                        metrics = ast.literal_eval(m.group(2))
                    except Exception:
                        continue
                current["rounds"].append({"round": round_num, **metrics})

    if current is not None:
        runs.append(current)

    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to nohup log file")
    parser.add_argument("--out", required=True, help="Base output dir (e.g. results_ablation/beta/yelp)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be saved without writing")
    args = parser.parse_args()

    runs = parse_log(args.log)

    print(f"Found {len(runs)} runs in {args.log}")
    for run in runs:
        beta_str = str(run['beta']).replace('.', '')
        subdir = os.path.join(args.out, f"beta{beta_str}")
        fname = f"{run['method']}_seed{run['seed']}_alpha{str(run['alpha']).replace('.','')}.json"
        out_path = os.path.join(subdir, fname)

        n_rounds = len(run['rounds'])
        print(f"  method={run['method']} beta={run['beta']} seed={run['seed']} "
              f"→ {n_rounds} rounds → {out_path}")

        if args.dry_run:
            continue

        if os.path.exists(out_path):
            print(f"    SKIP — already exists")
            continue

        os.makedirs(subdir, exist_ok=True)
        payload = {
            "method": run['method'],
            "seed":   run['seed'],
            "alpha":  run['alpha'],
            "rounds": run['rounds'],
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"    Saved ✓")


if __name__ == "__main__":
    main()
