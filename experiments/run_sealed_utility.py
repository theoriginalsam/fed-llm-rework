"""
Accuracy of federated LoRA fine-tuning with sealed noise (clip + Gaussian noise per client release).

Runs FlexLoRA (full-ΔW aggregation, so the release is the update the server sees)
on one dataset at several privacy levels, plus two references:
  none   no clipping, no noise (the ordinary run)
  clip   clipping only, no noise (separates the cost of clipping from noise)
  eps=x  clipping + Gaussian noise calibrated to (x, 1e-5)-DP per release (replace-one,
         sensitivity 2C over the whole update). Per-client totals compose over the
         client's releases and are logged.

Usage (on the GPU machine, from rework/):
  python experiments/run_sealed_utility.py --dataset yelp --seed 42
  python experiments/run_sealed_utility.py --dataset yelp --seed 42 --conditions none clip 1e4 1e3 100 10 1
  python experiments/run_sealed_utility.py --dataset yelp --seed 42 --conditions 100 --impl box   # full FedGT box in the loop

Results: <results-dir>/<dataset>/<condition>/flexlora_..._seed<seed>.json plus sealed_<condition>_seed<seed>.json
"""
import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.base_config import NUM_CLIENTS, NUM_ROUNDS, BATCH_SIZE
from src.privacy import SealedRelease
from src.server.fl_server import run_federated

# a trailing "d" = the server denoises the noisy aggregate (post-processing, no privacy cost)
# "base": one round whose aggregate the denoiser zeroes entirely (eps 1e-6), i.e. the untrained base model
DEFAULT_CONDITIONS = ["base", "none", "clip", "1e4", "1e4d", "1e3", "1e3d", "100d", "10d", "1d", "1e5", "1e5d"]


def load_data(dataset, tokenizer, seed, alpha):
    if dataset == "yelp":
        from config.dataset_configs import YELP_CONFIG as cfg
        from src.data.yelp import load_yelp as load
    elif dataset == "gsm8k":
        from config.dataset_configs import GSM8K_CONFIG as cfg
        from src.data.gsm8k import load_gsm8k as load
    elif dataset == "alpaca":
        from config.dataset_configs import ALPACA_CONFIG as cfg
        from src.data.alpaca import load_alpaca as load
    else:
        raise ValueError(dataset)
    clients, test = load(tokenizer=tokenizer, num_clients=NUM_CLIENTS, alpha=alpha, seed=seed)
    return clients, test, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="yelp", choices=["yelp", "gsm8k", "alpaca"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    ap.add_argument("--clip", type=float, default=1.5,
                    help="joint clip norm C over all q/v modules (campaign median joint q-norm is about 1.35)")
    ap.add_argument("--impl", choices=["torch", "box"], default="torch")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-rounds", type=int, default=NUM_ROUNDS)
    ap.add_argument("--results-dir", default="results_sealed")
    args = ap.parse_args()

    from experiments.run_yelp import load_base_model
    model, tokenizer = load_base_model(args.device)

    for cond in args.conditions:
        out_dir = os.path.join(args.results_dir, args.dataset, cond)
        os.makedirs(out_dir, exist_ok=True)
        summary_path = os.path.join(out_dir, f"sealed_{cond}_seed{args.seed}.json")
        if os.path.exists(summary_path):
            print(f"Skipping {cond}: done")
            continue
        sealed = None
        denoise = cond.endswith("d") or cond == "base"
        level = cond[:-1] if cond.endswith("d") else cond
        rounds = 1 if cond == "base" else args.num_rounds
        if cond == "base":
            sealed = SealedRelease(clip=args.clip, eps_per_release=1e-6, impl="torch")
        elif cond == "clip":
            sealed = SealedRelease(clip=args.clip, eps_per_release=None, impl="torch")
        elif cond != "none":
            sealed = SealedRelease(clip=args.clip, eps_per_release=float(level), impl=args.impl)
        print(f"\n=== {args.dataset} | {cond} | seed {args.seed} | sigma="
              f"{getattr(sealed, 'sigma', 0.0):.4g}")
        clients, test, cfg = load_data(args.dataset, tokenizer, args.seed, args.alpha)
        res = run_federated(
            method="flexlora", base_model=model, tokenizer=tokenizer,
            client_datasets=clients, test_dataset=test, dataset_config=cfg,
            seed=args.seed, alpha=args.alpha, results_dir=out_dir, device=args.device,
            num_rounds=rounds, batch_size=BATCH_SIZE, sealed_noise=sealed,
            sealed_denoise=denoise,
        )
        json.dump({"condition": cond, "denoise": denoise, "clip": args.clip, "impl": args.impl,
                   "sigma": getattr(sealed, "sigma", 0.0),
                   "eps_per_release": getattr(sealed, "eps_per_release", None),
                   "eps_client_final": ({str(c): sealed.epsilon(c) for c in sealed.releases}
                                        if sealed is not None and sealed.sigma > 0 else None),
                   "release_log": getattr(sealed, "log", None),
                   "rounds": res["rounds"]}, open(summary_path, "w"), indent=1, default=str)
    print("done")


if __name__ == "__main__":
    main()
