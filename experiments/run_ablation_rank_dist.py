"""
Rank distribution ablation.

Compares two rank distributions on Yelp alpha=0.1, 2 seeds:
  - balanced:  {r4:20, r8:20, r16:5, r32:5}  — current default
  - skewed:    {r4:35, r8:10, r16:3, r32:2}  — more extreme edge-heavy

Methods: hetlora_m, hetlora, spa_m, flexlora
Hypothesis: skewed distribution amplifies the rank heterogeneity problem,
making HetLoRA-M's Frobenius weighting more critical.

Usage:
  python experiments/run_ablation_rank_dist.py --dist skewed --method hetlora_m --seed 42
  python experiments/run_ablation_rank_dist.py --all
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config.base_config import MODEL_NAME, NUM_CLIENTS, NUM_ROUNDS, BATCH_SIZE
from config.dataset_configs import YELP_CONFIG
from src.data.yelp import load_yelp
from src.server.fl_server import run_federated

RANK_DISTRIBUTIONS = {
    "balanced": {"r4": 20, "r8": 20, "r16": 5, "r32": 5},
    "skewed":   {"r4": 35, "r8": 10, "r16": 3, "r32": 2},
}
METHODS = ["hetlora_m", "hetlora", "spa_m", "flexlora"]
ALPHA = 0.1
SEEDS = [42, 43]
RESULTS_DIR = "results_ablation/rank_dist"


def load_base_model(device):
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=str, default="skewed",
                        choices=list(RANK_DISTRIBUTIONS.keys()))
    parser.add_argument("--method", type=str, default="hetlora_m", choices=METHODS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    model, tokenizer = load_base_model(args.device)

    runs = (
        [(d, m, s) for d in RANK_DISTRIBUTIONS for m in METHODS for s in SEEDS]
        if args.all
        else [(args.dist, args.method, args.seed)]
    )

    for dist_name, method, seed in runs:
        # Save into dist-specific subdir — filename stays {method}_seed{seed}_alpha{alpha}.json
        subdir = os.path.join(RESULTS_DIR, "yelp", dist_name)
        os.makedirs(subdir, exist_ok=True)
        out_file = os.path.join(subdir,
                                f"{method}_seed{seed}_alpha{str(ALPHA).replace('.','')}.json")
        if os.path.exists(out_file):
            print(f"Skipping {dist_name} {method} seed={seed} — already done.")
            continue

        print(f"\n{'='*60}")
        print(f"Rank dist ablation: dist={dist_name} | method={method} | seed={seed}")
        print(f"  Distribution: {RANK_DISTRIBUTIONS[dist_name]}")
        print(f"{'='*60}")

        client_datasets, eval_samples = load_yelp(
            tokenizer=tokenizer,
            num_clients=NUM_CLIENTS,
            alpha=ALPHA,
            seed=seed,
        )

        run_federated(
            method=method,
            base_model=model,
            tokenizer=tokenizer,
            client_datasets=client_datasets,
            test_dataset=eval_samples,
            dataset_config=YELP_CONFIG,
            seed=seed,
            alpha=ALPHA,
            results_dir=subdir,
            device=args.device,
            num_rounds=NUM_ROUNDS,
            batch_size=BATCH_SIZE,
            hetlora_m_beta=0.5,
            rank_distribution=RANK_DISTRIBUTIONS[dist_name],
        )
        print(f"Saved → {out_file}")

    print("\nRank distribution ablation complete.")


if __name__ == "__main__":
    main()
