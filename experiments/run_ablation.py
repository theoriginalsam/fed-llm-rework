"""
Ablation experiments:
  A1 — Non-IID severity: alpha={0.5, 0.2, 0.1, 0.05} on Yelp
  A2 — Rank ratio gap: uniform vs extreme heterogeneity
  A3 — Number of clients: 20, 50, 100

Usage:
  python experiments/run_ablation.py --ablation A1
  python experiments/run_ablation.py --ablation A2
  python experiments/run_ablation.py --ablation A3
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config.base_config import MODEL_NAME, NUM_ROUNDS
from config.dataset_configs import YELP_CONFIG
from src.data.yelp import load_yelp
from src.server.fl_server import run_federated

SEEDS = [42, 43, 44]
KEY_METHODS = ["homo_r8", "hetero_pad", "flexlora", "hetero_spa"]


def load_base_model(device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def run_a1(model, tokenizer, device):
    """Alpha sweep: 0.5, 0.2, 0.1, 0.05."""
    results_dir = os.path.join("results", "ablation", "A1_alpha_sweep")
    alphas = [0.5, 0.2, 0.1, 0.05]
    for alpha in alphas:
        for method in KEY_METHODS:
            for seed in SEEDS:
                tag = f"{method}_alpha{str(alpha).replace('.','')}_seed{seed}"
                if os.path.exists(os.path.join(results_dir, f"{tag}.json")):
                    continue
                print(f"A1: {method} alpha={alpha} seed={seed}")
                client_ds, eval_ds = load_yelp(tokenizer, 50, alpha, seed)
                run_federated(method, model, tokenizer, client_ds, eval_ds,
                              YELP_CONFIG, seed, alpha, results_dir, device, NUM_ROUNDS)


def run_a2(model, tokenizer, device):
    """Rank heterogeneity: extreme (r4+r32 only) vs mild (r8+r16 only) vs uniform."""
    results_dir = os.path.join("results", "ablation", "A2_rank_gap")
    # This ablation requires modifying rank distribution — handled via config override
    # For now, log a note
    print("A2 ablation: modify RANK_DISTRIBUTION in config/base_config.py, then re-run.")
    print("Suggested configurations:")
    print("  Extreme: r4=25, r32=25 (skip r8/r16)")
    print("  Mild:    r8=25, r16=25 (skip r4/r32)")
    print("  Uniform: r8=50 (homogeneous)")
    print("Re-run run_yelp.py with each config to get A2 results.")


def run_a3(model, tokenizer, device):
    """Client count: 20, 50, 100."""
    results_dir = os.path.join("results", "ablation", "A3_client_count")
    for num_clients in [20, 50, 100]:
        for method in ["flexlora", "hetero_spa"]:
            for seed in SEEDS:
                tag = f"{method}_nc{num_clients}_seed{seed}"
                if os.path.exists(os.path.join(results_dir, f"{tag}.json")):
                    continue
                print(f"A3: {method} clients={num_clients} seed={seed}")
                client_ds, eval_ds = load_yelp(tokenizer, num_clients, alpha=0.5, seed=seed)
                run_federated(method, model, tokenizer, client_ds, eval_ds,
                              YELP_CONFIG, seed, 0.5, results_dir, device, NUM_ROUNDS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=str, choices=["A1", "A2", "A3"], required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model, tokenizer = load_base_model(args.device)

    if args.ablation == "A1":
        run_a1(model, tokenizer, args.device)
    elif args.ablation == "A2":
        run_a2(model, tokenizer, args.device)
    elif args.ablation == "A3":
        run_a3(model, tokenizer, args.device)


if __name__ == "__main__":
    main()
