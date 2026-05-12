"""
Yelp experiment runner.
Runs: all 5 methods × α={0.5, 0.1} × 5 seeds × 20 rounds.

Usage:
  python experiments/run_yelp.py --alpha 0.5 --seed 42 --method hetero_spa
  python experiments/run_yelp.py --alpha 0.5 --all  # run all methods
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config.base_config import MODEL_NAME, NUM_CLIENTS, NUM_ROUNDS, SEEDS, ALPHA_VALUES, METHODS, BATCH_SIZE
from config.dataset_configs import YELP_CONFIG
from src.data.yelp import load_yelp
from src.server.fl_server import run_federated


def load_base_model(device: str = "cuda"):
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
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="hetero_spa", choices=METHODS)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true", help="Run all methods × seeds × alphas")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--spa-tau", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (default: from base_config)")
    parser.add_argument("--results-dir", type=str, default="results_v2",
                        help="Root results directory (default: results_v2)")
    parser.add_argument("--num-rounds", type=int, default=None,
                        help="Override number of rounds (default: from base_config)")
    args = parser.parse_args()

    results_dir = os.path.join(args.results_dir, "yelp")
    os.makedirs(results_dir, exist_ok=True)

    model, tokenizer = load_base_model(args.device)

    if args.all:
        runs = [(m, a, s) for m in METHODS for a in ALPHA_VALUES for s in SEEDS]
    else:
        runs = [(args.method, args.alpha, args.seed)]

    for method, alpha, seed in runs:
        tag = f"{method}_alpha{str(alpha).replace('.','')}_seed{seed}"
        out_file = os.path.join(results_dir, f"{tag}.json")
        if os.path.exists(out_file):
            print(f"Skipping {tag} — already done.")
            continue

        print(f"\n{'='*60}")
        print(f"Running: {method} | alpha={alpha} | seed={seed}")
        print(f"{'='*60}")

        client_datasets, eval_samples = load_yelp(
            tokenizer=tokenizer,
            num_clients=NUM_CLIENTS,
            alpha=alpha,
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
            alpha=alpha,
            results_dir=results_dir,
            device=args.device,
            num_rounds=args.num_rounds if args.num_rounds is not None else NUM_ROUNDS,
            spa_tau=args.spa_tau,
            batch_size=args.batch_size if args.batch_size is not None else BATCH_SIZE,
        )

    print("\nAll Yelp experiments complete.")


if __name__ == "__main__":
    main()
