"""
Alpaca instruction-following experiment runner.
Methods: all 5 | alpha=0.5 | 3 seeds | 20 rounds.

Usage:
  python experiments/run_alpaca.py --method hetero_spa --seed 42
  python experiments/run_alpaca.py --all
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config.base_config import MODEL_NAME, NUM_CLIENTS, NUM_ROUNDS, METHODS
from config.dataset_configs import ALPACA_CONFIG
from src.data.alpaca import load_alpaca
from src.server.fl_server import run_federated

ALPACA_SEEDS = [42, 43, 44]  # 3 seeds for generation tasks


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="hetero_spa",
                        choices=METHODS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--spa-tau", type=float, default=0.01)
    parser.add_argument("--results-dir", type=str, default="results_v2")
    parser.add_argument("--hetlora-m-beta", type=float, default=0.5)
    parser.add_argument("--save-adapters-dir", type=str, default=None,
                        help="If set, save each selected client's per-round LoRA "
                             "adapter here for the FedGT spectral auditor")
    parser.add_argument("--clients-per-round", type=int, default=None,
                        help="Override clients selected per round (default: from "
                             "base_config)")
    parser.add_argument("--num-rounds", type=int, default=None,
                        help="Override number of rounds")
    args = parser.parse_args()

    results_dir = os.path.join(args.results_dir, "alpaca")
    os.makedirs(results_dir, exist_ok=True)

    model, tokenizer = load_base_model(args.device)

    runs = [(m, args.alpha, s) for m in METHODS for s in ALPACA_SEEDS] if args.all else [(args.method, args.alpha, args.seed)]

    for method, alpha, seed in runs:
        tag = f"{method}_alpha{str(alpha).replace('.','')}_seed{seed}"
        out_file = os.path.join(results_dir, f"{tag}.json")
        if os.path.exists(out_file):
            print(f"Skipping {tag}")
            continue

        print(f"\nRunning Alpaca: {method} | alpha={alpha} | seed={seed}")
        client_datasets, eval_samples = load_alpaca(tokenizer, NUM_CLIENTS, alpha=alpha, seed=seed)

        run_federated(
            method=method,
            base_model=model,
            tokenizer=tokenizer,
            client_datasets=client_datasets,
            test_dataset=eval_samples,
            dataset_config=ALPACA_CONFIG,
            seed=seed,
            alpha=alpha,
            results_dir=results_dir,
            device=args.device,
            num_rounds=args.num_rounds if args.num_rounds is not None else NUM_ROUNDS,
            spa_tau=args.spa_tau,
            hetlora_m_beta=args.hetlora_m_beta,
            save_adapters_dir=args.save_adapters_dir,
            **({"clients_per_round": args.clients_per_round}
               if args.clients_per_round is not None else {}),
        )

    print("Alpaca experiments complete.")


if __name__ == "__main__":
    main()
