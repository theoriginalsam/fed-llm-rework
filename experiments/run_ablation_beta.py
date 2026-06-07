"""
Beta ablation — HetLoRA-M and SPA-M at matched β values.

Sweeps beta in {0.3, 0.5, 0.7} for BOTH HetLoRA-M and SPA-M on Yelp alpha=0.1, 3 seeds.

Why both methods: if we only ablate HetLoRA-M, a reviewer can argue that HetLoRA-M wins
just because β=0.5 is better than SPA-M's default β_max=0.9, not because of adapter-space
vs ΔW-space momentum placement. Testing both at the same β isolates the architectural
difference (feedback loop) from the hyperparameter difference.

Expected result: HetLoRA-M should lead SPA-M at every β value, with the gap widest
at high β (high momentum amplifies the feedback loop in SPA-M).

Usage:
  python experiments/run_ablation_beta.py --method hetlora_m --beta 0.5 --seed 42
  python experiments/run_ablation_beta.py --all
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

BETA_VALUES = [0.3, 0.5, 0.7]
METHODS     = ["hetlora_m", "spa_m"]
ALPHA       = 0.1
SEEDS       = [42, 43, 44]
RESULTS_DIR = "results_ablation/beta"


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
    parser.add_argument("--method", type=str, default="hetlora_m", choices=METHODS)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true",
                        help="Run all methods × betas × seeds")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    model, tokenizer = load_base_model(args.device)

    runs = (
        [(m, b, s) for m in METHODS for b in BETA_VALUES for s in SEEDS]
        if args.all
        else [(args.method, args.beta, args.seed)]
    )

    for method, beta, seed in runs:
        beta_str = str(beta).replace('.', '')
        # Save into beta-specific subdir — filename stays {method}_seed{seed}_alpha{alpha}.json
        subdir = os.path.join(RESULTS_DIR, "yelp", f"beta{beta_str}")
        os.makedirs(subdir, exist_ok=True)
        out_file = os.path.join(subdir,
                                f"{method}_seed{seed}_alpha{str(ALPHA).replace('.','')}.json")
        if os.path.exists(out_file):
            print(f"Skipping beta={beta} {method} seed={seed} — already done.")
            continue

        print(f"\n{'='*60}")
        print(f"Beta ablation: method={method} | beta={beta} | alpha={ALPHA} | seed={seed}")
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
            hetlora_m_beta=beta,
            spa_m_beta=beta,
        )
        print(f"Saved → {out_file}")

    print("\nBeta ablation complete.")


if __name__ == "__main__":
    main()
