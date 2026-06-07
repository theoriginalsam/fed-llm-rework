"""
Participation rate (K) ablation.

Sweeps clients_per_round in {5, 10, 20} on Yelp alpha=0.1, 2 seeds.
Methods: hetlora_m, hetlora, spa_m — isolates when momentum helps vs hurts
as a function of how many clients participate each round.

Professor note: SPA-M's adaptive beta fires correctly only when K is large
enough that consecutive rounds share clients (subspace overlap).
HetLoRA-M should be robust across K since it never feeds momentum to clients.

Usage:
  python experiments/run_ablation_k.py --k 10 --method hetlora_m --seed 42
  python experiments/run_ablation_k.py --all
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

K_VALUES = [5, 10, 20]
METHODS = ["hetlora_m", "hetlora", "spa_m"]
ALPHA = 0.1
SEEDS = [42, 43]
RESULTS_DIR = "results_ablation/k_participation"


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
    parser.add_argument("--k", type=int, default=5, choices=K_VALUES)
    parser.add_argument("--method", type=str, default="hetlora_m", choices=METHODS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(os.path.join(RESULTS_DIR, "yelp"), exist_ok=True)
    model, tokenizer = load_base_model(args.device)

    runs = (
        [(k, m, s) for k in K_VALUES for m in METHODS for s in SEEDS]
        if args.all
        else [(args.k, args.method, args.seed)]
    )

    for k, method, seed in runs:
        tag = f"{method}_K{k}_alpha{str(ALPHA).replace('.','')}_seed{seed}"
        out_file = os.path.join(RESULTS_DIR, "yelp", f"{tag}.json")
        if os.path.exists(out_file):
            print(f"Skipping {tag} — already done.")
            continue

        print(f"\n{'='*60}")
        print(f"K ablation: K={k} | method={method} | alpha={ALPHA} | seed={seed}")
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
            results_dir=os.path.join(RESULTS_DIR, "yelp"),
            device=args.device,
            num_rounds=NUM_ROUNDS,
            batch_size=BATCH_SIZE,
            hetlora_m_beta=0.5,
            clients_per_round=k,
        )

        # Rename to include K in filename
        default_out = os.path.join(
            RESULTS_DIR, "yelp",
            f"{method}_seed{seed}_alpha{str(ALPHA).replace('.','')}.json"
        )
        if os.path.exists(default_out) and not os.path.exists(out_file):
            os.rename(default_out, out_file)
            print(f"Saved → {out_file}")

    print("\nK participation ablation complete.")


if __name__ == "__main__":
    main()
