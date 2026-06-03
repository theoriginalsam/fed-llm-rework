"""
Beta ablation for HetLoRA-M.

Sweeps beta in {0.3, 0.5, 0.7} on Yelp alpha=0.1, 3 seeds.
Isolates the effect of the EMA decay rate on convergence stability.

Usage:
  python experiments/run_ablation_beta.py --beta 0.3 --seed 42
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
ALPHA = 0.1          # extreme non-IID — where beta matters most
SEEDS = [42, 43, 44]
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
    parser.add_argument("--beta", type=float, default=0.5, choices=BETA_VALUES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(os.path.join(RESULTS_DIR, "yelp"), exist_ok=True)
    model, tokenizer = load_base_model(args.device)

    runs = [(b, s) for b in BETA_VALUES for s in SEEDS] if args.all else [(args.beta, args.seed)]

    for beta, seed in runs:
        tag = f"hetlora_m_beta{str(beta).replace('.','')}_alpha{str(ALPHA).replace('.','')}_seed{seed}"
        out_file = os.path.join(RESULTS_DIR, "yelp", f"{tag}.json")
        if os.path.exists(out_file):
            print(f"Skipping {tag} — already done.")
            continue

        print(f"\n{'='*60}")
        print(f"Beta ablation: beta={beta} | alpha={ALPHA} | seed={seed}")
        print(f"{'='*60}")

        client_datasets, eval_samples = load_yelp(
            tokenizer=tokenizer,
            num_clients=NUM_CLIENTS,
            alpha=ALPHA,
            seed=seed,
        )

        run_federated(
            method="hetlora_m",
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
            hetlora_m_beta=beta,
            # save under the tag so results don't collide with main grid
            # (ExperimentLogger uses method+seed+alpha; we rename after)
        )

        # Rename output to include beta in filename
        default_out = os.path.join(
            RESULTS_DIR, "yelp",
            f"hetlora_m_alpha{str(ALPHA).replace('.','')}_seed{seed}.json"
        )
        if os.path.exists(default_out) and not os.path.exists(out_file):
            os.rename(default_out, out_file)
            print(f"Saved → {out_file}")

    print("\nBeta ablation complete.")


if __name__ == "__main__":
    main()
