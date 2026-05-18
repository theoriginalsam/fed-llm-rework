"""
FlexLoRA-metric comparison experiment.

Mirrors FlexLoRA paper setup as closely as possible:
  - Model:   Qwen2.5-1.5B (≈ DataJuicer 1.3B used in FlexLoRA paper)
  - Dataset: Dolly-15K, task-heterogeneous partition (50 train + 10 unseen clients)
  - Metric:  ROUGE-L on unseen clients (zero-shot generalization) — FlexLoRA's primary metric
  - Methods: flexlora, spa_m, homo_r8 (oracle baseline)
  - Ranks:   {r4:20, r8:20, r16:5, r32:5} (same as main experiments)
  - Rounds:  20 | Seeds: 42, 43, 44 | Clients/round: 10

Key difference from main experiments:
  - Held-out unseen clients evaluated zero-shot each round (not fixed test set)
  - Generation task (ROUGE-L) instead of classification (accuracy)
  - Smaller model (1.5B vs 7B) — matches FlexLoRA's compute regime

Usage:
  python experiments/run_flexlora_comparison.py --method flexlora --seed 42
  python experiments/run_flexlora_comparison.py --method spa_m --seed 42 --device cuda:1
  python experiments/run_flexlora_comparison.py --all --device cuda:0
"""

import argparse
import copy
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from config.base_config import (
    CLIENTS_PER_ROUND, NUM_ROUNDS, STEPS_PER_ROUND, LR,
    BATCH_SIZE, GRAD_ACCUM_STEPS, MAX_RANK, TARGET_MODULES, RANK_DISTRIBUTION,
)
from src.aggregation.flexlora import FlexLoRAAggregator
from src.aggregation.spa_momentum import SPAMomentumAggregator
from src.aggregation.fedavg_homo import HomoAggregator
from src.clients.lora_client import train_client
from src.data.dolly import load_dolly_federated, NUM_TRAIN_CLIENTS
from src.evaluation.metrics import evaluate_unseen_clients, _load_eval_model
from src.server.fl_server import build_client_rank_map, project_wagg_to_client
from src.utils.logging_utils import ExperimentLogger

SMALL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
COMPARISON_METHODS = ["flexlora", "spa_m", "homo_r8"]
SEEDS = [42, 43, 44]
CLIENTS_PER_ROUND_COMPARISON = 10  # FlexLoRA used 10/round
RESULTS_DIR = "results_flexlora_comparison"


def load_base_model(device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        SMALL_MODEL, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def run_comparison(
    method: str,
    base_model,
    tokenizer,
    train_datasets,
    unseen_samples,
    seed: int,
    device: str,
    results_dir: str,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if method == "homo_r8":
        client_rank_map = {cid: 8 for cid in range(NUM_TRAIN_CLIENTS)}
        aggregator = HomoAggregator(rank=8, max_rank=MAX_RANK)
    else:
        client_rank_map = build_client_rank_map(RANK_DISTRIBUTION)
        if method == "flexlora":
            aggregator = FlexLoRAAggregator(max_rank=MAX_RANK)
        else:  # spa_m
            aggregator = SPAMomentumAggregator(max_rank=MAX_RANK, beta=0.9, gamma=1.0,
                                               use_consensus=True, consensus_rank=4)

    logger = ExperimentLogger(method, seed, alpha=0.0, results_dir=results_dir)
    logger.log(f"FlexLoRA comparison | method={method} seed={seed} model={SMALL_MODEL}")

    global_wagg = None
    round_results = []

    pbar = tqdm(range(1, NUM_ROUNDS + 1), desc=f"{method}|s{seed}", unit="round")

    for round_num in pbar:
        round_start = time.time()
        eligible = [cid for cid in range(NUM_TRAIN_CLIENTS) if len(train_datasets[cid]) > 0]
        selected = random.sample(eligible, min(CLIENTS_PER_ROUND_COMPARISON, len(eligible)))
        aggregator.reset()

        total_rw = sum(client_rank_map[cid] * len(train_datasets[cid]) for cid in selected)
        round_losses = []

        for cid in selected:
            rank = client_rank_map[cid]
            weight = (rank * len(train_datasets[cid])) / max(total_rw, 1)
            client_global = None if global_wagg is None else project_wagg_to_client(
                global_wagg, rank, method if method != "homo_r8" else "flexlora", 0.01, device
            )

            weights, loss = train_client(
                base_model=base_model,
                tokenizer=tokenizer,
                rank=rank,
                target_modules=TARGET_MODULES,
                global_weights=client_global,
                dataset=train_datasets[cid],
                steps=STEPS_PER_ROUND,
                batch_size=BATCH_SIZE,
                grad_accum=GRAD_ACCUM_STEPS,
                lr=LR,
                device=device,
                extract_method="full_w",
                pbar_desc=f"  R{round_num} C{cid}",
            )
            round_losses.append(loss)
            aggregator.update(weights, weight)

        global_wagg = {k: v.cpu() for k, v in aggregator.get_global().items()}

        # Eval rank: median
        all_ranks = sorted(client_rank_map.values())
        eval_rank = all_ranks[len(all_ranks) // 2]
        eval_lora = project_wagg_to_client(
            global_wagg, eval_rank,
            method if method != "homo_r8" else "flexlora",
            0.01, device,
        )

        # Zero-shot ROUGE-L on unseen clients (FlexLoRA's primary metric)
        eval_model = _load_eval_model(base_model, eval_lora, eval_rank, device)
        metrics = evaluate_unseen_clients(eval_model, tokenizer, unseen_samples, device)
        del eval_model
        torch.cuda.empty_cache()
        base_model.to(device)

        round_time = time.time() - round_start
        record = {
            "round": round_num,
            "avg_loss": float(np.mean(round_losses)),
            "round_time_s": round_time,
            **metrics,
        }
        round_results.append(record)
        pbar.set_postfix({"rouge_l": f"{metrics['rouge_l_unseen']:.4f}",
                          "loss": f"{np.mean(round_losses):.4f}"})
        logger.log(f"  Round {round_num} | rouge_l={metrics['rouge_l_unseen']:.4f} "
                   f"| loss={np.mean(round_losses):.4f} | time={round_time:.1f}s")

    logger.save(round_results)
    return {"method": method, "seed": seed, "rounds": round_results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="flexlora", choices=COMPARISON_METHODS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    model, tokenizer = load_base_model(args.device)

    print(f"Loading Dolly-15K (seed={args.seed})...")
    train_datasets, unseen_samples = load_dolly_federated(tokenizer, seed=args.seed)
    print(f"  {len(train_datasets)} train clients, {len(unseen_samples)} unseen eval clients")
    for i, ds in enumerate(train_datasets[:5]):
        print(f"  client {i}: {len(ds)} samples")

    runs = [(m, s) for m in COMPARISON_METHODS for s in SEEDS] if args.all else [(args.method, args.seed)]

    for method, seed in runs:
        out_file = os.path.join(RESULTS_DIR, f"{method}_seed{seed}.json")
        if os.path.exists(out_file):
            print(f"Skipping {method} seed={seed} (exists)")
            continue
        print(f"\n{'='*60}")
        print(f"Running: method={method} seed={seed}")
        print(f"{'='*60}")
        # Reload Dolly with per-run seed for data partitioning
        train_datasets, unseen_samples = load_dolly_federated(tokenizer, seed=seed)
        run_comparison(method, model, tokenizer, train_datasets, unseen_samples,
                       seed, args.device, RESULTS_DIR)


if __name__ == "__main__":
    main()
