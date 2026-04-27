"""
Federated Learning server orchestration loop.

Global state is W_agg: the full aggregated weight matrix per layer.
Each round, W_agg is projected to each client's rank before training.
This ensures correct injection regardless of rank heterogeneity.
"""

import random
import time
import json
import os
import torch
import numpy as np
from typing import Dict, List, Optional, Any

from config.base_config import (
    NUM_CLIENTS, CLIENTS_PER_ROUND, NUM_ROUNDS, STEPS_PER_ROUND,
    LR, BATCH_SIZE, GRAD_ACCUM_STEPS, MAX_RANK, TARGET_MODULES,
    RANK_DISTRIBUTION,
)
from src.aggregation.spa import SPAAggregator
from src.aggregation.flexlora import FlexLoRAAggregator
from src.aggregation.fedavg_homo import HomoAggregator, HeteroPadAggregator
from src.clients.lora_client import train_client
from src.evaluation.metrics import evaluate_model
from src.utils.logging_utils import ExperimentLogger


def build_client_rank_map(rank_distribution: Dict[str, int]) -> Dict[int, int]:
    rank_map = {}
    client_id = 0
    for rank_key, count in rank_distribution.items():
        rank = int(rank_key[1:])
        for _ in range(count):
            rank_map[client_id] = rank
            client_id += 1
    return rank_map


def get_fixed_rank(method: str) -> Optional[int]:
    if method == "homo_r4":
        return 4
    if method == "homo_r8":
        return 8
    return None


def project_wagg_to_client(
    global_wagg: Dict[str, torch.Tensor],
    rank: int,
    method: str,
    tau: float = 0.01,
    device: str = "cuda",
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Project global W_agg matrices to (A, B) for a client of given rank.
    Works for all methods: SPA, FlexLoRA, Homo, HeteroPad.
    """
    from src.aggregation.spa import SPAAggregator

    client_lora = {}
    for layer_key, w_agg in global_wagg.items():
        if method == "hetero_spa":
            B, A = SPAAggregator.project_to_rank(w_agg, rank, tau=tau, device=device)
        else:
            # FlexLoRA, Homo, HeteroPad all use tau=0
            B, A = SPAAggregator.project_to_rank(w_agg, rank, tau=0.0, device=device)
        client_lora[layer_key] = {"A": A, "B": B}
    return client_lora


def run_federated(
    method: str,
    base_model,
    tokenizer,
    client_datasets: List,
    test_dataset,
    dataset_config: Dict,
    seed: int,
    alpha: float,
    results_dir: str,
    device: str = "cuda",
    num_rounds: int = NUM_ROUNDS,
    spa_tau: float = 0.01,
) -> Dict[str, Any]:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    fixed_rank = get_fixed_rank(method)
    if fixed_rank is not None:
        client_rank_map = {cid: fixed_rank for cid in range(NUM_CLIENTS)}
    else:
        client_rank_map = build_client_rank_map(RANK_DISTRIBUTION)

    # Extract method: hetero_pad sends (A,B) pairs; others send full ΔW
    extract_method = "ab_pair" if method == "hetero_pad" else "full_w"

    # Build aggregator
    if method == "homo_r4":
        aggregator = HomoAggregator(rank=4, max_rank=MAX_RANK)
    elif method == "homo_r8":
        aggregator = HomoAggregator(rank=8, max_rank=MAX_RANK)
    elif method == "hetero_pad":
        aggregator = HeteroPadAggregator(max_rank=MAX_RANK)
    elif method == "flexlora":
        aggregator = FlexLoRAAggregator(max_rank=MAX_RANK)
    elif method == "hetero_spa":
        aggregator = SPAAggregator(max_rank=MAX_RANK, tau=spa_tau)
    else:
        raise ValueError(f"Unknown method: {method}")

    logger = ExperimentLogger(method, seed, alpha, results_dir)
    logger.log(f"Starting {method} | seed={seed} | alpha={alpha} | tau={spa_tau}")

    # Global state: W_agg per layer {layer_key: tensor(d_out, d_in)}
    # None until after first aggregation round
    global_wagg: Optional[Dict[str, torch.Tensor]] = None

    round_results = []

    for round_num in range(1, num_rounds + 1):
        round_start = time.time()
        logger.log(f"\n=== Round {round_num}/{num_rounds} ===")

        selected = random.sample(range(NUM_CLIENTS), CLIENTS_PER_ROUND)
        total_samples = sum(len(client_datasets[cid]) for cid in selected)
        aggregator.reset()

        round_losses = []

        for cid in selected:
            rank = client_rank_map[cid]
            weight = len(client_datasets[cid]) / total_samples

            # Project global W_agg to this client's rank
            if global_wagg is None:
                client_global = None
            else:
                client_global = project_wagg_to_client(
                    global_wagg, rank, method, spa_tau, device
                )

            weights, loss = train_client(
                base_model=base_model,
                tokenizer=tokenizer,
                rank=rank,
                target_modules=TARGET_MODULES,
                global_weights=client_global,
                dataset=client_datasets[cid],
                steps=STEPS_PER_ROUND,
                batch_size=BATCH_SIZE,
                grad_accum=GRAD_ACCUM_STEPS,
                lr=LR,
                device=device,
                extract_method=extract_method,
            )
            round_losses.append(loss)

            # Feed client update to aggregator
            if method == "hetero_pad":
                aggregator.update(weights, weight, {}, {})
            else:
                aggregator.update(weights, weight)

            logger.log(f"  Client {cid} (rank={rank}): loss={loss:.4f}")

        # Get new W_agg from aggregator
        if method == "hetero_pad":
            # HeteroPad accumulates padded A/B → reconstruct W_agg
            new_wagg = {}
            for layer_key in aggregator._a_accum:
                A = aggregator._a_accum[layer_key]   # (max_rank, d_in)
                B = aggregator._b_accum[layer_key]   # (d_out, max_rank)
                new_wagg[layer_key] = (B @ A).cpu()  # (d_out, d_in)
            global_wagg = new_wagg
        else:
            # SPA / FlexLoRA / Homo already accumulate W_agg directly
            global_wagg = {k: v.cpu() for k, v in aggregator.get_global().items()}

        # Evaluate at the minimum rank (most conservative / edge device perspective)
        eval_rank = min(client_rank_map.values())
        eval_lora = project_wagg_to_client(global_wagg, eval_rank, method, spa_tau, device)

        metrics = evaluate_model(
            base_model=base_model,
            tokenizer=tokenizer,
            global_lora_weights=eval_lora,
            rank=eval_rank,
            test_dataset=test_dataset,
            dataset_config=dataset_config,
            device=device,
        )

        round_time = time.time() - round_start
        record = {"round": round_num, "avg_loss": float(np.mean(round_losses)),
                  "round_time_s": round_time, **metrics}
        round_results.append(record)
        logger.log(f"  Round {round_num} | {metrics} | time={round_time:.1f}s")

    logger.save(round_results)
    return {"method": method, "seed": seed, "alpha": alpha, "rounds": round_results}
