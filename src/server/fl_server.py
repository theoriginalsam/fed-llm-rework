"""
Federated Learning server orchestration loop.

Handles all 5 methods uniformly via the aggregator registry.
Supports: homo_r4, homo_r8, hetero_pad, flexlora, hetero_spa.
"""

import random
import time
import json
import os
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from tqdm import tqdm

from config.base_config import (
    NUM_CLIENTS, CLIENTS_PER_ROUND, NUM_ROUNDS, STEPS_PER_ROUND,
    LR, BATCH_SIZE, GRAD_ACCUM_STEPS, MAX_RANK, TARGET_MODULES,
    RANK_DISTRIBUTION,
)
from src.aggregation import AGGREGATOR_REGISTRY
from src.aggregation.fedavg_homo import HomoAggregator, HeteroPadAggregator
from src.aggregation.spa import SPAAggregator
from src.aggregation.flexlora import FlexLoRAAggregator
from src.clients.lora_client import train_client
from src.evaluation.metrics import evaluate_model
from src.utils.logging_utils import ExperimentLogger


def build_client_rank_map(rank_distribution: Dict[str, int]) -> Dict[int, int]:
    """
    Returns {client_id: rank} for all clients.
    Client IDs 0..N-1 assigned in order of rank_distribution dict.
    """
    rank_map = {}
    client_id = 0
    for rank_key, count in rank_distribution.items():
        rank = int(rank_key[1:])  # "r4" -> 4
        for _ in range(count):
            rank_map[client_id] = rank
            client_id += 1
    return rank_map


def get_method_rank(method: str) -> Optional[int]:
    """For homo methods, return the fixed rank. For hetero, return None."""
    if method == "homo_r4":
        return 4
    elif method == "homo_r8":
        return 8
    return None


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
    """
    Main FL training loop.

    Args:
        method: one of "homo_r4", "homo_r8", "hetero_pad", "flexlora", "hetero_spa"
        base_model: frozen pre-trained model (HuggingFace)
        tokenizer: tokenizer for the model
        client_datasets: list of per-client datasets (length NUM_CLIENTS)
        test_dataset: evaluation dataset
        dataset_config: dict from dataset_configs.py
        seed: random seed for client selection
        alpha: Dirichlet non-IID parameter
        results_dir: where to save per-round results
        spa_tau: spectral threshold for SPA (0.0 = equivalent to FlexLoRA)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Build client rank assignment
    homo_rank = get_method_rank(method)
    if homo_rank is not None:
        client_rank_map = {cid: homo_rank for cid in range(NUM_CLIENTS)}
    else:
        client_rank_map = build_client_rank_map(RANK_DISTRIBUTION)

    # Determine extract method
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

    # Global weight state: {layer_key: {"A": tensor, "B": tensor}} or None
    global_weights: Optional[Dict] = None

    round_results = []

    for round_num in range(1, num_rounds + 1):
        round_start = time.time()
        logger.log(f"\n=== Round {round_num}/{num_rounds} ===")

        # Select clients for this round
        selected = random.sample(range(NUM_CLIENTS), CLIENTS_PER_ROUND)
        total_samples = sum(len(client_datasets[cid]) for cid in selected)
        aggregator.reset()

        round_losses = []

        for cid in selected:
            rank = client_rank_map[cid]
            dataset = client_datasets[cid]
            n_samples = len(dataset)
            weight = n_samples / total_samples

            # Get the global weights projected to this client's rank
            if global_weights is None:
                client_global = None
            elif method == "hetero_pad":
                # For pad, distribute A/B pair truncated to rank
                client_global = {
                    layer: {"A": mats["A"][:rank, :], "B": mats["B"][:, :rank]}
                    for layer, mats in global_weights.items()
                } if global_weights else None
            else:
                # SPA/FlexLoRA/FLoRA/Homo: project W_agg to rank each round
                client_global = {
                    layer: {"A": mats["A"][:rank, :] if mats["A"].shape[0] >= rank else mats["A"],
                            "B": mats["B"][:, :rank] if mats["B"].shape[1] >= rank else mats["B"]}
                    for layer, mats in global_weights.items()
                } if global_weights else None

            weights, loss = train_client(
                base_model=base_model,
                tokenizer=tokenizer,
                rank=rank,
                target_modules=TARGET_MODULES,
                global_weights=client_global,
                dataset=dataset,
                steps=STEPS_PER_ROUND,
                batch_size=BATCH_SIZE,
                grad_accum=GRAD_ACCUM_STEPS,
                lr=LR,
                device=device,
                extract_method=extract_method,
            )
            round_losses.append(loss)

            # Feed to aggregator
            if method == "hetero_pad":
                aggregator.update(weights, weight, d_out_map={}, d_in_map={})
            else:
                aggregator.update(weights, weight)

            logger.log(f"  Client {cid} (rank={rank}): loss={loss:.4f}")

        # Distribute global model back
        if method == "hetero_pad":
            dist = aggregator.distribute(client_rank_map, device)
        elif method in ("flexlora", "hetero_spa"):
            dist = aggregator.distribute(client_rank_map, device)
        elif method in ("homo_r4", "homo_r8"):
            dist = aggregator.distribute(client_rank_map, device)

        # Store first client's weights as representative global (same for all clients)
        first_cid = list(dist.keys())[0]
        global_weights = dist[first_cid]

        # Evaluate
        metrics = evaluate_model(
            base_model=base_model,
            tokenizer=tokenizer,
            global_lora_weights=global_weights,
            rank=client_rank_map[0],  # evaluate at min rank (most conservative)
            test_dataset=test_dataset,
            dataset_config=dataset_config,
            device=device,
        )

        round_time = time.time() - round_start
        round_record = {
            "round": round_num,
            "avg_loss": float(np.mean(round_losses)),
            "round_time_s": round_time,
            **metrics,
        }
        round_results.append(round_record)
        logger.log(f"  Round {round_num} | {metrics} | time={round_time:.1f}s")

    logger.save(round_results)
    return {"method": method, "seed": seed, "alpha": alpha, "rounds": round_results}
