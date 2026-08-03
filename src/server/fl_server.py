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
from tqdm import tqdm

from config.base_config import (
    NUM_CLIENTS, CLIENTS_PER_ROUND, NUM_ROUNDS, STEPS_PER_ROUND,
    LR, BATCH_SIZE, GRAD_ACCUM_STEPS, MAX_RANK, TARGET_MODULES,
    RANK_DISTRIBUTION,
)
from src.aggregation.spa import SPAAggregator
from src.aggregation.flexlora import FlexLoRAAggregator
from src.aggregation.fedavg_homo import HomoAggregator, HeteroPadAggregator
from src.aggregation.spa_momentum import SPAMomentumAggregator
from src.aggregation.hetlora import HetLoRAAggregator
from src.aggregation.hetlora_m import HetLoRAMomentumAggregator
from src.clients.lora_client import train_client
from src.evaluation.metrics import evaluate_model
from src.utils.logging_utils import ExperimentLogger
from analysis.ema_eval_hook import EMAEvalHook


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

    from src.aggregation.spa_momentum import SPAMomentumAggregator

    client_lora = {}
    for layer_key, w_agg in global_wagg.items():
        if method == "hetero_spa":
            B, A = SPAAggregator.project_to_rank(w_agg, rank, tau=tau, device=device)
        elif method == "spa_m":
            B, A = SPAMomentumAggregator.project_to_rank(w_agg, rank, device=device)
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
    batch_size: int = BATCH_SIZE,
    eval_rank_strategy: str = "median",
    rank_weighted: bool = True,
    hetlora_m_beta: float = 0.5,
    spa_m_beta: float = 0.9,
    clients_per_round: int = CLIENTS_PER_ROUND,
    rank_distribution: Optional[Dict[str, int]] = None,
    ema_eval: bool = False,
    save_adapters_dir: Optional[str] = None,
) -> Dict[str, Any]:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    fixed_rank = get_fixed_rank(method)
    if fixed_rank is not None:
        client_rank_map = {cid: fixed_rank for cid in range(NUM_CLIENTS)}
    else:
        dist = rank_distribution if rank_distribution is not None else RANK_DISTRIBUTION
        client_rank_map = build_client_rank_map(dist)

    # Extract method: hetero_pad and hetlora send (A,B) pairs; others send full ΔW
    extract_method = "ab_pair" if method in ("hetero_pad", "hetlora", "hetlora_m") else "full_w"

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
    elif method == "spa_m":
        aggregator = SPAMomentumAggregator(max_rank=MAX_RANK, beta=spa_m_beta, gamma=1.0,
                                           use_consensus=True, consensus_rank=4)
    elif method == "hetlora":
        aggregator = HetLoRAAggregator(max_rank=MAX_RANK)
    elif method == "hetlora_m":
        aggregator = HetLoRAMomentumAggregator(max_rank=MAX_RANK, beta=hetlora_m_beta)
    else:
        raise ValueError(f"Unknown method: {method}")

    logger = ExperimentLogger(method, seed, alpha, results_dir)
    logger.log(f"Starting {method} | seed={seed} | alpha={alpha} | tau={spa_tau}")

    _EMA_CONTROL_METHODS = {"homo_r8", "hetero_pad", "flexlora"}
    ema_hook = EMAEvalHook(beta=0.5, space="deltaw") if (ema_eval and method in _EMA_CONTROL_METHODS) else None

    # Global state: W_agg per layer {layer_key: tensor(d_out, d_in)}
    # HetLoRA uses global_ba {layer_key: {"A": ..., "B": ...}} at max_rank instead.
    global_wagg: Optional[Dict[str, torch.Tensor]] = None
    global_ba_hetlora: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    global_ba_hetlora_m_raw: Optional[Dict[str, Dict[str, torch.Tensor]]] = None  # → clients
    global_ba_hetlora_m_ema: Optional[Dict[str, Dict[str, torch.Tensor]]] = None  # → eval only

    round_results = []

    round_pbar = tqdm(range(1, num_rounds + 1),
                      desc=f"{method}|s{seed}|α{alpha}",
                      unit="round")

    for round_num in round_pbar:
        round_start = time.time()

        eligible = [cid for cid in range(NUM_CLIENTS) if len(client_datasets[cid]) > 0]
        selected = random.sample(eligible, min(clients_per_round, len(eligible)))
        aggregator.reset()

        # Rank-weighted aggregation: weight ∝ rank × dataset_size
        if rank_weighted:
            total_rw = sum(client_rank_map[cid] * len(client_datasets[cid]) for cid in selected)
        else:
            total_rw = sum(len(client_datasets[cid]) for cid in selected)

        round_losses = []

        for client_idx, cid in enumerate(selected):
            rank = client_rank_map[cid]
            if rank_weighted:
                weight = (rank * len(client_datasets[cid])) / total_rw
            else:
                weight = len(client_datasets[cid]) / total_rw

            if method == "hetlora":
                if global_ba_hetlora is None:
                    client_global = None
                else:
                    client_global = HetLoRAAggregator.distribute_to_client(
                        global_ba_hetlora, rank, device
                    )
            elif method == "hetlora_m":
                if global_ba_hetlora_m_raw is None:
                    client_global = None
                else:
                    client_global = HetLoRAAggregator.distribute_to_client(
                        global_ba_hetlora_m_raw, rank, device
                    )
            elif global_wagg is None:
                client_global = None
            else:
                client_global = project_wagg_to_client(
                    global_wagg, rank, method, spa_tau, device
                )

            adapter_prefix = None
            if save_adapters_dir is not None:
                adapter_prefix = os.path.join(
                    save_adapters_dir,
                    f"round{round_num:03d}_client{cid:03d}_r{rank}",
                )

            weights, loss = train_client(
                base_model=base_model,
                tokenizer=tokenizer,
                rank=rank,
                target_modules=TARGET_MODULES,
                global_weights=client_global,
                dataset=client_datasets[cid],
                steps=STEPS_PER_ROUND,
                batch_size=batch_size,
                grad_accum=GRAD_ACCUM_STEPS,
                lr=LR,
                device=device,
                extract_method=extract_method,
                pbar_desc=f"  R{round_num} C{client_idx+1}/{CLIENTS_PER_ROUND} r={rank}",
                adapter_save_prefix=adapter_prefix,
            )
            round_losses.append(loss)

            if method == "hetero_pad":
                aggregator.update(weights, weight, {}, {})
            elif method in ("hetlora", "hetlora_m"):
                # HetLoRA / HetLoRA-M ignore data-volume weight; Frobenius weights computed internally
                aggregator.update(weights)
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
        elif method == "hetlora":
            global_ba_hetlora = aggregator.get_global_ba()
            global_wagg = {k: v["B"] @ v["A"] for k, v in global_ba_hetlora.items()}
        elif method == "hetlora_m":
            global_ba_hetlora_m_ema = aggregator.get_global_ba()   # EMA state → eval
            global_ba_hetlora_m_raw = aggregator.get_raw_ba()       # raw aggregate → clients
            global_wagg = {k: v["B"] @ v["A"] for k, v in global_ba_hetlora_m_ema.items()}
        else:
            # SPA / FlexLoRA / Homo already accumulate W_agg directly
            global_wagg = {k: v.cpu() for k, v in aggregator.get_global().items()}

        # Eval rank strategy: median = typical deployment device
        all_ranks = sorted(client_rank_map.values())
        if eval_rank_strategy == "median":
            eval_rank = all_ranks[len(all_ranks) // 2]
        elif eval_rank_strategy == "max":
            eval_rank = all_ranks[-1]
        else:  # "min" — original conservative setting
            eval_rank = all_ranks[0]
        if method == "hetlora":
            eval_lora = HetLoRAAggregator.distribute_to_client(
                global_ba_hetlora, eval_rank, device
            )
        elif method == "hetlora_m":
            eval_lora = HetLoRAAggregator.distribute_to_client(
                global_ba_hetlora_m_ema, eval_rank, device
            )
        else:
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

        ema_metrics = {}
        if ema_hook is not None:
            smoothed_wagg = ema_hook.update(global_wagg)
            ema_eval_lora = project_wagg_to_client(smoothed_wagg, eval_rank, method, spa_tau, device)
            ema_metrics = evaluate_model(
                base_model=base_model,
                tokenizer=tokenizer,
                global_lora_weights=ema_eval_lora,
                rank=eval_rank,
                test_dataset=test_dataset,
                dataset_config=dataset_config,
                device=device,
            )

        round_time = time.time() - round_start
        record = {"round": round_num, "avg_loss": float(np.mean(round_losses)),
                  "round_time_s": round_time, **metrics}
        if ema_hook is not None:
            record["acc_raw_eval"] = metrics.get("accuracy")
            record["acc_ema_eval"] = ema_metrics.get("accuracy")
        round_results.append(record)

        # Update round-level bar with key metrics
        bar_metrics = {k: f"{v:.4f}" for k, v in metrics.items()
                       if isinstance(v, float)}
        bar_metrics["loss"] = f"{np.mean(round_losses):.4f}"
        round_pbar.set_postfix(bar_metrics)
        logger.log(f"  Round {round_num} | {metrics} | time={round_time:.1f}s")

    logger.save(round_results)
    return {"method": method, "seed": seed, "alpha": alpha, "rounds": round_results}
