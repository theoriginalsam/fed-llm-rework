"""
Quick end-to-end pipeline test.
Runs 2 rounds, 2 clients, 10 steps each — catches bugs in ~5 minutes.
Does NOT save to results/. Use this before launching full experiments.

Usage:
    python test_pipeline.py
"""

import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from src.aggregation.spa import SPAAggregator
from src.aggregation.flexlora import FlexLoRAAggregator
from src.aggregation.fedavg_homo import HomoAggregator, HeteroPadAggregator
from src.clients.lora_client import (
    make_lora_model, inject_lora_weights, extract_lora_weights, train_client
)
from config.base_config import MODEL_NAME, TARGET_MODULES

DEVICE = "cuda:0"
TEST_RANKS = [4, 8]


def make_toy_dataset(tokenizer, n=20):
    """Tiny in-memory dataset for smoke testing."""
    texts = [
        f"Review: The food was {'great' if i % 2 == 0 else 'terrible'}. Rating: {(i%5)+1} stars"
        for i in range(n)
    ]
    samples = []
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", max_length=64,
                        truncation=True, padding="max_length")
        samples.append({
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        })
    return samples


def test_aggregator(name, aggregator, base_model, tokenizer, client_ranks):
    print(f"\n--- Testing {name} ---")
    toy_ds = make_toy_dataset(tokenizer)

    global_weights = None
    for round_num in range(1, 3):
        aggregator.reset()
        total_n = sum(10 for _ in client_ranks)

        for rank in client_ranks:
            # Build minimal global weight dict for inject
            client_global = None
            if global_weights is not None:
                client_global = {
                    k: {"A": v["A"][:rank], "B": v["B"][:, :rank]}
                    for k, v in global_weights.items()
                    if v["A"].shape[0] >= rank
                } if global_weights else None

            weights, loss = train_client(
                base_model=base_model,
                tokenizer=tokenizer,
                rank=rank,
                target_modules=TARGET_MODULES,
                global_weights=client_global,
                dataset=toy_ds,
                steps=10,
                batch_size=2,
                grad_accum=2,
                lr=2e-4,
                device=DEVICE,
                extract_method="full_w" if name != "hetero_pad" else "ab_pair",
            )
            w = 10 / total_n

            if name == "hetero_pad":
                aggregator.update(weights, w, {}, {})
            else:
                aggregator.update(weights, w)

            print(f"  Round {round_num} | rank={rank} | loss={loss:.4f} | "
                  f"keys={list(weights.keys())[:2]}")

        # Distribute
        client_rank_map = {i: r for i, r in enumerate(client_ranks)}
        dist = aggregator.distribute(client_rank_map, DEVICE)
        global_weights = dist[0]  # representative global

        print(f"  Round {round_num} aggregation OK | "
              f"sample key shapes: A={list(global_weights.values())[0]['A'].shape}, "
              f"B={list(global_weights.values())[0]['B'].shape}")

    print(f"  {name} PASSED")


def main():
    print("Loading model (this takes ~1 min)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    vram = torch.cuda.memory_allocated(0) / 1e9
    print(f"Model loaded. VRAM used: {vram:.1f} GB")

    # Test all aggregators
    test_aggregator("homo_r8",    HomoAggregator(rank=8),         base_model, tokenizer, [8, 8])
    test_aggregator("hetero_pad", HeteroPadAggregator(max_rank=32), base_model, tokenizer, [4, 8])
    test_aggregator("flexlora",   FlexLoRAAggregator(max_rank=32), base_model, tokenizer, [4, 8])
    test_aggregator("hetero_spa", SPAAggregator(max_rank=32, tau=0.01), base_model, tokenizer, [4, 8])

    print("\n==========================================")
    print("ALL TESTS PASSED — ready for full experiments")
    print("==========================================")


if __name__ == "__main__":
    main()
