"""
End-to-end pipeline test — 2 rounds, 2 clients, 10 steps each (~5 min).
Validates the W_agg global state design: correct shapes for all ranks each round.

Usage: python test_pipeline.py
"""

import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.aggregation.spa import SPAAggregator
from src.aggregation.flexlora import FlexLoRAAggregator
from src.aggregation.fedavg_homo import HomoAggregator, HeteroPadAggregator
from src.aggregation.spa_momentum import SPAMomentumAggregator
from src.clients.lora_client import train_client
from config.base_config import MODEL_NAME, TARGET_MODULES

DEVICE = "cuda:0"
TEST_RANKS = [4, 8]   # simulate edge + mid-range client


def make_toy_dataset(tokenizer, n=20):
    texts = [
        f"Review: The food was {'great' if i%2==0 else 'terrible'}. Rating: {(i%5)+1} stars"
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


def project_wagg(global_wagg, rank, tau=0.0):
    """Project W_agg to (A, B) for given rank."""
    return {
        layer: dict(zip(("B", "A"),
                        SPAAggregator.project_to_rank(w, rank, tau=tau, device=DEVICE)))
        for layer, w in global_wagg.items()
    }


def test_aggregator(name, aggregator, base_model, tokenizer, client_ranks):
    print(f"\n--- Testing {name} ---")
    toy_ds = make_toy_dataset(tokenizer)
    global_wagg = None
    extract_method = "ab_pair" if name == "hetero_pad" else "full_w"

    for round_num in range(1, 3):
        aggregator.reset()
        total_n = len(client_ranks) * 10

        for rank in client_ranks:
            # Project global W_agg → (A, B) matched to this client's rank
            if global_wagg is None:
                client_global = None
            else:
                client_global = project_wagg(global_wagg, rank)

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
                extract_method=extract_method,
            )

            w = 10 / total_n
            if name == "hetero_pad":
                aggregator.update(weights, w, {}, {})
            else:
                aggregator.update(weights, w)

            print(f"  Round {round_num} | rank={rank} | loss={loss:.4f}")

        # Build new W_agg from aggregator
        if name == "hetero_pad":
            global_wagg = {
                k: (aggregator._b_accum[k] @ aggregator._a_accum[k]).cpu()
                for k in aggregator._a_accum
            }
        else:
            global_wagg = {k: v.cpu() for k, v in aggregator.get_global().items()}

        # Verify shapes are correct for ALL ranks after projection
        for test_rank in client_ranks:
            proj = project_wagg(global_wagg, test_rank)
            sample = next(iter(proj.values()))
            A_shape, B_shape = sample["A"].shape, sample["B"].shape
            assert A_shape[0] == test_rank, \
                f"A rank mismatch: expected {test_rank}, got {A_shape[0]}"
            assert B_shape[1] == test_rank, \
                f"B rank mismatch: expected {test_rank}, got {B_shape[1]}"
            print(f"  Round {round_num} | rank={test_rank} projection: "
                  f"A={A_shape}, B={B_shape} ✓")

    print(f"  {name} PASSED\n")


def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True,
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")

    test_aggregator("homo_r8",    HomoAggregator(rank=8),            base_model, tokenizer, [8, 8])
    test_aggregator("hetero_pad", HeteroPadAggregator(max_rank=32),  base_model, tokenizer, TEST_RANKS)
    test_aggregator("flexlora",   FlexLoRAAggregator(max_rank=32),   base_model, tokenizer, TEST_RANKS)
    test_aggregator("hetero_spa", SPAAggregator(max_rank=32, tau=0.01), base_model, tokenizer, TEST_RANKS)
    test_aggregator("spa_m",      SPAMomentumAggregator(max_rank=32, beta=0.9, gamma=1.0,
                                                        use_consensus=True, consensus_rank=4),
                    base_model, tokenizer, TEST_RANKS)

    print("=" * 50)
    print("ALL TESTS PASSED — ready for full experiments")
    print("=" * 50)


if __name__ == "__main__":
    main()
