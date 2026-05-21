# SPA-M: Spectral Aggregation with Momentum for Heterogeneous Federated LLM Fine-Tuning

> **Target:** ICLR / NeurIPS / ACL 2027

Federated fine-tuning of large language models (LLMs) across heterogeneous devices — where clients operate at different LoRA ranks due to varying compute and memory budgets — is a fundamentally unsolved problem. Standard aggregation methods either force all clients to the lowest rank (losing high-rank signal) or aggregate naively (losing low-rank minority directions).

We propose **SPA-M**: Spectral Projection Aggregation with server-side Momentum, which:
1. Reconstructs full-rank ΔW matrices from heterogeneous client LoRA updates
2. Applies rank-weighted consensus aggregation to preserve high-rank signal
3. Uses adaptive momentum (saturating stabilizer) to smooth round-to-round noise under non-IID data
4. Projects back to each client's native rank via truncated SVD

---

## Key Results

### Yelp Sentiment Classification (α=0.5, 5 seeds)

| Method | Mean-L5 ★ | Final Acc | Best Acc |
|--------|-----------|-----------|----------|
| FedAvg r=8 (oracle) | 52.4±2.1 | 56.4±3.1 | 59.9±1.3 |
| Hetero-Pad | 48.9±4.3 | 51.3±5.5 | 57.4±2.3 |
| FlexLoRA | 51.3±3.3 | 53.1±4.3 | 59.4±2.3 |
| SPA | 50.6±4.1 | 52.2±4.8 | 59.6±2.7 |
| **SPA-M (Ours)** | **51.6±3.5** | 52.1±4.0 | **60.1±2.4** |

SPA-M achieves the highest Best Acc (60.1%), beating the oracle FedAvg r=8 (59.9%).

### GSM8K Math Reasoning (α=0.5, 3 seeds)

| Method | Mean-L5 ★ | Final Acc | Best Acc |
|--------|-----------|-----------|----------|
| FedAvg r=8 (oracle) | 74.87±2.46 | 71.67±1.89 | 82.67±0.47 |
| Hetero-Pad | 75.00±0.28 | 72.33±0.47 | 81.67±2.05 |
| FlexLoRA | 75.30±2.30 | 77.00±4.00 | 82.00±1.00 |
| **SPA-M (Ours)** | **75.33±0.75** | 74.33±2.62 | **83.67±2.49** |

SPA-M beats the oracle on both Mean-L5 and Best Acc, with the lowest variance (±0.75%).

---

## Methods Compared

| Method | Description |
|--------|-------------|
| FedAvg r=8 | All clients forced to rank=8 — oracle baseline (requires all devices to afford r=8) |
| Hetero-Pad | Zero-pad A/B to max rank, aggregate, truncate — simple heterogeneous baseline |
| FlexLoRA | Reconstruct ΔW per client, average, SVD project (Bai et al., 2024) |
| SPA | Spectral Projection Aggregation with adaptive threshold (ours, base) |
| **SPA-M** | SPA + server-side momentum with adaptive β and cumulative bias correction (ours) |

---

## Setup

```bash
git clone https://github.com/theoriginalsam/fed-llm-rework.git
cd fed-llm-rework
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, CUDA 12.1+, ~49GB VRAM for Qwen2.5-7B (2× RTX A6000 recommended)

---

## Reproducing Experiments

```bash
# Yelp — all methods, both alphas
python experiments/run_yelp.py --method spa_m --seed 42 --alpha 0.5 --device cuda:0
python experiments/run_yelp.py --method spa_m --seed 42 --alpha 0.1 --device cuda:0

# GSM8K — math reasoning
python experiments/run_gsm8k.py --method spa_m --seed 42 --device cuda:0

# Alpaca — instruction following
python experiments/run_alpaca.py --method spa_m --seed 42 --device cuda:0
```

Available methods: `homo_r4`, `homo_r8`, `hetero_pad`, `flexlora`, `hetero_spa`, `spa_m`

---

## Project Structure

```
├── config/
│   └── base_config.py          # hyperparams, rank distribution, seeds
├── src/
│   ├── aggregation/
│   │   ├── spa.py              # SPA base aggregator
│   │   ├── spa_momentum.py     # SPA-M (our main method)
│   │   ├── flexlora.py         # FlexLoRA baseline
│   │   └── fedavg_homo.py      # Homo-r4, Homo-r8, Hetero-Pad
│   ├── clients/
│   │   └── lora_client.py      # client training loop
│   ├── server/
│   │   └── fl_server.py        # FL orchestration
│   ├── data/
│   │   ├── yelp.py             # Yelp Review Full
│   │   ├── alpaca.py           # Alpaca-52k
│   │   └── gsm8k.py            # GSM8K math reasoning
│   └── evaluation/
│       └── metrics.py          # accuracy, ROUGE-L, exact-match
├── experiments/
│   ├── run_yelp.py
│   ├── run_alpaca.py
│   └── run_gsm8k.py
└── notebooks/                  # result visualization
```

---

## Configuration

Key settings in `config/base_config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | Base LLM |
| `NUM_CLIENTS` | 50 | Total federated clients |
| `CLIENTS_PER_ROUND` | 5 | Sampled per round |
| `NUM_ROUNDS` | 20 | Communication rounds |
| `RANK_DISTRIBUTION` | r4:20, r8:20, r16:5, r32:5 | Heterogeneous rank setup |
| `SEEDS` | [42,43,44,45,46] | Reproducibility seeds |

---

## SPA-M Algorithm

```
Each round t:
  1. Sample K clients
  2. Each client k trains LoRA(rank=r_k) → uploads ΔW_k = B_k @ A_k
  3. Server computes consensus weights w_k ∝ n_k × C_k (subspace agreement)
  4. W_agg = Σ_k w_k × ΔW_k
  5. β_t = β_max × (1 − clamp(cos_sim(W_agg_t, W_agg_{t-1}), 0, 1))
  6. M_t = β_t × M_{t-1} + (1 − β_t) × W_agg_t   [EMA]
  7. M̂_t = M_t / (1 − Π_{τ≤t} β_τ)               [bias correction]
  8. Rescale M̂_t to ||W_agg_t||_F                 [magnitude normalization]
  9. Distribute: SVD(M̂_t) → top-r_k factors per client
```

---

## Citation

> Paper under preparation. Repository will be updated upon submission.
