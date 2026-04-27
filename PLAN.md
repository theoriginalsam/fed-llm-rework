# SPA Rework — Full-Scale Experiment Plan
**Status:** In Progress | **GPU:** Blackwell | **Target:** ICLR / NeurIPS / ACL 2027

---

## 0. Why We're Reworking

Reviews scored -2, -1, -2, +1. Three hard rejections. The core problems:

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| Missing baselines | FlexLoRA/FLoRA cited but not compared | Implement + run all |
| Narrow experiments | 1 dataset, α=0.5 only, 10 rounds, 3 seeds | 3 datasets, α={0.5,0.1}, 20 rounds, 5 seeds |
| Novelty overlap with FlexLoRA | SPA core ≈ FlexLoRA core | Formal spectral theorem + convergence bound |
| Hallucination rate undefined | Yelp is classification, not generation | Remove from classification; add generation task |
| Privacy claims unsupported | MIA AUC ≈ 0.50 for BOTH methods | Reframe honestly; remove "privacy-preserving" from title if needed |
| Comm cost framing wrong | Savings come from hetero ranks, not SPA | Attribute correctly in revised paper |

---

## 1. Folder Structure

```
rework/
├── PLAN.md                          ← this file
├── requirements.txt
├── config/
│   ├── base_config.py               ← shared hyperparams
│   ├── dataset_configs.py           ← per-dataset settings
│   └── model_configs.py             ← model loading utils
├── src/
│   ├── aggregation/
│   │   ├── __init__.py
│   │   ├── spa.py                   ← SPA (our method)
│   │   ├── flexlora.py              ← FlexLoRA baseline (Bai 2024)
│   │   ├── flora.py                 ← FLoRA baseline (Wang AAAI 2024)
│   │   ├── hetlora.py               ← HetLoRA baseline (EMNLP 2024)
│   │   └── fedavg_homo.py           ← Homo-r4, Homo-r8, Hetero-Pad
│   ├── clients/
│   │   ├── __init__.py
│   │   └── lora_client.py           ← client training logic
│   ├── server/
│   │   ├── __init__.py
│   │   └── fl_server.py             ← orchestration loop
│   ├── data/
│   │   ├── __init__.py
│   │   ├── yelp.py                  ← Yelp Review Full (classification)
│   │   ├── alpaca.py                ← Alpaca-52k (instruction following)
│   │   └── gsm8k.py                 ← GSM8K (math reasoning, generation)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py               ← accuracy, F1, perplexity, ROUGE, exact-match
│   │   ├── mia.py                   ← proper MIA with full protocol
│   │   └── efficiency.py            ← comm cost, SVD timing, memory
│   └── utils/
│       ├── __init__.py
│       ├── logging_utils.py
│       └── checkpoint.py
├── experiments/
│   ├── run_yelp.py                  ← Yelp: all methods, α={0.5,0.1}, 20 rounds, 5 seeds
│   ├── run_alpaca.py                ← Alpaca: all methods, α=0.5, 20 rounds, 3 seeds
│   ├── run_gsm8k.py                 ← GSM8K: all methods, α=0.5, 20 rounds, 3 seeds
│   ├── run_ablation.py              ← rank ratio ablation, client count ablation
│   └── run_overhead.py              ← SVD timing across all layers, all model sizes
├── analysis/
│   ├── spectral_analysis.py         ← singular value spectrum, energy capture
│   ├── convergence_analysis.py      ← empirical convergence curves + theoretical sketch
│   ├── communication_analysis.py    ← honest comm cost breakdown
│   └── privacy_analysis.py          ← MIA results, entropy, honest framing
├── results/
│   ├── yelp/
│   ├── alpaca/
│   ├── gsm8k/
│   └── ablation/
└── notebooks/
    ├── main_visualization.ipynb
    └── ablation_plots.ipynb
```

---

## 2. Methods to Implement

### 2a. Keep (from original Code/)
- **Homo-r4**: All 50 clients forced rank=4
- **Homo-r8**: All 50 clients forced rank=8
- **Hetero-Pad**: Heterogeneous ranks, zero-pad A/B to max_rank, aggregate, truncate

### 2b. New Baselines to Implement

#### FlexLoRA (CRITICAL)
**Paper**: "FlexLoRA: Any-Dimension Low-Rank Adaptation for Federated Learning" (Bai et al., 2024)  
**Algorithm**:
1. Each client k trains LoRA(rank=r_k) → uploads (A_k, B_k)
2. Server reconstructs: ΔW_k = B_k @ A_k  ∈ ℝ^{d_out × d_in}
3. Server aggregates: ΔW_avg = Σ_k (n_k/N) * ΔW_k
4. Server distributes to client j: SVD(ΔW_avg) → top-r_j components
5. A_j = sqrt(Σ_{1:r_j}) @ V^T_{1:r_j}, B_j = U_{:,1:r_j} @ sqrt(Σ_{1:r_j})

**Note**: This is nearly identical to SPA. The differentiation paper must make explicit.

#### FLoRA (Wang et al., AAAI 2024)
**Algorithm**:
1. Frozen base weights W_0
2. Each client uses same-rank LoRA (homogeneous); uploads (A_k, B_k)
3. Server stacks: A_global = [A_1; A_2; ...; A_K] — rank grows with K
4. During inference: h = W_0 x + (1/K) B_global A_global x
5. **Limitation**: Requires homogeneous ranks → adapt for hetero via truncation to min_rank

#### HetLoRA (EMNLP 2024)  
**Algorithm**:
1. Clients train with heterogeneous ranks
2. Low-rank client updates are zero-padded before aggregation (similar to Hetero-Pad)
3. BUT: uses structured sparsity masks so low-rank clients only update a subspace
4. Aggregate only the shared subspace, keep high-rank components from capable clients

### 2c. SPA (Ours) — Improvements
Keep core algorithm. Add:
- **Singular value threshold**: Drop components where σ_i < τ * σ_1 (adaptive denoising)
- **Adaptive rank suggestion**: After round 1, compute optimal r from energy threshold (≥95% energy)
- **Formal theorem**: Cite that truncated SVD is best rank-r approximation in Frobenius norm → projection is lossless up to rank constraint

---

## 3. Datasets

### Dataset 1: Yelp Review Full (KEEP + EXTEND)
- **Task**: 5-class sentiment classification
- **Size**: 650k train, 50k test
- **Partition**: Dirichlet α=0.5 AND α=0.1 across 50 clients
- **Metrics**: Accuracy, F1-macro, Perplexity
- **Remove**: "Hallucination rate" — not meaningful here
- **Rounds**: 20 | **Seeds**: 5 (42,43,44,45,46)

### Dataset 2: Alpaca-52k (NEW — Instruction Following)
- **Task**: Open-ended instruction following (generation)
- **Size**: 52k samples (use 40k train, 12k test)
- **Source**: tatsu-lab/alpaca on HuggingFace
- **Partition**: Dirichlet α=0.5 across 50 clients
- **Metrics**: ROUGE-L, BLEU, Perplexity
- **Hallucination rate**: Meaningful here (empty/off-topic responses)
- **Rounds**: 20 | **Seeds**: 3

### Dataset 3: GSM8K (NEW — Math Reasoning)
- **Task**: Grade-school math word problems (chain-of-thought generation)
- **Size**: 7.5k train, 1.3k test
- **Source**: openai/gsm8k on HuggingFace
- **Partition**: Dirichlet α=0.5 across 50 clients (small dataset → use IID + partitioned)
- **Metrics**: Exact-match accuracy on final answer, Perplexity
- **Rounds**: 20 | **Seeds**: 3
- **Why**: Shows generalization beyond classification; reasoning quality matters

---

## 4. Experimental Grid

### Main Experiments (Full Grid)

| Experiment | Dataset | Methods | α | Rounds | Seeds |
|-----------|---------|---------|---|--------|-------|
| E1 | Yelp | Homo-r4, Homo-r8, Hetero-Pad, FlexLoRA, SPA | 0.5 | 20 | 5 |
| E2 | Yelp | Homo-r8, Hetero-Pad, FlexLoRA, SPA | 0.1 | 20 | 3 |
| E3 | Alpaca | Homo-r4, Homo-r8, Hetero-Pad, FlexLoRA, SPA | 0.5 | 20 | 3 |
| E4 | GSM8K | Homo-r4, Homo-r8, Hetero-Pad, FlexLoRA, SPA | 0.5 | 20 | 3 |

### Ablation Experiments

| Experiment | Variable | Range | Dataset | Rounds | Seeds |
|-----------|---------|-------|---------|--------|-------|
| A1 | Non-IID severity | α={0.5, 0.2, 0.1, 0.05} | Yelp | 20 | 3 |
| A2 | Rank ratio gap | {r4+r32, r8+r16, all-same} | Yelp | 20 | 3 |
| A3 | Number of clients | {20, 50, 100} | Yelp | 20 | 3 |
| A4 | SVD overhead | All layers, d={512,1024,2048,4096} | — | timing | 1 |

### Compute Estimate (Blackwell GPU)
- Qwen2.5-7B in bfloat16: ~14GB VRAM — easily fits
- 100 steps/client, batch_size=4, grad_accum=4 → ~2 min/client
- 5 clients/round × 2 min = 10 min/round
- E1: 20 rounds × 5 methods × 5 seeds = 500 round-runs → ~83 hours (can parallelize)
- Total all experiments: ~150-200 GPU-hours — feasible on Blackwell in ~1 week

---

## 5. Evaluation Metrics (Fixed)

### Per Dataset
| Metric | Yelp | Alpaca | GSM8K | Notes |
|--------|------|--------|-------|-------|
| Accuracy | ✓ | — | ✓ (exact-match) | |
| F1-macro | ✓ | — | — | |
| ROUGE-L | — | ✓ | — | |
| BLEU | — | ✓ | — | |
| Perplexity | ✓ | ✓ | ✓ | |
| Hallucination Rate | REMOVED | ✓ | — | Only for open-gen |

### Privacy (Honest Framing)
- Run MIA (shadow model attack) with proper protocol:
  - Train shadow model on same distribution
  - Use confidence scores of member vs non-member samples
  - Report AUC with 95% CI across 5 runs
- Expected result: AUC ≈ 0.50 for both SPA and Hetero-Pad (no formal DP)
- **Framing**: "SPA does not degrade privacy relative to standard FL; spectral truncation filters
  gradient noise as a byproduct (entropy H=2.67 vs 1.47), but this is NOT a formal privacy guarantee"

### Efficiency (Honest)
- Report comm cost per method per round
- Acknowledge that comm savings come from heterogeneous ranks, not SPA specifically
- SPA vs Hetero-Pad: same comm cost, SPA has higher accuracy (this IS SPA's contribution)
- Report SVD timing per layer, total per round

---

## 6. Paper Novelty Positioning (Post-Rework)

### How SPA Differs from FlexLoRA

| Aspect | FlexLoRA | SPA |
|--------|---------|-----|
| Core algorithm | ΔW=BA → avg → SVD project | Same |
| Theoretical grounding | None | Frobenius optimality theorem + spectral noise filtering |
| Convergence analysis | None | Empirical + theoretical sketch |
| Spectral threshold | Fixed (rank = hardware) | Adaptive threshold τ on singular values |
| Evaluation scope | 1 dataset, 1 model | 3 datasets, 2 non-IID levels, ablations |
| Privacy analysis | None | Formal MIA protocol, honest framing |

### Title Revision Candidates
- "SPA: Spectral Aggregation for Heterogeneous Federated LLM Fine-Tuning" (drop "Privacy-Preserving")
- OR keep if privacy section is made honest: "...with Analysis of Privacy Properties"

---

## 7. Step-by-Step Execution

### Phase 1: Environment Setup (Day 1-2)
```bash
# On Blackwell machine
cd /Users/samir/Desktop/FedLLM/rework
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.44.2 peft==0.12.0 accelerate==0.33.0
pip install datasets evaluate scikit-learn scipy rouge-score sacrebleu nltk
pip install wandb  # for experiment tracking

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Phase 2: Baseline Implementation (Day 3-6)
1. Port existing SPA/Pad/Homo code to modular src/ structure
2. Implement FlexLoRA (1-2 days, most critical)
3. Implement FLoRA stub (1 day)
4. Implement HetLoRA stub (1 day)
5. Unit test each aggregation method on toy data

### Phase 3: Data Pipeline (Day 5-6, parallel with baseline impl)
1. Yelp: already working, extend to support α=0.1
2. Alpaca: download tatsu-lab/alpaca, write partition + tokenization
3. GSM8K: download openai/gsm8k, write CoT prompt + exact-match extractor

### Phase 4: Run Main Experiments (Day 7-18)
1. Run E1 (Yelp, α=0.5) first — most comparable to original results
2. Run E2 (Yelp, α=0.1) — harder non-IID
3. Run E3 (Alpaca) — instruction following
4. Run E4 (GSM8K) — reasoning
5. Run ablations A1-A3 in parallel if multiple GPUs available

### Phase 5: Analysis (Day 19-22)
1. SVD overhead profiler — time across all 32 layers of Qwen2.5-7B
2. Convergence curves — plot loss vs rounds for all methods
3. MIA with proper protocol
4. Per-class F1, spectral analysis

### Phase 6: Paper Revision (Day 23-28)
1. New Table I (feature comparison) — remove unsupported claims
2. New Table II (main results) — include FlexLoRA column
3. New Table III (GSM8K + Alpaca results)
4. Rewrite related work: acknowledge FlexLoRA similarity, explain differentiation
5. Fix privacy section: honest framing
6. Fix comm cost narrative
7. Add convergence sketch (even informal bounds help)
8. New Figure: SVD overhead analysis

---

## 8. Known Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| SPA ≤ FlexLoRA empirically | Medium | SPA has spectral threshold; likely better under extreme non-IID |
| GSM8K too small for FL (7.5k samples) | Low | Use IID partition + non-IID ablation separately |
| MIA AUC stays ≈ 0.5 for all | High | Expected; honest framing is the fix |
| Alpaca instruction quality varies | Low | Use cleaned Alpaca-cleaned version |
| Blackwell driver/CUDA compatibility | Low | Use latest PyTorch nightly if needed |

---

## 9. Checklist (Track Progress)

- [ ] Phase 1: Environment verified on Blackwell
- [ ] FlexLoRA implemented and unit-tested
- [ ] FLoRA implemented
- [ ] HetLoRA implemented
- [ ] Data pipeline: Yelp (α=0.1 added)
- [ ] Data pipeline: Alpaca
- [ ] Data pipeline: GSM8K
- [ ] E1 complete (Yelp, α=0.5, 20 rounds, 5 seeds)
- [ ] E2 complete (Yelp, α=0.1)
- [ ] E3 complete (Alpaca)
- [ ] E4 complete (GSM8K)
- [ ] Ablations complete (A1-A4)
- [ ] SVD overhead analysis
- [ ] MIA with proper protocol
- [ ] Paper revision: related work
- [ ] Paper revision: experiments section
- [ ] Paper revision: privacy section
- [ ] Paper revision: Table I cleaned
- [ ] Paper revision: title reconsidered
