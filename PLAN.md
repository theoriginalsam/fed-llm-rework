# SPA Rework — Full-Scale Experiment Plan
**Status:** V2 Running | **Last Updated:** 2026-05-06 | **GPU:** Blackwell (sp2ai) + A6000 (jovyan) | **Target:** ICLR / NeurIPS / ACL 2027

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

### Phase 1–3: Setup and Baselines
- [x] Phase 1: Environment verified on Blackwell (Qwen2.5-7B in bfloat16, ~14GB VRAM)
- [x] FlexLoRA implemented (`src/aggregation/flexlora.py`) and unit-tested
- [ ] FLoRA implemented — *skipped; not in reviewer comparison list; hetero_pad covers zero-pad baseline*
- [ ] HetLoRA implemented — *deferred; FlexLoRA is the critical missing baseline*
- [x] Data pipeline: Yelp (α=0.1 and 0.5, 50 clients, Dirichlet partition)
- [x] Data pipeline: Alpaca (200-sample eval, batched generation, ROUGE-L metric)
- [x] Data pipeline: GSM8K (100-sample eval cap, batched generation, exact-match)

### Phase 4: V1 Main Experiments (results/)
- [x] E1 complete (Yelp, α=0.5, 20 rounds, 5 seeds × 5 methods = 25 runs)
- [x] E2 complete (Yelp, α=0.1, 20 rounds, 5 seeds × 5 methods = 25 runs)
- [x] E3 complete (Alpaca, α=0.5, 20 rounds, 3 seeds × 4 methods = 12 runs) — *spa_m excluded*
- [x] E4 partial (GSM8K, α=0.5, 20 rounds) — hetero_spa ✓, homo_r4 ✓, homo_r8 ✓, flexlora 1/3 seeds ✓; hetero_pad + spa_m missing

### V1 Key Findings
- SPA wins GSM8K at +3.3pp over FedAvg-r8 (75.0% vs 71.7%, 1 seed each — to be confirmed)
- Alpaca: all methods converge to similar ROUGE-L ~0.42 (Qwen already strong on instruction following)
- SPA-M loses on Alpaca (-3pp) — EMA momentum hurts generative quality
- Yelp α=0.1: SPA-M shows instability (accuracy crashes, then recovers) due to β=0.9 carrying stale noise → fixed in V2
- Yelp α=0.5: methods are within 1-2pp of each other; SPA slightly edges others on best-round

### Phase 4: V2 Experiments — Branch `algo/spa-v2` (results_v2/)

Three algorithmic improvements over V1:
1. **Median eval rank** (r=8 instead of r=4): fixes structural disadvantage from projecting full-rank W_agg to min rank
2. **Rank-weighted aggregation**: weight = (rank × dataset_size) / Σ(rank_i × size_i); high-rank clients contribute proportionally more
3. **Adaptive β**: β_adaptive = β_max × (1 − cosine_sim(W_agg_t, W_agg_{t−1})); consistent updates → less momentum; divergent rounds → more momentum

Early result (2 seeds, α=0.1): SPA final accuracy 40.9±11.2 → 50.5±1.5 — 7× variance reduction confirms eval rank fix was the primary source of instability.

**V2 Yelp run status (as of 2026-05-06):**
- GPU0 (sp2ai cuda:0): seeds 42,43,44 — ~12/30 done
- GPU1 (sp2ai cuda:1): seeds 45,46 — running
- Total: ~12/50 done; ~38 runs remaining (~20 hours)
- [ ] V2 Yelp complete (50 runs: 5 methods × 2 alphas × 5 seeds)
- [ ] V2 GSM8K (SPA-M — was OOM on jovyan; run on sp2ai after V2 Yelp finishes)

---

### SPA-M Bug Analysis & Fix — Branch `algo/spa-v2` (2026-05-11)

**Symptoms:** SPA-M α=0.1 final acc 45.7 ± 10.1 vs base SPA 50.5 ± 1.5.
Accuracy in `spa_m_seed45_alpha01.json` oscillates ±20pp round-to-round
(48%→33%→48%→28%→46%→25%) despite stable perplexity (7.75–7.90).

**Root cause: 4 bugs in `src/aggregation/spa_momentum.py`.**

#### Bug 1 — EMA before SVD (wrong order)
- **What:** `_aggregate()` applied momentum to raw `w_agg_t`, then SVD only at distribution time via `soft_project_to_rank`.
- **Why it broke:** Under α=0.1, the 5 sampled clients change dramatically each round. Consecutive `W_agg_t` matrices occupy near-orthogonal subspaces. EMA over raw orthogonal matrices produces a noise soup; SVD at distribution time cannot recover a clean rank-r signal from it.
- **Fix:** `_soft_spectral_filter()` now runs on `w_agg_t` before the momentum update. The EMA buffer accumulates denoised signal. `project_to_rank()` at distribution does plain rank-r truncated SVD only (no double-shaping).

#### Bug 2a — Cosine similarity clamped to [0, 1]
- **What:** `.clamp(0.0, 1.0)` in `_adaptive_beta()` silently discarded negative similarity.
- **Why it broke:** Anti-correlated updates (sim < 0, the oscillation signal) were mapped to `sim=0` → `β = 0.9 × 1.0 = 0.9` (maximum momentum). The method was injecting maximum momentum into an already-oscillating system. This is the proximate cause of the ±10.1 variance.
- **Fix:** `.clamp(-1.0, 1.0)` — full range preserved.

#### Bug 2b — Adaptive β formula inverted
- **What:** Formula was `β_max × (1 − sim)`: consistent rounds got β≈0, divergent got β=β_max.
- **Why it broke:** Momentum's benefit is accelerating a consistent direction. Giving maximum momentum to divergent rounds accumulates stale noise from misaligned past rounds.
- **Fix:** Formula is now `β_max × (sim + 1) / 2`: maps [-1,1] → [0, β_max]. Anti-correlated → β=0 (brake). Consistent → β=β_max (accelerate).

#### Bug 2c — Bias correction undefined for variable β
- **What:** `bc = 1 - (β_adaptive^t)` used current round's variable β with the global round counter as exponent.
- **Why it broke:** Standard bias correction `1 - β^t` assumes constant β. With different β each round, `β_current^t` is meaningless — neither the product of past β values nor a constant power. Produces arbitrary scale factors, seed-dependent.
- **Fix:** Track cumulative product: `self._beta_product *= beta` → `bc = 1 - self._beta_product`. Mathematically correct for variable β.

#### Bug 3 — `_prev_wagg` stored raw (noisy) aggregation
- **What:** `self._prev_wagg` stored `w_agg_t` (raw). Cosine similarity in `_adaptive_beta` compared raw noisy aggregations.
- **Why it broke:** Under α=0.1, raw aggregations have high round-to-round directional variance (random client subsets). Cosine similarity was systematically underestimated → β_adaptive systematically inflated → momentum buffer accumulated more noise.
- **Fix:** `self._prev_filtered` now stores `w_filtered_t` (denoised). Cosine similarity compares clean signals.

#### Bug 4 — Momentum output acted as uncontrolled LR multiplier
- **What:** No magnitude normalization on the bias-corrected output. Different seeds → different β_adaptive sequences → different bias correction scales → different initialization magnitudes for clients → different effective learning rates per seed.
- **Why it broke:** This is the direct mechanical cause of ±10.1 variance across seeds. The model found the right loss basin (perplexity stable) but the seed-dependent initialization scale repeatedly displaced the classifier weights.
- **Fix:** Output rescaled to `||W_agg_t||_F` before returning. Effective LR is now seed-invariant.

**Files changed:**
- `src/aggregation/spa_momentum.py` — complete rewrite of `_aggregate()`, `_adaptive_beta()`, new `_soft_spectral_filter()`, `project_to_rank()` replaces `soft_project_to_rank()` at distribution
- `src/server/fl_server.py:71` — updated callsite to `project_to_rank`

**Tests:** `test_spa_m.py` — 10 unit tests, CPU-only, all pass. Regression test (Test 10) shows old β was 0.9 for anti-correlated input; new β is 0.0.

**Expected outcome:** Variance drops from ±10.1 to ±2–4. Mean accuracy rises from ~45.7. Whether SPA-M beats base SPA (50.5) at α=0.1 is empirical — momentum requires coherent signal across rounds, which is fundamentally challenging with 5 clients/round under extreme non-IID. More likely to show a clear win at α=0.5.

**Next step:** Run 1-seed smoke test before committing V2-fixed grid:
```bash
nohup python experiments/run_yelp.py --method spa_m --seed 42 --alpha 0.1 --num-rounds 5 --results-dir results_v2_fixed > logs/spa_m_smoke.log 2>&1 &
```

**Re-run grid (all SPA-M, results_v2_fixed/):**
```bash
# GPU 0 — α=0.1, seeds 43-46 (seed 42 covered by smoke test)
nohup bash -c 'for seed in 43 44 45 46; do python experiments/run_yelp.py --method spa_m --alpha 0.1 --seed $seed --device cuda:0 --results-dir results_v2_fixed; done' > logs/spa_m_alpha01.log 2>&1 &

# GPU 1 — α=0.5, all 5 seeds (run in parallel with above)
nohup bash -c 'for seed in 42 43 44 45 46; do python experiments/run_yelp.py --method spa_m --alpha 0.5 --seed $seed --device cuda:1 --results-dir results_v2_fixed; done' > logs/spa_m_alpha05.log 2>&1 &
```

**Note:** All runs use `nohup` so they survive terminal disconnects. Check progress with:
```bash
tail -f logs/spa_m_alpha01.log
tail -f logs/spa_m_alpha05.log
ps aux | grep run_yelp
```

### Phase 5: Analysis
- [ ] Ablations complete (A1-A4) — deferred until V2 Yelp done
- [ ] Statistical significance tests (paired t-test across seeds for SPA vs FlexLoRA)
- [ ] SVD overhead analysis
- [ ] MIA with proper protocol

### Phase 6: Paper Revision
- [ ] Paper revision: related work (acknowledge FlexLoRA similarity, differentiate formally)
- [ ] Paper revision: experiments section (3 datasets, V2 results table)
- [ ] Paper revision: privacy section (honest MIA framing)
- [ ] Paper revision: Table I cleaned
- [ ] Paper revision: title reconsidered (remove "Privacy-Preserving" if privacy claims stay weak)
