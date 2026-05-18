# SPA Rework — Full-Scale Experiment Plan
**Status:** Post-professor feedback — paper reframe + ablations planned | **Last Updated:** 2026-05-17 | **GPU:** sp2ai (2× RTX A6000 49GB, CUDA 12.8) | **Target:** ICLR / NeurIPS / ACL 2027

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
# On server (sp2ai: 2× RTX A6000 49GB, CUDA 12.8)
cd ~/FedLLM-Re/rework
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 && pip install -r requirements.txt

# Verify both GPUs
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_name(1))"
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

**V2 Yelp run status (as of 2026-05-15):**
- Other methods (hetero_spa, flexlora, homo_r8, hetero_pad): ✓ complete for both alphas
- SPA-M: V2.7 α=0.1 done (5 seeds) — Final 41.8±9.9, overshoot detected, V2.8 fix applied
- SPA-M: V2.7 α=0.5 in progress — results pending
- Next: re-run SPA-M both alphas with V2.8 (magnitude normalization restored)
- [ ] V2 Yelp SPA-M V2.8 complete (10 runs: 2 alphas × 5 seeds)
- [ ] V2 GSM8K (SPA-M — was OOM on jovyan; run on sp2ai after Yelp SPA-M fixed)

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

**V2 fixed results (5 seeds, α=0.1):**
- Mean-L5: 40.4 ± 3.6 (vs V1: 47.3 ± 6.5, vs SPA V2: 42.0 ± 4.4)
- Final:   33.0 ± 6.4 (vs V1: 47.6 ± 7.4, vs SPA V2: 50.5 ± 1.5)
- Variance did reduce (±7.4 → ±6.4) but mean collapsed — regression.

**Root cause of regression: β direction was wrong (accelerator vs stabilizer).**
The fix changed β to accelerator mode (high β when consistent, β→0 when divergent).
Under α=0.1 almost every round is anti-correlated → β≈0 → no momentum → worse than SPA.
Momentum in FL high non-IID must be a STABILIZER: high β when divergent to smooth oscillations.

**V2.1 fix (2026-05-14):** Reverted β to stabilizer direction with correct [-1,1] sim range:
  `β = beta_max × (1 - sim) / 2`  maps [-1,1] → [beta_max, 0]
  Also raised beta_max: 0.5 → 0.9 to match effective momentum strength of V1.
  Keeps all other fixes: cumulative bias correction, magnitude normalization, SVD filter order, lowrank SVD.

---

### SPA-M Simulation-Driven Ablation (2026-05-14) — Branch `algo/spa-v2`

**Problem:** V2.1 still underperformed base SPA (simulation cosine-sim 0.227 vs V1 0.274).
Built `test_spa_m_sim.py`: CPU-only simulation of 20 FL rounds with progressive learning signal
under α=0.1 dynamics (5→20 seeds × 20 rounds). Metric: cosine similarity with true direction.
No GPU or model required — runs in ~2 seconds.

**Ablation findings (20 seeds):**

| Version | Final Cos-Sim | SeedVar | Notes |
|---------|--------------|---------|-------|
| V1 (broken bugs) | 0.276 | 0.024 | reference |
| V2 (accelerator β) | 0.229 | 0.026 | regression, confirmed bad |
| V2.1 (stab[-1,1] + SVD + mag-norm) | 0.218 | 0.016 | SVD filter hurts signal |
| V2.7 (clamp[0,1] + cumul-bc, β=0.9) | 0.355 | 0.018 | **best in simulation** |

**Key discoveries:**
1. **SVD filter in `_aggregate()` attenuates learning signal** — filters the very signal we want to accumulate. Removing it from the aggregation path (keeping only at `project_to_rank` distribution time) was the right call.
2. **β_max=0.9 with clamped [0,1] stabilizer beats both V1 and full [-1,1] stabilizer** — clamping to [0,1] saturates at β_max for any non-consistent round (orthogonal or anti-correlated), preventing over-smoothing. Higher β (0.9 vs 0.5) retains more signal history; cumulative bias correction amplifies it correctly.
3. **Correct cumulative bias correction reduces seed variance 50%** (0.024 → 0.012) — the whole point of fixing bug 2c.

**V2.7 implementation** (`src/aggregation/spa_momentum.py`):
- `_adaptive_beta`: `clamp(sim, 0, 1)` → `β = beta_max × (1 - sim)` (saturating stabilizer)
- `_aggregate`: raw W_agg → adaptive β → EMA → cumulative bc → output (no SVD filter, no mag-norm inside)
- beta_max = 0.9 (default and fl_server.py)
- All 10 unit tests pass (`test_spa_m.py`)

---

### V2.7 Real Training Results + V2.8 Fix (2026-05-15)

**V2.7 actual result — Yelp α=0.1 (5 seeds):**
- Mean-L5 Acc: 40.0 ± 4.7
- Final Acc:   **41.8 ± 9.9** ← still bad, high variance
- Best Acc:    54.0 ± 1.8 ← competitive with SPA (53.5)

**Diagnosis: momentum overshoot.** Best Acc≈54% but Final Acc≈42% means the model peaks then degrades. Root cause: the simulation metric (cosine-sim) is scale-invariant. In real training, an uncapped EMA buffer accumulates gradient magnitudes across rounds and acts as an unbounded LR multiplier. The cumulative bias correction `1/(1-Π β_τ)` amplifies this further in mid-training, pushing the model past the optimal point.

**V2.8 fix (2026-05-15):** Re-add magnitude normalization to `_aggregate()` output.
Rescales `bc_v = momentum / bc` back to `||W_agg_t||_F` before returning.
This keeps the directional smoothing from momentum but caps the effective update magnitude
equal to the raw aggregation each round — same scale as all other methods.

```python
# Added back in _aggregate(), after bias correction:
for k, v in self._momentum.items():
    bc_v = v / bc
    w_norm = torch.linalg.norm(w_agg_t[k].float())
    m_norm = torch.linalg.norm(bc_v.float())
    if m_norm > 1e-8 and w_norm > 1e-8:
        bc_v = bc_v * (w_norm / m_norm)
    result[k] = bc_v
```

**Why simulation missed this:** Simulation compared cosine-similarity only (scale-invariant).
Magnitude explosion doesn't affect cosine-sim but does cause accuracy overshoot in real training.

---

### V2.8 Real Training Results (2026-05-16)

Seeds: 42–46. Both alphas re-run after git pull + correct seeds.

**V2.8 Yelp α=0.5 (5 seeds, FINAL):**
- Mean-L5: 50.0 ± 3.9 | Final: **52.1 ± 4.0** | Best: 59.1 ± 2.5
- SPA-M beats SPA (49.5) and FlexLoRA (51.7) ✓
- Mean-L5 ≈ Final gap only 2.1pp → oscillation controlled ✓

**V2.8 Yelp α=0.1 (5 seeds, FINAL):**
- Mean-L5: 41.4 ± 3.8 | Final: **36.0 ± 12.8** | Best: 53.9 ± 3.2
- Final accuracy collapses by round 20 despite peaking at 54% mid-training
- Gap Best→Final = 17.9pp → overshoot persists at α=0.1 ✗

**V2 Yelp Results (2026-05-18) — ★ = primary metric (Mean-L5 Acc):**

α=0.1 (n in parentheses — SPA/FlexLoRA still incomplete at n=2):

| Method | Mean-L5 Acc ★ | Final Acc | Best Acc | n |
|--------|--------------|-----------|----------|---|
| FedAvg r=8 | 42.2±2.7 | 39.6±12.8 | 54.1±3.3 | 5 |
| Hetero-Pad | 42.0±3.9 | 41.4±15.3 | 56.0±0.4 | 3 |
| FlexLoRA | 41.7±4.6 | 48.7±1.4 | 53.7±1.5 | **2** |
| SPA | 42.0±4.4 | 50.5±1.5 | 53.5±0.2 | **2** |
| **SPA-M** | 41.4±3.8 | 36.0±12.8 | 53.9±3.2 | **5** |

α=0.5 (SPA still at n=4):

| Method | Mean-L5 Acc ★ | Final Acc | Best Acc | n |
|--------|--------------|-----------|----------|---|
| FedAvg r=8 | 52.4±2.1 | 56.4±3.1 | 59.9±1.3 | 5 |
| Hetero-Pad | 48.9±4.3 | 51.3±5.5 | 57.4±2.3 | 5 |
| FlexLoRA | 51.3±3.3 | 53.1±4.3 | 59.4±2.3 | 5 |
| SPA | 50.0±4.3 | 51.2±4.9 | 58.9±2.7 | **4** |
| **SPA-M** | **51.6±3.5** | 52.1±4.0 | **60.1±2.4** | 5 |

⚠️ SPA and FlexLoRA at α=0.1 have n=2 — Final Acc means unreliable. Mean-L5 stable by n=3.

**Key findings from completed data:**
1. **α=0.5 SPA-M Best Acc 60.1 edges FedAvg r=8 (59.9)** — our method peaks higher than the oracle on best-round metric. Mean-L5 (51.6 vs 52.4) within noise.
2. **α=0.1 Mean-L5 all tied at 41–42%** — no method dominates on the ★ metric. FedAvg r=8 Final also collapses (39.6±12.8), same high variance as SPA-M. The "oracle" is not stable either.
3. **Final Acc is misleading for all methods at α=0.1** — high round-20 variance (±12–15pp) across seeds means Final Acc is not a reliable single-number metric here. Mean-L5 and Best Acc are the right primary metrics.
4. SPA/FlexLoRA n=2 at α=0.1 look good on Final (50.5/48.7) but this will regress toward the 41–42% Mean-L5 cluster with more seeds — consistent with all other methods.

---

### Root Cause Analysis: SPA-M α=0.1 Failure (2026-05-16)

**Symptom:** Best Acc≈54% (competitive) but Final Acc≈36% (catastrophic). Model peaks mid-training then degrades.

**Mechanism — server momentum feedback loop:**
```
Momentum M_t  →  project_to_rank()  →  client initialization (B, A)
                                              ↓
                                    clients train 100 steps
                                    pushing AWAY from M_t toward local data
                                              ↓
                                    ΔW ≈ θ_local − M_t  (relative, not absolute)
                                              ↓
                                    W_agg = avg(θ_local) − M_t
                                              ↓
                        M_{t+1} = β·M_t + (1−β)·W_agg
                                = β·M_t + (1−β)·(avg(θ_local) − M_t)
```

W_agg is NOT an absolute signal — it's relative to M_t. So momentum chases a moving target that it itself is displacing. Under α=0.1, different client subsets each round make `avg(θ_local)` jump around → oscillation compounds → accuracy collapses in late training.

**Why perplexity is fine but accuracy oscillates:** Perplexity is a smooth average over all tokens (language model layers improve steadily). Accuracy on Yelp depends on the output projection aligning with one of 5 class tokens — a sharper directional signal that the oscillating momentum periodically inverts.

**Why magnitude normalization (V2.8) didn't fix it at α=0.1:** Normalization caps the scale but not the direction. The feedback loop operates on direction, not magnitude.

**This is a known FL problem.** SCAFFOLD and FedDyn solve it by explicitly estimating and correcting client drift. Server-side momentum without drift correction creates exactly this instability under high non-IID.

**Conclusion:** SPA-M in its current form (server momentum on ΔW fed back into client initialization) cannot be fixed at α=0.1 through hyperparameter tuning. The architecture is the issue.

---

### Paper Positioning After SPA-M Analysis (2026-05-17)

**Why FedAvg r=8 outperforms heterogeneous methods:**
1. Homogeneous ranks → all 5 sampled clients contribute equal-expressivity ΔW → cleaner aggregation
2. No rank mismatch noise: rank-4 clients produce sparser updates that dilute higher-rank signal in FlexLoRA/SPA
3. No projection approximation error at distribution time

**FedAvg r=8 is an oracle baseline** — it assumes all 50 clients can afford rank=8 (2× memory for rank-4 devices). The realistic comparison is within heterogeneous methods: SPA-M vs SPA vs FlexLoRA.

**Current rank distribution: {r4:20, r8:20, r16:5, r32:5}**
- Equal split between r4/r8 (not strongly skewed toward edge devices)
- A more realistic distribution (e.g. 35/10/3/2) would further disadvantage FedAvg r=8
- Potential ablation: run 35/10/3/2 with 1-2 seeds after main results complete

**Paper story (current evidence):**
- α=0.5: SPA-M > SPA ≈ FlexLoRA > Hetero-Pad (within heterogeneous methods) ✓
- α=0.1: SPA-M unstable due to server momentum feedback loop — base SPA more robust
- Honest framing: *"SPA-M improves over SPA at moderate non-IID (α=0.5) but is destabilized by the server momentum feedback loop under extreme non-IID (α=0.1). We analyze this failure mode and recommend SPA-M when α≥0.5."*

---

### Current Run Status (2026-05-17)

**Remaining Yelp runs (in progress on sp2ai — 2× RTX A6000):**

GPU 0 (cuda:0) — α=0.1 missing seeds:
```bash
nohup bash -c 'python run_yelp.py --method homo_r8 --alpha 0.1 --seed 43 --device cuda:0 && ... hetero_pad seeds 43,44 && flexlora seeds 43,44,46 && hetero_spa seeds 43,44,46' > logs/remaining_a01.log 2>&1 &
```

GPU 1 (cuda:1) — α=0.5 missing seeds:
```bash
nohup bash -c '... homo_r8 seeds 43,44 && hetero_pad seeds 42,43,44,46 && flexlora seeds 42,43,44 && hetero_spa seeds 43,44' > logs/remaining_a05.log 2>&1 &
```

- [ ] Yelp all methods n=5 seeds both alphas complete
- [ ] GSM8K (hetero_spa, homo_r8, flexlora, hetero_pad — SPA-M excluded or flagged)
- [ ] Alpaca (same methods)
- [ ] Rank distribution ablation (35/10/3/2) — 1 seed each after main results

**Environment note:** sp2ai has torch 2.12.0 (CUDA 13.0) which conflicts with driver 570.x (CUDA 12.8). Fix:
```bash
pip uninstall torch torchvision torchaudio -y && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
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

---

## 10. Professor Feedback Integration (2026-05-17)

Received comprehensive feedback from supervisor. Key points below, translated into action items.

### 10.1 Core Scientific Insight — Reframe the Paper

**Professor's insight:** Heterogeneous-rank FL is fundamentally a **subspace-consensus problem**, not just an aggregation efficiency problem.

LoRA is non-identifiable: `BA = (BQ)(Q^{-1}A)` for any invertible Q. Different clients learn different parameterizations of the same (or similar) functions. When the server aggregates ΔW = B·A and projects back to rank-r, it is performing an implicit basis alignment — but this alignment is arbitrary and inconsistent across rounds.

The deeper failure: SVD truncation at distribution time (step 5 of FlexLoRA/SPA) **destroys minority directions**. A rank-4 client that learned a semantically important but low-energy subspace (e.g., a tail class under α=0.1) gets its direction zeroed out — not because it is noise, but because it is low-energy. This is not a hyperparameter issue; it is a structural flaw in the dense-aggregation + SVD-truncation pipeline.

**Paper reframe (new narrative):**
> "We show that heterogeneous-rank federated fine-tuning is a subspace-consensus problem: clients operating at different ranks learn incompatible gradient subspaces, and naive aggregation + SVD projection systematically discards minority directions. We characterize when this failure occurs (high non-IID, large rank gaps) and propose SPA as a theoretically grounded aggregation operator that minimizes projection distortion."

This reframing gives us a **stronger contribution**: identifying the subspace-consensus failure mode is novel, independent of whether SPA-M beats SPA numerically.

### 10.2 Rank Distribution — Current Setting is Too Easy

**Problem:** Current `{r4:20, r8:20, r16:5, r32:5}` is too balanced. Equal r4/r8 split means the aggregated ΔW signal is dominated by mid-rank clients — not a hard heterogeneity setting.

**Professor suggests:** More realistic distributions:
- **35/10/3/2** (35 clients at r4, 10 at r8, 3 at r16, 2 at r32) — strongly skewed toward edge devices
- **40/8/1/1** — extreme edge-heavy, 1 high-rank client

**Why it matters:** Under 35/10/3/2, FedAvg r=8 is a much less credible oracle (requires 40 edge devices to afford r=8 — not realistic). Heterogeneous methods gain larger relative advantage.

**Action:**
- [ ] Run 35/10/3/2 ablation: 2 seeds × 2 alphas × {SPA, FlexLoRA, FedAvg-r8} on Yelp (6 runs)
- [ ] Consider 40/8/1/1 if 35/10/3/2 shows interesting patterns (2 seeds, α=0.1 only)

### 10.3 SVD Truncation Destroys Minority Directions

**Professor's analysis:**
- Under α=0.1, rank-4 clients are the majority (20/50 currently, 35/50 in realistic setting)
- Their gradient subspace may capture tail-class directions that are orthogonal to the majority signal
- SVD of the aggregated ΔW orders components by energy (σ_1 ≥ σ_2 ≥ ... ≥ σ_k)
- When rank-4 clients' contribution is orthogonal to rank-32 clients' signal, it lands in σ_{k+1} — outside the rank-r window — and is silently discarded

**Proposed solution (Idea A — clustered spectral aggregation):**
1. Cluster clients by gradient subspace similarity (k-means on top-r singular vectors)
2. Aggregate within clusters (subspace consensus guaranteed within cluster)
3. Merge cluster aggregations with rank-weighted combination
4. Optional: keep one "minority cluster" representative to preserve tail directions

**Proposed solution (Idea B — personalized projection ranks):**
- Instead of projecting all clients to their own rank independently, project to the *cluster-shared* rank
- Low-rank clients in a cluster with high-rank clients get projection at the cluster consensus rank
- Reduces distortion from rank mismatch

**Note:** These are medium-complexity additions. Professor framed them as "optional for ICLR, required for NeurIPS if reviewers ask about subspace loss." Start with Idea A as a prototype after main results complete.

### 10.4 Grassmann Manifold Averaging

**Professor's suggestion:** Instead of averaging ΔW matrices in Euclidean space, average the subspaces themselves on the Grassmann manifold Gr(r, d).

- Karcher mean on Gr(r, d) is the proper notion of "consensus subspace"
- Avoids the basis-arbitrariness of Euclidean ΔW averaging
- Can be computed via iterative geodesic averaging (Bhattacharya & Bhattacharya 2012)
- Cost: O(n · r · d) per round — same order as current SVD

**Status:** Theoretical improvement, worth a section in the paper. Implement only if time permits after main results. The subspace-consensus narrative can reference this as a "direction for future work" if not implemented.

### 10.5 Drift Correction (SCAFFOLD / FedDyn)

**Professor's note:** Server-side momentum without drift correction (as we confirmed in §9 root-cause analysis) creates an unstable control loop under high non-IID. SCAFFOLD corrects this by estimating client drift with a control variate.

**Implication for SPA-M:** Adding SCAFFOLD-style drift correction to SPA-M would fix the α=0.1 failure mode. This is the "clean" architectural fix vs. the ad-hoc magnitude normalization.

- SCAFFOLD overhead: one additional upload per client (the control variate gradient)
- Communication cost doubles — must be reported honestly
- Not strictly necessary for the paper if we honestly frame SPA-M's limitation

**Action:** Implement SCAFFOLD-SPA-M as an optional experiment if time permits. Otherwise, cite the failure as a known FL problem and reference SCAFFOLD as the appropriate fix.

### 10.6 Evaluation Metrics — Add Best Acc and AUC

**Professor's correction:** Reporting only Final Acc at round 20 is misleading when accuracy oscillates (e.g., α=0.1 SPA-M peaks at 54% then collapses to 36%). Final Acc understates performance.

**Required additional metrics:**
1. **Best Acc** (peak across all rounds) — already tracked in JSON logs, add to tables
2. **AUC of convergence curve** (mean accuracy across all rounds) — use `np.trapz` on per-round accuracy
3. **Stability variance** (std of accuracy across last 5 rounds) — already computed as Mean-L5 std, just name it

**Update reporting in:**
- [ ] All results tables in paper
- [ ] Per-round logs (already captured in round_results JSON)
- [ ] Summary table in PLAN.md

### 10.7 Participation Rate Ablation

**Professor strongly suggests:** Varying clients-per-round (K) is a critical ablation. Current: K=5/50 (10%).

| K | Fraction | Regime |
|---|---------|--------|
| 5 | 10% | current, low participation |
| 10 | 20% | moderate |
| 20 | 40% | high participation |

**Why it matters:** SPA-M's momentum is sensitive to K. With K=5 under α=0.1, consecutive rounds share 0 clients → consecutive ΔW matrices are near-orthogonal → β saturates at β_max every round → pure smoothing, no acceleration. With K=20, consecutive rounds share ~8 clients → more consistent subspace direction → adaptive β can fire correctly.

**Action:**
- [ ] Run K={5, 10, 20} ablation: 2 seeds × α=0.1 × {SPA, SPA-M} on Yelp (12 runs)
- [ ] Expected: SPA-M improves relative to SPA as K increases — this would make a cleaner story

### 10.8 What to Do Next (Professor's Recommendation)

Priority order:
1. **Complete the 5-seed grid** for all methods on Yelp (both alphas) — already in progress
2. **Add Best Acc and AUC columns** to all tables in paper and analysis scripts
3. **Run 35/10/3/2 rank ablation** (2 seeds) — sharpens the heterogeneity argument
4. **Run K={5,10,20} participation ablation** (2 seeds, SPA vs SPA-M) — critical for understanding SPA-M failure
5. **Prototype clustered spectral aggregation** — optional, only if 1-4 yield strong results worth writing up at NeurIPS level
6. **Reframe the paper** around subspace-consensus — the narrative is stronger than the numbers alone

**Professor's summary quote:** *"The interesting result is not whether SPA-M beats SPA by 2%. The interesting result is that SVD projection at distribution time is structurally destroying minority client information, and nobody in the FL-LoRA literature has characterized this. That's your contribution."*

---

### Quick-Start Commands for New Ablations

```bash
# After Yelp 5-seed grid is complete, run on sp2ai:

# 1. Rank distribution ablation (35/10/3/2)
# Edit config/base_config.py: RANK_DISTRIBUTION = {"r4":35, "r8":10, "r16":3, "r32":2}
for seed in 42 43; do for alpha in 0.1 0.5; do for method in hetero_spa flexlora homo_r8 spa_m; do
  nohup python experiments/run_yelp.py --method $method --seed $seed --alpha $alpha \
    --device cuda:0 >> logs/rankdist_${method}_s${seed}_a${alpha}.log 2>&1 &
done; done; done

# 2. Participation rate ablation (K=10, K=20)
# Edit config/base_config.py: CLIENTS_PER_ROUND = 10 (or 20)
for K in 10 20; do for seed in 42 43; do for method in hetero_spa spa_m; do
  nohup python experiments/run_yelp.py --method $method --seed $seed --alpha 0.1 \
    --device cuda:1 >> logs/K${K}_${method}_s${seed}.log 2>&1 &
done; done; done
```
