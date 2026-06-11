# Ablation Run Tracker

Results base: `/home/sp2ai/FedLLM-Re/rework/results_ablation/`  
Branch: `algo/ablations`  
All ablations: Yelp α=0.1, eval rank = median (r=8)

**Status as of 2026-06-11: ALL THREE ABLATIONS COMPLETE. Paper updated (submission_v3.tex §5.6).**

---

## File Naming Convention

Logger always saves as:
```
{method}_seed{seed}_alpha{alpha}.json
e.g. hetlora_m_seed42_alpha01.json
```

**The beta/K/dist parameter is NOT in the filename — it is encoded by the subdirectory.**

```
results_ablation/
  beta/yelp/
    beta03/   ← β=0.3 runs
    beta05/   ← β=0.5 runs
    beta07/   ← β=0.7 runs
  k_participation/yelp/
    k5/       ← K=5 runs
    k10/      ← K=10 runs
    k20/      ← K=20 runs
  rank_dist/yelp/
    balanced/ ← balanced distribution runs
    skewed/   ← skewed distribution runs
```

---

## 1. Beta Ablation — COMPLETE ✓

**Script:** `experiments/run_ablation_beta.py --all --device cuda:X`  
**Methods:** hetlora_m, spa_m  
**Beta values:** 0.3, 0.5, 0.7  
**Seeds:** 42, 43, 44  
**Total runs:** 18/18 complete

### Final Results (2026-06-11)

| Method | β | AUC (%) | MeanL5 (%) | Best (%) | Seeds |
|--------|---|---------|-----------|---------|-------|
| HetLoRA-M | 0.3 | 41.31 ±4.85 | 44.53 ±7.23 | 56.89 | 3 |
| HetLoRA-M | 0.5 | 40.72 ±4.01 | 43.88 ±6.44 | 56.46 | 3 |
| HetLoRA-M | 0.7 | 39.85 ±5.89 | 41.77 ±8.88 | 56.78 | 3 |
| SPA-M | 0.3 | 40.42 | — | — | 3 |
| SPA-M | 0.5 | 40.54 | — | — | 3 |
| SPA-M | 0.7 | 40.19 | — | — | 3 |

**Key findings:**
- HetLoRA-M leads SPA-M at β=0.3 (+0.89 pp) and β=0.5 (+0.18 pp); ties within noise at β=0.7
- HetLoRA-M degrades more steeply with β (−1.46 pp/step vs −0.23 pp for SPA-M), consistent with high-β adapter EMA over-smoothing
- β=0.5 confirmed as good default (used in all main experiments)

### Recovery note
Beta ablation was initially run with the OLD script (before subdir fix), causing overwrites.
All data was recovered from `logs/ablation_beta.log` using `recover_from_log.py`.

### Rerun command (new script, saves to subdirs correctly)
```bash
nohup bash -c 'cd /home/sp2ai/FedLLM-Re/rework && python experiments/run_ablation_beta.py --all --device cuda:1' > logs/ablation_beta.log 2>&1 & echo "PID: $!"
```

---

## 2. K Participation Ablation — COMPLETE ✓

**Script:** `experiments/run_ablation_k.py --all --device cuda:X`  
**Methods:** hetlora_m, hetlora, spa_m  
**K values:** 5, 10, 20  
**Seeds:** 42, 43  
**Total runs:** 18/18 complete

### Final Results (2026-06-11)

| Method | K=5 AUC | K=10 AUC | K=20 AUC |
|--------|---------|---------|---------|
| HetLoRA-M | 42.28 ±4.10 | 44.29 ±5.92 | 45.39 |
| HetLoRA   | 42.32 ±0.76 | 44.28       | 45.06 |
| SPA-M     | 39.81 ±1.77 | 40.96       | 43.29 |

**Key findings:**
- HetLoRA-M leads SPA-M at all participation rates (+2.47, +3.33, +2.10 pp at K=5,10,20)
- HetLoRA-M and HetLoRA scale nearly identically — momentum advantage is architectural, not K-dependent
- Both momentum-off and momentum-on methods improve with more participants (expected)

### Recovery note
K ablation also started with OLD script. Recovered from `logs/ablation_k.log`.

### Rerun command (new script, saves to subdirs correctly)
```bash
nohup bash -c 'cd /home/sp2ai/FedLLM-Re/rework && python experiments/run_ablation_k.py --all --device cuda:0' > logs/ablation_k.log 2>&1 & echo "PID: $!"
```

---

## 3. Rank Distribution Ablation — COMPLETE ✓

**Script:** `experiments/run_ablation_rank_dist.py --all --device cuda:X`  
**Methods:** hetlora_m, hetlora, spa_m, flexlora  
**Distributions:** balanced, skewed  
**Seeds:** 42, 43  
**Total runs:** 16/16 complete

### Rank distributions
```
balanced: {r4:20, r8:20, r16:5,  r32:5}
skewed:   {r4:35, r8:10, r16:3,  r32:2}
```

### Final Results (2026-06-11)

| Method | Balanced AUC | Skewed AUC | MeanL5 Bal. | MeanL5 Skew. | Best Bal. | Best Skew. |
|--------|-------------|-----------|-------------|-------------|-----------|-----------|
| HetLoRA-M | 42.28 ±4.10 | 42.95 ±5.86 | 47.94 ±3.60 | 47.25 ±3.48 | 56.00 | 55.32 |
| HetLoRA   | 42.32 ±0.76 | 43.43 ±2.30 | 45.95 ±0.83 | 48.47 ±2.51 | 55.33 | 55.92 |
| SPA-M     | 39.52 ±2.13 | 40.70 ±0.27 | 42.37 ±2.51 | 46.32 ±3.18 | 53.16 | 52.68 |
| FlexLoRA  | 40.35 ±0.69 | 41.55 ±0.35 | 42.08 ±4.11 | 47.30 ±3.85 | 53.64 | 53.87 |

**Key findings:**
- HetLoRA-M leads SPA-M in both distributions (+2.76 pp balanced, +2.25 pp skewed) — advantage is stable
- HetLoRA slightly edges HetLoRA-M in skewed (43.43 vs 42.95), within noise given 2 seeds and ±5.86 std
- FlexLoRA sits between SPA-M and HetLoRA methods in both conditions
- Hypothesis (skewed amplifies HetLoRA-M advantage) not strongly confirmed — but method is robust

### Rerun command
```bash
nohup bash -c 'cd /home/sp2ai/FedLLM-Re/rework && python experiments/run_ablation_rank_dist.py --all --device cuda:0' > logs/ablation_rankdist.log 2>&1 & echo "PID: $!"
```

---

## Paper Status

**submission_v3.tex §5.6** updated with all three ablations (2026-06-11, 8 pages).

Table structure:
- β sweep block: Low/Mid/High = β=0.3/0.5/0.7 (3 seeds)
- K sweep block: Low/Mid/High = K=5/10/20 (2 seeds)
- Rank dist block: Bal./Skew. columns, 3 methods (hetlora_m, hetlora, spa_m)

FlexLoRA omitted from paper rank dist table (not a primary baseline; saves space to stay at 8 pages).

---

## Quick Recovery Cheatsheet

### The overwrite problem
The logger always saves `{method}_seed{seed}_alpha01.json` regardless of beta/K value.
When a new run starts with the same method+seed, it **silently overwrites** the previous file.
Old script had no subdir logic so all runs in a group landed in the same flat folder.

### The solution
1. **New scripts** (after fix) save into subdirs automatically — no manual moves needed.
2. **nohup logs** capture all stdout including every round's metrics — full recovery possible even after overwrites.
3. **recover_from_log.py** reads the log and reconstructs correct JSON files into subdirs.

### Recovery commands (run after processes finish)
```bash
# Beta ablation
python experiments/recover_from_log.py --type beta --log logs/ablation_beta.log --out results_ablation/beta/yelp

# K participation ablation
python experiments/recover_from_log.py --type k --log logs/ablation_k.log --out results_ablation/k_participation/yelp

# Always dry-run first to verify
python experiments/recover_from_log.py --type beta --log logs/ablation_beta.log --out results_ablation/beta/yelp --dry-run
```

### Identify what's done
```bash
# See all result files
find results_ablation -name "*.json" | grep -v checkpoint | sort

# Check log progress
grep "Beta ablation:" logs/ablation_beta.log | tail -5
grep "K ablation:" logs/ablation_k.log | tail -5
```

### The script skips already-completed runs
After recovery, rerunning the ablation script is safe — it checks if the subdir file exists and skips it.

---

## GPU Assignment

| GPU | Current job |
|-----|-------------|
| cuda:0 | idle (all runs complete) |
| cuda:1 | idle (all runs complete) |
