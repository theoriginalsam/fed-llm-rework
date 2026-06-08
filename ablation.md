# Ablation Run Tracker

Results base: `/home/sp2ai/FedLLM-Re/rework/results_ablation/`  
Branch: `algo/ablations`  
All ablations: Yelp α=0.1, eval rank = median (r=8)

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

## 1. Beta Ablation

**Script:** `experiments/run_ablation_beta.py --all --device cuda:X`  
**Methods:** hetlora_m, spa_m  
**Beta values:** 0.3, 0.5, 0.7  
**Seeds:** 42, 43, 44  
**Total runs:** 18 (2 methods × 3 betas × 3 seeds)

### Run order (sequential)
```
 1. hetlora_m  β=0.3  seed=42
 2. hetlora_m  β=0.3  seed=43
 3. hetlora_m  β=0.3  seed=44
 4. hetlora_m  β=0.5  seed=42
 5. hetlora_m  β=0.5  seed=43
 6. hetlora_m  β=0.5  seed=44
 7. hetlora_m  β=0.7  seed=42
 8. hetlora_m  β=0.7  seed=43
 9. hetlora_m  β=0.7  seed=44
10. spa_m      β=0.3  seed=42
11. spa_m      β=0.3  seed=43
12. spa_m      β=0.3  seed=44
13. spa_m      β=0.5  seed=42
14. spa_m      β=0.5  seed=43
15. spa_m      β=0.5  seed=44
16. spa_m      β=0.7  seed=42
17. spa_m      β=0.7  seed=43
18. spa_m      β=0.7  seed=44
```

### Status (as of 2026-06-07)

**WARNING:** Beta ablation was started with the OLD script (before subdir fix). It runs flat
and overwrites files across beta groups. All data is recoverable from `logs/ablation_beta.log`.

| Subdir | Contents | Seeds | Notes |
|--------|----------|-------|-------|
| `beta03/` | hetlora_m 42,43,44 + spa_m 42,43 ✓ | 5/6 | spa_m seed=44 in progress — move flat file when done |
| `beta05/` | hetlora_m 42,43,44 ✓ | 3/3 | Recovered from log (overwritten by β=0.7 before manual move) |
| `beta07/` | hetlora_m 42,43,44 ✓ | 3/3 | Recovered from log |
| `spa_m β=0.5/0.7` | — | 0 | Still running on cuda:1 |

**Preliminary results (2026-06-07):**

| Method | β | AUC (%) | MeanL5 (%) | Best (%) | Seeds |
|--------|---|---------|-----------|---------|-------|
| HetLoRA-M | 0.3 | 41.31 ±4.85 | 44.53 ±7.23 | 56.89 | 3 |
| HetLoRA-M | 0.5 | 40.72 ±4.01 | 43.88 ±6.44 | 56.46 | 3 |
| HetLoRA-M | 0.7 | 39.85 ±5.89 | 41.77 ±8.88 | 56.78 | 3 |
| SPA-M | 0.3 | 39.82 ±1.92 | 42.54 ±2.78 | 53.12 | 2 |
| SPA-M | 0.5 | pending | — | — | 0 |
| SPA-M | 0.7 | pending | — | — | 0 |

HetLoRA-M leads SPA-M at β=0.3 by **+1.49 pp**. HetLoRA-M degrades gracefully with β (−0.73 pp/step).

### After leaving for 20 hrs — recovery command
Both processes still run OLD script (nohup doesn't reload code). Overwrites will happen.
Recover everything from logs after runs complete:
```bash
git pull
python experiments/recover_from_log.py --type beta --log logs/ablation_beta.log --out results_ablation/beta/yelp
```
Script skips incomplete runs (< 20 rounds) and existing files automatically.

### Rerun command (new script, saves to subdirs correctly)
```bash
nohup bash -c 'cd /home/sp2ai/FedLLM-Re/rework && python experiments/run_ablation_beta.py --all --device cuda:1' > logs/ablation_beta.log 2>&1 & echo "PID: $!"
```

---

## 2. K Participation Ablation

**Script:** `experiments/run_ablation_k.py --all --device cuda:X`  
**Methods:** hetlora_m, hetlora, spa_m  
**K values:** 5, 10, 20  
**Seeds:** 42, 43  
**Total runs:** 18 (3 methods × 3 K values × 2 seeds)

### Run order (sequential)
```
 1. K=5   hetlora_m  seed=42
 2. K=5   hetlora_m  seed=43
 3. K=5   hetlora    seed=42
 4. K=5   hetlora    seed=43
 5. K=5   spa_m      seed=42
 6. K=5   spa_m      seed=43
 7. K=10  hetlora_m  seed=42
 8. K=10  hetlora_m  seed=43
 9. K=10  hetlora    seed=42
10. K=10  hetlora    seed=43
11. K=10  spa_m      seed=42
12. K=10  spa_m      seed=43
13. K=20  hetlora_m  seed=42
14. K=20  hetlora_m  seed=43
15. K=20  hetlora    seed=42
16. K=20  hetlora    seed=43
17. K=20  spa_m      seed=42
18. K=20  spa_m      seed=43
```

### Status (as of 2026-06-07)

**WARNING:** K ablation also started with OLD script. Overwrites will happen across K groups.
Recover from `logs/ablation_k.log` after runs complete.

| Subdir | Contents | Seeds | Notes |
|--------|----------|-------|-------|
| `k5/` | hetlora_m 42,43 / hetlora 42,43 / spa_m 42,43 ✓ | 6/6 | Moved manually |
| `k10/` | hetlora_m 42,43 / hetlora 42 ✓ | 3/6 | Moved manually (runs 7–9) |
| `k20/` | — | 0 | Still running |

**Preliminary results (2026-06-07):**

| Method | K=5 AUC | K=10 AUC | K=20 AUC |
|--------|---------|---------|---------|
| HetLoRA-M | 42.28 ±4.10 | 44.29 ±5.92 | pending |
| HetLoRA | 42.32 ±0.76 | 45.38 ±0.00 (1 seed) | pending |
| SPA-M | 39.81 ±1.77 | pending | pending |

### After leaving for 20 hrs — recovery command
```bash
git pull
python experiments/recover_from_log.py --type k --log logs/ablation_k.log --out results_ablation/k_participation/yelp
```

### Rerun command (new script, saves to subdirs correctly)
```bash
nohup bash -c 'cd /home/sp2ai/FedLLM-Re/rework && python experiments/run_ablation_k.py --all --device cuda:0' > logs/ablation_k.log 2>&1 & echo "PID: $!"
```

---

## 3. Rank Distribution Ablation

**Script:** `experiments/run_ablation_rank_dist.py --all --device cuda:X`  
**Methods:** hetlora_m, hetlora, spa_m, flexlora  
**Distributions:** balanced, skewed  
**Seeds:** 42, 43  
**Total runs:** 16 (4 methods × 2 dists × 2 seeds)

### Rank distributions
```
balanced: {r4:20, r8:20, r16:5,  r32:5}
skewed:   {r4:35, r8:10, r16:3,  r32:2}
```

### Run order (sequential)
```
 1. balanced  hetlora_m  seed=42
 2. balanced  hetlora_m  seed=43
 3. balanced  hetlora    seed=42
 4. balanced  hetlora    seed=43
 5. balanced  spa_m      seed=42
 6. balanced  spa_m      seed=43
 7. balanced  flexlora   seed=42
 8. balanced  flexlora   seed=43
 9. skewed    hetlora_m  seed=42
10. skewed    hetlora_m  seed=43
11. skewed    hetlora    seed=42
12. skewed    hetlora    seed=43
13. skewed    spa_m      seed=42
14. skewed    spa_m      seed=43
15. skewed    flexlora   seed=42
16. skewed    flexlora   seed=43
```

### Status (as of 2026-06-07)

| Subdir | Files moved | Notes |
|--------|-------------|-------|
| `balanced/` | — | Not yet started |
| `skewed/` | — | Not yet started |

### If GPU stops mid-run
```bash
grep -E "Rank dist ablation:|Skipping|complete" logs/ablation_rankdist.log | tail -30
```
Files in flat `rank_dist/yelp/` = from current dist group.  
First 8 files = balanced, next 8 = skewed.

### Rerun command
```bash
nohup bash -c 'cd /home/sp2ai/FedLLM-Re/rework && python experiments/run_ablation_rank_dist.py --all --device cuda:0' > logs/ablation_rankdist.log 2>&1 & echo "PID: $!"
```

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
| cuda:0 | K ablation (running) |
| cuda:1 | Beta ablation (running) |
