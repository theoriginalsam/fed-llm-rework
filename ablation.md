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

| Subdir | Files moved | How identified |
|--------|-------------|----------------|
| `beta03/` | hetlora_m seeds 42,43,44 | Runs 1–3 completed before GPU/script stop. Old script saved flat — moved manually. |
| `beta05/` | — | New script saves here automatically |
| `beta07/` | — | New script saves here automatically |
| `spa_m` all betas | — | Not yet started |

### If GPU stops mid-run
Check how far it got:
```bash
grep -E "Beta ablation:|Skipping|complete" logs/ablation_beta.log | tail -30
```
Count completed runs from the top of the run order table above.  
Files in flat `beta/yelp/` (not in a subdir) = from the last completed beta group.  
Move them to the right subdir, then rerun — the script skips existing files automatically.

### Rerun command
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

| Subdir | Files moved | How identified |
|--------|-------------|----------------|
| `k5/` | hetlora_m seeds 42,43 / hetlora seeds 42,43 / spa_m seeds 42,43 | Runs 1–6 completed (all K=5). Identified by sequential order — 6 files present = first full K group. Moved manually. |
| `k10/` | — | Script currently running here (cuda:0) |
| `k20/` | — | Not yet started |

### If GPU stops mid-run
```bash
grep -E "K ablation:|Skipping|complete" logs/ablation_k.log | tail -30
```
Files in flat `k_participation/yelp/` without a subdir = from current K group in progress.  
Count files to determine which K group: 1–2 files per method suggest partial K=10 run.  
Move completed files to `k10/` (or whichever K is active), rerun to finish the rest.

### Rerun command
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

### Identify what's done
```bash
# Count files per subdir
find results_ablation -name "*.json" | grep -v checkpoint | sort

# Check log for last completed run
grep "Beta ablation:\|K ablation:\|Rank dist ablation:\|complete" logs/ablation_beta.log | tail -20
grep "Beta ablation:\|K ablation:\|Rank dist ablation:\|complete" logs/ablation_k.log | tail -20
```

### Move flat files to correct subdir
If files landed flat (no subdir), use the run order tables above to identify which group they belong to, then:
```bash
mkdir -p results_ablation/<type>/yelp/<subdir>
mv results_ablation/<type>/yelp/{method}_seed{seed}_alpha01.json results_ablation/<type>/yelp/<subdir>/
```

### The script skips already-completed runs
After moving files and rerunning, the script checks if the output file exists in the subdir and skips it. Safe to rerun at any time.

---

## GPU Assignment

| GPU | Current job |
|-----|-------------|
| cuda:0 | K ablation (running) |
| cuda:1 | Beta ablation (running) |
