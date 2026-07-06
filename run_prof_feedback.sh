#!/usr/bin/env bash
# Post-professor-feedback experiments.
# Adds: (1) Yelp alpha=0.01 for all 5 methods × 5 seeds × 40 rounds
#        (2) GSM8K alpha=0.5 seeds 45+46 (fill missing to reach 5 seeds)
#        (3) GSM8K alpha=0.01 for all 5 methods × 5 seeds × 40 rounds
#
# Results saved to results_prof/{yelp,gsm8k}/ — separate from results_v2/
#
# Runtime estimate (2× A6000, ~3.3 h/Yelp-20r, ~4 h/Yelp-40r, ~5 h/GSM8K-40r):
#   Yelp 0.01  : 25 runs × ~4h ÷ 2 GPUs ≈ 2 days
#   GSM8K fill : 10 runs × ~5h ÷ 2 GPUs ≈ 1 day
#   GSM8K 0.01 : 25 runs × ~5h ÷ 2 GPUs ≈ 2.5 days (runs after fill)
#
# Usage (on sp2ai):
#   cd ~/FedLLM-Re/rework
#   bash run_prof_feedback.sh
#
# Monitor:
#   tail -f logs/prof_gpu0.log
#   tail -f logs/prof_gpu1.log
#   grep "DONE\|START\|Skipping" logs/prof_gpu0.log | tail -30

# ─── config ──────────────────────────────────────────────────────────────────
METHODS="homo_r8 hetero_pad flexlora hetlora hetlora_m"
RESULTS_DIR="results_prof"
YELP_ALPHA_NEW=0.01
YELP_ROUNDS_NEW=40          # longer for hard setting
GSM8K_ALPHA_EXISTING=0.5
GSM8K_SEEDS_FILL="45 46"   # only missing seeds for alpha=0.5
GSM8K_ALPHA_NEW=0.01
GSM8K_ROUNDS_NEW=40
ALL_SEEDS="42 43 44 45 46"
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p logs "${RESULTS_DIR}/yelp" "${RESULTS_DIR}/gsm8k"

# Split seeds: even → GPU 0, odd → GPU 1
GPU0_SEEDS=""; GPU1_SEEDS=""
for s in $ALL_SEEDS; do
    (( s % 2 == 0 )) && GPU0_SEEDS="$GPU0_SEEDS $s" || GPU1_SEEDS="$GPU1_SEEDS $s"
done
GPU0_FILL=""; GPU1_FILL=""
for s in $GSM8K_SEEDS_FILL; do
    (( s % 2 == 0 )) && GPU0_FILL="$GPU0_FILL $s" || GPU1_FILL="$GPU1_FILL $s"
done

echo "GPU 0 seeds (all):  $GPU0_SEEDS"
echo "GPU 1 seeds (all):  $GPU1_SEEDS"
echo "GPU 0 GSM8K fill:   $GPU0_FILL"
echo "GPU 1 GSM8K fill:   $GPU1_FILL"
echo ""

# ─── write worker scripts ─────────────────────────────────────────────────────
for GPU in 0 1; do
    if [ "$GPU" -eq 0 ]; then
        GSEEDS="$GPU0_SEEDS"; GFILL="$GPU0_FILL"
    else
        GSEEDS="$GPU1_SEEDS"; GFILL="$GPU1_FILL"
    fi

    cat > "/tmp/prof_gpu${GPU}.sh" <<WORKER
#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=${GPU}
cd $(pwd)
LOG="logs/prof_gpu${GPU}.log"

echo "=== [gpu${GPU}] prof_feedback worker started \$(date) ===" >> "\$LOG"

# ── BLOCK 1: Yelp alpha=0.01 (40 rounds) ─────────────────────────────────────
echo "[gpu${GPU}] === BLOCK 1: Yelp alpha=${YELP_ALPHA_NEW} rounds=${YELP_ROUNDS_NEW} ===" >> "\$LOG"
for METHOD in ${METHODS}; do
  for SEED in ${GSEEDS}; do
    ALPHA_TAG="${YELP_ALPHA_NEW/./}"
    OUTFILE="${RESULTS_DIR}/yelp/\${METHOD}_seed\${SEED}_alpha\${ALPHA_TAG}.json"
    if [ -f "\$OUTFILE" ]; then
      echo "[gpu${GPU}] Skipping Yelp \${METHOD} seed=\${SEED} alpha=${YELP_ALPHA_NEW} — already done." >> "\$LOG"
      continue
    fi
    echo "[gpu${GPU}][\$(date '+%H:%M')] START Yelp \${METHOD} seed=\${SEED} alpha=${YELP_ALPHA_NEW}" >> "\$LOG"
    python experiments/run_yelp.py \
      --method "\$METHOD" \
      --alpha "${YELP_ALPHA_NEW}" \
      --seed "\$SEED" \
      --num-rounds "${YELP_ROUNDS_NEW}" \
      --device "cuda:0" \
      --results-dir "${RESULTS_DIR}" \
      >> "\$LOG" 2>&1
    echo "[gpu${GPU}][\$(date '+%H:%M')] DONE  Yelp \${METHOD} seed=\${SEED} alpha=${YELP_ALPHA_NEW}" >> "\$LOG"
  done
done
echo "[gpu${GPU}] === BLOCK 1 COMPLETE ===" >> "\$LOG"

# ── BLOCK 2: GSM8K alpha=0.5 fill-in (seeds 45, 46 only) ─────────────────────
echo "[gpu${GPU}] === BLOCK 2: GSM8K alpha=${GSM8K_ALPHA_EXISTING} fill seeds [${GFILL}] ===" >> "\$LOG"
for METHOD in ${METHODS}; do
  for SEED in ${GFILL}; do
    ALPHA_TAG="${GSM8K_ALPHA_EXISTING/./}"
    OUTFILE="${RESULTS_DIR}/gsm8k/\${METHOD}_seed\${SEED}_alpha\${ALPHA_TAG}.json"
    if [ -f "\$OUTFILE" ]; then
      echo "[gpu${GPU}] Skipping GSM8K \${METHOD} seed=\${SEED} alpha=${GSM8K_ALPHA_EXISTING} — already done." >> "\$LOG"
      continue
    fi
    echo "[gpu${GPU}][\$(date '+%H:%M')] START GSM8K fill \${METHOD} seed=\${SEED}" >> "\$LOG"
    python experiments/run_gsm8k.py \
      --method "\$METHOD" \
      --alpha "${GSM8K_ALPHA_EXISTING}" \
      --seed "\$SEED" \
      --device "cuda:0" \
      --results-dir "${RESULTS_DIR}" \
      >> "\$LOG" 2>&1
    echo "[gpu${GPU}][\$(date '+%H:%M')] DONE  GSM8K fill \${METHOD} seed=\${SEED}" >> "\$LOG"
  done
done
echo "[gpu${GPU}] === BLOCK 2 COMPLETE ===" >> "\$LOG"

# ── BLOCK 3: GSM8K alpha=0.01 (40 rounds) ────────────────────────────────────
echo "[gpu${GPU}] === BLOCK 3: GSM8K alpha=${GSM8K_ALPHA_NEW} rounds=${GSM8K_ROUNDS_NEW} ===" >> "\$LOG"
for METHOD in ${METHODS}; do
  for SEED in ${GSEEDS}; do
    ALPHA_TAG="${GSM8K_ALPHA_NEW/./}"
    OUTFILE="${RESULTS_DIR}/gsm8k/\${METHOD}_seed\${SEED}_alpha\${ALPHA_TAG}.json"
    if [ -f "\$OUTFILE" ]; then
      echo "[gpu${GPU}] Skipping GSM8K \${METHOD} seed=\${SEED} alpha=${GSM8K_ALPHA_NEW} — already done." >> "\$LOG"
      continue
    fi
    echo "[gpu${GPU}][\$(date '+%H:%M')] START GSM8K \${METHOD} seed=\${SEED} alpha=${GSM8K_ALPHA_NEW}" >> "\$LOG"
    python experiments/run_gsm8k.py \
      --method "\$METHOD" \
      --alpha "${GSM8K_ALPHA_NEW}" \
      --seed "\$SEED" \
      --num-rounds "${GSM8K_ROUNDS_NEW}" \
      --device "cuda:0" \
      --results-dir "${RESULTS_DIR}" \
      >> "\$LOG" 2>&1
    echo "[gpu${GPU}][\$(date '+%H:%M')] DONE  GSM8K \${METHOD} seed=\${SEED} alpha=${GSM8K_ALPHA_NEW}" >> "\$LOG"
  done
done
echo "[gpu${GPU}] === BLOCK 3 COMPLETE ===" >> "\$LOG"

echo "=== [gpu${GPU}] ALL BLOCKS DONE \$(date) ===" >> "\$LOG"
WORKER
    chmod +x "/tmp/prof_gpu${GPU}.sh"
done

# ─── launch ──────────────────────────────────────────────────────────────────
nohup bash /tmp/prof_gpu0.sh >> logs/prof_gpu0.log 2>&1 &
PID0=$!
disown $PID0

nohup bash /tmp/prof_gpu1.sh >> logs/prof_gpu1.log 2>&1 &
PID1=$!
disown $PID1

echo "Launched GPU 0 (pid $PID0) — seeds $GPU0_SEEDS, GSM8K fill: $GPU0_FILL"
echo "Launched GPU 1 (pid $PID1) — seeds $GPU1_SEEDS, GSM8K fill: $GPU1_FILL"
echo "Results: ${RESULTS_DIR}/"
echo ""
echo "Monitor:"
echo "  tail -f logs/prof_gpu0.log"
echo "  tail -f logs/prof_gpu1.log"
echo ""
echo "Progress summary:"
echo "  grep 'DONE\|START\|Skipping\|COMPLETE' logs/prof_gpu0.log | tail -20"
echo "  grep 'DONE\|START\|Skipping\|COMPLETE' logs/prof_gpu1.log | tail -20"
echo ""
echo "Check still running:"
echo "  ps aux | grep prof_gpu"
