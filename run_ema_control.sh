#!/usr/bin/env bash
# EMA-eval control experiment (tab:ema_control in paper).
# Reruns homo_r8, hetero_pad, flexlora on Yelp alpha=0.1 with --ema-eval.
# Results saved to results_ema_control/yelp/ — separate from main results.
# Splits seeds across both GPUs; uses nohup so jobs survive SSH disconnect.
#
# Runtime: ~50 GPU-h total (15 runs × ~3.3h each) ≈ ~1.5 days on 2× A6000.
# 3 seeds instead of 5 cuts to ~overnight (edit SEEDS below).
#
# Usage (on sp2ai):
#   cd ~/FedLLM-Re/rework
#   bash run_ema_control.sh
#
# Monitor after launch:
#   tail -f logs/emactl_gpu0.log
#   tail -f logs/emactl_gpu1.log

METHODS="flexlora hetero_pad homo_r8"
ALPHA=0.1
SEEDS="42 43 44 45 46"   # change to "42 43 44" for 3-seed overnight run
RESULTS_DIR="results_ema_control"

mkdir -p logs "${RESULTS_DIR}/yelp"

# Split seeds: even → GPU 0, odd → GPU 1
GPU0_SEEDS=""; GPU1_SEEDS=""
for s in $SEEDS; do
    (( s % 2 == 0 )) && GPU0_SEEDS="$GPU0_SEEDS $s" || GPU1_SEEDS="$GPU1_SEEDS $s"
done

# Write per-GPU worker scripts (avoids function-export fragility)
for GPU in 0 1; do
    [ "$GPU" -eq 0 ] && GSEEDS="$GPU0_SEEDS" || GSEEDS="$GPU1_SEEDS"
    cat > "/tmp/emactl_gpu${GPU}.sh" <<WORKER
#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=${GPU}
cd $(pwd)
for METHOD in ${METHODS}; do
  for SEED in ${GSEEDS}; do
    OUTFILE="${RESULTS_DIR}/yelp/\${METHOD}_alpha${ALPHA/./}_seed\${SEED}.json"
    if [ -f "\$OUTFILE" ]; then
      echo "[gpu${GPU}] Skipping \${METHOD} seed=\${SEED} — already done."
      continue
    fi
    echo "[gpu${GPU}][\$(date '+%H:%M')] START \${METHOD} seed=\${SEED} alpha=${ALPHA}"
    python experiments/run_yelp.py \\
      --method "\$METHOD" \\
      --alpha "${ALPHA}" \\
      --seed "\$SEED" \\
      --ema-eval \\
      --device "cuda:0" \\
      --results-dir "${RESULTS_DIR}"
    echo "[gpu${GPU}][\$(date '+%H:%M')] DONE  \${METHOD} seed=\${SEED}"
  done
done
echo "[gpu${GPU}] ALL DONE"
WORKER
    chmod +x "/tmp/emactl_gpu${GPU}.sh"
done

# Launch with nohup — jobs survive SSH disconnect
nohup bash /tmp/emactl_gpu0.sh >> logs/emactl_gpu0.log 2>&1 &
PID0=$!
disown $PID0

nohup bash /tmp/emactl_gpu1.sh >> logs/emactl_gpu1.log 2>&1 &
PID1=$!
disown $PID1

echo "Launched GPU 0 (pid $PID0): seeds$GPU0_SEEDS"
echo "Launched GPU 1 (pid $PID1): seeds$GPU1_SEEDS"
echo "Output: ${RESULTS_DIR}/yelp/"
echo ""
echo "Monitor:"
echo "  tail -f logs/emactl_gpu0.log"
echo "  tail -f logs/emactl_gpu1.log"
echo ""
echo "Check running jobs: ps aux | grep emactl"
