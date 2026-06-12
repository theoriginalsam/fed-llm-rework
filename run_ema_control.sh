#!/usr/bin/env bash
# EMA-eval control experiment (tab:ema_control in paper).
# Reruns homo_r8, hetero_pad, flexlora on Yelp alpha=0.1 with --ema_eval.
# Results saved to results_ema_control/yelp/ — separate from main results.
# Splits seeds across both GPUs.
#
# Runtime: ~50 GPU-h total (15 runs × ~3.3h each) ≈ ~1.5 days on 2× A6000.
# 3 seeds instead of 5 cuts to ~overnight if needed (edit SEEDS below).
#
# Usage (on sp2ai):
#   cd ~/FedLLM-Re/rework
#   bash run_ema_control.sh
#
# To watch progress:
#   tail -f logs/emactl_gpu0.log
#   tail -f logs/emactl_gpu1.log

set -e

METHODS="flexlora hetero_pad homo_r8"
ALPHA=0.1
SEEDS="42 43 44 45 46"   # set to "42 43 44" for 3-seed overnight run

RESULTS_DIR="results_ema_control"
mkdir -p logs "${RESULTS_DIR}/yelp"

# Split seeds: even seeds → GPU 0, odd seeds → GPU 1
GPU0_SEEDS=""
GPU1_SEEDS=""
for s in $SEEDS; do
    if (( s % 2 == 0 )); then
        GPU0_SEEDS="$GPU0_SEEDS $s"
    else
        GPU1_SEEDS="$GPU1_SEEDS $s"
    fi
done

run_gpu() {
    local GPU=$1; shift
    local GPU_SEEDS="$*"
    export CUDA_VISIBLE_DEVICES=$GPU

    for METHOD in $METHODS; do
        for SEED in $GPU_SEEDS; do
            OUTFILE="${RESULTS_DIR}/yelp/${METHOD}_alpha${ALPHA/./}_seed${SEED}.json"
            if [ -f "$OUTFILE" ]; then
                echo "[gpu${GPU}] Skipping ${METHOD} seed=${SEED} — already done."
                continue
            fi
            echo "[gpu${GPU}][$(date '+%H:%M')] START ${METHOD} seed=${SEED} alpha=${ALPHA}"
            python experiments/run_yelp.py \
                --method "$METHOD" \
                --alpha "$ALPHA" \
                --seed "$SEED" \
                --ema-eval \
                --device "cuda:0" \
                --results-dir "$RESULTS_DIR"
            echo "[gpu${GPU}][$(date '+%H:%M')] DONE  ${METHOD} seed=${SEED}"
        done
    done
    echo "[gpu${GPU}] ALL DONE"
}

# Launch both GPU workers in background, each writing its own log
run_gpu 0 $GPU0_SEEDS >> logs/emactl_gpu0.log 2>&1 &
PID0=$!
run_gpu 1 $GPU1_SEEDS >> logs/emactl_gpu1.log 2>&1 &
PID1=$!

echo "Launched GPU 0 (pid $PID0): seeds $GPU0_SEEDS → logs/emactl_gpu0.log"
echo "Launched GPU 1 (pid $PID1): seeds $GPU1_SEEDS → logs/emactl_gpu1.log"
echo "Output files: ${RESULTS_DIR}/yelp/"
echo ""
echo "Monitor with:"
echo "  tail -f logs/emactl_gpu0.log"
echo "  tail -f logs/emactl_gpu1.log"

wait $PID0 && echo "GPU 0 finished."
wait $PID1 && echo "GPU 1 finished."

echo ""
echo "=== All EMA control runs complete ==="
echo "Next: run analysis/summarize_ema_control.py to build tab:ema_control"
