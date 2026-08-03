#!/usr/bin/env bash
# Fill SPA-M seeds 45 and 46 for Yelp and GSM8K (α=0.5 and α=0.1, 20 rounds).
# Results saved to results_prof/ alongside the other prof_feedback results.
#
# GPU 0: seed 46 (even)
# GPU 1: seed 45 (odd)
#
# Usage:
#   cd ~/FedLLM-Re/rework
#   bash run_spam_fill.sh
#
# Monitor:
#   tail -f logs/spam_gpu0.log
#   tail -f logs/spam_gpu1.log

RESULTS_DIR="results_prof"
mkdir -p logs "${RESULTS_DIR}/yelp" "${RESULTS_DIR}/gsm8k"

for GPU in 0 1; do
    if [ "$GPU" -eq 0 ]; then
        SEED=46
    else
        SEED=45
    fi

    cat > "/tmp/spam_gpu${GPU}.sh" <<WORKER
#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=${GPU}
cd $(pwd)
LOG="logs/spam_gpu${GPU}.log"

echo "=== [gpu${GPU}] SPA-M fill started \$(date) ===" >> "\$LOG"

for ALPHA in 0.5 0.1; do
    ALPHA_TAG="\${ALPHA/./}"

    # Yelp
    OUTFILE="${RESULTS_DIR}/yelp/spa_m_seed${SEED}_alpha\${ALPHA_TAG}.json"
    if [ -f "\$OUTFILE" ]; then
        echo "[gpu${GPU}] Skipping Yelp spa_m seed=${SEED} alpha=\${ALPHA} — already done." >> "\$LOG"
    else
        echo "[gpu${GPU}][\$(date '+%H:%M')] START Yelp spa_m seed=${SEED} alpha=\${ALPHA}" >> "\$LOG"
        python experiments/run_yelp.py \
            --method spa_m \
            --alpha "\${ALPHA}" \
            --seed ${SEED} \
            --device cuda:0 \
            --results-dir "${RESULTS_DIR}" \
            >> "\$LOG" 2>&1
        echo "[gpu${GPU}][\$(date '+%H:%M')] DONE  Yelp spa_m seed=${SEED} alpha=\${ALPHA}" >> "\$LOG"
    fi

    # GSM8K
    OUTFILE="${RESULTS_DIR}/gsm8k/spa_m_alpha\${ALPHA_TAG}_seed${SEED}.json"
    if [ -f "\$OUTFILE" ]; then
        echo "[gpu${GPU}] Skipping GSM8K spa_m seed=${SEED} alpha=\${ALPHA} — already done." >> "\$LOG"
    else
        echo "[gpu${GPU}][\$(date '+%H:%M')] START GSM8K spa_m seed=${SEED} alpha=\${ALPHA}" >> "\$LOG"
        python experiments/run_gsm8k.py \
            --method spa_m \
            --alpha "\${ALPHA}" \
            --seed ${SEED} \
            --device cuda:0 \
            --results-dir "${RESULTS_DIR}" \
            >> "\$LOG" 2>&1
        echo "[gpu${GPU}][\$(date '+%H:%M')] DONE  GSM8K spa_m seed=${SEED} alpha=\${ALPHA}" >> "\$LOG"
    fi

done

echo "=== [gpu${GPU}] SPA-M fill DONE \$(date) ===" >> "\$LOG"
WORKER
    chmod +x "/tmp/spam_gpu${GPU}.sh"
done

nohup bash /tmp/spam_gpu0.sh >> logs/spam_gpu0.log 2>&1 &
PID0=$!
disown $PID0

nohup bash /tmp/spam_gpu1.sh >> logs/spam_gpu1.log 2>&1 &
PID1=$!
disown $PID1

echo "Launched GPU 0 (pid $PID0) — seed 46"
echo "Launched GPU 1 (pid $PID1) — seed 45"
echo ""
echo "Monitor:"
echo "  tail -f logs/spam_gpu0.log"
echo "  tail -f logs/spam_gpu1.log"
echo ""
echo "Check done:"
echo "  grep 'DONE\|Skipping' logs/spam_gpu0.log logs/spam_gpu1.log"
