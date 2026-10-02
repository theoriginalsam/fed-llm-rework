#!/usr/bin/env bash
# Eval-side FedMoLoRA runs: HetLoRA trained normally, with the bias-corrected EMA
# (beta=0.5) evaluated alongside the raw aggregate every round. Each run yields
#   acc_raw_eval -> HetLoRA row
#   acc_ema_eval -> FedMoLoRA row (the method as described in the paper)
# paired within the same run. Yelp alpha=0.1 also logs beta 0.3 and 0.7 (beta ablation).
#
# 35 runs: Yelp {0.1, 0.01, 0.5} + GSM8K {0.5, 0.1, 0.01} + Alpaca, 5 seeds each.
# Jobs alternate across GPUs in priority order; finished runs are skipped, so the
# script can be re-launched after an interruption.
#
# Usage (on sp2ai):
#   cd ~/FedLLM-Re/rework
#   bash run_evalside.sh            # GPUs 0 and 1
#   GPUS="0" bash run_evalside.sh   # single GPU
#
# Monitor:
#   tail -f logs/evalside_gpu0.log logs/evalside_gpu1.log
#   ls results_evalside/*/ | wc -l          # finished runs (35 when done)

GPUS="${GPUS:-0 1}"
RESULTS_DIR="results_evalside"
SEEDS="42 43 44 45 46"

mkdir -p logs "${RESULTS_DIR}/yelp" "${RESULTS_DIR}/gsm8k" "${RESULTS_DIR}/alpaca"

# dataset alpha rounds extra-args (priority order: headline setting first)
JOBS=()
for s in $SEEDS; do JOBS+=("yelp 0.1 20 $s --ema-betas 0.3 0.7"); done
for s in $SEEDS; do JOBS+=("yelp 0.01 40 $s"); done
for s in $SEEDS; do JOBS+=("yelp 0.5 20 $s"); done
for s in $SEEDS; do JOBS+=("gsm8k 0.5 20 $s"); done
for s in $SEEDS; do JOBS+=("gsm8k 0.1 20 $s"); done
for s in $SEEDS; do JOBS+=("gsm8k 0.01 40 $s"); done
for s in $SEEDS; do JOBS+=("alpaca 0.1 20 $s"); done

read -r -a GPU_LIST <<< "$GPUS"
NGPU=${#GPU_LIST[@]}

for gi in "${!GPU_LIST[@]}"; do
    GPU=${GPU_LIST[$gi]}
    WORKER="/tmp/evalside_gpu${GPU}.sh"
    {
        echo "#!/usr/bin/env bash"
        echo "export CUDA_VISIBLE_DEVICES=${GPU}"
        echo "cd $(pwd)"
        for ji in "${!JOBS[@]}"; do
            (( ji % NGPU == gi )) || continue
            read -r DS ALPHA ROUNDS SEED EXTRA <<< "${JOBS[$ji]}"
            OUT="${RESULTS_DIR}/${DS}/hetlora_seed${SEED}_alpha${ALPHA/./}.json"
            cat <<JOB
if [ -f "${OUT}" ]; then
  echo "[gpu${GPU}] skip ${DS} alpha=${ALPHA} seed=${SEED} (done)"
else
  echo "[gpu${GPU}][\$(date '+%m-%d %H:%M')] START ${DS} alpha=${ALPHA} seed=${SEED} rounds=${ROUNDS}"
  python experiments/run_${DS}.py --method hetlora --alpha ${ALPHA} --seed ${SEED} \\
    --num-rounds ${ROUNDS} --ema-eval ${EXTRA} --device cuda:0 --results-dir ${RESULTS_DIR}
  echo "[gpu${GPU}][\$(date '+%m-%d %H:%M')] END   ${DS} alpha=${ALPHA} seed=${SEED} exit=\$?"
fi
JOB
        done
        echo "echo \"[gpu${GPU}] ALL DONE \$(date)\""
    } > "$WORKER"
    chmod +x "$WORKER"
    nohup bash "$WORKER" >> "logs/evalside_gpu${GPU}.log" 2>&1 &
    disown $!
    echo "Launched GPU ${GPU} (pid $!): $(grep -c 'START' "$WORKER") jobs"
done

echo "Output: ${RESULTS_DIR}/{yelp,gsm8k,alpaca}/"
echo "Monitor: tail -f logs/evalside_gpu*.log"
