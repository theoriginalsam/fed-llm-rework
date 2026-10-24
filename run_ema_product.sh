#!/usr/bin/env bash
# Product-EMA vs factor-EMA (reviewer request, BigData 2026 camera-ready).
# HetLoRA trained normally; every round evaluates four models of the same trajectory:
#   acc_raw_eval              raw aggregate, sliced to rank 8            (HetLoRA)
#   acc_ema_eval              EMA of B and A, sliced to rank 8           (FedMoLoRA)
#   acc_ema_factor_svd_eval   EMA of B and A, product SVD-truncated to 8
#   acc_ema_product_svd_eval  EMA of Delta W = B A, SVD-truncated to 8
# product vs factor_svd isolates where the average is taken.
#
# 10 runs: Yelp alpha=0.1 (20 rounds) and alpha=0.01 (40 rounds), seeds 42-46.
# Finished runs are skipped, so the script can be re-launched after an interruption.
#
# Usage (on the Blackwell box):
#   cd ~/FedLLM-Re/rework && git fetch origin && git checkout exp/ema-product
#   bash run_ema_product.sh
# Monitor:
#   tail -f logs/emaprod_gpu0.log logs/emaprod_gpu1.log
#   ls results_emaprod/yelp/*.json | wc -l        # 10 when done
#   python experiments/summarize_ema_product.py

GPUS="${GPUS:-0 1}"
RESULTS_DIR="results_emaprod"
SEEDS="42 43 44 45 46"

mkdir -p logs "${RESULTS_DIR}/yelp"

JOBS=()
for s in $SEEDS; do JOBS+=("0.1 20 $s"); done
for s in $SEEDS; do JOBS+=("0.01 40 $s"); done

read -r -a GPU_LIST <<< "$GPUS"
NGPU=${#GPU_LIST[@]}

for gi in "${!GPU_LIST[@]}"; do
    GPU=${GPU_LIST[$gi]}
    WORKER="/tmp/emaprod_gpu${GPU}.sh"
    {
        echo "#!/usr/bin/env bash"
        echo "export CUDA_VISIBLE_DEVICES=${GPU}"
        echo "export FEDLLM_COPY_ON_GPU=1"
        echo "cd $(pwd)"
        for ji in "${!JOBS[@]}"; do
            (( ji % NGPU == gi )) || continue
            read -r ALPHA ROUNDS SEED <<< "${JOBS[$ji]}"
            OUT="${RESULTS_DIR}/yelp/hetlora_seed${SEED}_alpha${ALPHA/./}.json"
            cat <<JOB
if [ -f "${OUT}" ]; then
  echo "[gpu${GPU}] skip alpha=${ALPHA} seed=${SEED} (done)"
else
  echo "[gpu${GPU}][\$(date '+%m-%d %H:%M')] START yelp alpha=${ALPHA} seed=${SEED} rounds=${ROUNDS}"
  python experiments/run_yelp.py --method hetlora --alpha ${ALPHA} --seed ${SEED} \\
    --num-rounds ${ROUNDS} --ema-eval --ema-product --device cuda:0 --results-dir ${RESULTS_DIR} < /dev/null
  rc=\$?
  echo "[gpu${GPU}][\$(date '+%m-%d %H:%M')] END   yelp alpha=${ALPHA} seed=${SEED} exit=\$rc"
fi
JOB
        done
        echo "echo \"[gpu${GPU}] ALL DONE \$(date)\""
    } > "$WORKER"
    chmod +x "$WORKER"
    nohup bash "$WORKER" >> "logs/emaprod_gpu${GPU}.log" 2>&1 &
    disown $!
    echo "Launched GPU ${GPU}: $(grep -c 'START' "$WORKER") jobs"
done

echo "Output: ${RESULTS_DIR}/yelp/"
echo "Monitor: tail -f logs/emaprod_gpu*.log"
