#!/bin/bash
# GPU 1: flexlora + homo_r8 + hetero_pad — all seeds, both alphas
export CUDA_VISIBLE_DEVICES=1
cd ~/FedLLM-Re/rework
mkdir -p logs results/yelp results/alpaca results/gsm8k results/ablation

echo "=== GPU 1 starting: flexlora + homo_r8 + hetero_pad ===" >> logs/gpu1.log

for METHOD in flexlora homo_r8 hetero_pad; do
  for ALPHA in 0.5 0.1; do
    for SEED in 42 43 44 45 46; do
      OUTFILE="results/yelp/${METHOD}_alpha${ALPHA/./}_seed${SEED}.json"
      if [ -f "$OUTFILE" ]; then
        echo "Skipping ${METHOD}_alpha${ALPHA}_seed${SEED} — done" >> logs/gpu1.log
        continue
      fi
      echo "[$(date)] Starting ${METHOD} alpha=${ALPHA} seed=${SEED}" >> logs/gpu1.log
      python experiments/run_yelp.py \
        --method $METHOD --alpha $ALPHA --seed $SEED \
        >> logs/gpu1.log 2>&1
      echo "[$(date)] Done" >> logs/gpu1.log
    done
  done
done

# Alpaca — flexlora + homo_r8 + hetero_pad
for METHOD in flexlora homo_r8 hetero_pad; do
  for SEED in 42 43 44; do
    echo "[$(date)] Alpaca $METHOD seed=$SEED" >> logs/gpu1.log
    python experiments/run_alpaca.py --method $METHOD --seed $SEED >> logs/gpu1.log 2>&1
  done
done

# GSM8K — all methods
for METHOD in hetero_spa flexlora homo_r8 hetero_pad homo_r4; do
  for SEED in 42 43 44; do
    echo "[$(date)] GSM8K $METHOD seed=$SEED" >> logs/gpu1.log
    python experiments/run_gsm8k.py --method $METHOD --seed $SEED >> logs/gpu1.log 2>&1
  done
done

echo "=== GPU 1 COMPLETE ===" >> logs/gpu1.log
