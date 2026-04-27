#!/bin/bash
# GPU 0: hetero_spa + homo_r4 — all seeds, both alphas
export CUDA_VISIBLE_DEVICES=0
cd ~/FedLLM-Re/rework
mkdir -p logs results/yelp results/alpaca results/gsm8k results/ablation

echo "=== GPU 0 starting: hetero_spa + homo_r4 ===" >> logs/gpu0.log

for METHOD in hetero_spa homo_r4; do
  for ALPHA in 0.5 0.1; do
    for SEED in 42 43 44 45 46; do
      TAG="${METHOD}_alpha${ALPHA/./}_seed${SEED}"
      OUTFILE="results/yelp/${METHOD}_alpha${ALPHA/./}_seed${SEED}.json"
      if [ -f "$OUTFILE" ]; then
        echo "Skipping $TAG — already done" >> logs/gpu0.log
        continue
      fi
      echo "[$(date)] Starting $TAG" >> logs/gpu0.log
      python experiments/run_yelp.py \
        --method $METHOD --alpha $ALPHA --seed $SEED \
        >> logs/gpu0.log 2>&1
      echo "[$(date)] Done $TAG" >> logs/gpu0.log
    done
  done
done

# Alpaca — hetero_spa only (most important)
for SEED in 42 43 44; do
  echo "[$(date)] Alpaca hetero_spa seed=$SEED" >> logs/gpu0.log
  python experiments/run_alpaca.py --method hetero_spa --seed $SEED >> logs/gpu0.log 2>&1
done

echo "=== GPU 0 COMPLETE ===" >> logs/gpu0.log
