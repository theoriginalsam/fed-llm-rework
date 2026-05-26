#!/usr/bin/env bash
# run_missing.sh — 38 missing runs, single GPU, sequential
# Yelp (2) → GSM8K alpha=0.1 (15) + hetlora_m alpha=0.5 (3) → Alpaca alpha=0.1 (15) + hetlora_m alpha=0.5 (3)
#
# Usage: bash run_missing.sh

set -e
cd "$(dirname "$0")"
mkdir -p logs

DEVICE=cuda:0

echo "Starting 38 missing runs on $DEVICE — $(date)"
echo "Logs: logs/missing_runs.log"

{

echo "=== Yelp: hetlora seed46 ==="
python experiments/run_yelp.py --method hetlora --alpha 0.5 --seed 46 --device $DEVICE --results-dir results_v2
python experiments/run_yelp.py --method hetlora --alpha 0.1 --seed 46 --device $DEVICE --results-dir results_v2

echo "=== GSM8K alpha=0.1: homo_r8 ==="
python experiments/run_gsm8k.py --method homo_r8   --alpha 0.1 --seed 42 --device $DEVICE
python experiments/run_gsm8k.py --method homo_r8   --alpha 0.1 --seed 43 --device $DEVICE
python experiments/run_gsm8k.py --method homo_r8   --alpha 0.1 --seed 44 --device $DEVICE

echo "=== GSM8K alpha=0.1: hetero_pad ==="
python experiments/run_gsm8k.py --method hetero_pad --alpha 0.1 --seed 42 --device $DEVICE
python experiments/run_gsm8k.py --method hetero_pad --alpha 0.1 --seed 43 --device $DEVICE
python experiments/run_gsm8k.py --method hetero_pad --alpha 0.1 --seed 44 --device $DEVICE

echo "=== GSM8K alpha=0.1: flexlora ==="
python experiments/run_gsm8k.py --method flexlora  --alpha 0.1 --seed 42 --device $DEVICE
python experiments/run_gsm8k.py --method flexlora  --alpha 0.1 --seed 43 --device $DEVICE
python experiments/run_gsm8k.py --method flexlora  --alpha 0.1 --seed 44 --device $DEVICE

echo "=== GSM8K alpha=0.1: hetlora ==="
python experiments/run_gsm8k.py --method hetlora   --alpha 0.1 --seed 42 --device $DEVICE
python experiments/run_gsm8k.py --method hetlora   --alpha 0.1 --seed 43 --device $DEVICE
python experiments/run_gsm8k.py --method hetlora   --alpha 0.1 --seed 44 --device $DEVICE

echo "=== GSM8K alpha=0.1: spa_m ==="
python experiments/run_gsm8k.py --method spa_m     --alpha 0.1 --seed 42 --device $DEVICE
python experiments/run_gsm8k.py --method spa_m     --alpha 0.1 --seed 43 --device $DEVICE
python experiments/run_gsm8k.py --method spa_m     --alpha 0.1 --seed 44 --device $DEVICE

echo "=== Alpaca alpha=0.1: homo_r8 ==="
python experiments/run_alpaca.py --method homo_r8   --alpha 0.1 --seed 42 --device $DEVICE
python experiments/run_alpaca.py --method homo_r8   --alpha 0.1 --seed 43 --device $DEVICE
python experiments/run_alpaca.py --method homo_r8   --alpha 0.1 --seed 44 --device $DEVICE

echo "=== Alpaca alpha=0.1: hetero_pad ==="
python experiments/run_alpaca.py --method hetero_pad --alpha 0.1 --seed 42 --device $DEVICE
python experiments/run_alpaca.py --method hetero_pad --alpha 0.1 --seed 43 --device $DEVICE
python experiments/run_alpaca.py --method hetero_pad --alpha 0.1 --seed 44 --device $DEVICE

echo "=== Alpaca alpha=0.1: flexlora ==="
python experiments/run_alpaca.py --method flexlora  --alpha 0.1 --seed 42 --device $DEVICE
python experiments/run_alpaca.py --method flexlora  --alpha 0.1 --seed 43 --device $DEVICE
python experiments/run_alpaca.py --method flexlora  --alpha 0.1 --seed 44 --device $DEVICE

echo "=== Alpaca alpha=0.1: hetlora ==="
python experiments/run_alpaca.py --method hetlora   --alpha 0.1 --seed 42 --device $DEVICE
python experiments/run_alpaca.py --method hetlora   --alpha 0.1 --seed 43 --device $DEVICE
python experiments/run_alpaca.py --method hetlora   --alpha 0.1 --seed 44 --device $DEVICE

echo "=== Alpaca alpha=0.1: spa_m ==="
python experiments/run_alpaca.py --method spa_m     --alpha 0.1 --seed 42 --device $DEVICE
python experiments/run_alpaca.py --method spa_m     --alpha 0.1 --seed 43 --device $DEVICE
python experiments/run_alpaca.py --method spa_m     --alpha 0.1 --seed 44 --device $DEVICE

echo "=== GSM8K hetlora_m alpha=0.5 seeds 42 43 44 ==="
python experiments/run_gsm8k.py --method hetlora_m  --alpha 0.5 --seed 42 --device $DEVICE --results-dir results_hetloram_b05 --hetlora-m-beta 0.5
python experiments/run_gsm8k.py --method hetlora_m  --alpha 0.5 --seed 43 --device $DEVICE --results-dir results_hetloram_b05 --hetlora-m-beta 0.5
python experiments/run_gsm8k.py --method hetlora_m  --alpha 0.5 --seed 44 --device $DEVICE --results-dir results_hetloram_b05 --hetlora-m-beta 0.5

echo "=== Alpaca hetlora_m alpha=0.5 seeds 42 43 44 ==="
python experiments/run_alpaca.py --method hetlora_m  --alpha 0.5 --seed 42 --device $DEVICE --results-dir results_hetloram_b05 --hetlora-m-beta 0.5
python experiments/run_alpaca.py --method hetlora_m  --alpha 0.5 --seed 43 --device $DEVICE --results-dir results_hetloram_b05 --hetlora-m-beta 0.5
python experiments/run_alpaca.py --method hetlora_m  --alpha 0.5 --seed 44 --device $DEVICE --results-dir results_hetloram_b05 --hetlora-m-beta 0.5

echo "=== All 38 runs complete — $(date) ==="

} > logs/missing_runs.log 2>&1 &

echo "PID: $!"
echo "Monitor: tail -f logs/missing_runs.log"
