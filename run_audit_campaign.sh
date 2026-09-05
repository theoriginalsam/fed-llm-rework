#!/usr/bin/env bash
#
# Audit campaign: generate LoRA adapters across datasets, methods and seeds
# so the FedGT spectral audit results can be tested for generality.
#
# Grid: 3 datasets x 2 methods x 3 seeds = 18 runs at 20 rounds each.
# Roughly 1.5 to 2 hours per run on an RTX PRO 6000 Blackwell, so budget
# 27 to 36 hours. Safe to stop and restart: finished runs are skipped.
#
# Usage:
#   ./run_audit_campaign.sh                  # whole grid
#   ./run_audit_campaign.sh --dry-run        # print what would run
#   DATASETS="yelp" ./run_audit_campaign.sh  # override any axis
#
set -u

DATASETS="${DATASETS:-yelp alpaca gsm8k}"
METHODS="${METHODS:-hetero_spa flexlora}"
SEEDS="${SEEDS:-42 43 44}"
ROUNDS="${ROUNDS:-20}"
CLIENTS_PER_ROUND="${CLIENTS_PER_ROUND:-10}"
ALPHA="${ALPHA:-0.5}"
ROOT="${ROOT:-results_campaign}"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

# GSM8K has much longer sequences than Yelp or Alpaca, so it needs a smaller
# batch to stay inside VRAM. Everything else uses the config default.
batch_for () { [ "$1" = "gsm8k" ] && echo "--batch-size 2" || echo ""; }

total=0; done_already=0; ok=0; failed=0
started=$(date +%s)

echo "=================================================================="
echo " FedGT audit campaign"
echo " datasets : $DATASETS"
echo " methods  : $METHODS"
echo " seeds    : $SEEDS"
echo " rounds   : $ROUNDS, clients per round: $CLIENTS_PER_ROUND"
echo " HF_HOME  : $HF_HOME"
echo " started  : $(date)"
echo "=================================================================="

for ds in $DATASETS; do
  for m in $METHODS; do
    for s in $SEEDS; do
      total=$((total+1))
      OUT="$ROOT/${ds}_${m}_s${s}"
      RUNNER="experiments/run_${ds}.py"

      if [ ! -f "$RUNNER" ]; then
        echo "[skip] no runner for $ds ($RUNNER missing)"; failed=$((failed+1)); continue
      fi
      if [ -d "$OUT/adapters" ] && [ "$(ls -A "$OUT/adapters" 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "[done] $OUT already has adapters, skipping"
        done_already=$((done_already+1)); continue
      fi

      CMD="python $RUNNER --method $m --alpha $ALPHA --seed $s \
--num-rounds $ROUNDS --clients-per-round $CLIENTS_PER_ROUND \
--results-dir $OUT --save-adapters-dir $OUT/adapters $(batch_for $ds)"

      echo
      echo "------------------------------------------------------------------"
      echo "[run ] $ds | $m | seed $s | $(date +%H:%M:%S)"
      echo "       $CMD"
      echo "------------------------------------------------------------------"

      if [ "$DRY" = "1" ]; then continue; fi

      t0=$(date +%s)
      if $CMD; then
        n=$(ls -A "$OUT/adapters" 2>/dev/null | wc -l)
        echo "[ok  ] $ds $m s$s finished in $((($(date +%s)-t0)/60)) min, $n adapter files"
        ok=$((ok+1))
      else
        echo "[FAIL] $ds $m s$s exited nonzero after $((($(date +%s)-t0)/60)) min"
        failed=$((failed+1))
      fi
    done
  done
done

echo
echo "=================================================================="
echo " campaign finished $(date)"
echo " total $total | ok $ok | already done $done_already | failed $failed"
echo " elapsed $(( ($(date +%s)-started)/3600 )) h"
echo " adapters under: $ROOT/*/adapters"
echo "=================================================================="
