#!/bin/bash
# Phase 5 -- best-combo confirmation. Own screen session:
#   screen -dmS probesweep5 bash scripts/perception_probe/run_phase5.sh
#   screen -r probesweep5
#
# 3 cells (default / combo / combo_plus), seed 0 = 3 runs. batch 8 (matches
# pre-sweep prod); --epochs 10 (batch 8 converges slower per-epoch than the
# batch-6 sweep, and combo's lower LR + shared pool slow it further -- 10 keeps
# best_epoch ~7-9, still shy of the overfit zone). Base = raw-target BCE + cumsum.
# Results: outputs/perception_probe/sweeps/phase5/summary.csv
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "$(date '+%F %T') phase 5 starting"
.venv/bin/python -u scripts/perception_probe/sweep.py --phase 5 --seeds 0 \
    --epochs 10 --patience 10 --num-workers 3 --prefetch-factor 2 \
    --batch-size 8 --max-steps 0 \
    > outputs/perception_probe/sweeps/phase5.driver.log 2>&1
echo "$(date '+%F %T') phase 5 done"
column -s, -t outputs/perception_probe/sweeps/phase5/summary.csv 2>/dev/null | head -6
