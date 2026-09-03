#!/bin/bash
# Phase 4 (alternative loss functions) sweep runner -- own screen session.
#   screen -dmS probesweep4 bash scripts/perception_probe/run_phase4.sh
#   screen -r probesweep4
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "$(date '+%F %T') phase 4 starting"
.venv/bin/python -u scripts/perception_probe/sweep.py --phase 4 --seeds 0 \
    --num-workers 2 --prefetch-factor 1 --batch-size 4 --max-steps 0 \
    > outputs/perception_probe/sweeps/phase4.driver.log 2>&1
echo "$(date '+%F %T') phase 4 done"
head -8 outputs/perception_probe/sweeps/phase4/summary.csv 2>/dev/null
