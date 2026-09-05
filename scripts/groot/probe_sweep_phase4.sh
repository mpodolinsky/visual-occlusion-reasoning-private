#!/bin/bash
# GR00T probe sweep phase 4 (alternative loss functions). Fork of
# scripts/perception_probe/run_phase4.sh. Own screen session:
#   screen -dmS grootprobe4 bash scripts/groot/probe_sweep_phase4.sh
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "$(date '+%F %T') phase 4 starting"
.venv/bin/python -u scripts/groot/probe_sweep.py --phase 4 --seeds 0 \
    --num-workers 2 --prefetch-factor 1 --batch-size 4 --max-steps 0 \
    > outputs/groot/probe_sweeps/phase4.driver.log 2>&1
echo "$(date '+%F %T') phase 4 done"
head -8 outputs/groot/probe_sweeps/phase4/summary.csv 2>/dev/null
