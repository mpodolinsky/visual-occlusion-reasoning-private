#!/bin/bash
# GR00T probe sweep phase 5 -- best-combo confirmation. Fork of
# scripts/perception_probe/run_phase5.sh. Own screen session:
#   screen -dmS grootprobe5 bash scripts/groot/probe_sweep_phase5.sh
#
# 3 cells (default / combo / combo_plus), seed 0. batch 8, --epochs 10.
# Base = raw-target BCE + cumsum. Results: outputs/groot/probe_sweeps/phase5/summary.csv
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "$(date '+%F %T') phase 5 starting"
.venv/bin/python -u scripts/groot/probe_sweep.py --phase 5 --seeds 0 \
    --epochs 10 --patience 10 --num-workers 3 --prefetch-factor 2 \
    --batch-size 8 --max-steps 0 \
    > outputs/groot/probe_sweeps/phase5.driver.log 2>&1
echo "$(date '+%F %T') phase 5 done"
column -s, -t outputs/groot/probe_sweeps/phase5/summary.csv 2>/dev/null | head -6
