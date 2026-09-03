#!/bin/bash
# Runs perception-probe sweep phases 1/2/3 concurrently (1 seed each), meant to
# be launched inside screen/tmux so it survives disconnection:
#
#   screen -dmS probesweep bash scripts/perception_probe/run_sweep_parallel.sh
#   screen -r probesweep      # reattach to watch
#
# UNCAPPED rerun: --max-steps 0 (full 104-step episodes, was clipped to 80 in
# the sweeps_maxsteps80/ run). Phase-2/3 base loss switched to cumsum
# (--raw-target-loss, no --rmean) since phase 1 showed cumsum >> rmean on unseen.
# Per-run: batch 6, workers 2, prefetch 1 -> each run ~6-7GB GPU / ~4GB shm, so
# 3 fit on the 24GB card and under 16GB /dev/shm. Progress:
# outputs/perception_probe/sweeps/phaseN.driver.log + phaseN/summary.csv.
set -u
cd "$(dirname "$0")/../.." || exit 1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
OUT=outputs/perception_probe/sweeps
mkdir -p "$OUT"

echo "$(date '+%F %T')  starting parallel sweep (phases 1 2 3, seed 0, --max-steps 0)"
pids=()
for P in 1 2 3; do
    $PY -u scripts/perception_probe/sweep.py \
        --phase "$P" --seeds 0 --num-workers 2 --prefetch-factor 1 \
        --batch-size 6 --max-steps 0 \
        > "$OUT/phase${P}.driver.log" 2>&1 &
    pids+=("$!")
    echo "  phase $P -> pid $!"
    sleep 120   # stagger so the 3 runs don't hit peak DataLoader memory in lockstep
done

fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done

echo "$(date '+%F %T')  ALL PHASES DONE (fail=$fail)"
for P in 1 2 3; do
    echo "=== phase $P top rows ==="
    head -6 "$OUT/phase${P}/summary.csv" 2>/dev/null
done
