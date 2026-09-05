#!/bin/bash
# GR00T perception-probe sweep phases 1/2/3 concurrently (1 seed each). Fork of
# scripts/perception_probe/run_sweep_parallel.sh. Launch inside screen/tmux:
#
#   screen -dmS grootprobe bash scripts/groot/probe_sweep_parallel.sh
#   screen -r grootprobe
#
# Per-run: batch 6, workers 2, prefetch 1 -> ~6-7GB GPU / ~4GB shm each, so 3
# fit on a 24GB card. Progress: outputs/groot/probe_sweeps/phaseN.driver.log +
# phaseN/summary.csv.
set -u
cd "$(dirname "$0")/../.." || exit 1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
OUT=outputs/groot/probe_sweeps
mkdir -p "$OUT"

echo "$(date '+%F %T')  starting parallel GR00T probe sweep (phases 1 2 3, seed 0, --max-steps 0)"
pids=()
for P in 1 2 3; do
    $PY -u scripts/groot/probe_sweep.py \
        --phase "$P" --seeds 0 --num-workers 2 --prefetch-factor 1 \
        --batch-size 6 --max-steps 0 \
        > "$OUT/phase${P}.driver.log" 2>&1 &
    pids+=("$!")
    echo "  phase $P -> pid $!"
    sleep 120
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
