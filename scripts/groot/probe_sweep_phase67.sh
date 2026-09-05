#!/bin/bash
# GR00T probe sweep phases 6 (training dynamics) + 7 (regularization). Fork of
# scripts/perception_probe/run_phase67.sh. Both on default arch, raw-target BCE +
# cumsum, batch 8, seed 0. 2 phases in parallel, staggered.
#   screen -dmS grootprobe67 bash scripts/groot/probe_sweep_phase67.sh
# Results: outputs/groot/probe_sweeps/phase{6,7}/summary.csv
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
OUT=outputs/groot/probe_sweeps
mkdir -p "$OUT"

echo "$(date '+%F %T') phases 6+7 starting"
pids=()
for P in 6 7; do
    $PY -u scripts/groot/probe_sweep.py --phase "$P" --seeds 0 \
        --epochs 10 --patience 20 --num-workers 2 --prefetch-factor 2 \
        --batch-size 8 --max-steps 0 \
        > "$OUT/phase${P}.driver.log" 2>&1 &
    pids+=("$!")
    echo "  phase $P -> pid $!"
    sleep 90
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
echo "$(date '+%F %T') phases 6+7 done (fail=$fail)"
for P in 6 7; do echo "=== phase $P ==="; column -s, -t "$OUT/phase${P}/summary.csv" 2>/dev/null | head -6; done
