#!/usr/bin/env bash
# Iterates collect_features.py over every suite and every task (0-9) with 25
# rollouts per scene variant. Resumable: collect_features.py skips any
# episode_*.npz that already exists on disk, so re-running this after an
# interruption just picks up where it left off.
#
# Requires the feature-serving websocket server to already be running
# (scripts/perception_probe/serve_pi05_with_features.py) before this starts.
#
# Usage:
#   NUM_TRIALS=5 scripts/perception_probe/collect_all.sh [extra args passed through to collect_features.py]
#
# Example:
#   NUM_TRIALS=5 scripts/perception_probe/collect_all.sh --host 127.0.0.1 --port 8000

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUITES=(libero_spatial_occluded libero_goal_occluded libero_object_occluded libero_10_occluded)
NUM_TASKS=10
NUM_TRIALS="${NUM_TRIALS:-25}"

LOG_DIR="${REPO_ROOT}/outputs/perception_probe/logs"
mkdir -p "${LOG_DIR}"

for suite in "${SUITES[@]}"; do
  for ((task_id=0; task_id<NUM_TASKS; task_id++)); do
    log_file="${LOG_DIR}/${suite}_task$(printf '%02d' "${task_id}").log"
    echo "==> ${suite} task ${task_id} (log: ${log_file})"
    "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/perception_probe/collect_features.py" \
      --suite "${suite}" \
      --task-id "${task_id}" \
      --num-trials "${NUM_TRIALS}" \
      "$@" \
      2>&1 | tee "${log_file}"
  done
done

echo "All suites/tasks collected."
