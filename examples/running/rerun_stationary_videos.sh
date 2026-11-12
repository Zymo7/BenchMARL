#!/usr/bin/env bash
# Makeup runs: record the 4 stationary-target condition videos that were
# overwritten during the push matrix (the disturbance script reuses
# disturbance_eval_<mode>/ across runs, so only the LAST run's video of
# each mode survives). The circle-condition videos were already preserved
# as <exp>/push_matrix_videos/*_circle_*.mp4.
#
# After each run the trajectory.mp4 is renamed into push_matrix_videos/
# under its condition name. Data is unchanged from the matrix runs (same
# seed/config => same trajectories, verified by the ref-Fg match), so the
# npz snapshots in /tmp/push_matrix remain the source of truth for numbers.
set -u
cd /home/zhaozeming/BenchMARL

PY=/home/zhaozeming/miniconda3/envs/benchmarl/bin/python
SCRIPT=examples/running/run_cmaes_han_flocking_disturbance.py

PLASTIC=examples/outputs/cmaeshan_flocking_hanmodel__b140e5f5_26_06_23-17_05_30-8-agents
STATIC=examples/outputs/cmaeshan_flocking_hanmodel__5de64e71_26_10_21-19_59_20

run_and_save () {
    local exp=$1 model=$2 mode=$3
    echo "=== rerun: $model | stationary | $mode ==="
    $PY $SCRIPT \
        --experiment-path "$exp" \
        --num-episodes 1 --max-steps 1600 --disturbance-step 800 \
        --disturbance-mode "$mode" \
        --push-magnitude 0.5 --push-direction radial-out \
        --frozen-agent-idx 2 \
        --fitness-mode flocking_orbit \
        --target-motion stationary \
        --target-pos-x 0.0 --target-pos-y 0.0 \
        --fps 20 --max-video-frames 1600 \
        > /tmp/rerun_${model}_${mode}.log 2>&1
    local src="$exp/disturbance_eval_${mode}/trajectory.mp4"
    if [ -f "$src" ]; then
        cp "$src" "$exp/push_matrix_videos/${model}_stationary_${mode}.mp4"
        echo "  saved -> ${model}_stationary_${mode}.mp4"
    else
        echo "  WARNING: $src missing"
    fi
}

run_and_save "$PLASTIC" plastic push
run_and_save "$PLASTIC" plastic none
run_and_save "$STATIC"  static  push
run_and_save "$STATIC"  static  none
echo "All stationary-condition videos re-recorded."
