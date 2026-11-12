#!/usr/bin/env bash
# Evader-pursuit comparison: plastic HAN vs trained static net.
# Sweet-spot speed from the scan: v=0.15 (40% of agent speed, S~1.0).
# Runs sequentially (parallel runs clobber <exp>/disturbance_eval_freeze/)
# and snapshots each npz + video under condition names immediately.
set -u
cd /home/zhaozeming/BenchMARL

PY=/home/zhaozeming/miniconda3/envs/benchmarl/bin/python
SCRIPT=examples/running/run_cmaes_han_flocking_disturbance.py

PLASTIC=examples/outputs/cmaeshan_flocking_hanmodel__b140e5f5_26_06_23-17_05_30-8-agents
STATIC=examples/outputs/cmaeshan_flocking_hanmodel__5de64e71_26_10_21-19_59_20

SNAP=/tmp/evader_compare
mkdir -p "$SNAP"
V=0.15
INTERVAL=30

run_one () {
    local exp=$1 model=$2
    local stamp="$SNAP/.stamp_${model}"
    touch "$stamp"
    echo "=== RUN: $model | evader v=$V ==="
    WANDB_MODE=offline $PY $SCRIPT \
        --experiment-path "$exp" \
        --num-episodes 1 --max-steps 1600 --disturbance-step 1599 \
        --disturbance-mode freeze \
        --fitness-mode flocking_orbit \
        --target-motion evader \
        --target-motion-speed $V --target-motion-interval $INTERVAL \
        --target-motion-bound 0.9 \
        --fps 20 --max-video-frames 1600 \
        > "$SNAP/${model}.log" 2>&1
    local rc=$?
    local npz="$exp/disturbance_eval_freeze/per_step_data.npz"
    local mp4="$exp/disturbance_eval_freeze/trajectory.mp4"
    if [ $rc -eq 0 ] && [ -f "$npz" ] && [ "$npz" -nt "$stamp" ]; then
        cp "$npz" "$SNAP/${model}_evader.npz"
        echo "  snapshot -> $SNAP/${model}_evader.npz"
    else
        echo "  WARNING: $model run failed or stale (rc=$rc)"
    fi
    if [ -f "$mp4" ] && [ "$mp4" -nt "$stamp" ]; then
        cp "$mp4" "$exp/evader_videos/${model}_evader_v015.mp4"
        echo "  video -> $exp/evader_videos/${model}_evader_v015.mp4"
    fi
}

mkdir -p "$PLASTIC/evader_videos" "$STATIC/evader_videos"
run_one "$PLASTIC" plastic
run_one "$STATIC"  static
echo "DONE. Snapshots in $SNAP"
