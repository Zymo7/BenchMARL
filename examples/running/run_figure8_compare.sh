#!/usr/bin/env bash
# Figure-8 target comparison: plastic HAN vs trained static net.
#
# The figure-8 period is auto-scaled (--target-motion-match-speed) so its
# mean speed matches the circle mode at the same radius:
#   circle   R=0.35, T=400 -> 0.005498 units/step
#   figure8  R=0.35, T=388 -> 0.005500 units/step   (0.04% error)
#
# Runs sequentially (parallel runs clobber <exp>/disturbance_eval_freeze/).
set -u
cd /home/zhaozeming/BenchMARL

PY=/home/zhaozeming/miniconda3/envs/benchmarl/bin/python
SCRIPT=examples/running/run_cmaes_han_flocking_disturbance.py

PLASTIC=examples/outputs/cmaeshan_flocking_hanmodel__b140e5f5_26_06_23-17_05_30-8-agents
STATIC=examples/outputs/cmaeshan_flocking_hanmodel__5de64e71_26_10_21-19_59_20
STATIC_MATCHED=examples/outputs/cmaeshan_flocking_hanmodel__7ae9ce55_26_10_22-11_43_24

SNAP=/tmp/figure8_compare
mkdir -p "$SNAP"

run_one () {
    local exp=$1 model=$2
    local stamp="$SNAP/.stamp_${model}"
    touch "$stamp"
    echo "=== RUN: $model | figure8 (speed-matched) ==="
    WANDB_MODE=offline $PY $SCRIPT \
        --experiment-path "$exp" \
        --num-episodes 1 --max-steps 1600 --disturbance-step 1599 \
        --disturbance-mode freeze \
        --fitness-mode flocking_orbit \
        --target-motion figure8 --target-motion-radius 0.35 \
        --target-motion-period 400 --target-motion-match-speed \
        --fps 20 --max-video-frames 1600 \
        > "$SNAP/${model}.log" 2>&1
    local rc=$?
    local npz="$exp/disturbance_eval_freeze/per_step_data.npz"
    local mp4="$exp/disturbance_eval_freeze/trajectory.mp4"
    if [ $rc -eq 0 ] && [ -f "$npz" ] && [ "$npz" -nt "$stamp" ]; then
        cp "$npz" "$SNAP/${model}_figure8.npz"
        echo "  snapshot -> $SNAP/${model}_figure8.npz"
    else
        echo "  WARNING: $model failed or stale (rc=$rc)"
    fi
    if [ -f "$mp4" ] && [ "$mp4" -nt "$stamp" ]; then
        mkdir -p "$exp/evader_videos"
        cp "$mp4" "$exp/evader_videos/${model}_figure8.mp4"
        echo "  video -> $exp/evader_videos/${model}_figure8.mp4"
    fi
}

run_one "$PLASTIC"        plastic
run_one "$STATIC"         static
run_one "$STATIC_MATCHED" static_matched
echo "DONE. Snapshots in $SNAP"
