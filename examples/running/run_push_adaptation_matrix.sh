#!/usr/bin/env bash
# Push-disturbance adaptation matrix for the moving-target experiment.
#
# Grid: 2 models (plastic / static) x 2 target motions (stationary / circle)
#       x 2 disturbance conditions (push / none).
#
# The plastic model is the reference experiment; the static model is the
# trained static-network control (CMA-ES optimized W directly, lr_hebb=0).
# Both are evaluated with the SAME task config (max_steps=1600, n_agents=8)
# so the comparison is apples-to-apples.
#
# The push is a one-shot radial-out velocity impulse (magnitude 0.5) on
# agent 2 at step 800 — identical to the pre-existing
# disturbance_eval_push baseline, so the results are directly comparable.
#
# Each run writes to <experiment>/disturbance_eval_push|/, which OVERWRITES
# on repeat — so the per-condition npz is copied out to a condition-named
# file by collect_push_matrix.py afterwards.
#
# Usage: bash examples/running/run_push_adaptation_matrix.sh

set -u
cd /home/zhaozeming/BenchMARL

PY=/home/zhaozeming/miniconda3/envs/benchmarl/bin/python
SCRIPT=examples/running/run_cmaes_han_flocking_disturbance.py

PLASTIC=examples/outputs/cmaeshan_flocking_hanmodel__b140e5f5_26_06_23-17_05_30-8-agents
STATIC=examples/outputs/cmaeshan_flocking_hanmodel__5de64e71_26_10_21-19_59_20

MAX_STEPS=1600
DIST_STEP=800
PUSH_MAG=0.5
FROZEN_IDX=2
FPS=20
VIDEO_FRAMES=200

SNAPSHOT_DIR=/tmp/push_matrix
mkdir -p "$SNAPSHOT_DIR"

run_one () {
    local exp=$1 motion=$2 mode=$3 label=$4
    local model=$5
    echo ""
    echo "=============================================================="
    echo "RUN: $label  (model=$model, motion=$motion, mode=$mode)"
    echo "=============================================================="
    $PY $SCRIPT \
        --experiment-path "$exp" \
        --num-episodes 1 \
        --max-steps $MAX_STEPS \
        --disturbance-step $DIST_STEP \
        --disturbance-mode "$mode" \
        --push-magnitude $PUSH_MAG \
        --push-direction radial-out \
        --frozen-agent-idx $FROZEN_IDX \
        --fitness-mode flocking_orbit \
        --target-motion "$motion" \
        --target-motion-radius 0.35 \
        --target-motion-period 400 \
        --target-pos-x 0.0 --target-pos-y 0.0 \
        --fps $FPS --max-video-frames $VIDEO_FRAMES \
        2>&1 | grep -vE "UserWarning|warnings.warn|register_pytree|^\s*$"
    # Snapshot the per-step data NOW: the disturbance script writes to
    # <exp>/disturbance_eval_<mode>/ (the directory name follows the
    # mode), and a later run of the same mode overwrites it.
    local npz="$exp/disturbance_eval_${mode}/per_step_data.npz"
    if [ -f "$npz" ]; then
        cp "$npz" "$SNAPSHOT_DIR/${model}_${motion}_${mode}.npz"
        echo "  snapshot -> $SNAPSHOT_DIR/${model}_${motion}_${mode}.npz"
    else
        echo "  WARNING: no per_step_data.npz at $npz for ${model}_${motion}_${mode}"
    fi
}

run_one "$PLASTIC" stationary push   "plastic | stationary | push"   plastic
run_one "$PLASTIC" circle    push   "plastic | circle    | push"   plastic
run_one "$PLASTIC" stationary none   "plastic | stationary | none"  plastic
run_one "$PLASTIC" circle    none   "plastic | circle    | none"   plastic
run_one "$STATIC"  stationary push   "static  | stationary | push"   static
run_one "$STATIC"  circle    push   "static  | circle    | push"   static
run_one "$STATIC"  stationary none   "static  | stationary | none"  static
run_one "$STATIC"  circle    none   "static  | circle    | none"   static

echo ""
echo "Matrix complete. Snapshots in $SNAPSHOT_DIR"
