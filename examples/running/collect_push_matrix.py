"""Collect the push-adaptation matrix into one comparison table.

The disturbance script always writes to ``<exp>/disturbance_eval_push/`` and
overwrites on repeat, so each condition's ``per_step_data.npz`` must be
copied aside before the next run. This script does that bookkeeping and
prints the final comparison.

Usage:
    python examples/running/collect_push_matrix.py --out /tmp/push_matrix
"""
import argparse
import shutil
from pathlib import Path

import numpy as np

PLASTIC = "examples/outputs/cmaeshan_flocking_hanmodel__b140e5f5_26_06_23-17_05_30-8-agents"
STATIC = "examples/outputs/cmaeshan_flocking_hanmodel__5de64e71_26_10_21-19_59_20"

CONDITIONS = [
    ("plastic", PLASTIC, "stationary", "push"),
    ("plastic", PLASTIC, "circle", "push"),
    ("plastic", PLASTIC, "stationary", "none"),
    ("plastic", PLASTIC, "circle", "none"),
    ("static", STATIC, "stationary", "push"),
    ("static", STATIC, "circle", "push"),
    ("static", STATIC, "stationary", "none"),
    ("static", STATIC, "circle", "none"),
]


def analyse(npz_path, dist_step, local_window=100):
    """Summarise one condition's per-step data.

    The reference level is the LOCAL pre-disturbance steady state (the
    last ``local_window`` steps before the disturbance), not the mean over
    the whole pre-window. The whole pre-window includes the warm-up
    phase (Fg climbs from ~2.2 to ~3.4 over the first ~800 steps), so
    using it as the reference would make "recovery" meaningless — Fg
    simply keeps climbing past it.
    """
    d = np.load(npz_path)
    Fg, At, Dt, Cg, S = d["Fg"], d["At"], d["Dt"], d["Cg"], d["S"]
    T = len(Fg)
    pre = slice(0, dist_step)
    post = slice(dist_step, T)
    local = slice(max(0, dist_step - local_window), dist_step)

    local_mean = Fg[local].mean()
    band = 0.05 * abs(local_mean)
    rec = None
    for i, v in enumerate(Fg[post]):
        if abs(v - local_mean) <= band:
            rec = i
            break
    return {
        "T": T,
        "local_pre_Fg": float(local_mean),
        "pre_Fg": float(Fg[pre].mean()),
        "post_Fg": float(Fg[post].mean()),
        "local_pre_At": float(At[local].mean()),
        "post_At": float(At[post].mean()),
        "local_pre_Dt": float(Dt[local].mean()),
        "post_Dt": float(Dt[post].mean()),
        "local_pre_Cg": float(Cg[local].mean()),
        "post_Cg": float(Cg[post].mean()),
        "local_pre_S": float(S[local].mean()),
        "post_S": float(S[post].mean()),
        "min_post_Fg": float(Fg[post].min()),
        "drop_local": float(local_mean - Fg[post].min()),
        "drop_mean": float(local_mean - Fg[post].mean()),
        "recovery_steps": rec,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshots", type=str, default="/tmp/push_matrix",
                   help="Directory holding the per-condition npz snapshots "
                        "(written by run_push_adaptation_matrix.sh after "
                        "each run, because the disturbance script always "
                        "overwrites <exp>/disturbance_eval_push/).")
    p.add_argument("--dist-step", type=int, default=800)
    args = p.parse_args()

    snaps = Path(args.snapshots)

    results = {}
    for model, _exp, motion, mode in CONDITIONS:
        tag = f"{model}_{motion}_{mode}"
        src = snaps / f"{tag}.npz"
        if not src.exists():
            print(f"  MISSING: {src}")
            continue
        results[tag] = analyse(src, args.dist_step)

    print()
    print("=" * 108)
    print("PUSH-ADAPTATION MATRIX  (push at step 800, radial-out, mag 0.5)")
    print("Reference = local pre-disturbance steady state (last 100 steps before the push)")
    print("=" * 108)
    hdr = (f"{'condition':<26} {'ref Fg':>7} {'post Fg':>8} {'min Fg':>8} "
           f"{'drop':>7} {'recov':>7} | {'ΔAt':>7} {'ΔDt':>7} {'ΔCg':>7} {'ΔS':>7}")
    print(hdr)
    print("-" * len(hdr))
    for model in ("plastic", "static"):
        for motion in ("stationary", "circle"):
            for mode in ("push", "none"):
                tag = f"{model}_{motion}_{mode}"
                if tag not in results:
                    continue
                r = results[tag]
                rec = r["recovery_steps"]
                rec_s = f"{rec}" if rec is not None else "never"
                print(f"{tag:<26} {r['local_pre_Fg']:>7.3f} {r['post_Fg']:>8.3f} "
                      f"{r['min_post_Fg']:>8.3f} {r['drop_local']:>+7.3f} {rec_s:>7} | "
                      f"{r['post_At']-r['local_pre_At']:>+7.3f} "
                      f"{r['post_Dt']-r['local_pre_Dt']:>+7.3f} "
                      f"{r['post_Cg']-r['local_pre_Cg']:>+7.3f} "
                      f"{r['post_S']-r['local_pre_S']:>+7.3f}")

    # Net effect of the push = (drop with push) - (drop without push).
    print()
    print("=" * 108)
    print("NET EFFECT OF THE PUSH  (drop with push  -  drop without push)")
    print("=" * 108)
    print(f"{'model / target':<26} {'Δdrop':>9} {'Δmin Fg':>9} {'Δrecovery':>11}")
    print("-" * 60)
    for model in ("plastic", "static"):
        for motion in ("stationary", "circle"):
            p_tag = f"{model}_{motion}_push"
            n_tag = f"{model}_{motion}_none"
            if p_tag not in results or n_tag not in results:
                continue
            rp, rn = results[p_tag], results[n_tag]
            rec_p = rp["recovery_steps"]
            rec_n = rn["recovery_steps"]
            rec_s = (f"{rec_p}" if rec_p is not None else "never")
            print(f"{model + ' / ' + motion:<26} "
                  f"{rp['drop_local'] - rn['drop_local']:>+9.3f} "
                  f"{rp['min_post_Fg'] - rn['min_post_Fg']:>+9.3f} "
                  f"{rec_s:>11}")
    print()
    print("(a positive Δdrop means the push caused extra degradation "
          "beyond run-to-run drift)")


if __name__ == "__main__":
    main()
