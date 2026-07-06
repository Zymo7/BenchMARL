"""End-to-end evaluation driver for the HAN-vs-static-MLP comparison.

This script orchestrates the full evaluation flow:

  1. For each (algorithm, scenario, n_agents) combination, run
     ``run_comparison_eval.py`` to compute ``F_orbit`` summary.json
     files.
  2. Then invoke ``plot_comparison.py`` to render the comparison
     figures (orbit scalability + frozen-agent adaptation).

Usage:
    /home/zhaozeming/miniconda3/envs/benchmarl/bin/python \\
        examples/run_eval.py \\
        --han-path outputs/cmaeshan_flocking_hanmodel__b140e5f5_26_06_23-17_05_30-8agents \\
        --static-mlp-path outputs/cmaesstaticmlp_flocking_staticmlpmodel__48952390_26_06_30-10_16_35 \\
        --n-agents 4 5 8 \\
        --scenarios orbit frozen_agent \\
        --max-steps 800 --disturbance-step 400 \\
        --out comparison_plots

If ``--han-path`` / ``--static-mlp-path`` are not provided, the
script auto-discovers the most recent matching experiment folder
under ``examples/outputs/`` (the naming convention used by
``run_cmaes_han_flocking_custom.py`` and
``run_cmaes_static_mlp_flocking_custom.py``).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


# Path constants used by subprocess invocations.
REPO_ROOT = Path(__file__).parent.parent          # .../BenchMARL
EXAMPLES_DIR = REPO_ROOT / "examples"
RUNNING_DIR = EXAMPLES_DIR / "running"
OUTPUTS_DIR = EXAMPLES_DIR / "outputs"


# Auto-discovery patterns matching the folder names produced by
# ``run_cmaes_han_flocking_custom.py`` and
# ``run_cmaes_static_mlp_flocking_custom.py``.
HAN_FOLDER_RE = re.compile(r"cmaeshan_flocking_hanmodel__")
SLM_FOLDER_RE = re.compile(r"cmaesstaticmlp_flocking_staticmlpmodel__")


def _find_latest_run(pattern: re.Pattern) -> Optional[Path]:
    """Return the most-recently-modified subfolder of OUTPUTS_DIR
    whose name matches ``pattern``, or ``None``."""
    if not OUTPUTS_DIR.exists():
        return None
    candidates = sorted(
        (p for p in OUTPUTS_DIR.iterdir()
         if p.is_dir() and pattern.search(p.name)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _abs_path(p: str) -> Path:
    """Resolve ``p`` relative to CWD; raise if it doesn't exist."""
    path = Path(p).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Pass an absolute path or run from "
            f"the directory containing the experiment folder."
        )
    return path


def _run(cmd: List[str], log_path: Optional[Path] = None) -> int:
    """Run a subprocess, optionally tee its output to a log file.

    Returns the process exit code. We use ``cwd=RUNNING_DIR`` because
    the eval and plot scripts depend on `from flocking_patch import
    ...` which lives in that directory.
    """
    print(f"  $ {' '.join(cmd)}")
    sys.stdout.flush()
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as logf:
            return subprocess.call(
                cmd,
                cwd=str(RUNNING_DIR),
                stdout=logf,
                stderr=subprocess.STDOUT,
            )
    return subprocess.call(cmd, cwd=str(RUNNING_DIR))


def main():
    ap = argparse.ArgumentParser(
        description="Run all HAN-vs-static-MLP evaluations and "
                    "generate comparison plots in one go."
    )
    ap.add_argument("--han-path", type=str, default=None,
                    help="Path to the HAN training folder. If omitted, "
                         "auto-discover the most recent "
                         "cmaeshan_* folder under outputs/.")
    ap.add_argument("--static-mlp-path", type=str, default=None,
                    help="Path to the static-MLP training folder. If "
                         "omitted, auto-discover the most recent "
                         "cmaesstaticmlp_* folder under outputs/.")
    ap.add_argument("--n-agents", type=int, nargs="+", default=[4],
                    help="Agent counts to evaluate at. Default: 4.")
    ap.add_argument("--scenarios", type=str, nargs="+",
                    default=["orbit", "frozen_agent"],
                    choices=["orbit", "frozen_agent"])
    ap.add_argument("--disturbance-modes", type=str, nargs="+",
                    default=["freeze", "push"],
                    choices=["freeze", "push"],
                    help="For frozen_agent scenario: which disturbance "
                         "modes to evaluate. Default: both.")
    ap.add_argument("--max-steps", type=int, default=800,
                    help="Episode horizon (must match training).")
    ap.add_argument("--disturbance-step", type=int, default=400,
                    help="For frozen_agent scenario only.")
    ap.add_argument("--push-magnitude", type=float, default=0.5,
                    help="Push impulse magnitude (push mode only).")
    ap.add_argument("--push-direction", type=str, default="radial-out",
                    choices=["radial-out", "fixed-x"])
    ap.add_argument("--num-episodes", type=int, default=3,
                    help="Number of eval episodes per (algo, scenario, n)")
    ap.add_argument("--han-hidden-size", type=int, default=10,
                    help="HAN hidden size used at training time.")
    ap.add_argument("--static-mlp-hidden-size", type=int, default=40,
                    help="Static-MLP hidden size used at training time.")
    ap.add_argument("--out", type=str, default="comparison_plots",
                    help="Output directory for plots.")
    ap.add_argument("--log-dir", type=str, default=None,
                    help="If set, write per-run subprocess logs into "
                         "this directory (useful when running many "
                         "configurations).")
    ap.add_argument("--skip-plot", action="store_true",
                    help="Skip the final plot step (just run evals).")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Resolve the two experiment folders (auto-discover if missing)
    # ------------------------------------------------------------------
    if args.han_path:
        han_path = _abs_path(args.han_path)
    else:
        han_path = _find_latest_run(HAN_FOLDER_RE)
        if han_path is None:
            print("ERROR: --han-path not given and no cmaeshan_* folder "
                  "found under outputs/.", file=sys.stderr)
            sys.exit(2)
        print(f"[auto-discover] HAN: {han_path}")

    if args.static_mlp_path:
        slm_path = _abs_path(args.static_mlp_path)
    else:
        slm_path = _find_latest_run(SLM_FOLDER_RE)
        if slm_path is None:
            print("ERROR: --static-mlp-path not given and no "
                  "cmaesstaticmlp_* folder found under outputs/.",
                  file=sys.stderr)
            sys.exit(2)
        print(f"[auto-discover] static-MLP: {slm_path}")

    print()
    print("=" * 70)
    print("HAN-vs-static-MLP evaluation driver")
    print("=" * 70)
    print(f"  HAN       : {han_path}")
    print(f"  static-MLP: {slm_path}")
    print(f"  n_agents  : {args.n_agents}")
    print(f"  scenarios : {args.scenarios}")
    print(f"  max_steps : {args.max_steps}")
    if "frozen_agent" in args.scenarios:
        print(f"  disturbance_step: {args.disturbance_step}")
        print(f"  disturbance_modes: {args.disturbance_modes}")
        print(f"  push_magnitude:   {args.push_magnitude}")
        print(f"  push_direction:   {args.push_direction}")
    print(f"  num_episodes per run: {args.num_episodes}")
    print()

    # ------------------------------------------------------------------
    # Run all evaluations
    # ------------------------------------------------------------------
    log_dir = Path(args.log_dir) if args.log_dir else None
    failures = []

    def _eval(algo: str, scenario: str, n: int,
              mode: str = "freeze") -> None:
        log_path = (log_dir / f"{algo}_{scenario}_{mode}_n{n}.log"
                    if log_dir else None)
        cmd = [
            sys.executable, "run_comparison_eval.py",
            "--algo", algo,
            "--max-steps", str(args.max_steps),
            "--n-agents", str(n),
            "--scenario", scenario,
            "--num-episodes", str(args.num_episodes),
            "--han-hidden-size", str(args.han_hidden_size),
            "--static-mlp-hidden-size", str(args.static_mlp_hidden_size),
        ]
        if algo == "han":
            cmd += ["--han-exp-path", str(han_path)]
        else:
            cmd += ["--static-mlp-exp-path", str(slm_path)]

        if scenario == "frozen_agent":
            cmd += ["--disturbance-step", str(args.disturbance_step),
                    "--disturbance-mode", mode,
                    "--push-magnitude", str(args.push_magnitude),
                    "--push-direction", args.push_direction]

        print(f"\n[{algo}/{scenario}/{mode}/n={n}] launching...")
        rc = _run(cmd, log_path=log_path)
        if rc != 0:
            failures.append((algo, scenario, mode, n, rc))
            print(f"  FAILED rc={rc} (log: {log_path})")
        else:
            print(f"  OK")

    for n in args.n_agents:
        for scenario in args.scenarios:
            if scenario == "frozen_agent":
                for mode in args.disturbance_modes:
                    for algo in ("han", "cmaes-static-mlp"):
                        _eval(algo, scenario, n, mode=mode)
            else:
                for algo in ("han", "cmaes-static-mlp"):
                    _eval(algo, scenario, n, mode="freeze")

    if failures:
        print("\n!!! Some evaluations failed:")
        for algo, scenario, mode, n, rc in failures:
            print(f"   {algo} / {scenario} / {mode} / n={n}  rc={rc}")
        if not args.skip_plot:
            print("Plotting anyway with whatever data is on disk.")
    else:
        print("\nAll evaluations completed successfully.")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    if args.skip_plot:
        print("Skipping plots (--skip-plot).")
        return

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = EXAMPLES_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_log = (log_dir / "plot.log" if log_dir else None)
    cmd = [
        sys.executable, "plot_comparison.py",
        "--han-path", str(han_path),
        "--static-mlp-path", str(slm_path),
        "--n-agents", *map(str, args.n_agents),
        "--scenario", "both" if len(args.scenarios) > 1 else args.scenarios[0],
        "--mode", ("both"
                   if (len(args.disturbance_modes) > 1
                       and "frozen_agent" in args.scenarios)
                   else (args.disturbance_modes[0]
                         if args.disturbance_modes else "freeze")),
        "--out", str(out_dir),
    ]
    print(f"\n[plot] launching: {' '.join(cmd)}")
    rc = _run(cmd, log_path=plot_log)
    if rc != 0:
        print(f"  plot FAILED rc={rc} (log: {plot_log})")
        sys.exit(rc)
    print(f"\nAll plots written to: {out_dir}")
    for f in sorted(out_dir.glob("*.png")):
        print(f"  {f}")


if __name__ == "__main__":
    main()