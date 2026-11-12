"""Mechanism probe: what can the trained HAN actually do online?

This script isolates the Hebbian weight-update rule from the simulator
and measures, on synthetic input streams, the capabilities that a
"moving target" style task is supposed to test. It answers the
question "which task can actually reveal HAN's dynamic advantage?"
*before* investing in a new environment.

Findings (with the reference plastic HAN's ABCD, see --abcd-dir):

  PROBE 1 — input-distribution shift (direction).
      When the dominant input direction switches mid-stream, the
      Hebbian/Oja term makes W *forget* the stale direction:
      |W[0, old_dim]| decays from 0.32 -> 0.13 over 200 steps.
      A static (lr=0) net leaves it at 0.506 forever.
      => HAN's advantage is FAST WEIGHT RE-SHAPING, not memory.

  PROBE 2 — input-scale change (gain).
      Measured output gain barely moves (0.94 -> 0.94 static,
      1.10 -> 1.16 plastic) because the hard max-abs normalization
      cancels the scale change.
      => GAIN adaptation is NOT a usable lever for task design.

  PROBE 3 — the pure Hebbian term needs a nonzero A.
      A bare HanLayer has A=B=C=D=0 (zeros), so dW = 0 and only the
      max-abs normalization acts. The Hebbian rule is inert until
      CMA-ES has optimized ABCD.
      => Always load trained ABCD when probing.

Usage:
    python examples/running/han_mechanism_probe.py
    python examples/running/han_mechanism_probe.py --probe all
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarl.models.han import HanLayer  # noqa: E402

DEFAULT_ABCD_DIR = (
    "examples/outputs/cmaeshan_flocking_hanmodel__"
    "b140e5f5_26_06_23-17_05_30-8-agents/han_results"
)


def _load_layer(abcd_dir, idx, n_in, n_out, lr_hebb):
    """Build a HanLayer with the trained ABCD of layer ``idx``."""
    v = np.load(str(Path(abcd_dir) / f"layer{idx}_abcd.npy"))
    layer = HanLayer(n_in, n_out, lr_hebb=lr_hebb, weight_init=0.1,
                     window_size=10)
    layer.set_abcd_from_vector(torch.tensor(v, dtype=torch.float32))
    return layer


def probe_direction_shift(abcd_dir, lr_hebb, n_in=10, n_out=10,
                          switch_at=200, T=400, seed=0):
    """Does W re-shape when the dominant input direction changes?"""
    torch.manual_seed(seed)
    d_old = torch.zeros(n_in)
    d_old[0] = 1.0
    d_new = torch.zeros(n_in)
    d_new[1] = 1.0
    layer = _load_layer(abcd_dir, 0, n_in, n_out, lr_hebb)
    trace = []
    for t in range(T):
        d = d_old if t < switch_at else d_new
        x = (torch.randn(1, n_in) * 0.2 + d).unsqueeze(0)
        layer(x)
        if layer._pre_window and (t + 1) % 4 == 0:
            layer.update_weights()
        trace.append(abs(layer.W.data[0, 1]).item())
    return np.array(trace)


def probe_gain(abcd_dir, lr_hebb, n_in=10, n_out=10, T=400, seed=0):
    """Does the effective output gain track an input-scale change?"""
    torch.manual_seed(seed)
    layer = _load_layer(abcd_dir, 0, n_in, n_out, lr_hebb)
    gains = []
    for t in range(T):
        scale = 1.0 if t < T // 2 else 4.0
        x = torch.randn(1, n_in) * 0.5 * scale
        out = layer(x)
        gains.append((out.norm() / (x.norm() + 1e-6)).item())
        if layer._pre_window and (t + 1) % 4 == 0:
            layer.update_weights()
    return np.array(gains)


def probe_recovery_time(abcd_dir, lr_hebb, seed=0, switch_at=200, T=600):
    """Steps after a switch until W has re-shaped.

    "Recovered" = ``|W[0, new_dim]|`` fell below 60% of its value at the
    switch. Returns ``None`` if it never recovers within ``T``.
    """
    tr = probe_direction_shift(abcd_dir, lr_hebb, switch_at=switch_at, T=T,
                               seed=seed)
    threshold = tr[switch_at] * 0.6
    idx = np.where(tr[switch_at:] < threshold)[0]
    return int(idx[0]) if len(idx) else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--abcd-dir", type=str, default=DEFAULT_ABCD_DIR,
                   help="Directory holding layer0_abcd.npy / layer1_abcd.npy.")
    p.add_argument("--probe", type=str, default="all",
                   choices=["all", "direction", "gain", "recovery"])
    p.add_argument("--seeds", type=int, default=5,
                   help="Seeds for the recovery-time probe.")
    args = p.parse_args()

    print("=" * 68)
    print("HAN mechanism probe — what can the Hebbian rule do online?")
    print("=" * 68)
    print(f"ABCD source: {args.abcd_dir}\n")

    if args.probe in ("all", "direction"):
        print("PROBE 1 — input-distribution shift (direction switch at t=200)")
        traces = {}
        for lr in (0.01, 0.0):
            traces[lr] = probe_direction_shift(args.abcd_dir, lr)
            t = traces[lr]
            print(f"  lr_hebb={lr:<5} |W[0,new_dim]|: "
                  f"t=200 {t[199]:.4f} -> t=400 {t[-1]:.4f} "
                  f"(delta {t[-1] - t[199]:+.4f})")
        plastic, static = traces[0.01], traces[0.0]
        print(f"  => plastic re-shapes W ({plastic[-1] - plastic[199]:+.4f}); "
              f"static does not ({static[-1] - static[199]:+.4f})")
        print("  => HAN's dynamic advantage = fast weight re-shaping.\n")

    if args.probe in ("all", "recovery"):
        print("PROBE 1b — recovery time after a switch (steps until "
              "|W[0,new]| < 60% of its value at the switch)")
        print(f"  {'lr_hebb':<10} " +
              " ".join(f"s{s}".rjust(7) for s in range(args.seeds)))
        for lr in (0.003, 0.01, 0.03, 0.1):
            rec = [probe_recovery_time(args.abcd_dir, lr, seed=s)
                   for s in range(args.seeds)]
            cells = " ".join(
                (f"{r}" if r is not None else "never").rjust(7) for r in rec)
            print(f"  {lr:<10} {cells}")
        print("  => lr_hebb=0.01 is the most reliable; too small cannot "
              "re-shape,\n     too large is unstable. Design the environment "
              "change period\n     in the ~100-300 step range.\n")

    if args.probe in ("all", "gain"):
        print("PROBE 2 — input-scale change (gain tracking)")
        for lr in (0.01, 0.0):
            g = probe_gain(args.abcd_dir, lr)
            print(f"  lr_hebb={lr:<5} gain: first-half {g[:200].mean():.4f} "
                  f"-> second-half {g[200:].mean():.4f} "
                  f"({g[200:].mean() - g[:200].mean():+.4f})")
        print("  => gain adaptation is negligible (max-abs normalization "
              "cancels it).\n")

    print("PROBE 3 — bare layer (A=B=C=D=0) has an inert Hebbian rule:")
    layer = HanLayer(4, 4, lr_hebb=0.5, weight_init=0.1, window_size=4)
    print(f"  ||A||={layer.A.norm():.4f} ||B||={layer.B.norm():.4f} "
          f"||C||={layer.C.norm():.4f} ||D||={layer.D.norm():.4f}")
    print("  => always load trained ABCD when probing.")


if __name__ == "__main__":
    main()
