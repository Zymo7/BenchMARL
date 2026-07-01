"""Generate comparison plots for the HAN-vs-static-MLP experiment.

Reads the ``summary.json`` files written by ``run_comparison_eval.py``
from one or more experiment folders and produces grouped bar charts:

  1. F_orbit vs n_agents  (scalability)        — for the ``orbit`` scenario
  2. baseline vs full_post F_orbit             — for the ``frozen_agent`` scenario
                                                  (the Δ after freeze is the
                                                  headline adaptation metric)

Usage:
    python plot_comparison.py \\
        --han-path  <han exp folder> \\
        --static-mlp-path <static_mlp exp folder> \\
        --n-agents 4 5 8 \\
        --scenario both \\
        --out comparison_plots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _load_summaries(exp_path: Path, scenario: str,
                    n_agents_list: List[int]) -> Dict[int, dict]:
    """Load per-n_agents summary.json for one algorithm + scenario.

    Tries multiple naming conventions to stay backward-compatible with
    eval runs that pre-date the {algo}-suffix naming:
      - ``{scenario}_n{N}_{algo}``  (current, e.g. orbit_n4_cmaes-static-mlp)
      - ``{scenario}_n{N}``         (legacy, e.g. orbit_n4)
      - ``{scenario}_n{N}_han`` / ``_ippo`` / ``_static-mlp`` (older arms)
    """
    out = {}
    for n in n_agents_list:
        candidates = [
            exp_path / "comparison_eval" / f"{scenario}_n{n}" / "summary.json",
            (exp_path / "comparison_eval"
             / f"{scenario}_n{n}_ippo" / "summary.json"),
            (exp_path / "comparison_eval"
             / f"{scenario}_n{n}_static-mlp" / "summary.json"),
            (exp_path / "comparison_eval"
             / f"{scenario}_n{n}_cmaes-static-mlp" / "summary.json"),
            (exp_path / "comparison_eval"
             / f"{scenario}_n{n}_han" / "summary.json"),
        ]
        for p in candidates:
            if p.exists():
                with open(p) as f:
                    out[n] = json.load(f)
                break
    return out


def _aggregate_episode_metric(summary: dict, phase: str = "full") \
        -> Optional[tuple]:
    """Return (mean over episodes, std over episodes) of a phase's mean."""
    eps = summary.get("per_episode", [])
    vals = [ep.get(phase, {}).get("mean") for ep in eps]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    arr = np.array(vals)
    return float(arr.mean()), float(arr.std())


def _bar(ax, x_positions, means, stds, label, color):
    ax.bar(x_positions, means, yerr=stds, width=0.35, label=label,
           color=color, alpha=0.85, capsize=4, edgecolor="black",
           linewidth=0.5)


def plot_orbit_scalability(han: Dict[int, dict], baseline: Dict[int, dict],
                           out_path: Path, baseline_label: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = sorted(set(list(han.keys()) + list(baseline.keys())))
    fig, ax = plt.subplots(figsize=(8, 5))

    han_means, han_stds = [], []
    base_means, base_stds = [], []
    for n in ns:
        h = _aggregate_episode_metric(han[n], "full") if n in han else None
        b = (_aggregate_episode_metric(baseline[n], "full")
             if n in baseline else None)
        han_means.append(h[0] if h else 0.0)
        han_stds.append(h[1] if h else 0.0)
        base_means.append(b[0] if b else 0.0)
        base_stds.append(b[1] if b else 0.0)

    x = np.arange(len(ns))
    _bar(ax, x - 0.18, han_means, han_stds, "HAN (CMA-ES)", "C0")
    _bar(ax, x + 0.18, base_means, base_stds, baseline_label, "C3")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} agents" for n in ns])
    ax.set_ylabel("F_orbit (scheme E, higher = better)")
    ax.set_title(
        f"Flocking-orbit scalability: HAN vs {baseline_label}\n"
        "(trained on 4 agents; evaluated on N)"
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for xi, m in zip(x - 0.18, han_means):
        if m > 0:
            ax.text(xi, m + 0.03, f"{m:.2f}", ha="center", fontsize=9)
    for xi, m in zip(x + 0.18, base_means):
        if m > 0:
            ax.text(xi, m + 0.03, f"{m:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_frozen_agent_adaptation(han: Dict[int, dict], baseline: Dict[int, dict],
                                 out_path: Path, baseline_label: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = sorted(set(list(han.keys()) + list(baseline.keys())))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    han_base, han_post = [], []
    base_base, base_post = [], []
    for n in ns:
        hb = (_aggregate_episode_metric(han[n], "baseline")
              if n in han else None)
        hp = (_aggregate_episode_metric(han[n], "full_post")
              if n in han else None)
        bb = (_aggregate_episode_metric(baseline[n], "baseline")
              if n in baseline else None)
        bp = (_aggregate_episode_metric(baseline[n], "full_post")
              if n in baseline else None)
        han_base.append(hb[0] if hb else 0.0)
        han_post.append(hp[0] if hp else 0.0)
        base_base.append(bb[0] if bb else 0.0)
        base_post.append(bp[0] if bp else 0.0)

    x = np.arange(len(ns))
    w = 0.2
    ax.bar(x - 1.5 * w, han_base, width=w, label="HAN baseline", color="C0")
    ax.bar(x - 0.5 * w, han_post, width=w, label="HAN after freeze",
           color="C0", alpha=0.5)
    ax.bar(x + 0.5 * w, base_base, width=w,
           label=f"{baseline_label} baseline", color="C3")
    ax.bar(x + 1.5 * w, base_post, width=w,
           label=f"{baseline_label} after freeze", color="C3", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} agents" for n in ns])
    ax.set_ylabel("F_orbit")
    ax.set_title("Frozen-agent disturbance: baseline vs after-freeze")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    han_delta = [p - b for b, p in zip(han_base, han_post)]
    base_delta = [p - b for b, p in zip(base_base, base_post)]
    _bar(ax, x - 0.18, han_delta, [0] * len(ns), "HAN Δ", "C0")
    _bar(ax, x + 0.18, base_delta, [0] * len(ns),
         f"{baseline_label} Δ", "C3")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} agents" for n in ns])
    ax.set_ylabel("Δ F_orbit (after freeze − baseline)")
    ax.set_title("Adaptation robustness (Δ closer to 0 = more robust)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for xi, d in zip(x - 0.18, han_delta):
        ax.text(xi, d + (0.02 if d >= 0 else -0.04), f"{d:+.2f}",
                ha="center", fontsize=9)
    for xi, d in zip(x + 0.18, base_delta):
        ax.text(xi, d + (0.02 if d >= 0 else -0.04), f"{d:+.2f}",
                ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    p = argparse.ArgumentParser(
        description="Plot HAN-vs-static-MLP comparison from "
                    "eval summary.json files."
    )
    p.add_argument("--han-path", type=str, required=True)
    p.add_argument("--static-mlp-path", type=str, required=True,
                   help="Path to the CMA-ES static-MLP training folder "
                        "(the script's --static-mlp-exp-path counterpart).")
    p.add_argument("--baseline-label", type=str, default="static-MLP (CMA-ES)",
                   help="Legend label for the baseline bars.")
    p.add_argument("--n-agents", type=int, nargs="+", default=[4, 5, 8])
    p.add_argument("--scenario", type=str, default="both",
                   choices=["orbit", "frozen_agent", "both"])
    p.add_argument("--out", type=str, default="comparison_plots")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    han_path = Path(args.han_path)
    slm_path = Path(args.static_mlp_path)

    if args.scenario in ("orbit", "both"):
        han = _load_summaries(han_path, "orbit", args.n_agents)
        slm = _load_summaries(slm_path, "orbit", args.n_agents)
        if not han and not slm:
            print("  [orbit] No summary.json found — run "
                  "run_comparison_eval.py --scenario orbit first.")
        else:
            print(f"[orbit] HAN ns={sorted(han)}, "
                  f"static-MLP ns={sorted(slm)}")
            plot_orbit_scalability(
                han, slm, out_dir / "orbit_scalability.png",
                baseline_label=args.baseline_label,
            )

    if args.scenario in ("frozen_agent", "both"):
        han = _load_summaries(han_path, "frozen_agent", args.n_agents)
        slm = _load_summaries(slm_path, "frozen_agent", args.n_agents)
        if not han and not slm:
            print("  [frozen_agent] No summary.json found — run "
                  "run_comparison_eval.py --scenario frozen_agent first.")
        else:
            print(f"[frozen] HAN ns={sorted(han)}, "
                  f"static-MLP ns={sorted(slm)}")
            plot_frozen_agent_adaptation(
                han, slm, out_dir / "frozen_agent_adaptation.png",
                baseline_label=args.baseline_label,
            )

    print(f"\nAll plots in: {out_dir}")


if __name__ == "__main__":
    main()