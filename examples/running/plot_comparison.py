"""Generate comparison plots for the HAN-vs-static-MLP experiment.

Reads the ``summary.json`` files written by ``run_comparison_eval.py``
from one or more experiment folders and produces:

  1. ``orbit_scalability.png``  : F_orbit vs n_agents (legacy, kept).
  2. ``frozen_agent_adaptation.png`` : baseline vs full_post + Δ bars
     (legacy, kept).
  3. ``t_stable_<mode>.png``    : T_stable and T_recover grouped bars
     (HAN vs static-MLP × N). One file per disturbance mode
     (``freeze`` / ``push``).
  4. ``f_quality_<mode>.png``   : stable_quality and after_quality
     grouped bars. One per mode.
  5. ``error_vs_N.png``         : line plot, x=N, y=1-stable_quality.
  6. ``t_stable_overlay.png``   : line plot of T_stable over N.

Legacy summaries (without ``_freeze``/``_push`` suffix) are loaded as
fallback, and the legacy plots keep working.

Usage:
    python plot_comparison.py \\
        --han-path  <han exp folder> \\
        --static-mlp-path <static_mlp exp folder> \\
        --n-agents 4 6 8 10 12 \\
        --scenario both \\
        --mode both \\
        --out comparison_plots
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ============================================================================
# Summary loaders
# ============================================================================
def _load_summaries(
    exp_path: Path, scenario: str, n_agents_list: List[int],
    mode: Optional[str] = None,
) -> Dict[int, dict]:
    """Load per-n_agents summary.json for one algorithm + scenario.

    When ``mode`` is ``freeze`` or ``push``, prefer the per-mode sub-folder
    (e.g. ``frozen_agent_n4_han_freeze``); fall back to the legacy name
    (``frozen_agent_n4_han``) for backward compat.

    For scenario ``orbit`` the ``mode`` arg is ignored (orbit has no
    disturbance).
    """
    out: Dict[int, dict] = {}
    for n in n_agents_list:
        candidates: List[Path] = []
        if scenario == "frozen_agent" and mode in ("freeze", "push"):
            candidates += [
                (exp_path / "comparison_eval"
                 / f"frozen_agent_n{n}_{algo}_{mode}" / "summary.json")
                for algo in ("han", "cmaes-static-mlp", "ippo", "static-mlp")
            ]
        if scenario == "frozen_agent":
            candidates += [
                exp_path / "comparison_eval" / f"frozen_agent_n{n}" / "summary.json",
                (exp_path / "comparison_eval"
                 / f"frozen_agent_n{n}_ippo" / "summary.json"),
                (exp_path / "comparison_eval"
                 / f"frozen_agent_n{n}_static-mlp" / "summary.json"),
                (exp_path / "comparison_eval"
                 / f"frozen_agent_n{n}_cmaes-static-mlp" / "summary.json"),
                (exp_path / "comparison_eval"
                 / f"frozen_agent_n{n}_han" / "summary.json"),
            ]
        else:  # orbit
            candidates += [
                exp_path / "comparison_eval" / f"orbit_n{n}" / "summary.json",
                (exp_path / "comparison_eval"
                 / f"orbit_n{n}_han" / "summary.json"),
                (exp_path / "comparison_eval"
                 / f"orbit_n{n}_cmaes-static-mlp" / "summary.json"),
            ]
        for p in candidates:
            if p.exists():
                with open(p) as f:
                    out[n] = json.load(f)
                break
    return out


def _aggregate_phase_metric(summary: dict, phase: str) -> Optional[tuple]:
    """Return (mean, std) of a phase's mean over per-episode records."""
    eps = summary.get("per_episode", [])
    vals = [ep.get(phase, {}).get("mean") for ep in eps]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    arr = np.array(vals)
    return float(arr.mean()), float(arr.std())


def _aggregate_scalar(summary: dict, key: str) -> Optional[tuple]:
    """Return (mean, std) of a scalar per-episode key."""
    eps = summary.get("per_episode", [])
    vals = [ep.get(key) for ep in eps]
    vals = [v for v in vals
            if v is not None
            and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None
    arr = np.array(vals, dtype=float)
    return float(arr.mean()), float(arr.std())


# ============================================================================
# Plotting helpers
# ============================================================================
def _bar(ax, x_positions, means, stds, label, color):
    ax.bar(x_positions, means, yerr=stds, width=0.35, label=label,
           color=color, alpha=0.85, capsize=4, edgecolor="black",
           linewidth=0.5)


def _hatched_bar(ax, x, height, label, color, *, fraction=0.06):
    """Empty hatched bar for 'never stable' / 'never recovered' cases.

    Renders as a small `fraction` of the panel height at the bottom,
    so it remains visible regardless of the y-axis range.
    """
    y = height * fraction
    ax.bar([x], [y], width=0.35, label=label,
           color=color, alpha=0.15, edgecolor=color, hatch="//",
           linewidth=1.0)
    ax.text(x, y, "—", ha="center", va="bottom",
            fontsize=10, color=color, fontweight="bold")


# ============================================================================
# Legacy plots (unchanged signatures, kept for backward compat)
# ============================================================================
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
        h = _aggregate_phase_metric(han[n], "full") if n in han else None
        b = (_aggregate_phase_metric(baseline[n], "full")
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
        hb = (_aggregate_phase_metric(han[n], "baseline")
              if n in han else None)
        hp = (_aggregate_phase_metric(han[n], "full_post")
              if n in han else None)
        bb = (_aggregate_phase_metric(baseline[n], "baseline")
              if n in baseline else None)
        bp = (_aggregate_phase_metric(baseline[n], "full_post")
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


# ============================================================================
# New geometric-metric plots
# ============================================================================
def _collect_steps(han: Dict[int, dict], baseline: Dict[int, dict],
                   ns: List[int]) -> List[int]:
    """Best-effort guess at T_max from a per-episode record."""
    for n in ns:
        s = han.get(n) or baseline.get(n)
        if s and "max_steps" in s:
            return [s["max_steps"]] * len(ns)
    return [800] * len(ns)


def plot_t_stable(han: Dict[int, dict], baseline: Dict[int, dict],
                  out_path: Path, baseline_label: str,
                  mode_label: str, *, with_recover: bool = True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = sorted(set(list(han.keys()) + list(baseline.keys())))
    if not ns:
        print(f"  [skip] {out_path.name}: no data")
        return
    T_maxs = _collect_steps(han, baseline, ns)

    han_t_stable, han_t_recover = [], []
    base_t_stable, base_t_recover = [], []
    for i, n in enumerate(ns):
        T_max = T_maxs[i]
        h_ts = _aggregate_scalar(han[n], "T_stable") if n in han else None
        b_ts = (_aggregate_scalar(baseline[n], "T_stable")
                if n in baseline else None)
        h_tr = _aggregate_scalar(han[n], "T_recover") if n in han else None
        b_tr = (_aggregate_scalar(baseline[n], "T_recover")
                if n in baseline else None)
        han_t_stable.append((h_ts[0] if h_ts else T_max, h_ts[1] if h_ts else 0.0))
        base_t_stable.append((b_ts[0] if b_ts else T_max,
                              b_ts[1] if b_ts else 0.0))
        if with_recover:
            han_t_recover.append((h_tr[0] if h_tr else T_max + 1,
                                  h_tr[1] if h_tr else 0.0))
            base_t_recover.append((b_tr[0] if b_tr else T_max + 1,
                                   b_tr[1] if b_tr else 0.0))

    x = np.arange(len(ns))
    if with_recover:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 5))
        axes = [axes]

    # --- Left: T_stable ---
    ax = axes[0]
    han_means = [v[0] for v in han_t_stable]
    han_stds = [v[1] for v in han_t_stable]
    base_means = [v[0] for v in base_t_stable]
    base_stds = [v[1] for v in base_t_stable]
    for xi, m, s, n in zip(x - 0.18, han_means, han_stds, ns):
        T_max = T_maxs[list(ns).index(n)]
        if m >= T_max:
            _hatched_bar(ax, xi, T_max, "HAN (never)", "C0")
        else:
            ax.bar([xi], [m], width=0.35, yerr=[s], color="C0",
                   alpha=0.85, edgecolor="black", linewidth=0.5,
                   capsize=4, label="HAN (CMA-ES)")
            ax.text(xi, m + T_max * 0.02, f"{m:.0f}", ha="center",
                    fontsize=8)
    for xi, m, s, n in zip(x + 0.18, base_means, base_stds, ns):
        T_max = T_maxs[list(ns).index(n)]
        if m >= T_max:
            _hatched_bar(ax, xi, T_max, f"{baseline_label} (never)", "C3")
        else:
            ax.bar([xi], [m], width=0.35, yerr=[s], color="C3",
                   alpha=0.85, edgecolor="black", linewidth=0.5,
                   capsize=4, label=baseline_label)
            ax.text(xi, m + T_max * 0.02, f"{m:.0f}", ha="center",
                    fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}" for n in ns])
    ax.set_xlabel("n_agents")
    ax.set_ylabel("T_stable (steps)")
    ax.set_title(f"T_stable — {mode_label} (lower = faster convergence)")
    ax.set_ylim(0, max(T_maxs) * 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    handles = []
    seen = set()
    for h in ax.containers:
        lbl = h.get_label()
        if lbl and lbl not in seen:
            seen.add(lbl)
            handles.append(h)
    if handles:
        ax.legend(handles=handles[:2], fontsize=9)

    # --- Right: T_recover ---
    if with_recover:
        ax = axes[1]
        han_means = [v[0] for v in han_t_recover]
        han_stds = [v[1] for v in han_t_recover]
        base_means = [v[0] for v in base_t_recover]
        base_stds = [v[1] for v in base_t_recover]
        for xi, m, s, n in zip(x - 0.18, han_means, han_stds, ns):
            T_max = T_maxs[list(ns).index(n)]
            if m > T_max:
                _hatched_bar(ax, xi, T_max, "HAN (never)", "C0")
            else:
                ax.bar([xi], [m], width=0.35, yerr=[s], color="C0",
                       alpha=0.85, edgecolor="black", linewidth=0.5,
                       capsize=4, label="HAN (CMA-ES)")
                ax.text(xi, m + T_max * 0.02, f"{m:.0f}", ha="center",
                        fontsize=8)
        for xi, m, s, n in zip(x + 0.18, base_means, base_stds, ns):
            T_max = T_maxs[list(ns).index(n)]
            if m > T_max:
                _hatched_bar(ax, xi, T_max, f"{baseline_label} (never)", "C3")
            else:
                ax.bar([xi], [m], width=0.35, yerr=[s], color="C3",
                       alpha=0.85, edgecolor="black", linewidth=0.5,
                       capsize=4, label=baseline_label)
                ax.text(xi, m + T_max * 0.02, f"{m:.0f}", ha="center",
                        fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}" for n in ns])
        ax.set_xlabel("n_agents")
        ax.set_ylabel("T_recover (steps from disturbance)")
        ax.set_title(f"T_recover — {mode_label} (lower = faster recovery)")
        ax.set_ylim(0, max(T_maxs) * 1.1)
        ax.grid(True, axis="y", alpha=0.3)
        handles = []
        seen = set()
        for h in ax.containers:
            lbl = h.get_label()
            if lbl and lbl not in seen:
                seen.add(lbl)
                handles.append(h)
        if handles:
            ax.legend(handles=handles[:2], fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_f_quality(han: Dict[int, dict], baseline: Dict[int, dict],
                   out_path: Path, baseline_label: str,
                   mode_label: str, *, with_after: bool = True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = sorted(set(list(han.keys()) + list(baseline.keys())))
    if not ns:
        print(f"  [skip] {out_path.name}: no data")
        return

    han_stable, han_after = [], []
    base_stable, base_after = [], []
    for n in ns:
        h_s = _aggregate_scalar(han[n], "stable_quality") if n in han else None
        b_s = (_aggregate_scalar(baseline[n], "stable_quality")
               if n in baseline else None)
        h_a = _aggregate_scalar(han[n], "after_quality") if n in han else None
        b_a = (_aggregate_scalar(baseline[n], "after_quality")
               if n in baseline else None)
        han_stable.append(h_s if h_s else (float("nan"), 0.0))
        base_stable.append(b_s if b_s else (float("nan"), 0.0))
        if with_after:
            han_after.append(h_a if h_a else (float("nan"), 0.0))
            base_after.append(b_a if b_a else (float("nan"), 0.0))

    x = np.arange(len(ns))
    if with_after:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 5))
        axes = [axes]

    # --- Left: stable_quality ---
    ax = axes[0]
    for xi, (m, s) in zip(x - 0.18, han_stable):
        if not math.isnan(m):
            ax.bar([xi], [m], width=0.35, yerr=[s], color="C0",
                   alpha=0.85, edgecolor="black", linewidth=0.5,
                   capsize=4, label="HAN (CMA-ES)")
            ax.text(xi, min(m + 0.03, 1.0), f"{m:.2f}", ha="center",
                    fontsize=8)
    for xi, (m, s) in zip(x + 0.18, base_stable):
        if not math.isnan(m):
            ax.bar([xi], [m], width=0.35, yerr=[s], color="C3",
                   alpha=0.85, edgecolor="black", linewidth=0.5,
                   capsize=4, label=baseline_label)
            ax.text(xi, min(m + 0.03, 1.0), f"{m:.2f}", ha="center",
                    fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}" for n in ns])
    ax.set_xlabel("n_agents")
    ax.set_ylabel("F_quality_stable (higher = better ring)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Stable-window ring quality — {mode_label}")
    ax.grid(True, axis="y", alpha=0.3)
    handles = []
    seen = set()
    for h in ax.containers:
        lbl = h.get_label()
        if lbl and lbl not in seen:
            seen.add(lbl)
            handles.append(h)
    if handles:
        ax.legend(handles=handles[:2], fontsize=9)

    if with_after:
        ax = axes[1]
        for xi, (m, s) in zip(x - 0.18, han_after):
            if not math.isnan(m):
                ax.bar([xi], [m], width=0.35, yerr=[s], color="C0",
                       alpha=0.85, edgecolor="black", linewidth=0.5,
                       capsize=4, label="HAN (CMA-ES)")
                ax.text(xi, min(m + 0.03, 1.0), f"{m:.2f}", ha="center",
                        fontsize=8)
        for xi, (m, s) in zip(x + 0.18, base_after):
            if not math.isnan(m):
                ax.bar([xi], [m], width=0.35, yerr=[s], color="C3",
                       alpha=0.85, edgecolor="black", linewidth=0.5,
                       capsize=4, label=baseline_label)
                ax.text(xi, min(m + 0.03, 1.0), f"{m:.2f}", ha="center",
                        fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}" for n in ns])
        ax.set_xlabel("n_agents")
        ax.set_ylabel("F_quality_after (higher = better recovery)")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"After-recovery ring quality — {mode_label}")
        ax.grid(True, axis="y", alpha=0.3)
        handles = []
        seen = set()
        for h in ax.containers:
            lbl = h.get_label()
            if lbl and lbl not in seen:
                seen.add(lbl)
                handles.append(h)
        if handles:
            ax.legend(handles=handles[:2], fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_error_vs_N(han: Dict[int, dict], baseline: Dict[int, dict],
                    out_path: Path, baseline_label: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = sorted(set(list(han.keys()) + list(baseline.keys())))
    if not ns:
        print(f"  [skip] {out_path.name}: no data")
        return

    han_err_mean, han_err_std = [], []
    base_err_mean, base_err_std = [], []
    for n in ns:
        h = _aggregate_scalar(han[n], "stable_quality") if n in han else None
        b = (_aggregate_scalar(baseline[n], "stable_quality")
             if n in baseline else None)
        # Error = 1 - stable_quality. If never stable, treat as error = 1.
        if h:
            han_err_mean.append(1.0 - h[0] if not math.isnan(h[0]) else 1.0)
            han_err_std.append(h[1] if not math.isnan(h[1]) else 0.0)
        else:
            han_err_mean.append(1.0)
            han_err_std.append(0.0)
        if b:
            base_err_mean.append(1.0 - b[0] if not math.isnan(b[0]) else 1.0)
            base_err_std.append(b[1] if not math.isnan(b[1]) else 0.0)
        else:
            base_err_mean.append(1.0)
            base_err_std.append(0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.array(ns)
    han = np.array(han_err_mean); han_s = np.array(han_err_std)
    base = np.array(base_err_mean); base_s = np.array(base_err_std)
    ax.plot(x, han, "o-", color="C0", label="HAN (CMA-ES)", linewidth=2)
    ax.fill_between(x, han - han_s, han + han_s, color="C0", alpha=0.2)
    ax.plot(x, base, "s-", color="C3", label=baseline_label, linewidth=2)
    ax.fill_between(x, base - base_s, base + base_s, color="C3", alpha=0.2)
    ax.set_xlabel("n_agents")
    ax.set_ylabel("Error = 1 − F_quality_stable (lower = better)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        "Orbit quality vs swarm size\n"
        "(flatter curve = more N-robust; HAN is expected flatter)"
    )
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    p = argparse.ArgumentParser(
        description="Plot HAN-vs-static-MLP comparison from "
                    "eval summary.json files."
    )
    p.add_argument("--han-path", type=str, required=True)
    p.add_argument("--static-mlp-path", type=str, required=True)
    p.add_argument("--baseline-label", type=str, default="static-MLP (CMA-ES)")
    p.add_argument("--n-agents", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    p.add_argument("--scenario", type=str, default="both",
                   choices=["orbit", "frozen_agent", "both"])
    p.add_argument("--mode", type=str, default="both",
                   choices=["freeze", "push", "both"],
                   help="Disturbance mode to plot for frozen_agent "
                        "scenario. Ignored for orbit.")
    p.add_argument("--out", type=str, default="comparison_plots")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    han_path = Path(args.han_path)
    slm_path = Path(args.static_mlp_path)

    if args.scenario in ("orbit", "both"):
        han = _load_summaries(han_path, "orbit", args.n_agents, mode=None)
        slm = _load_summaries(slm_path, "orbit", args.n_agents, mode=None)
        if not han and not slm:
            print("  [orbit] No summary.json found.")
        else:
            print(f"[orbit] HAN ns={sorted(han)}, "
                  f"static-MLP ns={sorted(slm)}")
            plot_orbit_scalability(
                han, slm, out_dir / "orbit_scalability.png",
                baseline_label=args.baseline_label,
            )
            plot_error_vs_N(
                han, slm, out_dir / "error_vs_N.png",
                baseline_label=args.baseline_label,
            )

    if args.scenario in ("frozen_agent", "both"):
        modes = (["freeze", "push"] if args.mode == "both"
                 else [args.mode])
        for mode in modes:
            han = _load_summaries(
                han_path, "frozen_agent", args.n_agents, mode=mode)
            slm = _load_summaries(
                slm_path, "frozen_agent", args.n_agents, mode=mode)
            if not han and not slm:
                print(f"  [{mode}] No summary.json found — skipping.")
                continue
            print(f"[{mode}] HAN ns={sorted(han)}, "
                  f"static-MLP ns={sorted(slm)}")
            plot_t_stable(
                han, slm,
                out_dir / f"t_stable_{mode}.png",
                baseline_label=args.baseline_label,
                mode_label=mode,
            )
            plot_f_quality(
                han, slm,
                out_dir / f"f_quality_{mode}.png",
                baseline_label=args.baseline_label,
                mode_label=mode,
            )
            # Legacy adaptation plot only for freeze (for backward compat).
            if mode == "freeze":
                # Also try the legacy sub-folder name (no _freeze suffix).
                han_legacy = _load_summaries(
                    han_path, "frozen_agent", args.n_agents, mode=None)
                slm_legacy = _load_summaries(
                    slm_path, "frozen_agent", args.n_agents, mode=None)
                if han_legacy or slm_legacy:
                    plot_frozen_agent_adaptation(
                        han_legacy, slm_legacy,
                        out_dir / "frozen_agent_adaptation.png",
                        baseline_label=args.baseline_label,
                    )

    print(f"\nAll plots in: {out_dir}")


if __name__ == "__main__":
    main()
