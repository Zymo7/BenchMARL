"""Unified evaluation script for the HAN-vs-static-MLP comparison.

Loads a trained model (HAN or CMA-ES static-MLP), runs evaluation
episodes under one of two scenarios:

  - ``--scenario orbit``         : plain flocking_orbit, no disturbance.
  - ``--scenario frozen_agent``  : mid-episode freeze of one agent.

The evaluation fitness is always ``_compute_flocking_orbit_fitness``
(the 方案 E formula in ``CmaesHanOptimizer``), regardless of which
algorithm produced the policy. This lets us compare HAN and static-MLP
head-to-head on the *same* metric.

Supported policies
------------------
1. ``--algo han``  : CMA-ES-trained HAN. Provide ``--han-exp-path``
   pointing to the experiment folder that contains ``han_results/``.
2. ``--algo cmaes-static-mlp`` : CMA-ES-trained static-MLP. Provide
   ``--static-mlp-exp-path`` pointing to the experiment folder that
   contains ``static_mlp_results/``.

Scalability
-----------
``--n-agents`` overrides the trained n_agents at eval time, exercising the
zero-shot transfer of each policy to a different swarm size (4 → 5 → 8).

Outputs (under ``<algo-exp-path>/comparison_eval/<scenario>_n<NN>/``)
  - per_step_data.npz : raw per-step arrays
  - summary.json      : mean fitness + phase-wise breakdown
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

# Apply the shared flocking monkey-patch (10-dim nn obs + centered target).
from flocking_patch import configure as _flocking_patch_configure  # noqa: F401

from benchmarl.algorithms.cmaes_han import CmaesHanConfig
from benchmarl.algorithms.cmaes_han_optimizer import CmaesHanOptimizer
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig, StaticMlpConfig
from benchmarl.models.han import HanConfig


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified HAN-vs-IPPO evaluation: flocking_orbit or "
                    "frozen-agent disturbance, with arbitrary n_agents.",
    )

    p.add_argument("--algo", type=str, required=True,
                   choices=["han", "cmaes-static-mlp"],
                   help="Which trained policy to evaluate. "
                        "``han`` for the CMA-ES Hebbian network; "
                        "``cmaes-static-mlp`` for the CMA-ES static-MLP "
                        "baseline.")
    p.add_argument("--han-exp-path", type=str, default=None,
                   help="Path to the HAN training folder (contains "
                        "han_results/). Required for --algo han.")
    p.add_argument("--static-mlp-exp-path", type=str, default=None,
                   help="Path to the static-MLP training folder "
                        "(contains static_mlp_results/). Required for "
                        "--algo cmaes-static-mlp.")
    p.add_argument("--scenario", type=str, required=True,
                   choices=["orbit", "frozen_agent"],
                   help="Eval scenario: plain orbit or mid-episode freeze.")
    p.add_argument("--n-agents", type=int, default=4,
                   help="Agent count at eval time. Train at 4 and reuse "
                        "the policy on 5/8 to test scalability.")
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--num-episodes", type=int, default=3)
    p.add_argument("--disturbance-step", type=int, default=400,
                   help="For --scenario frozen_agent only.")
    p.add_argument("--frozen-agent-idx", type=int, default=0,
                   help="For --scenario frozen_agent only.")

    # Fitness / scenario parameters (must match training time).
    p.add_argument("--neighbor-radius", type=float, default=0.5)
    p.add_argument("--safety-distance", type=float, default=0.15)
    p.add_argument("--orbit-radius", type=float, default=0.7)
    p.add_argument("--orbit-radius-tolerance", type=float, default=0.3)
    p.add_argument("--dt-floor", type=float, default=0.1)
    p.add_argument("--target-pos-x", type=float, default=0.0)
    p.add_argument("--target-pos-y", type=float, default=0.0)

    # HAN hyperparams (only used if --algo han).
    p.add_argument("--han-hidden-size", type=int, default=10)
    p.add_argument("--han-lr-hebb", type=float, default=0.01)
    p.add_argument("--han-weight-init", type=float, default=1.0)
    p.add_argument("--han-window-size", type=int, default=10)
    p.add_argument("--han-f-nn", type=int, default=4)
    p.add_argument("--han-f-hebb", type=int, default=1)

    # static-MLP hyperparams (only used if --algo cmaes-static-mlp).
    p.add_argument("--static-mlp-hidden-size", type=int, default=40,
                   help="Must match the hidden size used at static-MLP "
                        "training time (default 40 → 560 weights).")
    p.add_argument("--static-mlp-bias", action="store_true",
                   help="Include bias on Linear layers. Default off, "
                        "matching HAN's bias-less W matrix.")

    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--fps", type=int, default=20,
                   help="Video FPS (videos only saved on request).")
    p.add_argument("--save-video", action="store_true",
                   help="Render and save trajectory videos for the first "
                        "episode of each run.")
    p.add_argument("--max-video-frames", type=int, default=400)
    p.add_argument("--smooth-window", type=int, default=20)
    return p.parse_args()


# ============================================================================
# Experiment construction
# ============================================================================
def _get_task(n_agents: int, max_steps: int) -> VmasClass:
    """Build a vmas/flocking task with the patched env (10-dim nn obs).

    The IPPO optimization uses the default VMAS collision + dist_shaping
    rewards plus the orbit-shaping terms configured via
    ``flocking_patch.configure()`` (At + Dt). We use the same env
    settings at eval time so the rollout's reward stream matches what
    the trained policy expects. (For HAN evaluation the env reward is
    irrelevant — HAN ignores it and uses CMA-ES on F_orbit computed
    from raw positions — but using the same env keeps the two
    algorithms on equal footing for the rollout bookkeeping.)
    """
    task = VmasTask.FLOCKING.get_from_yaml()
    task.config["n_agents"] = n_agents
    task.config["max_steps"] = max_steps
    if "n_obstacles" in task.config:
        task.config["n_obstacles"] = 0  # match training
    return task


def _build_han_experiment(args, task) -> Experiment:
    model_config = HanConfig(
        hidden_size=args.han_hidden_size,
        lr_hebb=args.han_lr_hebb,
        weight_init=args.han_weight_init,
        window_size=args.han_window_size,
        f_nn=args.han_f_nn,
        f_hebb=args.han_f_hebb,
        activation_class=torch.nn.Tanh,
    )
    critic_model_config = MlpConfig(
        num_cells=[64, 64],
        activation_class=torch.nn.Tanh,
        layer_class=torch.nn.Linear,
    )
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.max_n_iters = 1
    experiment_config.save_folder = str(
        Path(__file__).parent.parent / "outputs")
    experiment_config.loggers = []  # no wandb
    experiment_config.train_device = args.device
    experiment = Experiment(
        task=task,
        algorithm_config=CmaesHanConfig.get_from_yaml(),
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
    )
    experiment._setup()
    return experiment


def _build_cmaes_static_mlp_experiment(args, task) -> Experiment:
    """Build a CMA-ES static-MLP experiment.

    Uses :class:`CmaesStaticMlpConfig` (algorithm) plus
    :class:`StaticMlpConfig` (model). The bias flag must match
    training time so the ``state_dict`` layout is the same.
    """
    from benchmarl.algorithms.cmaes_static_mlp import (
        CmaesStaticMlpConfig,
    )
    actor_model_config = StaticMlpConfig(
        num_cells=[args.static_mlp_hidden_size],
        activation_class=torch.nn.Tanh,
        layer_class=torch.nn.Linear,
        bias=args.static_mlp_bias,
    )
    critic_model_config = StaticMlpConfig(
        num_cells=[args.static_mlp_hidden_size],
        activation_class=torch.nn.Tanh,
        layer_class=torch.nn.Linear,
        bias=args.static_mlp_bias,
    )
    algorithm_config = CmaesStaticMlpConfig.get_from_yaml()
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.max_n_iters = 1
    experiment_config.save_folder = str(
        Path(__file__).parent.parent / "outputs")
    experiment_config.loggers = []  # no wandb
    experiment_config.train_device = args.device
    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=actor_model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
    )
    experiment._setup()
    return experiment


# ============================================================================
# Policy loaders
# ============================================================================
def _load_han_policy(experiment, exp_path: Path, device: str):
    """Load HAN policy state + ABCD params from a CMA-ES run."""
    han_dir = exp_path / "han_results"
    with open(han_dir / "results.json") as f:
        metadata = json.load(f)
    print(f"  HAN metadata: {metadata.get('n_layers')} layers, "
          f"fitness={metadata.get('best_fitness'):.3f}, "
          f"f_nn={metadata.get('f_nn')}, f_hebb={metadata.get('f_hebb')}")

    policy_path = han_dir / "policy_state.pt"
    experiment.policy.load_state_dict(
        torch.load(str(policy_path), map_location=device)
    )
    han_model = experiment.algorithm.get_han_model()
    abcd_path = han_dir / "abcd_params.npy"
    if abcd_path.exists():
        abcd = np.load(str(abcd_path))
        han_model.set_abcd_from_vector(torch.tensor(abcd, device=device))
        han_model.reset_all_weights()
        print(f"  Loaded ABCD: {len(abcd)} params")
    return han_model, metadata


def _load_static_mlp_policy(experiment, exp_path: Path, device: str,
                              label: str = "static-mlp"):
    """Load CMA-ES static-MLP policy state + flat weights.

    The on-disk layout (produced by ``run_cmaes_static_mlp_flocking_custom.py``) is:
        ``<exp>/static_mlp_results/policy_state.pt``
        ``<exp>/static_mlp_results/weights.npy``
        ``<exp>/static_mlp_results/results.json``
    """
    slm_dir = exp_path / "static_mlp_results"
    if not slm_dir.exists():
        raise FileNotFoundError(f"{slm_dir} not found")

    metadata = {}
    metadata_path = slm_dir / "results.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"  {label} metadata: total_weights={metadata.get('total_weights')}, "
              f"fitness={metadata.get('best_fitness'):.3f}, "
              f"fitness_mode={metadata.get('fitness_mode')}")

    policy_path = slm_dir / "policy_state.pt"
    if not policy_path.exists():
        raise FileNotFoundError(f"{policy_path} not found")
    experiment.policy.load_state_dict(
        torch.load(str(policy_path), map_location=device)
    )

    weights_path = slm_dir / "weights.npy"
    if weights_path.exists():
        weights = np.load(str(weights_path))
        static_mlp_model = experiment.algorithm.get_static_mlp_model()
        static_mlp_model.set_weights_from_vector(
            torch.tensor(weights, device=device)
        )
    return metadata


# ============================================================================
# Rollout
# ============================================================================
def _run_episode(
    env,
    group: str,
    policy,
    max_steps: int,
    optimizer: CmaesHanOptimizer,
    scenario: str,
    disturbance_step: int,
    frozen_idx: int,
    save_video: bool,
    max_video_frames: int,
    fps: int,
):
    """Run a single episode under the requested scenario.

    The policy is invoked for ALL agents every step. In the
    ``frozen_agent`` scenario, the action for the frozen agent is
    overridden to 0 from ``disturbance_step`` onward, and its position
    is anchored to the value it had at the first frozen step (so it
    behaves as a static obstacle, mirroring the HAN disturbance script).
    """
    td = env.reset()

    # Re-center the target for a symmetric orbit (VMAS default = (0, -1)).
    core = optimizer._get_vmas_core()
    target = getattr(getattr(core, "scenario", None), "_target", None)
    if target is not None and hasattr(target.state, "pos"):
        new_tgt = torch.tensor(
            [optimizer.experiment.task.config.get("target_pos_x", 0.0),
             optimizer.experiment.task.config.get("target_pos_y", 0.0)],
            device=target.state.pos.device,
            dtype=target.state.pos.dtype,
        ).unsqueeze(0).expand_as(target.state.pos)
        target.set_pos(new_tgt, batch_index=None)

    action_key = (group, "action")
    pos_history, rot_history, vel_history, target_pos_history = [], [], [], []
    frames = []
    done = False
    step = 0

    frozen_anchor_pos = None
    frozen_agent_obj = None

    while not done and step < max_steps:
        td = policy(td)

        if scenario == "frozen_agent" and step >= disturbance_step:
            td[action_key][:, frozen_idx] = 0.0
            if frozen_agent_obj is None:
                core = optimizer._vmas_core
                policy_agents_now = (
                    core.world.policy_agents
                    if core is not None and hasattr(core, "world")
                    else None
                )
                if (policy_agents_now is not None
                        and frozen_idx < len(policy_agents_now)):
                    frozen_agent_obj = policy_agents_now[frozen_idx]
                    frozen_anchor_pos = (
                        frozen_agent_obj.state.pos[0].detach().clone()
                    )

        if save_video and len(frames) < max_video_frames:
            try:
                frame = env.render(mode="rgb_array")
                if frame is not None:
                    frames.append(
                        torch.tensor(frame.copy())
                        .permute(2, 0, 1).unsqueeze(0)
                    )
            except Exception:
                pass

        td = env.step(td)

        if frozen_agent_obj is not None and frozen_anchor_pos is not None:
            with torch.no_grad():
                frozen_agent_obj.state.pos[0] = frozen_anchor_pos
                frozen_agent_obj.state.vel[0] = 0.0
                if hasattr(frozen_agent_obj.state, "force"):
                    frozen_agent_obj.state.force[0] = 0.0

        # Record per-step data on env 0.
        core = optimizer._vmas_core
        policy_agents = (
            core.world.policy_agents
            if core is not None and hasattr(core, "world")
            else None
        )
        if policy_agents is not None and len(policy_agents) > 0:
            pos_stack = torch.stack(
                [a.state.pos[0] for a in policy_agents], dim=0
            )
            vel_stack = torch.stack(
                [a.state.vel[0] for a in policy_agents], dim=0
            )
            for a_obj in policy_agents:
                vel0 = a_obj.state.vel[0]
                a_obj.state.rot[0, 0] = torch.atan2(vel0[1], vel0[0])
            rot_stack = torch.stack(
                [a.state.rot[0, 0] for a in policy_agents], dim=0
            )
            pos_history.append(pos_stack.detach().cpu())
            rot_history.append(rot_stack.detach().cpu())
            vel_history.append(vel_stack.detach().cpu())
            target = getattr(getattr(core, "scenario", None), "_target", None)
            if target is not None and hasattr(target.state, "pos"):
                target_pos_history.append(target.state.pos[0].detach().cpu())

        done = td.get(("next", "done")).any().item()
        td = td.get("next")
        step += 1

    return {
        "pos_history": pos_history,
        "rot_history": rot_history,
        "vel_history": vel_history,
        "target_pos_history": target_pos_history,
        "frames": frames,
        "steps": step,
    }


# ============================================================================
# Per-step metrics
# ============================================================================
def _compute_per_step_metrics(optimizer, pos_history, rot_history,
                              vel_history, target_pos_history,
                              frozen_agent_idx: Optional[int] = 0) -> Dict:
    T = len(pos_history)
    Fg_arr = np.zeros(T)
    At_arr = np.zeros(T)
    Dt_arr = np.zeros(T)
    Cg_arr = np.zeros(T)
    S_arr = np.zeros(T)
    frozen_speed_arr = np.zeros(T) if frozen_agent_idx is not None else None

    nr = float(optimizer.neighbor_radius)
    sd = float(optimizer.safety_distance)
    r_star = float(optimizer.orbit_radius)
    r_sigma = float(optimizer.orbit_radius_tolerance)
    dt_floor = float(optimizer.dt_floor)
    speed_threshold = 0.02

    for t in range(T):
        pos = pos_history[t]
        rot = rot_history[t]
        vel = vel_history[t]
        tgt = target_pos_history[t]
        N = pos.shape[0]
        eye = torch.eye(N, dtype=torch.bool)

        # At
        r_vec = pos - tgt
        r_norm = torch.linalg.vector_norm(r_vec, dim=-1, keepdim=True)
        r_unit = r_vec / (r_norm + 1e-6)
        tangent = torch.stack([-r_unit[:, 1], r_unit[:, 0]], dim=-1)
        v_dir = torch.stack([torch.cos(rot), torch.sin(rot)], dim=-1)
        dot = (v_dir * tangent).sum(dim=-1)
        align = ((dot + 1.0) * 0.5).clamp(0.0, 1.0)
        speed = torch.linalg.vector_norm(vel, dim=-1)
        speed_factor = (speed / (speed + speed_threshold)).clamp(0.0, 1.0)
        contrib = align * speed_factor
        valid = (r_norm.squeeze(-1) > 1e-6).float()
        denom = valid.sum().item()
        At = (contrib * valid).sum().item() / denom if denom > 0 else 0.0

        # Dt
        r_dist = r_norm.squeeze(-1)
        Dt_raw = torch.exp(-((r_dist - r_star) ** 2) / (2.0 * r_sigma ** 2))
        Dt = max(Dt_raw.mean().item(), dt_floor)

        # Cg
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        dist = torch.linalg.vector_norm(diff, dim=-1)
        adj = (dist < nr) & (~eye)
        num_groups = CmaesHanOptimizer._count_connected_components(
            optimizer, adj)
        Cg = 1.0 / max(int(num_groups), 1)

        # S
        dist_for_collision = dist.masked_fill(eye, float("inf"))
        in_collision = (dist_for_collision < sd).any(dim=-1)
        S = 1.0 - in_collision.float().mean().item()

        # Weighted F_orbit (方案 E: 1.5·At + Dt + 0.2·Cg + 0.8·S)
        Fg = 1.5 * At + Dt + 0.2 * Cg + 0.8 * S

        Fg_arr[t] = Fg
        At_arr[t] = At
        Dt_arr[t] = Dt
        Cg_arr[t] = Cg
        S_arr[t] = S
        if frozen_speed_arr is not None:
            frozen_speed_arr[t] = float(vel[frozen_agent_idx].norm().item())

    return {
        "Fg": Fg_arr, "At": At_arr, "Dt": Dt_arr,
        "Cg": Cg_arr, "S": S_arr,
        "frozen_speed": frozen_speed_arr,
    }


def _phase_summary(metrics: Dict, T: int, scenario: str,
                   disturbance_step: int) -> Dict:
    """Mean±std of F_orbit over each phase."""
    Fg = metrics["Fg"]

    def stat(arr, sl):
        a = arr[sl]
        if len(a) == 0:
            return float("nan"), float("nan")
        return float(a.mean()), float(a.std())

    if scenario == "frozen_agent" and disturbance_step < T:
        phases = {
            "baseline": (0, disturbance_step),
            "immediate_post": (disturbance_step, min(disturbance_step + 100, T)),
            "long_post": (disturbance_step + 100, T),
            "full_post": (disturbance_step, T),
        }
    else:
        # For "orbit" scenario, just report whole-episode + half-half.
        mid = T // 2
        phases = {
            "first_half": (0, mid),
            "second_half": (mid, T),
            "full": (0, T),
        }
    return {name: {"mean": stat(Fg, slice(lo, hi))[0],
                   "std":  stat(Fg, slice(lo, hi))[1]}
            for name, (lo, hi) in phases.items()}


def _save_video(frames, path, fps):
    if not frames:
        return
    import torchvision
    vid = torch.cat(frames, dim=0).unsqueeze(0)
    for idx in (-1, -2):
        if vid.shape[idx] % 2 != 0:
            vid = vid.index_select(idx, torch.arange(1, vid.shape[idx]))
    vid_rgb = vid[0].permute(0, 2, 3, 1)
    torchvision.io.write_video(path, vid_rgb.numpy(), fps=fps)
    print(f"  Saved video: {path} ({len(frames)} frames @ {fps}fps)")


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()

    # Configure shared flocking patch.
    _flocking_patch_configure(
        target_pos_x=args.target_pos_x,
        target_pos_y=args.target_pos_y,
        neighbor_radius=args.neighbor_radius,
    )

    if args.algo == "han" and args.han_exp_path is None:
        raise ValueError("--han-exp-path required for --algo han")
    if args.algo == "cmaes-static-mlp" and args.static_mlp_exp_path is None:
        raise ValueError(
            f"--static-mlp-exp-path required for --algo {args.algo}"
        )

    if args.algo == "han":
        exp_path = Path(args.han_exp_path)
    else:  # cmaes-static-mlp
        exp_path = Path(args.static_mlp_exp_path)
    algo_label = args.algo
    out_dir = (exp_path / "comparison_eval"
               / f"{args.scenario}_n{args.n_agents}_{algo_label}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Comparison eval — algo={algo_label}, scenario={args.scenario}, "
          f"n_agents={args.n_agents}")
    print("=" * 70)
    print(f"  exp_path: {exp_path}")
    print(f"  out_dir : {out_dir}")

    # Build the experiment under the *target* n_agents, so the env has the
    # right number of slots.
    task = _get_task(args.n_agents, args.max_steps)
    if args.algo == "han":
        experiment = _build_han_experiment(args, task)
        han_model, _ = _load_han_policy(
            experiment, exp_path, args.device)
    else:
        experiment = _build_cmaes_static_mlp_experiment(args, task)
        _load_static_mlp_policy(experiment, exp_path, args.device,
                                label=args.algo)
        han_model = None  # not used

    device = experiment.config.train_device
    # Build a "shell" optimizer so we can reuse its fitness math and env
    # accessors. We do NOT need its CMA-ES machinery; pass a no-op
    # pop_size/max_gens/...
    han_model_for_optimizer = han_model if han_model is not None else _FakeHan()
    optimizer = CmaesHanOptimizer(
        experiment=experiment,
        han_model=han_model_for_optimizer,
        fitness_mode="flocking_orbit",
        pop_size=1, max_gens=0, n_eval_episodes=1,
        device=device,
        collision_penalty_weight=2.0,
        safety_distance=args.safety_distance,
        neighbor_radius=args.neighbor_radius,
        movement_target_displacement=1.0,
        orbit_radius=args.orbit_radius,
        orbit_radius_tolerance=args.orbit_radius_tolerance,
        dt_floor=args.dt_floor,
    )
    # We need target_pos_x/y to be readable inside _run_episode. The
    # optimizer doesn't store these; we attach them on the experiment
    # task so the rollout helper can find them.
    experiment.task.config["target_pos_x"] = args.target_pos_x
    experiment.task.config["target_pos_y"] = args.target_pos_y

    group = list(experiment.group_map.keys())[0]
    env = experiment.test_env

    # Run episodes.
    all_summaries = []
    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep+1}/{args.num_episodes} ---")
        if han_model is not None:
            han_model.reset_all_weights()

        with torch.no_grad(), set_exploration_type(
                ExplorationType.DETERMINISTIC):
            data = _run_episode(
                env, group, experiment.policy, args.max_steps, optimizer,
                scenario=args.scenario,
                disturbance_step=args.disturbance_step,
                frozen_idx=args.frozen_agent_idx,
                save_video=(args.save_video and ep == 0),
                max_video_frames=args.max_video_frames,
                fps=args.fps,
            )

        print(f"  ran {data['steps']} steps, "
              f"captured {len(data['frames'])} frames")

        frozen_idx_for_metrics = (
            args.frozen_agent_idx if args.scenario == "frozen_agent" else None
        )
        metrics = _compute_per_step_metrics(
            optimizer, data["pos_history"], data["rot_history"],
            data["vel_history"], data["target_pos_history"],
            frozen_idx_for_metrics,
        )
        summary = _phase_summary(
            metrics, data["steps"], args.scenario, args.disturbance_step,
        )
        all_summaries.append(summary)
        if "full" in summary:
            print(f"  F_orbit full       = {summary['full']['mean']:.3f} "
                  f"± {summary['full']['std']:.3f}")
        else:
            # frozen_agent scenario: print the key contrast.
            base = summary["baseline"]["mean"]
            post = summary["full_post"]["mean"]
            print(f"  F_orbit baseline   = {base:.3f}")
            print(f"  F_orbit full_post  = {post:.3f}")
            print(f"  Δ after freeze     = {post - base:+.3f}")

        # Save per-step arrays.
        np.savez(
            out_dir / f"per_step_ep{ep}.npz",
            Fg=metrics["Fg"], At=metrics["At"], Dt=metrics["Dt"],
            Cg=metrics["Cg"], S=metrics["S"],
            frozen_speed=metrics["frozen_speed"]
                       if metrics["frozen_speed"] is not None
                       else np.zeros(data["steps"]),
            pos=np.stack([p.numpy() for p in data["pos_history"]]),
            target=np.stack([t.numpy() for t in data["target_pos_history"]]),
            rot=np.stack([r.numpy() for r in data["rot_history"]]),
            vel=np.stack([v.numpy() for v in data["vel_history"]]),
        )
        if args.save_video and ep == 0:
            _save_video(data["frames"],
                        str(out_dir / "trajectory_ep0.mp4"),
                        args.fps)

    # Save aggregate summary.
    aggregate = {
        "algo": args.algo,
        "scenario": args.scenario,
        "n_agents": args.n_agents,
        "max_steps": args.max_steps,
        "num_episodes": args.num_episodes,
        "disturbance_step": args.disturbance_step
                            if args.scenario == "frozen_agent" else None,
        "frozen_agent_idx": args.frozen_agent_idx
                            if args.scenario == "frozen_agent" else None,
        "per_episode": all_summaries,
        "exp_path": str(exp_path),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\n  Saved summary: {out_dir / 'summary.json'}")
    print(f"  All outputs:   {out_dir}")


class _FakeHan:
    """Stub used when --algo ippo: the optimizer only needs the
    ``reset_all_weights``/``set_abcd_*`` API, which is never called here
    (we only reuse the per-step fitness math)."""
    def reset_all_weights(self):
        pass
    def get_abcd_vector(self):
        return torch.zeros(0)
    def set_abcd_from_vector(self, v):
        pass


if __name__ == "__main__":
    main()
