"""Dynamic-adaptation evaluation for trained HAN on flocking.

Loads a trained CMA-ES+HAN model and runs a single episode with a
mid-episode disturbance: at step ``--disturbance-step`` the action of
agent ``--frozen-agent-idx`` is overridden to zero (it "freezes" in
place, but remains in the physics simulation and can be pushed around by
the other agents). The remaining 3 agents continue to run the HAN
policy.

Goal: observe whether the trained HAN policy can maintain flocking
behavior across a sudden reduction in the active agent count, which
probes the policy's distributed/robustness properties (vs. an
overfit-to-N-agents policy that collapses).

Outputs (under ``<experiment-path>/disturbance_eval/``):
  - fitness_curve.png : 2-panel plot (fitness components + geometry)
  - trajectory.mp4    : rollout video with disturbance marker
  - per_step_data.npz : raw per-step arrays (for further analysis)

Usage:
    /home/zhaozeming/miniconda3/envs/benchmarl/bin/python \\
        examples/running/run_cmaes_han_flocking_disturbance.py \\
        --experiment-path outputs/<your flocking experiment> \\
        --fitness-mode flocking_orbit \\
        --disturbance-step 400 --frozen-agent-idx 2 \\
        --hidden-size 10 --window-size 10 --f-nn 4 --f-hebb 1 \\
        --orbit-radius 0.7 --orbit-radius-tolerance 0.3 --dt-floor 0.1 \\
        --neighbor-radius 0.5 --safety-distance 0.15
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

# --------------------------------------------------------------------------
# Monkey-patch: apply the shared VMAS flocking patches (stationary +
# centered target + nearest-neighbor observation). MUST match the
# training script's observation layout exactly so the trained ABCD
# vector is valid at eval time. Configuration happens after argparse.
from flocking_patch import configure as _flocking_patch_configure  # noqa: F401

from benchmarl.algorithms.cmaes_han import CmaesHanConfig
from benchmarl.algorithms.cmaes_han_optimizer import CmaesHanOptimizer
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig
from benchmarl.models.han import HanConfig


def parse_args():
    p = argparse.ArgumentParser(
        description="Dynamic-adaptation evaluation: freeze one agent "
                    "mid-episode to test HAN robustness."
    )
    p.add_argument("--experiment-path", type=str, required=True,
                   help="Trained experiment folder (contains han_results/).")
    p.add_argument("--fitness-mode", type=str, default="flocking_orbit",
                   choices=CmaesHanOptimizer.FITNESS_MODES)
    p.add_argument("--collision-penalty-weight", type=float, default=2.0)
    p.add_argument("--safety-distance", type=float, default=0.15)
    p.add_argument("--neighbor-radius", type=float, default=0.5)
    p.add_argument("--movement-target-displacement", type=float, default=1.0)
    p.add_argument("--orbit-radius", type=float, default=0.7)
    p.add_argument("--orbit-radius-tolerance", type=float, default=0.3)
    p.add_argument("--dt-floor", type=float, default=0.1)

    p.add_argument("--hidden-size", type=int, default=10,
                   help="HAN hidden layer size. Default 10 matches "
                        "the 10-dim flocking observation "
                        "(pos+vel+target_rel+nn_rel_pos+nn_rel_vel).")
    p.add_argument("--lr-hebb", type=float, default=0.01)
    p.add_argument("--weight-init", type=float, default=1.0)
    p.add_argument("--window-size", type=int, default=10)
    p.add_argument("--f-nn", type=int, default=4)
    p.add_argument("--f-hebb", type=int, default=1)

    # Disturbance
    p.add_argument("--disturbance-step", type=int, default=800,
                   help="Step at which to freeze the target agent.")
    p.add_argument("--frozen-agent-idx", type=int, default=2,
                   help="Index of the agent to freeze at the disturbance step.")
    p.add_argument("--num-episodes", type=int, default=1,
                   help="Number of disturbance episodes to run.")
    p.add_argument("--max-steps", type=int, default=800,
                   help="Total episode length (must be > --disturbance-step).")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--max-video-frames", type=int, default=800)
    p.add_argument("--smooth-window", type=int, default=20,
                   help="Moving-average window for the plot curves.")
    p.add_argument("--target-pos-x", type=float, default=0.0,
                   help="Override the target agent's x position after "
                        "reset (default 0.0; VMAS default is also 0).")
    p.add_argument("--target-pos-y", type=float, default=0.0,
                   help="Override the target agent's y position after "
                        "reset. VMAS default is -y_dim = -1.0, which "
                        "places the target at the bottom edge of the "
                        "world and biases the flocking orbit "
                        "asymmetrically. Setting to 0.0 centers it.")
    return p.parse_args()


args = parse_args()

# Configure the shared flocking patches to match training exactly
_flocking_patch_configure(
    target_pos_x=args.target_pos_x,
    target_pos_y=args.target_pos_y,
    neighbor_radius=args.neighbor_radius,
)


def _get_task():
    task = VmasTask.FLOCKING.get_from_yaml()
    # Match the training environment: no obstacles. (The default yaml
    # ships n_obstacles=5; the training script overrides it to 0.)
    if "n_obstacles" in task.config:
        task.config["n_obstacles"] = 0
    return task


def _create_model_config():
    return HanConfig(
        hidden_size=args.hidden_size,
        lr_hebb=args.lr_hebb,
        weight_init=args.weight_init,
        window_size=args.window_size,
        f_nn=args.f_nn,
        f_hebb=args.f_hebb,
        activation_class=torch.nn.Tanh,
    )


def _create_critic_model_config():
    return MlpConfig(
        num_cells=[64, 64],
        activation_class=torch.nn.Tanh,
        layer_class=torch.nn.Linear,
    )


def _setup_experiment_for_cmaes(task, model_config, critic_model_config, output_dir):
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.max_n_iters = 1
    experiment_config.save_folder = str(output_dir)
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


def _load_trained_model(experiment, exp_path, device):
    """Load policy_state.pt + ABCD params from a trained experiment."""
    han_dir = Path(exp_path) / "han_results"
    metadata_path = han_dir / "results.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    print(f"Loaded metadata: {metadata['n_layers']} layers, "
          f"fitness={metadata['best_fitness']:.4f}, "
          f"fitness_mode={metadata.get('fitness_mode', 'n/a')}")

    policy_path = han_dir / "policy_state.pt"
    if not policy_path.exists():
        raise FileNotFoundError(f"policy_state.pt not found at {policy_path}")
    experiment.policy.load_state_dict(
        torch.load(str(policy_path), map_location=device)
    )

    han_model = experiment.algorithm.get_han_model()
    abcd_path = han_dir / "abcd_params.npy"
    if abcd_path.exists():
        abcd = np.load(str(abcd_path))
        han_model.set_abcd_from_vector(torch.tensor(abcd, device=device))
        han_model.reset_all_weights()
        print(f"Loaded ABCD params: {len(abcd)} parameters")
    return han_model, metadata


def _build_optimizer(experiment, han_model, device):
    """Build a CmaesHanOptimizer just so we can reuse its fitness
    component math (it has no _run_one_episode called in this script)."""
    return CmaesHanOptimizer(
        experiment=experiment,
        han_model=han_model,
        fitness_mode=args.fitness_mode,
        pop_size=1, max_gens=0, n_eval_episodes=1,
        device=device,
        collision_penalty_weight=args.collision_penalty_weight,
        safety_distance=args.safety_distance,
        neighbor_radius=args.neighbor_radius,
        movement_target_displacement=args.movement_target_displacement,
        orbit_radius=args.orbit_radius,
        orbit_radius_tolerance=args.orbit_radius_tolerance,
        dt_floor=args.dt_floor,
    )


def _compute_per_step_metrics(optimizer, pos_history, rot_history,
                              target_pos_history, vel_history,
                              frozen_agent_idx=0):
    """Per-step decomposition of the flocking_orbit fitness.

    Returns a dict of 1-D numpy arrays of length T, one value per step:
      Fg, At, Dt, Cg, S
    plus a 'frozen_speed' array tracking the frozen agent's speed.
    """
    T = len(pos_history)
    Fg_arr = np.zeros(T)
    At_arr = np.zeros(T)
    Dt_arr = np.zeros(T)
    Cg_arr = np.zeros(T)
    S_arr = np.zeros(T)
    fgs = []  # individual term sums for the "weighted" Fg
    frozen_speed_arr = np.zeros(T)

    nr = float(optimizer.neighbor_radius)
    sd = float(optimizer.safety_distance)
    r_star = float(optimizer.orbit_radius)
    r_sigma = float(optimizer.orbit_radius_tolerance)
    dt_floor = float(optimizer.dt_floor)

    # Speed threshold: agents slower than this contribute At=0.
    # Matches the post-fix optimizer behavior.
    speed_threshold = 0.02

    for t in range(T):
        pos = pos_history[t]
        rot = rot_history[t]
        vel = vel_history[t]
        tgt = target_pos_history[t]
        N = pos.shape[0]

        eye = torch.eye(N, dtype=torch.bool)

        # --- At: speed-modulated radial-tangential alignment ---
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
        if denom > 0:
            At = (contrib * valid).sum().item() / denom
        else:
            At = 0.0

        # --- Dt ---
        r_dist = r_norm.squeeze(-1)
        Dt_raw = torch.exp(-((r_dist - r_star) ** 2) / (2.0 * r_sigma ** 2))
        Dt = max(Dt_raw.mean().item(), dt_floor)

        # --- Cg: connected components ---
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        dist = torch.linalg.vector_norm(diff, dim=-1)
        adj = (dist < nr) & (~eye)
        num_groups = CmaesHanOptimizer._count_connected_components(
            optimizer, adj)
        Cg = 1.0 / max(int(num_groups), 1)

        # --- S ---
        dist_for_collision = dist.masked_fill(eye, float("inf"))
        in_collision = (dist_for_collision < sd).any(dim=-1)
        S = 1.0 - in_collision.float().mean().item()

        Fg = At + Dt + 0.5 * Cg + 0.5 * S

        Fg_arr[t] = Fg
        At_arr[t] = At
        Dt_arr[t] = Dt
        Cg_arr[t] = Cg
        S_arr[t] = S
        # vel_history[t] shape is (N, 2); track frozen agent's speed.
        # vel_history[t] shape is (N, 2); track frozen agent's speed.
        # Use per-step velocity norm (frozen agent index selects the
        # agent within this step's velocity matrix).
        frozen_speed_arr[t] = float(
            vel[frozen_agent_idx].norm().item()
        )
    return {
        "Fg": Fg_arr, "At": At_arr, "Dt": Dt_arr,
        "Cg": Cg_arr, "S": S_arr,
        "frozen_speed": frozen_speed_arr,
    }


def run_disturbance_episode(optimizer, env, group, max_steps, policy,
                            disturbance_step, frozen_idx, render=False):
    """Run a single episode with a freeze disturbance.

    Returns per-step data: pos_history, rot_history, vel_history,
    target_pos_history, frames (optional, for video).
    """
    optimizer._get_vmas_core()
    td = env.reset()

    # Override the target's initial position. VMAS flocking sets the
    # target at (0, -y_dim) = (0, -1) by default, which biases the
    # orbit asymmetrically (more space above the target than below).
    # Re-centering the target makes the orbit_radius symmetric around
    # it. This must run AFTER env.reset() and BEFORE the first policy
    # call so HAN sees the corrected target in its observation
    # (agent.state.pos - target.state.pos).
    core = optimizer._vmas_core
    target = getattr(getattr(core, "scenario", None), "_target", None)
    if target is not None and hasattr(target.state, "pos"):
        new_target_pos = torch.tensor(
            [args.target_pos_x, args.target_pos_y],
            device=target.state.pos.device,
            dtype=target.state.pos.dtype,
        ).unsqueeze(0).expand_as(target.state.pos)
        target.set_pos(new_target_pos, batch_index=None)
        print(f"  Target repositioned to ({args.target_pos_x}, "
              f"{args.target_pos_y}) for centered orbit.")

    pos_history = []
    rot_history = []
    vel_history = []
    target_pos_history = []
    frames = []
    done = False
    step = 0

    action_key = (group, "action")

    # Anchor position for the frozen agent — captured the first time the
    # disturbance activates, then restored after every subsequent step
    # so the frozen agent behaves like a fixed obstacle (other agents
    # must learn to flow around it).
    frozen_anchor_pos = None  # (2,) on env 0
    frozen_agent_obj = None   # reference to the frozen VMAS Agent

    while not done and step < max_steps:
        # 1) Policy produces action for ALL agents.
        td = policy(td)

        # 2) DISTURBANCE: from the configured step onward, override the
        #    frozen agent's action to zero AND pin its position so it
        #    behaves like a static obstacle (other agents can't push it).
        # NOTE on indexing: td[action_key] has shape (num_envs, n_agents,
        # action_dim). We collect data from env index 0 only, so the
        # override targets env 0 / agent `frozen_idx`. Slicing [:, idx]
        # would freeze that agent in ALL envs (so pos_history etc. show
        # the same per-env behavior we expect).
        if step >= disturbance_step:
            td[action_key][:, frozen_idx] = 0.0
            # First disturbance step: capture the agent's current
            # position as the anchor (so we can restore it after each
            # step). On subsequent steps we use the same anchor.
            if frozen_agent_obj is None:
                core = optimizer._vmas_core
                policy_agents_now = (core.world.policy_agents
                                     if core is not None
                                     and hasattr(core, "world")
                                     else None)
                if (policy_agents_now is not None
                        and frozen_idx < len(policy_agents_now)):
                    frozen_agent_obj = policy_agents_now[frozen_idx]
                    frozen_anchor_pos = (
                        frozen_agent_obj.state.pos[0].detach().clone()
                    )

        if render:
            try:
                frame = env.render(mode="rgb_array")
                if frame is not None:
                    frames.append(
                        torch.tensor(frame.copy())
                        .permute(2, 0, 1).unsqueeze(0)
                    )
            except Exception:
                pass

        # 3) Step the env with the (possibly overridden) action.
        td = env.step(td)

        # 3b) If the disturbance is active, force the frozen agent back
        # to its anchor position so it behaves like a static obstacle.
        # Without this, the other agents' collision impulses (collision
        # force = 400) slowly drift it away, defeating the "static
        # obstacle" semantics.
        if frozen_agent_obj is not None and frozen_anchor_pos is not None:
            with torch.no_grad():
                frozen_agent_obj.state.pos[0] = frozen_anchor_pos
                frozen_agent_obj.state.vel[0] = 0.0
                # Also zero the accumulated force so VMAS doesn't try to
                # apply residual impulses at the next substep.
                if hasattr(frozen_agent_obj.state, "force"):
                    frozen_agent_obj.state.force[0] = 0.0

        # 4) Read the vmas core for absolute positions, velocities,
        #    target. Patch state.rot from velocity direction (needed
        #    for the At term since Holonomic doesn't update rot).
        # IMPORTANT: core.agents includes the target as the first entry
        # in VMAS flocking (target is added before policy agents). We
        # only want policy_agents for the fitness computation (matches
        # the training code's behavior in cmaes_han_optimizer.py).
        core = optimizer._vmas_core
        policy_agents = (core.world.policy_agents
                         if core is not None and hasattr(core, "world")
                         else None)
        if policy_agents is not None and len(policy_agents) > 0:
            pos_stack = torch.stack(
                [a.state.pos[0] for a in policy_agents], dim=0
            )  # (N, 2)
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


def smooth(arr, win):
    if win <= 1 or len(arr) == 0:
        return np.asarray(arr)
    arr = np.asarray(arr, dtype=float)
    kernel = np.ones(win) / win
    out = np.convolve(arr, kernel, mode="same")
    # Fix endpoints (convolution-with-same shrinks them)
    half = win // 2
    out[:half] = arr[:half]
    out[-half:] = arr[-half:]
    return out


def plot_fitness_curve(metrics, pos_history, target_pos_history,
                       disturbance_step, output_path, smooth_window):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = len(metrics["Fg"])
    steps_axis = np.arange(T)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- Panel 1: fitness components ---
    ax1.plot(steps_axis, smooth(metrics["Fg"], smooth_window),
             label="F_orbit (total)", color="black", linewidth=2)
    ax1.plot(steps_axis, smooth(metrics["At"], smooth_window),
             label="At (tangential align)", color="C0")
    ax1.plot(steps_axis, smooth(metrics["Dt"], smooth_window),
             label="Dt (distance band)", color="C1")
    ax1.plot(steps_axis, smooth(0.5 * metrics["Cg"], smooth_window),
             label="0.5·Cg (cohesion)", color="C2")
    ax1.plot(steps_axis, smooth(0.5 * metrics["S"], smooth_window),
             label="0.5·S (separation)", color="C3")
    ax1.axvline(disturbance_step, color="red", linestyle="--", alpha=0.7,
                label=f"Disturbance (step {disturbance_step})")
    ax1.axvspan(0, disturbance_step, alpha=0.08, color="green",
                label="Baseline phase")
    ax1.axvspan(disturbance_step, T, alpha=0.08, color="orange",
                label="Recovery phase")
    ax1.set_ylabel("Fitness component")
    ax1.set_ylim(0, 4.2)
    ax1.set_title("flocking_orbit fitness over time (smoothed)")
    ax1.legend(loc="upper right", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: geometry ---
    pos_arr = torch.stack(pos_history).numpy()       # (T, N, 2)
    tgt_arr = torch.stack(target_pos_history).numpy()  # (T, 2)
    centroid = pos_arr.mean(axis=1)                   # (T, 2)
    centroid_to_target = np.linalg.norm(centroid - tgt_arr, axis=1)
    # Group spread = mean pairwise distance (excluding self).
    spreads = []
    for t in range(T):
        p = pos_arr[t]
        diff = p[:, None, :] - p[None, :, :]
        d = np.linalg.norm(diff, axis=-1)
        N = d.shape[0]
        mask = ~np.eye(N, dtype=bool)
        spreads.append(d[mask].mean())
    spreads = np.array(spreads)
    # Frozen agent distance to target.
    frozen_to_target = np.linalg.norm(
        pos_arr[:, args.frozen_agent_idx, :] - tgt_arr, axis=1)

    ax2.plot(steps_axis, smooth(centroid_to_target, smooth_window),
             label="Centroid → target", color="purple", linewidth=2)
    ax2.plot(steps_axis, smooth(spreads, smooth_window),
             label="Group spread (mean pairwise dist)", color="teal")
    ax2.plot(steps_axis, smooth(frozen_to_target, smooth_window),
             label=f"Agent #{args.frozen_agent_idx} (frozen) → target",
             color="brown", linestyle=":")
    ax2.axvline(disturbance_step, color="red", linestyle="--", alpha=0.7)
    ax2.axvspan(0, disturbance_step, alpha=0.08, color="green")
    ax2.axvspan(disturbance_step, T, alpha=0.08, color="orange")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Distance (VMAS world units)")
    ax2.set_title("Geometric properties")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  Saved fitness curve plot to: {output_path}")


def save_video(frames, output_path, fps):
    import torchvision
    if not frames:
        print("  No frames captured — skipping video.")
        return
    vid = torch.cat(frames, dim=0).unsqueeze(0)
    for idx in (-1, -2):
        if vid.shape[idx] % 2 != 0:
            vid = vid.index_select(idx, torch.arange(1, vid.shape[idx]))
    vid_rgb = vid[0].permute(0, 2, 3, 1)
    torchvision.io.write_video(output_path, vid_rgb.numpy(), fps=fps)
    print(f"  Saved video: {output_path} ({len(frames)} frames @ {fps}fps)")


def print_phase_summary(metrics, pos_history, target_pos_history,
                        disturbance_step):
    Fg = metrics["Fg"]
    T = len(Fg)
    if disturbance_step >= T:
        print(f"\n  [warn] disturbance_step={disturbance_step} >= T={T}")
        return
    # Three windows: pre-disturbance, immediate (0..100 post),
    # long (100+ post), full post.
    pre = slice(0, disturbance_step)
    immediate = slice(disturbance_step, min(disturbance_step + 100, T))
    long_post = slice(disturbance_step + 100, T)
    full_post = slice(disturbance_step, T)

    def stat(arr, sl):
        a = arr[sl]
        if len(a) == 0:
            return float("nan"), float("nan")
        return float(a.mean()), float(a.std())

    print()
    print("=" * 70)
    print("Phase summary (mean ± std)")
    print("=" * 70)
    print(f"  Baseline        [0..{disturbance_step}):       "
          f"Fg={stat(Fg, pre)[0]:.3f}±{stat(Fg, pre)[1]:.3f}, "
          f"At={stat(metrics['At'], pre)[0]:.3f}, "
          f"Dt={stat(metrics['Dt'], pre)[0]:.3f}, "
          f"Cg={stat(metrics['Cg'], pre)[0]:.3f}, "
          f"S={stat(metrics['S'], pre)[0]:.3f}")
    print(f"  Immediate post  [{disturbance_step}..{disturbance_step+100}):  "
          f"Fg={stat(Fg, immediate)[0]:.3f}±{stat(Fg, immediate)[1]:.3f}")
    print(f"  Long post       [{disturbance_step+100}..{T}):    "
          f"Fg={stat(Fg, long_post)[0]:.3f}±{stat(Fg, long_post)[1]:.3f}")
    print(f"  Full post       [{disturbance_step}..{T}):     "
          f"Fg={stat(Fg, full_post)[0]:.3f}±{stat(Fg, full_post)[1]:.3f}")
    if not math.isnan(stat(Fg, pre)[0]):
        drop = stat(Fg, pre)[0] - stat(Fg, full_post)[0]
        print(f"\n  Fitness drop after disturbance: {drop:+.3f}")
        if drop < 0.1:
            print("  → HAN is ROBUST to the disturbance (fitness barely changed).")
        elif drop < 0.4:
            print("  → HAN shows partial adaptation (some drop, partial recovery).")
        else:
            print("  → HAN does NOT adapt well to the disturbance.")
    print("=" * 70)


if __name__ == "__main__":
    if args.disturbance_step >= args.max_steps:
        raise ValueError(
            f"--disturbance-step ({args.disturbance_step}) must be < "
            f"--max-steps ({args.max_steps})"
        )
    if args.frozen_agent_idx < 0:
        raise ValueError("--frozen-agent-idx must be >= 0")

    exp_path = Path(args.experiment_path)
    output_dir = exp_path / "disturbance_eval"
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("Dynamic-Adaptation Eval — Flocking with Frozen Agent")
    print("=" * 70)
    print(f"  experiment: {exp_path}")
    print(f"  fitness_mode: {args.fitness_mode}")
    print(f"  disturbance_step: {args.disturbance_step}")
    print(f"  frozen_agent_idx: {args.frozen_agent_idx}")
    print(f"  max_steps: {args.max_steps}")
    print(f"  num_episodes: {args.num_episodes}")
    print(f"  output_dir: {output_dir}")
    print()

    task = _get_task()
    model_config = _create_model_config()
    critic_model_config = _create_critic_model_config()

    # Build the experiment (test_env, policy, etc.) under outputs/ — but we
    # will save disturbance outputs under <exp>/disturbance_eval/.
    base_output = Path(__file__).parent.parent / "outputs"
    base_output.mkdir(exist_ok=True)
    experiment = _setup_experiment_for_cmaes(
        task, model_config, critic_model_config, base_output)

    device = experiment.config.train_device
    han_model, metadata = _load_trained_model(experiment, exp_path, device)
    optimizer = _build_optimizer(experiment, han_model, device)

    group = list(experiment.group_map.keys())[0]
    env = experiment.test_env

    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep+1}/{args.num_episodes} ---")
        han_model.reset_all_weights()
        with torch.no_grad(), set_exploration_type(
                ExplorationType.DETERMINISTIC):
            data = run_disturbance_episode(
                optimizer, env, group, args.max_steps, experiment.policy,
                disturbance_step=args.disturbance_step,
                frozen_idx=args.frozen_agent_idx,
                render=True,
            )

        print(f"  ran {data['steps']} steps, "
              f"captured {len(data['frames'])} frames")

        metrics = _compute_per_step_metrics(
            optimizer, data["pos_history"], data["rot_history"],
            data["target_pos_history"], data["vel_history"],
            args.frozen_agent_idx,
        )
        print_phase_summary(
            metrics, data["pos_history"], data["target_pos_history"],
            args.disturbance_step)

        # Save artifacts.
        np.savez(
            output_dir / "per_step_data.npz",
            Fg=metrics["Fg"], At=metrics["At"], Dt=metrics["Dt"],
            Cg=metrics["Cg"], S=metrics["S"],
            frozen_speed=metrics["frozen_speed"],
            pos=np.stack([p.numpy() for p in data["pos_history"]]),
            target=np.stack([t.numpy() for t in data["target_pos_history"]]),
            rot=np.stack([r.numpy() for r in data["rot_history"]]),
            vel=np.stack([v.numpy() for v in data["vel_history"]]),
        )
        print(f"  Saved raw per-step data: {output_dir / 'per_step_data.npz'}")

        plot_fitness_curve(
            metrics, data["pos_history"], data["target_pos_history"],
            args.disturbance_step, str(output_dir / "fitness_curve.png"),
            args.smooth_window,
        )
        save_video(data["frames"], str(output_dir / "trajectory.mp4"),
                   args.fps)

    print(f"\nAll outputs written to: {output_dir}")