"""Standalone cross-formation evaluation script.

Loads a policy + ABCD from a training run, builds a fresh env with a
DIFFERENT formation_type than what was used during training, runs N
evaluation rollouts, and saves videos + per-episode stats to a fresh
output directory (does NOT overwrite the source training run's outputs).

Usage:
    python eval_cross_formation.py \
        --experiment-path /path/to/cmaeshan_..._run/ \
        --formation-type grid \
        --output-dir /path/to/output/ \
        --n-episodes 10 \
        --max-video-frames 200
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

from benchmarl.algorithms.cmaes_han import CmaesHanConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig
from benchmarl.models.hgn import HgnConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-path", type=str, required=True,
                   help="Source training run directory (must contain "
                        "han_results/abcd_params.npy and policy_state.pt)")
    p.add_argument("--formation-type", type=str, default="grid",
                   choices=["circle", "line", "v", "grid"])
    p.add_argument("--formation-radius", type=float, default=0.5)
    p.add_argument("--spawn-radius", type=float, default=0.4)
    p.add_argument("--spawn-cluster-radius", type=float, default=0.25)
    p.add_argument("--n-agents", type=int, default=6)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--n-episodes", type=int, default=10,
                   help="Number of evaluation episodes to record")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, required=True,
                   help="Where to save the new videos and stats "
                        "(DOES NOT touch the source directory)")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--max-video-frames", type=int, default=200)
    # HGN config must match the source training run, otherwise ABCD has
    # the wrong dimensionality.
    p.add_argument("--d-h", type=int, default=18)
    p.add_argument("--n-message-steps", type=int, default=1)
    p.add_argument("--topology", type=str, default="full")
    p.add_argument("--lr-hebb", type=float, default=0.01)
    p.add_argument("--weight-init", type=float, default=1.0)
    p.add_argument("--window-size", type=int, default=10)
    p.add_argument("--f-nn", type=int, default=4)
    p.add_argument("--f-hebb", type=int, default=1)
    p.add_argument("--success-reward", type=float, default=5.0)
    p.add_argument("--final-weight", type=float, default=1.0)
    p.add_argument("--formation-collision-penalty", type=float, default=2.0)
    p.add_argument("--formation-timeout-penalty", type=float, default=2.0)
    p.add_argument("--formation-reach-radius", type=float, default=0.10)
    return p.parse_args()


def get_agents_and_scenario(env):
    """Walk wrapper chain to find vmas World.agents and scenario.

    Robust to either torchrl's TransformedEnv → VmasEnv → Environment
    (last has `.scenario` directly) or older chains where scenario is on
    world.scenario.
    """
    node = env
    while True:
        sub = None
        for attr in ('base_env', '_env', 'env'):
            cand = getattr(node, attr, None)
            if cand is not None and cand is not node:
                sub = cand
                break
        if sub is None:
            break
        node = sub

    # Drill one more level if we're still on a wrapper.
    if hasattr(node, 'scenario'):
        scenario = node.scenario
    elif hasattr(node, '_env') and hasattr(node._env, 'scenario'):
        scenario = node._env.scenario
        node = node._env
    else:
        scenario = None

    # World.agents lives on the vmas Environment object, not the
    # torchrl wrapper.
    world = getattr(node, '_env', None) or node
    if not hasattr(world, 'agents'):
        # Last-ditch: walk through env._env
        if hasattr(node, 'env'):
            world = node.env
    if not hasattr(world, 'agents'):
        raise RuntimeError(
            f"Could not locate vmas World.agents; node type = {type(node).__name__}")

    if scenario is None and hasattr(world, 'scenario'):
        scenario = world.scenario
    if scenario is None:
        raise RuntimeError("Could not locate vmas scenario object")
    return world.agents, scenario, world


def main():
    args = parse_args()
    src = Path(args.experiment_path)
    assert (src / "han_results" / "policy_state.pt").exists(), (
        f"No policy_state.pt in {src}/han_results/")
    assert (src / "han_results" / "abcd_params.npy").exists(), (
        f"No abcd_params.npy in {src}/han_results/")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "videos"
    video_dir.mkdir(exist_ok=True)
    print(f"[eval_cross_formation] output dir: {out_dir}")

    # Load source metadata
    src_meta = json.load(open(src / "han_results" / "results.json"))
    print(f"[eval_cross_formation] source run: "
          f"best_fitness={src_meta['best_fitness']}, "
          f"generations_completed={src_meta['generations_completed']}, "
          f"layers={[(l['in'], l['out']) for l in src_meta['layer_shapes']]}")

    # Build env with the NEW formation type (not the one used at training).
    task = VmasTask.BENCHMARL_HGN_FORMATION
    task_cfg = task.get_from_yaml().config.copy()
    task_cfg.update(dict(
        n_agents=args.n_agents,
        formation_type=args.formation_type,
        formation_radius=args.formation_radius,
        spawn_radius=args.spawn_radius,
        spawn_cluster_radius=args.spawn_cluster_radius,
        max_steps=args.max_steps,
    ))
    task_obj = task.get_task(config=task_cfg)
    print(f"[eval_cross_formation] task config: "
          f"formation_type={task_cfg['formation_type']}, "
          f"n_agents={task_cfg['n_agents']}, "
          f"max_steps={task_cfg['max_steps']}")

    algorithm_config = CmaesHanConfig.get_from_yaml()
    model_config = HgnConfig(
        d_h=args.d_h,
        n_message_steps=args.n_message_steps,
        topology=args.topology,
        lr_hebb=args.lr_hebb,
        weight_init=args.weight_init,
        window_size=args.window_size,
        f_nn=args.f_nn,
        f_hebb=args.f_hebb,
        activation_class=torch.nn.Tanh,
    )
    critic_model_config = MlpConfig(
        num_cells=[64, 64],
        activation_class=torch.nn.Tanh,
        layer_class=torch.nn.Linear,
    )
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.max_n_iters = 1

    experiment = Experiment(
        task=task_obj,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=args.seed,
        config=experiment_config,
    )
    experiment._setup()

    # Load policy + ABCD from source.
    policy_state = torch.load(
        src / "han_results" / "policy_state.pt",
        map_location="cpu",
    )
    experiment.policy.load_state_dict(policy_state)
    hgn = experiment.algorithm.get_hgn_model()
    abcd = np.load(src / "han_results" / "abcd_params.npy")
    hgn.set_abcd_from_vector(torch.tensor(abcd, dtype=torch.float32))
    hgn.reset_all_weights()
    print(f"[eval_cross_formation] loaded ABCD: "
          f"shape={abcd.shape}, L2={np.linalg.norm(abcd):.2f}")

    # Evaluate
    env = experiment.test_env
    policy = experiment.policy
    formation_targets = None

    per_episode = []
    for ep in range(args.n_episodes):
        # Per-episode reset of W (so we measure policy behaviour,
        # not Hebbian adaptation).
        hgn.reset_all_weights()
        hgn.set_abcd_from_vector(torch.tensor(abcd, dtype=torch.float32))

        td = env.reset()
        agents, scenario, world = get_agents_and_scenario(env)
        if formation_targets is None:
            formation_targets = scenario.formation_targets.clone()

        frames = []
        pos_history = []

        with torch.no_grad(), set_exploration_type(
                ExplorationType.DETERMINISTIC):
            for step in range(args.max_steps):
                td = policy(td)
                td = env.step(td)

                if args.max_video_frames and len(frames) < args.max_video_frames:
                    try:
                        frame = env.render(mode="rgb_array")
                        if frame is not None:
                            frames.append(
                                torch.tensor(frame.copy())
                                .permute(2, 0, 1).unsqueeze(0))
                    except Exception:
                        pass

                done = td.get(("next", "done"))
                if done.dim() == 2:
                    done = done[0]
                if done.item():
                    break

        # Per-episode statistics
        final_pos = torch.stack([a.state.pos[0] for a in agents], dim=0)
        d = torch.linalg.vector_norm(
            final_pos - formation_targets, dim=-1)
        reached = bool((d <= args.formation_reach_radius).all().item())
        mean_d = d.mean().item()
        max_d = d.max().item()

        if reached:
            fitness = float(args.success_reward)
        else:
            fitness = -args.final_weight * mean_d
            if step >= args.max_steps - 1 and not reached:
                fitness -= args.formation_timeout_penalty

        per_episode.append({
            "episode": ep,
            "steps": step + 1,
            "reached": reached,
            "mean_dist_to_target": mean_d,
            "max_dist_to_target": max_d,
            "per_agent_dist": d.tolist(),
            "fitness": fitness,
        })
        print(f"  ep {ep:2d}: steps={step+1:3d}, "
              f"reached={reached}, mean_dist={mean_d:.3f}, "
              f"max_dist={max_d:.3f}, fitness={fitness:+.2f}")

        # Save video
        if frames:
            import torchvision
            vid = torch.cat(frames, dim=0).unsqueeze(0)
            for idx in (-1, -2):
                if vid.shape[idx] % 2 != 0:
                    vid = vid.index_select(
                        idx, torch.arange(1, vid.shape[idx]))
            video_path = video_dir / f"eval_{args.formation_type}_ep{ep}.mp4"
            vid_rgb = vid[0].permute(0, 2, 3, 1)
            torchvision.io.write_video(
                str(video_path), vid_rgb.numpy(), fps=args.fps)
            print(f"    video: {video_path}")

    # Aggregate statistics
    reached_n = sum(1 for r in per_episode if r["reached"])
    mean_fitness = float(np.mean([r["fitness"] for r in per_episode]))
    mean_dist = float(np.mean([r["mean_dist_to_target"] for r in per_episode]))
    mean_max_dist = float(np.mean([r["max_dist_to_target"] for r in per_episode]))
    success_rate = reached_n / args.n_episodes

    print()
    print("=" * 60)
    print(f"Cross-formation generalization test summary")
    print(f"  Source formation:        {src_meta.get('fitness_mode', '?')}")
    print(f"  Eval formation:          {args.formation_type}")
    print(f"  Eval formation_radius:   {args.formation_radius}")
    print(f"  Episodes:                {args.n_episodes}")
    print(f"  Success rate:            {success_rate*100:.1f}%")
    print(f"  Mean fitness:            {mean_fitness:+.3f}")
    print(f"  Mean dist to target:     {mean_dist:.3f}")
    print(f"  Mean max dist to target: {mean_max_dist:.3f}")
    print("=" * 60)

    # Save JSON summary
    summary = {
        "source_experiment_path": str(src),
        "source_best_fitness": src_meta["best_fitness"],
        "source_generations": src_meta["generations_completed"],
        "eval_formation_type": args.formation_type,
        "eval_formation_radius": args.formation_radius,
        "eval_n_agents": args.n_agents,
        "eval_max_steps": args.max_steps,
        "eval_seed": args.seed,
        "n_episodes": args.n_episodes,
        "success_rate": success_rate,
        "mean_fitness": mean_fitness,
        "mean_dist_to_target": mean_dist,
        "mean_max_dist_to_target": mean_max_dist,
        "per_episode": per_episode,
        "formation_targets": formation_targets.tolist(),
    }
    summary_path = out_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()