"""CMA-ES HGN training on the formation-control task.

Spawns a swarm clustered in a small region around the origin; the agents
must reach and stabilize at a fixed target formation (circle / line / V /
grid) using a Hebbian Graph Network actor (shared plastic edge matrix +
per-step message passing + per-agent plastic node update + per-agent
plastic output head). CMA-ES optimizes the ABCD Hebbian parameters.

Run::

    python examples/running/run_cmaes_hgn_formation.py \\
        --formation-type circle --n-agents 6 \\
        --cmaes-gens 15 --pop-size 30 --n-eval-episodes 3
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmarl.algorithms.cmaes_han import CmaesHanConfig
from benchmarl.algorithms.cmaes_han_optimizer import CmaesHanOptimizer
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig
from benchmarl.models.hgn import HgnConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="CMA-ES HGN — Multi-agent formation control"
    )
    # Formation task configuration
    parser.add_argument("--n-agents", type=int, default=6)
    parser.add_argument("--formation-type", type=str, default="circle",
                        choices=["circle", "line", "v", "grid"])
    parser.add_argument("--formation-radius", type=float, default=0.6)
    parser.add_argument("--spawn-radius", type=float, default=1.0)
    parser.add_argument("--spawn-cluster-radius", type=float, default=0.25)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--n-obstacles", type=int, default=0,
                        help="Number of static obstacles (Phase 2).")
    parser.add_argument("--obstacle-radius", type=float, default=0.10)
    parser.add_argument("--moving-target", action="store_true",
                        help="Translate the formation over the episode (Phase 3).")
    parser.add_argument("--target-speed", type=float, default=0.1)

    # Fitness configuration
    parser.add_argument("--fitness-mode", type=str,
                        default="hgn_formation_v1",
                        choices=CmaesHanOptimizer.FITNESS_MODES)
    parser.add_argument("--success-reward", type=float, default=5.0)
    parser.add_argument("--final-weight", type=float, default=1.0)
    parser.add_argument("--formation-collision-penalty", type=float, default=2.0)
    parser.add_argument("--formation-timeout-penalty", type=float, default=2.0)
    parser.add_argument("--formation-reach-radius", type=float, default=0.10)

    # CMA-ES parameters
    parser.add_argument("--cmaes-gens", type=int, default=15)
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--sigma0", type=float, default=0.5)
    parser.add_argument("--n-eval-episodes", type=int, default=3)

    # HGN network parameters
    parser.add_argument("--d-h", type=int, default=18)
    parser.add_argument("--n-message-steps", type=int, default=2)
    parser.add_argument("--topology", type=str, default="full",
                        choices=["full", "from_pos"])
    parser.add_argument("--edge-radius", type=float, default=None)
    parser.add_argument("--lr-hebb", type=float, default=0.01)
    parser.add_argument("--weight-init", type=float, default=1.0)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--f-nn", type=int, default=4)
    parser.add_argument("--f-hebb", type=int, default=1)

    # Compatibility kwargs (unused by formation, listed for API parity)
    parser.add_argument("--collision-penalty-weight", type=float, default=2.0)
    parser.add_argument("--safety-distance", type=float, default=0.15)
    parser.add_argument("--neighbor-radius", type=float, default=0.5)
    parser.add_argument("--movement-target-displacement", type=float, default=1.0)
    parser.add_argument("--orbit-radius", type=float, default=0.7)
    parser.add_argument("--orbit-radius-tolerance", type=float, default=0.3)
    parser.add_argument("--dt-floor", type=float, default=0.1)

    # Evaluation
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--experiment-path", type=str, default=None)
    parser.add_argument("--n-final-eval", type=int, default=10)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-video-frames", type=int, default=200)

    return parser.parse_args()


args = parse_args()


def _get_task():
    task = VmasTask.BENCHMARL_HGN_FORMATION.get_from_yaml()
    task.config["n_agents"] = args.n_agents
    task.config["formation_type"] = args.formation_type
    task.config["formation_radius"] = args.formation_radius
    task.config["spawn_radius"] = args.spawn_radius
    task.config["spawn_cluster_radius"] = args.spawn_cluster_radius
    task.config["max_steps"] = args.max_steps
    task.config["n_obstacles"] = args.n_obstacles
    task.config["obstacle_radius"] = args.obstacle_radius
    task.config["moving_target"] = args.moving_target
    task.config["target_speed"] = args.target_speed
    return task


def _create_model_config():
    return HgnConfig(
        d_h=args.d_h,
        n_message_steps=args.n_message_steps,
        topology=args.topology,
        edge_radius=args.edge_radius,
        lr_hebb=args.lr_hebb,
        weight_init=args.weight_init,
        window_size=args.window_size,
        f_nn=args.f_nn,
        f_hebb=args.f_hebb,
        activation_class=torch.nn.Tanh,
    )


def _setup_experiment_for_cmaes():
    """Build a BenchMARL Experiment wired with HgnConfig + CmaesHan."""
    task = _get_task()
    algorithm_config = CmaesHanConfig.get_from_yaml()
    model_config = _create_model_config()
    critic_model_config = MlpConfig(
        num_cells=[64, 64],
        activation_class=torch.nn.Tanh,
        layer_class=torch.nn.Linear,
    )

    experiment_config = ExperimentConfig.get_from_yaml()
    # CMA-ES drives training, not PPO; we set max_n_iters=1 so the
    # experiment framework runs a single dummy PPO-style pass and then
    # the CmaesHanOptimizer takes over.
    experiment_config.max_n_iters = 1

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
    )
    experiment._setup()
    return experiment


def _load_experiment_for_eval(experiment_path: str):
    """Reload a saved experiment from disk for evaluate-only mode."""
    exp_path = Path(experiment_path)
    config_path = exp_path / "experiment_config.json"
    task_config_path = exp_path / "task_config.json"
    with open(config_path, "r") as f:
        experiment_cfg_dict = json.load(f)
    with open(task_config_path, "r") as f:
        task_cfg_dict = json.load(f)

    task = VmasTask.BENCHMARL_HGN_FORMATION
    task_obj = task.get_task(config=task_cfg_dict)

    algorithm_config = CmaesHanConfig.get_from_yaml()
    model_config = HgnConfig(
        d_h=args.d_h,
        n_message_steps=args.n_message_steps,
        topology=args.topology,
        edge_radius=args.edge_radius,
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

    experiment_config = ExperimentConfig(**experiment_cfg_dict)
    experiment_config.max_n_iters = 1

    experiment = Experiment(
        task=task_obj,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
    )
    experiment._setup()

    # Reload ABCD and policy weights.
    policy_state = torch.load(exp_path / "han_results" / "policy_state.pt",
                              map_location=experiment.config.train_device)
    experiment.policy.load_state_dict(policy_state)
    return experiment


def main():
    if args.evaluate_only:
        experiment = _load_experiment_for_eval(args.experiment_path)
    else:
        experiment = _setup_experiment_for_cmaes()

    hgn_model = experiment.algorithm.get_hgn_model()
    if hgn_model is None:
        raise RuntimeError(
            "HGN model not found in the policy; did the registration succeed?"
        )

    print(f"HGN model: total_abcd_params={hgn_model.total_abcd_params}")
    print(f"HGN model: layers={[type(l).__name__ for l in hgn_model.get_all_han_layers()]}")

    optimizer = CmaesHanOptimizer(
        experiment=experiment,
        han_model=hgn_model,
        fitness_mode=args.fitness_mode,
        pop_size=args.pop_size,
        sigma0=args.sigma0,
        max_gens=args.cmaes_gens,
        n_eval_episodes=args.n_eval_episodes,
        device=experiment.config.train_device,
        success_reward=args.success_reward,
        final_weight=args.final_weight,
        formation_collision_penalty=args.formation_collision_penalty,
        formation_timeout_penalty=args.formation_timeout_penalty,
        formation_reach_radius=args.formation_reach_radius,
        # Compatibility kwargs (unused for formation fitness)
        collision_penalty_weight=args.collision_penalty_weight,
        safety_distance=args.safety_distance,
        neighbor_radius=args.neighbor_radius,
        movement_target_displacement=args.movement_target_displacement,
        orbit_radius=args.orbit_radius,
        orbit_radius_tolerance=args.orbit_radius_tolerance,
        dt_floor=args.dt_floor,
    )

    if args.evaluate_only:
        output_dir = args.experiment_path
    else:
        best_abcd = optimizer.run()
        output_dir = str(experiment.folder_name)
        optimizer.save(output_dir=output_dir)
        optimizer.plot_convergence(output_dir=output_dir)

    optimizer.evaluate(
        output_dir=output_dir,
        n_episodes=args.n_final_eval,
        fps=args.fps,
        max_video_frames=args.max_video_frames,
    )


if __name__ == "__main__":
    main()