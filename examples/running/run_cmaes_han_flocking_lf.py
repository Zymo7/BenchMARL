"""CMA-ES HAN training on flocking_lf (Leader/Follower arrival).

Variant of the original flocking task where the swarm has two roles:
- Leader agents observe the static target's relative position and can
  move directly toward it.
- Follower agents do NOT see the target; they only observe the
  relative position of their nearest in-range neighbor.

The HAN layer must propagate the leader's knowledge of the target to
the followers so the entire swarm reaches the target. Fitness is the
mean final distance to the target across all agents (lower is better).
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
from benchmarl.models.han import HanConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="CMA-ES HAN on flocking_lf (leader/follower arrival)"
    )

    # Task parameters
    parser.add_argument("--n-leaders", type=int, default=1)
    parser.add_argument("--n-followers", type=int, default=4)
    parser.add_argument(
        "--neighbor-radius", type=float, default=0.5,
        help="Radius for follower (and leader) neighbor detection. "
             "Agents outside this radius are not seen as neighbors.",
    )
    parser.add_argument("--target-pos-x", type=float, default=0.0)
    parser.add_argument("--target-pos-y", type=float, default=0.0)
    parser.add_argument("--min-spawn-dist", type=float, default=0.3)
    parser.add_argument("--spawn-radius", type=float, default=0.9)
    parser.add_argument(
        "--clustered-spawn", action="store_true",
        help="Spawn all agents clustered in a disc instead of the "
             "annulus around target. Used for evaluation with fixed "
             "initial configuration.",
    )
    parser.add_argument("--spawn-cluster-center-x", type=float, default=0.5)
    parser.add_argument("--spawn-cluster-center-y", type=float, default=0.0)
    parser.add_argument("--spawn-cluster-radius", type=float, default=0.15)
    parser.add_argument("--max-steps", type=int, default=1600)

    # Fitness mode
    parser.add_argument(
        "--fitness-mode", type=str, default="flocking_lf_arrival",
        choices=CmaesHanOptimizer.FITNESS_MODES,
        help="Fitness function mode. The default 'flocking_lf_arrival' "
             "minimizes mean final distance to target across all agents.",
    )

    # CMA-ES parameters
    parser.add_argument("--cmaes-gens", type=int, default=30)
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--sigma0", type=float, default=0.3)
    parser.add_argument("--n-eval-episodes", type=int, default=2)

    # HAN network parameters (input is 6 dims: is_leader + target_rel + nn_rel + nn_dist)
    parser.add_argument("--hidden-size", type=int, default=6)
    parser.add_argument("--lr-hebb", type=float, default=0.01)
    parser.add_argument("--weight-init", type=float, default=0.1)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--f-nn", type=int, default=4)
    parser.add_argument("--f-hebb", type=int, default=1)

    # Other CmaesHanOptimizer kwargs (kept for API compatibility; not
    # used by the default 'flocking_lf_arrival' fitness).
    parser.add_argument("--collision-penalty-weight", type=float, default=2.0)
    parser.add_argument("--safety-distance", type=float, default=0.15)
    parser.add_argument("--movement-target-displacement", type=float, default=1.0)
    parser.add_argument("--orbit-radius", type=float, default=0.7)
    parser.add_argument("--orbit-radius-tolerance", type=float, default=0.3)
    parser.add_argument("--dt-floor", type=float, default=0.1)

    # Evaluation
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--experiment-path", type=str, default=None)
    parser.add_argument("--n-final-eval", type=int, default=10)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-video-frames", type=int, default=400)

    return parser.parse_args()


args = parse_args()


def _get_task():
    task = VmasTask.FLOCKING_LF.get_from_yaml()
    task.config["n_leaders"] = args.n_leaders
    task.config["n_followers"] = args.n_followers
    task.config["neighbor_radius"] = args.neighbor_radius
    task.config["target_pos_x"] = args.target_pos_x
    task.config["target_pos_y"] = args.target_pos_y
    task.config["min_spawn_dist"] = args.min_spawn_dist
    task.config["spawn_radius"] = args.spawn_radius
    task.config["clustered_spawn"] = args.clustered_spawn
    task.config["spawn_cluster_center_x"] = args.spawn_cluster_center_x
    task.config["spawn_cluster_center_y"] = args.spawn_cluster_center_y
    task.config["spawn_cluster_radius"] = args.spawn_cluster_radius
    task.config["max_steps"] = args.max_steps
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


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    task = _get_task()
    model_config = _create_model_config()
    critic_model_config = _create_critic_model_config()

    print("=" * 60)
    print("CMA-ES HAN — Flocking LF (Leader/Follower arrival)")
    print("=" * 60)
    print(f"Task: flocking_lf | n_leaders={args.n_leaders}, n_followers={args.n_followers}")
    print(f"  neighbor_radius={args.neighbor_radius}, "
          f"target=({args.target_pos_x}, {args.target_pos_y})")
    if args.clustered_spawn:
        print(f"  spawn: CLUSTERED at ({args.spawn_cluster_center_x}, "
              f"{args.spawn_cluster_center_y}), "
              f"radius={args.spawn_cluster_radius}, max_steps={args.max_steps}")
    else:
        print(f"  spawn: annulus [{args.min_spawn_dist}, {args.spawn_radius}] "
              f"around target, max_steps={args.max_steps}")
    print(f"Fitness mode: {args.fitness_mode}")
    print(f"HAN: hidden={args.hidden_size}, window={args.window_size}, "
          f"f_nn={args.f_nn}, f_hebb={args.f_hebb}")
    print(f"CMA-ES: pop={args.pop_size}, gens={args.cmaes_gens}, sigma0={args.sigma0}")
    print(f"Eval episodes per candidate: {args.n_eval_episodes}")
    print()

    if args.evaluate_only:
        if args.experiment_path is None:
            raise ValueError("--experiment-path required for evaluate-only")
        exp_path = Path(args.experiment_path)
        han_dir = exp_path / "han_results"
        with open(han_dir / "results.json") as f:
            metadata = json.load(f)
        print(f"Loaded metadata: {metadata['n_layers']} layers, "
              f"fitness={metadata['best_fitness']}")

        experiment = _setup_experiment_for_cmaes(
            task, model_config, critic_model_config, output_dir
        )
        policy_path = han_dir / "policy_state.pt"
        experiment.policy.load_state_dict(
            torch.load(str(policy_path),
                       map_location=experiment.config.train_device)
        )
        han_model = experiment.algorithm.get_han_model()
        abcd_path = han_dir / "abcd_params.npy"
        if abcd_path.exists():
            abcd = np.load(str(abcd_path))
            han_model.set_abcd_from_vector(
                torch.tensor(abcd, device=experiment.config.train_device)
            )
            han_model.reset_all_weights()

        optimizer = CmaesHanOptimizer(
            experiment=experiment,
            han_model=han_model,
            fitness_mode=args.fitness_mode,
            pop_size=1, max_gens=0, n_eval_episodes=1,
            device=experiment.config.train_device,
            collision_penalty_weight=args.collision_penalty_weight,
            safety_distance=args.safety_distance,
            neighbor_radius=args.neighbor_radius,
            movement_target_displacement=args.movement_target_displacement,
            orbit_radius=args.orbit_radius,
            orbit_radius_tolerance=args.orbit_radius_tolerance,
            dt_floor=args.dt_floor,
        )
        if abcd_path.exists():
            optimizer._best_abcd_so_far = np.load(str(abcd_path))

        optimizer.evaluate(
            output_dir=str(exp_path),
            n_episodes=args.n_final_eval,
            fps=args.fps,
            max_video_frames=args.max_video_frames,
        )
    else:
        experiment = _setup_experiment_for_cmaes(
            task, model_config, critic_model_config, output_dir
        )
        han_model = experiment.algorithm.get_han_model()
        if han_model is None:
            raise RuntimeError("No HanModel in policy")

        layers = han_model.get_all_han_layers()
        print(f"HanModel: {len(layers)} layers, {han_model.total_abcd_params} ABCD params")
        for i, layer in enumerate(layers):
            print(f"  Layer {i}: {layer.in_features} -> {layer.out_features} "
                  f"({layer.num_abcd_params} ABCD params)")
        print()

        optimizer = CmaesHanOptimizer(
            experiment=experiment,
            han_model=han_model,
            fitness_mode=args.fitness_mode,
            pop_size=args.pop_size,
            sigma0=args.sigma0,
            max_gens=args.cmaes_gens,
            n_eval_episodes=args.n_eval_episodes,
            device=experiment.config.train_device,
            collision_penalty_weight=args.collision_penalty_weight,
            safety_distance=args.safety_distance,
            neighbor_radius=args.neighbor_radius,
            movement_target_displacement=args.movement_target_displacement,
            orbit_radius=args.orbit_radius,
            orbit_radius_tolerance=args.orbit_radius_tolerance,
            dt_floor=args.dt_floor,
        )

        best_abcd = optimizer.run()

        if (optimizer._current_gen < optimizer.max_gens
                and optimizer._best_abcd_so_far is not None):
            optimizer.apply_best_so_far()
            best_abcd = optimizer._best_abcd_so_far

        print(f"\nTraining complete. Best ABCD shape: {best_abcd.shape}")

        optimizer.save(output_dir=str(experiment.folder_name))
        optimizer.plot_convergence(output_dir=str(experiment.folder_name))
        optimizer.evaluate(
            output_dir=str(experiment.folder_name),
            n_episodes=args.n_final_eval,
            fps=args.fps,
            max_video_frames=args.max_video_frames,
        )
