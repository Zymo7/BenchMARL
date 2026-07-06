"""CMA-ES static-MLP training on flocking with a stationary target.

Parallel of ``run_cmaes_han_flocking_custom.py`` for the
"static-MLP" arm of the HAN-vs-static-MLP comparison. CMA-ES directly
optimizes the flat MLP weight vector (no Hebbian plasticity); all
other machinery — the shared ``flocking_patch`` env, the
``flocking_orbit`` fitness, the rollout loop, the disturbance eval
script — is identical, so the only experimental difference is the
network architecture.

Architecture / param count (matched to HAN's b140e5f5):
  - HAN  : 4 × (10*10 + 10*4) = 560 parameters  (A,B,C,D per weight)
  - static-MLP : 10*40 + 40*4   = 560 parameters  (bias=False)

Run with the same flags as ``run_cmaes_han_flocking_custom.py`` so
the two arms can be compared head-to-head:

    /home/zhaozeming/miniconda3/envs/benchmarl/bin/python \\
        examples/running/run_cmaes_static_mlp_flocking_custom.py \\
        --task flocking \\
        --fitness-mode flocking_orbit \\
        --cmaes-gens 30 --pop-size 30 --sigma0 0.3 \\
        --hidden-size 40 \\
        --orbit-radius 0.7 --orbit-radius-tolerance 0.3 --dt-floor 0.1 \\
        --neighbor-radius 0.5 --safety-distance 0.15 \\
        --target-pos-x 0.0 --target-pos-y 0.0 \\
        --n-final-eval 10 --max-video-frames 400 --fps 20
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

# Apply the shared VMAS flocking monkey-patches (stationary + centered
# target + nearest-neighbor observation). These patches MUST match the
# HAN training / disturbance scripts so the observation layout is
# identical.
from flocking_patch import configure as _flocking_patch_configure  # noqa: F401


from benchmarl.algorithms.cmaes_static_mlp import (
    CmaesStaticMlpConfig,
)
from benchmarl.algorithms.cmaes_static_mlp_optimizer import (
    CmaesStaticMlpOptimizer,
)
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig
from benchmarl.models.static_mlp import StaticMlpConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="CMA-ES static-MLP on flocking with a stationary target"
    )

    # Fitness / scenario
    parser.add_argument(
        "--fitness-mode", type=str, default="flocking_orbit",
        choices=CmaesStaticMlpOptimizer.FITNESS_MODES,
        help="Fitness function mode. Default flocking_orbit.",
    )
    parser.add_argument("--collision-penalty-weight", type=float, default=2.0)
    parser.add_argument("--safety-distance", type=float, default=0.15)
    parser.add_argument("--neighbor-radius", type=float, default=0.5)
    parser.add_argument(
        "--task", type=str, default="flocking",
        choices=["navigation_static_dynamic_obs", "navigation_dynamic_obs",
                 "flocking"],
    )
    parser.add_argument("--n-static-obstacles", type=int, default=0)
    parser.add_argument("--n-dynamic-obstacles", type=int, default=0)
    parser.add_argument("--movement-target-displacement", type=float, default=1.0)
    parser.add_argument("--orbit-radius", type=float, default=0.7)
    parser.add_argument("--orbit-radius-tolerance", type=float, default=0.3)
    parser.add_argument("--dt-floor", type=float, default=0.1)

    # CMA-ES
    parser.add_argument("--cmaes-gens", type=int, default=30)
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--sigma0", type=float, default=0.3)
    parser.add_argument("--n-eval-episodes", type=int, default=2)

    # Static-MLP hyper-params
    parser.add_argument("--hidden-size", type=int, default=40,
                        help="Static-MLP hidden width. 40 gives "
                             "10*40 + 40*4 = 560 weights, matching "
                             "HAN's 4*140=560 ABCD params.")
    parser.add_argument("--bias", action="store_true",
                        help="If set, include bias terms on the Linear "
                             "layers (default off to match HAN's "
                             "bias-less W matrix).")

    # Eval / output
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--experiment-path", type=str, default=None)
    parser.add_argument("--n-final-eval", type=int, default=10)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-video-frames", type=int, default=800)
    parser.add_argument("--target-pos-x", type=float, default=0.0,
                        help="Initial x position of the target. "
                             "VMAS default is 0; override to recenter.")
    parser.add_argument("--target-pos-y", type=float, default=0.0,
                        help="Initial y position of the target. "
                             "VMAS default is -y_dim = -1, which places "
                             "the target at the bottom edge and biases "
                             "the orbit asymmetrically. Set to 0.0 to "
                             "center the target in the world.")
    return parser.parse_args()


args = parse_args()

# Configure the shared flocking monkey-patches with CLI values. MUST
# match what the HAN training and disturbance scripts configure.
_flocking_patch_configure(
    target_pos_x=args.target_pos_x,
    target_pos_y=args.target_pos_y,
    neighbor_radius=args.neighbor_radius,
)


def _get_task():
    if args.task == "navigation_static_dynamic_obs":
        task = VmasTask.NAVIGATION_STATIC_DYNAMIC_OBS.get_from_yaml()
    elif args.task == "navigation_dynamic_obs":
        task = VmasTask.NAVIGATION_DYNAMIC_OBS.get_from_yaml()
    elif args.task == "flocking":
        task = VmasTask.FLOCKING.get_from_yaml()
    else:
        raise ValueError(f"Unknown task: {args.task}")
    if "n_static_obstacles" in task.config:
        task.config["n_static_obstacles"] = args.n_static_obstacles
    if "n_dynamic_obstacles" in task.config:
        task.config["n_dynamic_obstacles"] = args.n_dynamic_obstacles
    if "n_obstacles" in task.config:
        task.config["n_obstacles"] = 0
    return task


def _create_model_config():
    """StaticMlpConfig: bias=False by default for parameter parity with HAN."""
    return StaticMlpConfig(
        num_cells=[args.hidden_size],
        activation_class=torch.nn.Tanh,
        layer_class=torch.nn.Linear,
        bias=args.bias,
    )


def _create_critic_model_config():
    return MlpConfig(
        num_cells=[64, 64],
        activation_class=torch.nn.Tanh,
        layer_class=torch.nn.Linear,
    )


def _setup_experiment_for_cmaes(task, model_config, critic_model_config,
                                output_dir):
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.max_n_iters = 1
    experiment_config.save_folder = str(output_dir)
    experiment_config.loggers = []  # disable wandb
    experiment = Experiment(
        task=task,
        algorithm_config=CmaesStaticMlpConfig.get_from_yaml(),
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
    print("CMA-ES static-MLP — Flocking with stationary target")
    print("=" * 60)
    print(f"  Task: {args.task} | fitness_mode: {args.fitness_mode}")
    print(f"  Flocking patch: stationary target @ "
          f"({args.target_pos_x}, {args.target_pos_y}), "
          f"neighbor_radius={args.neighbor_radius}, obs=10-dim (nn pos+vel)")
    print(f"  orbit_radius={args.orbit_radius}, "
          f"orbit_radius_tolerance={args.orbit_radius_tolerance}, "
          f"dt_floor={args.dt_floor}")

    if args.evaluate_only:
        if args.experiment_path is None:
            raise ValueError("--experiment-path required for evaluate-only")
        exp_path = Path(args.experiment_path)
        slm_dir = exp_path / "static_mlp_results"
        with open(slm_dir / "results.json") as f:
            metadata = json.load(f)
        print(f"  Loaded metadata: total_weights={metadata['total_weights']}, "
              f"fitness={metadata['best_fitness']:.4f}, "
              f"fitness_mode={metadata.get('fitness_mode', 'n/a')}")

        experiment = _setup_experiment_for_cmaes(
            task, model_config, critic_model_config, output_dir)
        policy_path = slm_dir / "policy_state.pt"
        experiment.policy.load_state_dict(
            torch.load(str(policy_path),
                       map_location=experiment.config.train_device))
        static_mlp_model = experiment.algorithm.get_static_mlp_model()
        weights_path = slm_dir / "weights.npy"
        if weights_path.exists():
            weights = np.load(str(weights_path))
            static_mlp_model.set_weights_from_vector(
                torch.tensor(weights,
                             device=experiment.config.train_device)
            )

        optimizer = CmaesStaticMlpOptimizer(
            experiment=experiment,
            static_mlp_model=static_mlp_model,
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

        optimizer.evaluate(
            output_dir=str(exp_path),
            n_episodes=args.n_final_eval,
            fps=args.fps,
            max_video_frames=args.max_video_frames,
        )
    else:
        experiment = _setup_experiment_for_cmaes(
            task, model_config, critic_model_config, output_dir)
        static_mlp_model = experiment.algorithm.get_static_mlp_model()
        if static_mlp_model is None:
            raise RuntimeError("No StaticMlpModel in policy")
        print(f"  StaticMlpModel: {static_mlp_model.total_weights} weights "
              f"(matches HAN's 560 ABCD params)")

        optimizer = CmaesStaticMlpOptimizer(
            experiment=experiment,
            static_mlp_model=static_mlp_model,
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

        best_weights = optimizer.run()
        if (optimizer._current_gen < optimizer.max_gens
                and optimizer._best_weights_so_far is not None):
            optimizer.apply_best_so_far()

        optimizer.save(output_dir=str(experiment.folder_name))
        optimizer.plot_convergence(output_dir=str(experiment.folder_name))
        optimizer.evaluate(
            output_dir=str(experiment.folder_name),
            n_episodes=args.n_final_eval,
            fps=args.fps,
            max_video_frames=args.max_video_frames,
        )