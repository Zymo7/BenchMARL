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
    parser = argparse.ArgumentParser(description="CMA-ES Hebbian Attractor Network (HAN) Training on Navigation")

    # Fitness mode
    parser.add_argument(
        "--fitness-mode",
        type=str,
        default="navigation_v2",
        choices=CmaesHanOptimizer.FITNESS_MODES,
        help="Fitness function mode. 'navigation_avoidance_v2' adds "
             "inter-agent collision penalty based on per-agent collision "
             "ratios (fraction of steps each agent spent within "
             "safety_distance of a neighbor within neighbor_radius). "
             "'flocking_global' implements Ramos 2019 Global flocking "
             "(alignment + cohesion + separation). 'flocking_orbit' "
             "adds a target-orbit term (radial-tangential alignment + "
             "distance band around orbit_radius).",
    )
    parser.add_argument(
        "--collision-penalty-weight",
        type=float,
        default=2.0,
        help="Weight for collision penalty. Used in 'navigation_avoidance' "
             "(per-step obstacle collisions) and in 'navigation_avoidance_v2' "
             "(per-agent mean collision ratio).",
    )
    parser.add_argument(
        "--safety-distance",
        type=float,
        default=0.15,
        help="Inter-agent distance below which two agents are considered "
             "in collision. Used by 'navigation_avoidance_v2'.",
    )
    parser.add_argument(
        "--neighbor-radius",
        type=float,
        default=0.5,
        help="Radius within which other agents are considered neighbors. "
             "Only neighbors within this radius are checked for collision. "
             "Used by 'navigation_avoidance_v2'.",
    )

    # Task selection
    parser.add_argument(
        "--task",
        type=str,
        default="navigation_static_dynamic_obs",
        choices=["navigation_static_dynamic_obs", "navigation_dynamic_obs", "flocking"],
        help="VMAS task to use. 'flocking' uses VMAS's built-in flocking "
             "scenario; pair with --fitness-mode flocking_global or "
             "flocking_orbit.",
    )
    parser.add_argument(
        "--n-static-obstacles",
        type=int,
        default=2,
        help="Number of static obstacles. Set to 0 to disable environment "
             "obstacles (so only inter-agent collisions matter).",
    )
    parser.add_argument(
        "--n-dynamic-obstacles",
        type=int,
        default=0,
        help="Number of dynamic obstacles. Set to 0 to disable.",
    )
    parser.add_argument(
        "--movement-target-displacement",
        type=float,
        default=1.0,
        help="Target average displacement (in VMAS world units) used to "
             "compute the movement bonus M in the flocking_global fitness. "
             "Default 1.0 is a reasonable scale for the VMAS flocking world "
             "(x_dim=y_dim=1, max possible displacement ~sqrt(2)).",
    )
    parser.add_argument(
        "--orbit-radius",
        type=float,
        default=0.7,
        help="Target radius (VMAS world units) of the orbit each agent "
             "should keep around the flocking target. Used by the "
             "'flocking_orbit' fitness as the center of the Gaussian "
             "distance band Dt. Default 0.7 fits the VMAS flocking world "
             "(agent-to-target distance is typically 1..2 with target at "
             "(0,-1)).",
    )
    parser.add_argument(
        "--orbit-radius-tolerance",
        type=float,
        default=0.3,
        help="Standard deviation (VMAS world units) of the Gaussian "
             "distance band Dt in 'flocking_orbit'. Larger = more "
             "lenient (agents can stray further from orbit_radius).",
    )
    parser.add_argument(
        "--dt-floor",
        type=float,
        default=0.1,
        help="Lower bound for Dt in 'flocking_orbit'. Prevents the "
             "distance term from going to zero when all agents are far "
             "from orbit_radius (keeps some signal flowing back to CMA-ES).",
    )

    # CMA-ES parameters
    parser.add_argument("--cmaes-gens", type=int, default=15, help="CMA-ES generations")
    parser.add_argument("--pop-size", type=int, default=30, help="CMA-ES population size")
    parser.add_argument("--sigma0", type=float, default=0.5, help="CMA-ES initial step size")
    parser.add_argument("--n-eval-episodes", type=int, default=3, help="Episodes per fitness evaluation")

    # HAN network parameters
    parser.add_argument("--hidden-size", type=int, default=18, help="Hidden layer size (single hidden layer)")
    parser.add_argument("--lr-hebb", type=float, default=0.01, help="Hebbian learning rate")
    parser.add_argument("--weight-init", type=float, default=0.1, help="Weight initialization scale")
    parser.add_argument("--window-size", type=int, default=10,
                        help="Sliding-window length M for time-averaged ABCD update")
    parser.add_argument("--f-nn", type=int, default=4,
                        help="Action-inference frequency (env steps per f_nn unit)")
    parser.add_argument("--f-hebb", type=int, default=1,
                        help="Weight-update frequency (env steps per f_hebb unit)")

    # Evaluation
    parser.add_argument("--evaluate-only", action="store_true", help="Skip training, only evaluate")
    parser.add_argument("--experiment-path", type=str, default=None, help="Existing experiment folder")
    parser.add_argument("--n-final-eval", type=int, default=10, help="Number of final evaluation episodes")
    parser.add_argument("--fps", type=int, default=20, help="Video fps (higher -> shorter playback)")
    parser.add_argument("--max-video-frames", type=int, default=400, help="Cap frames recorded per episode")

    return parser.parse_args()


args = parse_args()


def _get_task():
    if args.task == "navigation_static_dynamic_obs":
        task = VmasTask.NAVIGATION_STATIC_DYNAMIC_OBS.get_from_yaml()
    elif args.task == "navigation_dynamic_obs":
        task = VmasTask.NAVIGATION_DYNAMIC_OBS.get_from_yaml()
    elif args.task == "flocking":
        task = VmasTask.FLOCKING.get_from_yaml()
    else:
        raise ValueError(f"Unknown task: {args.task}")
    # Apply obstacle-count overrides via task.config dict.
    if "n_static_obstacles" in task.config:
        task.config["n_static_obstacles"] = args.n_static_obstacles
    if "n_dynamic_obstacles" in task.config:
        task.config["n_dynamic_obstacles"] = args.n_dynamic_obstacles
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
    """Set up the experiment infrastructure without running PPO training."""
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
    print("CMA-ES Hebbian Attractor Network (HAN) Training")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"  n_static_obstacles={args.n_static_obstacles}, n_dynamic_obstacles={args.n_dynamic_obstacles}")
    print(f"Fitness mode: {args.fitness_mode}")
    if args.fitness_mode in ("navigation_avoidance", "navigation_avoidance_v2"):
        print(f"  collision_penalty_weight={args.collision_penalty_weight}")
    if args.fitness_mode == "navigation_avoidance_v2":
        print(f"  safety_distance={args.safety_distance}, neighbor_radius={args.neighbor_radius}")
    print(f"Network: input -> {args.hidden_size} -> output (single hidden layer)")
    print(f"HAN: window_size={args.window_size}, f_nn={args.f_nn}, f_hebb={args.f_hebb}, "
          f"update_interval={args.f_nn // args.f_hebb if args.f_hebb > 0 and args.f_nn >= args.f_hebb else 'disabled'}")
    print(f"CMA-ES: pop={args.pop_size}, gens={args.cmaes_gens}, sigma0={args.sigma0}")
    print(f"Eval episodes per candidate: {args.n_eval_episodes}")
    print()

    if args.evaluate_only:
        if args.experiment_path is None:
            raise ValueError("--experiment-path is required for evaluate-only mode")

        exp_path = Path(args.experiment_path)
        han_dir = exp_path / "han_results"

        metadata_path = han_dir / "results.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        print(f"Loaded metadata: {metadata['n_layers']} layers, fitness={metadata['best_fitness']}")

        experiment = _setup_experiment_for_cmaes(task, model_config, critic_model_config, output_dir)

        policy_path = han_dir / "policy_state.pt"
        if not policy_path.exists():
            raise FileNotFoundError(f"policy_state.pt not found at {policy_path}")
        experiment.policy.load_state_dict(
            torch.load(str(policy_path), map_location=experiment.config.train_device)
        )

        han_model = experiment.algorithm.get_han_model()
        abcd_path = han_dir / "abcd_params.npy"
        if abcd_path.exists():
            abcd = np.load(str(abcd_path))
            han_model.set_abcd_from_vector(torch.tensor(abcd, device=experiment.config.train_device))
            han_model.reset_all_weights()
            print(f"Loaded ABCD params: {len(abcd)} parameters")

        optimizer = CmaesHanOptimizer(
            experiment=experiment,
            han_model=han_model,
            fitness_mode=args.fitness_mode,
            pop_size=1,
            max_gens=0,
            n_eval_episodes=1,
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
        print("Initializing experiment (environment, policy, algorithm)...")
        experiment = _setup_experiment_for_cmaes(task, model_config, critic_model_config, output_dir)

        han_model = experiment.algorithm.get_han_model()
        if han_model is None:
            raise RuntimeError("Could not find HanModel in the policy")

        layers = han_model.get_all_han_layers()
        print(f"\nHanModel found with {len(layers)} layers:")
        for i, layer in enumerate(layers):
            print(f"  Layer {i}: {layer.in_features} -> {layer.out_features} "
                  f"({layer.num_abcd_params} ABCD params, window={layer.window_size})")
        print(f"  Total ABCD parameters: {han_model.total_abcd_params}")
        print(f"  Tick: update_interval = {han_model._update_interval} env steps")
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

        if optimizer._current_gen < optimizer.max_gens and optimizer._best_abcd_so_far is not None:
            print(f"\nCMA-ES interrupted at gen {optimizer._current_gen}/{optimizer.max_gens}")
            optimizer.apply_best_so_far()
            best_abcd = optimizer._best_abcd_so_far

        print(f"\nTraining complete. Best ABCD shape: {best_abcd.shape}")

        print("\nSaving results...")
        optimizer.save(output_dir=str(experiment.folder_name))

        optimizer.plot_convergence(output_dir=str(experiment.folder_name))

        print(f"\nFinal evaluation ({args.n_final_eval} episodes)...")
        optimizer.evaluate(
            output_dir=str(experiment.folder_name),
            n_episodes=args.n_final_eval,
            fps=args.fps,
            max_video_frames=args.max_video_frames,
        )
