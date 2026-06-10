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
        help="Fitness function mode.",
    )
    parser.add_argument(
        "--collision-penalty-weight",
        type=float,
        default=1.0,
        help="Weight for collision penalty in navigation_avoidance mode",
    )

    # Task selection
    parser.add_argument(
        "--task",
        type=str,
        default="navigation_static_dynamic_obs",
        choices=["navigation_static_dynamic_obs", "navigation_dynamic_obs"],
        help="VMAS task to use",
    )

    # CMA-ES parameters
    parser.add_argument("--cmaes-gens", type=int, default=60, help="CMA-ES generations")
    parser.add_argument("--pop-size", type=int, default=30, help="CMA-ES population size")
    parser.add_argument("--sigma0", type=float, default=0.5, help="CMA-ES initial step size")
    parser.add_argument("--n-eval-episodes", type=int, default=3, help="Episodes per fitness evaluation")

    # HAN network parameters
    parser.add_argument("--hidden-size", type=int, default=18, help="Hidden layer size (single hidden layer)")
    parser.add_argument("--lr-hebb", type=float, default=0.01, help="Hebbian learning rate")
    parser.add_argument("--weight-init", type=float, default=1.0, help="Weight initialization scale")
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
        return VmasTask.NAVIGATION_STATIC_DYNAMIC_OBS.get_from_yaml()
    elif args.task == "navigation_dynamic_obs":
        return VmasTask.NAVIGATION_DYNAMIC_OBS.get_from_yaml()
    else:
        raise ValueError(f"Unknown task: {args.task}")


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
    print(f"Fitness mode: {args.fitness_mode}")
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
