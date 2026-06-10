import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from benchmarl.algorithms.cmaes_hebbian import CmaesHebbianConfig
from benchmarl.algorithms.cmaes_full_hebbian_optimizer import CmaesFullHebbianOptimizer
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig
from benchmarl.models.full_hebbian import FullHebbianConfig


def parse_args():
    parser = argparse.ArgumentParser(description="CMA-ES Full Hebbian Training on Navigation")

    # Fitness mode
    parser.add_argument(
        "--fitness-mode",
        type=str,
        default="navigation_v2",
        choices=CmaesFullHebbianOptimizer.FITNESS_MODES,
        help="Fitness function mode. 'navigation': raw env reward. "
             "'navigation_avoidance': env reward minus a collision penalty. "
             "'navigation_v2': progress (initial-vs-final goal distance, normalized) "
             "+ sparse success bonus + small final-distance term. Recommended for CMA-ES.",
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
    parser.add_argument("--cmaes-gens", type=int, default=50, help="CMA-ES generations")
    parser.add_argument("--pop-size", type=int, default=30, help="CMA-ES population size")
    parser.add_argument("--sigma0", type=float, default=0.5, help="CMA-ES initial step size")
    parser.add_argument("--n-eval-episodes", type=int, default=3, help="Episodes per fitness evaluation")

    # Network parameters
    parser.add_argument("--hidden-size", type=int, default=9, help="Hidden layer size")
    parser.add_argument("--lr-hebb", type=float, default=0.01, help="Hebbian learning rate")
    parser.add_argument("--weight-init", type=float, default=1.0, help="Weight initialization scale")
    parser.add_argument("--w-max", type=float, default=1.0, help="Per-element |W| clip after each Hebbian update (<=0 disables)")

    # Evaluation
    parser.add_argument("--evaluate-only", action="store_true", help="Skip training, only evaluate")
    parser.add_argument("--experiment-path", type=str, default=None, help="Existing experiment folder")
    parser.add_argument("--n-final-eval", type=int, default=10, help="Number of final evaluation episodes")
    parser.add_argument("--fps", type=int, default=20, help="Video fps (higher -> shorter playback)")
    parser.add_argument("--max-video-frames", type=int, default=400, help="Cap frames recorded per episode; "
                        "env keeps running to compute reward, only frame capture is truncated")

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
    return FullHebbianConfig(
        hidden_size=args.hidden_size,
        lr_hebb=args.lr_hebb,
        weight_init=args.weight_init,
        w_max=args.w_max,
        activation_class=torch.nn.Tanh,
    )


def _create_critic_model_config():
    return MlpConfig(
        num_cells=[64, 64],
        activation_class=torch.nn.Tanh,
        layer_class=torch.nn.Linear,
    )


def _setup_experiment_for_cmaes(task, model_config, critic_model_config, output_dir):
    """Set up the experiment infrastructure without running PPO training.

    We create the experiment, call _setup() to initialize env/algorithm/policy,
    but skip the training loop entirely and go straight to CMA-ES.
    """
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.max_n_iters = 1
    experiment_config.save_folder = str(output_dir)

    experiment = Experiment(
        task=task,
        algorithm_config=CmaesHebbianConfig.get_from_yaml(),
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
    )

    # Call _setup() which initializes env, algorithm, policy, and collector
    # but does NOT start training
    experiment._setup()

    return experiment


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    task = _get_task()
    model_config = _create_model_config()
    critic_model_config = _create_critic_model_config()

    print("=" * 60)
    print("CMA-ES Full Hebbian Network Training")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Fitness mode: {args.fitness_mode}")
    print(f"Network: input -> {args.hidden_size} -> {args.hidden_size} -> output")
    print(f"CMA-ES: pop={args.pop_size}, gens={args.cmaes_gens}, sigma0={args.sigma0}")
    print(f"Eval episodes per candidate: {args.n_eval_episodes}")
    print()

    if args.evaluate_only:
        # ========== Evaluate Only Mode ==========
        if args.experiment_path is None:
            raise ValueError("--experiment-path is required for evaluate-only mode")

        exp_path = Path(args.experiment_path)
        hebbian_dir = exp_path / "hebbian_results"

        metadata_path = hebbian_dir / "results.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        print(f"Loaded metadata: {metadata['n_layers']} layers, fitness={metadata['best_fitness']}")

        experiment = _setup_experiment_for_cmaes(task, model_config, critic_model_config, output_dir)

        # Load policy
        policy_path = hebbian_dir / "policy_state.pt"
        if not policy_path.exists():
            raise FileNotFoundError(f"policy_state.pt not found at {policy_path}")
        experiment.policy.load_state_dict(
            torch.load(str(policy_path), map_location=experiment.config.train_device)
        )

        # Load ABCD params
        full_hebbian = experiment.algorithm.get_full_hebbian_model()
        abcd_path = hebbian_dir / "abcd_params.npy"
        if abcd_path.exists():
            abcd = np.load(str(abcd_path))
            full_hebbian.set_abcd_from_vector(torch.tensor(abcd, device=experiment.config.train_device))
            full_hebbian.reset_all_weights()
            print(f"Loaded ABCD params: {len(abcd)} parameters")

        optimizer = CmaesFullHebbianOptimizer(
            experiment=experiment,
            full_hebbian_model=full_hebbian,
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
        # ========== Full Training Mode ==========
        print("Initializing experiment (environment, policy, algorithm)...")
        experiment = _setup_experiment_for_cmaes(task, model_config, critic_model_config, output_dir)

        # Get the FullHebbian model
        full_hebbian = experiment.algorithm.get_full_hebbian_model()
        if full_hebbian is None:
            raise RuntimeError("Could not find FullHebbianModel in the policy")

        layers = full_hebbian.get_all_hebbian_layers()
        print(f"\nFullHebbianModel found with {len(layers)} layers:")
        for i, layer in enumerate(layers):
            print(f"  Layer {i}: {layer.in_features} -> {layer.out_features} ({layer.num_abcd_params} ABCD params)")
        print(f"  Total ABCD parameters: {full_hebbian.total_abcd_params}")
        print()

        # Create CMA-ES optimizer
        optimizer = CmaesFullHebbianOptimizer(
            experiment=experiment,
            full_hebbian_model=full_hebbian,
            fitness_mode=args.fitness_mode,
            pop_size=args.pop_size,
            sigma0=args.sigma0,
            max_gens=args.cmaes_gens,
            n_eval_episodes=args.n_eval_episodes,
            device=experiment.config.train_device,
            collision_penalty_weight=args.collision_penalty_weight,
        )

        # Run CMA-ES optimization
        best_abcd = optimizer.run()

        # Handle mid-interrupt
        if optimizer._current_gen < optimizer.max_gens and optimizer._best_abcd_so_far is not None:
            print(f"\nCMA-ES interrupted at gen {optimizer._current_gen}/{optimizer.max_gens}")
            optimizer.apply_best_so_far()
            best_abcd = optimizer._best_abcd_so_far

        print(f"\nTraining complete. Best ABCD shape: {best_abcd.shape}")

        # Save results
        print("\nSaving results...")
        optimizer.save(output_dir=str(experiment.folder_name))

        # Convergence plot
        optimizer.plot_convergence(output_dir=str(experiment.folder_name))

        # Final evaluation
        print(f"\nFinal evaluation ({args.n_final_eval} episodes)...")
        optimizer.evaluate(
            output_dir=str(experiment.folder_name),
            n_episodes=args.n_final_eval,
            fps=args.fps,
            max_video_frames=args.max_video_frames,
        )
