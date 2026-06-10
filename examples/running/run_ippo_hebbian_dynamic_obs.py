import argparse
import json
from pathlib import Path

import numpy as np

import torch
from benchmarl.algorithms.ippo_hebbian import IppoHebbianConfig
from benchmarl.algorithms.cmaes_optimizer import CmaesHebbianOptimizer
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig
from benchmarl.models.hebbian import HebbianConfig
from benchmarl.models.common import SequenceModelConfig


def parse_args():
    parser = argparse.ArgumentParser(description="IPPO-Hebbian on Navigation Dynamic Obs")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip training, only run evaluation (use with --experiment-path)",
    )
    parser.add_argument(
        "--experiment-path",
        type=str,
        default=None,
        help="Path to an existing experiment folder (for evaluate-only mode)",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--max-n-iters",
        type=int,
        default=200,
        help="Max training iterations for PPO phase (default: 200)",
    )
    parser.add_argument(
        "--cmaes-gens",
        type=int,
        default=25,
        help="Max CMA-ES generations (default: 30)",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=30,
        help="CMA-ES population size (default: 30)",
    )
    return parser.parse_args()


args = parse_args()


def _create_model_configs():
    """Create model configurations for IPPO-Hebbian."""
    model_config = SequenceModelConfig(
        model_configs=[
            MlpConfig(num_cells=[64, 64], activation_class=torch.nn.Tanh, layer_class=torch.nn.Linear),
            HebbianConfig(lr_hebb=0.01, weight_init=1.0),
        ],
        intermediate_sizes=[64],
    )
    critic_model_config = MlpConfig(
        num_cells=[64, 64],
        activation_class=torch.nn.Tanh,
        layer_class=torch.nn.Linear,
    )
    return model_config, critic_model_config


def _create_experiment(task, model_config, critic_model_config, output_dir, max_n_iters=200):
    """Create and return an Experiment instance."""
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.max_n_iters = max_n_iters
    experiment_config.save_folder = str(output_dir)
    return Experiment(
        task=task,
        algorithm_config=IppoHebbianConfig.get_from_yaml(),
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
    )


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    if args.evaluate_only:
        # ========== Evaluate Only Mode ==========
        if args.experiment_path is None:
            raise ValueError("--experiment-path is required for evaluate-only mode")

        exp_path = Path(args.experiment_path)
        hebbian_dir = exp_path / "hebbian_results"

        print("=" * 60)
        print("Evaluate-Only Mode")
        print("=" * 60)
        print(f"Loading experiment from: {exp_path}")

        # Load metadata
        metadata_path = hebbian_dir / "results.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        print(f"Hebbian metadata: in={metadata['in_features']}, out={metadata['out_features']}")

        # Recreate experiment with same config
        task = VmasTask.NAVIGATION_DYNAMIC_OBS.get_from_yaml()
        model_config, critic_model_config = _create_model_configs()
        experiment = _create_experiment(task, model_config, critic_model_config, output_dir)

        # Load policy state dict
        policy_path = hebbian_dir / "policy_state.pt"
        if not policy_path.exists():
            raise FileNotFoundError(
                f"policy_state.pt not found at {policy_path}.\n"
                "This file is saved automatically during training.\n"
                "For old experiments without it, please re-run training first."
            )
        policy_state = torch.load(str(policy_path), map_location=experiment.config.train_device)
        experiment.policy.load_state_dict(policy_state)
        print("Loaded full policy from policy_state.pt")

        # Get Hebbian layer and ensure ABCD params are loaded
        hebbian_layer = experiment.algorithm.get_hebbian_layer()
        abcd_path = hebbian_dir / "abcd_params.npy"
        if policy_path.exists() and abcd_path.exists():
            # Full policy loaded; also set ABCD for optimizer tracking
            abcd = np.load(str(abcd_path))
            hebbian_layer.set_abcd_from_vector(torch.tensor(abcd, device=experiment.config.train_device))
            hebbian_layer.reset_weights()
            print("Loaded ABCD params from abcd_params.npy")

        # Create optimizer (for evaluation) and run
        optimizer = CmaesHebbianOptimizer(
            experiment=experiment,
            hebbian_layer=hebbian_layer,
            pop_size=1,
            max_gens=0,
            n_eval_episodes=1,
            device=experiment.config.train_device,
        )
        if abcd_path.exists():
            abcd = np.load(str(abcd_path))
            optimizer._best_abcd_so_far = abcd

        print(f"\nRunning evaluation with {args.n_eval_episodes} episodes...")
        optimizer.evaluate(
            output_dir=str(exp_path),
            n_episodes=args.n_eval_episodes,
        )
        print("Evaluation complete!")

    else:
        # ========== Full Training Mode ==========
        print("=" * 60)
        print("IPPO-Hebbian Training on Navigation Dynamic Obs")
        print("=" * 60)
        print(f"PPO max_n_iters: {args.max_n_iters}, CMA-ES max_gens: {args.cmaes_gens}")

        # Create experiment
        task = VmasTask.NAVIGATION_DYNAMIC_OBS.get_from_yaml()
        model_config, critic_model_config = _create_model_configs()
        experiment = _create_experiment(
            task, model_config, critic_model_config, output_dir, max_n_iters=args.max_n_iters
        )

        # ========================================
        # Phase 1: PPO training of MLP layers
        # ========================================
        print("Phase 1: Training MLP layers with PPO")
        print("=" * 60)
        experiment.run()

        # ========================================
        # Phase 2: CMA-ES optimization of ABCD
        # ========================================
        print("\n" + "=" * 60)
        print("Phase 2: Optimizing Hebbian ABCD parameters with CMA-ES")
        print("=" * 60)

        # Extract Hebbian layer from the trained algorithm
        hebbian_layer = experiment.algorithm.get_hebbian_layer()
        if hebbian_layer is None:
            raise RuntimeError("Could not find Hebbian layer in the policy")

        print(f"Hebbian layer: {hebbian_layer.in_features} -> {hebbian_layer.out_features}")
        print(f"ABCD parameters to optimize: {hebbian_layer.num_abcd_params}")

        # CMA-ES optimization parameters
        optimizer = CmaesHebbianOptimizer(
            experiment=experiment,
            hebbian_layer=hebbian_layer,
            pop_size=args.pop_size,
            sigma0=0.5,
            max_gens=args.cmaes_gens,
            n_eval_episodes=6,
            device=experiment.config.train_device,
        )

        best_abcd = optimizer.run()

        # Handle mid-interrupt
        if optimizer._current_gen < optimizer.max_gens and optimizer._best_abcd_so_far is not None:
            print(f"\nCMA-ES was interrupted at generation {optimizer._current_gen}/{optimizer.max_gens}")
            print(f"Using best solution found so far (fitness={-optimizer._best_fitness_so_far:.2f})")
            optimizer.apply_best_so_far()
            best_abcd = optimizer._best_abcd_so_far

        print(f"\nTraining complete. Best ABCD parameters shape: {best_abcd.shape}")

        # ========================================
        # Save Hebbian ABCD results
        # ========================================
        print("\n" + "=" * 60)
        print("Saving Hebbian results...")
        print("=" * 60)
        optimizer.save(output_dir=str(experiment.folder_name))

        # ========================================
        # CMA-ES Convergence Plot
        # ========================================
        print("\n" + "=" * 60)
        print("Generating CMA-ES convergence plot...")
        print("=" * 60)
        optimizer.plot_convergence(output_dir=str(experiment.folder_name))

        # ========================================
        # Phase 2 Evaluation: Save videos
        # ========================================
        print("\n" + "=" * 60)
        print(f"Phase 2 Evaluation: Running evaluation with {args.n_eval_episodes} episodes")
        print("=" * 60)
        optimizer.evaluate(
            output_dir=str(experiment.folder_name),
            n_episodes=args.n_eval_episodes,
        )