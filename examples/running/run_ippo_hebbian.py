import os
from pathlib import Path

import torch
from benchmarl.algorithms.ippo_hebbian import IppoHebbianConfig
from benchmarl.algorithms.cmaes_optimizer import CmaesHebbianOptimizer
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig
from benchmarl.models.hebbian import HebbianConfig
from benchmarl.models.common import SequenceModelConfig

if __name__ == "__main__":

    # === Network architecture ===
    # Layer 1-2: Standard MLP (64 neurons each, Tanh activation) -> trained by PPO
    # Layer 3: Hebbian output layer (64 -> action_dim*2) -> ABCD optimized by CMA-ES
    model_config = SequenceModelConfig(
        model_configs=[
            MlpConfig(num_cells=[64, 64], activation_class=torch.nn.Tanh, layer_class=torch.nn.Linear),
            HebbianConfig(lr_hebb=0.01, weight_init=1.0),
        ],
        intermediate_sizes=[64],
    )

    # Critic uses standard MLP (no Hebbian layer)
    critic_model_config = MlpConfig(
        num_cells=[64, 64],
        activation_class=torch.nn.Tanh,
        layer_class=torch.nn.Linear,
    )

    # Load configs from yaml
    experiment_config = ExperimentConfig.get_from_yaml()
    task = VmasTask.NAVIGATION.get_from_yaml()
    algorithm_config = IppoHebbianConfig.get_from_yaml()

    # Phase 1: limit training iterations for PPO phase
    experiment_config.max_n_iters = 200  # For Full training
    # experiment_config.max_n_iters = 5 # For quick testing
    
    # Save outputs to outputs/ folder (same as hydra-based runs)
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    experiment_config.save_folder = str(output_dir)

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
    )

    # ========================================
    # Phase 1: PPO training of MLP layers
    # ========================================
    print("=" * 60)
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
    # Adjust these based on your needs:
    # - pop_size: population size (more = better exploration, slower)
    # - max_gens: maximum generations (more = better convergence, slower)
    # - n_eval_episodes: episodes per fitness evaluation (more = stable, slower)
    optimizer = CmaesHebbianOptimizer(
        experiment=experiment,
        hebbian_layer=hebbian_layer,
        pop_size=30,      # Population size
        sigma0=0.5,       # Initial step size
        max_gens=30,      # Maximum generations
        # max_gens=5,      # quick testing
        n_eval_episodes=6,  # Episodes per evaluation
        device=experiment.config.train_device,
    )

    best_abcd = optimizer.run()

    # Handle mid-interrupt: use best_so_far if CMA-ES was interrupted
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
    # Phase 2 Evaluation: Save videos with Hebbian layer
    # ========================================
    print("\n" + "=" * 60)
    print("Phase 2 Evaluation: Running evaluation with Hebbian layer")
    print("=" * 60)

    optimizer.evaluate(
        output_dir=str(experiment.folder_name),
        n_episodes=10,
    )
