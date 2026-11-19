"""CMA-ES HAN training on single-agent navigation with static obstacles.

A single holonomic agent must reach a green goal while avoiding N static
red obstacles. The 9-dim observation per agent is:

    [ agent.pos (2),
      agent.vel (2),
      goal_rel (2),
      nearest_obstacle_rel (2),   # 0 if no obstacle within sense range
      has_obstacle_flag (1) ]

The fitness combines navigation_v2-style progress + success + final_term
with a time-averaged obstacle-distance penalty exp(-k * r), where
r normalises the agent→obstacle distance into [0, 1] over the safety
margin (r=1 → ~0 penalty; r=0 → penalty 1.0).
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
        description="CMA-ES HAN — Single-agent obstacle-avoidance navigation"
    )

    # Task configuration
    parser.add_argument("--n-static-obstacles", type=int, default=3,
                        help="Number of static obstacles. 0 disables obstacles.")
    parser.add_argument("--obstacle-radius", type=float, default=0.15)
    parser.add_argument("--agent-radius", type=float, default=0.10)
    parser.add_argument("--obstacle-sense-range", type=float, default=0.6,
                        help="Detection range for the nearest-obstacle feature.")
    parser.add_argument("--world-spawning-x", type=float, default=1.0)
    parser.add_argument("--world-spawning-y", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=800)

    # Fitness configuration (navigation_obs_avoidance only — other modes
    # are listed for API parity but the script forces the obs_avoidance mode)
    parser.add_argument("--fitness-mode", type=str,
                        default="navigation_obs_avoidance",
                        choices=CmaesHanOptimizer.FITNESS_MODES)
    # Sparse fitness formula:
    #   if reached AND not collided:
    #       fitness = +success_reward
    #   else:
    #       fitness = -final_weight      * final_goal_dist
    #               - penalty_weight     * penalty_ratio
    #               - nav_timeout_penalty   (when total_steps hit max)
    # where penalty_ratio = (steps with d_obs <= safety_distance) / total_steps.
    parser.add_argument("--success-reward", type=float, default=5.0,
                        help="Reward for reaching the goal without "
                             "touching any obstacle.")
    parser.add_argument("--final-weight", type=float, default=1.0,
                        help="Weight on final_goal_dist in the failure branch.")
    parser.add_argument("--penalty-weight", type=float, default=2.0,
                        help="Weight on penalty_ratio in the failure branch.")
    parser.add_argument("--progress-weight", type=float, default=0.0,
                        help="Weight of the relative goal progress "
                             "(init-final)/init in [-1, 1] in the failure "
                             "branch. 0 keeps the previous fitness.")
    parser.add_argument("--nav-timeout-penalty", type=float, default=2.0,
                        help="Extra penalty when the episode ran to "
                             "max_steps without success.")
    parser.add_argument("--obstacle-safety-distance", type=float, default=0.3,
                        help="Radius around each obstacle where steps "
                             "count toward penalty_ratio.")
    # ``--obstacle-penalty-weight`` is kept under its old name as a
    # backward-compat alias for ``--penalty-weight``.
    parser.add_argument("--obstacle-penalty-weight", type=float, default=2.0,
                        help="Backward-compat alias for --penalty-weight.")

    # CMA-ES parameters
    parser.add_argument("--cmaes-gens", type=int, default=30)
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--sigma0", type=float, default=0.5)
    parser.add_argument("--n-eval-episodes", type=int, default=3)

    # HAN network parameters
    parser.add_argument("--hidden-size", type=int, default=12)
    parser.add_argument("--lr-hebb", type=float, default=0.01)
    parser.add_argument("--weight-init", type=float, default=0.1)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--f-nn", type=int, default=4)
    parser.add_argument("--f-hebb", type=int, default=1)

    # Hebbian mechanism knobs (see HAN_diagnosis_and_rework_plan.md).
    parser.add_argument("--lr-oja", type=float, default=0.0,
                        help="Strength of the Oja stabilisation term "
                             "(0 disables it).")
    parser.add_argument("--d-scale", type=float, default=0.0,
                        help="Scale of the constant D (drift) term. 0 makes "
                             "the update purely data-driven.")
    parser.add_argument("--stats-mode", type=str, default="per_sample",
                        choices=["per_sample", "population"],
                        help="How the Hebbian statistics are formed.")
    parser.add_argument("--normalization", type=str, default="spectral",
                        choices=["spectral", "column_maxabs", "column_l2",
                                 "max_abs", "none"],
                        help="Weight normalization mode.")
    parser.add_argument("--no-normalize-stats", action="store_true",
                        help="Disable RMS-normalisation of the recorded "
                             "activations before the outer product.")
    parser.add_argument("--lr-reward", type=float, default=0.0,
                        help="Strength of the reward-modulated (three-factor) "
                             "Hebbian term. 0 disables it.")
    parser.add_argument("--modulation-source", type=str, default="env",
                        choices=["env", "shaped"],
                        help="Third factor of the reward term: the env reward "
                             "or the shaped signal m_t = goal progress - "
                             "lambda * obstacle danger.")
    parser.add_argument("--modulation-obs-weight", type=float, default=1.0,
                        help="lambda: weight of the obstacle-danger part of m_t.")
    parser.add_argument("--advantage-clip", type=float, default=1.0,
                        help="Clip on |m - baseline| per sample in the reward term.")

    # Compatibility kwargs (unused by navigation_obs_avoidance, but
    # listed so the CmaesHanOptimizer API stays uniform).
    parser.add_argument("--collision-penalty-weight", type=float, default=2.0)
    parser.add_argument("--safety-distance", type=float, default=0.15)
    parser.add_argument("--neighbor-radius", type=float, default=0.5)
    parser.add_argument("--movement-target-displacement", type=float, default=1.0)
    parser.add_argument("--orbit-radius", type=float, default=0.7)
    parser.add_argument("--orbit-radius-tolerance", type=float, default=0.3)
    parser.add_argument("--dt-floor", type=float, default=0.1)

    # Evaluation
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Skip training, only evaluate from --experiment-path.")
    parser.add_argument("--experiment-path", type=str, default=None)
    parser.add_argument("--n-final-eval", type=int, default=10)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-video-frames", type=int, default=800)

    # Static-network control.
    #
    # CMA-ES optimises the plastic weight matrices W directly instead of
    # the ABCD Hebbian parameters, and the Hebbian rule is switched off.
    # This is the *meaningful* baseline for "what does plasticity buy?":
    # a trained static feedforward network of the same architecture.
    # (Using ``--lr-hebb 0`` instead would leave W at its random
    # initialisation, i.e. it would measure a random network.)
    parser.add_argument("--static-mode", action="store_true",
                        help="Train a static network control (CMA-ES "
                             "optimises W; the Hebbian rule is disabled).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Experiment seed. The same seed makes the "
                             "plastic and static runs face identical task "
                             "instances, which is required for a fair "
                             "comparison.")

    return parser.parse_args()


args = parse_args()

# The static control must have the Hebbian rule completely inert: with
# ``lr_hebb == 0`` the update ``W + 0*delta_W`` leaves W at its random
# initialisation, so CMA-ES would be optimising ABCD parameters that
# influence nothing. Instead the static control lets CMA-ES optimise W
# directly (see ``CmaesHanOptimizer.static_mode``).
if args.static_mode and args.lr_hebb != 0.0:
    print(f"NOTE: --static-mode forces --lr-hebb 0 "
          f"(was {args.lr_hebb}); plasticity would defeat the control.")
    args.lr_hebb = 0.0


def _get_task():
    task = VmasTask.NAVIGATION_OBS_AVOIDANCE.get_from_yaml()
    task.config["n_static_obstacles"] = args.n_static_obstacles
    task.config["obstacle_radius"] = args.obstacle_radius
    task.config["agent_radius"] = args.agent_radius
    task.config["obstacle_sense_range"] = args.obstacle_sense_range
    task.config["world_spawning_x"] = args.world_spawning_x
    task.config["world_spawning_y"] = args.world_spawning_y
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
        lr_oja=args.lr_oja,
        d_scale=args.d_scale,
        stats_mode=args.stats_mode,
        normalization=args.normalization,
        normalize_stats=not args.no_normalize_stats,
        lr_reward=args.lr_reward,
        modulation_key=("modulation" if args.modulation_source == "shaped"
                        else "reward"),
        advantage_clip=args.advantage_clip,
        activation_class=torch.nn.Tanh,
    )


def _create_critic_model_config():
    return MlpConfig(
        num_cells=[64, 64],
        activation_class=torch.nn.Tanh,
        layer_class=torch.nn.Linear,
    )


def _setup_experiment_for_cmaes(task, model_config, critic_model_config, output_dir,
                               seed=0):
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.max_n_iters = 1
    experiment_config.save_folder = str(output_dir)

    experiment = Experiment(
        task=task,
        algorithm_config=CmaesHanConfig.get_from_yaml(),
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=seed,
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
    print("CMA-ES HAN — Single-agent obstacle-avoidance navigation")
    print("=" * 60)
    print(f"Task: navigation_obs_avoidance")
    print(f"  n_static_obstacles={args.n_static_obstacles}, "
          f"obstacle_radius={args.obstacle_radius}, "
          f"agent_radius={args.agent_radius}")
    print(f"  obstacle_sense_range={args.obstacle_sense_range}, "
          f"max_steps={args.max_steps}")
    print(f"Fitness mode: {args.fitness_mode}")
    if args.fitness_mode == "navigation_obs_avoidance":
        print(f"  success_reward={args.success_reward}, "
              f"final_weight={args.final_weight}, "
              f"penalty_weight={args.penalty_weight}, "
              f"nav_timeout_penalty={args.nav_timeout_penalty}, "
              f"progress_weight={args.progress_weight}")
        print(f"  obstacle_safety_distance={args.obstacle_safety_distance}")
    print(f"HAN: hidden={args.hidden_size}, "
          f"window_size={args.window_size}, "
          f"f_nn={args.f_nn}, f_hebb={args.f_hebb}")
    print(f"  mechanism: stats_mode={args.stats_mode}, "
          f"normalize_stats={not args.no_normalize_stats}, "
          f"normalization={args.normalization}, "
          f"lr_oja={args.lr_oja}, d_scale={args.d_scale}, "
          f"lr_reward={args.lr_reward}")
    print(f"  modulation: source={args.modulation_source}, "
          f"obs_weight={args.modulation_obs_weight}, "
          f"advantage_clip={args.advantage_clip}")
    print(f"CMA-ES: pop={args.pop_size}, gens={args.cmaes_gens}, "
          f"sigma0={args.sigma0}")
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
            task, model_config, critic_model_config, output_dir, seed=args.seed
        )
        policy_path = han_dir / "policy_state.pt"
        experiment.policy.load_state_dict(
            torch.load(str(policy_path),
                       map_location=experiment.config.train_device)
        )
        han_model = experiment.algorithm.get_han_model()
        abcd_path = han_dir / "abcd_params.npy"
        w_path = han_dir / "w_params.npy"
        if args.static_mode and w_path.exists():
            # Static control: the optimised parameters are W, not ABCD.
            han_model.set_w_from_vector(
                torch.tensor(
                    np.load(str(w_path)), device=experiment.config.train_device
                )
            )
            han_model.reset_all_weights()
        elif abcd_path.exists():
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
            obstacle_penalty_weight=args.obstacle_penalty_weight,
            obstacle_safety_distance=args.obstacle_safety_distance,
            obstacle_agent_radius=args.agent_radius,
            obstacle_obstacle_radius=args.obstacle_radius,
            success_reward=args.success_reward,
            final_weight=args.final_weight,
            penalty_weight=args.penalty_weight,
            nav_timeout_penalty=args.nav_timeout_penalty,
            progress_weight=args.progress_weight,
            modulation_source=args.modulation_source,
            modulation_obs_weight=args.modulation_obs_weight,
            static_mode=args.static_mode,
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
            task, model_config, critic_model_config, output_dir, seed=args.seed
        )
        han_model = experiment.algorithm.get_han_model()
        if han_model is None:
            raise RuntimeError("No HanModel in policy")

        layers = han_model.get_all_han_layers()
        print(f"HanModel: {len(layers)} layers, "
              f"{han_model.total_abcd_params} ABCD params")
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
            obstacle_penalty_weight=args.obstacle_penalty_weight,
            obstacle_safety_distance=args.obstacle_safety_distance,
            obstacle_agent_radius=args.agent_radius,
            obstacle_obstacle_radius=args.obstacle_radius,
            success_reward=args.success_reward,
            final_weight=args.final_weight,
            penalty_weight=args.penalty_weight,
            nav_timeout_penalty=args.nav_timeout_penalty,
            progress_weight=args.progress_weight,
            modulation_source=args.modulation_source,
            modulation_obs_weight=args.modulation_obs_weight,
            static_mode=args.static_mode,
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