"""CMA-ES HAN training on simple_tag_v1 (pursuer vs. evader, no obstacles).

Three adversaries chase a single good agent in a square arena without
landmarks. The episode terminates as soon as any adversary comes into
contact with the good (caught = True) or after ``max_steps``.

The HAN network for every agent receives a constant-size observation
that does NOT depend on the number of agents:

    [ self_pos (2),
      self_vel (2),                              # only if observe_vel
      nearest_neighbor_rel (2),                  # closest other agent
      nearest_neighbor_vel (2),                 # only if observe_vel
      nearest_good_rel (2) ]                     # closest good agent

= 10 dims with vel, 6 dims without. Search radius for "nearest" is
controlled by ``--nearest-radius``; entities outside the radius are
zero-padded in the corresponding slot but do NOT change obs length.

There is no role flag: leaders / pursuers all share the same architecture
and the same ABCD parameters (just like the other flocking_* tasks).

Fitness (mode = ``simple_tag_capture``) rewards:
- a one-shot ``catch_reward`` if the evader was caught at any step;
- minus the mean per-step distance from the closest adversary to the
  good agent (proximity pressure);
- minus ``timeout_penalty`` if the rollout ran out the clock without a
  catch.
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
        description="CMA-ES HAN on simple_tag_v1 (no-obstacle pursuit)"
    )

    # Task parameters
    parser.add_argument("--num-good-agents", type=int, default=1)
    parser.add_argument("--num-adversaries", type=int, default=3)
    parser.add_argument(
        "--observe-vel", type=str, default="True",
        help="Include velocity in the per-agent observation.",
    )
    parser.add_argument(
        "--nearest-radius", type=float, default=1.0,
        help="Search radius for the nearest-neighbor / nearest-good "
             "slots in the observation. Outside this radius the slots "
             "are zero-padded. Does NOT change obs dimensionality.",
    )
    parser.add_argument("--bound", type=float, default=1.0)
    parser.add_argument(
        "--done-when-caught", type=str, default="True",
        help="End the episode as soon as any adversary is in contact "
             "with the good agent.",
    )
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--spawn-radius", type=float, default=0.8)

    # Fitness
    parser.add_argument(
        "--fitness-mode", type=str, default="simple_tag_capture",
        choices=CmaesHanOptimizer.FITNESS_MODES,
    )
    parser.add_argument("--catch-reward", type=float, default=5.0)
    parser.add_argument("--proximity-weight", type=float, default=1.0)
    parser.add_argument("--timeout-penalty", type=float, default=1.0)

    # CMA-ES
    parser.add_argument("--cmaes-gens", type=int, default=30)
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--sigma0", type=float, default=0.5)
    parser.add_argument("--n-eval-episodes", type=int, default=2)

    # HAN network
    parser.add_argument("--hidden-size", type=int, default=10)
    parser.add_argument("--lr-hebb", type=float, default=0.01)
    parser.add_argument("--weight-init", type=float, default=0.1)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--f-nn", type=int, default=4)
    parser.add_argument("--f-hebb", type=int, default=1)

    # Other CmaesHanOptimizer kwargs (kept for API compatibility; not
    # used by the simple_tag_capture fitness).
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
    parser.add_argument("--max-video-frames", type=int, default=800)

    return parser.parse_args()


# argparse converts str values to bool at the call-site below to keep
# the CLI flags consistent with what BenchMARL YAML expects.
def _as_bool(value):
    if isinstance(value, bool):
        return value
    return value.lower() not in {"false", "0", "no", ""}


args = parse_args()


def _get_task():
    task = VmasTask.SIMPLE_TAG_V1.get_from_yaml()
    cfg = task.config
    cfg["num_good_agents"] = args.num_good_agents
    cfg["num_adversaries"] = args.num_adversaries
    cfg["observe_vel"] = _as_bool(args.observe_vel)
    cfg["nearest_radius"] = args.nearest_radius
    cfg["bound"] = args.bound
    cfg["done_when_caught"] = _as_bool(args.done_when_caught)
    cfg["spawn_radius"] = args.spawn_radius
    cfg["max_steps"] = args.max_steps
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
    print("CMA-ES HAN — Simple Tag v1 (no-obstacle pursuit)")
    print("=" * 60)
    print(f"Task: simple_tag_v1 | n_good={args.num_good_agents}, "
          f"n_adv={args.num_adversaries}, bound={args.bound}")
    print(f"  observe_vel={_as_bool(args.observe_vel)}, "
          f"nearest_radius={args.nearest_radius}, "
          f"done_when_caught={_as_bool(args.done_when_caught)}, "
          f"max_steps={args.max_steps}")
    print(f"Fitness mode: {args.fitness_mode} "
          f"(catch_reward={args.catch_reward}, "
          f"proximity_weight={args.proximity_weight}, "
          f"timeout_penalty={args.timeout_penalty})")
    print(f"HAN: hidden={args.hidden_size}, window={args.window_size}, "
          f"f_nn={args.f_nn}, f_hebb={args.f_hebb}")
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
            catch_reward=args.catch_reward,
            proximity_weight=args.proximity_weight,
            timeout_penalty=args.timeout_penalty,
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
            catch_reward=args.catch_reward,
            proximity_weight=args.proximity_weight,
            timeout_penalty=args.timeout_penalty,
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
