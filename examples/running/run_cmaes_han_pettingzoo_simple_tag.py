"""Train a HAN pursuer policy on PettingZoo MPE ``simple_tag_v3``.

The task contains three adversaries (pursuers), one good agent (evader), and no
obstacles. CMA-ES optimizes the ABCD parameters of the HAN shared by the three
pursuers. The evader is intentionally not optimized by the same objective: it
uses a deterministic policy that moves away from its nearest pursuer.

The capture fitness combines:

* a one-shot reward when any pursuer touches the evader;
* dense pressure from the nearest-pursuer distance;
* a timeout penalty when no capture occurs.

PettingZoo's continuous MPE action space is used throughout.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmarl.algorithms.cmaes_han import CmaesHanConfig
from benchmarl.algorithms.cmaes_han_optimizer import CmaesHanOptimizer
from benchmarl.environments import PettingZooTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig
from benchmarl.models.han import HanConfig


PURSUER_GROUP = "adversary"
EVADER_GROUP = "agent"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "CMA-ES HAN on PettingZoo simple_tag_v3 "
            "(3 pursuers, 1 evader, no obstacles)"
        )
    )

    # Task
    parser.add_argument("--num-adversaries", type=int, default=3)
    parser.add_argument("--num-good-agents", type=int, default=1)
    parser.add_argument("--num-obstacles", type=int, default=0)
    parser.add_argument("--max-cycles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)

    # Capture fitness
    parser.add_argument(
        "--fitness-mode",
        type=str,
        default="simple_tag_capture",
        choices=CmaesHanOptimizer.FITNESS_MODES,
    )
    parser.add_argument("--capture-distance", type=float, default=0.125)
    parser.add_argument("--catch-reward", type=float, default=10.0)
    parser.add_argument("--proximity-weight", type=float, default=1.0)
    parser.add_argument("--timeout-penalty", type=float, default=1.0)

    # CMA-ES
    parser.add_argument("--cmaes-gens", type=int, default=30)
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--sigma0", type=float, default=0.5)
    parser.add_argument("--n-eval-episodes", type=int, default=3)

    # HAN
    parser.add_argument("--hidden-size", type=int, default=10)
    parser.add_argument("--lr-hebb", type=float, default=0.01)
    parser.add_argument("--weight-init", type=float, default=0.1)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--f-nn", type=int, default=4)
    parser.add_argument("--f-hebb", type=int, default=1)

    # Evaluation/output
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--experiment-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-final-eval", type=int, default=10)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-video-frames", type=int, default=100)

    return parser.parse_args()


def _validate_args(args):
    if args.num_good_agents != 1:
        raise ValueError(
            "This runner currently supports exactly one evader "
            "(--num-good-agents 1)."
        )
    if args.num_adversaries < 1:
        raise ValueError("--num-adversaries must be at least 1.")
    if args.num_obstacles != 0:
        raise ValueError(
            "This no-obstacle runner requires --num-obstacles 0."
        )
    if args.capture_distance <= 0:
        raise ValueError("--capture-distance must be positive.")


def _get_task(args):
    task = PettingZooTask.SIMPLE_TAG.get_from_yaml()
    task.config["num_good"] = args.num_good_agents
    task.config["num_adversaries"] = args.num_adversaries
    task.config["num_obstacles"] = args.num_obstacles
    task.config["max_cycles"] = args.max_cycles
    return task


def _create_model_config(args):
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


def _setup_experiment(args, task, output_dir):
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.max_n_iters = 1
    experiment_config.prefer_continuous_actions = True
    experiment_config.share_policy_params = True
    experiment_config.save_folder = str(output_dir)

    experiment = Experiment(
        task=task,
        algorithm_config=CmaesHanConfig.get_from_yaml(),
        model_config=_create_model_config(args),
        critic_model_config=_create_critic_model_config(),
        seed=args.seed,
        config=experiment_config,
    )
    experiment._setup()

    expected_groups = {PURSUER_GROUP, EVADER_GROUP}
    missing = expected_groups.difference(experiment.group_map)
    if missing:
        raise RuntimeError(
            f"PettingZoo simple_tag groups changed; missing {sorted(missing)}. "
            f"Found groups: {experiment.group_map}"
        )
    if len(experiment.group_map[PURSUER_GROUP]) != args.num_adversaries:
        raise RuntimeError(
            "Unexpected pursuer group size: "
            f"{experiment.group_map[PURSUER_GROUP]}"
        )
    if len(experiment.group_map[EVADER_GROUP]) != 1:
        raise RuntimeError(
            f"Unexpected evader group: {experiment.group_map[EVADER_GROUP]}"
        )
    return experiment


def _create_optimizer(args, experiment, han_model, training):
    return CmaesHanOptimizer(
        experiment=experiment,
        han_model=han_model,
        fitness_mode=args.fitness_mode,
        pop_size=args.pop_size if training else 1,
        sigma0=args.sigma0,
        max_gens=args.cmaes_gens if training else 0,
        n_eval_episodes=args.n_eval_episodes if training else 1,
        device=experiment.config.train_device,
        catch_reward=args.catch_reward,
        proximity_weight=args.proximity_weight,
        timeout_penalty=args.timeout_penalty,
        train_group=PURSUER_GROUP,
        tag_evader_group=EVADER_GROUP,
        tag_num_adversaries=args.num_adversaries,
        tag_num_obstacles=args.num_obstacles,
        tag_capture_distance=args.capture_distance,
    )


def _print_summary(args, experiment, han_model):
    print("=" * 72)
    print("CMA-ES HAN — PettingZoo simple_tag_v3")
    print("=" * 72)
    print(
        f"Task: {args.num_adversaries} pursuers, "
        f"{args.num_good_agents} evader, {args.num_obstacles} obstacles, "
        f"max_cycles={args.max_cycles}"
    )
    print(f"Groups: {experiment.group_map}")
    print(
        "Control: adversary=shared HAN, "
        "agent=fixed escape-from-nearest policy"
    )
    print(
        f"Fitness: catch={args.catch_reward}, "
        f"proximity={args.proximity_weight}, "
        f"timeout={args.timeout_penalty}, "
        f"capture_distance={args.capture_distance}"
    )
    print(
        f"HAN: hidden={args.hidden_size}, window={args.window_size}, "
        f"f_nn={args.f_nn}, f_hebb={args.f_hebb}, "
        f"ABCD params={han_model.total_abcd_params}"
    )
    print(
        f"CMA-ES: pop={args.pop_size}, gens={args.cmaes_gens}, "
        f"sigma0={args.sigma0}, "
        f"episodes/candidate={args.n_eval_episodes}"
    )
    print()


def main():
    args = parse_args()
    _validate_args(args)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path(__file__).parent.parent / "outputs"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    task = _get_task(args)
    experiment = _setup_experiment(args, task, output_dir)
    han_model = experiment.algorithm.get_han_model(PURSUER_GROUP)
    if han_model is None:
        raise RuntimeError(
            f"No HanModel found in pursuer group {PURSUER_GROUP!r}."
        )

    _print_summary(args, experiment, han_model)

    if args.evaluate_only:
        if args.experiment_path is None:
            raise ValueError(
                "--experiment-path is required with --evaluate-only."
            )
        experiment_path = Path(args.experiment_path)
        han_dir = experiment_path / "han_results"
        with open(han_dir / "results.json") as file:
            metadata = json.load(file)
        if metadata.get("train_group", PURSUER_GROUP) != PURSUER_GROUP:
            raise ValueError(
                "Checkpoint was not trained for the adversary group."
            )

        experiment.policy.load_state_dict(
            torch.load(
                han_dir / "policy_state.pt",
                map_location=experiment.config.train_device,
            )
        )
        abcd = np.load(han_dir / "abcd_params.npy")
        han_model.set_abcd_from_vector(
            torch.tensor(abcd, device=experiment.config.train_device)
        )
        han_model.reset_all_weights()

        optimizer = _create_optimizer(
            args, experiment, han_model, training=False
        )
        optimizer._best_abcd_so_far = abcd
        optimizer.evaluate(
            output_dir=str(experiment_path),
            n_episodes=args.n_final_eval,
            fps=args.fps,
            max_video_frames=args.max_video_frames,
        )
        return

    optimizer = _create_optimizer(args, experiment, han_model, training=True)
    best_abcd = optimizer.run()
    if (
        optimizer._current_gen < optimizer.max_gens
        and optimizer._best_abcd_so_far is not None
    ):
        optimizer.apply_best_so_far()
        best_abcd = optimizer._best_abcd_so_far

    print(f"\nTraining complete. Best ABCD shape: {best_abcd.shape}")
    optimizer.save(output_dir=str(experiment.folder_name))
    optimizer.plot_convergence(output_dir=str(experiment.folder_name))
    if args.n_final_eval > 0:
        optimizer.evaluate(
            output_dir=str(experiment.folder_name),
            n_episodes=args.n_final_eval,
            fps=args.fps,
            max_video_frames=args.max_video_frames,
        )


if __name__ == "__main__":
    main()
