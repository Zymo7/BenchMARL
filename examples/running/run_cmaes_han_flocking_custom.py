"""CMA-ES HAN training on flocking with a CUSTOM target motion script.

This is a fork of `run_cmaes_han.py` that monkey-patches the VMAS flocking
scenario's `action_script_creator` BEFORE the test env is built. Edit the
`custom_target_action_script` function below to change how the green target
moves; everything else (HAN, CMA-ES, fitness) is unchanged.

Usage:
    /home/zhaozeming/miniconda3/envs/benchmarl/bin/python \\
        examples/running/run_cmaes_han_flocking_custom.py \\
        --task flocking \\
        --fitness-mode flocking_orbit \\
        --cmaes-gens 30 --pop-size 20 --sigma0 0.3 \\
        ... (all other flags same as run_cmaes_han.py)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------------
# Monkey-patch: replace the VMAS flocking scenario's target motion BEFORE
# VmasClass imports/uses the scenario. This must happen at import time of
# this script (which is before VmasTask.FLOCKING.get_from_yaml()).
# --------------------------------------------------------------------------
# Mechanism: VMAS's vmas.scenarios.load() re-executes the scenario file via
# importlib every time make_env(scenario="flocking") is called. So patching
# a one-off imported copy of the module has no effect. We instead wrap the
# `load` function so that, after it loads flocking.py, we override the
# Scenario.action_script_creator to use our custom closure.
#
# IMPORTANT: `vmas/__init__.py` reassigns `scenarios` to a sorted list of
# scenario names, shadowing the subpackage. We must use
# `importlib.import_module('vmas.scenarios')` to get the real subpackage
# (which has `load`); the `vmas.scenarios` attribute itself is a list
# after `vmas/__init__.py` finishes loading.
import importlib
import vmas  # noqa: F401 — triggers vmas.__init__, vmas.scenarios = list

_vmas_scenarios_pkg = importlib.import_module("vmas.scenarios")
_original_load = _vmas_scenarios_pkg.load


def _patched_load(name):
    module = _original_load(name)
    if name == "flocking.py" or name == "flocking":
        # Replace action_script_creator with a function returning our
        # custom action script. Matches the original VMAS calling
        # convention: self.action_script_creator() → action_script.
        module.Scenario.action_script_creator = _patched_action_script_creator
        # Replace reset_world_at so the target's INITIAL position is
        # set to (--target-pos-x, --target-pos-y) instead of the VMAS
        # default (0, -y_dim). The default places the target at the
        # bottom edge of the world and biases the flocking orbit
        # asymmetrically. The original VMAS reset_world_at is also
        # responsible for spawning random agent positions, obstacle
        # positions, distance shaping, etc.; we keep all that logic and
        # only override the target_pos initialization.
        module.Scenario.reset_world_at = _patched_reset_world_at
    return module


def _patched_reset_world_at(self, env_index=None):
    """Like the original VMAS flocking reset_world_at, but with the
    target's initial position replaced by (TARGET_POS_X, TARGET_POS_Y).

    Reads module-level ``_TARGET_POS_X`` / ``_TARGET_POS_Y`` so it can be
    configured from CLI args.
    """
    # Lazy-import the same constants the original uses, so all other
    # behaviour (spawning, distance shaping, lidar init) is unchanged.
    from vmas.simulator.utils import ScenarioUtils, Y, X  # noqa: F401
    import torch as _torch

    target_pos = _torch.zeros(
        (
            (1, self.world.dim_p)
            if env_index is not None
            else (self.world.batch_dim, self.world.dim_p)
        ),
        device=self.world.device,
        dtype=_torch.float32,
    )
    target_pos[:, X] = _TARGET_POS_X
    target_pos[:, Y] = _TARGET_POS_Y
    self._target.set_pos(target_pos, batch_index=env_index)
    ScenarioUtils.spawn_entities_randomly(
        self.obstacles + self.world.policy_agents,
        self.world,
        env_index,
        self._min_dist_between_entities,
        x_bounds=(-self.x_dim, self.x_dim),
        y_bounds=(-self.y_dim, self.y_dim),
        occupied_positions=target_pos.unsqueeze(1),
    )

    for agent in self.world.policy_agents:
        if env_index is None:
            agent.distance_shaping = (
                _torch.stack(
                    [
                        _torch.linalg.vector_norm(
                            agent.state.pos - a.state.pos, dim=-1
                        )
                        for a in self.world.agents
                        if a != agent
                    ],
                    dim=1,
                )
                - self.desired_distance
            ).pow(2).mean(-1) * self.dist_shaping_factor
        else:
            agent.distance_shaping[env_index] = (
                _torch.stack(
                    [
                        _torch.linalg.vector_norm(
                            agent.state.pos[env_index] - a.state.pos[env_index]
                        )
                        for a in self.world.agents
                        if a != agent
                    ],
                    dim=0,
                )
                - self.desired_distance
            ).pow(2).mean(-1) * self.dist_shaping_factor

    if env_index is None:
        self.t = _torch.zeros(self.world.batch_dim, device=self.world.device)
    else:
        self.t[env_index] = 0


_vmas_scenarios_pkg.load = _patched_load
# --------------------------------------------------------------------------
# End monkey-patch. Now safe to import everything else.
# --------------------------------------------------------------------------




def custom_target_action_script(agent, world, scenario_self):
    """The action script called by VMAS every env step. Set
    ``agent.action.u`` to the (2,) velocity (or force) you want the target
    to take this step.

    Args:
        agent: the target Agent whose action we set.
        world: the VMAS World (unused by default).
        scenario_self: the VMAS Scenario instance — its ``t`` attribute
            is the step counter (incremented in the scenario's reward()).
    """
    t = scenario_self.t / 30.0  # original scaling; matches VMAS flocking default
    # === EDIT BELOW to change target motion ===
    # SANITY-CHECK: keep target STATIONARY at its initial position (0, -1).
    u_x = torch.zeros_like(t)
    u_y = torch.zeros_like(t)
    # === END EDIT ===
    agent.action.u = torch.stack([u_x, u_y], dim=1)


# Mirrors the original VMAS flocking signature:
#   def action_script_creator(self):
#       def action_script(agent, world):
#           t = self.t / 30
#           agent.action.u = ...
#       return action_script
# VMAS calls `self.action_script_creator()` (with self = Scenario instance)
# inside make_world(), then uses the returned closure as the agent's
# action_script. The closure captures self = Scenario. We replicate the
# pattern but delegate to custom_target_action_script for clarity.
def _patched_action_script_creator(self):
    def action_script(agent, world):
        custom_target_action_script(agent, world, self)
    return action_script


from benchmarl.algorithms.cmaes_han import CmaesHanConfig
from benchmarl.algorithms.cmaes_han_optimizer import CmaesHanOptimizer
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig
from benchmarl.models.han import HanConfig


def parse_args():
    # Identical CLI to run_cmaes_han.py (kept in sync manually — if you add
    # args there, mirror them here).
    parser = argparse.ArgumentParser(
        description="CMA-ES HAN on flocking with a CUSTOM target motion"
    )

    parser.add_argument(
        "--fitness-mode", type=str, default="flocking_orbit",
        choices=CmaesHanOptimizer.FITNESS_MODES,
        help="Fitness function mode. Default flocking_orbit.",
    )
    parser.add_argument("--collision-penalty-weight", type=float, default=2.0)
    parser.add_argument("--safety-distance", type=float, default=0.15)
    parser.add_argument("--neighbor-radius", type=float, default=0.5)
    parser.add_argument(
        "--task", type=str, default="flocking",
        choices=["navigation_static_dynamic_obs", "navigation_dynamic_obs", "flocking"],
    )
    parser.add_argument("--n-static-obstacles", type=int, default=0)
    parser.add_argument("--n-dynamic-obstacles", type=int, default=0)
    parser.add_argument("--movement-target-displacement", type=float, default=1.0)
    parser.add_argument("--orbit-radius", type=float, default=0.7)
    parser.add_argument("--orbit-radius-tolerance", type=float, default=0.3)
    parser.add_argument("--dt-floor", type=float, default=0.1)

    parser.add_argument("--cmaes-gens", type=int, default=40)
    parser.add_argument("--pop-size", type=int, default=40)
    parser.add_argument("--sigma0", type=float, default=0.3)
    parser.add_argument("--n-eval-episodes", type=int, default=2)

    parser.add_argument("--hidden-size", type=int, default=18)
    parser.add_argument("--lr-hebb", type=float, default=0.01)
    parser.add_argument("--weight-init", type=float, default=0.1)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--f-nn", type=int, default=4)
    parser.add_argument("--f-hebb", type=int, default=1)

    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--experiment-path", type=str, default=None)
    parser.add_argument("--n-final-eval", type=int, default=10)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-video-frames", type=int, default=400)
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

# Module-level constants read by _patched_reset_world_at. Set after
# args is parsed so CLI overrides flow through.
_TARGET_POS_X = args.target_pos_x
_TARGET_POS_Y = args.target_pos_y


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
    print("CMA-ES HAN — Flocking with CUSTOM target motion")
    print("=" * 60)
    print(f"Task: {args.task} | fitness_mode: {args.fitness_mode}")
    print(f"Patched action_script_creator: "
          f"{_patched_action_script_creator.__qualname__}")
    print(f"orbit_radius={args.orbit_radius}, "
          f"orbit_radius_tolerance={args.orbit_radius_tolerance}, "
          f"dt_floor={args.dt_floor}")
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
            task, model_config, critic_model_config, output_dir)
        policy_path = han_dir / "policy_state.pt"
        experiment.policy.load_state_dict(
            torch.load(str(policy_path),
                       map_location=experiment.config.train_device))
        han_model = experiment.algorithm.get_han_model()
        abcd_path = han_dir / "abcd_params.npy"
        if abcd_path.exists():
            abcd = np.load(str(abcd_path))
            han_model.set_abcd_from_vector(
                torch.tensor(abcd, device=experiment.config.train_device))
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
            task, model_config, critic_model_config, output_dir)
        han_model = experiment.algorithm.get_han_model()
        if han_model is None:
            raise RuntimeError("No HanModel in policy")
        print(f"HanModel: {han_model.total_abcd_params} ABCD params\n")

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

        optimizer.save(output_dir=str(experiment.folder_name))
        optimizer.plot_convergence(output_dir=str(experiment.folder_name))
        optimizer.evaluate(
            output_dir=str(experiment.folder_name),
            n_episodes=args.n_final_eval,
            fps=args.fps,
            max_video_frames=args.max_video_frames,
        )