import json
import math
import os
import time

import numpy as np
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type


class CmaesHanOptimizer:
    """CMA-ES optimizer for the ABCD parameters of a HanModel.

    Single-phase training: CMA-ES directly optimizes all ABCD Hebbian
    parameters across every :class:`HanLayer` in the network. The plastic
    weights ``W`` themselves evolve online during evaluation through the
    HAN rule (decoupled ``update_weights`` with sliding-window averaging
    and hard layer-wise max-abs normalization).

    Supports the same fitness modes as :class:`CmaesFullHebbianOptimizer`.
    """

    FITNESS_MODES = [
        "navigation",
        "navigation_avoidance",
        "navigation_v2",
        "navigation_avoidance_v2",
        "navigation_obs_avoidance",
        "flocking_global",
        "flocking_orbit",
        "flocking_lf_arrival",
        "flocking_light_intensity",
        "flocking_signal_intensity",
        "simple_tag_capture",
        "hgn_formation_v1",
    ]

    def __init__(
        self,
        experiment,
        han_model,
        fitness_mode: str = "navigation_avoidance",
        pop_size: int = 30,
        sigma0: float = 0.5,
        max_gens: int = 100,
        n_eval_episodes: int = 10,
        device: str = "cpu",
        collision_penalty_weight: float = 2.0,
        safety_distance: float = 0.15,
        neighbor_radius: float = 0.5,
        movement_target_displacement: float = 1.0,
        patch_heading_from_vel: bool = True,
        orbit_radius: float = 0.7,
        orbit_radius_tolerance: float = 0.3,
        dt_floor: float = 0.1,
        catch_reward: float = 5.0,
        proximity_weight: float = 1.0,
        timeout_penalty: float = 1.0,
        train_group: str = None,
        tag_evader_group: str = None,
        tag_num_adversaries: int = 3,
        tag_num_obstacles: int = 0,
        tag_capture_distance: float = 0.125,
        # navigation_obs_avoidance fitness parameters.
        #
        # The fitness is now intentionally simple so that HAN's online
        # Hebbian weight updates can converge toward a stable attractor
        # rather than chasing a multi-component composite reward:
        #
        #     if reached AND not collided:
        #         fitness = +success_reward
        #     else:
        #         fitness = - final_weight   * final_goal_dist
        #                 - penalty_weight  * penalty_ratio
        #                 - timeout_penalty  (when the episode ran out
        #                                     of steps without success)
        #
        # ``penalty_ratio`` = (steps with d_obs <= d_safety) / total_steps,
        # which is a single episode-end scalar (no time segmentation).
        #
        # The old ``obstacle_penalty_weight / obstacle_penalty_k /
        # obstacle_safety_distance`` knob trio is preserved under the
        # same names but now they only feed ``penalty_ratio`` via the
        # safety-distance threshold; ``obstacle_penalty_weight`` is
        # reused as the alias for ``penalty_weight`` to keep older
        # run-scripts working.
        obstacle_penalty_weight: float = 2.0,
        obstacle_safety_distance: float = 0.3,
        obstacle_agent_radius: float = 0.10,
        obstacle_obstacle_radius: float = 0.15,
        success_reward: float = 5.0,
        final_weight: float = 1.0,
        penalty_weight: float = 2.0,
        nav_timeout_penalty: float = 2.0,
        # HGN formation-control fitness parameters.
        formation_reach_radius: float = 0.10,
        formation_collision_penalty: float = 2.0,
        formation_timeout_penalty: float = 2.0,
        formation_tail_frac: float = 0.10,
    ):
        self.experiment = experiment
        self.han_model = han_model
        self.fitness_mode = fitness_mode
        self.pop_size = pop_size
        self.sigma0 = sigma0
        self.max_gens = max_gens
        self.n_eval_episodes = n_eval_episodes
        self.device = device
        self.collision_penalty_weight = collision_penalty_weight
        self.safety_distance = safety_distance
        self.neighbor_radius = neighbor_radius
        self.movement_target_displacement = movement_target_displacement
        self.patch_heading_from_vel = patch_heading_from_vel
        self.orbit_radius = orbit_radius
        self.orbit_radius_tolerance = orbit_radius_tolerance
        self.dt_floor = dt_floor
        self.catch_reward = catch_reward
        self.proximity_weight = proximity_weight
        self.timeout_penalty = timeout_penalty
        self.train_group = (
            train_group
            if train_group is not None
            else list(experiment.group_map.keys())[0]
        )
        self.tag_evader_group = tag_evader_group
        self.tag_num_adversaries = tag_num_adversaries
        self.tag_num_obstacles = tag_num_obstacles
        self.tag_capture_distance = tag_capture_distance
        # Navigation-obs-avoidance fitness parameters.
        self.success_reward = success_reward
        self.final_weight = final_weight
        # ``obstacle_penalty_weight`` is preserved under its old name as
        # the backward-compat alias for ``penalty_weight``.
        self.penalty_weight = (
            penalty_weight if penalty_weight != 2.0 else obstacle_penalty_weight
        )
        self.nav_timeout_penalty = nav_timeout_penalty
        self.obstacle_safety_distance = obstacle_safety_distance
        self.obstacle_agent_radius = obstacle_agent_radius
        self.obstacle_obstacle_radius = obstacle_obstacle_radius
        # HGN formation-control fitness parameters.
        self.formation_reach_radius = formation_reach_radius
        self.formation_collision_penalty = formation_collision_penalty
        self.formation_timeout_penalty = formation_timeout_penalty
        self.formation_tail_frac = formation_tail_frac

        self.policy = experiment.policy
        self.rollout_policy = (
            experiment.group_policies[self.train_group]
            if self.tag_evader_group is not None
            else self.policy
        )

        self._initial_abcd = han_model.get_abcd_vector().clone()
        self._best_abcd_so_far = None
        self._best_fitness_so_far = float('inf')
        self._current_gen = 0

    def get_abcd_vector(self) -> np.ndarray:
        return self.han_model.get_abcd_vector().detach().cpu().numpy()

    def set_abcd_from_vector(self, x: np.ndarray):
        self.han_model.set_abcd_from_vector(torch.tensor(x, device=self.device))

    def _get_vmas_core(self):
        """Walk the wrapper chain to reach the underlying vmas Environment.

        Returns the object that exposes ``agents`` with ``state.pos`` —
        the source of truth for absolute agent positions. Result is
        cached on the optimizer because the wrapper chain does not change.
        """
        if hasattr(self, "_vmas_core") and self._vmas_core is not None:
            return self._vmas_core
        env = self.experiment.test_env
        node = env
        while True:
            nxt = getattr(node, "base_env", None) or getattr(node, "_env", None)
            if nxt is None or nxt is node:
                break
            node = nxt
        self._vmas_core = node
        return node

    def _get_pettingzoo_tag_relative_positions(
        self, evader_observation: torch.Tensor
    ) -> torch.Tensor:
        """Extract adversary positions relative to the good agent.

        PettingZoo MPE ``simple_tag_v3`` observations are ordered as
        ``[self_vel, self_pos, landmark_rel_pos, other_agent_rel_pos, ...]``.
        This project uses one good agent, so all ``other_agent_rel_pos`` entries
        in the good-agent observation are the adversaries.
        """
        start = 4 + 2 * self.tag_num_obstacles
        end = start + 2 * self.tag_num_adversaries
        if evader_observation.shape[-1] < end:
            raise ValueError(
                "PettingZoo simple_tag observation is too short for "
                f"{self.tag_num_adversaries} adversaries and "
                f"{self.tag_num_obstacles} obstacles: "
                f"got {evader_observation.shape[-1]}, need at least {end}"
            )
        return evader_observation[..., start:end].reshape(
            *evader_observation.shape[:-1], self.tag_num_adversaries, 2
        )

    def _set_pettingzoo_tag_evader_action(self, tensordict):
        """Apply a deterministic escape policy to the single good agent.

        The CMA-ES candidate controls only the adversary group. The good agent
        moves directly away from its nearest adversary using the continuous MPE
        action layout ``[noop, left, right, down, up]``.
        """
        if self.tag_evader_group is None:
            return

        observation = tensordict.get(
            (self.tag_evader_group, "observation"), None
        )
        if observation is None:
            raise KeyError(
                f"Missing observation for tag evader group "
                f"{self.tag_evader_group!r}"
            )

        relative_positions = self._get_pettingzoo_tag_relative_positions(
            observation.float()
        )
        distances = torch.linalg.vector_norm(relative_positions, dim=-1)
        nearest_index = distances.argmin(dim=-1, keepdim=True)
        gather_index = nearest_index.unsqueeze(-1).expand(
            *nearest_index.shape, 2
        )
        nearest_relative = torch.gather(
            relative_positions, dim=-2, index=gather_index
        ).squeeze(-2)
        escape_direction = -nearest_relative
        escape_direction = escape_direction / (
            torch.linalg.vector_norm(
                escape_direction, dim=-1, keepdim=True
            ).clamp_min(1e-6)
        )

        action = torch.zeros(
            *escape_direction.shape[:-1],
            5,
            device=escape_direction.device,
            dtype=escape_direction.dtype,
        )
        action[..., 1] = (-escape_direction[..., 0]).clamp(min=0.0, max=1.0)
        action[..., 2] = escape_direction[..., 0].clamp(min=0.0, max=1.0)
        action[..., 3] = (-escape_direction[..., 1]).clamp(min=0.0, max=1.0)
        action[..., 4] = escape_direction[..., 1].clamp(min=0.0, max=1.0)
        tensordict.set((self.tag_evader_group, "action"), action)

    def _record_pettingzoo_tag_step(self, tensordict, step: int):
        """Return capture/proximity bookkeeping for one PettingZoo tag step."""
        observation = tensordict.get(
            ("next", self.tag_evader_group, "observation"), None
        )
        if observation is None:
            raise KeyError(
                f"Missing next observation for tag evader group "
                f"{self.tag_evader_group!r}"
            )

        relative_positions = self._get_pettingzoo_tag_relative_positions(
            observation.float()
        )
        distances = torch.linalg.vector_norm(relative_positions, dim=-1)
        nearest_distance = distances.min(dim=-1).values
        caught = nearest_distance < self.tag_capture_distance
        record = {
            "step": step,
            "caught_b": caught.detach().cpu(),
            "mean_adv_to_good_b": nearest_distance.detach().cpu(),
        }
        return record, bool(caught.reshape(-1)[0].item())

    @staticmethod
    def _render_rgb_array(env):
        """Render both VMAS-style and PettingZoo-style TorchRL wrappers."""
        try:
            return env.render(mode="rgb_array")
        except TypeError:
            return env.render()

    def _compute_fitness(self, episode_reward: float, collision_count: int, total_steps: int,
                         initial_goal_dists: torch.Tensor = None,
                         final_goal_dists: torch.Tensor = None,
                         success: bool = False,
                         agent_collision_ratios: torch.Tensor = None,
                         pos_history=None,
                         rot_history=None,
                         target_pos_history=None,
                         caught_step_records=None,
                         first_env_caught_at=None,
                         obstacle_dist_history=None,
                         ) -> float:
        """Compute fitness value from episode data based on fitness mode.

        Args:
            episode_reward: total reward accumulated during the episode.
            collision_count: total number of obstacle collisions (from env).
            total_steps: number of environment steps in the episode.
            initial_goal_dists: per-agent distance to goal at episode start.
            final_goal_dists: per-agent distance to goal at episode end.
            success: True if all agents reached their goals.
            agent_collision_ratios: (n_agents,) per-agent fraction of steps
                where the agent was within safety_distance of a neighbor
                (within neighbor_radius). Used by ``navigation_avoidance_v2``.
            pos_history: list of (n_agents, 2) CPU tensors, one per step.
                Required for ``flocking_global`` and ``flocking_orbit``.
            rot_history: list of (n_agents,) CPU tensors (heading angle in
                radians, one per agent per step). Required for
                ``flocking_global`` and ``flocking_orbit``.
            target_pos_history: list of (2,) CPU tensors (target absolute
                position per step). Required for ``flocking_orbit``.
        """
        if self.fitness_mode == "navigation":
            return episode_reward
        elif self.fitness_mode == "navigation_avoidance":
            return episode_reward - self.collision_penalty_weight * collision_count
        elif self.fitness_mode == "navigation_v2":
            if initial_goal_dists is None or final_goal_dists is None:
                raise ValueError("navigation_v2 mode requires initial/final goal distances")
            eps = 1e-6
            progress = ((initial_goal_dists - final_goal_dists).clamp(min=0.0)
                        / (initial_goal_dists + eps))
            mask = (initial_goal_dists > eps)
            if mask.any():
                mean_progress = progress[mask].mean().item()
            else:
                mean_progress = 0.0
            success_term = 1.0 if success else 0.0
            mean_final = final_goal_dists.mean().item() if mask.any() else 0.0
            final_term = -mean_final
            w_progress = 3.0
            w_success = 5.0
            w_final = 1.0
            return w_progress * mean_progress + w_success * success_term + w_final * final_term
        elif self.fitness_mode == "navigation_avoidance_v2":
            if initial_goal_dists is None or final_goal_dists is None:
                raise ValueError("navigation_avoidance_v2 mode requires initial/final goal distances")
            if agent_collision_ratios is None:
                raise ValueError("navigation_avoidance_v2 mode requires agent_collision_ratios")
            eps = 1e-6
            progress = ((initial_goal_dists - final_goal_dists).clamp(min=0.0)
                        / (initial_goal_dists + eps))
            mask = (initial_goal_dists > eps)
            if mask.any():
                mean_progress = progress[mask].mean().item()
            else:
                mean_progress = 0.0
            success_term = 1.0 if success else 0.0
            mean_final = final_goal_dists.mean().item() if mask.any() else 0.0
            final_term = -mean_final
            # Per-agent collision penalty: mean across agents of the
            # fraction of steps each agent spent in collision.
            mean_collision_ratio = agent_collision_ratios.mean().item()
            w_progress = 3.0
            w_success = 5.0
            w_final = 1.0
            w_collision = self.collision_penalty_weight
            return (w_progress * mean_progress
                    + w_success * success_term
                    + w_final * final_term
                    - w_collision * mean_collision_ratio)
        elif self.fitness_mode == "navigation_obs_avoidance":
            # Sparse fitness: give a fixed reward for "arrived safely",
            # otherwise charge for how far away the agent ended up plus
            # how much of the episode it spent inside the safety band of
            # an obstacle. The composite is just two scalars + a binary
            # success term so that HAN's online weight updates have a
            # simple target to converge to.
            if initial_goal_dists is None or final_goal_dists is None:
                raise ValueError(
                    "navigation_obs_avoidance mode requires initial/final "
                    "goal distances"
                )
            if obstacle_dist_history is None:
                raise ValueError(
                    "navigation_obs_avoidance mode requires "
                    "obstacle_dist_history"
                )
            if len(obstacle_dist_history) == 0 or total_steps <= 0:
                # Degenerate: nothing to evaluate.
                return 0.0

            # Episode-level collision flag: did the agent ever touch an
            # obstacle (distance <= d_min)?
            d_min = (
                self.obstacle_agent_radius
                + self.obstacle_obstacle_radius
            )
            d_safe = max(self.obstacle_safety_distance, d_min + 1e-6)
            obs_dists = torch.stack(
                [d.float() for d in obstacle_dist_history]
            )
            collided = bool((obs_dists <= d_min + 1e-6).any().item())

            # Mean final distance to goal across the (single) agent.
            mean_final = final_goal_dists.mean().item()

            # Fraction of episode steps the agent spent inside any
            # obstacle's safety band.
            penalty_ratio = (
                (obs_dists <= d_safe).float().mean().item()
            )

            if success and not collided:
                # The "good" outcome: agent reached the goal without
                # touching any obstacle. Single, easy-to-optimise target.
                return float(self.success_reward)

            # Failure branch: agent did not arrive safely. Encourage
            # proximity to the goal while penalising how much of the
            # episode was spent in the safety band.
            fitness = (
                - self.final_weight * mean_final
                - self.penalty_weight * penalty_ratio
            )
            # Extra timeout penalty when the episode ran the full
            # ``total_steps`` without success (i.e. the agent wandered
            # or got stuck rather than converging).
            if not success and total_steps >= self.experiment.max_steps:
                fitness -= self.nav_timeout_penalty
            return float(fitness)
        elif self.fitness_mode == "flocking_global":
            if pos_history is None or rot_history is None:
                raise ValueError(
                    "flocking_global mode requires pos_history and rot_history"
                )
            return self._compute_flocking_fitness(pos_history, rot_history)
        elif self.fitness_mode == "flocking_orbit":
            if pos_history is None or rot_history is None or target_pos_history is None:
                raise ValueError(
                    "flocking_orbit mode requires pos_history, rot_history, "
                    "and target_pos_history"
                )
            return self._compute_flocking_orbit_fitness(
                pos_history, rot_history, target_pos_history
            )
        elif self.fitness_mode == "flocking_lf_arrival":
            if pos_history is None or target_pos_history is None:
                raise ValueError(
                    "flocking_lf_arrival mode requires pos_history and "
                    "target_pos_history"
                )
            return self._compute_flocking_lf_arrival_fitness(
                pos_history, target_pos_history
            )
        elif self.fitness_mode == "flocking_light_intensity":
            if pos_history is None or target_pos_history is None:
                raise ValueError(
                    "flocking_light_intensity mode requires pos_history "
                    "and target_pos_history"
                )
            return self._compute_flocking_light_intensity_fitness(
                pos_history, target_pos_history
            )
        elif self.fitness_mode == "flocking_signal_intensity":
            if pos_history is None or target_pos_history is None:
                raise ValueError(
                    "flocking_signal_intensity mode requires pos_history "
                    "and target_pos_history"
                )
            return self._compute_flocking_signal_intensity_fitness(
                pos_history, target_pos_history
            )
        elif self.fitness_mode == "simple_tag_capture":
            return self._compute_simple_tag_capture_fitness(
                initial_goal_dists=initial_goal_dists,
                caught_step_records=caught_step_records,
                first_env_caught_at=first_env_caught_at,
                total_steps=total_steps,
                collision_count=collision_count,
            )
        elif self.fitness_mode == "hgn_formation_v1":
            return self._compute_hgn_formation_fitness(
                pos_history=pos_history,
                target_pos_history=target_pos_history,
                total_steps=total_steps,
            )
        else:
            raise ValueError(f"Unknown fitness mode: {self.fitness_mode}")

    def _count_connected_components(self, adj: torch.Tensor) -> int:
        """Count connected components of an undirected adjacency matrix.

        Args:
            adj: (N, N) bool tensor where adj[i, j] is True if i and j
                are connected. Diagonal entries are ignored.
        Returns:
            Number of connected components (>= 1).
        """
        N = adj.shape[0]
        if N == 0:
            return 0
        # Build undirected edges (upper triangular) and run DSU.
        parent = list(range(N))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Move to CPU once; graph is small.
        adj_cpu = adj.detach().cpu().bool()
        iu, iv = torch.triu_indices(N, N, offset=1)
        edge_mask = adj_cpu[iu, iv]
        edge_i = iu[edge_mask].tolist()
        edge_j = iv[edge_mask].tolist()
        for a, b in zip(edge_i, edge_j):
            union(a, b)

        roots = {find(x) for x in range(N)}
        return len(roots)

    def _compute_flocking_fitness(self, pos_history, rot_history) -> float:
        """Ramos et al. 2019 Global flocking fitness (Eq. 11).

        Fg = (1/T) * sum_t [ Cg(t) + S(t) + Ag(t) ] + M

        Args:
            pos_history: list of length T with (N, 2) CPU tensors.
            rot_history: list of length T with (N,) CPU tensors (heading
                angle in radians, already patched from velocity direction).
        """
        if not pos_history:
            return 0.0
        T = len(pos_history)
        # Sanity: every step should have N agents.
        N = pos_history[0].shape[0]
        if N < 2:
            # Degenerate: at most one agent, no cohesion/alignment to compute.
            Ag = 0.0
            Cg = 0.0
            S = 1.0
            return Ag + Cg + S + 0.0

        fitness_sum = 0.0
        # Cache neighbor_radius/safety_distance as python floats for speed.
        nr = float(self.neighbor_radius)
        sd = float(self.safety_distance)
        eye = torch.eye(N, dtype=torch.bool)

        for t in range(T):
            pos = pos_history[t]                            # (N, 2)
            rot = rot_history[t]                            # (N,)

            # Ag(t) — global alignment order parameter.
            vx = torch.cos(rot)
            vy = torch.sin(rot)
            Ag = torch.sqrt(vx.mean() ** 2 + vy.mean() ** 2).item()

            # Cg(t) — global cohesion via connected components over
            # the neighbor-radius graph.
            diff = pos.unsqueeze(0) - pos.unsqueeze(1)      # (N, N, 2)
            dist = torch.linalg.vector_norm(diff, dim=-1)   # (N, N)
            adj = (dist < nr) & (~eye)
            num_groups = CmaesHanOptimizer._count_connected_components(self, adj)
            Cg = 1.0 / max(int(num_groups), 1)

            # S(t) — separation: 1 - fraction of agents colliding.
            # Exclude self-pairs (diagonal is 0 and would falsely count as
            # a collision for every agent).
            dist_for_collision = dist.masked_fill(eye, float("inf"))
            in_collision = (dist_for_collision < sd).any(dim=-1)  # (N,)
            S = 1.0 - in_collision.float().mean().item()

            fitness_sum += Cg + S + Ag

        Fg = fitness_sum / T

        # M — movement bonus from Ramos et al. (avg displacement vs target).
        d = torch.linalg.vector_norm(
            pos_history[-1] - pos_history[0], dim=-1
        ).mean().item()
        D = float(self.movement_target_displacement)
        M = min(d / D, 1.0) if D > 0 else 0.0

        return Fg + M

    def _compute_flocking_orbit_fitness(self, pos_history, rot_history,
                                         target_pos_history) -> float:
        """Orbit-target flocking fitness.

        Inspired by VMAS flocking's distance-shaping reward and Ramos
        et al. 2019's global flocking fitness, but biased toward forming
        an orbit around the moving ``_target`` rather than free flocking.

        F_orbit = (1/T) * sum_t [ At(t) + Dt(t)
                                  + w_C * Cg(t) + w_S * S(t) ]

        where:
          At(t)  - radial-tangential alignment. Each agent's velocity
                   direction is dot-compared against the tangent of the
                   agent-to-target ray (CCW 90 deg rotation). Agents
                   flying along the tangent get At ~ 1, agents flying
                   radially get At ~ 0.5, agents on the wrong tangent
                   side get At ~ 0.
          Dt(t)  - Gaussian band around ``orbit_radius`` of the distance
                   from each agent to the target. Encourages staying on
                   the orbit rather than crashing into / fleeing the
                   target.
          Cg(t)  - global cohesion: 1 / num_connected_components over
                   the neighbor-radius graph (same as flocking_global).
          S(t)   - separation: 1 - (fraction of agents colliding).

        Args:
            pos_history: list of length T with (N, 2) CPU tensors.
            rot_history: list of length T with (N,) CPU tensors (heading
                angle per agent, already patched from velocity direction).
            target_pos_history: list of length T with (2,) CPU tensors
                (target absolute position at each step).
        """
        T = len(pos_history)
        if T == 0:
            return 0.0
        N = pos_history[0].shape[0]
        if N < 1:
            return 0.0

        # Weighting: connectivity & separation are softer than the
        # orbit terms so they don't dominate the fitness.
        # Adjusted weights (方案 E): stronger At separation, weaker cohesion.
        w_C = 0.2
        # w_S = 1.0
        w_S = 0.8
        r_star = float(self.orbit_radius)
        r_sigma = float(self.orbit_radius_tolerance)
        dt_floor = float(self.dt_floor)
        nr = float(self.neighbor_radius)
        sd = float(self.safety_distance)
        eye = torch.eye(N, dtype=torch.bool)
        eps = 1e-6

        sum_At = 0.0
        sum_Dt = 0.0
        sum_Cg = 0.0
        sum_S = 0.0

        for t in range(T):
            pos = pos_history[t]                       # (N, 2)
            rot = rot_history[t]                       # (N,)
            tgt = target_pos_history[t]                # (2,)

            # --- At: radial-tangential alignment ---
            # r_vec: from target to agent (N, 2). Normalize to unit
            # radial direction.
            r_vec = pos - tgt                          # (N, 2)
            r_norm = torch.linalg.vector_norm(r_vec, dim=-1, keepdim=True)
            r_unit = r_vec / (r_norm + eps)            # (N, 2)
            # Tangent = CCW 90 deg rotation of r_unit.
            # rot90_CCW([x, y]) = [-y, x]
            tangent = torch.stack([-r_unit[:, 1], r_unit[:, 0]], dim=-1)
            # Agent velocity direction from heading angle.
            v_dir = torch.stack([torch.cos(rot), torch.sin(rot)], dim=-1)
            # Dot product along the tangent.
            dot = (v_dir * tangent).sum(dim=-1)        # (N,)
            # Map from [-1, 1] -> [0, 1] (clipped) then average.
            at = ((dot + 1.0) * 0.5).clamp(0.0, 1.0)
            # If an agent is exactly on the target (r_norm ~= 0), tangent
            # is undefined: mask it out by setting At contribution to the
            # neutral 0.5 via r_norm > eps gate.
            valid = (r_norm.squeeze(-1) > eps).float()
            at_sum = (at * valid).sum().item()
            at_count = valid.sum().item()
            At = at_sum / at_count if at_count > 0 else 0.5

            # --- Dt: Gaussian distance band ---
            # Higher when r_norm is close to r_star; floored at dt_floor.
            r_dist = r_norm.squeeze(-1)                # (N,)
            Dt_raw = torch.exp(-((r_dist - r_star) ** 2) / (2.0 * r_sigma ** 2))
            Dt = max(Dt_raw.mean().item(), dt_floor)

            # --- Cg: connected components over neighbor graph ---
            diff = pos.unsqueeze(0) - pos.unsqueeze(1)
            dist = torch.linalg.vector_norm(diff, dim=-1)
            adj = (dist < nr) & (~eye)
            num_groups = CmaesHanOptimizer._count_connected_components(self, adj)
            Cg = 1.0 / max(int(num_groups), 1)

            # --- S: separation ---
            dist_for_collision = dist.masked_fill(eye, float("inf"))
            in_collision = (dist_for_collision < sd).any(dim=-1)
            S = 1.0 - in_collision.float().mean().item()

            sum_At += At
            sum_Dt += Dt
            sum_Cg += Cg
            sum_S += S

        F_orbit = (1.5 * sum_At + sum_Dt + w_C * sum_Cg + w_S * sum_S) / T
        # Range: At, Dt, Cg, S each in [0, 1]; weights keep F_orbit in [0, 4].
        # Adjusted weights: 1.5×At (emphasize tangential alignment), 0.2×Cg
        # (reduce clustering reward), 0.8×S (strengthen separation penalty).
        return F_orbit

    def _compute_flocking_lf_arrival_fitness(
        self, pos_history, target_pos_history
    ) -> float:
        """Leader/follower arrival fitness for ``flocking_lf``.

        The agent swarm must converge to a static target landmark. The
        target is visible only to leader agents; followers must rely on
        reaching the target via information propagation through HAN.

        Fitness = -mean_over_agents(final_dist_to_target)
        (i.e. the optimizer *minimizes* mean final distance; the sign
        flip in ``fitness()`` flips it back to a positive fitness value
        that grows as agents get closer to the target).

        Args:
            pos_history: list of length T with (N, 2) CPU tensors.
            target_pos_history: list of length T with (2,) CPU tensors
                (target absolute position per step; constant for
                ``flocking_lf`` since the target is a static landmark).
        """
        T = len(pos_history)
        if T == 0:
            return 0.0
        final_pos = pos_history[-1]                   # (N, 2)
        final_target = target_pos_history[-1]         # (2,)
        # Per-agent distance to target at episode end.
        dist = torch.linalg.vector_norm(
            final_pos - final_target.unsqueeze(0), dim=-1
        )                                             # (N,)
        mean_final_dist = dist.mean().item()
        # Higher fitness = closer to target (sign flipped in fitness()).
        return -mean_final_dist

    def _compute_flocking_light_intensity_fitness(
        self, pos_history, target_pos_history
    ) -> float:
        """Time-averaged light-intensity fitness for ``flocking_light``.

        Φ(pos) = 1 / (||pos - target|| + ε)

        Fitness = mean_over_time( mean_over_agents( Φ(pos_t) ) )

        Rewarding the instantaneous field value at every step (rather
        than only the final distance) means the swarm is encouraged to
        move toward the target quickly AND stay there — no late-episode
        surge can game it. ``ε`` is read from the scenario so this stays
        consistent with the field the agents actually observe.

        Args:
            pos_history: list of length T with (N, 2) CPU tensors.
            target_pos_history: list of length T with (2,) CPU tensors.
        """
        T = len(pos_history)
        if T == 0:
            return 0.0
        eps = 0.01
        core = self._get_vmas_core()
        scenario = getattr(core, "scenario", None)
        if scenario is not None:
            eps = float(getattr(scenario, "light_eps", eps))

        intensity_sum = 0.0
        for t in range(T):
            pos = pos_history[t]                    # (N, 2)
            target = target_pos_history[t]          # (2,)
            dist = torch.linalg.vector_norm(
                pos - target.unsqueeze(0), dim=-1
            )                                       # (N,)
            intensity = 1.0 / (dist + eps)          # (N,)
            intensity_sum += intensity.mean().item()
        return intensity_sum / T

    def _compute_flocking_signal_intensity_fitness(
        self, pos_history, target_pos_history
    ) -> float:
        """Time-averaged light-intensity fitness for ``flocking_signal``.

        Identical to ``_compute_flocking_light_intensity_fitness`` in
        formula, but ``flocking_signal`` has a moving target so the
        light field Φ is recomputed per step using the current target
        position.

        Fitness = mean_over_time( mean_over_agents( Φ(pos_t, t) ) )

        Args:
            pos_history: list of length T with (N, 2) CPU tensors.
            target_pos_history: list of length T with (2,) CPU tensors
                (target absolute position per step — varies because the
                target moves).
        """
        T = len(pos_history)
        if T == 0:
            return 0.0
        eps = 0.01
        core = self._get_vmas_core()
        scenario = getattr(core, "scenario", None)
        if scenario is not None:
            eps = float(getattr(scenario, "light_eps", eps))

        intensity_sum = 0.0
        for t in range(T):
            pos = pos_history[t]
            target = target_pos_history[t]
            dist = torch.linalg.vector_norm(
                pos - target.unsqueeze(0), dim=-1
            )
            intensity = 1.0 / (dist + eps)
            intensity_sum += intensity.mean().item()
        return intensity_sum / T

    def _compute_simple_tag_capture_fitness(
        self,
        initial_goal_dists: torch.Tensor = None,
        caught_step_records: list = None,
        first_env_caught_at: int = None,
        total_steps: int = 0,
        collision_count: int = 0,
    ) -> float:
        """Capture fitness for ``simple_tag_v1`` (adversary pursuit).

        Score components (env index 0; the per-batch-env metric the
        CMA-ES evaluator reads from):

        + ``catch_term``  = ``catch_reward`` if the good agent was
            caught at any step (else 0).
        - ``proximity_term`` = mean per-step distance from the closest
          adversary to each good agent (env 0). Lower is better.
        - ``timeout_term``  = ``timeout_penalty`` if the rollout hit
            ``max_steps`` without a catch (else 0).

        The exact reward weights default to ``(5.0, 1.0, 1.0)``. These
        place catch above all other signals, while still rewarding
        faster convergence and proximity pressure before the catch.

        Args:
            initial_goal_dists: (n_agents,) obs-distance at step 0.
                Currently unused but kept for symmetry with the other
                fitness modes.
            caught_step_records: list of dicts collected by
                ``_run_one_episode`` when ``fitness_mode ==
                'simple_tag_capture'``. Each entry has ``"step"``,
                ``"caught_b"`` ((B,) bool) and ``"mean_adv_to_good_b"``
                ((B,) float). May be None or empty.
            first_env_caught_at: int step at which env index 0 first
                caught the good agent (``None`` if it never did).
            total_steps: total steps the rollout actually executed.
            collision_count: total collisions recorded by the env
                reward signal (kept for symmetry, currently unused).
        """
        if not caught_step_records:
            return -float(self.timeout_penalty)

        # Per-step mean pursuit distance for env index 0.
        per_step_dist_e0 = torch.stack(
            [rec["mean_adv_to_good_b"][0] for rec in caught_step_records],
            dim=0,
        )
        mean_proximity = per_step_dist_e0.mean().item()

        caught = first_env_caught_at is not None
        catch_term = float(self.catch_reward) if caught else 0.0
        timeout_term = float(self.timeout_penalty) if (
            not caught and total_steps >= 1
        ) else 0.0

        # Fitness: maximize catch_term, minimize proximity & timeout.
        return (
            catch_term
            - self.proximity_weight * mean_proximity
            - timeout_term
        )

    def _compute_hgn_formation_fitness(
        self,
        pos_history: list = None,
        target_pos_history: list = None,
        total_steps: int = 0,
    ) -> float:
        """HGN formation-control fitness (``hgn_formation_v1``).

        Sparse + single-target design, mirroring ``obs_avoidance_fitness.md``
        v3 — one plateau target, one monotonic failure signal, no temporal
        segmentation:

            if reached AND not collided:
                fitness = +formation_success_reward  (default +5.0)
            else:
                fitness = -formation_final_weight * mean_dist
                        - formation_collision_penalty (if tail collided)
                        - formation_timeout_penalty   (if ran max_steps)

        ``target_pos_history`` is unused here — targets are *static* (or
        moving uniformly in moving-target mode), and the scenario caches
        them on the env. We recover them from the simulator's scenario
        object via ``self._get_vmas_core().scenario.formation_targets``.
        """
        if not pos_history:
            return 0.0

        final_pos = pos_history[-1]                       # (N, 2)
        core = getattr(self, "_vmas_core", None)
        if core is None:
            core = self._get_vmas_core()
        scenario = getattr(core, "scenario", None)
        if scenario is None or not hasattr(scenario, "formation_targets"):
            # No scenario targets: fall back to first target_pos_history
            # entry (set by reset_world_at / pre_step). This branch is only
            # hit if the user pointed ``hgn_formation_v1`` at a non-formation
            # env.
            if not target_pos_history:
                return 0.0
            target = target_pos_history[-1]              # (2,) single target
            # Cannot compare per-agent distances with a single global
            # target. Return 0 — caller should pick the right mode.
            return 0.0

        target = scenario.formation_targets                # (N, 2)
        if target.device != final_pos.device:
            target = target.to(final_pos.device)

        # Per-agent distance to assigned formation slot.
        d = torch.linalg.vector_norm(final_pos - target, dim=-1)   # (N,)
        mean_dist = d.mean().item()

        # Reached iff every agent is within reach_radius of its slot.
        reached = bool((d <= self.formation_reach_radius).all().item())

        # Collided iff any pair of agents got within 2*agent_radius in the
        # tail of the episode.
        collided = self._formation_tail_collided(pos_history)

        if reached and not collided:
            return float(self.success_reward)

        # Failure branch.
        fitness = -self.final_weight * mean_dist
        if collided:
            fitness -= self.formation_collision_penalty
        if total_steps >= self.experiment.max_steps and not reached:
            fitness -= self.formation_timeout_penalty
        return float(fitness)

    def _formation_tail_collided(self, pos_history: list) -> bool:
        """Return True if any pair of agents touched in the tail of the
        episode. Pair collision radius uses the VMAS default 0.05 + 0.05
        (= 2 * default agent_radius)."""
        if len(pos_history) < 2:
            return False
        tail = max(1, int(len(pos_history) * self.formation_tail_frac))
        tail_history = pos_history[-tail:]
        collision_dist = 2 * 0.05  # default agent_radius
        for pos in tail_history:
            if pos.shape[0] < 2:
                continue
            diff = pos.unsqueeze(0) - pos.unsqueeze(1)
            dist = torch.linalg.vector_norm(diff, dim=-1)
            N = dist.shape[0]
            eye = torch.eye(N, device=dist.device, dtype=torch.bool)
            dist = dist.masked_fill(eye, float("inf"))
            if (dist < collision_dist).any().item():
                return True
        return False

    def _run_one_episode(self, env, group, max_steps, policy, on_frame=None):
        """Run a single episode and return everything needed by any fitness mode."""
        td = env.reset()

        # VMAS fitness modes read simulator state directly. PettingZoo tag
        # instead uses the good-agent observation, so it does not depend on
        # wrapper internals.
        if self.tag_evader_group is None:
            self._get_vmas_core()

        obs = td.get((group, "observation"))
        initial_goal_dists = torch.linalg.vector_norm(
            obs[..., :2].float(), dim=-1
        )
        # For the navigation_obs_avoidance task the observation layout is
        # [pos(2), vel(2), goal_rel(2), nearest_obstacle_rel(2), has_flag(1)],
        # so obs[..., :2] is the absolute agent position rather than the
        # goal-relative offset. We instead compute initial / final goal
        # distances from simulator state (single-env, env index 0).
        use_sim_goal_dist = self.fitness_mode == "navigation_obs_avoidance"

        episode_reward = 0.0
        collision_count = 0
        done = False
        step = 0
        success = False

        # Per-agent counter of steps where the agent was within
        # safety_distance of any neighbor inside neighbor_radius.
        # Shape: (n_agents,). Set up after we observe the first batch.
        agent_collision_steps = None
        n_agents = None

        # Per-step absolute positions (N, 2) and heading angles (N,) on
        # CPU. Only populated when flocking_global fitness mode is active;
        # the per-step cost is tiny (N ~ 4..10, T ~ 100..300) so we
        # always allocate to keep _run_one_episode uniform.
        pos_history = []
        rot_history = []  # type: list[torch.Tensor]
        target_pos_history = []  # type: list[torch.Tensor]
        initial_pos = None

        # Captured by ``simple_tag_capture`` fitness:
        #   caught_step_records: list of dicts {step_idx, caught_b,
        #                                     mean_adv_to_good, adv_spread}
        # single-env metadata used to break out of the loop fast and to
        # expose the catch step to the fitness function.
        caught_step_records = []
        # Stop the rollout as soon as the per-env "first env" has caught
        # the good agent (saves CMA-ES evaluation time on successful
        # candidates).
        first_env_caught_at = None

        # Per-step obstacle distance (scalar, env index 0) used by the
        # navigation_obs_avoidance fitness mode. Populated only when
        # the active fitness mode requests it.
        obstacle_dist_history = []

        while not done and step < max_steps:
            td = policy(td)
            self._set_pettingzoo_tag_evader_action(td)
            td = env.step(td)

            if on_frame is not None:
                on_frame(td, step)

            reward = td.get(("next", group, "reward"))
            episode_reward += reward.sum().item()

            info = td.get(("next", group, "info"), None)
            if info is not None:
                col_rew = info.get("agent_collision_rew")
                if col_rew is not None:
                    collision_count += (col_rew < 0).sum().item()

            if (
                self.fitness_mode == "simple_tag_capture"
                and self.tag_evader_group is not None
            ):
                record, caught_now = self._record_pettingzoo_tag_step(td, step)
                caught_step_records.append(record)
                if first_env_caught_at is None and caught_now:
                    first_env_caught_at = step

            # Inter-agent collision detection: use ABSOLUTE agent
            # positions read from the vmas core environment. We cannot
            # recover absolute positions from obs[:2] because each agent
            # has its own goal (obs[:2] = agent.pos - agent.goal), and
            # the goal offset does NOT cancel when subtracting two
            # different agents' obs[:2]. The pairwise difference
            # (obs[i,:2] - obs[j,:2]) = (pos[i] - pos[j]) + (goal[j] -
            # goal[i]), which is biased by per-agent goal offsets.
            core = getattr(self, "_vmas_core", None)
            if core is not None and hasattr(core, "agents"):
                if n_agents is None:
                    n_agents = len(core.agents)
                    agent_collision_steps = torch.zeros(n_agents, device=self.device)
                # Stack absolute positions: (num_envs, n_agents, 2).
                # For collision we use the FIRST parallel env (env index
                # 0) since the optimizer's _run_one_episode is a
                # single-trajectory rollout.
                pos_stack = torch.stack(
                    [a.state.pos[0] for a in core.agents], dim=0
                )  # (n_agents, 2)
                a = pos_stack.shape[0]
                diff = pos_stack.unsqueeze(0) - pos_stack.unsqueeze(1)  # (a, a, 2)
                dist = torch.linalg.vector_norm(diff, dim=-1)           # (a, a)
                eye = torch.eye(a, device=dist.device, dtype=torch.bool)
                dist = dist.masked_fill(eye, float("inf"))
                in_radius = dist < self.neighbor_radius
                in_collision = dist < self.safety_distance
                agent_in_collision = (in_radius & in_collision).any(dim=-1)  # (a,)
                agent_collision_steps += agent_in_collision.float()

                # Flocking fitness bookkeeping: patch each agent's heading
                # from its current velocity direction (VMAS Holonomic does
                # not update state.rot automatically), then record per-step
                # absolute position + heading for later aggregation.
                if self.patch_heading_from_vel:
                    for agent in core.agents:
                        vel0 = agent.state.vel[0]                # (2,)
                        # atan2(vy, vx) → scalar; assign into rot[0, 0]
                        # which is shape (batch_dim, 1).
                        agent.state.rot[0, 0] = torch.atan2(
                            vel0[1], vel0[0]
                        )
                pos_cpu = pos_stack.detach().cpu()
                rot_cpu = torch.stack(
                    [a.state.rot[0, 0] for a in core.agents], dim=0
                ).detach().cpu()
                if initial_pos is None:
                    initial_pos = pos_cpu.clone()
                pos_history.append(pos_cpu)
                rot_history.append(rot_cpu)

                # navigation_obs_avoidance bookkeeping: per-step minimum
                # distance from the agent to any obstacle (env index 0).
                # Obstacles are landmarks that are NOT the goal. Only
                # collected when the active fitness mode requests it.
                if use_sim_goal_dist:
                    scenario = getattr(core, "scenario", None)
                    obstacles = (
                        scenario.obstacles if scenario is not None else []
                    )
                    if obstacles:
                        obs_pos_stack = torch.stack(
                            [o.state.pos[0] for o in obstacles], dim=0
                        )                                        # (N, 2)
                        diff_obs = (
                            pos_stack[0].unsqueeze(0)
                            - obs_pos_stack
                        )                                        # (N, 2)
                        obs_dist = torch.linalg.vector_norm(
                            diff_obs, dim=-1
                        )                                        # (N,)
                        obstacle_dist_history.append(
                            obs_dist.min().detach().cpu()
                        )
                    else:
                        # No obstacles: record a large value so the
                        # exp(-k*r) penalty is ~0.
                        obstacle_dist_history.append(
                            torch.tensor(float(self.obstacle_safety_distance * 4))
                        )

                # Record target absolute position (env index 0) for
                # flocking_orbit. The target is the action_script-driven
                # agent attached to the VMAS flocking scenario.
                target = getattr(getattr(core, "scenario", None), "_target", None)
                if target is not None and hasattr(target.state, "pos"):
                    target_pos_history.append(
                        target.state.pos[0].detach().cpu()
                    )

                # simple_tag-style bookkeeping: only active when
                # ``fitness_mode == 'simple_tag_capture'``. Records
                # per-step catch flag (per batch env) and per-step
                # mean pursuit distance for env index 0.
                if (
                    self.fitness_mode == "simple_tag_capture"
                    and self.tag_evader_group is None
                ):
                    scenario = getattr(core, "scenario", None)
                    advs = scenario.adversaries() if scenario is not None else []
                    goods = scenario.good_agents() if scenario is not None else []
                    if advs and goods:
                        adv_pos_all = torch.stack(
                            [a.state.pos for a in advs], dim=1
                        )    # (B, n_adv, 2)
                        good_pos_all = torch.stack(
                            [g.state.pos for g in goods], dim=1
                        )  # (B, n_good, 2)
                        # Pairwise distance per batch row.
                        diff_ag = adv_pos_all.unsqueeze(2) - good_pos_all.unsqueeze(1)
                        dist_ag = torch.linalg.vector_norm(diff_ag, dim=-1)  # (B, n_adv, n_good)
                        # Contact: any (adv, good) pair within radii sum.
                        adv_r = torch.tensor(
                            [a.shape.radius for a in advs],
                            device=self.device, dtype=adv_pos_all.dtype,
                        ).view(1, -1, 1)
                        good_r = torch.tensor(
                            [g.shape.radius for g in goods],
                            device=self.device, dtype=adv_pos_all.dtype,
                        ).view(1, 1, -1)
                        contact_ag = dist_ag < (adv_r + good_r)            # (B, n_adv, n_good)
                        caught_step_b = contact_ag.any(dim=(1, 2))         # (B,)
                        min_per_good = dist_ag.min(dim=1).values           # (B, n_good)
                        mean_adv_to_good_step = min_per_good.mean(dim=-1)  # (B,)
                        caught_step_records.append({
                            "step": step,
                            "caught_b": caught_step_b.detach().cpu(),
                            "mean_adv_to_good_b": mean_adv_to_good_step.detach().cpu(),
                        })
                        if (first_env_caught_at is None
                                and bool(caught_step_b[0].item())):
                            first_env_caught_at = step

            # IMPORTANT: only env index 0 is the one we evaluate. The
            # rollout is a parallel batch and other batch rows may
            # terminate early (initial-spawn collisions, etc.). Using
            # ``any()`` over the whole batch would mistakenly end the
            # rollout for env[0] as well.
            done_t = td.get(("next", "done"))
            if done_t.ndim > 1:
                done = bool(done_t[0].item())
            else:
                done = bool(done_t.item())
            if done:
                success = True
            td = td.get("next")
            step += 1

            # For simple_tag-style tasks: once env[0] has caught the
            # good agent, we can break the rollout early to save
            # CMA-ES evaluation time (the catch flag is the dominant
            # fitness signal). Other fitness modes are unaffected.
            if (self.fitness_mode == "simple_tag_capture"
                    and first_env_caught_at is not None
                    and first_env_caught_at >= 0
                    and len(caught_step_records) > 0):
                break

        final_obs = td.get((group, "observation"))
        final_goal_dists = torch.linalg.vector_norm(
            final_obs[..., :2].float(), dim=-1
        )
        # When the obs layout encodes absolute positions rather than
        # goal-relative offsets, recompute goal distances from simulator
        # state. ``initial_goal_dists`` was also recomputed below to
        # keep the contract consistent across modes.
        if use_sim_goal_dist and getattr(self, "_vmas_core", None) is not None:
            core = self._vmas_core
            agent_pos_init = core.agents[0].state.pos[0]
            agent_pos_final = agent_pos_init  # default: fall back if step==0
            goal_pos = core.agents[0].goal.state.pos[0]
            # Walk the pos_history collected in this episode: the last
            # CPU entry is the post-step absolute position of the agent.
            # If the loop never ran, fall back to the current state.pos.
            if pos_history:
                # pos_history is parallel-agent (n_agents, 2); for single
                # agent tasks index 0 is our agent.
                agent_pos_final = pos_history[-1][0].to(self.device)
            init_dist = torch.linalg.vector_norm(
                agent_pos_init - goal_pos
            )
            final_dist = torch.linalg.vector_norm(
                agent_pos_final - goal_pos
            )
            initial_goal_dists = init_dist.unsqueeze(0)
            final_goal_dists = final_dist.unsqueeze(0)

        # Per-agent collision ratio = (steps in collision) / total_steps.
        # If n_agents is 1, no inter-agent collisions possible.
        if n_agents is not None and n_agents > 1 and step > 0:
            agent_collision_ratios = agent_collision_steps / float(step)
        else:
            agent_collision_ratios = torch.zeros(n_agents if n_agents is not None else 1,
                                                 device=self.device)

        return {
            "episode_reward": episode_reward,
            "collision_count": collision_count,
            "step": step,
            "success": success,
            "initial_goal_dists": initial_goal_dists,
            "final_goal_dists": final_goal_dists,
            "agent_collision_ratios": agent_collision_ratios,
            "pos_history": pos_history,
            "rot_history": rot_history,
            "target_pos_history": target_pos_history,
            "initial_pos": initial_pos,
            # simple_tag-specific: per-step records + catch step (None
            # if never caught).
            "caught_step_records": caught_step_records,
            "first_env_caught_at": first_env_caught_at,
            # navigation_obs_avoidance-specific: per-step min agent→obstacle
            # distance (CPU scalar, env index 0). Empty list if the
            # fitness mode does not request it.
            "obstacle_dist_history": obstacle_dist_history,
        }

    def fitness(self, x: np.ndarray) -> float:
        """Evaluate fitness of a candidate ABCD parameter vector."""
        t0 = time.time()
        self.set_abcd_from_vector(x)
        # reset_all_weights also clears the sliding windows and resets ticks.
        self.han_model.reset_all_weights()
        # Cache the VMAS core env when simulator-state fitness is in use.
        if self.tag_evader_group is None:
            self._get_vmas_core()

        fitnesses = []
        group = self.train_group
        env = self.experiment.test_env
        max_steps = self.experiment.max_steps

        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            for ep in range(self.n_eval_episodes):
                self.han_model.reset_all_weights()
                stats = self._run_one_episode(
                    env, group, max_steps, self.rollout_policy
                )
                ep_fitness = self._compute_fitness(
                    stats["episode_reward"], stats["collision_count"], stats["step"],
                    initial_goal_dists=stats["initial_goal_dists"],
                    final_goal_dists=stats["final_goal_dists"],
                    success=stats["success"],
                    agent_collision_ratios=stats["agent_collision_ratios"],
                    pos_history=stats["pos_history"],
                    rot_history=stats["rot_history"],
                    target_pos_history=stats["target_pos_history"],
                    caught_step_records=stats["caught_step_records"],
                    first_env_caught_at=stats["first_env_caught_at"],
                    obstacle_dist_history=stats["obstacle_dist_history"],
                )
                fitnesses.append(ep_fitness)

        elapsed = time.time() - t0
        self._last_eval_time = elapsed
        self._last_eval_steps = 0

        return -np.mean(fitnesses)

    def run(self) -> np.ndarray:
        """Run CMA-ES optimization."""
        import cma

        x0 = self.get_abcd_vector()
        total_params = len(x0)

        layers = self.han_model.get_all_han_layers()
        layer_info = ", ".join(
            f"L{i}: {l.in_features}x{l.out_features} ({l.num_abcd_params})"
            for i, l in enumerate(layers)
        )

        print(f"CMA-ES: optimizing {total_params} ABCD parameters across {len(layers)} layers")
        print(f"  Layers: [{layer_info}]")
        print(f"  pop_size={self.pop_size}, sigma0={self.sigma0}, max_gens={self.max_gens}")
        print(f"  fitness_mode={self.fitness_mode}, n_eval_episodes={self.n_eval_episodes}")
        print(f"  HanModel: window_size={layers[0].window_size}, "
              f"f_nn={self.han_model.f_nn}, f_hebb={self.han_model.f_hebb}, "
              f"update_interval={self.han_model._update_interval}")

        opts = {
            "popsize": self.pop_size,
            "maxiter": self.max_gens,
            "verbose": 1,
            "tolfun": 1e10,
            "tolfunhist": 1e10,
            "tolflatfitness": self.max_gens + 10,
            "tolstagnation": self.max_gens * 2,
            "tolx": 1e-12,
        }

        es = cma.CMAEvolutionStrategy(x0, self.sigma0, opts)
        total_start = time.time()

        for gen in range(1, self.max_gens + 1):
            self._current_gen = gen
            gen_start = time.time()
            solutions = es.ask()
            fitnesses = [self.fitness(x) for x in solutions]
            es.tell(solutions, fitnesses)
            es.logger.add()
            gen_time = time.time() - gen_start

            best_fitness = min(fitnesses)
            mean_fitness = sum(fitnesses) / len(fitnesses)
            elapsed_total = time.time() - total_start
            remaining = (elapsed_total / gen) * (self.max_gens - gen)

            if best_fitness < self._best_fitness_so_far:
                self._best_fitness_so_far = best_fitness
                self._best_abcd_so_far = es.result.xbest.copy()

            print(
                f"  Gen {gen}/{self.max_gens}: "
                f"best={-best_fitness:.2f}, mean={-mean_fitness:.2f} | "
                f"gen_time={gen_time:.1f}s | "
                f"elapsed={elapsed_total:.0f}s, ETA={remaining:.0f}s"
            )

        result = es.result
        best_x = result.xbest
        best_fitness = -result.fbest

        if self._best_abcd_so_far is not None and self._best_fitness_so_far < result.fbest:
            best_x = self._best_abcd_so_far
            best_fitness = -self._best_fitness_so_far

        total_time = time.time() - total_start
        print(f"\nCMA-ES finished in {total_time:.1f}s ({total_time/60:.1f}min):")
        print(f"  Best fitness: {best_fitness:.2f}")
        print(f"  Best ABCD norm: {np.linalg.norm(best_x):.4f}")

        self.set_abcd_from_vector(best_x)
        self.han_model.reset_all_weights()

        return best_x

    def plot_convergence(self, output_dir: str):
        """Generate CMA-ES convergence plot."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fit_path = os.path.join(os.getcwd(), "outcmaes", "fit.dat")
        if not os.path.exists(fit_path):
            print(f"  Skipping convergence plot: {fit_path} not found")
            return

        data = np.loadtxt(fit_path, comments="%")
        # np.loadtxt returns a 1-D array when the file has a single row
        # (e.g. max_gens=1). Normalize to 2-D so column indexing works.
        if data.ndim == 1:
            data = data.reshape(1, -1)
        gen = data[:, 0]
        bestever = data[:, 4]
        best = data[:, 5]
        median = data[:, 6]
        worst = data[:, 7]
        sigma = data[:, 2]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(gen, -best, label="Best", color="blue")
        ax1.plot(gen, -bestever, label="Best Ever", color="green", linestyle="--")
        ax1.plot(gen, -median, label="Median", color="orange", alpha=0.7)
        ax1.fill_between(gen, -worst, -best, alpha=0.15, color="blue")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness")
        ax1.set_title(f"CMA-ES Convergence (HAN, mode={self.fitness_mode})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.semilogy(gen, sigma, color="red")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Sigma (step size)")
        ax2.set_title("CMA-ES Step Size")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "cmaes_convergence.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Convergence plot saved to: {plot_path}")

    def apply_best_so_far(self):
        if self._best_abcd_so_far is not None:
            self.set_abcd_from_vector(self._best_abcd_so_far)
            self.han_model.reset_all_weights()
            return self._best_abcd_so_far
        return None

    def save(self, output_dir: str):
        """Save optimization results."""
        abcd_dir = os.path.join(output_dir, "han_results")
        os.makedirs(abcd_dir, exist_ok=True)

        best_abcd = self._best_abcd_so_far if self._best_abcd_so_far is not None else self.get_abcd_vector()
        np.save(os.path.join(abcd_dir, "abcd_params.npy"), best_abcd)

        layers = self.han_model.get_all_han_layers()
        for i, layer in enumerate(layers):
            np.save(os.path.join(abcd_dir, f"layer{i}_abcd.npy"),
                    layer.get_abcd_vector().detach().cpu().numpy())

        torch.save(self.policy.state_dict(), os.path.join(abcd_dir, "policy_state.pt"))

        metadata = {
            "fitness_mode": self.fitness_mode,
            "train_group": self.train_group,
            "tag_evader_group": self.tag_evader_group,
            "tag_num_adversaries": self.tag_num_adversaries,
            "tag_num_obstacles": self.tag_num_obstacles,
            "tag_capture_distance": self.tag_capture_distance,
            "total_abcd_params": self.han_model.total_abcd_params,
            "n_layers": len(layers),
            "layer_shapes": [
                {"in": l.in_features, "out": l.out_features, "n_abcd": l.num_abcd_params}
                for l in layers
            ],
            "window_size": layers[0].window_size if layers else None,
            "f_nn": self.han_model.f_nn,
            "f_hebb": self.han_model.f_hebb,
            "update_interval": self.han_model._update_interval,
            "best_fitness": float(-self._best_fitness_so_far) if self._best_fitness_so_far != float('inf') else None,
            "generations_completed": self._current_gen,
            "max_generations": self.max_gens,
        }
        with open(os.path.join(abcd_dir, "results.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  HAN results saved to: {abcd_dir}/")
        return abcd_dir

    def evaluate(
        self,
        output_dir: str,
        n_episodes: int = 10,
        fps: int = 20,
        max_video_frames: int = None,
    ):
        """Run evaluation episodes and save videos."""
        import torchvision

        group = self.train_group
        env = self.experiment.test_env
        max_steps = self.experiment.max_steps

        video_dir = os.path.join(output_dir, "videos_han")
        os.makedirs(video_dir, exist_ok=True)

        print(f"  Video output dir: {video_dir}")
        print(f"  max_steps={max_steps}, n_episodes={n_episodes}, fps={fps}, "
              f"max_video_frames={max_video_frames}")

        all_rewards = []
        all_fitnesses = []

        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            for ep in range(n_episodes):
                self.han_model.reset_all_weights()
                frames = []
                frame_errors = []

                ep_max_frames = max_video_frames
                if ep_max_frames is not None:
                    def on_frame(_td, _step, _cap=ep_max_frames):
                        if len(frames) < _cap:
                            try:
                                frame = self._render_rgb_array(env)
                                if frame is not None:
                                    frames.append(
                                        torch.tensor(frame.copy()).permute(2, 0, 1).unsqueeze(0)
                                    )
                            except Exception as e:
                                frame_errors.append(f"ep{ep} step{_step}: {e}")
                else:
                    def on_frame(_td, _step):
                        try:
                            frame = self._render_rgb_array(env)
                            if frame is not None:
                                frames.append(
                                    torch.tensor(frame.copy()).permute(2, 0, 1).unsqueeze(0)
                                )
                        except Exception as e:
                            frame_errors.append(f"ep{ep} step{_step}: {e}")

                try:
                    frame = self._render_rgb_array(env)
                    if frame is not None:
                        frames.append(torch.tensor(frame.copy()).permute(2, 0, 1).unsqueeze(0))
                except Exception as e:
                    frame_errors.append(f"ep{ep} reset: {e}")

                stats = self._run_one_episode(
                    env,
                    group,
                    max_steps,
                    self.rollout_policy,
                    on_frame=on_frame,
                )

                all_rewards.append(stats["episode_reward"])
                ep_fitness = self._compute_fitness(
                    stats["episode_reward"], stats["collision_count"], stats["step"],
                    initial_goal_dists=stats["initial_goal_dists"],
                    final_goal_dists=stats["final_goal_dists"],
                    success=stats["success"],
                    agent_collision_ratios=stats["agent_collision_ratios"],
                    pos_history=stats["pos_history"],
                    rot_history=stats["rot_history"],
                    target_pos_history=stats["target_pos_history"],
                    caught_step_records=stats["caught_step_records"],
                    first_env_caught_at=stats["first_env_caught_at"],
                    obstacle_dist_history=stats["obstacle_dist_history"],
                )
                all_fitnesses.append(ep_fitness)

                if frames:
                    vid = torch.cat(frames, dim=0).unsqueeze(0)
                    for idx in (-1, -2):
                        if vid.shape[idx] % 2 != 0:
                            vid = vid.index_select(idx, torch.arange(1, vid.shape[idx]))
                    video_path = os.path.join(video_dir, f"eval_han_{ep}.mp4")
                    vid_rgb = vid[0].permute(0, 2, 3, 1)
                    torchvision.io.write_video(video_path, vid_rgb.numpy(), fps=fps)
                    print(f"  Saved ep{ep}: {len(frames)} frames, reward={stats['episode_reward']:.2f}, fitness={ep_fitness:.2f}")
                else:
                    print(f"  SKIP ep{ep}: no frames, reward={stats['episode_reward']:.2f}")

        print(f"\nEvaluation ({n_episodes} episodes, mode={self.fitness_mode}):")
        print(f"  Mean reward: {np.mean(all_rewards):.2f}")
        print(f"  Mean fitness: {np.mean(all_fitnesses):.2f}")
        print(f"  Videos saved to: {video_dir}")

        return np.mean(all_fitnesses)
