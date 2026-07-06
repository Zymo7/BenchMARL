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
        "flocking_global",
        "flocking_orbit",
        "flocking_lf_arrival",
        "flocking_light_intensity",
        "flocking_signal_intensity",
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

        self.policy = experiment.policy

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

    def _compute_fitness(self, episode_reward: float, collision_count: int, total_steps: int,
                         initial_goal_dists: torch.Tensor = None,
                         final_goal_dists: torch.Tensor = None,
                         success: bool = False,
                         agent_collision_ratios: torch.Tensor = None,
                         pos_history=None,
                         rot_history=None,
                         target_pos_history=None,
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
        w_S = 1.0
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

    def _run_one_episode(self, env, group, max_steps, policy, on_frame=None):
        """Run a single episode and return everything needed by any fitness mode."""
        td = env.reset()

        # Ensure the vmas core reference is resolved. Normally cached
        # lazily inside fitness() before the rollout loop, but the
        # evaluate-only entry path can call _run_one_episode directly.
        self._get_vmas_core()

        obs = td.get((group, "observation"))
        initial_goal_dists = torch.linalg.vector_norm(
            obs[..., :2].float(), dim=-1
        )

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

        while not done and step < max_steps:
            td = policy(td)
            td = env.step(td)

            if on_frame is not None:
                on_frame(td, step)

            reward = td.get(("next", group, "reward"))
            episode_reward += reward.sum().item()

            info = td.get(("next", group, "info"))
            if info is not None:
                col_rew = info.get("agent_collision_rew")
                if col_rew is not None:
                    collision_count += (col_rew < 0).sum().item()

            # Inter-agent collision detection: use ABSOLUTE agent
            # positions read from the vmas core environment. We cannot
            # recover absolute positions from obs[:2] because each agent
            # has its own goal (obs[:2] = agent.pos - agent.goal), and
            # the goal offset does NOT cancel when subtracting two
            # different agents' obs[:2]. The pairwise difference
            # (obs[i,:2] - obs[j,:2]) = (pos[i] - pos[j]) + (goal[j] -
            # goal[i]), which is biased by per-agent goal offsets.
            core = self._vmas_core
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

                # Record target absolute position (env index 0) for
                # flocking_orbit. The target is the action_script-driven
                # agent attached to the VMAS flocking scenario.
                target = getattr(getattr(core, "scenario", None), "_target", None)
                if target is not None and hasattr(target.state, "pos"):
                    target_pos_history.append(
                        target.state.pos[0].detach().cpu()
                    )

            done = td.get(("next", "done")).any().item()
            if done:
                success = True
            td = td.get("next")
            step += 1

        final_obs = td.get((group, "observation"))
        final_goal_dists = torch.linalg.vector_norm(
            final_obs[..., :2].float(), dim=-1
        )

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
        }

    def fitness(self, x: np.ndarray) -> float:
        """Evaluate fitness of a candidate ABCD parameter vector."""
        t0 = time.time()
        self.set_abcd_from_vector(x)
        # reset_all_weights also clears the sliding windows and resets ticks.
        self.han_model.reset_all_weights()
        # Cache the vmas core env (used for absolute position readout
        # during inter-agent collision detection).
        self._get_vmas_core()

        fitnesses = []
        group = list(self.experiment.group_map.keys())[0]
        env = self.experiment.test_env
        max_steps = self.experiment.max_steps

        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            for ep in range(self.n_eval_episodes):
                self.han_model.reset_all_weights()
                stats = self._run_one_episode(env, group, max_steps, self.policy)
                ep_fitness = self._compute_fitness(
                    stats["episode_reward"], stats["collision_count"], stats["step"],
                    initial_goal_dists=stats["initial_goal_dists"],
                    final_goal_dists=stats["final_goal_dists"],
                    success=stats["success"],
                    agent_collision_ratios=stats["agent_collision_ratios"],
                    pos_history=stats["pos_history"],
                    rot_history=stats["rot_history"],
                    target_pos_history=stats["target_pos_history"],
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
        print(f"  HanModel: window_size={self.han_model.layers[0].window_size}, "
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

        group = list(self.experiment.group_map.keys())[0]
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
                                frame = env.render(mode="rgb_array")
                                if frame is not None:
                                    frames.append(
                                        torch.tensor(frame.copy()).permute(2, 0, 1).unsqueeze(0)
                                    )
                            except Exception as e:
                                frame_errors.append(f"ep{ep} step{_step}: {e}")
                else:
                    def on_frame(_td, _step):
                        try:
                            frame = env.render(mode="rgb_array")
                            if frame is not None:
                                frames.append(
                                    torch.tensor(frame.copy()).permute(2, 0, 1).unsqueeze(0)
                                )
                        except Exception as e:
                            frame_errors.append(f"ep{ep} step{_step}: {e}")

                try:
                    frame = env.render(mode="rgb_array")
                    if frame is not None:
                        frames.append(torch.tensor(frame.copy()).permute(2, 0, 1).unsqueeze(0))
                except Exception as e:
                    frame_errors.append(f"ep{ep} reset: {e}")

                stats = self._run_one_episode(env, group, max_steps, self.policy, on_frame=on_frame)

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
