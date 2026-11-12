"""Shared VMAS flocking scenario patches.

Ensures training (run_cmaes_han_flocking_custom.py) and evaluation
(run_cmaes_han_flocking_disturbance.py) see identical observations.

Patches applied to ``vmas.scenarios.flocking.Scenario``:
  1. ``action_script_creator`` → target motion is configurable via
     ``configure(target_motion=...)``:
       - ``"stationary"`` (default): target frozen at (x, y) — the
         original behavior of this patch.
       - ``"circle"``: target orbits (x, y) at ``target_motion_radius``
         with ``target_motion_period`` steps per revolution.
       - ``"figure8"``: Gerono lemniscate through (x, y), extents
         (±R, ±R/2), same period.
       - ``"random_walk"``: constant speed ``target_motion_speed``,
         heading resampled every ``target_motion_interval`` steps and
         on wall hits (clamped to ±0.95).
     Motion is *kinematic*: the script sets ``u = 0`` (VMAS asserts an
     action script sets ``u``) and drives the position analytically via
     ``set_pos`` — the same mechanism as ``vmas flocking_signal``. A
     force-driven orbit (tangential ``u`` + drag) produces a lagging
     rosette rather than a circle, so analytic positions are the right
     tool. Phase/heading state lives on the scenario and is reset per
     episode in ``reset_world_at`` for reproducible eval trajectories.
  2. ``reset_world_at`` → target initialized at ``(target_pos_x, target_pos_y)``
     instead of the VMAS default ``(0, -y_dim)``.
  3. ``observation`` → 10-dim:
       [pos(2), vel(2), target_rel(2),
        nn_rel_pos(2), nn_rel_vel(2)]
     where ``nn_*`` is the nearest neighbor within ``neighbor_radius``.
  4. ``reward`` → preserves original collision + dist_shaping rewards
     AND adds two orbit-shaping terms so PPO-based algorithms (e.g.
     IPPO) can learn to orbit the target — VMAS flocking's default
     reward only encodes "stay together, don't collide", which would
     train a PPO policy to clump rather than orbit.
       + w_at * At    (tangential alignment vs target, speed-modulated)
       + w_dt * Dt    (Gaussian distance band around ``orbit_radius``)

     The two orbit-shaping weights default to 0 (vanilla VMAS
     reward). Enable them via ``configure(orbit_reward_weight_at=...,
     orbit_reward_weight_dt=...)``. The HAN-vs-IPPO comparison uses
     these orbit-shaping terms for IPPO and a per-step decomposition
     of the orbit fitness (At, Dt, Cg, S) for HAN's CMA-ES fitness.

Only ``name == "flocking.py" or name == "flocking"`` is patched; all
other VMAS scenarios are untouched.
"""
import importlib
import math

import torch
import vmas  # noqa: F401 — triggers vmas.__init__, vmas.scenarios = list
from vmas.simulator.core import Agent, World, Landmark
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import ScenarioUtils, X, Y


# ---------------------------------------------------------------------------
# Module-level config (set by ``configure()`` after argparse).
# ---------------------------------------------------------------------------
_TARGET_POS_X = 0.0
_TARGET_POS_Y = 0.0
_NEIGHBOR_RADIUS = 0.5
_ORBIT_RADIUS = 0.7
_ORBIT_RADIUS_TOLERANCE = 0.3
_DT_FLOOR = 0.1
_ORBIT_REWARD_WEIGHT_AT = 0.0  # off by default → identical to vanilla
_ORBIT_REWARD_WEIGHT_DT = 0.0  # off by default → identical to vanilla
_SPEED_THRESHOLD = 0.02  # At is gated by speed > threshold
# Moving-target motion (kinematic, see module docstring §1).
_TARGET_MOTION = "stationary"
_TARGET_MOTION_RADIUS = 0.35
_TARGET_MOTION_PERIOD = 400
_TARGET_MOTION_SPEED = 0.01
_TARGET_MOTION_INTERVAL = 50
# "evader" mode: fast target (speed comparable to the agents' own ~0.37
# units/step) that bounces off the walls of the ±_TARGET_MOTION_BOUND box.
_TARGET_MOTION_BOUND = 0.9
# Speed-matching factor between the figure-8 and the circle.
#
# For the Gerono lemniscate  x = R·sin φ,  y = (R/2)·sin 2φ  with
# φ = 2πt/T, the instantaneous speed is
#     (2πR/T)·sqrt(cos²φ + cos²2φ)
# whose mean over one period is 0.9700·(2πR/T) — i.e. the figure-8
# covers slightly LESS distance per revolution than a circle of the same
# radius. So to give both trajectories the same mean speed at the same
# radius, the figure-8 needs a slightly SHORTER period:
#     T_figure8 = 0.9700 · T_circle
# Measured: R=0.35 → circle T=400 gives 0.005498/step,
#                     figure-8 T=388 gives 0.005499/step (0.02% error).
_FIGURE8_SPEED_FACTOR = 0.97
# Per-batch-env motion state, lazily created by the action script and
# reset per episode by ``_patched_reset_world_at`` (env_index=None only —
# partial resets keep their state, matching how the trainable agents'
# ``self.t`` behaves).
_TARGET_MOTION_STATE = {}


def configure(
    target_pos_x=0.0,
    target_pos_y=0.0,
    neighbor_radius=0.5,
    orbit_radius=0.7,
    orbit_radius_tolerance=0.3,
    dt_floor=0.1,
    orbit_reward_weight_at=0.0,
    orbit_reward_weight_dt=0.0,
    target_motion="stationary",
    target_motion_radius=0.35,
    target_motion_period=400,
    target_motion_speed=0.01,
    target_motion_interval=50,
    target_motion_bound=0.9,
    target_motion_match_speed=False,
):
    """Set the parameters read by the patched methods.

    The two reward weights default to 0.0, which keeps the env
    backward-compatible with the original VMAS flocking reward
    (collision + distance shaping only). Set them > 0 to add
    tangential-alignment (At) and distance-band (Dt) shaping on top.

    ``target_motion`` selects the target trajectory:
      ``"stationary"`` | ``"circle"`` | ``"figure8"`` | ``"random_walk"``
      | ``"evader"``.
    ``circle``/``figure8`` use (``target_motion_radius``,
    ``target_motion_period``); ``random_walk`` uses
    (``target_motion_speed``, ``target_motion_interval``);
    ``evader`` uses (``target_motion_speed``, ``target_motion_interval``,
    ``target_motion_bound``).

    ``target_motion_match_speed=True`` rescales the period of the
    ``figure8`` mode so that its mean speed matches the ``circle`` mode
    at the same radius (see ``_FIGURE8_SPEED_FACTOR``). This makes the
    two trajectories directly comparable in terms of how fast the target
    moves.
    """
    global _TARGET_POS_X, _TARGET_POS_Y, _NEIGHBOR_RADIUS
    global _ORBIT_RADIUS, _ORBIT_RADIUS_TOLERANCE, _DT_FLOOR
    global _ORBIT_REWARD_WEIGHT_AT, _ORBIT_REWARD_WEIGHT_DT
    global _TARGET_MOTION, _TARGET_MOTION_RADIUS, _TARGET_MOTION_PERIOD
    global _TARGET_MOTION_SPEED, _TARGET_MOTION_INTERVAL
    global _TARGET_MOTION_BOUND
    _TARGET_POS_X = float(target_pos_x)
    _TARGET_POS_Y = float(target_pos_y)
    _NEIGHBOR_RADIUS = float(neighbor_radius)
    _ORBIT_RADIUS = float(orbit_radius)
    _ORBIT_RADIUS_TOLERANCE = float(orbit_radius_tolerance)
    _DT_FLOOR = float(dt_floor)
    _ORBIT_REWARD_WEIGHT_AT = float(orbit_reward_weight_at)
    _ORBIT_REWARD_WEIGHT_DT = float(orbit_reward_weight_dt)
    valid_motions = (
        "stationary", "circle", "figure8", "random_walk", "evader",
    )
    if target_motion not in valid_motions:
        raise ValueError(
            f"target_motion must be one of {valid_motions}, "
            f"got {target_motion!r}"
        )
    _TARGET_MOTION = target_motion
    _TARGET_MOTION_RADIUS = float(target_motion_radius)
    period = max(int(target_motion_period), 1)
    if target_motion == "figure8" and target_motion_match_speed:
        # Match the circle's mean speed at the same radius.
        period = max(int(round(period * _FIGURE8_SPEED_FACTOR)), 1)
    _TARGET_MOTION_PERIOD = period
    _TARGET_MOTION_SPEED = float(target_motion_speed)
    _TARGET_MOTION_INTERVAL = max(int(target_motion_interval), 1)
    _TARGET_MOTION_BOUND = float(target_motion_bound)


# ---------------------------------------------------------------------------
# 1) Target motion action scripts.
# ---------------------------------------------------------------------------
def _stationary_action_script(agent, world, scenario_self):
    """Stationary target: u = 0 for all envs."""
    device = agent.device
    batch_dim = agent.batch_dim
    agent.action.u = torch.zeros((batch_dim, 2), device=device)


def _zeros_u(agent):
    agent.action.u = torch.zeros(
        (agent.batch_dim, 2), device=agent.device
    )


def _motion_position(mode, step, B, device, dtype):
    """Analytic target position at phase ``step`` for every batch env.

    Returns a (B, 2) tensor of positions relative to
    (_TARGET_POS_X, _TARGET_POS_Y). All trajectories stay inside the
    ±0.5 default-radius orbits, well within the ±1.0 world bounds.
    """
    base = torch.zeros((B, 2), device=device, dtype=dtype)
    if mode == "circle":
        phi = 2.0 * math.pi * step.to(dtype) / _TARGET_MOTION_PERIOD
        base[:, 0] = _TARGET_MOTION_RADIUS * torch.cos(phi)
        base[:, 1] = _TARGET_MOTION_RADIUS * torch.sin(phi)
    elif mode == "figure8":
        # Gerono lemniscate: x = R sin(phi), y = (R/2) sin(2 phi).
        phi = 2.0 * math.pi * step.to(dtype) / _TARGET_MOTION_PERIOD
        base[:, 0] = _TARGET_MOTION_RADIUS * torch.sin(phi)
        base[:, 1] = (
            0.5 * _TARGET_MOTION_RADIUS * torch.sin(2.0 * phi)
        )
    # "random_walk" integrates heading on the scenario; handled by
    # _random_walk_position, not here.
    return base


def _random_walk_position(B, device, dtype):
    """Advance (or initialize) the random-walk state and return (B, 2)
    absolute positions.

    State lives in the module-level ``_TARGET_MOTION_STATE`` dict keyed
    by batch-env index, created lazily here and cleared per episode in
    ``_patched_reset_world_at``. Heading resamples every
    ``_TARGET_MOTION_INTERVAL`` steps or when the next position would
    leave the ±0.95 box (choose a direction pointing back inside).
    """
    pos0 = torch.tensor(
        [_TARGET_POS_X, _TARGET_POS_Y], device=device, dtype=dtype
    )
    positions = pos0.repeat(B, 1)
    for b in range(B):
        st = _TARGET_MOTION_STATE.get(b)
        if st is None:
            st = {
                "pos": pos0.clone(),
                "theta": 2.0 * math.pi * torch.rand(1).item(),
                "steps": 0,
            }
            _TARGET_MOTION_STATE[b] = st
        # Resample heading periodically.
        st["steps"] += 1
        if st["steps"] >= _TARGET_MOTION_INTERVAL:
            st["theta"] = 2.0 * math.pi * torch.rand(1).item()
            st["steps"] = 0
        nxt = st["pos"] + _TARGET_MOTION_SPEED * torch.tensor(
            [math.cos(st["theta"]), math.sin(st["theta"])],
            device=device, dtype=dtype,
        )
        # Wall hit: head back toward the configured center.
        if (nxt[0].abs() > 0.95) or (nxt[1].abs() > 0.95):
            st["theta"] = math.atan2(
                (pos0 - st["pos"])[1].item(),
                (pos0 - st["pos"])[0].item(),
            )
            st["steps"] = 0
            nxt = st["pos"] + _TARGET_MOTION_SPEED * torch.tensor(
                [math.cos(st["theta"]), math.sin(st["theta"])],
                device=device, dtype=dtype,
            )
        st["pos"] = nxt
        positions[b] = nxt
    return positions


def _evader_position(B, device, dtype):
    """Fast random-walking target that bounces off the world walls.

    This is the "pursuit / encirclement" motion: the target moves at
    ``_TARGET_MOTION_SPEED`` (comparable to the agents' own speed, ~0.37
    units/step) with its heading resampled every
    ``_TARGET_MOTION_INTERVAL`` steps, and reflects off the walls of the
    ±``_TARGET_MOTION_BOUND`` box.

    Unlike ``circle``/``figure8`` (whose speed is only ~1.5% of the
    agents' speed, so the target looks stationary to a policy), the
    evader is genuinely hard to track — the swarm has to re-aim
    continuously.

    State lives in the module-level ``_TARGET_MOTION_STATE`` dict keyed
    by batch-env index, created lazily here and cleared per episode in
    ``_patched_reset_world_at``.
    """
    bound = _TARGET_MOTION_BOUND
    speed = _TARGET_MOTION_SPEED
    positions = torch.zeros((B, 2), device=device, dtype=dtype)
    for b in range(B):
        st = _TARGET_MOTION_STATE.get(b)
        if st is None:
            st = {
                "pos": torch.tensor(
                    [_TARGET_POS_X, _TARGET_POS_Y], device=device, dtype=dtype
                ),
                "theta": 2.0 * math.pi * torch.rand(1).item(),
                "steps": 0,
            }
            _TARGET_MOTION_STATE[b] = st

        # Resample heading periodically -> unpredictable direction.
        st["steps"] += 1
        if st["steps"] >= _TARGET_MOTION_INTERVAL:
            st["theta"] = 2.0 * math.pi * torch.rand(1).item()
            st["steps"] = 0

        pos = st["pos"]
        theta = st["theta"]
        # Reflect off the walls (at most one reflection per axis).
        for _ in range(2):
            nxt = pos + speed * torch.tensor(
                [math.cos(theta), math.sin(theta)], device=device, dtype=dtype
            )
            hit_x = bool(nxt[0].abs() > bound)
            hit_y = bool(nxt[1].abs() > bound)
            if not (hit_x or hit_y):
                break
            if hit_x:
                theta = math.pi - theta
            if hit_y:
                theta = -theta
            st["theta"] = theta
        nxt = pos + speed * torch.tensor(
            [math.cos(theta), math.sin(theta)], device=device, dtype=dtype
        )
        nxt[0] = torch.clamp(nxt[0], -bound, bound)
        nxt[1] = torch.clamp(nxt[1], -bound, bound)
        st["pos"] = nxt
        positions[b] = nxt
    return positions


def _moving_action_script(agent, world, scenario_self):
    """Kinematic moving target: u = 0, position set analytically.

    VMAS asserts that an action script sets ``u`` (core.py
    ``action_callback``), so we keep the force at zero and drive
    ``state.pos`` directly — the same mechanism as the ``flocking_signal``
    scenario's ``pre_step``. ``action_callback`` runs inside
    ``env_process_action`` *before* ``world.step()``, and with u = 0 and
    zero drag-displaced velocity the physics step leaves the position
    where we set it.
    """
    _zeros_u(agent)
    mode = _TARGET_MOTION
    if mode == "stationary":
        return
    B = agent.batch_dim
    state = scenario_self._target.state
    if mode == "random_walk":
        new_pos = _random_walk_position(
            B, agent.device, state.pos.dtype
        )
        agent.set_pos(new_pos, batch_index=None)
        return
    if mode == "evader":
        new_pos = _evader_position(B, agent.device, state.pos.dtype)
        agent.set_pos(new_pos, batch_index=None)
        return
    # circle / figure8: closed-form trajectory from the per-env phase.
    step = getattr(scenario_self, "_target_motion_step", None)
    if step is None:
        step = torch.zeros(B, device=agent.device, dtype=torch.long)
        scenario_self._target_motion_step = step
    step += 1
    rel = _motion_position(mode, step, B, agent.device, state.pos.dtype)
    center = torch.stack(
        [
            torch.full((B,), _TARGET_POS_X, device=agent.device),
            torch.full((B,), _TARGET_POS_Y, device=agent.device),
        ],
        dim=-1,
    ).to(state.pos.dtype)
    agent.set_pos(center + rel, batch_index=None)


def _patched_action_script_creator(self):
    def action_script(agent, world):
        if _TARGET_MOTION == "stationary":
            _stationary_action_script(agent, world, self)
        else:
            _moving_action_script(agent, world, self)
    return action_script


# ---------------------------------------------------------------------------
# 1b) Patched make_world: same as VMAS flocking, but target is
# collide=False (pure visual marker, no physical body for agents to
# bounce off).
# ---------------------------------------------------------------------------
def _patched_make_world(self, batch_dim, device, **kwargs):
    """Identical to VMAS flocking's make_world, except ``_target``
    is created with ``collide=False``. Keeps the rest of the spawn /
    obstacle logic unchanged.
    """
    from vmas.simulator.utils import Color  # local import to avoid cycles
    from vmas.simulator.core import Sphere   # local import

    n_agents = kwargs.pop("n_agents", 4)
    n_obstacles = kwargs.pop("n_obstacles", 5)
    self._min_dist_between_entities = kwargs.pop("min_dist_between_entities", 0.15)

    self.n_lidar_rays = kwargs.pop("n_lidar_rays", 12)

    self.collision_reward = kwargs.pop("collision_reward", -0.1)
    self.dist_shaping_factor = kwargs.pop("dist_shaping_factor", 1)
    ScenarioUtils.check_kwargs_consumed(kwargs)

    self.plot_grid = True
    self.desired_distance = 0.1
    self.min_collision_distance = 0.005
    self.x_dim = 1
    self.y_dim = 1

    # Make world
    world = World(batch_dim, device, collision_force=400, substeps=5)

    # === Custom: target has no physical body (collide=False). ===
    self._target = Agent(
        name="target",
        collide=False,                     # ← changed from True
        color=Color.GREEN,
        render_action=True,
        action_script=self.action_script_creator(),
    )
    world.add_agent(self._target)

    goal_entity_filter = lambda e: not isinstance(e, Agent)
    for i in range(n_agents):
        agent = Agent(
            name=f"agent_{i}",
            collide=True,
            sensors=None,  # ← removed Lidar sensor to avoid per-step overhead
            render_action=True,
        )
        agent.collision_rew = torch.zeros(batch_dim, device=device)
        agent.dist_rew = agent.collision_rew.clone()

        world.add_agent(agent)

    # Add landmarks (obstacles)
    self.obstacles = []
    for i in range(n_obstacles):
        obstacle = Landmark(
            name=f"obstacle_{i}",
            collide=True,
            movable=False,
            shape=Sphere(radius=0.1),
            color=Color.RED,
        )
        world.add_landmark(obstacle)
        self.obstacles.append(obstacle)

    return world


# ---------------------------------------------------------------------------
# 2) reset_world_at with configurable target initial position.
# ---------------------------------------------------------------------------
def _patched_reset_world_at(self, env_index=None):
    """Original VMAS flocking reset_world_at, but target initial position
    is set to (TARGET_POS_X, TARGET_POS_Y) instead of (0, -y_dim)."""
    target_pos = torch.zeros(
        (
            (1, self.world.dim_p)
            if env_index is not None
            else (self.world.batch_dim, self.world.dim_p)
        ),
        device=self.world.device,
        dtype=torch.float32,
    )
    target_pos[:, X] = _TARGET_POS_X
    target_pos[:, Y] = _TARGET_POS_Y
    self._target.set_pos(target_pos, batch_index=env_index)
    if env_index is None:
        # Reset moving-target motion state so each episode starts from
        # the same trajectory phase (reproducible eval rollouts).
        self._target_motion_step = None
        _TARGET_MOTION_STATE.clear()
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
                torch.stack(
                    [
                        torch.linalg.vector_norm(
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
                torch.stack(
                    [
                        torch.linalg.vector_norm(
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
        self.t = torch.zeros(self.world.batch_dim, device=self.world.device)
    else:
        self.t[env_index] = 0


# ---------------------------------------------------------------------------
# 3) observation: configurable between "nn" (10-dim) and "lidar" (18-dim).
# ---------------------------------------------------------------------------
def _nearest_neighbor_pos_vel(self, agent):
    """Return (rel_pos (B,2), rel_vel (B,2)) of the nearest OTHER
    policy agent within ``_NEIGHBOR_RADIUS`` of ``agent``.
    If no agent is in range, return zeros for both.
    Optimized: avoid creating intermediate tensors per agent.
    """
    others = [a for a in self.world.policy_agents if a is not agent]
    if not others:
        B = agent.state.pos.shape[0]
        zero = torch.zeros((B, 2), device=agent.state.pos.device,
                           dtype=agent.state.pos.dtype)
        return zero, zero

    n_others = len(others)
    # Stack all positions at once: (n_others, B, 2) -> transpose to (B, n_others, 2)
    # Avoid per-agent list comprehension overhead by pre-allocating.
    pos_list = []
    vel_list = []
    for a in others:
        pos_list.append(a.state.pos)
        vel_list.append(a.state.vel)

    others_pos = torch.stack(pos_list, dim=1)  # (B, n_others, 2)
    others_vel = torch.stack(vel_list, dim=1)  # (B, n_others, 2)

    agent_pos = agent.state.pos.unsqueeze(1)   # (B, 1, 2)
    agent_vel = agent.state.vel.unsqueeze(1)   # (B, 1, 2)

    diffs_pos = others_pos - agent_pos         # (B, n_others, 2)
    dists = torch.linalg.vector_norm(diffs_pos, dim=-1)  # (B, n_others)

    # Mask out-of-range agents
    masked_dists = torch.where(
        dists < _NEIGHBOR_RADIUS,
        dists,
        torch.full_like(dists, float("inf"))
    )
    any_in_range = (dists < _NEIGHBOR_RADIUS).any(dim=-1)  # (B,)
    best_idx = torch.argmin(masked_dists, dim=-1)          # (B,)
    B = agent.state.pos.shape[0]

    # Gather best relative position and velocity
    gather_idx = best_idx.view(B, 1, 1).expand(-1, 1, 2)
    best_rel_pos = torch.gather(diffs_pos, 1, gather_idx).squeeze(1)

    diffs_vel = others_vel - agent_vel
    best_rel_vel = torch.gather(diffs_vel, 1, gather_idx).squeeze(1)

    zero = torch.zeros_like(best_rel_pos)
    out_rel_pos = torch.where(any_in_range.unsqueeze(-1), best_rel_pos, zero)
    out_rel_vel = torch.where(any_in_range.unsqueeze(-1), best_rel_vel, zero)
    return out_rel_pos, out_rel_vel




def _patched_observation(self, agent):
    """10-dim observation: nn (nearest-neighbor pos+vel)."""
    pos = agent.state.pos
    vel = agent.state.vel
    target_rel = pos - self._target.state.pos
    nn_rel_pos, nn_rel_vel = _nearest_neighbor_pos_vel(self, agent)
    return torch.cat([pos, vel, target_rel, nn_rel_pos, nn_rel_vel], dim=-1)


# ---------------------------------------------------------------------------
# 4) reward: original VMAS flocking reward + optional At / Dt orbit
# shaping. The orbit terms are 0 by default (configure with
# ``orbit_reward_weight_at`` / ``orbit_reward_weight_dt`` > 0 to enable).
# ---------------------------------------------------------------------------
def _patched_reward(self, agent):
    """Reward = original collision + dist_shaping + w_at * At + w_dt * Dt.

    At, Dt follow the same definitions as the HAN fitness
    ``_compute_flocking_orbit_fitness`` in
    ``benchmarl/algorithms/cmaes_han_optimizer.py``:

        At = (dot(vel_dir, tangent) + 1) / 2   (CCW tangent of agent→target)
             multiplied by ``speed / (speed + speed_threshold)`` so a
             stationary agent contributes At = 0.
        Dt = exp( - (r - orbit_radius)² / (2 * orbit_radius_tolerance²) )
             floored at ``dt_floor``.

    Both At and Dt are gated: if the agent sits exactly on the target
    (r → 0) the tangent is undefined, so the contribution is masked to 0.
    """
    is_first = self.world.policy_agents.index(agent) == 0

    if is_first:
        self.t += 1
        # Avoid collisions with each other (vanilla VMAS logic, copied
        # verbatim — we don't call the original ``reward`` because the
        # state mutation is interleaved with our own bookkeeping).
        if self.collision_reward != 0:
            for a in self.world.policy_agents:
                a.collision_rew[:] = 0
            for i, a in enumerate(self.world.agents):
                for j, b in enumerate(self.world.agents):
                    if j <= i:
                        continue
                    collision = (
                        self.world.get_distance(a, b)
                        <= self.min_collision_distance
                    )
                    if a.action_script is None:
                        a.collision_rew[collision] += self.collision_reward
                    if b.action_script is None:
                        b.collision_rew[collision] += self.collision_reward

    # Stay-close-together (separation) shaping — vanilla VMAS logic.
    agents_dist_shaping = (
        torch.stack(
            [
                torch.linalg.vector_norm(agent.state.pos - a.state.pos, dim=-1)
                for a in self.world.agents
                if a != agent
            ],
            dim=1,
        )
        - self.desired_distance
    ).pow(2).mean(-1) * self.dist_shaping_factor
    agent.dist_rew = agent.distance_shaping - agents_dist_shaping
    agent.distance_shaping = agents_dist_shaping

    base_reward = agent.collision_rew + agent.dist_rew

    # Orbit shaping: tangential alignment + distance band.
    if (_ORBIT_REWARD_WEIGHT_AT <= 0.0
            and _ORBIT_REWARD_WEIGHT_DT <= 0.0):
        return base_reward

    pos = agent.state.pos                                  # (B, 2)
    vel = agent.state.vel                                  # (B, 2)
    tgt = self._target.state.pos                           # (B, 2)
    r_vec = pos - tgt                                      # (B, 2)
    r_norm = torch.linalg.vector_norm(r_vec, dim=-1)       # (B,)
    eps = 1e-6
    valid = (r_norm > eps).float()                         # (B,)
    r_unit = r_vec / (r_norm.unsqueeze(-1) + eps)          # (B, 2)

    # At: speed-modulated CCW-tangent alignment.
    # rot90_CCW([x, y]) = [-y, x]
    tangent = torch.stack([-r_unit[:, 1], r_unit[:, 0]], dim=-1)  # (B, 2)
    speed = torch.linalg.vector_norm(vel, dim=-1)          # (B,)
    speed_factor = (speed / (speed + _SPEED_THRESHOLD)).clamp(0.0, 1.0)
    # vel_dir normalized; guarded against zero speed.
    vel_dir = vel / (speed.unsqueeze(-1) + eps)            # (B, 2)
    dot = (vel_dir * tangent).sum(dim=-1)                  # (B,)
    at_raw = ((dot + 1.0) * 0.5).clamp(0.0, 1.0)           # (B,) ∈ [0, 1]
    at = at_raw * speed_factor * valid                     # (B,)

    # Dt: Gaussian band around orbit radius, floored.
    dt_raw = torch.exp(
        -((r_norm - _ORBIT_RADIUS) ** 2)
        / (2.0 * _ORBIT_RADIUS_TOLERANCE ** 2)
    )                                                      # (B,)
    dt = torch.clamp(dt_raw, min=_DT_FLOOR) * valid        # (B,)

    orbit_reward = (
        _ORBIT_REWARD_WEIGHT_AT * at
        + _ORBIT_REWARD_WEIGHT_DT * dt
    )
    return base_reward + orbit_reward


# ---------------------------------------------------------------------------
# Install the load() wrapper.
# ---------------------------------------------------------------------------
_vmas_scenarios_pkg = importlib.import_module("vmas.scenarios")
_original_load = _vmas_scenarios_pkg.load


def _patched_load(name):
    module = _original_load(name)
    if name == "flocking.py" or name == "flocking":
        module.Scenario.action_script_creator = _patched_action_script_creator
        module.Scenario.make_world = _patched_make_world
        module.Scenario.reset_world_at = _patched_reset_world_at
        module.Scenario.observation = _patched_observation
        module.Scenario.reward = _patched_reward
    return module


_vmas_scenarios_pkg.load = _patched_load
