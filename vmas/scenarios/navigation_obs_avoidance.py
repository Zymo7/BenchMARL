#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
"""Single-agent navigation with N static obstacles.

Used by the HAN obstacle-avoidance experiment in BenchMARL.

Observation (per agent, single holonomic agent):
    [ agent.pos (2),
      agent.vel (2),
      goal_rel (2),
      nearest_obstacle_rel (2),   # 0 if no obstacle within sense range
      has_obstacle_flag (1) ]      # 1 if any obstacle within sense range

Total observation dimension: 9.

Reward (matches VMAS ``navigation`` shaping + a small obstacle touch penalty
that is exposed via ``info["agent_collision_rew"]`` for downstream fitness
functions to inspect):
    pos_shaping(t-1) - pos_shaping(t)   # per-step distance shaping
    + final_reward when agent reaches goal
    + agent_collision_penalty * number_of_obstacle_touches
"""
import typing
from typing import Callable, Dict, List

import torch
from torch import Tensor

from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils, DRAG, LINEAR_FRICTION, ANGULAR_FRICTION

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        ################
        # Scenario configuration
        ################
        self.plot_grid = False

        self.n_static_obstacles = kwargs.pop("n_static_obstacles", 3)
        self.world_spawning_x = kwargs.pop("world_spawning_x", 1.0)
        self.world_spawning_y = kwargs.pop("world_spawning_y", 1.0)
        self.agent_radius = kwargs.pop("agent_radius", 0.10)
        self.obstacle_radius = kwargs.pop("obstacle_radius", 0.15)
        # Sense range (in world units) for the nearest-obstacle feature.
        # Obstacles beyond this distance are not represented in the observation.
        self.obstacle_sense_range = kwargs.pop("obstacle_sense_range", 0.6)

        self.shared_rew = kwargs.pop("shared_rew", False)
        self.final_reward = kwargs.pop("final_reward", 0.01)
        self.agent_collision_penalty = kwargs.pop(
            "agent_collision_penalty", -1.0
        )

        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.min_collision_distance = 0.005

        ScenarioUtils.check_kwargs_consumed(kwargs)

        ################
        # Make world
        ################
        world = World(
            batch_dim,
            device,
            substeps=5,
            collision_force=500,
            dt=0.1,
            gravity=(0.0, 0.0),
            drag=DRAG,
            linear_friction=LINEAR_FRICTION,
            angular_friction=ANGULAR_FRICTION,
        )

        ################
        # Add single holonomic agent
        ################
        agent = Agent(
            name="agent_0",
            collide=True,
            color=Color.BLUE,
            render_action=True,
            shape=Sphere(radius=self.agent_radius),
            u_range=[1, 1],
            u_multiplier=[1, 1],
            dynamics=Holonomic(),
        )
        agent.pos_rew = torch.zeros(batch_dim, device=device)
        agent.agent_collision_rew = torch.zeros(batch_dim, device=device)
        world.add_agent(agent)

        ################
        # Add goal landmark (non-collidable)
        ################
        goal = Landmark(
            name="goal_0",
            collide=False,
            movable=False,
            color=Color.GREEN,
            shape=Sphere(radius=self.agent_radius),
        )
        world.add_landmark(goal)
        agent.goal = goal
        self.goal = goal

        ################
        # Add static obstacles (collidable)
        ################
        self.obstacles: List[Landmark] = []
        for i in range(self.n_static_obstacles):
            obstacle = Landmark(
                name=f"obstacle_{i}",
                collide=True,
                movable=False,
                color=Color.RED,
                shape=Sphere(radius=self.obstacle_radius),
            )
            world.add_landmark(obstacle)
            self.obstacles.append(obstacle)

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = torch.zeros(batch_dim, device=device)
        self.on_goal = torch.zeros(batch_dim, device=device, dtype=torch.bool)

        return world

    def reset_world_at(self, env_index: int = None):
        # Spawn all entities (agents + obstacles + goal) ensuring pairwise
        # separation by ``min_distance_between_entities``.
        ScenarioUtils.spawn_entities_randomly(
            [self.world.agents[0]] + self.obstacles + [self.goal],
            self.world,
            env_index,
            self.min_distance_between_entities,
            x_bounds=(-self.world_spawning_x, self.world_spawning_x),
            y_bounds=(-self.world_spawning_y, self.world_spawning_y),
        )

        agent = self.world.agents[0]
        if env_index is None:
            agent.goal_dist = torch.linalg.vector_norm(
                agent.state.pos - agent.goal.state.pos, dim=-1
            )
        else:
            agent.goal_dist[env_index] = torch.linalg.vector_norm(
                agent.state.pos[env_index] - agent.goal.state.pos[env_index]
            )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        if is_first:
            self.pos_rew[:] = 0
            self.final_rew[:] = 0

            distance_to_goal = torch.linalg.vector_norm(
                agent.state.pos - agent.goal.state.pos, dim=-1
            )
            self.on_goal = distance_to_goal < agent.goal.shape.radius

            # VMAS-style positional shaping: reward delta in distance to goal.
            agent.pos_rew = agent.goal_dist - distance_to_goal
            agent.goal_dist = distance_to_goal
            self.pos_rew += agent.pos_rew

            self.final_rew[self.on_goal] = self.final_reward

            # Obstacle-touch penalty: 1 per step while the agent overlaps
            # any obstacle by more than the min collision distance.
            agent.agent_collision_rew[:] = 0
            for obstacle in self.obstacles:
                if self.world.collides(agent, obstacle):
                    distance = self.world.get_distance(agent, obstacle)
                    agent.agent_collision_rew[
                        distance <= self.min_collision_distance
                    ] += self.agent_collision_penalty

        return agent.pos_rew + self.final_rew + agent.agent_collision_rew

    def observation(self, agent: Agent):
        pos = agent.state.pos                            # (B, 2)
        vel = agent.state.vel                            # (B, 2)
        goal_rel = agent.goal.state.pos - pos            # (B, 2)

        if self.obstacles:
            obs_pos = torch.stack(
                [o.state.pos for o in self.obstacles], dim=1
            )                                            # (B, N, 2)
            diff = obs_pos - pos.unsqueeze(1)            # (B, N, 2)
            dist = torch.linalg.vector_norm(diff, dim=-1)  # (B, N)
            nearest_idx = dist.argmin(dim=-1)            # (B,)
            nearest_dist = torch.gather(
                dist, 1, nearest_idx.unsqueeze(-1)
            )                                            # (B, 1)
            nearest_dir = torch.gather(
                diff,
                1,
                nearest_idx[:, None, None].expand(-1, 1, 2),
            ).squeeze(1)                                 # (B, 2)
            in_range = (nearest_dist < self.obstacle_sense_range).float()
            # Scale the direction vector so that the magnitude carries the
            # "distance" information up to the sense range (rather than
            # the raw 0..inf distance). Outside the sense range we mask
            # the direction entirely (multiplied by in_range).
            cap = nearest_dist.clamp(max=self.obstacle_sense_range)
            unit_dir = nearest_dir / nearest_dist.clamp_min(1e-6)
            nearest_rel = in_range * unit_dir * cap       # (B, 2)
        else:
            in_range = torch.zeros_like(pos[:, :1])
            nearest_rel = torch.zeros_like(pos)

        return torch.cat(
            [pos, vel, goal_rel, nearest_rel, in_range], dim=-1
        )                                                # (B, 9)

    def done(self) -> Tensor:
        return self.on_goal

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "pos_rew": agent.pos_rew,
            "final_rew": self.final_rew,
            "agent_collision_rew": agent.agent_collision_rew,
        }

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        # Render goal as a green ring so it stays visible from above.
        from vmas.simulator import rendering

        geoms: List[Geom] = []
        goal_color = Color.GREEN.value
        ring = rendering.Line(
            self.goal.state.pos[env_index],
            self.goal.state.pos[env_index]
            + torch.zeros(2, device=self.goal.state.pos.device),
            width=2,
        )
        ring.set_color(*goal_color)
        geoms.append(ring)
        return geoms