#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass


@dataclass
class TaskConfig:
    # Scenario limits
    max_steps: int = 200

    # Static obstacles
    n_static_obstacles: int = 3
    obstacle_radius: float = 0.15
    obstacle_sense_range: float = 0.6

    # World spawning area
    world_spawning_x: float = 1.0
    world_spawning_y: float = 1.0

    # Reward configuration
    shared_rew: bool = False
    final_reward: float = 0.01
    agent_collision_penalty: float = -1.0

    # Agent geometry
    agent_radius: float = 0.10