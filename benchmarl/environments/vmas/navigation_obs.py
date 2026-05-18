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

    # Agent dynamics composition (total n_agents = holonomic + diff_drive + car)
    n_agents_holonomic: int = 3
    n_agents_diff_drive: int = 0
    n_agents_car: int = 0

    # Number of obstacles
    n_obstacles: int = 3

    # World spawning area
    world_spawning_x: float = 1.0
    world_spawning_y: float = 1.0

    # Lidar sensor configuration
    lidar_range: float = 0.3
    n_lidar_rays: int = 12

    # Reward configuration
    shared_rew: bool = False
    final_reward: float = 0.01
    agent_collision_penalty: float = -1.0

    # Agent geometry
    agent_radius: float = 0.1

    # Rendering (communication lines range, 0 = disabled)
    comms_rendering_range: float = 0.0