from typing import List
from dataclasses import dataclass, field


@dataclass
class TaskConfig:
    # Scenario limits
    max_steps: int = 200

    # Number of agents
    n_agents: int = 3

    # Static obstacles
    n_static_obstacles: int = 2
    static_obstacle_radius: float = 0.15

    # Dynamic obstacles
    n_dynamic_obstacles: int = 2
    dynamic_obstacle_radius: float = 0.1
    dynamic_obstacle_modes: List[str] = field(default_factory=lambda: ["linear", "circular"])

    # Dynamic obstacle parameters
    obstacle_speed: float = 0.02
    obstacle_linear_range: float = 0.5
    obstacle_circular_radius: float = 0.3
    obstacle_random_noise: float = 0.1

    # World spawning area
    world_spawning_x: float = 1.0
    world_spawning_y: float = 1.0

    # Lidar sensor configuration
    lidar_range: float = 0.3
    n_lidar_rays: int = 12

    # Reward configuration
    shared_rew: bool = False
    final_reward: float = 0.01
    agent_collision_penalty: float = -2.0
    obstacle_collision_penalty: float = -1.0

    # Agent geometry
    agent_radius: float = 0.1

    # Rendering
    comms_rendering_range: float = 0.0
