#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    """Configuration for the HGN formation-control task.

    The VMAS scenario is ``benchmarl_hgn_formation`` (see vmas install dir).
    Agents spawn clustered in a small box and must reach a fixed target
    formation (circle / line / V / grid) parameterized by ``formation_type``.
    """

    max_steps: int = MISSING
    n_agents: int = MISSING
    formation_type: str = MISSING
    formation_radius: float = MISSING
    spawn_radius: float = MISSING
    spawn_cluster_center_x: float = MISSING
    spawn_cluster_center_y: float = MISSING
    spawn_cluster_radius: float = MISSING
    n_obstacles: int = MISSING
    obstacle_radius: float = MISSING
    agent_radius: float = MISSING
    moving_target: bool = MISSING
    target_speed: float = MISSING
