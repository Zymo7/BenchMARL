#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    n_leaders: int = MISSING
    n_followers: int = MISSING
    neighbor_radius: float = MISSING
    target_pos_x: float = MISSING
    target_pos_y: float = MISSING
    min_spawn_dist: float = MISSING
    spawn_radius: float = MISSING
    light_eps: float = MISSING
    target_speed: float = MISSING
    direction_change_interval: float = MISSING
    x_dim: float = MISSING
    y_dim: float = MISSING