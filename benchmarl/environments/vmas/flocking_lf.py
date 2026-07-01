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
    clustered_spawn: bool = MISSING
    spawn_cluster_center_x: float = MISSING
    spawn_cluster_center_y: float = MISSING
    spawn_cluster_radius: float = MISSING