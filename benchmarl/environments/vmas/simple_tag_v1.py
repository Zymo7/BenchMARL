#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    num_good_agents: int = MISSING
    num_adversaries: int = MISSING
    observe_vel: bool = MISSING
    nearest_radius: float = MISSING
    bound: float = MISSING
    done_when_caught: bool = MISSING
    spawn_radius: float = MISSING
