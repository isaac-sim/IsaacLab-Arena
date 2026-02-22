# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from enum import Enum


class RLFramework(Enum):
    RSL_RL = "rsl_rl"
    SKRL = "skrl"
    RL_GAMES = "rl_games"
    SB3 = "sb3"

    def get_entry_point_string(self) -> str:
        return f"{self.value}_cfg_entry_point"
