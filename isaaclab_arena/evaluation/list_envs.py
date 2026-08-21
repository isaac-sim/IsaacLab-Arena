# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.assets.registries import EnvironmentRegistry
from isaaclab_arena_environments.cli import ensure_environments_registered

ensure_environments_registered()

[print(n) for n in sorted(EnvironmentRegistry().get_all_keys())]
