# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""cuRobo build-time IK reachability extension."""

from __future__ import annotations

try:
    # Importing this submodule registers ReachabilityValidator as the ``ik_reachable`` build-time check.
    # Delist the validator check if the cuRobo deps are not installed.
    from isaaclab_arena_curobo import ik_reachability_validator as _ik_reachability_validator  # noqa: F401
except ImportError:
    pass
