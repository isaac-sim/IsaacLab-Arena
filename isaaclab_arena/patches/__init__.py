# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Small, self-contained workarounds for upstream Isaac Sim / Isaac Lab issues.

Each module patches around one specific upstream bug and documents the condition under which it can be
removed. Prefer deleting a patch over extending it once the upstream fix lands.
"""

from isaaclab_arena.patches.camera_render_pose import CameraLocalOffsetWriter

__all__ = ["CameraLocalOffsetWriter"]
