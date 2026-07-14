# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from isaaclab_arena.assets.register import register_camera_profile
from isaaclab_arena.embodiments.camera_profile import CameraProfileBase


@register_camera_profile
class DroidWorkspaceExteriorRgbdProfile(CameraProfileBase):
    """DROID workspace exterior RGB-D camera profile used by the GaP bridge."""

    name = "droid_workspace_exterior_rgbd"
    description = "Exterior workspace RGB-D camera named exterior_cam, 800x512, with live pose and intrinsics."
    compatible_embodiments = frozenset({"droid_abs_joint_pos"})

    @classmethod
    def apply(cls, embodiment):
        from isaaclab_arena.variations.camera_extrinsics_variation import CameraExtrinsicsVariation
        from isaaclab_arena_environments.maple_cameras import MapleDroidPerceptionCameraCfg

        embodiment.camera_config = MapleDroidPerceptionCameraCfg()
        variation = CameraExtrinsicsVariation(camera_name="exterior_cam")
        if variation.name not in embodiment.variations:
            embodiment.add_variation(variation)
