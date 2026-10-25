# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from isaaclab_arena.policy.camera_requirement import CameraRequirement


def test_droid_cameras_render_above_the_target_size():
    from isaaclab_arena_gr00t.policy.gr00t_core import camera_requirements_for_policy

    policy_config = SimpleNamespace(
        pov_cam_name_sim=["external_camera_rgb", "wrist_camera_rgb"], target_image_size=(180, 320, 3)
    )
    requirements = camera_requirements_for_policy(policy_config)
    assert requirements == {
        "external_camera": CameraRequirement(360, 640),
        "wrist_camera": CameraRequirement(360, 640),
    }
    # A 1280x720 DROID camera renders at 640x360, so the policy's pad-and-resize step still runs.
    assert requirements["wrist_camera"].render_size(720, 1280) == (360, 640)


def test_camera_at_target_size_is_unchanged():
    from isaaclab_arena_gr00t.policy.gr00t_core import camera_requirements_for_policy

    policy_config = SimpleNamespace(pov_cam_name_sim="robot_head_cam_rgb", target_image_size=(480, 640, 3))
    requirements = camera_requirements_for_policy(policy_config)
    assert requirements["robot_head_cam"].render_size(480, 640) == (480, 640)
