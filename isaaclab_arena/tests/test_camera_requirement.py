# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.policy.camera_requirement import CameraRequirement


def test_render_size_fits_native_aspect_inside_the_box():
    assert CameraRequirement(224, 224).render_size(720, 1280) == (126, 224)
    assert CameraRequirement(180, 320).render_size(720, 1280) == (180, 320)
    assert CameraRequirement(224, 224).render_size(1280, 720) == (224, 126)


def test_render_size_never_upscales():
    assert CameraRequirement(960, 1280).render_size(480, 640) == (480, 640)
    assert CameraRequirement(360, 640).render_size(720, 1280) == (360, 640)
