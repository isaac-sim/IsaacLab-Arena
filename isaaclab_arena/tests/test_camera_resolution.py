# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test camera-rig resolution configuration."""


def test_droid_camera_rig_keeps_production_dimensions_by_default():
    """Keep the maintained production dimensions unchanged when no override is requested."""
    from isaaclab_arena.embodiments.droid.droid import DroidCameraCfg

    camera_rig = DroidCameraCfg()

    assert camera_rig.camera_names() == ["external_camera", "external_camera_2", "wrist_camera"]
    for camera_name in camera_rig.camera_names():
        camera_cfg = getattr(camera_rig, camera_name)
        assert (camera_cfg.height, camera_cfg.width) == (720, 1280)


def test_camera_rig_resolution_applies_to_base_and_tiled_cameras():
    """Apply an explicit image size to every camera before the rig is tiled."""
    from isaaclab_arena.embodiments.droid.droid import DroidCameraCfg

    camera_rig = DroidCameraCfg()
    camera_rig.set_resolution(height=360, width=640)

    for configured_rig in (camera_rig, camera_rig.get_cfg()):
        for camera_name in configured_rig.camera_names():
            camera_cfg = getattr(configured_rig, camera_name)
            assert (camera_cfg.height, camera_cfg.width) == (360, 640)
