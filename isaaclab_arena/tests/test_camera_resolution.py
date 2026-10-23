# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test camera-rig resolution configuration."""

import pytest


def test_droid_camera_rig_keeps_production_dimensions_by_default():
    """Keep the maintained production dimensions unchanged when no override is requested."""
    from isaaclab_arena.embodiments.droid.droid import DroidCameraCfg

    camera_rig = DroidCameraCfg()

    assert camera_rig.camera_names() == ["external_camera", "external_camera_2", "wrist_camera"]
    for camera_name in camera_rig.camera_names():
        camera_cfg = getattr(camera_rig, camera_name)
        assert (camera_cfg.height, camera_cfg.width) == (720, 1280)


def test_camera_rig_fit_resolution_applies_to_base_and_tiled_cameras():
    """Fit every camera before the rig is tiled, so tiled and untiled rigs render the same size."""
    from isaaclab_arena.embodiments.droid.droid import DroidCameraCfg

    camera_rig = DroidCameraCfg()
    camera_rig.fit_resolution(360, 640)

    for configured_rig in (camera_rig, camera_rig.get_cfg()):
        for camera_name in configured_rig.camera_names():
            camera_cfg = getattr(configured_rig, camera_name)
            assert (camera_cfg.height, camera_cfg.width) == (360, 640)


@pytest.mark.parametrize(
    ("height", "width", "max_height", "max_width", "expected"),
    [
        (720, 1280, 224, 224, (126, 224)),
        (720, 1280, 360, 640, (360, 640)),
        (720, 1280, 180, 320, (180, 320)),
        (480, 640, 224, 224, (168, 224)),
        (512, 512, 224, 224, (224, 224)),
        (720, 1280, 2000, 2000, (1125, 2000)),
    ],
)
def test_fit_image_size_matches_resize_with_pad_content_area(height, width, max_height, max_width, expected):
    """Match the content area tf.image.resize_with_pad produces inside the bounding box."""
    from isaaclab_arena.utils.cameras import fit_image_size

    assert fit_image_size(height, width, max_height, max_width) == expected


def test_camera_rig_fit_resolution_preserves_aspect_ratio_per_camera():
    """Fit every camera of the rig into the policy's box instead of forcing one size on all of them."""
    from isaaclab_arena.embodiments.droid.droid import DroidCameraCfg

    camera_rig = DroidCameraCfg()
    for camera_name in ("external_camera", "external_camera_2"):
        camera_cfg = getattr(camera_rig, camera_name)
        camera_cfg.height, camera_cfg.width = 480, 640
    camera_rig.fit_resolution(224, 224)

    assert (camera_rig.external_camera.height, camera_rig.external_camera.width) == (168, 224)
    assert (camera_rig.external_camera_2.height, camera_rig.external_camera_2.width) == (168, 224)
    assert (camera_rig.wrist_camera.height, camera_rig.wrist_camera.width) == (126, 224)


def test_camera_rig_fit_resolution_rejects_a_fit_that_changes_the_aspect_ratio(monkeypatch):
    """Guard the camera intrinsics: a fit that strays from the native aspect ratio is an error."""
    from isaaclab_arena.embodiments.droid.droid import DroidCameraCfg
    from isaaclab_arena.utils import cameras

    monkeypatch.setattr(cameras, "fit_image_size", lambda height, width, max_height, max_width: (224, 224))

    with pytest.raises(AssertionError, match="changes the aspect ratio"):
        DroidCameraCfg().fit_resolution(224, 224)
