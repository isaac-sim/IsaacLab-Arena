# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test the renderer selection of camera rigs."""

import pytest


def test_camera_rig_selects_newton_warp_for_every_camera():
    """Render every camera with Newton Warp, following the link it is mounted on, and leave the source rig reusable."""
    from isaaclab_newton.renderers import NewtonWarpRendererCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    from isaaclab_arena.embodiments.droid.droid import DroidCameraCfg

    source_rig = DroidCameraCfg()
    source_rig.set_renderer("newton_warp")
    rig = source_rig.get_cfg()

    for camera_name in rig.camera_names():
        camera_cfg = getattr(rig, camera_name)
        assert isinstance(camera_cfg.renderer_cfg, NewtonWarpRendererCfg)
        assert camera_cfg.update_latest_camera_pose
        assert isinstance(getattr(source_rig, camera_name).renderer_cfg, IsaacRtxRendererCfg)


def test_camera_rig_keeps_untiled_cameras_on_isaac_rtx():
    """Reject Newton Warp for untiled cameras, which only the Isaac RTX renderer provides."""
    from isaaclab_arena.embodiments.droid.droid import DroidCameraCfg

    rig = DroidCameraCfg()
    rig.set_use_tiled_camera(False)
    rig.set_renderer("newton_warp")

    with pytest.raises(AssertionError, match="Newton Warp renders tiled cameras only"):
        rig.get_cfg()
