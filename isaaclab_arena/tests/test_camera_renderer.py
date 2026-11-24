# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test the renderer selection of camera rigs."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_camera_rig_selects_newton_warp_for_every_camera(simulation_app):
    # The renderer configs import Kit modules, so they are imported after the SimulationApp starts.
    from isaaclab_newton.renderers import NewtonWarpRendererCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    from isaaclab_arena.embodiments.droid.droid import DroidCameraCfg

    source_rig = DroidCameraCfg()
    # Use nondefault output settings to verify they carry over to Newton Warp.
    source_rig.wrist_camera.renderer_cfg.depth_clipping_behavior = "max"
    source_rig.wrist_camera.renderer_cfg.semantic_filter = "class:*"
    source_rig.set_renderer("newton_warp")
    rig = source_rig.get_cfg()

    for camera_name in rig.camera_names():
        camera_cfg = getattr(rig, camera_name)
        assert isinstance(camera_cfg.renderer_cfg, NewtonWarpRendererCfg)
        assert camera_cfg.update_latest_camera_pose
        assert isinstance(getattr(source_rig, camera_name).renderer_cfg, IsaacRtxRendererCfg)
    assert rig.wrist_camera.renderer_cfg.depth_clipping_behavior == "max"
    assert rig.wrist_camera.renderer_cfg.semantic_filter == "class:*"
    return True


def test_camera_rig_selects_newton_warp_for_every_camera():
    """Render every camera with Newton Warp, keeping its output settings, and leave the source rig reusable."""
    assert run_function_with_persistent_simulation_app(_test_camera_rig_selects_newton_warp_for_every_camera)


def _test_camera_rig_rejects_newton_warp_for_untiled_cameras(simulation_app):
    from isaaclab_arena.embodiments.droid.droid import DroidCameraCfg

    rig = DroidCameraCfg()
    rig.set_use_tiled_camera(False)
    rig.set_renderer("newton_warp")

    with pytest.raises(AssertionError, match="Newton Warp renders tiled cameras only"):
        rig.get_cfg()
    return True


def test_camera_rig_rejects_newton_warp_for_untiled_cameras():
    """Reject Newton Warp for untiled cameras, which only the Isaac RTX renderer provides."""
    assert run_function_with_persistent_simulation_app(_test_camera_rig_rejects_newton_warp_for_untiled_cameras)
