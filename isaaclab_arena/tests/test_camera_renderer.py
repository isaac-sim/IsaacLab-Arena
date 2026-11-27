# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test the renderer selection of camera rigs."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena.utils.camera_renderer import CameraRenderer


def _test_camera_rig_selects_renderer_for_every_camera(simulation_app, renderer):
    # The renderer configs import Kit modules, so they are imported after the SimulationApp starts.
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    from isaaclab_arena.embodiments.droid.droid import DroidCameraCfg

    source_rig = DroidCameraCfg()
    # Use nondefault output settings to verify they carry over.
    source_rig.wrist_camera.renderer_cfg.colorize_semantic_segmentation = False
    source_rig.wrist_camera.renderer_cfg.depth_clipping_behavior = "max"
    source_rig.wrist_camera.renderer_cfg.semantic_filter = "class:*"
    source_rig.set_renderer(renderer)
    rig = source_rig.get_cfg()

    for camera_name in rig.camera_names():
        camera_cfg = getattr(rig, camera_name)
        assert camera_cfg.renderer_cfg.renderer_type == renderer
        assert camera_cfg.update_latest_camera_pose
        assert isinstance(getattr(source_rig, camera_name).renderer_cfg, IsaacRtxRendererCfg)
    assert not rig.wrist_camera.renderer_cfg.colorize_semantic_segmentation
    if renderer is CameraRenderer.NEWTON_WARP:
        assert rig.wrist_camera.renderer_cfg.depth_clipping_behavior == "max"
        assert rig.wrist_camera.renderer_cfg.semantic_filter == "class:*"
    return True


@pytest.mark.parametrize("renderer", [CameraRenderer.NEWTON_WARP, CameraRenderer.OVRTX])
def test_camera_rig_selects_renderer_for_every_camera(renderer):
    """Render every camera with the renderer, keeping its output settings, and leave the source rig reusable."""
    assert run_function_with_persistent_simulation_app(
        _test_camera_rig_selects_renderer_for_every_camera, renderer=renderer
    )


def _test_camera_rig_rejects_untiled_cameras(simulation_app, renderer):
    from isaaclab_arena.embodiments.droid.droid import DroidCameraCfg

    rig = DroidCameraCfg()
    rig.set_use_tiled_camera(False)
    rig.set_renderer(renderer)

    with pytest.raises(AssertionError, match=f"{renderer} renders tiled cameras only"):
        rig.get_cfg()
    return True


@pytest.mark.parametrize("renderer", [CameraRenderer.NEWTON_WARP, CameraRenderer.OVRTX])
def test_camera_rig_rejects_untiled_cameras(renderer):
    """Reject Newton Warp and OVRTX for untiled cameras, which only the Isaac RTX renderer provides."""
    assert run_function_with_persistent_simulation_app(_test_camera_rig_rejects_untiled_cameras, renderer=renderer)


def _test_camera_rig_rejects_newton_warp_for_other_renderers(simulation_app):
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    from isaaclab_arena.embodiments.droid.droid import DroidCameraCfg

    rig = DroidCameraCfg()
    rig.wrist_camera.renderer_cfg = NewtonWarpRendererCfg()
    rig.set_renderer(CameraRenderer.NEWTON_WARP)

    with pytest.raises(AssertionError, match="newton_warp replaces the Isaac RTX renderer"):
        rig.get_cfg()
    return True


def test_camera_rig_rejects_newton_warp_for_other_renderers():
    """Reject Newton Warp for a camera that does not start from the Isaac RTX renderer."""
    assert run_function_with_persistent_simulation_app(_test_camera_rig_rejects_newton_warp_for_other_renderers)


def _test_camera_rig_rejects_renderer_strings(simulation_app):
    from isaaclab_arena.embodiments.droid.droid import DroidCameraCfg

    with pytest.raises(AssertionError, match="Expected a CameraRenderer"):
        DroidCameraCfg().set_renderer("newton_warp")
    return True


def test_camera_rig_rejects_renderer_strings():
    """Reject a plain string, which would otherwise select Isaac RTX without an error."""
    assert run_function_with_persistent_simulation_app(_test_camera_rig_rejects_renderer_strings)
