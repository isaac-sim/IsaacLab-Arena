# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena.tests.utils.subprocess import run_subprocess


def _test_separate_camera_renderer_supported_settings(simulation_app):
    from isaaclab.renderers import RenderBufferKind, RendererCfg
    from isaaclab.sensors import CameraCfg

    from isaaclab_arena.utils.isaaclab_utils.separate_camera_renderer import (
        SeparateCameraRenderer,
        assert_camera_cfg_compatible_with_independent_rendering,
    )

    for data_types in (["rgb"], ["rgba"], ["rgb", "rgba"]):
        camera_cfg = CameraCfg(data_types=data_types)
        assert_camera_cfg_compatible_with_independent_rendering(camera_cfg)
    renderer = SeparateCameraRenderer(camera_cfg.renderer_cfg)
    assert set(renderer.supported_output_types()) == {RenderBufferKind.RGB, RenderBufferKind.RGBA}
    for field, value, message in [
        ("data_types", [], "only RGB/RGBA"),
        ("data_types", ["depth"], "only RGB/RGBA"),
        ("isp_cfg", object(), "do not support ISP"),
        ("background_color", (1.0, 0.0, 0.0), "custom background"),
        ("renderer_cfg", RendererCfg(renderer_type="other"), "Isaac RTX"),
    ]:
        camera_cfg = CameraCfg(data_types=["rgb"])
        setattr(camera_cfg, field, value)
        with pytest.raises(AssertionError, match=message):
            assert_camera_cfg_compatible_with_independent_rendering(camera_cfg)
    camera_cfg = CameraCfg(data_types=["rgb"])
    camera_cfg.renderer_cfg.class_type = object
    with pytest.raises(AssertionError, match="custom renderer"):
        assert_camera_cfg_compatible_with_independent_rendering(camera_cfg)
    return True


def _test_separate_camera_renderer_buffers_and_lifecycle(simulation_app):
    import torch

    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.isaaclab_utils.separate_camera_renderer import SeparateCameraRenderer

    embodiment = DroidAbsoluteJointPositionEmbodiment(enable_cameras=True)
    rig = embodiment.camera_config
    rig.set_use_tiled_camera(False)
    for name in rig.camera_names():
        camera_cfg = getattr(rig, name)
        camera_cfg.width, camera_cfg.height = 64, 48
        camera_cfg.data_types = ["rgb", "rgba"]
    arena_env = IsaacLabArenaEnvironment(name="test_separate_camera_renderer", embodiment=embodiment, scene=Scene())
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=2)).make_registered()
    try:
        env.reset()
        cameras = [env.unwrapped.scene[name] for name in rig.camera_names()]
        renderer = cameras[0]._renderer
        assert isinstance(renderer, SeparateCameraRenderer)
        assert all(camera._renderer is renderer for camera in cameras)
        render_data = [camera._render_data for camera in cameras]
        products = [child.render_product.path for data in render_data for child in data]
        assert len(set(products)) == 2 * len(cameras)
        for camera in cameras:
            output = camera.data.output
            assert output["rgb"].torch.shape == (2, 48, 64, 3)
            assert output["rgba"].torch.shape == (2, 48, 64, 4)
            torch.testing.assert_close(output["rgb"].torch, output["rgba"].torch[..., :3])
        env.reset()
        assert [child.render_product.path for camera in cameras for child in camera._render_data] == products
    finally:
        env.close()
    assert all(child.render_product is None for data in render_data for child in data)
    for data in render_data:
        renderer.cleanup(data)
    renderer.cleanup(None)
    return True


@pytest.mark.with_cameras
def test_separate_camera_renderer_supported_settings():
    assert run_function_with_persistent_simulation_app(
        _test_separate_camera_renderer_supported_settings, headless=True, enable_cameras=True
    )


@pytest.mark.with_cameras
@pytest.mark.with_subprocess
def test_separate_camera_renderer_buffers_and_lifecycle():
    run_subprocess([TestConstants.python_path, __file__])


if __name__ == "__main__":
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.enable_cameras = True
    with SimulationAppContext(args_cli) as simulation_app_context:
        assert _test_separate_camera_renderer_buffers_and_lifecycle(simulation_app_context.app_launcher.app)
