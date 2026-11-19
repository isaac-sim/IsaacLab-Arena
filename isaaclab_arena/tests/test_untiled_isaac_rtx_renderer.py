# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess


def _test_untiled_isaac_rtx_renderer_produces_images_and_cleans_up(simulation_app):
    import torch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.utils.isaaclab_utils.untiled_isaac_rtx_renderer import UnTiledIsaacRtxRenderer
    from isaaclab_arena_environments.pick_and_place_maple_table_environment import (
        PickAndPlaceMapleTableEnvironment,
        PickAndPlaceMapleTableEnvironmentCfg,
    )

    arena_env = PickAndPlaceMapleTableEnvironment().build(PickAndPlaceMapleTableEnvironmentCfg(enable_cameras=True))
    rig = arena_env.embodiment.camera_config
    rig.set_use_tiled_camera(False)
    for name in rig.camera_names():
        getattr(rig, name).data_types = ["rgb", "rgba"]
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=2)).make_registered()
    try:
        env.reset()
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        for _ in range(2):
            observations, _, _, _, _ = env.step(actions)
        for name in rig.camera_names():
            image = observations["camera_obs"][f"{name}_rgb"]
            camera_cfg = getattr(rig, name)
            assert image.shape == (2, camera_cfg.height, camera_cfg.width, 3)
            assert image.any(), "The renderer must produce a nonblank image"
            assert image.float().std() > 1.0, "The image must contain scene detail"

        cameras = [env.unwrapped.scene[name] for name in rig.camera_names()]
        renderer = cameras[0]._renderer
        assert isinstance(renderer, UnTiledIsaacRtxRenderer)
        assert all(camera._renderer is renderer for camera in cameras)
        render_data = [camera._render_data for camera in cameras]
        products = [child.render_product.path for data in render_data for child in data]
        assert len(set(products)) == 2 * len(cameras)
        for camera in cameras:
            output = camera.data.output
            assert output["rgba"].torch.shape == (*output["rgb"].torch.shape[:-1], 4)
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
@pytest.mark.with_subprocess
def test_untiled_isaac_rtx_renderer_produces_images_and_cleans_up():
    run_subprocess([TestConstants.python_path, __file__])


if __name__ == "__main__":
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.enable_cameras = True
    with SimulationAppContext(args_cli) as simulation_app_context:
        assert _test_untiled_isaac_rtx_renderer_produces_images_and_cleans_up(simulation_app_context.app_launcher.app)
