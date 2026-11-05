# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check Bluefield camera offsets across repeated and partial resets."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_torso_camera_reset(simulation_app):
    import torch

    from isaaclab.sensors import CameraCfg
    from isaaclab.sim import PinholeCameraCfg
    from isaaclab.utils.configclass import configclass
    from pxr import UsdGeom

    from isaaclab_arena.embodiments.no_embodiment import NoEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.cameras import ArenaCameraCfg
    from isaaclab_arena_environments.mdp.bluefield_camera_position_variation import BluefieldCameraPositionVariation

    @configclass
    class CameraRig(ArenaCameraCfg):
        robot_head_cam: CameraCfg = CameraCfg(
            prim_path="{ENV_REGEX_NS}/torso_link/RobotHeadCam",
            width=64,
            height=48,
            spawn=PinholeCameraCfg(),
            offset=CameraCfg.OffsetCfg(pos=(0.2, -0.1, 0.6), rot=(0.0, 0.7071068, 0.0, 0.7071068), convention="ros"),
        )

    variation = BluefieldCameraPositionVariation()
    first = [0.01, -0.02, 0.03]
    variation.cfg.sampler_cfg.low = first
    variation.cfg.sampler_cfg.high = first
    variation.apply_cfg(variation.cfg)
    variation.enable()
    embodiment = NoEmbodiment()
    embodiment.enable_cameras = True
    embodiment.camera_config = CameraRig()
    embodiment.add_variation(variation)
    description = IsaacLabArenaEnvironment(name="bluefield_camera_offset_test", embodiment=embodiment, scene=Scene())
    env = ArenaEnvBuilder(description, ArenaEnvBuilderCfg(num_envs=2, solve_relations=False)).make_registered()
    try:
        env.reset()
        base = env.unwrapped
        camera = base.scene["robot_head_cam"]
        nominal = torch.tensor(camera.cfg.offset.pos, device=base.device)
        positions, orientations = camera._view.get_local_poses()
        original_orientation = orientations.torch.clone()
        torch.testing.assert_close(positions.torch, (nominal + nominal.new_tensor(first)).expand(2, -1))
        term = base.event_manager.get_term_cfg("bluefield_ego_camera_offset_torso")
        second = [-0.02, 0.005, -0.01]
        term.params["sampler"].low[:] = torch.tensor(second)
        term.params["sampler"].high[:] = torch.tensor(second)
        for _ in range(2):
            base.reset(env_ids=torch.tensor([1], device=base.device))
            positions, orientations = camera._view.get_local_poses()
            expected = torch.stack([nominal + nominal.new_tensor(first), nominal + nominal.new_tensor(second)])
            torch.testing.assert_close(positions.torch, expected)
            torch.testing.assert_close(orientations.torch, original_orientation)
            # Check the render-side USD prim as well as the physics view.
            for i in range(2):
                prim = base.sim.stage.GetPrimAtPath(f"/World/envs/env_{i}/torso_link/RobotHeadCam")
                local = UsdGeom.Xformable(prim).GetLocalTransformation().ExtractTranslation()
                torch.testing.assert_close(nominal.new_tensor(tuple(local)), expected[i])
    finally:
        env.close()
    return True


@pytest.mark.with_cameras
def test_torso_camera_reset():
    assert run_function_with_persistent_simulation_app(_test_torso_camera_reset, enable_cameras=True)
