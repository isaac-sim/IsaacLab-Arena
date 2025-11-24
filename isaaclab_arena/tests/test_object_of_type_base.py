# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import tqdm

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

NUM_STEPS = 10
HEADLESS = True
MOVEMENT_EPS = 0.001


def _test_object_of_type_base(simulation_app):

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.object_library import LibraryObject
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.dummy_task import DummyTask
    from isaaclab_arena.utils.pose import Pose

    asset_registry = AssetRegistry()

    class CrackerBoxNoPhysics(LibraryObject):
        """
        Cracker box without physics.
        """

        name = "cracker_box_no_physics"
        tags = ["object"]
        usd_path = (
            "omniverse://isaac-dev.ov.nvidia.com/Projects/isaac_arena/assets_for_tests/cracker_box_base_asset.usd"
        )
        default_prim_path = "{ENV_REGEX_NS}/target_cracker_box_no_physics"
        object_type = ObjectType.BASE

        def __init__(self, prim_path: str = default_prim_path, initial_pose: Pose | None = None):
            super().__init__(prim_path=prim_path, initial_pose=initial_pose)

    # Scene
    background = asset_registry.get_asset_by_name("kitchen")()
    embodiment = asset_registry.get_asset_by_name("franka")()
    cracker_box = CrackerBoxNoPhysics()

    cracker_box.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

    scene = Scene(assets=[background, cracker_box])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="reference_object_test",
        embodiment=embodiment,
        scene=scene,
        task=DummyTask(),
        teleop_device=None,
    )

    try:

        # NOTE(alexmillane, 2025-09-15): The real test is just if the env compiles
        # here. If you we're to try to spawn a rigid object with the test usd path,
        # it would fail as the USD doesn't have a rigid body enabled.
        args_cli = get_isaaclab_arena_cli_parser().parse_args([])
        env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
        env = env_builder.make_registered()
        env.reset()

        position_before_simulation = torch.tensor(cracker_box.get_initial_pose().position_xyz)

        # Run some zero actions.
        for _ in tqdm.tqdm(range(NUM_STEPS)):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)

            # Check the the object is floating.
            position_after_simulation, _ = env.scene["cracker_box_no_physics"].get_world_poses()
            movement = position_after_simulation.cpu() - position_before_simulation.cpu()
            assert torch.norm(movement).item() < MOVEMENT_EPS, "Object moved. Should not have physics."

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def test_object_of_type_base():
    result = run_simulation_app_function(
        _test_object_of_type_base,
        headless=HEADLESS,
    )
    assert result, "Test failed"


if __name__ == "__main__":
    test_object_of_type_base()
