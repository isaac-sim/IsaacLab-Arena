# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import traceback

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

NUM_STEPS = 50
HEADLESS = True


def _test_g1_locomanip_object_on_destination_termination(simulation_app) -> bool:

    from isaaclab.managers import SceneEntityCfg

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.locomanip_pick_and_place_task import LocomanipPickAndPlaceTask
    from isaaclab_arena.tasks.terminations import object_on_destination
    from isaaclab_arena.utils.pose import Pose

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args([])

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("galileo_locomanip")()
    brown_box = asset_registry.get_asset_by_name("brown_box")()
    blue_sorting_bin = asset_registry.get_asset_by_name("blue_sorting_bin")()

    blue_sorting_bin.set_initial_pose(
        Pose(
            position_xyz=(-0.2450, -1.6272, -0.2641),
            rotation_xyzw=(0.0, 0.0, 1.0, 0.0),
        )
    )
    brown_box.set_initial_pose(
        Pose(
            position_xyz=(-0.2450, -1.6272, 0.5),
            rotation_xyzw=(0, 0, 0, 1),
        )
    )

    scene = Scene(assets=[background, brown_box, blue_sorting_bin])

    task = LocomanipPickAndPlaceTask(
        pick_up_object=brown_box,
        destination_location=blue_sorting_bin,
        background_scene=background,
        force_threshold=0.5,
        velocity_threshold=0.1,
    )

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="galileo_g1_locomanip_pick_and_place",
        scene=scene,
        task=task,
    )

    builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env = builder.make_registered()
    env.reset()

    try:
        condition_met_vec = []
        terminated_vec = []
        for _ in range(NUM_STEPS):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                _, _, terminated, _, _ = env.step(actions)

                condition_met_vec.append(
                    object_on_destination(
                        env,
                        object_cfg=SceneEntityCfg(brown_box.name),
                        contact_sensor_cfg=SceneEntityCfg("pick_up_object_contact_sensor"),
                        force_threshold=0.5,
                        velocity_threshold=0.1,
                    )
                )
                terminated_vec.append(terminated.item())

    except Exception:
        print("Error: {e}")
        traceback.print_exc()
        return False

    finally:
        env.close()

    assert condition_met_vec[0].item() is False, "Object started in the bin"
    assert any(condition_met_vec), "Object did not end in the bin"
    assert any(terminated_vec), "The task was not terminated"
    assert condition_met_vec[-1].item() is False, "Object was not moved above the bin"

    return True


def test_g1_locomanip_object_on_destination_termination():
    result = run_simulation_app_function(
        _test_g1_locomanip_object_on_destination_termination,
        headless=HEADLESS,
    )
    assert result, "Test failed"


if __name__ == "__main__":
    test_g1_locomanip_object_on_destination_termination()
