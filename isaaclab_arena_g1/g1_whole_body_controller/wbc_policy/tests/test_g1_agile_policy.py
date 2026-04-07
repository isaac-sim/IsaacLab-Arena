# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Integration test for the G1 AGILE WBC policy.

Loads the G1 robot with the AGILE lower-body policy in an empty scene and
verifies that the robot remains standing (root z > 0.5 m) over several
hundred simulation steps.
"""

import torch
import traceback

import pytest
import warp as wp

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

NUM_STEPS = 500
MIN_ROOT_HEIGHT = 0.5
HEADLESS = True
ENABLE_CAMERAS = True


def _get_agile_test_env(num_envs: int = 1):
    """Create a G1 robot with the AGILE WBC policy in an empty scene."""
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.g1.g1 import G1WBCAgileJointEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    scene = Scene(assets=[])
    embodiment = G1WBCAgileJointEmbodiment(enable_cameras=ENABLE_CAMERAS)
    embodiment.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    env = IsaacLabArenaEnvironment(
        name="g1_agile_standing_test",
        embodiment=embodiment,
        scene=scene,
    )

    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    env_builder = ArenaEnvBuilder(env, args_cli)
    env = env_builder.make_registered()
    env.reset()
    return env


def _test_agile_standing(simulation_app) -> bool:
    """The robot should stay above 0.5 m when given zero upper-body actions."""
    try:
        env = _get_agile_test_env(num_envs=1)

        for step in range(NUM_STEPS):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                # Set base height command to 0.75 m (avoids squatting to match 0-height)
                actions[:, -4] = 0.75
                env.step(actions)

                root_z = wp.to_torch(env.unwrapped.scene["robot"].data.root_link_pose_w)[0, 2].item()
                assert (
                    root_z > MIN_ROOT_HEIGHT
                ), f"Robot fell at step {step}: root z = {root_z:.3f} m (min {MIN_ROOT_HEIGHT} m)"

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    finally:
        env.close()

    return True


@pytest.mark.with_cameras
def test_agile_standing():
    result = run_simulation_app_function(
        _test_agile_standing,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )
    assert result, f"Test {_test_agile_standing.__name__} failed"


if __name__ == "__main__":
    test_agile_standing()
