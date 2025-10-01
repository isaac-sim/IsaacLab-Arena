# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import tqdm

from isaac_arena.tests.utils.subprocess import run_simulation_app_function_in_separate_process

NUM_STEPS = 100
HEADLESS = False

# Test description.
# We start 2 envs. In these two envs:
# - 1 : The object falls in the drawer resulting in a success.
# - 2 : The object falls out of the drawer resulting in a failure.
# We expect the success rate to be 0.5 and the object moved rate to be 1.0.
# We allow for
# - Success rate error: 10% because the two environments reset at different rates
#   due to the different height that the object falls from.
# - Object moved rate error: 5% to allow for the case where in the last run
#   the object doesn't move much (in practice I haven't seen this happen).
EXPECTED_SUCCESS_RATE = 0.5
ALLOWABLE_SUCCESS_RATE_ERROR = 0.1
EXPECTED_OBJECT_MOVED_RATE = 1.0
ALLOWABLE_OBJECT_MOVED_RATE_ERROR = 0.05


def _test_metrics(simulation_app):
    """Returns a scene which we use for these tests."""

    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    from isaac_arena.assets.asset_registry import AssetRegistry
    from isaac_arena.assets.object_reference import ObjectReference
    from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
    from isaac_arena.environments.compile_env import ArenaEnvBuilder
    from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
    from isaac_arena.metrics.metrics import compute_metrics
    from isaac_arena.scene.scene import Scene
    from isaac_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaac_arena.terms.events import set_object_pose_per_env
    from isaac_arena.utils.pose import Pose

    asset_registry = AssetRegistry()

    background = asset_registry.get_asset_by_name("kitchen")()
    embodiment = asset_registry.get_asset_by_name("franka")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()
    destination_location = ObjectReference(
        name="destination_location",
        prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
        parent_asset=background,
    )

    scene = Scene(assets=[background, cracker_box])
    isaac_arena_environment = IsaacArenaEnvironment(
        name="robot_initial_position",
        embodiment=embodiment,
        scene=scene,
        task=PickAndPlaceTask(cracker_box, destination_location, background),
        teleop_device=None,
    )

    # Build the cfg, but dont register so we can make some adjustments.
    NUM_ENVS = 2
    args_cli = get_isaac_arena_cli_parser().parse_args([])
    args_cli.num_envs = NUM_ENVS
    env_builder = ArenaEnvBuilder(isaac_arena_environment, args_cli)
    env_cfg = env_builder.compose_manager_cfg()

    # Replace the pose reset term:
    # - from: constant per env,
    # - to: per env pose
    pose_list = [
        # Success (in the drawer)
        Pose(position_xyz=(0.0, -0.5, 0.2), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)),
        # Fail (out of the drawer)
        Pose(position_xyz=(-0.5, -0.5, 0.2), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)),
    ]
    env_cfg.events.reset_pick_up_object_pose = EventTermCfg(
        func=set_object_pose_per_env,
        mode="reset",
        params={
            "pose_list": pose_list,
            "asset_cfg": SceneEntityCfg(cracker_box.name),
        },
    )

    env = env_builder.make_registered(env_cfg)
    env.reset()

    try:

        # Run some zero actions.
        for _ in tqdm.tqdm(range(NUM_STEPS)):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)

        metrics = compute_metrics(env)
        print(f"Metrics: {metrics}")
        assert "success_rate" in metrics
        assert "object_moved_rate" in metrics
        success_rate = metrics["success_rate"]
        object_moved_rate = metrics["object_moved_rate"]
        print(f"Success rate: {success_rate}")
        print(f"Object moved rate: {object_moved_rate}")
        assert abs(success_rate - EXPECTED_SUCCESS_RATE) < ALLOWABLE_SUCCESS_RATE_ERROR
        assert abs(object_moved_rate - EXPECTED_OBJECT_MOVED_RATE) < ALLOWABLE_OBJECT_MOVED_RATE_ERROR

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def test_metrics():
    result = run_simulation_app_function_in_separate_process(
        _test_metrics,
        headless=HEADLESS,
    )
    assert result, f"Test {test_metrics.__name__} failed"


if __name__ == "__main__":
    test_metrics()
