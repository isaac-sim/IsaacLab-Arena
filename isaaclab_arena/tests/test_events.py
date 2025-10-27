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

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

NUM_STEPS = 10
HEADLESS = True
INITIAL_POSITION_EPS = 0.1  # The cracker box falls slightly.


def _test_set_object_pose_per_env_event(simulation_app):
    """Returns a scene which we use for these tests."""

    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.terms.events import set_object_pose_per_env
    from isaaclab_arena.utils.pose import Pose

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
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="robot_initial_position",
        embodiment=embodiment,
        scene=scene,
        task=PickAndPlaceTask(cracker_box, destination_location, background),
        teleop_device=None,
    )

    # Build the cfg, but dont register so we can make some adjustments.
    NUM_ENVS = 2
    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.num_envs = NUM_ENVS
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env_cfg = env_builder.compose_manager_cfg()

    # Replace the pose reset term:
    # - from: constant per env,
    # - to: per env pose
    pose_list = [
        Pose(position_xyz=(0.4, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)),
        Pose(position_xyz=(0.4, 0.4, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)),
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
    # disable control on stop
    env.unwrapped.sim._app_control_on_stop_handle = None
    env.reset()

    try:

        # Run some zero actions.
        for _ in tqdm.tqdm(range(NUM_STEPS)):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)

        # Check that the cracker box ended up in the correct position.
        cracker_box_poses = cracker_box.get_object_pose(env)
        initial_poses = torch.cat(
            (
                pose_list[0].to_tensor(device=env.device).unsqueeze(0),
                pose_list[1].to_tensor(device=env.device).unsqueeze(0),
            ),
            dim=0,
        )
        position_errors = torch.norm(cracker_box_poses[:, :3] - initial_poses[:, :3], dim=1)
        print(f"Cranker box poses: {cracker_box_poses}")
        print(f"Initial poses: {initial_poses}")
        print(f"Position errors: {position_errors}")
        assert torch.all(position_errors < INITIAL_POSITION_EPS), "Position errors are too large"

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def test_set_object_post_per_env_event():
    result = run_simulation_app_function(
        _test_set_object_pose_per_env_event,
        headless=HEADLESS,
    )
    assert result, f"Test {test_set_object_post_per_env_event.__name__} failed"


if __name__ == "__main__":
    test_set_object_post_per_env_event()
