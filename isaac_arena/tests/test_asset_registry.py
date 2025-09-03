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

import gymnasium as gym
import torch
import tqdm

from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
from isaac_arena.geometry.pose import Pose
from isaac_arena.tests.utils.subprocess import run_simulation_app_function_in_separate_process

NUM_STEPS = 2
HEADLESS = True
OBJECT_SEPARATION = 0.2


def _test_default_assets_registered(simulation_app):
    from isaac_arena.assets.registry import AssetRegistry

    asset_registry = AssetRegistry()
    assert asset_registry is not None
    num_assets = len(asset_registry.components)
    print(f"Number of assets registered: {num_assets}")
    assert num_assets > 0
    num_background_assets = len(asset_registry.get_assets_by_tag("background"))
    print(f"Number of background assets registered: {num_background_assets}")
    assert num_background_assets > 0
    num_assets = len(asset_registry.get_assets_by_tag("object"))
    print(f"Number of pick up object assets registered: {num_assets}")
    assert num_assets > 0
    return True


def test_default_assets_registered():
    # Basic test that just adds all our pick-up objects to the scene and checks that nothing crashes.
    result = run_simulation_app_function_in_separate_process(
        _test_default_assets_registered,
    )
    assert result, "Test failed"


def _test_all_assets_in_registry(simulation_app):
    # Import the necessary classes.
    from isaaclab_tasks.utils import parse_env_cfg

    from isaac_arena.assets.registry import AssetRegistry
    from isaac_arena.embodiments.franka.franka import FrankaEmbodiment
    from isaac_arena.environments.compile_env import ArenaEnvBuilder
    from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
    from isaac_arena.scene.pick_and_place_scene import PickAndPlaceScene, RigidObjectCfg
    from isaac_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaac_arena.utils.configclass import combine_configclass_instances, make_configclass

    # Base Environment
    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("packing_table_pick_and_place")()
    asset = asset_registry.get_asset_by_name("cracker_box")()
    isaac_arena_environment = IsaacArenaEnvironment(
        name="kitchen_pick_and_place",
        embodiment=FrankaEmbodiment(),
        scene=PickAndPlaceScene(background, asset),
        task=PickAndPlaceTask(),
        teleop_device=None,
    )

    # Compile the environment.
    args_parser = get_isaac_arena_cli_parser()
    args_cli = args_parser.parse_args([])

    builder = ArenaEnvBuilder(isaac_arena_environment, args_cli)
    base_cfg = builder.compose_manager_cfg()

    # Get the position of the pick-up object.
    object_position = asset.get_object_cfg().init_state.pos

    # NOTE(alexmillane): We're doing backflips here to add an object. This indicates there
    # is something suboptimal with our interfaces.

    # Go through all the pick-up objects and make a configclass containing all of them.
    fields = []
    for idx, asset_cls in enumerate(asset_registry.get_assets_by_tag("object")):
        asset = asset_cls()
        asset.set_prim_path("{ENV_REGEX_NS}/new_object_" + str(idx))
        pose = Pose(
            position_xyz=(
                object_position[0] + (idx + 1) * OBJECT_SEPARATION,
                object_position[1],
                object_position[2],
            ),
            rotation_wxyz=(1, 0, 0, 0),
        )
        asset.set_initial_pose(pose)
        object_cfg = asset.get_object_cfg()
        fields.append(
            (f"object_{idx}", RigidObjectCfg, object_cfg),
        )
    AdditionalObjectCfg = make_configclass("AdditionalObjectCfg", fields)
    additional_object_cfg = AdditionalObjectCfg()

    # Add the new objects to the scene.
    new_scene_cfg = combine_configclass_instances("SceneCfg", base_cfg.scene, additional_object_cfg)
    base_cfg.scene = new_scene_cfg
    print(base_cfg)

    # Run some zero actions.
    entry_point = "isaaclab.envs:ManagerBasedRLEnv"
    gym.register(
        id=isaac_arena_environment.name,
        entry_point=entry_point,
        kwargs={"env_cfg_entry_point": base_cfg},
        disable_env_checker=True,
    )
    env_cfg = parse_env_cfg(
        isaac_arena_environment.name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(isaac_arena_environment.name, cfg=env_cfg)
    env.reset()
    for _ in tqdm.tqdm(range(NUM_STEPS)):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

    # Close the environment.
    env.close()

    return True


def test_all_assets_in_registry():
    # Basic test that just adds all our pick-up objects to the scene and checks that nothing crashes.
    result = run_simulation_app_function_in_separate_process(
        _test_all_assets_in_registry,
        headless=HEADLESS,
    )
    assert result, "Test failed"
