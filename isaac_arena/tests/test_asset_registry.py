# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

import torch
import tqdm

from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
from isaac_arena.geometry.pose import Pose
from isaac_arena.scene.asset_registry import AssetRegistry
from isaac_arena.tests.utils.subprocess import run_simulation_app_function_in_separate_process

NUM_STEPS = 2
HEADLESS = True
OBJECT_SEPARATION = 0.2


def _test_default_assets_registered(simulation_app):
    asset_registry = AssetRegistry()
    assert asset_registry is not None
    num_assets = len(asset_registry.registry)
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
    from isaac_arena.embodiments.franka import FrankaEmbodiment
    from isaac_arena.environments.compile_env import compile_gym_env, compile_manager_based_env_cfg
    from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
    from isaac_arena.scene.pick_and_place_scene import PickAndPlaceScene, RigidObjectCfg
    from isaac_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaac_arena.utils.configclass import combine_configclass_instances, make_configclass

    # Base Environment
    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("packing_table_pick_and_place")
    asset = asset_registry.get_asset_by_name("cracker_box")
    isaac_arena_environment = IsaacArenaEnvironment(
        name="kitchen_pick_and_place",
        embodiment=FrankaEmbodiment(),
        scene=PickAndPlaceScene(background, asset),
        task=PickAndPlaceTask(),
    )

    # Compile the environment.
    arena_env_cfg = compile_manager_based_env_cfg(isaac_arena_environment)

    # Get the position of the pick-up object.
    object_position = asset.get_object_cfg().init_state.pos

    # NOTE(alexmillane): We're doing backflips here to add an object. This indicates there
    # is something suboptimal with our interfaces.

    # Go through all the pick-up objects and make a configclass containing all of them.
    fields = []
    for idx, asset in enumerate(asset_registry.get_assets_by_tag("object")):
        asset.set_prim_path("{ENV_REGEX_NS}/new_object_" + str(idx))
        asset.set_name("new_object_" + str(idx))
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
    new_scene_cfg = combine_configclass_instances("SceneCfg", arena_env_cfg.scene, additional_object_cfg)
    arena_env_cfg.scene = new_scene_cfg
    print(arena_env_cfg)

    # Run some zero actions.
    args_parser = get_isaac_arena_cli_parser()
    args_cli = args_parser.parse_args([])
    env = compile_gym_env(isaac_arena_environment.name, arena_env_cfg, args_cli)
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
