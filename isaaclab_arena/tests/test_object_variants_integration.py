# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import tqdm
import traceback

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True
NUM_ENVS = 10
# Kept lower than NUM_ENVS because each env renders three 720p cameras. Must stay > 1 so the
# object variants still produce a heterogeneous clone plan.
NUM_ENVS_WITH_CAMERAS = 4
NUM_STEPS_WITH_CAMERAS = 2
VARIANT_OBJECT_1_PRIM_PATH = "/World/envs/env_.*/ObjectVariants_1"
VARIANT_OBJECT_2_PRIM_PATH = "/World/envs/env_.*/ObjectVariants_2"


def _test_single_variant_object(simulation_app):
    from isaaclab.sim.utils.stage import get_current_stage

    from isaaclab_arena.assets.object_choice import ObjectChoice
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.utils.usd.helpers import get_asset_usd_path_from_prim_path

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("kitchen")()
    embodiment = asset_registry.get_asset_by_name("franka_ik")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()
    destination_location = ObjectReference(
        name="destination_location",
        prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
        parent_asset=background,
    )
    varied_object = ObjectChoice(
        name="single_object_variants", objects=[cracker_box], prim_path=VARIANT_OBJECT_1_PRIM_PATH
    )
    varied_object.set_initial_pose(Pose(position_xyz=(0.1, 0.0, 0.1), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    scene = Scene(assets=[background, varied_object, destination_location])
    task = PickAndPlaceTask(
        pick_up_object=varied_object, destination_location=destination_location, background_scene=background
    )
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="single_object_variants_test",
        embodiment=embodiment,
        scene=scene,
        task=task,
        teleop_device=None,
    )
    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.num_envs = NUM_ENVS
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, arena_env_builder_cfg_from_argparse(args_cli))
    env = env_builder.make_registered()
    env.reset()

    try:
        for i in range(NUM_ENVS):
            # Construct the actual prim path for this environment
            path = get_asset_usd_path_from_prim_path(
                prim_path=VARIANT_OBJECT_1_PRIM_PATH.replace(".*", str(i)), stage=get_current_stage()
            )
            assert path is not None, "Path is None"
            assert "cracker_box.usd" in path, "Path does not contain cracker_box.usd"
            assert varied_object.get_initial_pose() is not None, "Initial pose is None"

        assert env.unwrapped.scene[varied_object.name].data.root_pose_w is not None, "Root pose is None"
        assert (
            env.unwrapped.scene.sensors[task.contact_sensor_name].data.force_matrix_w is not None
        ), "Contact sensor data is None"
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()
    return True


def _test_object_variants_across_environments(simulation_app):
    from isaaclab.sim.utils.stage import get_current_stage

    from isaaclab_arena.assets.object_choice import ObjectChoice
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.usd.helpers import get_asset_usd_path_from_prim_path

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("kitchen")()
    embodiment = asset_registry.get_asset_by_name("franka_ik")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()
    sugar_box = asset_registry.get_asset_by_name("sugar_box")()
    destination_location = ObjectReference(
        name="destination_location",
        prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
        parent_asset=background,
    )
    varied_object = ObjectChoice(
        name="multiple_objects_with_variants",
        objects=[cracker_box, sugar_box],
        prim_path=VARIANT_OBJECT_2_PRIM_PATH,
    )
    scene = Scene(assets=[background, varied_object, destination_location])
    task = PickAndPlaceTask(
        pick_up_object=varied_object, destination_location=destination_location, background_scene=background
    )
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="object_variants_across_environments_test",
        embodiment=embodiment,
        scene=scene,
        task=task,
        teleop_device=None,
    )
    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.num_envs = NUM_ENVS
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, arena_env_builder_cfg_from_argparse(args_cli))
    env = env_builder.make_registered()
    env.reset()

    assert env.unwrapped.scene[varied_object.name].data.root_pose_w is not None, "Root pose is None"
    assert (
        env.unwrapped.scene.sensors[task.contact_sensor_name].data.force_matrix_w is not None
    ), "Contact sensor data is None"

    # replace * in VARIANT_OBJECT_PRIM_PATH with env_index
    object_paths = []
    try:
        for i in range(NUM_ENVS):

            path = get_asset_usd_path_from_prim_path(
                prim_path=VARIANT_OBJECT_2_PRIM_PATH.replace(".*", str(i)), stage=get_current_stage()
            )
            assert path is not None, "Path is None"
            object_paths.append(path)
        assert len(object_paths) == NUM_ENVS, "Spawned object count does not match NUM_ENVS"
        expected_paths = [
            varied_object.spawn_cfg.assets_cfg[variant_index].usd_path
            for variant_index in varied_object.variant_indices_by_env
        ]
        # Native asset retrieval can change the directory; prepared USD filenames remain stable.
        assert [os.path.basename(path) for path in object_paths] == [os.path.basename(path) for path in expected_paths]
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()
    return True


def _test_multiple_objects_with_variants(simulation_app):
    from isaaclab.sim.utils.stage import get_current_stage

    from isaaclab_arena.assets.object_choice import ObjectChoice
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.usd.helpers import get_asset_usd_path_from_prim_path

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("packing_table")()
    embodiment = asset_registry.get_asset_by_name("franka_ik")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()
    sugar_box = asset_registry.get_asset_by_name("sugar_box")()
    mustard_bottle = asset_registry.get_asset_by_name("mustard_bottle")()

    first_object = ObjectChoice(
        name="multiple_objects_with_variants_1",
        objects=[cracker_box, sugar_box],
        prim_path=VARIANT_OBJECT_1_PRIM_PATH,
    )
    second_object = ObjectChoice(
        name="multiple_objects_with_variants_2",
        objects=[sugar_box, mustard_bottle],
        prim_path=VARIANT_OBJECT_2_PRIM_PATH,
    )
    scene = Scene(assets=[background, first_object, second_object])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="multiple_objects_with_variants_test",
        embodiment=embodiment,
        scene=scene,
    )
    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.num_envs = NUM_ENVS
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, arena_env_builder_cfg_from_argparse(args_cli))
    env = env_builder.make_registered()
    env.reset()

    try:
        object_1_paths = []
        object_2_paths = []
        for i in range(NUM_ENVS):

            path_1 = get_asset_usd_path_from_prim_path(
                prim_path=VARIANT_OBJECT_1_PRIM_PATH.replace(".*", str(i)), stage=get_current_stage()
            )
            path_2 = get_asset_usd_path_from_prim_path(
                prim_path=VARIANT_OBJECT_2_PRIM_PATH.replace(".*", str(i)), stage=get_current_stage()
            )
            object_1_paths.append(path_1)
            object_2_paths.append(path_2)
            assert path_1 is not None, (
                "Path_1 from Prim Path " + VARIANT_OBJECT_1_PRIM_PATH.replace(".*", str(i)) + " is None"
            )
            assert path_2 is not None, (
                "Path_2 from Prim Path " + VARIANT_OBJECT_2_PRIM_PATH.replace(".*", str(i)) + " is None"
            )
        assert len(object_1_paths) == NUM_ENVS, "First object count does not match NUM_ENVS"
        assert len(object_2_paths) == NUM_ENVS, "Second object count does not match NUM_ENVS"
        for varied_object, spawned_paths in ((first_object, object_1_paths), (second_object, object_2_paths)):
            expected_paths = [
                varied_object.spawn_cfg.assets_cfg[variant_index].usd_path
                for variant_index in varied_object.variant_indices_by_env
            ]
            assert [os.path.basename(path) for path in spawned_paths] == [
                os.path.basename(path) for path in expected_paths
            ]
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()
    return True


def _test_object_variants_with_robot_mounted_cameras(simulation_app) -> bool:
    """Object variants clone correctly in a scene whose cameras are mounted on the robot.

    An object spawns one USD variant per environment, which puts the scene on Isaac Lab's
    heterogeneous clone-plan path: every cfg gets its own destination template instead of a
    single env-root one. DROID's cameras live under the robot, so their templates nest
    inside the robot's, and resolving them used to raise. Needs more than one env; a single
    env takes the homogeneous fast path and never builds the nested templates.
    """
    from isaaclab_arena.assets.object_choice import ObjectChoice
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args(["--enable_cameras"])
    args_cli.num_envs = NUM_ENVS_WITH_CAMERAS

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("packing_table")()
    sweet_potato = asset_registry.get_asset_by_name("sweet_potato")()
    jug = asset_registry.get_asset_by_name("jug")()

    object_variants = ObjectChoice(name="object_variants", objects=[sweet_potato, jug])
    object_variants.set_initial_pose(
        Pose(position_xyz=(0.0758066475391388, -0.5088448524475098, 0.5), rotation_xyzw=(0, 0, 0, 1))
    )

    scene = Scene(assets=[background, object_variants])

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="object_variants_with_cameras_test",
        embodiment=DroidAbsoluteJointPositionEmbodiment(enable_cameras=True),
        scene=scene,
    )

    # Scene construction is what used to raise, so reaching the first observation is the
    # regression signal.
    builder = ArenaEnvBuilder(isaaclab_arena_environment, arena_env_builder_cfg_from_argparse(args_cli))
    env = builder.make_registered()
    env.reset()

    # The observation dict is the env's own buffer, so it must be read before env.close().
    try:
        for _ in tqdm.tqdm(range(NUM_STEPS_WITH_CAMERAS)):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                obs, _, _, _, _ = env.step(actions)

                assert "camera_obs" in obs, f"No camera observation group; got groups {sorted(obs)}"
                camera_obs = obs["camera_obs"]
                # Every camera nested under the robot must survive cloning: the two external
                # ones under panda_link0 and the wrist one under the gripper.
                for camera_key in ("external_camera_rgb", "external_camera_2_rgb", "wrist_camera_rgb"):
                    assert camera_key in camera_obs, f"Missing {camera_key!r}; got {sorted(camera_obs)}"
                    images = camera_obs[camera_key]
                    num_envs = images.shape[0]
                    assert num_envs == NUM_ENVS_WITH_CAMERAS, f"{camera_key} has {num_envs} envs"
                    assert images.shape[3] == 3, f"{camera_key} rgb observation does not have three channels"
                    assert images.any(), f"{camera_key} observation contains only 0s"
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()

    return True


def test_single_variant_object():
    result = run_function_with_persistent_simulation_app(
        _test_single_variant_object,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_single_variant_object.__name__} failed"


def test_object_variants_across_environments():
    result = run_function_with_persistent_simulation_app(
        _test_object_variants_across_environments,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_object_variants_across_environments.__name__} failed"


def test_multiple_objects_with_variants():
    result = run_function_with_persistent_simulation_app(
        _test_multiple_objects_with_variants,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_multiple_objects_with_variants.__name__} failed"


@pytest.mark.with_cameras
def test_object_variants_with_robot_mounted_cameras():
    result = run_function_with_persistent_simulation_app(
        _test_object_variants_with_robot_mounted_cameras,
        headless=HEADLESS,
        enable_cameras=True,
    )
    assert result, f"Test {_test_object_variants_with_robot_mounted_cameras.__name__} failed"
