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
# object set still produces a heterogeneous clone plan.
NUM_ENVS_WITH_CAMERAS = 4
NUM_STEPS_WITH_CAMERAS = 2
OBJECT_SET_1_PRIM_PATH = "/World/envs/env_.*/ObjectSet_1"
OBJECT_SET_2_PRIM_PATH = "/World/envs/env_.*/ObjectSet_2"


def _test_empty_object_set(simulation_app):
    from isaaclab_arena.assets.object_set import RigidObjectSet

    with pytest.raises(AssertionError, match="at least one member"):
        RigidObjectSet(name="empty_object_set", objects=[])

    return True


def test_empty_object_set():
    assert run_function_with_persistent_simulation_app(_test_empty_object_set, headless=HEADLESS)


def _test_articulation_object_set(simulation_app):
    from isaaclab.sim import CuboidCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.assets.object_type import ObjectType

    articulation = Object(
        name="articulation", object_type=ObjectType.ARTICULATION, spawn_cfg=CuboidCfg(size=(1.0, 1.0, 1.0))
    )
    with pytest.raises(AssertionError, match="rigid Object members only"):
        RigidObjectSet(name="articulation_object_set", objects=[articulation])

    return True


def test_articulation_object_set():
    assert run_function_with_persistent_simulation_app(_test_articulation_object_set, headless=HEADLESS)


def _test_object_set_copies_native_members_and_uses_assigned_geometry(simulation_app):
    from isaaclab.sim import CuboidCfg, MultiAssetSpawnerCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.scene.object_variant_assignment import assign_object_variants
    from isaaclab_arena.utils.pose import Pose

    sizes = ((1.0, 2.0, 3.0), (2.0, 3.0, 4.0))
    members = [
        Object(name=f"box_{index}", object_type=ObjectType.RIGID, spawn_cfg=CuboidCfg(size=size))
        for index, size in enumerate(sizes)
    ]
    members[0].set_initial_pose(Pose(position_xyz=(1.0, 2.0, 3.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    object_set = RigidObjectSet(name="boxes", objects=members)
    assert isinstance(object_set.spawn_cfg, MultiAssetSpawnerCfg)
    assert len(object_set.spawn_cfg.assets_cfg) == 2
    assert not object_set.spawn_cfg.random_choice
    assert object_set.get_initial_pose() is None
    members[0].spawn_cfg.size = (7.0, 8.0, 9.0)
    assert object_set.spawn_cfg.assets_cfg[0].size == sizes[0]

    assign_object_variants([object_set], num_envs=5)
    assert object_set.asset_indices_by_env == (0, 1, 0, 1, 0)
    expected_sizes = torch.tensor([sizes[index] for index in object_set.asset_indices_by_env])
    torch.testing.assert_close(object_set.get_bounding_box_per_env(5).size, expected_sizes)
    assert len(object_set.spawn_cfg.assets_cfg) == 2
    with pytest.raises(AssertionError, match="per-environment bounding boxes"):
        object_set.get_bounding_box()
    with pytest.raises(AssertionError, match="one native spawn configuration per member"):
        RigidObjectSet(name="nested", objects=[object_set])

    singleton = RigidObjectSet(name="singleton", objects=[members[1]])
    assert isinstance(singleton.spawn_cfg, CuboidCfg)
    assert not singleton.has_multiple_assets
    torch.testing.assert_close(singleton.get_bounding_box_per_env(3).size, torch.tensor([sizes[1]] * 3))
    assert singleton.get_contact_sensor_prim_path() == singleton.get_prim_path()

    return True


def test_object_set_copies_native_members_and_uses_assigned_geometry():
    assert run_function_with_persistent_simulation_app(
        _test_object_set_copies_native_members_and_uses_assigned_geometry, headless=HEADLESS
    )


def _test_single_object_in_one_object_set(simulation_app):
    from isaaclab.sim.utils.stage import get_current_stage

    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.object_set import RigidObjectSet
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
    obj_set = RigidObjectSet(name="single_object_set", objects=[cracker_box], prim_path=OBJECT_SET_1_PRIM_PATH)
    obj_set.set_initial_pose(Pose(position_xyz=(0.1, 0.0, 0.1), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    scene = Scene(assets=[background, obj_set, destination_location])
    task = PickAndPlaceTask(
        pick_up_object=obj_set, destination_location=destination_location, background_scene=background
    )
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="single_object_set_test",
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
                prim_path=OBJECT_SET_1_PRIM_PATH.replace(".*", str(i)), stage=get_current_stage()
            )
            assert path is not None, "Path is None"
            assert "cracker_box.usd" in path, "Path does not contain cracker_box.usd"
            assert obj_set.get_initial_pose() is not None, "Initial pose is None"

        assert env.unwrapped.scene[obj_set.name].data.root_pose_w is not None, "Root pose is None"
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


def _test_multi_objects_in_one_object_set(simulation_app):
    from isaaclab.sim.utils.stage import get_current_stage

    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.object_set import RigidObjectSet
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
    obj_set = RigidObjectSet(
        name="multi_object_sets", objects=[cracker_box, sugar_box], prim_path=OBJECT_SET_2_PRIM_PATH
    )
    scene = Scene(assets=[background, obj_set, destination_location])
    task = PickAndPlaceTask(
        pick_up_object=obj_set, destination_location=destination_location, background_scene=background
    )
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="multi_objects_in_one_object_set_test",
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

    assert env.unwrapped.scene[obj_set.name].data.root_pose_w is not None, "Root pose is None"
    assert (
        env.unwrapped.scene.sensors[task.contact_sensor_name].data.force_matrix_w is not None
    ), "Contact sensor data is None"

    # replace * in OBJECT_SET_PRIM_PATH with env_index
    object_paths = []
    try:
        for i in range(NUM_ENVS):

            path = get_asset_usd_path_from_prim_path(
                prim_path=OBJECT_SET_2_PRIM_PATH.replace(".*", str(i)), stage=get_current_stage()
            )
            assert path is not None, "Path is None"
            object_paths.append(path)
        assert len(object_paths) == NUM_ENVS, "Object_paths length is not equal to NUM_ENVS"
        expected_paths = [
            obj_set.spawn_cfg.assets_cfg[variant_index].usd_path for variant_index in obj_set.asset_indices_by_env
        ]
        # Native retrieval can change directories; compare the final prepared filenames.
        assert [os.path.basename(path) for path in object_paths] == [os.path.basename(path) for path in expected_paths]
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()
    return True


def _test_multi_object_sets(simulation_app):
    from isaaclab.sim.utils.stage import get_current_stage

    from isaaclab_arena.assets.object_set import RigidObjectSet
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

    obj_set_1 = RigidObjectSet(
        name="multi_object_sets_1", objects=[cracker_box, sugar_box], prim_path=OBJECT_SET_1_PRIM_PATH
    )
    obj_set_2 = RigidObjectSet(
        name="multi_object_sets_2", objects=[sugar_box, mustard_bottle], prim_path=OBJECT_SET_2_PRIM_PATH
    )
    scene = Scene(assets=[background, obj_set_1, obj_set_2])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="multi_object_sets_test",
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
                prim_path=OBJECT_SET_1_PRIM_PATH.replace(".*", str(i)), stage=get_current_stage()
            )
            path_2 = get_asset_usd_path_from_prim_path(
                prim_path=OBJECT_SET_2_PRIM_PATH.replace(".*", str(i)), stage=get_current_stage()
            )
            object_1_paths.append(path_1)
            object_2_paths.append(path_2)
            assert path_1 is not None, (
                "Path_1 from Prim Path " + OBJECT_SET_1_PRIM_PATH.replace(".*", str(i)) + " is None"
            )
            assert path_2 is not None, (
                "Path_2 from Prim Path " + OBJECT_SET_2_PRIM_PATH.replace(".*", str(i)) + " is None"
            )
        assert len(object_1_paths) == NUM_ENVS, "Object_1_paths length is not equal to NUM_ENVS"
        assert len(object_2_paths) == NUM_ENVS, "Object_2_paths length is not equal to NUM_ENVS"
        for object_set, spawned_paths in ((obj_set_1, object_1_paths), (obj_set_2, object_2_paths)):
            expected_paths = [
                object_set.spawn_cfg.assets_cfg[variant_index].usd_path
                for variant_index in object_set.asset_indices_by_env
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


def _test_object_set_with_robot_mounted_cameras(simulation_app) -> bool:
    """An object set clones correctly in a scene whose cameras are mounted on the robot.

    An object set spawns one USD variant per env, which puts the scene on Isaac Lab's
    heterogeneous clone-plan path: every cfg gets its own destination template instead of a
    single env-root one. DROID's cameras live under the robot, so their templates nest
    inside the robot's, and resolving them used to raise. Needs more than one env; a single
    env takes the homogeneous fast path and never builds the nested templates.
    """
    from isaaclab_arena.assets.object_set import RigidObjectSet
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

    object_set = RigidObjectSet(name="object_set", objects=[sweet_potato, jug])
    object_set.set_initial_pose(
        Pose(position_xyz=(0.0758066475391388, -0.5088448524475098, 0.5), rotation_xyzw=(0, 0, 0, 1))
    )

    scene = Scene(assets=[background, object_set])

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="object_set_with_cameras_test",
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


def test_single_object_in_one_object_set():
    result = run_function_with_persistent_simulation_app(
        _test_single_object_in_one_object_set,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_single_object_in_one_object_set.__name__} failed"


def test_multi_objects_in_one_object_set():
    result = run_function_with_persistent_simulation_app(
        _test_multi_objects_in_one_object_set,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_multi_objects_in_one_object_set.__name__} failed"


def test_multi_object_sets():
    result = run_function_with_persistent_simulation_app(
        _test_multi_object_sets,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_multi_object_sets.__name__} failed"


@pytest.mark.with_cameras
def test_object_set_with_robot_mounted_cameras():
    result = run_function_with_persistent_simulation_app(
        _test_object_set_with_robot_mounted_cameras,
        headless=HEADLESS,
        enable_cameras=True,
    )
    assert result, f"Test {_test_object_set_with_robot_mounted_cameras.__name__} failed"
