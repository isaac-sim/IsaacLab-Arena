# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True
NUM_ENVS = 3
OBJECT_SET_1_PRIM_PATH = "/World/envs/env_.*/ObjectSet_1"
OBJECT_SET_2_PRIM_PATH = "/World/envs/env_.*/ObjectSet_2"
OBJECT_SET_JUG_PRIM_PATH = "/World/envs/env_.*/ObjectSet_Jug"
OBJECT_SET_BOTTLES_PRIM_PATH = "/World/envs/env_.*/ObjectSet_Bottles"


def _build_and_reset_env(simulation_app, scene_assets, env_name="object_set_test", task=None):
    """Build arena env with given scene and optional task, then reset. Returns env (caller must close)."""
    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    asset_registry = AssetRegistry()
    embodiment = asset_registry.get_asset_by_name("franka")()
    scene = Scene(assets=scene_assets)
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name=env_name,
        embodiment=embodiment,
        scene=scene,
        task=task,
        teleop_device=None,
    )
    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.num_envs = NUM_ENVS
    args_cli.headless = HEADLESS
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env = env_builder.make_registered()
    env.reset()
    return env


def _run_pick_and_place_object_set_test(
    simulation_app,
    obj_set,
    object_set_prim_path,
    path_contains,
    initial_pose=None,
):
    """Build env with one object set and PickAndPlaceTask, run common assertions, close. path_contains: str or list[str] of length NUM_ENVS."""
    from isaacsim.core.utils.stage import get_current_stage

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.usd_helpers import get_asset_usd_path_from_prim_path

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("kitchen")()
    destination_location = ObjectReference(
        name="destination_location",
        prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
        parent_asset=background,
    )
    if initial_pose is not None:
        obj_set.set_initial_pose(initial_pose)
    scene_assets = [background, obj_set]
    task = PickAndPlaceTask(
        pick_up_object=obj_set,
        destination_location=destination_location,
        background_scene=background,
    )
    env = _build_and_reset_env(
        simulation_app,
        scene_assets,
        env_name="pick_and_place_object_set_test",
        task=task,
    )
    try:
        if isinstance(path_contains, str):
            path_contains = [path_contains] * NUM_ENVS
        for i in range(NUM_ENVS):
            path = get_asset_usd_path_from_prim_path(
                prim_path=object_set_prim_path.replace(".*", str(i)),
                stage=get_current_stage(),
            )
            assert path is not None, "Path is None"
            assert path_contains[i] in path, f"Path does not contain {path_contains[i]!r}: {path}"
        if initial_pose is not None:
            assert obj_set.get_initial_pose() is not None, "Initial pose is None"
        assert env.scene[obj_set.name].data.root_pose_w is not None, "Root pose is None"
        assert (
            env.scene.sensors["pick_up_object_contact_sensor"].data.force_matrix_w is not None
        ), "Contact sensor data is None"
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        env.close()


def _test_empty_object_set(simulation_app):
    from isaaclab_arena.assets.object_set import RigidObjectSet

    try:
        RigidObjectSet(name="empty_object_set", objects=[])
    except Exception:
        return True
    return False


def _test_articulation_object_set(simulation_app):
    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.assets.object_set import RigidObjectSet

    asset_registry = AssetRegistry()
    microwave = asset_registry.get_asset_by_name("microwave")()
    try:
        RigidObjectSet(name="articulation_object_set", objects=[microwave])
    except Exception:
        return True
    return False


def _test_single_object_in_one_object_set(simulation_app):
    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.utils.pose import Pose

    asset_registry = AssetRegistry()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()
    obj_set = RigidObjectSet(
        name="single_object_set", objects=[cracker_box, cracker_box], prim_path=OBJECT_SET_1_PRIM_PATH
    )
    return _run_pick_and_place_object_set_test(
        simulation_app,
        obj_set,
        OBJECT_SET_1_PRIM_PATH,
        path_contains="cracker_box.usd",
        initial_pose=Pose(position_xyz=(0.1, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)),
    )


def _test_multi_objects_in_one_object_set(simulation_app):
    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.assets.object_set import RigidObjectSet

    asset_registry = AssetRegistry()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()
    sugar_box = asset_registry.get_asset_by_name("sugar_box")()
    obj_set = RigidObjectSet(
        name="multi_object_sets", objects=[cracker_box, sugar_box], prim_path=OBJECT_SET_2_PRIM_PATH
    )
    path_contains = ["cracker_box.usd" if i % 2 == 0 else "sugar_box.usd" for i in range(NUM_ENVS)]
    return _run_pick_and_place_object_set_test(
        simulation_app,
        obj_set,
        OBJECT_SET_2_PRIM_PATH,
        path_contains=path_contains,
    )


def _test_multi_object_sets(simulation_app):
    from isaacsim.core.utils.stage import get_current_stage

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.utils.usd_helpers import get_asset_usd_path_from_prim_path

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("packing_table")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()
    sugar_box = asset_registry.get_asset_by_name("sugar_box")()
    mustard_bottle = asset_registry.get_asset_by_name("mustard_bottle")()

    obj_set_1 = RigidObjectSet(
        name="multi_object_sets_1", objects=[cracker_box, sugar_box], prim_path=OBJECT_SET_1_PRIM_PATH
    )
    obj_set_2 = RigidObjectSet(
        name="multi_object_sets_2", objects=[sugar_box, mustard_bottle], prim_path=OBJECT_SET_2_PRIM_PATH
    )
    env = _build_and_reset_env(
        simulation_app,
        [background, obj_set_1, obj_set_2],
        env_name="multi_object_sets_test",
        task=None,
    )
    try:
        for i in range(NUM_ENVS):
            path_1 = get_asset_usd_path_from_prim_path(
                prim_path=OBJECT_SET_1_PRIM_PATH.replace(".*", str(i)), stage=get_current_stage()
            )
            path_2 = get_asset_usd_path_from_prim_path(
                prim_path=OBJECT_SET_2_PRIM_PATH.replace(".*", str(i)), stage=get_current_stage()
            )
            assert path_1 is not None, (
                "Path_1 from Prim Path " + OBJECT_SET_1_PRIM_PATH.replace(".*", str(i)) + " is None"
            )
            assert path_2 is not None, (
                "Path_2 from Prim Path " + OBJECT_SET_2_PRIM_PATH.replace(".*", str(i)) + " is None"
            )
            if i % 2 == 0:
                assert "cracker_box.usd" in path_1, "Path_1 does not contain cracker_box.usd for env index " + str(i)
                assert "sugar_box.usd" in path_2, "Path_2 does not contain sugar_box.usd for env index " + str(i)
            else:
                assert "sugar_box.usd" in path_1, "Path_1 does not contain sugar_box.usd for env index " + str(i)
                assert (
                    "mustard_bottle.usd" in path_2
                ), "Path_2 does not contain mustard_bottle.usd for env index " + str(i)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        env.close()


def _test_object_set_with_jug(simulation_app):
    """Test object set with Jug asset (depth-1 rigid body); exercises cache pipeline and contact sensor."""
    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.utils.pose import Pose

    asset_registry = AssetRegistry()
    jug = asset_registry.get_asset_by_name("jug")()
    obj_set = RigidObjectSet(
        name="ObjectSet_Jug",
        objects=[jug, jug],
        prim_path=OBJECT_SET_JUG_PRIM_PATH,
    )
    return _run_pick_and_place_object_set_test(
        simulation_app,
        obj_set,
        OBJECT_SET_JUG_PRIM_PATH,
        path_contains="jug",
        initial_pose=Pose(position_xyz=(0.1, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)),
    )


def _test_object_set_with_ranch_and_bbq_bottles(simulation_app):
    """Test object set with ranch_dressing_bottle and bbq_sauce_bottle."""
    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.assets.object_set import RigidObjectSet

    asset_registry = AssetRegistry()
    ranch_dressing_bottle = asset_registry.get_asset_by_name("ranch_dressing_bottle")()
    bbq_sauce_bottle = asset_registry.get_asset_by_name("bbq_sauce_bottle")()
    obj_set = RigidObjectSet(
        name="ObjectSet_Bottles",
        objects=[ranch_dressing_bottle, bbq_sauce_bottle],
        prim_path=OBJECT_SET_BOTTLES_PRIM_PATH,
    )
    path_contains = ["ranch_dressing" if i % 2 == 0 else "bbq_sauce_bottle" for i in range(NUM_ENVS)]
    return _run_pick_and_place_object_set_test(
        simulation_app,
        obj_set,
        OBJECT_SET_BOTTLES_PRIM_PATH,
        path_contains=path_contains,
    )


def test_empty_object_set():
    result = run_simulation_app_function(
        _test_empty_object_set,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_empty_object_set.__name__} failed"


def test_articulation_object_set():
    result = run_simulation_app_function(
        _test_articulation_object_set,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_articulation_object_set.__name__} failed"


def test_single_object_in_one_object_set():
    result = run_simulation_app_function(
        _test_single_object_in_one_object_set,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_single_object_in_one_object_set.__name__} failed"


def test_multi_objects_in_one_object_set():
    result = run_simulation_app_function(
        _test_multi_objects_in_one_object_set,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_multi_objects_in_one_object_set.__name__} failed"


def test_multi_object_sets():
    result = run_simulation_app_function(
        _test_multi_object_sets,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_multi_object_sets.__name__} failed"


def test_object_set_with_jug():
    result = run_simulation_app_function(
        _test_object_set_with_jug,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_object_set_with_jug.__name__} failed"


def test_object_set_with_ranch_and_bbq_bottles():
    result = run_simulation_app_function(
        _test_object_set_with_ranch_and_bbq_bottles,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_object_set_with_ranch_and_bbq_bottles.__name__} failed"


if __name__ == "__main__":
    test_empty_object_set()
    test_articulation_object_set()
    test_single_object_in_one_object_set()
    test_multi_objects_in_one_object_set()
    test_multi_object_sets()
    test_object_set_with_jug()
    test_object_set_with_ranch_and_bbq_bottles()
