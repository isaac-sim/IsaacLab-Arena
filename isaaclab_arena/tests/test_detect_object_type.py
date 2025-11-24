# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import tqdm

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True
NUM_STEPS = 10


def _test_detect_object_type(simulation_app):

    from pxr import Usd, UsdPhysics

    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.object_utils import detect_object_type

    # ObjectType.BASE
    print("Detecting ObjectType.BASE")
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/test1", "Xform")
    stage.DefinePrim("/test2", "Xform")
    assert detect_object_type(stage=stage) == ObjectType.BASE

    # ObjectType.RIGID
    print("Detecting ObjectType.RIGID")
    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/rigid_body", "Xform")
    prim.ApplyAPI(UsdPhysics.RigidBodyAPI)
    stage.DefinePrim("/rigid_body/child_1", "Xform")
    stage.DefinePrim("/rigid_body/child_2", "Xform")
    assert detect_object_type(stage=stage) == ObjectType.RIGID

    # ObjectType.ARTICULATION
    print("Detecting ObjectType.ARTICULATION")
    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/articulation_root", "Xform")
    prim.ApplyAPI(UsdPhysics.ArticulationRootAPI)
    stage.DefinePrim("/articulation_root/child_1", "Xform")
    stage.DefinePrim("/articulation_root/child_2", "Xform")
    assert detect_object_type(stage=stage) == ObjectType.ARTICULATION

    # Expect FAIL - multiple object types at the same depth
    print("Expect Fail: Detecting multiple object types at the same depth")
    with pytest.raises(ValueError) as exception_info:
        stage = Usd.Stage.CreateInMemory()
        prim = stage.DefinePrim("/rigid_body", "Xform")
        prim.ApplyAPI(UsdPhysics.RigidBodyAPI)
        prim = stage.DefinePrim("/articulation_root", "Xform")
        prim.ApplyAPI(UsdPhysics.ArticulationRootAPI)
        # Should raise an error
        detect_object_type(stage=stage)
    assert "Found multiple rigid body or articulation roots at depth" in str(exception_info.value)
    return True


def _test_detect_object_type_for_all_objects(simulation_app):
    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.assets.object_utils import detect_object_type

    asset_registry = AssetRegistry()
    for object_asset in asset_registry.get_assets_by_tag("object"):
        print(f"Automatically classifying: {object_asset.name}")
        detected_object_type = detect_object_type(usd_path=object_asset.usd_path)
        print(f"database object type: {object_asset.object_type}")
        print(f"detected object type: {detected_object_type}")
        assert detected_object_type == object_asset.object_type
    return True


def _test_auto_object_type(simulation_app):

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

    asset_registry = AssetRegistry()

    background = asset_registry.get_asset_by_name("kitchen")()
    embodiment = asset_registry.get_asset_by_name("franka")()

    try:
        # Try out an auto object.
        cracker_box = Object(
            name="cracker_box",
            prim_path="{ENV_REGEX_NS}/cracker_box",
            object_type=None,
            usd_path=asset_registry.get_asset_by_name("cracker_box")().usd_path,
        )

        microwave = Object(
            name="microwave",
            prim_path="{ENV_REGEX_NS}/microwave",
            object_type=None,
            usd_path=asset_registry.get_asset_by_name("microwave")().usd_path,
        )

        scene = Scene(assets=[background, cracker_box, microwave])
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name="auto_object_type_test",
            embodiment=embodiment,
            scene=scene,
            # NOTE(alexmillane, 2025-09-16): We use the pick and place task to ensure
            # that we can use an auto-detected ridid-object in a task.
            task=PickAndPlaceTask(cracker_box, cracker_box, background),
        )

        args_cli = get_isaaclab_arena_cli_parser().parse_args([])
        env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
        env = env_builder.make_registered()
        env.reset()

        # Run some zero actions.
        for _ in tqdm.tqdm(range(NUM_STEPS)):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)

        # Check that we detect the correct object type.
        assert cracker_box.object_type == ObjectType.RIGID
        assert microwave.object_type == ObjectType.ARTICULATION

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def test_detect_object_type():
    result = run_simulation_app_function(
        _test_detect_object_type,
        headless=HEADLESS,
    )
    assert result, "Test failed"


def test_detect_object_type_for_all_objects():
    result = run_simulation_app_function(
        _test_detect_object_type_for_all_objects,
        headless=HEADLESS,
    )
    assert result, "Test failed"


def test_auto_object_type():
    result = run_simulation_app_function(
        _test_auto_object_type,
        headless=HEADLESS,
    )
    assert result, "Test failed"


if __name__ == "__main__":
    test_detect_object_type()
    test_detect_object_type_for_all_objects()
    test_auto_object_type()
