# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import tqdm

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function
from isaaclab_arena.utils.pose import Pose

NUM_STEPS = 2
HEADLESS = True
OBJECT_SEPARATION = 0.5


def _test_default_assets_registered(simulation_app):
    from isaaclab_arena.assets.registries import AssetRegistry

    asset_registry = AssetRegistry()
    assert asset_registry is not None
    num_background_assets = len(asset_registry.get_assets_by_tag("background"))
    print(f"Number of background assets registered: {num_background_assets}")
    assert num_background_assets > 0
    num_assets = len(asset_registry.get_assets_by_tag("object"))
    print(f"Number of pick up object assets registered: {num_assets}")
    assert num_assets > 0
    num_ground_plane_assets = len(asset_registry.get_assets_by_tag("ground_plane"))
    print(f"Number of ground plane assets registered: {num_ground_plane_assets}")
    assert num_ground_plane_assets > 0
    num_light_assets = len(asset_registry.get_assets_by_tag("light"))
    print(f"Number of light assets registered: {num_light_assets}")
    assert num_light_assets > 0
    return True


def test_default_assets_registered():
    # Basic test that just adds all our pick-up objects to the scene and checks that nothing crashes.
    result = run_simulation_app_function(
        _test_default_assets_registered,
    )
    assert result, "Test failed"


def _test_all_assets_in_registry(simulation_app):
    # Import the necessary classes.

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    # Base Environment
    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("packing_table")()
    asset = asset_registry.get_asset_by_name("cracker_box")()

    first_position = (0.5, 0.0, 0.0)
    objects_in_registry_names: list[str] = []
    objects_in_registry: list[Object] = []
    for idx, asset_cls in enumerate(asset_registry.get_assets_by_tag("object")):
        asset = asset_cls()
        # Set their pose
        pose = Pose(
            position_xyz=(
                first_position[0] + (idx + 1) * OBJECT_SEPARATION,
                first_position[1],
                first_position[2],
            ),
            rotation_xyzw=(0, 0, 0, 1),
        )
        asset.set_initial_pose(pose)
        objects_in_registry.append(asset)
        objects_in_registry_names.append(asset.name)
    # Add lights
    for asset_cls in asset_registry.get_assets_by_tag("light"):
        asset = asset_cls()
        objects_in_registry.append(asset)
        objects_in_registry_names.append(asset.name)
    # Add ground plane
    ground_plane = asset_registry.get_asset_by_name("ground_plane")()
    objects_in_registry.append(ground_plane)
    objects_in_registry_names.append(ground_plane.name)
    assert len(objects_in_registry) > 0

    scene = Scene(assets=[background, *objects_in_registry])

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="dummy_task",
        embodiment=FrankaIKEmbodiment(),
        scene=scene,
    )

    # Compile the environment.
    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args([])

    builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env = builder.make_registered()
    env.reset()

    # Run
    for _ in tqdm.tqdm(range(NUM_STEPS)):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

    # Check all the assets made it into the scene.
    for asset_name in objects_in_registry_names:
        assert asset_name in env.unwrapped.scene.keys(), f"Asset {asset_name} not found in scene"

    # Check all the assets have the correct pose.
    for asset_name in objects_in_registry_names:
        assert asset_name in env.unwrapped.scene.keys(), f"Asset {asset_name} not found in scene"

    env.close()

    return True


def test_all_assets_in_registry():
    # Basic test that just adds all our pick-up objects to the scene and checks that nothing crashes.
    result = run_simulation_app_function(
        _test_all_assets_in_registry,
        headless=HEADLESS,
    )
    assert result, "Test failed"


def _test_hdr_images_registered(simulation_app):
    from isaaclab_arena.assets.hdr_image import HDRImage
    from isaaclab_arena.assets.registries import HDRImageRegistry

    hdr_registry = HDRImageRegistry()
    all_keys = hdr_registry.get_all_keys()
    print(f"Number of HDR images registered: {len(all_keys)}")
    assert len(all_keys) > 0, "No HDR images registered"

    for key in all_keys:
        hdr_cls = hdr_registry.get_hdr_by_name(key)
        assert hdr_cls is not None, f"HDR image class for '{key}' is None"
        # Instantiate and check attributes
        hdr = hdr_cls()
        assert isinstance(hdr, HDRImage), f"Expected HDRImage instance, got {type(hdr)}"
        assert hdr.name == key, f"HDR name mismatch: expected '{key}', got '{hdr.name}'"
        assert hdr.texture_file, f"HDR '{key}' has empty texture_file"
        assert hdr.tags is not None, f"HDR '{key}' has None tags"
        print(f"  HDR '{key}': texture_file={hdr.texture_file}, tags={hdr.tags}")

    return True


def test_hdr_images_registered():
    result = run_simulation_app_function(
        _test_hdr_images_registered,
    )
    assert result, "Test failed"


def _test_hdr_image_spawn(simulation_app):
    """Spawn a DomeLight with one HDR image to verify the integration works end-to-end.

    Individual HDR image attributes (name, texture_file, tags) are already
    validated for *all* registered images by ``_test_hdr_images_registered``.
    This test only needs to confirm that the DomeLight + HDRImage mechanism
    works in a live simulation, so a single HDR is sufficient.
    """
    from isaaclab_arena.assets.registries import AssetRegistry, HDRImageRegistry
    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    asset_registry = AssetRegistry()
    hdr_registry = HDRImageRegistry()

    all_keys = hdr_registry.get_all_keys()
    assert len(all_keys) > 0, "No HDR images registered"

    # Pick the first registered HDR image as representative.
    hdr = hdr_registry.get_hdr_by_name(all_keys[0])()
    light = asset_registry.get_asset_by_name("light")()
    light.add_hdr(hdr)
    ground_plane = asset_registry.get_asset_by_name("ground_plane")()

    scene = Scene(assets=[light, ground_plane])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="hdr_spawn_test",
        embodiment=FrankaIKEmbodiment(),
        scene=scene,
    )

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args([])

    builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env = builder.make_registered()
    env.reset()

    for _ in tqdm.tqdm(range(NUM_STEPS), desc=f"HDR: {all_keys[0]}"):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

    env.close()
    print(f"  HDR '{all_keys[0]}' spawned successfully")

    return True


def test_hdr_image_spawn():
    # Test that spawns a DomeLight with an HDR image and runs the simulation.
    result = run_simulation_app_function(
        _test_hdr_image_spawn,
        headless=HEADLESS,
    )
    assert result, "Test failed"


def _test_multi_light_in_scene(simulation_app):
    from pxr import UsdLux

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.usd_helpers import get_all_prims

    asset_registry = AssetRegistry()
    light = asset_registry.get_asset_by_name("light")()
    light_duplicate = asset_registry.get_asset_by_name("light")()
    ground_plane = asset_registry.get_asset_by_name("ground_plane")()
    ground_plane_duplicate = asset_registry.get_asset_by_name("ground_plane")()
    scene = Scene(assets=[light, light_duplicate, ground_plane, ground_plane_duplicate])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="dummy_task",
        embodiment=FrankaIKEmbodiment(),
        scene=scene,
    )
    # Compile the environment.
    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args(["--num_envs", "2"])

    builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env = builder.make_registered()
    env.reset()
    for _ in tqdm.tqdm(range(NUM_STEPS)):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)
    all_prims_in_stage = get_all_prims(env.unwrapped.scene.stage)
    # Check that there is only one light in the stage
    # We dont add lights from anywhere else in this scene.
    light_prims = [prim for prim in all_prims_in_stage if prim.IsA(UsdLux.DomeLight)]
    assert len(light_prims) == 1
    env.close()
    return True


def test_multi_light_in_scene():
    result = run_simulation_app_function(
        _test_multi_light_in_scene,
        headless=HEADLESS,
    )
    assert result, "Test failed"


if __name__ == "__main__":
    test_default_assets_registered()
    test_all_assets_in_registry()
    test_hdr_images_registered()
    test_hdr_image_spawn()
    test_multi_light_in_scene()
