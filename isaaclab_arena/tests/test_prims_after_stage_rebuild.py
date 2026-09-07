# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for prim transforms across an experiment-runner stage rebuild.

Checks the same rebuild as ``test_render_after_stage_rebuild`` one layer lower down. Instead of
comparing rendered pixels, it reports prims that USD places away from the origin while Fabric still
holds an identity ``omni:fabric:worldMatrix``. The render delegate reads that attribute, so a prim
left at identity is drawn at the world origin -- which is the mechanism behind the geometry that the
render test sees at the wrong pose. Failing here names the offending prim paths directly, so it is
the cheaper of the two to diagnose.

Regression test for the Isaac Lab render-after-rebuild bug:
https://github.com/isaac-sim/IsaacLab/issues/7472
"""

import numpy as np
import torch
from functools import partial

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True
# Stage builds performed in one process: the first build plus the rebuilds that follow it.
NUM_BUILDS = 2
# Steps taken after reset before transforms are compared.
NUM_STEPS = 5
# Absolute tolerance on a matrix entry, both for calling a prim origin-placed and for calling its
# Fabric matrix an identity.
TRANSFORM_TOLERANCE = 1e-4
# Minimum number of prims the comparison must actually reach, so a build that exposes no Fabric world
# matrices at all cannot pass the check vacuously.
MIN_PRIMS_COMPARED = 1


def _build_droid_env(disable_fabric: bool):
    """Build a single-env, camera-enabled DROID environment on a lit packing table.

    Args:
        disable_fabric: Whether to build on CPU with Fabric disabled.
    """
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    asset_registry = AssetRegistry()
    scene = Scene(
        assets=[
            asset_registry.get_asset_by_name("light")(),
            asset_registry.get_asset_by_name("packing_table")(),
        ]
    )
    arena_env = IsaacLabArenaEnvironment(
        name="test_prims_after_stage_rebuild",
        # Enabling cameras turns on the Fabric Scene Delegate, the render path this bug lives on.
        embodiment=DroidAbsoluteJointPositionEmbodiment(enable_cameras=True),
        scene=scene,
    )
    cli_args = ["--num_envs", "1", "--enable_cameras"]
    if disable_fabric:
        cli_args.extend(["--disable_fabric", "--device", "cpu"])
    args_cli = get_isaaclab_arena_cli_parser().parse_args(cli_args)
    builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli))
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    # Arena turns this off by default. Without it a reset can return while assets are still streaming,
    # so prims would still be settling into their poses when the transforms are read.
    env_cfg.wait_for_textures = True
    return builder.make_registered(env_cfg, env_kwargs)


def _reset_and_step(env) -> None:
    """Reset and step the environment so the scene settles before transforms are read."""
    env.reset()
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        for _ in range(NUM_STEPS):
            env.step(actions)


def _flatten(value) -> tuple[float, ...]:
    """Flatten a nested matrix value into a flat tuple of floats."""
    if hasattr(value, "__len__"):
        flat: list[float] = []
        for item in value:
            flat.extend(_flatten(item))
        return tuple(flat)
    return (float(value),)


def _prims_left_at_identity_in_fabric() -> tuple[list[str], int]:
    """Find prims that USD places away from the origin but Fabric still reports at identity.

    Returns:
        The paths of the stale prims, and the number of prims the comparison reached at all.
    """
    import omni.usd
    import usdrt
    from isaaclab.sim.utils.stage import get_current_stage_id
    from pxr import Usd, UsdGeom

    usd_stage = omni.usd.get_context().get_stage()
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    usd_transforms: dict[str, tuple[float, ...]] = {}
    for prim in usd_stage.Traverse():
        if prim.IsA(UsdGeom.Xformable):
            matrix = xform_cache.GetLocalToWorldTransform(prim)
            usd_transforms[prim.GetPath().pathString] = tuple(value for row in matrix for value in row)

    fabric_stage = usdrt.Usd.Stage.Attach(get_current_stage_id())
    stale_prim_paths: list[str] = []
    num_prims_compared = 0
    for prim in fabric_stage.Traverse():
        usd_value = usd_transforms.get(str(prim.GetPath()))
        if usd_value is None:
            continue
        attribute = prim.GetAttribute("omni:fabric:worldMatrix")
        if not attribute or not attribute.IsValid():
            continue
        value = attribute.Get()
        if value is None:
            continue
        fabric_value = _flatten(value)
        if len(fabric_value) != 16:
            continue
        # Only prims USD places off the origin can be distinguished from a legitimate identity.
        if max(abs(entry) for entry in usd_value[12:15]) <= TRANSFORM_TOLERANCE:
            continue
        num_prims_compared += 1
        if np.allclose(fabric_value, np.identity(4).flatten(), atol=TRANSFORM_TOLERANCE):
            stale_prim_paths.append(str(prim.GetPath()))
    return stale_prim_paths, num_prims_compared


def _test_prims_after_stage_rebuild(simulation_app, disable_fabric: bool) -> bool:
    from isaaclab_arena.evaluation.resource_cleanup import close_environment

    stale_prim_paths_per_build: list[list[str]] = []
    num_prims_compared_per_build: list[int] = []
    for build_index in range(NUM_BUILDS):
        env = _build_droid_env(disable_fabric)
        try:
            _reset_and_step(env)
            stale_prim_paths, num_prims_compared = _prims_left_at_identity_in_fabric()
        finally:
            # The same teardown the experiment runner performs between rebuilds.
            close_environment(env)
        stale_prim_paths_per_build.append(stale_prim_paths)
        num_prims_compared_per_build.append(num_prims_compared)
        print(
            f"[build {build_index}] "
            f"{len(stale_prim_paths)} of {num_prims_compared} off-origin prim(s) left at identity in Fabric.",
            flush=True,
        )

    # Checked in both variants: the Fabric Scene Delegate mirrors USD transforms into Fabric for the
    # renderer whenever cameras are on, independently of whether physics drives Fabric, so the
    # Fabric-off build has world matrices to compare too.
    for build_index, num_prims_compared in enumerate(num_prims_compared_per_build):
        assert num_prims_compared >= MIN_PRIMS_COMPARED, (
            f"Build {build_index} exposed no off-origin prim with a Fabric world matrix, so the comparison "
            "would pass vacuously. The Fabric Scene Delegate is likely not running."
        )
    for build_index, stale_prim_paths in enumerate(stale_prim_paths_per_build):
        assert not stale_prim_paths, (
            f"Build {build_index} left {len(stale_prim_paths)} of {num_prims_compared_per_build[build_index]} "
            f"off-origin prim(s) at identity in Fabric, drawn at the origin: {', '.join(stale_prim_paths)}"
        )
    return True


@pytest.mark.with_cameras
def test_prims_after_stage_rebuild_without_fabric():
    """Rebuilds keep their transforms with Fabric off, which is the path the experiment runner takes."""
    assert run_function_with_persistent_simulation_app(
        partial(_test_prims_after_stage_rebuild, disable_fabric=True),
        headless=HEADLESS,
        enable_cameras=True,
        force_disable_fabric=True,
    )


# TODO(alexmillane, 2026-09-02): [lab-render-after-rebuild-bug] Remove once the render after rebuild bug
# is solved in Lab. Under GPU+Fabric every build after the first leaves prims at an identity Fabric world
# matrix, which is the mechanism this test detects.
@pytest.mark.skip(reason="[lab-render-after-rebuild-bug] Rebuilds lose Fabric world matrices under GPU+Fabric.")
@pytest.mark.with_cameras
def test_prims_after_stage_rebuild_with_fabric():
    """Rebuilds should keep the Fabric world matrix of every prim USD places off the origin."""
    assert run_function_with_persistent_simulation_app(
        partial(_test_prims_after_stage_rebuild, disable_fabric=False),
        headless=HEADLESS,
        enable_cameras=True,
        # Opt out of the suite-wide override, which would otherwise build this variant Fabric-off too.
        force_disable_fabric=False,
    )
