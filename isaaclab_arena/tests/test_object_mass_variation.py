# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function
from isaaclab_arena.variations.object_mass_variation import ObjectMassVariationCfg
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

HEADLESS = True
TEST_ASSET_NAME = "sphere"
TEST_EVENT_NAME = f"{TEST_ASSET_NAME}_mass_variation"
TEST_MASS = 0.125
TEST_HYDRA_MASS = 0.25


def get_test_environment(
    *,
    enabled: bool,
    mass: float = TEST_MASS,
    mass_low: float | None = None,
    mass_high: float | None = None,
    recompute_inertia: bool = True,
):
    """Build a minimal arena env with an optional enabled object mass variation.

    By default the mass sampler is a point mass (``low == high == mass``). Pass ``mass_low``/``mass_high``
    to sample over a range instead (used to draw distinct per-env masses).
    """
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    sphere = AssetRegistry().get_asset_by_name(TEST_ASSET_NAME)()
    assert sphere.name == TEST_ASSET_NAME

    low = [mass_low if mass_low is not None else mass]
    high = [mass_high if mass_high is not None else mass]
    variation = sphere.get_variation("mass")
    variation.apply_cfg(
        ObjectMassVariationCfg(
            sampler_cfg=UniformSamplerCfg(low=low, high=high),
            recompute_inertia=recompute_inertia,
        )
    )
    if enabled:
        variation.enable()
    assert variation.enabled is enabled

    return IsaacLabArenaEnvironment(
        name="test_object_mass_variation",
        scene=Scene(assets=[sphere]),
    )


def _test_object_mass_variation_registration(simulation_app):
    import torch
    from types import SimpleNamespace
    from unittest.mock import patch

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    registry = AssetRegistry()
    sphere = registry.get_asset_by_name("sphere")()
    assert "mass" in sphere.variations

    dome_light = registry.get_asset_by_name("light")()
    assert "mass" not in dome_light.variations

    table = registry.get_asset_by_name("table")()
    assert "mass" not in table.variations

    can_a = Object(name="can_a", object_type=ObjectType.RIGID, usd_path="/tmp/can_a.usd")
    can_b = Object(name="can_b", object_type=ObjectType.RIGID, usd_path="/tmp/can_b.usd")
    can_a.bounding_box = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.2))
    can_b.bounding_box = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.2, 0.2, 0.3))
    with (
        patch("isaaclab_arena.assets.object_set.detect_object_type", return_value=ObjectType.RIGID),
        patch("isaaclab_arena.assets.object_set.find_shallowest_rigid_body", return_value="/rigid"),
        patch("isaaclab_arena.assets.object_set.torch.randint", return_value=torch.tensor([0, 1])),
    ):
        obj_set = RigidObjectSet(name="cans", objects=[can_a, can_b], random_choice=True)
    assert "mass" in obj_set.variations

    class DefaultPrim:
        def GetPath(self):
            return "/World/parent"

    class Stage:
        def GetDefaultPrim(self):
            return DefaultPrim()

        def GetPrimAtPath(self, prim_path):
            return object()

    class OpenStage:
        def __init__(self, path):
            pass

        def __enter__(self):
            return Stage()

        def __exit__(self, exc_type, exc, tb):
            return False

    parent = SimpleNamespace(
        name="parent",
        usd_path="/tmp/parent.usd",
        scale=(1.0, 1.0, 1.0),
        initial_pose=None,
    )
    with (
        patch("isaaclab_arena.assets.object_reference.open_stage", OpenStage),
        patch(
            "isaaclab_arena.assets.object_reference.get_prim_pose_in_default_prim_frame", return_value=Pose.identity()
        ),
    ):
        object_ref = ObjectReference(
            name="object_ref",
            prim_path="{ENV_REGEX_NS}/parent/object_ref",
            parent_asset=parent,
            object_type=ObjectType.RIGID,
        )
    assert "mass" in object_ref.variations
    return True


def _test_disabled_object_mass_variation_not_in_events_cfg(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = get_test_environment(enabled=False)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

    assert not hasattr(env_cfg.events, TEST_EVENT_NAME), (
        f"Disabled variation must not add '{TEST_EVENT_NAME}' to env_cfg.events; "
        f"got event fields: {sorted(vars(env_cfg.events))}."
    )
    return True


def _test_enabled_object_mass_variation_in_events_cfg(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.variations.object_mass_variation import ApplyObjectMassFromSampler

    arena_env = get_test_environment(enabled=True)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

    assert hasattr(
        env_cfg.events, TEST_EVENT_NAME
    ), f"Expected env_cfg.events to contain '{TEST_EVENT_NAME}'; got event fields: {sorted(vars(env_cfg.events))}."
    event_cfg = getattr(env_cfg.events, TEST_EVENT_NAME)
    assert event_cfg.func is ApplyObjectMassFromSampler
    assert event_cfg.mode == "reset"
    assert event_cfg.params["asset_cfg"].name == TEST_ASSET_NAME
    assert event_cfg.params["recompute_inertia"] is True
    return True


def _test_object_mass_variation_realized_and_recorded(simulation_app):
    import warp as wp

    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    env = ArenaEnvBuilder(
        get_test_environment(enabled=True),
        arena_env_builder_cfg_from_argparse(args_cli),
    ).make_registered()
    try:
        env.reset()
        asset = env.unwrapped.scene[TEST_ASSET_NAME]

        mass = wp.to_torch(asset.data.body_mass).clone()
        torch.testing.assert_close(
            mass[0, 0],
            torch.tensor(TEST_MASS, device=mass.device, dtype=mass.dtype),
            atol=1e-6,
            rtol=1e-6,
        )

        record = env.unwrapped.variation_recorder[f"{TEST_ASSET_NAME}.mass"]
        episode_idx = env.unwrapped.get_episode_index(0)
        recorded_mass = record.sample_for_episode(0, episode_idx)
        assert recorded_mass.tolist() == pytest.approx([TEST_MASS], abs=1e-6)

        first_inertia = wp.to_torch(asset.data.body_inertia).clone()
        env.reset()
        second_inertia = wp.to_torch(asset.data.body_inertia).clone()
        torch.testing.assert_close(second_inertia, first_inertia, atol=1e-6, rtol=1e-6)
    finally:
        env.close()
    return True


def _test_object_mass_variation_scales_inertia(simulation_app):
    """With recompute_inertia, each env's inertia scales linearly with its sampled mass.

    Draws distinct per-env masses over a range, then checks ``inertia / mass`` is the same across
    envs (i.e. inertia == default_inertia * mass / default_mass), directly validating the scaling.
    """
    import warp as wp

    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "4", "--seed", "0"])
    env = ArenaEnvBuilder(
        get_test_environment(enabled=True, mass_low=0.1, mass_high=3.0, recompute_inertia=True),
        arena_env_builder_cfg_from_argparse(args_cli),
    ).make_registered()
    try:
        env.reset()
        asset = env.unwrapped.scene[TEST_ASSET_NAME]
        masses = wp.to_torch(asset.data.body_mass)[:, 0].clone()  # (num_envs,)
        inertias = wp.to_torch(asset.data.body_inertia)[:, 0].clone()  # (num_envs, K)

        # The test is only meaningful if the envs actually drew different masses.
        assert (masses.max() - masses.min()).item() > 0.1, f"expected a spread of per-env masses, got {masses}."

        # inertia == default_inertia * (mass / default_mass) => inertia / mass is constant across envs.
        inertia_per_mass = inertias / masses[:, None]
        reference = inertia_per_mass[0]
        significant = reference.abs() > 1e-9  # ignore identically-zero inertia components
        for env_id in range(1, masses.shape[0]):
            torch.testing.assert_close(
                inertia_per_mass[env_id][significant],
                reference[significant],
                rtol=1e-4,
                atol=1e-8,
            )
    finally:
        env.close()
    return True


def _test_hydra_override_applies_object_mass_variation(simulation_app):
    import warp as wp

    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    env = ArenaEnvBuilder(
        get_test_environment(enabled=False),
        arena_env_builder_cfg_from_argparse(args_cli),
        hydra_overrides=[
            f"{TEST_ASSET_NAME}.mass.enabled=true",
            f"{TEST_ASSET_NAME}.mass.sampler_cfg.low=[{TEST_HYDRA_MASS}]",
            f"{TEST_ASSET_NAME}.mass.sampler_cfg.high=[{TEST_HYDRA_MASS}]",
            f"{TEST_ASSET_NAME}.mass.recompute_inertia=false",
        ],
    ).make_registered()
    try:
        env.reset()
        asset = env.unwrapped.scene[TEST_ASSET_NAME]
        mass = wp.to_torch(asset.data.body_mass).clone()
        torch.testing.assert_close(
            mass[0, 0],
            torch.tensor(TEST_HYDRA_MASS, device=mass.device, dtype=mass.dtype),
            atol=1e-6,
            rtol=1e-6,
        )

        record = env.unwrapped.variation_recorder[f"{TEST_ASSET_NAME}.mass"]
        episode_idx = env.unwrapped.get_episode_index(0)
        recorded_mass = record.sample_for_episode(0, episode_idx)
        assert recorded_mass.tolist() == pytest.approx([TEST_HYDRA_MASS], abs=1e-6)
        assert record.cfg.recompute_inertia is False
    finally:
        env.close()
    return True


def _test_object_mass_variation_partial_reset_accepts_sequence_env_ids(simulation_app):
    import warp as wp

    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "3"])
    env = ArenaEnvBuilder(
        get_test_environment(enabled=True),
        arena_env_builder_cfg_from_argparse(args_cli),
    ).make_registered()
    try:
        env.reset()
        asset = env.unwrapped.scene[TEST_ASSET_NAME]
        mass = wp.to_torch(asset.data.body_mass).clone()
        manual_masses = torch.tensor([[0.2], [0.3], [0.4]], device=mass.device, dtype=mass.dtype)
        env_ids = torch.arange(3, device=asset.device, dtype=torch.int32)
        body_ids = torch.tensor([0], device=asset.device, dtype=torch.int32)
        asset.set_masses_index(masses=manual_masses, body_ids=body_ids, env_ids=env_ids)

        env.unwrapped.reset(env_ids=[1])

        mass_after_reset = wp.to_torch(asset.data.body_mass).clone()
        expected_masses = manual_masses.clone()
        expected_masses[1, 0] = TEST_MASS
        torch.testing.assert_close(mass_after_reset[:, 0], expected_masses[:, 0], atol=1e-6, rtol=1e-6)

        record = env.unwrapped.variation_recorder[f"{TEST_ASSET_NAME}.mass"]
        episode_idx = env.unwrapped.get_episode_index(1)
        recorded_mass = record.sample_for_episode(1, episode_idx)
        assert recorded_mass.tolist() == pytest.approx([TEST_MASS], abs=1e-6)
    finally:
        env.close()
    return True


def _test_object_mass_sample_below_floor_fails(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    # Sample a mass below the fixed physical floor (_MIN_PHYSICAL_MASS_KG = 1e-6) to trip the guard.
    builder = ArenaEnvBuilder(
        get_test_environment(enabled=True, mass=1e-9),
        arena_env_builder_cfg_from_argparse(args_cli),
    )
    env = None
    try:
        with pytest.raises(AssertionError, match="minimum physical mass"):
            env = builder.make_registered()
            env.reset()
    finally:
        if env is not None:
            env.close()
    return True


def test_object_mass_variation_registration():
    assert run_simulation_app_function(
        _test_object_mass_variation_registration,
        headless=HEADLESS,
    )


def test_disabled_object_mass_variation_not_in_events_cfg():
    assert run_simulation_app_function(
        _test_disabled_object_mass_variation_not_in_events_cfg,
        headless=HEADLESS,
    )


def test_enabled_object_mass_variation_in_events_cfg():
    assert run_simulation_app_function(
        _test_enabled_object_mass_variation_in_events_cfg,
        headless=HEADLESS,
    )


def test_object_mass_variation_realized_and_recorded():
    assert run_simulation_app_function(
        _test_object_mass_variation_realized_and_recorded,
        headless=HEADLESS,
    )


def test_object_mass_variation_scales_inertia():
    assert run_simulation_app_function(
        _test_object_mass_variation_scales_inertia,
        headless=HEADLESS,
    )


def test_hydra_override_applies_object_mass_variation():
    assert run_simulation_app_function(
        _test_hydra_override_applies_object_mass_variation,
        headless=HEADLESS,
    )


def test_object_mass_variation_partial_reset_accepts_sequence_env_ids():
    assert run_simulation_app_function(
        _test_object_mass_variation_partial_reset_accepts_sequence_env_ids,
        headless=HEADLESS,
    )


def test_object_mass_sample_below_floor_fails():
    assert run_simulation_app_function(
        _test_object_mass_sample_below_floor_fails,
        headless=HEADLESS,
    )
