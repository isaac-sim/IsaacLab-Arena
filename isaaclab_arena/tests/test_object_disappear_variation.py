# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True


def get_env_local_distances(env, asset_name):
    """Return the asset's per-env horizontal distance from its environment origin."""
    import warp as wp

    pose_w = wp.to_torch(env.unwrapped.scene[asset_name].data.root_pose_w)
    return (pose_w[:, :3] - env.unwrapped.scene.env_origins)[:, :2].norm(dim=-1)


def get_test_environment(*, enabled: bool, probability: float):
    """Build a minimal arena env with an optional enabled disappear variation on a sphere."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.variations.bernoulli_sampler import BernoulliSamplerCfg
    from isaaclab_arena.variations.object_disappear_variation import ObjectDisappearVariationCfg

    sphere = AssetRegistry().get_asset_by_name("sphere")()
    sphere.get_variation("disappear").apply_cfg(
        ObjectDisappearVariationCfg(
            enabled=enabled,
            sampler_cfg=BernoulliSamplerCfg(probability=probability),
        )
    )

    return IsaacLabArenaEnvironment(
        name="test_object_disappear_variation",
        scene=Scene(assets=[sphere]),
    )


def _test_object_disappear_variation_registration(simulation_app):
    from isaaclab_arena.assets.registries import AssetRegistry

    registry = AssetRegistry()
    # Rigid objects carry the variation; non-rigid assets have nothing to teleport.
    assert "disappear" in registry.get_asset_by_name("sphere")().variations
    assert "disappear" not in registry.get_asset_by_name("light")().variations
    assert "disappear" not in registry.get_asset_by_name("table")().variations
    return True


def _test_disabled_disappear_variation_not_in_events_cfg(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = get_test_environment(enabled=False, probability=1.0)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

    assert not hasattr(env_cfg.events, "sphere_disappear"), (
        "Disabled variation must not add 'sphere_disappear' to env_cfg.events; "
        f"got event fields: {sorted(vars(env_cfg.events))}."
    )
    return True


def _test_envs_draw_independently(simulation_app):
    """Each env draws its own outcome, so one reset leaves a mix of present and disappeared."""
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "16", "--seed", "0"])
    env = ArenaEnvBuilder(
        get_test_environment(enabled=True, probability=0.5),
        arena_env_builder_cfg_from_argparse(args_cli),
    ).make_registered()
    try:
        env.reset()
        distances = get_env_local_distances(env, "sphere")
        # 100 m is far past any scene layout but short of the 1000 m park position.
        gone = distances > 100.0
        # P(all envs agree) = 2 * 0.5^16, so a split is essentially certain if draws are independent.
        assert bool(gone.any()) and not bool(
            gone.all()
        ), f"Expected a mix of present and disappeared envs; got distances {distances.tolist()}."
    finally:
        env.close()
    return True


def _test_disappeared_object_survives_relation_placement(simulation_app):
    """Relation placement rewrites poses on reset; the teleport must still win."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.relations import IsAnchor, On
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.variations.bernoulli_sampler import BernoulliSamplerCfg
    from isaaclab_arena.variations.object_disappear_variation import ObjectDisappearVariationCfg

    asset_registry = AssetRegistry()
    table = asset_registry.get_asset_by_name("table")()
    box = asset_registry.get_asset_by_name("cracker_box")()
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))
    table.add_relation(IsAnchor())
    box.add_relation(On(table))
    box.get_variation("disappear").apply_cfg(
        ObjectDisappearVariationCfg(
            enabled=True,
            sampler_cfg=BernoulliSamplerCfg(probability=1.0),
        )
    )
    arena_env = IsaacLabArenaEnvironment(
        name="test_object_disappear_variation_relations",
        scene=Scene(assets=[table, box]),
    )

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    env = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).make_registered()
    try:
        assert hasattr(
            env.unwrapped.cfg.events, "placement_reset"
        ), "Test setup is wrong: relation solving did not register a placement reset event."
        env.reset()
        distance = get_env_local_distances(env, "cracker_box")[0]
        assert (
            distance > 100.0
        ), f"Relation placement must not put a disappeared object back into the scene; got {float(distance)} m."
    finally:
        env.close()
    return True


def _test_hydra_override_enables_disappear(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    env = ArenaEnvBuilder(
        get_test_environment(enabled=False, probability=0.0),
        arena_env_builder_cfg_from_argparse(args_cli),
        hydra_overrides=[
            "sphere.disappear.enabled=true",
            "sphere.disappear.sampler_cfg.probability=1.0",
        ],
    ).make_registered()
    try:
        env.reset()
        distance = get_env_local_distances(env, "sphere")[0]
        assert distance > 100.0, f"Hydra override must enable the variation; got {float(distance)} m."

        record = env.unwrapped.variation_recorder["sphere.disappear"]
        episode_idx = env.unwrapped.get_episode_index(0)
        assert record.sample_for_episode(0, episode_idx) is True
    finally:
        env.close()
    return True


def test_object_disappear_variation_registration():
    assert run_function_with_persistent_simulation_app(
        _test_object_disappear_variation_registration,
        headless=HEADLESS,
    )


def test_disabled_disappear_variation_not_in_events_cfg():
    assert run_function_with_persistent_simulation_app(
        _test_disabled_disappear_variation_not_in_events_cfg,
        headless=HEADLESS,
    )


def test_envs_draw_independently():
    assert run_function_with_persistent_simulation_app(
        _test_envs_draw_independently,
        headless=HEADLESS,
    )


def test_disappeared_object_survives_relation_placement():
    assert run_function_with_persistent_simulation_app(
        _test_disappeared_object_survives_relation_placement,
        headless=HEADLESS,
    )


def test_hydra_override_enables_disappear():
    assert run_function_with_persistent_simulation_app(
        _test_hydra_override_enables_disappear,
        headless=HEADLESS,
    )
