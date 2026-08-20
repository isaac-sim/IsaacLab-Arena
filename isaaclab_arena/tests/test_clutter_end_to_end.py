# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""In-sim tests for clutter declared with the ``cluttered_on`` relation.

Covers the path the sim-free tests cannot: that members reach placement at all, that drop
poses survive as spawn poses, and that the pile comes to rest on its support.
"""

from __future__ import annotations

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

SUPPORT_ASSET = "office_table_background"
CLUTTER_ASSETS = ["tomato_soup_can", "cracker_box", "sugar_box", "mustard_bottle", "dex_cube", "mug"]
OBJECT_COUNT = 6
MAX_SETTLE_STEPS = 2000
POLL_EVERY = 50


def _build_scene(seed: int, layouts_per_env: int | None = None):
    """A kinematic table with one declared clutter group resting on it."""
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.relations import ClutteredOn, IsAnchor
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.utils.pose import Pose

    registry = AssetRegistry()
    light = registry.get_asset_by_name("light")(spawner_cfg=sim_utils.DomeLightCfg(intensity=1500.0))
    ground = registry.get_asset_by_name("ground_plane")()

    support = registry.get_asset_by_name(SUPPORT_ASSET)()
    support.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    support.add_relation(IsAnchor())

    members = []
    for index in range(OBJECT_COUNT):
        asset_name = CLUTTER_ASSETS[index % len(CLUTTER_ASSETS)]
        member = registry.get_asset_by_name(asset_name)(instance_name=f"{asset_name}_{index}")
        member.add_relation(ClutteredOn(support, group="tools"))
        members.append(member)

    placer_params = None
    if layouts_per_env is not None:
        from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
        from isaaclab_arena.relations.relation_solver_params import RelationSolverParams

        placer_params = ObjectPlacerParams(
            solver_params=RelationSolverParams(verbose=False, save_position_history=False),
            min_unique_layouts_per_env=layouts_per_env,
        )

    scene = Scene(assets=[ground, light, support, *members])
    arena_env = IsaacLabArenaEnvironment(
        name=f"clutter_test_{seed}", scene=scene, task=NoTask(), placer_params=placer_params
    )
    return arena_env, support, members


def _build_and_reset(seed: int, num_envs: int = 1, layouts_per_env: int | None = None):
    """Build the env and reset it, returning (env, support, members, region, poses_fn).

    ``poses_fn(env_id)`` returns that environment's member poses in its own local frame, so
    per-env results are directly comparable against the same region.
    """
    import torch

    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.relations.bounding_box_helpers import get_bounding_box_per_env
    from isaaclab_arena.relations.clutter_pour import region_above_support

    arena_env, support, members = _build_scene(seed, layouts_per_env=layouts_per_env)
    # Build args from the real parser rather than a hand-rolled Namespace: the builder reads
    # more fields than are obvious, and a missing one fails only once the env is constructed.
    args = get_isaaclab_arena_cli_parser().parse_args([])
    args.num_envs = num_envs
    args.placement_seed = seed
    for name, default in (("language_instruction", None), ("mimic", False)):
        if not hasattr(args, name):
            setattr(args, name, default)
    env = ArenaEnvBuilder(arena_env, args).make_registered()
    env.reset()

    scene = env.unwrapped.scene
    member_keys = [member.get_scene_key() for member in members]
    region = region_above_support(
        tuple(float(value) for value in support.get_initial_pose().position_xyz),
        get_bounding_box_per_env(support, num_envs),
    )

    def poses(env_id: int = 0):
        states = torch.stack([scene[key].data.root_state_w[env_id] for key in member_keys])
        return states[:, :3] - scene.env_origins[env_id], states[:, 3:7]

    return env, support, members, region, poses


def _pour_and_settle(seed: int):
    """Build, settle, and return (region, settled_at, spawn positions, resting positions, names)."""
    from isaaclab_arena.relations.clutter_validation import ClutterSettleParams, SettleTracker
    from isaaclab_arena.utils import physics_settle

    env, _support, members, region, poses = _build_and_reset(seed)
    names = [member.name for member in members]

    spawn_positions, _ = poses()
    tracker = SettleTracker(ClutterSettleParams())
    settled_at = None
    stepped = 0
    while stepped < MAX_SETTLE_STEPS:
        chunk = min(POLL_EVERY, MAX_SETTLE_STEPS - stepped)
        physics_settle.step_physics(env, chunk)
        stepped += chunk
        if tracker.update(*poses()):
            settled_at = stepped
            break

    positions, _ = poses()
    result = (region, settled_at, spawn_positions.clone(), positions.clone(), names)
    env.close()
    return result


def _test_clutter_settles_on_its_support(simulation_app) -> bool:
    from isaaclab_arena.relations.clutter_validation import check_resting_poses

    region, settled_at, spawn_positions, positions, names = _pour_and_settle(seed=0)

    # Members must reach placement and be released above the support, not left at the origin.
    above = int((spawn_positions[:, 2] > region.floor_z).sum())
    assert above == len(names), f"only {above}/{len(names)} members spawned above the support surface"

    assert settled_at is not None, f"pile never settled within {MAX_SETTLE_STEPS} steps"

    verdict = check_resting_poses(positions, region)
    assert verdict.ok, f"pile came to rest badly: {verdict.describe(names)}"

    # Resting on the support, not merely inside its footprint at ground level.
    lowest = float(positions[:, 2].min())
    assert lowest > region.floor_z, f"lowest member rests at {lowest:.3f}, at or below support top {region.floor_z:.3f}"
    return True


def _test_pile_is_already_settled_at_reset(simulation_app) -> bool:
    """A reset must place the resting pile, not the poses it was released from."""
    from isaaclab_arena.relations.clutter_validation import ClutterSettleParams, check_resting_poses
    from isaaclab_arena.utils import physics_settle

    env, support, members, region, poses = _build_and_reset(seed=0)
    positions, rotations = poses()

    verdict = check_resting_poses(positions, region, ClutterSettleParams(containment_margin_m=0.05))
    assert not verdict.diverged, "reset produced non-finite poses"

    # Stepping a settled pile barely moves it; a pile still falling moves a long way.
    physics_settle.step_physics(env, 60)
    moved_positions, moved_rotations = poses()
    drift = float((moved_positions - positions).norm(dim=-1).max())
    env.close()

    assert drift < 0.02, f"pile moved {drift:.3f} m after reset, so the reset wrote falling poses"
    return True


def _test_every_parallel_env_gets_its_own_settled_pile(simulation_app) -> bool:
    """Each environment must receive a pile of its own, settled on its own support."""
    import torch

    from isaaclab_arena.relations.clutter_validation import ClutterSettleParams, check_resting_poses

    num_envs = 4
    env, _support, members, region, poses = _build_and_reset(seed=0, num_envs=num_envs)
    names = [member.name for member in members]
    params = ClutterSettleParams(containment_margin_m=0.05)

    per_env_positions = []
    for env_id in range(num_envs):
        positions, _ = poses(env_id)
        per_env_positions.append(positions.clone())
        verdict = check_resting_poses(positions, region, params)
        assert verdict.ok, f"env {env_id} came to rest badly: {verdict.describe(names)}"
        lowest = float(positions[:, 2].min())
        assert lowest > region.floor_z, f"env {env_id} lowest member rests at {lowest:.3f}, below the support"

    env.close()

    # A pool that hands every env the same layout would defeat per-env variation.
    distinct = any(not torch.allclose(per_env_positions[0], other, atol=1e-4) for other in per_env_positions[1:])
    assert distinct, "every parallel env received an identical pile"
    return True


def _test_spilled_layouts_are_rejected_from_the_cache(simulation_app) -> bool:
    """Every cached layout must hold its pile on the support, across repeated draws."""
    from isaaclab_arena.relations.clutter_validation import ClutterSettleParams, check_resting_poses

    layouts_per_env = 6
    env, _support, members, region, poses = _build_and_reset(seed=1, layouts_per_env=layouts_per_env)
    names = [member.name for member in members]
    params = ClutterSettleParams(containment_margin_m=0.02)

    for draw in range(layouts_per_env):
        env.reset()
        positions, _ = poses()
        verdict = check_resting_poses(positions, region, params)
        assert verdict.ok, f"cached layout {draw} spilled: {verdict.describe(names)}"
    env.close()
    return True


def _test_pile_stays_settled_after_the_pool_refills(simulation_app) -> bool:
    """Every reset must place a settled pile, including past the cached set.

    Settling steps physics and so cannot run inside a reset. A pool that solved fresh layouts
    when its queue ran dry would hand back poses nothing had settled, and the pile would be
    released mid-air from the reset that exhausted the cache onwards.
    """
    from isaaclab_arena.utils import physics_settle

    layouts_per_env = 2
    env, _support, _members, _region, poses = _build_and_reset(seed=0, layouts_per_env=layouts_per_env)

    worst_drift = 0.0
    # Well past the cached set, so a queue that refilled instead of rewinding would show.
    for reset_index in range(layouts_per_env * 4):
        env.reset()
        before, _ = poses()
        physics_settle.step_physics(env, 200)
        after, _ = poses()
        drift = float((after - before).norm(dim=-1).max())
        worst_drift = max(worst_drift, drift)
        print(f"  reset {reset_index}: drift {drift:.4f} m")
    env.close()

    assert worst_drift < 0.05, f"a reset placed a pile that was not settled: worst drift {worst_drift:.3f} m"
    return True


def _test_same_seed_reproduces_the_pile(simulation_app) -> bool:
    import torch

    _, _, _, first, _ = _pour_and_settle(seed=7)
    _, _, _, second, _ = _pour_and_settle(seed=7)
    assert torch.allclose(first, second, atol=1e-4), "same seed produced different piles"
    return True


def test_clutter_settles_on_its_support():
    assert run_function_with_persistent_simulation_app(_test_clutter_settles_on_its_support)


def test_pile_is_already_settled_at_reset():
    assert run_function_with_persistent_simulation_app(_test_pile_is_already_settled_at_reset)


def test_every_parallel_env_gets_its_own_settled_pile():
    assert run_function_with_persistent_simulation_app(_test_every_parallel_env_gets_its_own_settled_pile)


def test_spilled_layouts_are_rejected_from_the_cache():
    assert run_function_with_persistent_simulation_app(_test_spilled_layouts_are_rejected_from_the_cache)


def test_pile_stays_settled_after_the_pool_refills():
    assert run_function_with_persistent_simulation_app(_test_pile_stays_settled_after_the_pool_refills)


def test_same_seed_reproduces_the_pile():
    assert run_function_with_persistent_simulation_app(_test_same_seed_reproduces_the_pile)
