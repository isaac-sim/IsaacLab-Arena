# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""In-sim tests for clutter declared with the ``cluttered_on`` relation.

Covers the path the sim-free tests cannot: that members reach placement at all, that drop
poses survive as spawn poses, and that the pile comes to rest on its support.
"""

from __future__ import annotations

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

SUPPORT_ASSET = "office_table_background"
CLUTTER_ASSETS = ["tomato_soup_can", "cracker_box", "sugar_box", "mustard_bottle", "dex_cube", "mug"]
OBJECT_COUNT = 6
MAX_SETTLE_STEPS = 2000
POLL_EVERY = 50


def _build_scene(seed: int):
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

    scene = Scene(assets=[ground, light, support, *members])
    arena_env = IsaacLabArenaEnvironment(name=f"clutter_test_{seed}", scene=scene, task=NoTask())
    return arena_env, support, members


def _pour_and_settle(seed: int):
    """Build, settle, and return (region, settled_at, resting positions, member names)."""
    import torch

    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.relations.bounding_box_helpers import get_bounding_box_per_env
    from isaaclab_arena.relations.clutter_pour import region_above_support
    from isaaclab_arena.relations.clutter_validation import ClutterSettleParams, SettleTracker
    from isaaclab_arena.utils import physics_settle

    arena_env, support, members = _build_scene(seed)
    # Build args from the real parser rather than a hand-rolled Namespace: the builder reads
    # more fields than are obvious, and a missing one fails only once the env is constructed.
    args = get_isaaclab_arena_cli_parser().parse_args([])
    args.num_envs = 1
    args.placement_seed = seed
    for name, default in (("language_instruction", None), ("mimic", False)):
        if not hasattr(args, name):
            setattr(args, name, default)
    env = ArenaEnvBuilder(arena_env, args).make_registered()
    env.reset()

    scene = env.unwrapped.scene
    member_keys = [member.get_scene_key() for member in members]
    names = [member.name for member in members]
    region = region_above_support(
        tuple(float(value) for value in support.get_initial_pose().position_xyz),
        get_bounding_box_per_env(support, 1),
    )

    def poses():
        states = torch.stack([scene[key].data.root_state_w[0] for key in member_keys])
        return states[:, :3] - scene.env_origins[0], states[:, 3:7]

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


def _test_same_seed_reproduces_the_pile(simulation_app) -> bool:
    import torch

    _, _, _, first, _ = _pour_and_settle(seed=7)
    _, _, _, second, _ = _pour_and_settle(seed=7)
    assert torch.allclose(first, second, atol=1e-4), "same seed produced different piles"
    return True


def test_clutter_settles_on_its_support():
    assert run_simulation_app_function(_test_clutter_settles_on_its_support)


def test_same_seed_reproduces_the_pile():
    assert run_simulation_app_function(_test_same_seed_reproduces_the_pile)
