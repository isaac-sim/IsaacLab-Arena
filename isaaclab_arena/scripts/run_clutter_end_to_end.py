# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run a clutter pile through the real placement path and report how it comes to rest.

Unlike the pour experiment, which writes poses directly, this declares clutter with the
``cluttered_on`` relation and lets the solver, pour planner and validators do the work:

    ./isaaclab_arena/scripts/run_clutter_end_to_end.py --headless --objects 12
"""

from __future__ import annotations

import argparse


def add_experiment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--objects", type=int, default=12, help="Number of clutter objects to pour.")
    parser.add_argument("--group", type=str, default="tools", help="Clutter group name.")
    parser.add_argument("--spread", type=float, default=1.0, help="Fraction of the support footprint to use.")
    parser.add_argument("--gap_m", type=float, default=0.03, help="Vertical gap between stacked drop poses.")
    parser.add_argument("--max_steps", type=int, default=2000, help="Physics step budget for settling.")
    parser.add_argument("--poll_every", type=int, default=50, help="Physics steps between pose polls.")
    parser.add_argument("--layout_seed", type=int, default=0, help="Placement seed.")
    parser.add_argument(
        "--containment_margin_m",
        type=float,
        default=0.0,
        help="How far outside the support an object may rest before counting as fallen off.",
    )


SUPPORT_ASSET = "office_table_background"
CLUTTER_ASSETS = [
    "tomato_soup_can",
    "cracker_box",
    "sugar_box",
    "mustard_bottle",
    "dex_cube",
    "mug",
    "power_drill",
]


def _build_environment(args_cli):
    """A table with a declared clutter group resting on it."""
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
    for index in range(args_cli.objects):
        asset_name = CLUTTER_ASSETS[index % len(CLUTTER_ASSETS)]
        member = registry.get_asset_by_name(asset_name)(instance_name=f"{asset_name}_{index}")
        member.add_relation(ClutteredOn(support, group=args_cli.group, spread=args_cli.spread, gap_m=args_cli.gap_m))
        members.append(member)

    scene = Scene(assets=[ground, light, support, *members])
    arena_env = IsaacLabArenaEnvironment(name="clutter_end_to_end", scene=scene, task=NoTask())
    return arena_env, support, members


def _run(simulation_app, args_cli) -> bool:
    import torch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.relations.bounding_box_helpers import get_bounding_box_per_env
    from isaaclab_arena.relations.clutter_pour import region_above_support
    from isaaclab_arena.relations.clutter_validation import ClutterSettleParams, SettleTracker, check_resting_poses
    from isaaclab_arena.utils import physics_settle

    arena_env, support, members = _build_environment(args_cli)
    args_cli.num_envs = 1
    for name, default in (("language_instruction", None), ("mimic", False)):
        if not hasattr(args_cli, name):
            setattr(args_cli, name, default)
    env = ArenaEnvBuilder(arena_env, args_cli).make_registered()
    env.reset()

    scene = env.unwrapped.scene
    member_keys = [member.get_scene_key() for member in members]
    names = [member.name for member in members]

    support_bbox = get_bounding_box_per_env(support, 1)
    support_position = support.get_initial_pose().position_xyz
    # Judge the settled pile against the whole support, not the shrunk region it was poured
    # into: a tight pour is meant to relax outward as it settles.
    region = region_above_support(tuple(float(v) for v in support_position), support_bbox)
    print(
        f"\nsupport top z = {region.floor_z:.3f}; region "
        f"x[{region.min_x:.3f},{region.max_x:.3f}] y[{region.min_y:.3f},{region.max_y:.3f}]"
    )

    def _poses() -> tuple[torch.Tensor, torch.Tensor]:
        states = torch.stack([scene[key].data.root_state_w[0] for key in member_keys])
        return states[:, :3] - scene.env_origins[0], states[:, 3:7]

    spawn_positions, _ = _poses()
    above = int((spawn_positions[:, 2] > region.floor_z).sum())
    print(f"spawned above the support surface: {above}/{len(members)}")

    params = ClutterSettleParams(containment_margin_m=args_cli.containment_margin_m)
    tracker = SettleTracker(params)
    settled_at = None
    stepped = 0
    while stepped < args_cli.max_steps:
        chunk = min(args_cli.poll_every, args_cli.max_steps - stepped)
        physics_settle.step_physics(env, chunk)
        stepped += chunk
        positions, rotations = _poses()
        if tracker.update(positions, rotations) and settled_at is None:
            settled_at = stepped
            break

    positions, _ = _poses()
    verdict = check_resting_poses(positions, region, params)

    print(f"\nsettled at: {settled_at if settled_at is not None else 'NOT SETTLED'} (budget {args_cli.max_steps})")
    print(f"rest verdict: {verdict.describe(names)}")
    for name, position in zip(names, positions):
        print(f"  {name:24s} ({position[0]:7.3f},{position[1]:7.3f},{position[2]:7.3f})")

    env.close()
    return settled_at is not None and verdict.ok


def main() -> None:
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    parser = get_isaaclab_arena_cli_parser()
    add_experiment_args(parser)
    args_cli, unknown = parser.parse_known_args()
    assert not unknown, f"unrecognised arguments: {unknown}"
    # Clutter refuses to pour without a seed, since an unreproducible pile defeats seeding.
    if args_cli.placement_seed is None:
        args_cli.placement_seed = args_cli.layout_seed

    with SimulationAppContext(args_cli) as simulation_app:
        ok = _run(simulation_app, args_cli)
        print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
