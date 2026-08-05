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
    parser.add_argument(
        "--record_video",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Write an mp4 of the pour to PATH. Adds a fixed camera and forces --enable_cameras. "
            "Note that enabling cameras diverges Newton above roughly 20 objects."
        ),
    )
    parser.add_argument("--video_every", type=int, default=4, help="Capture a frame every N physics steps.")
    parser.add_argument("--video_fps", type=int, default=30, help="Frame rate of the written mp4.")
    parser.add_argument(
        "--region_hint",
        type=float,
        default=1.6,
        help="Approximate support width in metres, used only to frame the video camera.",
    )
    parser.add_argument(
        "--show_pour",
        action="store_true",
        help=(
            "Skip settling the pool at build time so the pour happens on this script's steps "
            "and can be watched. The pile is normally already at rest when a reset writes it."
        ),
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


def _disable_build_time_settle(env_cfg):
    """Leave the pool holding drop poses so the pour itself is visible."""
    env_cfg.settle_clutter_on_build = False
    return env_cfg


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
    callbacks = []
    if args_cli.record_video:
        import functools

        from isaaclab_arena.scripts.run_clutter_pour_experiment import _add_video_camera

        half = 0.5 * max(args_cli.region_hint, 0.1)
        callbacks.append(functools.partial(_add_video_camera, region_half_x=half, region_half_y=half))
    if args_cli.show_pour:
        callbacks.append(_disable_build_time_settle)

    def env_cfg_callback(env_cfg):
        for callback in callbacks:
            env_cfg = callback(env_cfg)
        return env_cfg

    env_cfg_callback = env_cfg_callback if callbacks else None

    arena_env = IsaacLabArenaEnvironment(
        name="clutter_end_to_end", scene=scene, task=NoTask(), env_cfg_callback=env_cfg_callback
    )
    return arena_env, support, members


def _run(simulation_app, args_cli) -> bool:
    import torch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.relations.bounding_box_helpers import get_bounding_box_per_env
    from isaaclab_arena.relations.clutter_pour import region_above_support
    from isaaclab_arena.relations.clutter_validation import ClutterSettleParams, SettleTracker, check_resting_poses
    from isaaclab_arena.scripts.run_clutter_pour_experiment import _capture_frame, _write_video
    from isaaclab_arena.utils import physics_settle

    arena_env, support, members = _build_environment(args_cli)
    # The CLI parser supplies --num_envs; honour it so parallel envs can be inspected too.
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
    frames: list | None = [] if args_cli.record_video else None
    settled_at = None
    stepped = 0
    # Capturing and polling are independent schedules. Stepping to whichever falls next keeps
    # each on its own stride, so the settle verdict is built from polls at poll_every whatever
    # the capture rate; advancing a deadline by its own period rather than to the step that
    # overshot it keeps that stride from drifting later and later.
    next_poll = args_cli.poll_every
    next_frame = args_cli.video_every if frames is not None else args_cli.max_steps
    while stepped < args_cli.max_steps:
        target = min(next_poll, next_frame, args_cli.max_steps)
        physics_settle.step_physics(env, target - stepped, render=frames is not None)
        stepped = target
        if frames is not None and stepped >= next_frame:
            _capture_frame(scene, frames)
            next_frame += args_cli.video_every
        if stepped < next_poll:
            continue
        next_poll += args_cli.poll_every
        positions, rotations = _poses()
        if tracker.update(positions, rotations) and settled_at is None:
            settled_at = stepped
            break

    positions, _ = _poses()
    if frames is not None:
        # Linger on the settled pile so the video ends on the result rather than mid-fall.
        for _ in range(args_cli.video_fps):
            _capture_frame(scene, frames)
        _write_video(frames, args_cli.record_video, args_cli.video_fps)
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
    if args_cli.record_video:
        # The pour camera only produces frames when the renderer is up.
        args_cli.enable_cameras = True

    with SimulationAppContext(args_cli) as simulation_app:
        ok = _run(simulation_app, args_cli)
        print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
