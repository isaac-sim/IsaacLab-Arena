# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Drop a computed clutter layout into a live scene and report how it settles.

Answers the questions that geometry alone cannot: does a drop layout settle into a
usable pile within a bounded step budget, does it stay in the bin, and how many steps
does each physics backend need? Run it under both backends to compare:

    ./isaaclab_arena/scripts/run_clutter_pour_experiment.py --headless
    ./isaaclab_arena/scripts/run_clutter_pour_experiment.py --headless --presets newton

This is a diagnostic, not part of the placement pipeline.
"""

from __future__ import annotations

import argparse
import math


def add_experiment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--objects", type=int, default=8, help="Number of clutter objects to pour.")
    parser.add_argument(
        "--region",
        type=str,
        default="bin",
        help="Where to pour: 'bin' (grey bin) or 'floor' (open ground, sized by --region_size).",
    )
    parser.add_argument(
        "--region_size",
        type=float,
        nargs=2,
        default=(1.0, 1.0),
        metavar=("X", "Y"),
        help="Floor region size in metres; only used with --region floor.",
    )
    parser.add_argument("--settle_steps", type=int, default=400, help="Physics steps to settle for.")
    parser.add_argument("--report_every", type=int, default=50, help="Steps between velocity reports.")
    parser.add_argument("--lin_vel_thresh", type=float, default=0.1, help="Settled linear speed (m/s).")
    parser.add_argument(
        "--ang_vel_thresh",
        type=float,
        default=0.8,
        help=(
            "Settled angular speed (rad/s), reported only. Objects in stable contact keep "
            "micro-rocking at 0.25-0.65 rad/s indefinitely, so this cannot decide settling."
        ),
    )
    parser.add_argument(
        "--move_thresh_m",
        type=float,
        default=0.002,
        help="Settled when no object translates more than this between checks.",
    )
    parser.add_argument(
        "--turn_thresh_deg",
        type=float,
        default=2.0,
        help="Settled when no object rotates more than this between checks.",
    )
    parser.add_argument("--drop_order", type=str, default="flattest_first", help="as_listed|flattest_first|shuffle")
    parser.add_argument("--xy_sampling", type=str, default="grid_cells", help="grid_cells|uniform")
    parser.add_argument("--clutter_spread", type=float, default=1.0, help="Scales the usable region.")
    parser.add_argument("--layout_seed", type=int, default=0, help="Layout seed.")
    parser.add_argument(
        "--layers",
        type=int,
        default=1,
        help="Pour in this many batches, settling between each so a later layer lands on the real pile.",
    )
    parser.add_argument(
        "--render_settle",
        action="store_true",
        help="Render every settle step so the pour is visible. Use with --viz kit.",
    )
    parser.add_argument(
        "--hold_seconds",
        type=float,
        default=0.0,
        help="Keep stepping after settling so the result stays on screen. Use with --viz kit.",
    )
    parser.add_argument(
        "--pause_before_pour",
        type=float,
        default=0.0,
        help="Seconds to hold the pre-pour scene, so the drop layout is visible before it falls.",
    )


# Grocery-sized props standing in for tools: a mix of flat, bulky and elongated shapes.
CLUTTER_ASSETS = [
    "tomato_soup_can",
    "cracker_box",
    "sugar_box",
    "mustard_bottle",
    "dex_cube",
    "mug",
    "broccoli",
    "power_drill",
]
BIN_ASSET = "grey_bin_robolab"


def _build_environment(args_cli):
    """Bin on a table with N clutter objects, all anchored so the solver leaves them alone."""
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.relations import AtPosition, IsAnchor
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.utils.pose import Pose

    registry = AssetRegistry()
    light = registry.get_asset_by_name("light")(spawner_cfg=sim_utils.DomeLightCfg(intensity=1500.0))
    ground = registry.get_asset_by_name("ground_plane")()

    # The bin is always the solver's anchor (the ground plane has no bounding box); in floor
    # mode it is parked well clear of the pour region so objects land on open ground.
    bin_position = BIN_POSITION if args_cli.region == "bin" else (5.0, 0.0, 0.0)
    bin_asset = registry.get_asset_by_name(BIN_ASSET)()
    bin_asset.set_initial_pose(Pose(position_xyz=bin_position, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    bin_asset.add_relation(IsAnchor())

    # Park the clutter off to one side; the experiment writes their real poses after reset.
    # The palette repeats for large counts, so each repeat needs its own instance name --
    # duplicates would otherwise resolve to a single scene prim.
    objects = []
    for index in range(args_cli.objects):
        asset_name = CLUTTER_ASSETS[index % len(CLUTTER_ASSETS)]
        obj = registry.get_asset_by_name(asset_name)(instance_name=f"{asset_name}_{index}")
        obj.add_relation(AtPosition(x=1.5 + 0.3 * index, y=0.0, z=0.1))
        objects.append(obj)

    scene = Scene(assets=[ground, light, bin_asset, *objects])
    arena_env = IsaacLabArenaEnvironment(name="clutter_pour_experiment", scene=scene, task=NoTask())
    return arena_env, objects, bin_asset


BIN_POSITION = (0.0, 0.0, 0.0)
BIN_INTERIOR_FRACTION = 0.92
"""Fraction of the bin's outer footprint treated as usable interior (walls are not in the bbox)."""


def _run(simulation_app, args_cli) -> bool:
    import torch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.relations.bounding_box_helpers import get_bounding_box_per_env
    from isaaclab_arena.relations.clutter_drop_poses import (
        ClutterDropParams,
        ClutterRegion,
        DropOrder,
        XySampling,
        compute_drop_poses,
    )
    from isaaclab_arena.utils import physics_settle

    arena_env, objects, bin_asset = _build_environment(args_cli)
    # Reuse the parsed CLI namespace; the builder also reads a couple of policy-runner args.
    args_cli.num_envs = 1
    for name, default in (("language_instruction", None), ("mimic", False)):
        if not hasattr(args_cli, name):
            setattr(args_cli, name, default)
    env = ArenaEnvBuilder(arena_env, args_cli).make_registered()
    env.reset()

    scene = env.unwrapped.scene
    device = env.unwrapped.device

    # Drop from just above the bin rim so objects fall in regardless of interior geometry.
    if args_cli.region == "bin":
        bin_bbox = get_bounding_box_per_env(bin_asset, 1)
        bin_size = bin_bbox.size[0]
        floor_z = BIN_POSITION[2] + float(bin_bbox.top_surface_z[0])
        half_x = float(bin_size[0]) * 0.5 * BIN_INTERIOR_FRACTION
        half_y = float(bin_size[1]) * 0.5 * BIN_INTERIOR_FRACTION
        print(
            f"bin outer size = {float(bin_size[0]):.3f} x {float(bin_size[1]):.3f} x {float(bin_size[2]):.3f} m; "
            f"usable half-extents = {half_x:.3f} x {half_y:.3f} m; rim z = {floor_z:.3f}"
        )
    else:
        half_x, half_y = float(args_cli.region_size[0]) * 0.5, float(args_cli.region_size[1]) * 0.5
        floor_z = 0.0
        print(f"floor region = {args_cli.region_size[0]:.2f} x {args_cli.region_size[1]:.2f} m at z = {floor_z:.3f}")

    region = ClutterRegion(
        min_x=BIN_POSITION[0] - half_x,
        min_y=BIN_POSITION[1] - half_y,
        max_x=BIN_POSITION[0] + half_x,
        max_y=BIN_POSITION[1] + half_y,
        floor_z=floor_z,
    )
    params = ClutterDropParams(
        clutter_spread=args_cli.clutter_spread,
        xy_sampling=XySampling(args_cli.xy_sampling),
        drop_order=DropOrder(args_cli.drop_order),
    )

    bboxes = [get_bounding_box_per_env(obj, 1) for obj in objects]
    generator = torch.Generator()
    generator.manual_seed(args_cli.layout_seed)

    names = [obj.name for obj in objects]
    layers = max(1, args_cli.layers)
    per_layer = math.ceil(len(objects) / layers)
    steps_per_layer = max(1, args_cli.settle_steps // layers)
    print(f"\n=== pouring {len(objects)} objects in {layers} layer(s), backend={args_cli.presets or 'physx'} ===")

    steps_done = 0
    settled_at = None

    for layer_index in range(layers):
        lo, hi = layer_index * per_layer, min((layer_index + 1) * per_layer, len(objects))
        if lo >= hi:
            break
        layer_names, layer_bboxes = names[lo:hi], bboxes[lo:hi]

        # Later layers land on the pile the earlier ones actually formed, read from the sim,
        # so the stack interleaves instead of reserving worst-case height up front.
        layer_floor = floor_z if layer_index == 0 else _pile_top(scene, names[:lo], bboxes[:lo])
        layer_region = ClutterRegion(region.min_x, region.min_y, region.max_x, region.max_y, layer_floor)
        poses = compute_drop_poses(layer_bboxes, layer_region, params, generator)

        highest = max(
            pose.position[2] + float(bbox.rotated_by_quat(torch.tensor([pose.rotation_xyzw])).top_surface_z[0])
            for bbox, pose in zip(layer_bboxes, poses)
        )
        print(
            f"\n--- layer {layer_index + 1}/{layers}: {len(layer_names)} objects, "
            f"floor_z={layer_floor:.3f}, spans {highest - layer_floor:.3f} m ---"
        )

        for name, pose in zip(layer_names, poses):
            asset = scene[name]
            state = asset.data.default_root_state.clone()[0:1]
            state[0, 0:3] = torch.tensor(pose.position, device=device)
            state[0, 3:7] = torch.tensor(list(pose.rotation_xyzw), device=device)
            state[0, 7:13] = 0.0
            asset.write_root_state_to_sim(state)
        scene.write_data_to_sim()

        if args_cli.pause_before_pour > 0.0:
            _hold(env, args_cli.pause_before_pour, freeze=layer_names)

        # A new layer changes the object set, so the previous snapshot no longer lines up.
        previous_poses = None
        layer_steps = 0
        while layer_steps < steps_per_layer:
            chunk = min(args_cli.report_every, steps_per_layer - layer_steps)
            physics_settle.step_physics(env, chunk, render=args_cli.render_settle)
            layer_steps += chunk
            steps_done += chunk

            current = _read_poses(scene, names[:hi])
            moved, turned = _pose_delta(previous_poses, current) if previous_poses else (float("inf"), float("inf"))
            previous_poses = current
            settled = moved <= args_cli.move_thresh_m and turned <= args_cli.turn_thresh_deg
            max_lin, max_ang, worst = _max_speeds(scene, names[:hi])
            print(
                f"  step {steps_done:4d}: settled={settled!s:5s} "
                f"moved={moved:7.4f}m turned={turned:6.2f}deg  |  "
                f"max_lin={max_lin:6.3f} max_ang={max_ang:6.3f} ({worst})"
            )
            if settled and settled_at is None and layer_index == layers - 1:
                settled_at = steps_done

    print("\n=== result ===")
    print(f"  settled first observed at: {settled_at if settled_at is not None else 'NOT SETTLED'}")

    escaped = []
    for name in names:
        position = _root_position(scene[name])
        inside = (
            region.min_x <= float(position[0]) <= region.max_x and region.min_y <= float(position[1]) <= region.max_y
        )
        if not inside:
            escaped.append((name, [round(float(v), 3) for v in position]))
        print(f"  {name:22s} final xyz=({position[0]:6.3f},{position[1]:6.3f},{position[2]:6.3f}) in_region={inside}")

    print(f"\n  escaped: {len(escaped)}/{len(names)}" + (f" -> {escaped}" if escaped else ""))

    if args_cli.hold_seconds > 0.0:
        print(f"\n=== holding result for {args_cli.hold_seconds:.0f}s ===")
        _hold(env, args_cli.hold_seconds)

    env.close()
    return settled_at is not None and not escaped


def _hold(env, seconds: float, freeze: list[str] | None = None) -> None:
    """Keep the viewer alive, optionally re-freezing objects so they stay put on screen."""
    import time
    import torch

    from isaaclab_arena.utils import physics_settle

    deadline = time.time() + seconds
    frozen = None
    if freeze:
        frozen = {name: env.unwrapped.scene[name].data.root_state_w[0:1].clone() for name in freeze}
    while time.time() < deadline:
        if frozen:
            for name, state in frozen.items():
                held = state.clone()
                held[0, 7:13] = torch.zeros(6, device=held.device)
                env.unwrapped.scene[name].write_root_state_to_sim(held)
            env.unwrapped.scene.write_data_to_sim()
        physics_settle.step_physics(env, 1, render=True)


def _read_poses(scene, names):
    """Current world positions and orientations, as CPU tensors."""
    import torch

    import warp as wp

    positions = torch.stack([wp.to_torch(scene[name].data.root_pos_w)[0].cpu() for name in names])
    rotations = torch.stack([wp.to_torch(scene[name].data.root_quat_w)[0].cpu() for name in names])
    return positions, rotations


def _pose_delta(previous, current) -> tuple[float, float]:
    """Largest translation (m) and rotation (deg) any object underwent between two snapshots.

    Preferred over a velocity threshold: an object wedged in a pile keeps micro-rocking at a
    non-trivial angular speed indefinitely while going nowhere, so angular velocity never
    crosses a strict threshold even though the layout is stable. Displacement measures what
    actually matters -- whether the pile is still changing.
    """
    import torch

    previous_positions, previous_rotations = previous
    positions, rotations = current
    moved = float((positions - previous_positions).norm(dim=1).max())
    # Angle between unit quaternions, sign-invariant because q and -q are the same rotation.
    dots = (rotations * previous_rotations).sum(dim=1).abs().clamp(max=1.0)
    turned = float(torch.rad2deg(2.0 * torch.acos(dots)).max())
    return moved, turned


def _pile_top(scene, names, bboxes) -> float:
    """Highest point of the objects already poured, as they currently rest."""
    import torch

    import warp as wp

    top = 0.0
    for name, bbox in zip(names, bboxes):
        data = scene[name].data
        z = float(wp.to_torch(data.root_pos_w)[0][2])
        quat = wp.to_torch(data.root_quat_w)[0]
        rotated = bbox.rotated_by_quat(torch.tensor([quat.tolist()], dtype=torch.float32))
        top = max(top, z + float(rotated.top_surface_z[0]))
    return top


def _max_speeds(scene, names):
    """Largest linear/angular speed across the clutter, and which object holds it."""
    import warp as wp

    max_lin, max_ang, worst = 0.0, 0.0, ""
    for name in names:
        data = scene[name].data
        lin = float(wp.to_torch(data.root_lin_vel_w)[0].norm())
        ang = float(wp.to_torch(data.root_ang_vel_w)[0].norm())
        if lin > max_lin:
            max_lin, worst = lin, name
        max_ang = max(max_ang, ang)
    return max_lin, max_ang, worst


def _root_position(asset):
    import warp as wp

    return wp.to_torch(asset.data.root_pos_w)[0]


def main() -> None:
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    parser = get_isaaclab_arena_cli_parser()
    add_experiment_args(parser)
    # Reject unknown flags rather than ignoring them: a silently-dropped option looks like a
    # feature that ran and did nothing.
    args_cli, unknown = parser.parse_known_args()
    assert not unknown, f"unrecognised arguments: {unknown}"

    with SimulationAppContext(args_cli) as simulation_app:
        ok = _run(simulation_app, args_cli)
        print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
