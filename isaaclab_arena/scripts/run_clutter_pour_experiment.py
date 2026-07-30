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


def add_experiment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--objects", type=int, default=8, help="Number of clutter objects to pour.")
    parser.add_argument("--settle_steps", type=int, default=400, help="Physics steps to settle for.")
    parser.add_argument("--report_every", type=int, default=50, help="Steps between velocity reports.")
    parser.add_argument("--lin_vel_thresh", type=float, default=0.1, help="Settled linear speed (m/s).")
    parser.add_argument("--ang_vel_thresh", type=float, default=0.1, help="Settled angular speed (rad/s).")
    parser.add_argument("--drop_order", type=str, default="flattest_first", help="as_listed|flattest_first|shuffle")
    parser.add_argument("--xy_sampling", type=str, default="grid_cells", help="grid_cells|uniform")
    parser.add_argument("--clutter_spread", type=float, default=1.0, help="Scales the usable region.")
    parser.add_argument("--layout_seed", type=int, default=0, help="Layout seed.")


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

    bin_asset = registry.get_asset_by_name(BIN_ASSET)()
    bin_asset.set_initial_pose(Pose(position_xyz=BIN_POSITION, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    bin_asset.add_relation(IsAnchor())

    # Park the clutter off to one side; the experiment writes their real poses after reset.
    objects = []
    for index, asset_name in enumerate(CLUTTER_ASSETS[: args_cli.objects]):
        obj = registry.get_asset_by_name(asset_name)()
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
    bin_bbox = get_bounding_box_per_env(bin_asset, 1)
    bin_size = bin_bbox.size[0]
    floor_z = BIN_POSITION[2] + float(bin_bbox.top_surface_z[0])
    half_x = float(bin_size[0]) * 0.5 * BIN_INTERIOR_FRACTION
    half_y = float(bin_size[1]) * 0.5 * BIN_INTERIOR_FRACTION
    print(
        f"bin outer size = {float(bin_size[0]):.3f} x {float(bin_size[1]):.3f} x {float(bin_size[2]):.3f} m; "
        f"usable half-extents = {half_x:.3f} x {half_y:.3f} m; rim z = {floor_z:.3f}"
    )

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
    poses = compute_drop_poses(bboxes, region, params, generator)

    names = [obj.name for obj in objects]
    print(f"\n=== drop layout (floor_z={floor_z:.3f}, backend={args_cli.presets or 'physx'}) ===")
    highest = 0.0
    for name, bbox, pose in zip(names, bboxes, poses):
        rotated = bbox.rotated_by_quat(torch.tensor([pose.rotation_xyzw], dtype=torch.float32))
        bottom = pose.position[2] + float(rotated.bottom_surface_z[0])
        highest = max(highest, pose.position[2] + float(rotated.top_surface_z[0]))
        print(f"  {name:22s} xy=({pose.position[0]:6.3f},{pose.position[1]:6.3f}) bottom_z={bottom:.3f}")
    print(f"  layout spans {highest - floor_z:.3f} m above the floor")

    # Write the drop layout into the sim.
    for name, pose in zip(names, poses):
        asset = scene[name]
        root = asset.data.default_root_state.clone()[0:1] if hasattr(asset.data, "default_root_state") else None
        state = torch.zeros((1, 13), device=device) if root is None else root
        state[0, 0:3] = torch.tensor(pose.position, device=device)
        x, y, z, w = pose.rotation_xyzw
        state[0, 3:7] = torch.tensor([x, y, z, w], device=device)
        state[0, 7:13] = 0.0
        asset.write_root_state_to_sim(state)
    scene.write_data_to_sim()

    print(f"\n=== settling ({args_cli.settle_steps} steps) ===")
    steps_done = 0
    settled_at = None
    while steps_done < args_cli.settle_steps:
        chunk = min(args_cli.report_every, args_cli.settle_steps - steps_done)
        physics_settle.step_physics(env, chunk)
        steps_done += chunk
        settled = physics_settle.are_all_objects_settled_per_env(
            env, [0], names, args_cli.lin_vel_thresh, args_cli.ang_vel_thresh
        )[0]
        max_lin, max_ang, worst = _max_speeds(scene, names)
        print(
            f"  step {steps_done:4d}: settled={settled!s:5s} max_lin={max_lin:6.3f} max_ang={max_ang:6.3f}  ({worst})"
        )
        if settled and settled_at is None:
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
    env.close()
    return settled_at is not None and not escaped


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
    args_cli, _ = parser.parse_known_args()

    with SimulationAppContext(args_cli) as simulation_app:
        ok = _run(simulation_app, args_cli)
        print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
