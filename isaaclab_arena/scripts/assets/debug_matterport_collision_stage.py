#!/usr/bin/env python3
# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build a debug stage for inspecting Matterport collision setup in Isaac Sim.

This script mirrors the runtime collision wiring used by
`MatterportBackground` but is focused on inspection instead of evaluation.
It can:

1. Spawn the visual Matterport USD.
2. Optionally enable explicit child-mesh colliders.
3. Optionally attach legacy or split collision overlays.
4. Make hidden collision overlays visible for viewport inspection.
5. Save the resulting stage to USD so it can be reopened in Isaac Sim.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


def _path_or_none(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path.expanduser().resolve())


parser = argparse.ArgumentParser(description="Create a debug USD stage for Matterport collision inspection.")
parser.add_argument("--visual_usd_path", type=Path, required=True, help="Path to the visual Matterport USD.")
parser.add_argument(
    "--output_usd_path",
    type=Path,
    default=None,
    help="Optional path to export the composed debug stage.",
)
parser.add_argument(
    "--combined_collision_usd_path",
    type=Path,
    default=None,
    help="Optional legacy combined collision overlay USD.",
)
parser.add_argument(
    "--floor_collision_usd_path",
    type=Path,
    default=None,
    help="Optional floor-only collision overlay USD.",
)
parser.add_argument(
    "--obstacle_collision_usd_path",
    type=Path,
    default=None,
    help="Optional obstacle-only collision overlay USD.",
)
parser.add_argument(
    "--prim_path",
    type=str,
    default="/World/matterport",
    help="Prim path where the Matterport scene will be spawned.",
)
parser.add_argument(
    "--mesh_collider_type",
    type=str,
    default="triangle",
    choices=("triangle", "convex_decomposition", "sdf"),
    help="Approximation used for explicit child-mesh colliders and overlays.",
)
parser.add_argument(
    "--enable_child_mesh_colliders",
    action="store_true",
    default=False,
    help="Apply explicit collision schemas to descendant visual USD meshes.",
)
parser.add_argument(
    "--disable_ground_plane",
    action="store_true",
    default=False,
    help="Do not spawn the fallback invisible z=0 ground plane.",
)
parser.add_argument(
    "--show_collision_prims",
    action="store_true",
    default=False,
    help="Make hidden collision overlay prims visible after spawning them.",
)
parser.add_argument(
    "--show_ground_plane",
    action="store_true",
    default=False,
    help="Make the fallback ground plane visible if it is spawned.",
)
parser.add_argument(
    "--flatten_output",
    action="store_true",
    default=False,
    help="Export a flattened stage instead of only the root layer.",
)
parser.add_argument(
    "--settle_frames",
    type=int,
    default=4,
    help="Number of app update frames after spawning before export.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _iter_descendant_prims(root_prim):
    from pxr import Usd

    for child in root_prim.GetFilteredChildren(Usd.TraverseInstanceProxies()):
        yield child
        yield from _iter_descendant_prims(child)


def _count_meshes(stage, prim_path: str) -> int:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return 0
    return sum(1 for descendant in _iter_descendant_prims(prim) if descendant.IsA(UsdGeom.Mesh))


def _set_visible(stage, prim_path: str) -> bool:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False
    UsdGeom.Imageable(prim).MakeVisible()
    return True


def _overlay_prim_names() -> list[str]:
    names: list[str] = []
    if args_cli.combined_collision_usd_path is not None:
        names.append("collision")
    if args_cli.floor_collision_usd_path is not None:
        names.append("collisionFloor")
    if args_cli.obstacle_collision_usd_path is not None:
        names.append("collisionObstacle")
    return names


def main() -> None:
    import omni.usd
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.matterport_background import _spawn_matterport_with_ground

    if args_cli.combined_collision_usd_path and (
        args_cli.floor_collision_usd_path or args_cli.obstacle_collision_usd_path
    ):
        raise ValueError("Use either the combined collision overlay or split floor/obstacle overlays, not both.")

    visual_usd_path = str(args_cli.visual_usd_path.expanduser().resolve())
    output_usd_path = args_cli.output_usd_path.expanduser().resolve() if args_cli.output_usd_path else None

    usd_context = omni.usd.get_context()
    usd_context.new_stage()
    stage = usd_context.get_stage()
    if stage is None:
        raise RuntimeError("Failed to create a fresh USD stage.")

    cfg = sim_utils.UsdFileCfg(usd_path=visual_usd_path)
    cfg.ground_plane_z = None if args_cli.disable_ground_plane else 0.0
    cfg.explicit_mesh_colliders = args_cli.enable_child_mesh_colliders
    cfg.mesh_collider_approximation = args_cli.mesh_collider_type
    cfg.collision_overlay_usd_path = _path_or_none(args_cli.combined_collision_usd_path)
    cfg.floor_collision_usd_path = _path_or_none(args_cli.floor_collision_usd_path)
    cfg.obstacle_collision_usd_path = _path_or_none(args_cli.obstacle_collision_usd_path)

    _spawn_matterport_with_ground(args_cli.prim_path, cfg)

    for _ in range(max(args_cli.settle_frames, 0)):
        simulation_app.update()

    visible_overlays: list[str] = []
    if args_cli.show_collision_prims:
        for overlay_name in _overlay_prim_names():
            overlay_path = f"{args_cli.prim_path}/{overlay_name}"
            if _set_visible(stage, overlay_path):
                visible_overlays.append(overlay_path)

    ground_plane_visible = False
    if args_cli.show_ground_plane:
        ground_plane_visible = _set_visible(stage, "/World/GroundPlane")

    visual_mesh_count = _count_meshes(stage, args_cli.prim_path)
    print(f"[debug_matterport_collision_stage] visual_root={args_cli.prim_path} meshes={visual_mesh_count}")
    for overlay_name in _overlay_prim_names():
        overlay_path = f"{args_cli.prim_path}/{overlay_name}"
        print(f"[debug_matterport_collision_stage] overlay={overlay_path} meshes={_count_meshes(stage, overlay_path)}")

    if visible_overlays:
        print("[debug_matterport_collision_stage] visible_overlays=" + ", ".join(visible_overlays))
    if ground_plane_visible:
        print("[debug_matterport_collision_stage] ground_plane made visible at /World/GroundPlane")

    if output_usd_path is not None:
        output_usd_path.parent.mkdir(parents=True, exist_ok=True)
        if args_cli.flatten_output:
            stage.Export(str(output_usd_path))
            export_mode = "flattened stage"
        else:
            stage.GetRootLayer().Export(str(output_usd_path))
            export_mode = "root layer"
        print(f"[debug_matterport_collision_stage] exported {export_mode} to {output_usd_path}")

    if not args_cli.headless:
        print("[debug_matterport_collision_stage] GUI mode active. Inspect the stage and close Isaac Sim when done.")
        while simulation_app.is_running():
            simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
