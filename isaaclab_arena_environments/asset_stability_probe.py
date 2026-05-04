# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Asset-gen physics smoke test.

A parameterized env that spawns ONE asset above the center of a table
and lets it fall. Pair with
:mod:`isaaclab_arena.scene_gen.run_stability_check` to classify the
asset as stable / fell_off / tipped / unsettled / spawn_collision —
catching mass / collision-mesh / scale authoring bugs in newly
registered assets without writing a per-asset env.

Run::

    /isaac-sim/python.sh isaaclab_arena/scene_gen/run_stability_check.py \\
        --viz kit --num_envs 1 \\
        asset_stability_probe \\
        --asset_name avocado01_fruits_veggies_robolab

Override the table or drop height::

    ... asset_stability_probe \\
        --asset_name <name> \\
        --background <bg_name> \\
        --drop_height 0.05
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


class AssetStabilityProbeEnvironment(ExampleEnvironmentBase):
    """Drop one asset onto a table and let physics tell us if it's well-authored."""

    name: str = "asset_stability_probe"

    def get_env(self, args_cli: argparse.Namespace) -> "IsaacLabArenaEnvironment":
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab.envs.common import ViewerCfg
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.no_task import NoTask
        from isaaclab_arena.utils.pose import Pose

        # ---- Background, ground, light --------------------------------
        background = self.asset_registry.get_asset_by_name(args_cli.background)()
        background.set_initial_pose(Pose(position_xyz=(0.5, 0.0, 0.0), rotation_wxyz=(0.7071068, 0.0, 0.0, 0.7071068)))
        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        ground_plane.set_initial_pose(Pose(position_xyz=(0.0, 0.0, -1.05)))
        light = self.asset_registry.get_asset_by_name("light")()

        # ---- Tabletop anchor ------------------------------------------
        # Same path convention used by the LLM-gen pipeline; works for any
        # tabletop background that exposes a /table sub-prim.
        tabletop_anchor = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/" + args_cli.background + "/table",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )
        tabletop_anchor.add_relation(IsAnchor())

        # ``embodiment=None`` lets ArenaEnvBuilder fall back to NoEmbodiment
        # automatically — avoids depending on whether ``no_embodiment`` is
        # auto-registered in the asset registry on this branch.
        embodiment = None
        if args_cli.embodiment is not None:
            embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)()

        # ---- The asset under test -------------------------------------
        asset = self.asset_registry.get_asset_by_name(args_cli.asset_name)()

        # Fixed spawn pose at the runtime tabletop center, ``drop_height``
        # meters above the tabletop top surface. Bypassing the relation
        # resolver here on purpose: a smoke test should give every asset
        # the same simple drop test instead of letting the resolver
        # silently work around a bad collision mesh.
        _tbl_bbox = tabletop_anchor.get_world_bounding_box()
        _tbl_min_xyz = [float(_tbl_bbox.min_point[i]) for i in range(3)]
        _tbl_max_xyz = [float(_tbl_bbox.max_point[i]) for i in range(3)]
        _spawn_x = (_tbl_min_xyz[0] + _tbl_max_xyz[0]) / 2.0
        _spawn_y = (_tbl_min_xyz[1] + _tbl_max_xyz[1]) / 2.0
        _spawn_z = _tbl_max_xyz[2] + float(args_cli.drop_height)
        print(
            f"[asset_stability_probe] tabletop AABB: "
            f"min=({_tbl_min_xyz[0]:.3f}, {_tbl_min_xyz[1]:.3f}, {_tbl_min_xyz[2]:.3f}) -> "
            f"max=({_tbl_max_xyz[0]:.3f}, {_tbl_max_xyz[1]:.3f}, {_tbl_max_xyz[2]:.3f}); "
            f"spawning '{args_cli.asset_name}' at ({_spawn_x:.3f}, {_spawn_y:.3f}, {_spawn_z:.3f})",
            flush=True,
        )
        asset.set_initial_pose(
            Pose(
                position_xyz=(_spawn_x, _spawn_y, _spawn_z),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        scene = Scene(assets=[background, ground_plane, light, tabletop_anchor, asset])

        # ---- Camera framed on the spawn point -------------------------
        _viewer_lookat = (
            (_tbl_min_xyz[0] + _tbl_max_xyz[0]) / 2.0,
            (_tbl_min_xyz[1] + _tbl_max_xyz[1]) / 2.0,
            _tbl_max_xyz[2],
        )
        _viewer_eye = (
            _viewer_lookat[0] + 0.6,
            _viewer_lookat[1] + 0.6,
            _viewer_lookat[2] + 0.8,
        )

        def _set_viewer_cfg(env_cfg):
            env_cfg.viewer = ViewerCfg(eye=_viewer_eye, lookat=_viewer_lookat)
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=NoTask(),
            env_cfg_callback=_set_viewer_cfg,
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--asset_name",
            type=str,
            required=True,
            help="Name of the asset to drop on the table (must be in the asset registry).",
        )
        parser.add_argument(
            "--background",
            type=str,
            default="maple_table_robolab",
            help="Name of the tabletop background asset to spawn the test asset on. Default: maple_table_robolab.",
        )
        parser.add_argument(
            "--drop_height",
            type=float,
            default=0.10,
            help=(
                "Height in meters above the tabletop top surface where the asset spawns. "
                "Larger values give a noisier first-step jump that separates 'exploded "
                "into the table' from 'settled normally'. Default: 0.10."
            ),
        )
        parser.add_argument(
            "--embodiment",
            type=str,
            default=None,
            help=(
                "Optional: embodiment to spawn alongside the asset. Default is None, "
                "which falls back to NoEmbodiment so the asset is the only mover in the scene."
            ),
        )
