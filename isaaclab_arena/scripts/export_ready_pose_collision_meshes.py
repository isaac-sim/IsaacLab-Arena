# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Export every registered robot's ready-pose collision mesh, for upload to Nucleus.

Extracting a robot's posed mesh costs 0.1-2.5 s per process. Publishing the result once means every
scene that spawns the robot reads it instead. The output holds a folder per robot and is uploaded
verbatim into ``collision_mesh_store.ARENA_ROBOT_LIBRARY_DIR``, merging with each robot's existing
folder: the relative paths are the ones the loader looks for.

Re-run this whenever a robot's USD or configured joint positions change, since the loader validates
the pose an artifact was extracted at and falls back to extraction when it no longer matches.

Run inside the container:

    /isaac-sim/python.sh isaaclab_arena/scripts/export_ready_pose_collision_meshes.py --headless \\
        --out_dir /tmp/ready_pose_export

Check an export before uploading it by pointing the loader at it:

    export ISAACLAB_ARENA_ROBOT_LIBRARY_DIR=/tmp/ready_pose_export
"""

from __future__ import annotations

import argparse
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

if TYPE_CHECKING:
    # Typing only: embodiment_base pulls in Isaac Lab, which is unavailable before sim init.
    from isaaclab_arena.embodiments.embodiment_base import PlacementGeometrySource


def add_export_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the export flags."""
    group = parser.add_argument_group("Ready-Pose Mesh Export Arguments")
    group.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/tmp/ready_pose_export"),
        help="Directory to write per-robot folders into, merged as-is into the Arena robot library.",
    )
    group.add_argument(
        "--robots",
        nargs="+",
        default=None,
        help="Registered embodiment names to export. Defaults to every registered embodiment.",
    )


def _embodiment_classes(names: list[str] | None) -> list[type]:
    """Return the registered embodiment classes to export, in a stable order."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase

    registry = AssetRegistry()
    candidates = names if names is not None else registry.get_all_keys()
    classes = []
    for name in sorted(candidates):
        asset = registry.get_asset_by_name(name)
        if isinstance(asset, type) and issubclass(asset, EmbodimentBase):
            classes.append(asset)
    assert classes, f"no registered embodiments among {candidates}"
    return classes


@dataclass(frozen=True)
class PlannedArtifact:
    """One artifact the export will write, and the embodiment it was planned from."""

    embodiment_name: str
    """Embodiment the mesh is extracted from, for reporting which robot an artifact came from."""

    source: PlacementGeometrySource
    """USD, scale and joint positions the artifact is written from."""


def plan_artifacts(sources: Mapping[str, PlacementGeometrySource]) -> dict[str, PlannedArtifact]:
    """Map each artifact's published relative path to the embodiment and source it is written from.

    Robots differing only in action space share a USD, so they collapse to one artifact. Robots
    missing from ``ROBOT_LIBRARY_FOLDERS`` have nowhere to publish and are left out.

    Args:
        sources: Placement geometry source per embodiment name.
    """
    from isaaclab_arena.utils.collision_mesh_store import published_relative_path

    plan: dict[str, PlannedArtifact] = {}
    for name, source in sorted(sources.items()):
        relative_path = published_relative_path(source.usd_path)
        if relative_path is None:
            continue
        # Artifacts are validated by USD stem, so two robots sharing one would be served each other's
        # mesh. Refuse to publish that rather than let placement use the wrong shape.
        planned = plan.setdefault(relative_path, PlannedArtifact(name, source))
        assert planned.source.usd_path == source.usd_path, (
            f"{relative_path} would be written for both {planned.source.usd_path} and"
            f" {source.usd_path}; rename one of the source USDs"
        )
    return plan


def export_ready_pose_meshes(out_dir: Path, names: list[str] | None) -> tuple[list[str], list[str]]:
    """Write each robot's ready-pose mesh into out_dir, returning the artifacts and the failures.

    Args:
        out_dir: Staging directory, whose per-robot folders are uploaded as-is to the robot library.
        names: Registered embodiment names to export, or None for every registered embodiment.
    """
    from isaaclab_arena.utils.collision_mesh_store import export_ready_pose_mesh, published_relative_path
    from isaaclab_arena.utils.usd_helpers import extract_trimesh_from_usd_at_joint_pos

    sources = {}
    failed = []
    for embodiment_class in _embodiment_classes(names):
        try:
            sources[embodiment_class.name] = embodiment_class().get_placement_geometry_source()
        except Exception as error:
            failed.append(f"{embodiment_class.__name__}: {error}")
            traceback.print_exc()

    skipped = sorted(name for name, source in sources.items() if published_relative_path(source.usd_path) is None)
    written = []
    for relative_path, planned in plan_artifacts(sources).items():
        source = planned.source
        try:
            # Unit scale: one artifact serves every spawn scale, as the loader rescales on read.
            mesh = extract_trimesh_from_usd_at_joint_pos(source.usd_path, source.joint_pos, (1.0, 1.0, 1.0))
            export_ready_pose_mesh(source.usd_path, source.joint_pos, mesh, out_dir)
            written.append(f"{relative_path}  ({len(mesh.vertices)} vertices, from {planned.embodiment_name})")
        except Exception as error:
            failed.append(f"{planned.embodiment_name}: {error}")
            traceback.print_exc()
    if skipped:
        print(f"\nSkipped {len(skipped)} embodiment(s) missing from ROBOT_LIBRARY_FOLDERS: {', '.join(skipped)}")
    return sorted(written), failed


def main() -> None:
    args_parser = get_isaaclab_arena_cli_parser()
    add_export_arguments(args_parser)
    args_cli, _ = args_parser.parse_known_args()

    with SimulationAppContext(args_cli):
        from isaaclab_arena.utils.collision_mesh_store import ARENA_ROBOT_LIBRARY_DIR

        written, failed = export_ready_pose_meshes(args_cli.out_dir, args_cli.robots)

        print(f"\nWrote {len(written)} artifact(s) to {args_cli.out_dir}, upload them to {ARENA_ROBOT_LIBRARY_DIR}:")
        for line in written:
            print(f"  {line}")
        if failed:
            print(f"\n{len(failed)} embodiment(s) failed to export:")
            for line in failed:
                print(f"  {line}")
            # Exit non-zero so a partial export is not mistaken for a complete one and uploaded.
            raise SystemExit(1)


if __name__ == "__main__":
    main()
