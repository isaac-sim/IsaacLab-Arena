# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


"""Named object poses for complete, reusable environment layouts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset


@dataclass
class PlacementLayouts:
    """L complete layouts for N named objects, expressed in environment frame E.

    The same list index selects one complete layout across every object.
    """

    poses: dict[str, list[Pose]]
    """N object names mapped to L poses each; positions have shape (3,), quaternions (4,)."""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Require complete layouts containing finite poses and unit quaternions."""
        assert self.poses, "Placement layouts must contain objects"
        assert all(isinstance(name, str) and name for name in self.poses), "Object names must be nonempty strings"
        counts = {len(poses) for poses in self.poses.values()}
        assert len(counts) == 1 and next(iter(counts)) > 0, "All objects must have the same nonzero number of poses"
        for poses in self.poses.values():
            for pose in poses:
                assert all(
                    isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
                    for value in (*pose.position_xyz, *pose.rotation_xyzw)
                ), "Non-finite pose"
                assert math.isclose(
                    sum(value * value for value in pose.rotation_xyzw), 1.0, abs_tol=1e-4
                ), "Placement poses must have unit quaternions"

    def validate_assets(self, assets: list[PlaceableAsset]) -> None:
        """Require concrete scene keys and complete coverage of relation-placed assets."""
        from isaaclab_arena.assets.object_set import RigidObjectSet
        from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
        from isaaclab_arena.relations.relations import RandomAroundSolution, get_relation

        self.validate()
        assert not any(
            isinstance(asset, RigidObjectSet) for asset in assets
        ), "Cached layouts require concrete assets, not object sets"
        by_key = {asset.get_scene_key(): asset for asset in assets}
        assert len(by_key) == len(assets), "Cached placement assets must have distinct scene keys"
        unknown = set(self.poses) - set(by_key)
        assert not unknown, f"Unknown cached scene objects: {unknown}"
        required = {
            key
            for key, asset in by_key.items()
            if not asset.is_anchor
            and (asset.get_spatial_relations() or (isinstance(asset, EmbodimentBase) and asset.get_relations()))
        }
        missing = required - set(self.poses)
        assert not missing, f"Cache is missing placed objects: {missing}"
        for name in self.poses:
            assert (
                get_relation(by_key[name], RandomAroundSolution) is None
            ), f"Cached object '{name}' cannot randomize on reset"

    @property
    def num_layouts(self) -> int:
        """Number of complete layouts."""
        return len(next(iter(self.poses.values())))

    @classmethod
    def from_episode_jsonl(cls, path: str | Path) -> PlacementLayouts:
        """Read complete layouts in line order, ignoring other episode metadata."""
        poses: dict[str, list[Pose]] = {}
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line, object_pairs_hook=_unique_json_mapping)
                    values = record["variations"]["scene.relation_placement"]["poses"]
                    assert isinstance(values, dict) and values, "Placement poses must be a nonempty mapping"
                    if not poses:
                        poses = {name: [] for name in values}
                    assert values.keys() == poses.keys(), "Every record must contain the same objects"
                    for name, value in values.items():
                        assert isinstance(value, dict) and set(value) == {
                            "position_xyz",
                            "rotation_xyzw",
                        }, f"Object '{name}' requires position_xyz and rotation_xyzw only"
                        for field, size in (("position_xyz", 3), ("rotation_xyzw", 4)):
                            assert (
                                isinstance(value[field], list) and len(value[field]) == size
                            ), f"Object '{name}' {field} must contain {size} numbers"
                        assert all(
                            isinstance(component, Real) and not isinstance(component, bool)
                            for field in value.values()
                            for component in field
                        ), f"Object '{name}' pose components must be numbers"
                        poses[name].append(Pose.from_dict(value))
                except (AssertionError, KeyError, TypeError, ValueError) as error:
                    raise AssertionError(f"{path}, line {line_number}: {error}") from error
        try:
            return cls(poses)
        except AssertionError as error:
            raise AssertionError(f"{path}: {error}") from error

    def write_episode_jsonl(self, path: str | Path, source: str) -> None:
        """Write layouts with their source label in the episode variations envelope without overwriting."""
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            for index in range(self.num_layouts):
                placement = {
                    "layout_id": f"layout_{index:06d}",
                    "source": source,
                    "poses": {name: poses[index].to_dict() for name, poses in self.poses.items()},
                }
                record = {"variations": {"scene.relation_placement": placement}}
                stream.write(json.dumps(record, allow_nan=False) + "\n")


def _unique_json_mapping(items: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON keys instead of silently replacing object poses."""
    result = dict(items)
    assert len(result) == len(items), "Duplicate key in placement record"
    return result
