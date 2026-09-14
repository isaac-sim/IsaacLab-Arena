# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


"""Named object poses for complete, reusable environment layouts."""

from __future__ import annotations

import yaml
from dataclasses import dataclass
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
        assert self.poses, "A placement cache must contain objects"
        assert all(isinstance(name, str) and name for name in self.poses), "Object names must be nonempty strings"
        counts = {len(poses) for poses in self.poses.values()}
        assert len(counts) == 1 and next(iter(counts)) > 0, "All objects must have the same nonzero number of poses"
        for poses in self.poses.values():
            for pose in poses:
                # Pose construction checks dimensions; cache poses also require finite values and unit quaternions.
                Pose.from_dict(pose.to_dict())

    def validate_assets(self, assets: list[PlaceableAsset]) -> None:
        """Require concrete scene keys and complete coverage of relation-placed assets."""
        from isaaclab_arena.assets.object_set import RigidObjectSet
        from isaaclab_arena.relations.relations import RandomAroundSolution, get_relation

        assert not any(
            isinstance(asset, RigidObjectSet) for asset in assets
        ), "Cached layouts require concrete assets, not object sets"
        by_key = {asset.get_scene_key(): asset for asset in assets}
        unknown = set(self.poses) - set(by_key)
        assert not unknown, f"Unknown cached scene objects: {unknown}"
        required = {key for key, asset in by_key.items() if asset.get_spatial_relations() and not asset.is_anchor}
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
    def from_yaml(cls, path: str | Path) -> PlacementLayouts:
        """Read an object-name-to-pose-list YAML mapping."""
        with Path(path).open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        assert isinstance(data, dict), "Placement cache must be a mapping"
        assert all(isinstance(values, list) for values in data.values()), "Each object must have a list of poses"
        poses = {}
        for name, values in data.items():
            poses[name] = []
            for index, value in enumerate(values):
                try:
                    poses[name].append(Pose.from_dict(value))
                except AssertionError as error:
                    raise AssertionError(f"{path}: object '{name}', layout {index}: {error}") from error
        return cls(poses)

    def write_yaml(self, path: str | Path) -> None:
        """Write the layouts as ordinary YAML, refusing to overwrite an existing file."""
        data = {name: [pose.to_dict() for pose in poses] for name, poses in self.poses.items()}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, sort_keys=False)
