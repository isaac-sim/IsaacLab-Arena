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

from isaaclab_arena.utils.pose import Pose


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

    @property
    def num_layouts(self) -> int:
        """Number of complete layouts."""
        return len(next(iter(self.poses.values())))

    def write_episode_jsonl(self, path: str | Path) -> None:
        """Write settled layouts in the episode variations envelope without overwriting."""
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            for index in range(self.num_layouts):
                placement = {
                    "layout_id": f"layout_{index:06d}",
                    "source": "settled",
                    "poses": {name: poses[index].to_dict() for name, poses in self.poses.items()},
                }
                record = {"variations": {"scene.relation_placement": placement}}
                stream.write(json.dumps(record, allow_nan=False) + "\n")
