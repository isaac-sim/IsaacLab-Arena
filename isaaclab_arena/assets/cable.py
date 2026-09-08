# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from isaaclab.assets import CableObjectCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.sim.spawners.shapes import CableCfg

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.terms.events import reset_cable_to_default
from isaaclab_arena.utils.pose import Pose


class Cable(Asset):
    """A procedurally spawned Isaac Lab cable managed as an Arena scene asset.

    Cable simulation requires the Newton physics backend. Build environments containing this asset
    with ``presets="newton"``.
    """

    def __init__(
        self,
        name: str,
        prim_path: str,
        spawn: CableCfg,
        initial_pose: Pose | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Configure a cable scene asset."""

        super().__init__(name=name, tags=tags)
        assert prim_path, "Cable prim_path must be non-empty."
        self.prim_path = prim_path
        self.initial_pose = initial_pose
        self.object_cfg = CableObjectCfg(prim_path=prim_path, spawn=spawn)
        if initial_pose is not None:
            self.object_cfg.init_state.pos = initial_pose.position_xyz
            self.object_cfg.init_state.rot = initial_pose.rotation_xyzw
        self._reset_event_cfg = EventTermCfg(
            func=reset_cable_to_default,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg(self.name)},
        )

    def get_object_cfg(self) -> tuple[str, CableObjectCfg]:
        """Return the scene key and Isaac Lab cable configuration."""
        return self.name, self.object_cfg

    def get_event_cfg(self) -> tuple[str, EventTermCfg]:
        """Return the event that restores the cable's default segment state."""
        return self.name, self._reset_event_cfg

    def get_initial_pose(self) -> Pose | None:
        """Return the initial pose."""
        return self.initial_pose
