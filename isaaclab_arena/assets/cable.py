# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena asset wrapper for an Isaac Lab cable object."""

from __future__ import annotations

from isaaclab.assets import CableObjectCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.utils.pose import Pose


class Cable(Asset):
    """A procedurally spawned cable managed as a first-class Arena scene asset."""

    def __init__(
        self,
        name: str,
        prim_path: str,
        spawn: SpawnerCfg,
        initial_pose: Pose | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Initialize a cable asset.

        Args:
            name: Isaac Lab scene key for the cable.
            prim_path: USD prim path used to spawn the cable.
            spawn: Isaac Lab cable spawner configuration.
            initial_pose: Optional environment-local initial pose.
            tags: Optional Arena asset tags.
        """
        super().__init__(name=name, tags=tags)
        assert prim_path, "Cable prim_path must be non-empty."
        self.prim_path = prim_path
        self.initial_pose = initial_pose
        self.object_cfg = CableObjectCfg(prim_path=prim_path, spawn=spawn)
        if initial_pose is not None:
            self.object_cfg.init_state.pos = initial_pose.position_xyz
            self.object_cfg.init_state.rot = initial_pose.rotation_xyzw

    def get_object_cfg(self) -> tuple[str, CableObjectCfg]:
        """Return the scene key and Isaac Lab cable configuration."""
        return self.name, self.object_cfg

    def get_event_cfg(self) -> tuple[str, None]:
        """Return no per-asset event; task scene resets restore cable defaults."""
        return self.name, None

    def get_initial_pose(self) -> Pose | None:
        """Return the configured environment-local initial pose."""
        return self.initial_pose
