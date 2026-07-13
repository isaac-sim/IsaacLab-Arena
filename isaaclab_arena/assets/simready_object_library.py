# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generic SimReady USD object registered for agent-generated environments."""

from __future__ import annotations

from typing import Any

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose


@register_asset
class SimReadyUsdObject(Object):
    """Spawn a SimReady asset from an explicit USD path supplied in graph params."""

    name = "simready_usd_object"
    tags = ["object", "sim-ready"]
    object_type = ObjectType.RIGID

    def __init__(
        self,
        usd_path: str,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
        **kwargs: Any,
    ):
        assert usd_path, "simready_usd_object requires params.usd_path"
        kwargs.pop("tags", None)
        super().__init__(
            name=instance_name or self.name,
            prim_path=prim_path,
            tags=self.tags,
            usd_path=usd_path,
            object_type=self.object_type,
            scale=scale if scale is not None else (1.0, 1.0, 1.0),
            initial_pose=initial_pose,
            **kwargs,
        )
