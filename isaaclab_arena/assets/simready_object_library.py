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

SIMREADY_USD_OBJECT_REGISTRY_NAME = "simready_usd_object"
"""Registry name a generated spec uses to spawn a searched SimReady asset."""

ISAAC_SIMREADY_GA_S3_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/SimReady"
)

DEFAULT_SIMREADY_SERVICE_URL = "https://search.simready.omniverse.nvidia.com/"
"""Hosted SimReady search, used when the ``service`` source is selected."""

# SimReady GA props author collision/rigid APIs under the Physics=physics variant.
# Without this selection, Usd.Stage.Open sees geometry but no RigidBodyAPI, and
# PickAndPlace contact sensors fail with "No rigid body found".
SIMREADY_PHYSICS_VARIANTS: dict[str, str] = {"Physics": "physics"}


@register_asset
class SimReadyUsdObject(Object):
    """Spawn a SimReady asset from an explicit USD path supplied in graph params."""

    name = SIMREADY_USD_OBJECT_REGISTRY_NAME
    tags = ["object", "sim-ready"]

    object_type: ObjectType | None = None
    """Worked out per instance, since SimReady props range from single bodies to hinged furniture."""

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
        spawn_cfg_addon = dict(kwargs.pop("spawn_cfg_addon", None) or {})
        spawn_cfg_addon.setdefault("variants", dict(SIMREADY_PHYSICS_VARIANTS))
        super().__init__(
            name=instance_name or self.name,
            prim_path=prim_path,
            tags=self.tags,
            usd_path=usd_path,
            object_type=self.object_type,
            scale=scale if scale is not None else (1.0, 1.0, 1.0),
            initial_pose=initial_pose,
            spawn_cfg_addon=spawn_cfg_addon,
            **kwargs,
        )
