# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Manager-term adapters for spatial predicates that need Arena assets."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.tasks.predicates.spatial import object_on_destination, objects_on_destinations

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


@dataclass(frozen=True, slots=True, repr=False)
class ArenaAssetHandle:
    """Keep an Arena asset by identity while manager term configurations are copied.

    Termination configurations are created before per-environment variants and relation-solved
    poses are finalized, while progress predicates are copied into event and recorder terms.
    The handle keeps all of those manager terms connected to the original asset.
    """

    asset: ObjectBase
    """The original Arena asset used by the spatial predicate."""

    def __deepcopy__(self, memo: dict[int, object]) -> ArenaAssetHandle:
        """Return this handle unchanged when its containing configuration is copied."""
        memo[id(self)] = self
        return self

    def __repr__(self) -> str:
        return f"{type(self).__name__}(asset={self.asset.name!r})"


def object_on_destination_term(
    env: ManagerBasedRLEnv,
    object_asset_handle: ArenaAssetHandle,
    destination_asset_handle: ArenaAssetHandle,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object_contact_sensor"),
    force_threshold: float = 1.0,
    velocity_threshold: float = 0.5,
    support_cone_half_angle_deg: float = 45.0,
) -> torch.Tensor:
    """Evaluate ``object_on_destination`` with assets held by a manager term configuration."""
    return object_on_destination(
        env=env,
        object_asset=object_asset_handle.asset,
        destination_asset=destination_asset_handle.asset,
        object_cfg=object_cfg,
        contact_sensor_cfg=contact_sensor_cfg,
        force_threshold=force_threshold,
        velocity_threshold=velocity_threshold,
        support_cone_half_angle_deg=support_cone_half_angle_deg,
    )


def objects_on_destinations_term(
    env: ManagerBasedRLEnv,
    object_asset_handle_list: list[ArenaAssetHandle],
    destination_asset_handle_list: list[ArenaAssetHandle],
    object_cfg_list: list[SceneEntityCfg] = [SceneEntityCfg("pick_up_object")],
    contact_sensor_cfg_list: list[SceneEntityCfg] = [SceneEntityCfg("pick_up_object_contact_sensor")],
    force_threshold: float = 1.0,
    velocity_threshold: float = 0.5,
    support_cone_half_angle_deg: float = 45.0,
) -> torch.Tensor:
    """Evaluate ``objects_on_destinations`` with assets held by a manager term configuration."""
    return objects_on_destinations(
        env=env,
        object_asset_list=[asset_handle.asset for asset_handle in object_asset_handle_list],
        destination_asset_list=[asset_handle.asset for asset_handle in destination_asset_handle_list],
        object_cfg_list=object_cfg_list,
        contact_sensor_cfg_list=contact_sensor_cfg_list,
        force_threshold=force_threshold,
        velocity_threshold=velocity_threshold,
        support_cone_half_angle_deg=support_cone_half_angle_deg,
    )
