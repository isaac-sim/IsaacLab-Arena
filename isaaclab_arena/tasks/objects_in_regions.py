# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""A geometric success task for placing paired objects in oriented regions."""

from __future__ import annotations

import math
import numpy as np
import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

import isaaclab.envs.mdp as mdp_isaac_lab
import warp as wp
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.terminations import SuccessMode, check_success
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


Bounds = tuple[float, float, float, float, float, float]


def points_in_oriented_regions(
    object_positions_w: torch.Tensor,
    region_positions_w: torch.Tensor,
    region_quaternions_w: torch.Tensor,
    bounds_xyzxyz: torch.Tensor,
) -> torch.Tensor:
    """Return whether every corresponding point lies in its oriented local box.

    The bounds are expressed in the local coordinates of their paired region and
    are inclusive, so a point lying exactly on a face is successful.
    """

    object_pos_region, _ = subtract_frame_transforms(
        region_positions_w,
        region_quaternions_w,
        object_positions_w,
    )
    lower = bounds_xyzxyz[:, :3]
    upper = bounds_xyzxyz[:, 3:]
    return ((object_pos_region >= lower) & (object_pos_region <= upper)).all(dim=-1)


def objects_in_regions(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    region_names: list[str],
    bounds_xyzxyz: list[Bounds],
) -> torch.Tensor:
    """Return success only where all paired objects are inside their regions."""

    if not object_names or len(object_names) != len(region_names) or len(object_names) != len(bounds_xyzxyz):
        raise ValueError("objects, regions, and bounds must be non-empty lists of the same length")

    pair_results: list[torch.Tensor] = []
    for object_name, region_name, bounds in zip(object_names, region_names, bounds_xyzxyz, strict=True):
        object_instance: Any = env.scene[object_name]
        region_instance: Any = env.scene[region_name]
        object_positions_w = wp.to_torch(object_instance.data.root_pos_w)
        if hasattr(region_instance, "data"):
            region_positions_w = wp.to_torch(region_instance.data.root_pos_w)
            region_quaternions_w = wp.to_torch(region_instance.data.root_quat_w)
        else:
            region_positions_w, region_quaternions_w = (
                wp.to_torch(value) for value in region_instance.get_world_poses()
            )
        bounds_tensor = torch.as_tensor(bounds, dtype=object_positions_w.dtype, device=object_positions_w.device)
        pair_results.append(
            points_in_oriented_regions(
                object_positions_w,
                region_positions_w,
                region_quaternions_w,
                bounds_tensor.unsqueeze(0).expand(env.num_envs, -1),
            )
        )
    return torch.stack(pair_results, dim=0).all(dim=0)


@register_task
class ObjectsInRegionsTask(TaskBase):
    """Require every object to be within the local bounds of its paired region."""

    def __init__(
        self,
        object_list: list[Asset],
        region_list: list[Asset],
        bounds_xyzxyz: list[Bounds],
        episode_length_s: float = 480.0,
        task_description: str | None = None,
    ):
        self._validate_inputs(object_list, region_list, bounds_xyzxyz)
        super().__init__(
            episode_length_s=episode_length_s,
            task_description=task_description or "Place every object inside its paired region.",
        )
        self.objects = list(object_list)
        self.regions = list(region_list)
        self.bounds = [tuple(float(value) for value in bounds) for bounds in bounds_xyzxyz]
        self.events_cfg = None
        self.termination_cfg = self.make_termination_cfg()
        self.metrics = [SuccessRateMetric()]

    @staticmethod
    def _validate_inputs(object_list: list[Asset], region_list: list[Asset], bounds_xyzxyz: list[Bounds]) -> None:
        if not object_list or not region_list or not bounds_xyzxyz:
            raise ValueError("objects, regions, and bounds must contain at least one paired entry")
        if len(object_list) != len(region_list) or len(object_list) != len(bounds_xyzxyz):
            raise ValueError("objects, regions, and bounds must have the same length")
        for bounds in bounds_xyzxyz:
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 6:
                raise ValueError("each region bounds entry must contain exactly six values")
            try:
                values = tuple(float(value) for value in bounds)
            except (TypeError, ValueError) as error:
                raise ValueError("region bounds must be numeric") from error
            if not all(math.isfinite(value) for value in values):
                raise ValueError("region bounds must be finite")
            if any(lower > upper for lower, upper in zip(values[:3], values[3:], strict=True)):
                raise ValueError("region bounds lower values must not exceed upper values")

    def get_scene_cfg(self):
        return None

    def make_termination_cfg(self):
        region_predicate = TerminationTermCfg(
            func=objects_in_regions,
            params={
                "object_names": [object_.name for object_ in self.objects],
                "region_names": [region.name for region in self.regions],
                "bounds_xyzxyz": self.bounds,
            },
        )
        return TerminationsCfg(
            success=TerminationTermCfg(
                func=check_success,
                params={"mode": SuccessMode.ALL, "predicates": [region_predicate]},
            )
        )

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_events_cfg(self):
        return self.events_cfg

    def get_mimic_env_cfg(self, _arm_mode):
        return None

    def get_metrics(self) -> list[MetricBase]:
        return self.metrics

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.objects[0],
            offset=np.array([-1.5, -1.5, 1.5]),
        )


@configclass
class TerminationsCfg:
    """Termination configuration for :class:`ObjectsInRegionsTask`."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING
