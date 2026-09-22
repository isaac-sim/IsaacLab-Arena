# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compartment-scoring task for the easy tool-sorting environments."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from functools import partial
from typing import Any

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

Bounds = tuple[float, float, float, float, float, float]


def objects_in_regions(
    env,
    object_names: list[str],
    region_names: list[str],
    bounds_xyzxyz: list[Bounds],
) -> torch.Tensor:
    """Return whether every object root lies in its paired region-local bounds."""
    pair_results = []
    for object_name, region_name, bounds in zip(object_names, region_names, bounds_xyzxyz, strict=True):
        object_data = env.scene[object_name].data
        region_data = env.scene[region_name].data
        object_position_w = object_data.root_pos_w.torch
        region_position_w = region_data.root_pos_w.torch
        region_quaternion_w = region_data.root_quat_w.torch
        object_position_r = quat_apply_inverse(
            region_quaternion_w,
            object_position_w - region_position_w,
        )
        lower = object_position_r.new_tensor(bounds[:3])
        upper = object_position_r.new_tensor(bounds[3:])
        pair_results.append(((object_position_r >= lower) & (object_position_r <= upper)).all(dim=-1))
    return torch.stack(pair_results, dim=0).all(dim=0)


@configclass
class EventsCfg:
    """Standard scene reset."""

    reset_all: EventTermCfg = EventTermCfg(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )


@register_task
class ObjectsInRegionsTask(TaskBase):
    """Require every object's root position to lie in its paired bin compartment."""

    def __init__(
        self,
        object_list: list[Asset],
        region_list: list[Asset],
        bounds_xyzxyz: Sequence[Sequence[float]],
        episode_length_s: float = 192.0,
        task_description: str | None = None,
    ) -> None:
        assert object_list, "Tool sorting requires at least one object."
        assert (
            len(object_list) == len(region_list) == len(bounds_xyzxyz)
        ), "Tool sorting requires one region and one bounds entry for every object."
        bounds = tuple(tuple(float(value) for value in region_bounds) for region_bounds in bounds_xyzxyz)
        assert all(
            len(region_bounds) == 6 for region_bounds in bounds
        ), "Each tool-sort region must use [xmin, ymin, zmin, xmax, ymax, zmax] bounds."
        super().__init__(
            episode_length_s=episode_length_s,
            task_description=task_description or "Sort each tool into its assigned destination-bin compartment.",
        )
        self.objects = tuple(object_list)
        self.regions = tuple(region_list)
        self.bounds = bounds
        self.events_cfg = EventsCfg()

    def get_scene_cfg(self) -> Any:
        return None

    def get_termination_cfg(self) -> TaskTerminationCfg:
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                ProgressObjective(
                    name="sort_tools",
                    predicate_sequence=[
                        partial(
                            objects_in_regions,
                            object_names=[object_.name for object_ in self.objects],
                            region_names=[region.name for region in self.regions],
                            bounds_xyzxyz=list(self.bounds),
                        ),
                    ],
                ),
            ],
        )

    def get_events_cfg(self) -> Any:
        return self.events_cfg

    def get_mimic_env_cfg(self, arm_mode) -> Any:
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]
