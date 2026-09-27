# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# TODO(alexmillane): [object-in-missing-feature]: Move to a more general ObjectIn task once we have it.

"""Syringe containment and settling tracked by ProgressTracker."""

from __future__ import annotations

import math
import torch

from isaaclab.managers import TerminationTermCfg
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
from isaaclab_arena.tasks.predicates.spatial import velocity_below_threshold
from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.tasks.terminations import check_success


def center_of_mass_in_region(env, object_name: str, region_name: str, bounds: tuple[float, ...]) -> torch.Tensor:
    """Check the object's center of mass against inclusive receiver-local bounds."""
    # ArenaWorld's geometry centroid is not the mass center required by CAP scoring.
    center_W = env.scene[object_name].data.root_com_pos_w.torch
    T_W_R = env.arena_world.get_pose_w(region_name)
    center_R = quat_apply_inverse(T_W_R[:, 3:], center_W - T_W_R[:, :3])
    limits = torch.as_tensor(bounds, device=center_R.device, dtype=center_R.dtype)
    return ((center_R >= limits[:3]) & (center_R <= limits[3:])).all(dim=-1)


# TODO(alexmillane, 2026.09.17) [policy-requested-termination-requested-feature]: Remove this task-specific
# policy-requested termination once we add a framework-wide method for allow the policy to request an
# episode termination.
def cap_episode_finished(env) -> torch.Tensor:
    """End a disconnected CAP episode after settling; never count it as success."""
    return torch.full((env.num_envs,), getattr(env, "cap_episode_finished", False), device=env.device, dtype=torch.bool)


@register_task
class SyringeSortTask(TaskBase):
    """Require syringes to remain settled inside their disposal regions."""

    def __init__(
        self,
        object_list: list[Asset],
        region_list: list[Asset],
        bounds_xyzxyz: list[tuple[float, ...]],
        linear_velocity_threshold: float = 0.01,
        angular_velocity_threshold: float = 0.05,
        consecutive_success_steps: int = 50,
        episode_length_s: float = 228.0,
        task_description: str | None = None,
    ):
        assert object_list and len(object_list) == len(region_list) == len(bounds_xyzxyz)
        assert isinstance(consecutive_success_steps, int) and not isinstance(consecutive_success_steps, bool)
        assert consecutive_success_steps > 0
        for value in (
            linear_velocity_threshold,
            angular_velocity_threshold,
            episode_length_s,
        ):
            assert math.isfinite(value) and value > 0
        for bounds in bounds_xyzxyz:
            assert len(bounds) == 6 and all(math.isfinite(value) for value in bounds)
            assert all(lo <= hi for lo, hi in zip(bounds[:3], bounds[3:], strict=True))
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        self.objects = object_list
        self.regions = region_list
        self.consecutive_success_steps = consecutive_success_steps
        self.bounds = bounds_xyzxyz
        self.linear_velocity_threshold = linear_velocity_threshold
        self.angular_velocity_threshold = angular_velocity_threshold

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self) -> TaskTerminationCfg:
        predicates = []
        for obj, region, bounds in zip(self.objects, self.regions, self.bounds, strict=True):
            predicates.extend([
                TerminationTermCfg(
                    func=center_of_mass_in_region,
                    params={"object_name": obj.name, "region_name": region.name, "bounds": tuple(bounds)},
                ),
                TerminationTermCfg(
                    func=velocity_below_threshold,
                    params={
                        "subject_name": obj.name,
                        "linear_velocity_threshold": self.linear_velocity_threshold,
                        "angular_velocity_threshold": self.angular_velocity_threshold,
                    },
                ),
            ])
        settled_in_regions = TrueForConsecutiveStepsCfg(
            predicate=TerminationTermCfg(func=check_success, params={"predicates": predicates}),
            required_steps=self.consecutive_success_steps,
        )
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[ProgressObjective(name="syringe_sort", predicate_sequence=[settled_in_regions])],
            failures={"cap_finished": TerminationTermCfg(func=cap_episode_finished)},
        )

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode):
        return None

    def get_metrics(self):
        return [SuccessRateMetric()]
