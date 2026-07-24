# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.predicates.spatial import object_is_below_height, objects_in_proximity
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_transition import Relocate, TaskTransition
from isaaclab_arena.tasks.terminations import SuccessMode, check_success
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object


@register_task
class DeformablePickAndPlaceTask(TaskBase):
    """Pick-and-place task for deformable objects using centroid-based success.

    Mimic/datagen is not yet supported for deformables (``get_mimic_env_cfg`` raises), so this task is
    intentionally not marked ``@agent_ready``.
    """

    def __init__(
        self,
        pick_up_object: Asset,
        destination_location: Asset,
        background_scene: Asset,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        max_separation: tuple[float, float, float] = (0.08, 0.08, 0.08),
        velocity_threshold: float | None = 0.15,
    ):
        super().__init__(episode_length_s=episode_length_s)
        self.pick_up_object = pick_up_object
        self.destination_location = destination_location
        self.background_scene = background_scene
        assert len(max_separation) == 3, f"max_separation must be (x, y, z), got {max_separation!r}"
        self.max_separation = max_separation
        self.velocity_threshold = velocity_threshold
        self.termination_cfg = self.make_termination_cfg()
        self.task_description = (
            f"Pick up the deformable {pick_up_object.name}, and place it onto {destination_location.name}"
            if task_description is None
            else task_description
        )

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        return self.termination_cfg

    def make_termination_cfg(self):
        max_x_separation, max_y_separation, max_z_separation = self.max_separation
        success = TerminationTermCfg(
            func=check_success,
            params={
                "mode": SuccessMode.ALL,
                "predicates": [
                    TerminationTermCfg(
                        func=objects_in_proximity,
                        params={
                            "object_cfg": SceneEntityCfg(self.pick_up_object.name),
                            "target_object_cfg": SceneEntityCfg(self.destination_location.name),
                            "max_x_separation": max_x_separation,
                            "max_y_separation": max_y_separation,
                            "max_z_separation": max_z_separation,
                            "velocity_threshold": self.velocity_threshold,
                        },
                    )
                ],
            },
        )
        object_dropped = TerminationTermCfg(
            func=object_is_below_height,
            params={
                "object_name": self.pick_up_object.name,
                "minimum_height": self.background_scene.object_min_z,
            },
        )
        return DeformablePickAndPlaceTerminationsCfg(
            success=success,
            object_dropped=object_dropped,
        )

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError("Mimic data generation is not wired for deformable pick-and-place yet.")

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.pick_up_object,
            offset=np.array([-1.0, -1.0, 0.8]),
        )

    @classmethod
    def success_state_transition(cls, pick_up_object: str, destination_location: str, **_) -> TaskTransition:
        return TaskTransition(
            subject=pick_up_object,
            effects=(Relocate(subject=pick_up_object, relation="on", target=destination_location),),
        )


@configclass
class DeformablePickAndPlaceTerminationsCfg:
    """Termination terms for deformable pick-and-place."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)

    success: TerminationTermCfg = MISSING

    object_dropped: TerminationTermCfg = MISSING
