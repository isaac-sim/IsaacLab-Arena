# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.assets.register import agent_ready, register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
from isaaclab_arena.tasks.common.open_close_door_mimic import RotateDoorMimicEnvCfg
from isaaclab_arena.tasks.rotate_revolute_joint_task import RotateRevoluteJointTask
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg


@agent_ready
@register_task
class CloseDoorTask(RotateRevoluteJointTask):
    def __init__(
        self,
        openable_object: Openable,
        closedness_threshold: float | None = None,
        reset_openness: float = 1.0,  # Start with door OPEN for close task
        episode_length_s: float | None = None,
        task_description: str | None = None,
    ):
        super().__init__(
            openable_object=openable_object,
            target_joint_percentage_threshold=closedness_threshold,
            reset_joint_percentage=reset_openness,  # Reset to OPEN
            episode_length_s=episode_length_s,
            task_description=task_description,
        )

        self.task_description = (
            f"Reach out to the {openable_object.name} and close it." if task_description is None else task_description
        )

    def get_termination_cfg(self) -> TaskTerminationCfg:
        params = {}
        if self.target_joint_percentage_threshold is not None:
            params["threshold"] = self.target_joint_percentage_threshold
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                ProgressObjective(
                    name="close_door",
                    predicate_sequence=[partial(self.openable_object.is_closed, **params)],
                )
            ],
        )

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        return RotateDoorMimicEnvCfg(
            arm_mode=arm_mode,
            openable_object_name=self.openable_object.name,
        )
