# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task in which a robot closes an openable object such as a door."""

from functools import partial

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.assets.register import agent_ready, register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.common.open_close_door_mimic import RotateDoorMimicEnvCfg
from isaaclab_arena.tasks.rotate_revolute_joint_task import RotateRevoluteJointTask
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg


@agent_ready
@register_task
class CloseDoorTask(RotateRevoluteJointTask):
    """Task that succeeds when an openable object is closed; the object starts open."""

    def __init__(
        self,
        openable_object: Openable,
        closedness_threshold: float | None = None,
        reset_openness: float = 1.0,  # Start with door OPEN for close task
        episode_length_s: float | None = None,
        task_description: str | None = None,
    ):
        """Initialize the close-door task.

        Args:
            openable_object: Object to close.
            closedness_threshold: Threshold for the object to count as closed; None selects the
                affordance's own threshold.
            reset_openness: Openness (0 closed to 1 open) the object is reset to at the start of an episode.
            episode_length_s: Episode time limit in seconds.
            task_description: Natural-language task description; derived from the object name when None.
        """
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
        """Return a success termination that triggers when the openable object is closed."""
        params = {}
        if self.target_joint_percentage_threshold is not None:
            params["threshold"] = self.target_joint_percentage_threshold
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name="close_door",
                    predicate_sequence=[partial(self.openable_object.is_closed, **params)],
                )
            ],
        )

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        """Return the Isaac Lab Mimic environment configuration for rotating the door.

        Args:
            arm_mode: Which arm or arms the Mimic subtasks are generated for.
        """
        return RotateDoorMimicEnvCfg(
            arm_mode=arm_mode,
            openable_object_name=self.openable_object.name,
        )
