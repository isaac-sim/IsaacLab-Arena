# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Base task for rotating the revolute joint of an openable object to a target joint percentage."""

import numpy as np
from dataclasses import MISSING

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.revolute_joint_moved_rate import RevoluteJointMovedRateMetric
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.tasks.task_transition import TaskTransition
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object


@register_task
class RotateRevoluteJointTask(TaskBase):
    """Task that rotates the revolute joint of an openable object.

    Subclasses define the success termination.
    """

    def __init__(
        self,
        openable_object: Openable,
        target_joint_percentage_threshold: float,
        reset_joint_percentage: float,
        episode_length_s: float | None = None,
        task_description: str | None = None,
    ):
        """Initialize the task.

        Args:
            openable_object: Object whose revolute joint is rotated.
            target_joint_percentage_threshold: Target joint position as a fraction of the joint range.
            reset_joint_percentage: Joint position, as a fraction of the joint range, that the joint is reset to
                at the start of an episode.
            episode_length_s: Episode time limit in seconds.
            task_description: Natural-language task description; derived from the object name and target when None.
        """
        super().__init__(episode_length_s=episode_length_s)
        self.openable_object = openable_object
        self.target_joint_percentage_threshold = target_joint_percentage_threshold
        self.reset_joint_percentage = reset_joint_percentage
        self.min_openness_change = 0.05
        """Openness change, as a fraction of the joint range, past which the joint counts as having moved."""
        self.task_description = (
            f"Rotate the {self.openable_object.name} joint to the target {target_joint_percentage_threshold} joint"
            " percentage."
            if task_description is None
            else task_description
        )
        self.events_cfg = RotateRevoluteJointEventCfg(
            self.openable_object, reset_openable_object_revolute_joint_percentage=self.reset_joint_percentage
        )
        self.scene_config = None
        self.mimic_env_cfg = None

    def get_scene_cfg(self):
        """Return None; the task adds nothing to the scene."""
        return self.scene_config

    def get_events_cfg(self):
        """Return the reset event that sets the joint to its reset position."""
        return self.events_cfg

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        """Not implemented; always raises NotImplementedError.

        Args:
            arm_mode: Which arm or arms the Mimic subtasks are generated for.
        """
        raise NotImplementedError("Function {self.get_mimic_env_cfg.__name__} not implemented yet.")

    def get_termination_cfg(self) -> TaskTerminationCfg:
        """Not implemented; always raises NotImplementedError."""
        raise NotImplementedError("Function {self.get_termination_cfg.__name__} not implemented yet.")

    def get_metrics(self) -> list[MetricBase]:
        """Return the success rate and the rate at which the joint is moved."""
        return [
            SuccessRateMetric(),
            RevoluteJointMovedRateMetric(
                self.openable_object,
                reset_joint_percentage=self.reset_joint_percentage,
                joint_percentage_delta_threshold=self.min_openness_change,
            ),
        ]

    def get_viewer_cfg(self) -> ViewerCfg:
        """Return a viewer camera configuration that looks at the object."""
        return get_viewer_cfg_look_at_object(lookat_object=self.openable_object, offset=np.array([-1.3, -1.3, 1.3]))

    @classmethod
    def success_state_transition(cls, openable_object: str, **_) -> TaskTransition:
        """Success is the openable object's joint state change.
        The env-graph spec has not yet defined joint openness, so no effects are
        emitted today — only the subject node (the openable object) is recorded.
        # TODO(xinjieyao, 2026.06.02): allow env graph spec to define node state change

        Args:
            openable_object: Name of the object whose joint is rotated.
        """
        return TaskTransition(subject=openable_object)


@configclass
class RotateRevoluteJointEventCfg:
    """Reset events for the rotate-revolute-joint task."""

    reset_openable_object_revolute_joint_percentage: EventTermCfg = MISSING

    def __init__(self, openable_object: Openable, reset_openable_object_revolute_joint_percentage: float | None):
        """Create the reset event that rotates the joint to its reset position.

        Args:
            openable_object: Object whose revolute joint is reset.
            reset_openable_object_revolute_joint_percentage: Joint position, as a fraction of the joint range,
                that the joint is reset to; None selects the affordance's own value.
        """
        assert isinstance(openable_object, Openable), "Object pose must be an instance of Openable"
        params = {}
        if reset_openable_object_revolute_joint_percentage is not None:
            params["percentage"] = reset_openable_object_revolute_joint_percentage
        self.reset_openable_object_revolute_joint_percentage = EventTermCfg(
            func=openable_object.rotate_revolute_joint,
            mode="reset",
            params=params,
        )
