# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import MISSING
from functools import partial

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.assets.register import agent_ready, register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
from isaaclab_arena.tasks.common.open_close_door_mimic import RotateDoorMimicEnvCfg
from isaaclab_arena.tasks.rotate_revolute_joint_task import RotateRevoluteJointTask

DEFAULT_OPENNESS_THRESHOLD = 0.5
"""Fraction of the joint range the door must travel past for the task to succeed."""

DEFAULT_MOVED_OPENNESS_DELTA = 0.05
"""How far the openness must change from the reset openness to count as having moved the door."""


@agent_ready
@register_task
class OpenDoorTask(RotateRevoluteJointTask):
    """Open-door task. Success fires once the door's joint travels past ``openness_threshold``.

    Progress is tracked with a single two-step objective: the door moving at all, then the door
    reaching the success threshold. See ``get_progress_objectives``.
    """

    def __init__(
        self,
        openable_object: Openable,
        openness_threshold: float | None = DEFAULT_OPENNESS_THRESHOLD,
        reset_openness: float | None = 0.0,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        moved_openness_delta: float = DEFAULT_MOVED_OPENNESS_DELTA,
    ):
        """Initializes the open-door task.

        Args:
            openable_object: The door-like object to open.
            openness_threshold: Fraction of the joint range the door must travel past to succeed.
                None falls back to the object's own ``openable_threshold``.
            reset_openness: The openness the door is reset to at the start of each episode.
            episode_length_s: The episode length in seconds.
            task_description: The language instruction for the task.
            moved_openness_delta: How far the openness must change from ``reset_openness`` before the
                door counts as having moved, for progress tracking.
        """
        self.moved_openness_delta = moved_openness_delta
        super().__init__(
            openable_object=openable_object,
            target_joint_percentage_threshold=openness_threshold,
            reset_joint_percentage=reset_openness,
            episode_length_s=episode_length_s,
            task_description=task_description,
        )

        self.termination_cfg = self.make_termination_cfg()
        self.task_description = (
            f"Reach out to the {openable_object.name} and open it." if task_description is None else task_description
        )

    def make_termination_cfg(self):
        params = {}
        if self.target_joint_percentage_threshold is not None:
            params["threshold"] = self.target_joint_percentage_threshold
        success = TerminationTermCfg(
            func=self.openable_object.is_open,
            params=params,
        )
        return TerminationsCfg(success=success)

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_progress_objectives(self) -> list[ProgressObjective]:
        """Returns a single objective whose chain is: the door moved at all, then the door is open.

        The chain is sequential, so a partially opened door scores 0.5 and a door past the success
        threshold scores 1.0 and completes the objective.
        """
        is_open_params = {}
        if self.target_joint_percentage_threshold is not None:
            is_open_params["threshold"] = self.target_joint_percentage_threshold
        reset_openness = 0.0 if self.reset_joint_percentage is None else self.reset_joint_percentage
        return [
            ProgressObjective(
                name="open_door",
                predicate_groups=[
                    partial(
                        self.openable_object.has_moved,
                        reference_percentage=reset_openness,
                        min_change=self.moved_openness_delta,
                    ),
                    partial(self.openable_object.is_open, **is_open_params),
                ],
                description=f"Move the {self.openable_object.name} door, then open it past the success threshold.",
            ),
        ]

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        return RotateDoorMimicEnvCfg(
            arm_mode=arm_mode,
            openable_object_name=self.openable_object.name,
        )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)

    # Dependent on the openable object, so this is passed in from the task at
    # construction time.
    success: TerminationTermCfg = MISSING
