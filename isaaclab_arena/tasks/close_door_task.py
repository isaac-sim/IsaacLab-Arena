# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.embodiments.common.mimic_arm_mode import MimicArmMode
from isaaclab_arena.tasks.common.open_close_door_mimic import RotateDoorMimicEnvCfg
from isaaclab_arena.tasks.rotate_revolute_joint_task import RotateRevoluteJointTask


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

        self.termination_cfg = self.make_termination_cfg()
        self.task_description = (
            f"Reach out to the {openable_object.name} and close it." if task_description is None else task_description
        )

    def make_termination_cfg(self):
        params = {}
        if self.target_joint_percentage_threshold is not None:
            params["threshold"] = self.target_joint_percentage_threshold
        success = TerminationTermCfg(
            func=self.openable_object.is_closed,
            params=params,
        )
        return TerminationsCfg(success=success)

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_mimic_env_cfg(self, arm_mode: MimicArmMode):
        return RotateDoorMimicEnvCfg(
            arm_mode=arm_mode,
            openable_object_name=self.openable_object.name,
        )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)

    # Dependent on the openable object, so this is passed in from the task at
    # construction time.
    success: TerminationTermCfg = MISSING
