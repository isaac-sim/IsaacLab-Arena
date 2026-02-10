# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import MISSING

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.managers import RewardTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.tasks.common.open_close_door_mimic import RotateDoorMimicEnvCfg
from isaaclab_arena.tasks.rotate_revolute_joint_task import RotateRevoluteJointTask
from isaaclab_arena.tasks.rewards import openable_rewards

class OpenDoorTask(RotateRevoluteJointTask):
    def __init__(
        self,
        openable_object: Openable,
        openness_threshold: float | None = None,
        reset_openness: float | None = 0.0,
        episode_length_s: float | None = None,
        task_description: str | None = None,
    ):
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

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
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

class OpenDoorTaskRL(OpenDoorTask):
    def __init__(
        self,
        openable_object: Openable,
        openness_threshold: float | None = None,
        reset_openness: float | None = 0.0,
        episode_length_s: float | None = None,
        task_description: str | None = None,
    ):
        super().__init__(
            openable_object=openable_object,
            openness_threshold=openness_threshold,
            reset_openness=reset_openness,
            episode_length_s=episode_length_s,
            task_description=task_description,
        )
        self.openable_object = openable_object
        self.observation_cfg = OpenDoorObservationsCfg(
            openable_object=self.openable_object,
        )
        self.commands_cfg = None
        self.rewards_cfg = OpenDoorRewardCfg(
            openable_object=self.openable_object
        )

    def get_observation_cfg(self):
        return self.observation_cfg

    def get_rewards_cfg(self):
        return self.rewards_cfg

@configclass
class OpenDoorObservationsCfg:
    """Observation specifications for the Open Object task."""

    task_obs: ObsGroup = MISSING

    def __init__(self, openable_object: Openable):
        @configclass
        class TaskObsCfg(ObsGroup):
            """Observations for the Open Object task."""

            door_openness = ObsTerm(
                func=lambda env: openable_object.get_openness(env).unsqueeze(-1)
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        self.task_obs = TaskObsCfg()

@configclass
class OpenDoorRewardCfg:
    """Reward terms for the Open Object task."""

    def __init__(self, openable_object: Openable):
        self.reaching_object = RewardTermCfg(
            func=openable_rewards.openable_distance_to_target_openness,
            params={
                "openable_object": openable_object
            },
            weight=1.0,
        )