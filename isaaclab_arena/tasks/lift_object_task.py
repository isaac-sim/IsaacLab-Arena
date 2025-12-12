# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import CommandTermCfg, EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.tasks.observations import general_observations
from isaaclab_arena.tasks.rewards import general_rewards
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object


class LiftObjectTask(TaskBase):
    def __init__(
        self,
        lift_object: Asset,
        background_scene: Asset,
        minimum_height_to_lift: float = 0.04,
        episode_length_s: float = 5.0,
    ):
        super().__init__(episode_length_s=episode_length_s)
        self.lift_object = lift_object
        self.background_scene = background_scene
        self.minimum_height_to_lift = minimum_height_to_lift
        self.scene_config = None
        self.events_cfg = LiftObjectEventsCfg(lift_object=self.lift_object)
        self.termination_cfg = self.make_termination_cfg()

    def get_scene_cfg(self):
        return self.scene_config

    def get_observation_cfg(self):
        return self.observation_config

    def get_termination_cfg(self):
        return self.termination_cfg

    def make_termination_cfg(self):
        object_dropped = TerminationTermCfg(
            func=mdp_isaac_lab.root_height_below_minimum,
            params={
                "minimum_height": self.background_scene.object_min_z,
                "asset_cfg": SceneEntityCfg(self.lift_object.name),
            },
        )
        return LiftObjectTerminationsCfg(object_dropped=object_dropped)

    def get_events_cfg(self):
        return self.events_cfg

    def get_mimic_env_cfg(self, embodiment_name: str):
        raise NotImplementedError("Function not implemented yet.")

    def get_prompt(self):
        raise NotImplementedError("Function not implemented yet.")

    def get_metrics(self) -> list[MetricBase]:
        return []

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.lift_object,
            offset=np.array([-1.5, -1.5, 1.5]),
        )


@configclass
class LiftObjectEventsCfg:
    """Configuration for Lift Object."""

    reset_lift_object_pose: EventTermCfg = MISSING

    def __init__(self, lift_object: Asset):
        self.reset_lift_object_pose = EventTermCfg(
            func=mdp_isaac_lab.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.1, 0.1),
                    "y": (-0.25, 0.25),
                    "z": (0.0, 0.0),
                },
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg(lift_object.name),
            },
        )


@configclass
class LiftObjectTerminationsCfg:
    """Termination terms for the Lift Object task."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    object_dropped: TerminationTermCfg = MISSING


class LiftObjectTaskRL(LiftObjectTask):
    def __init__(
        self,
        lift_object: Asset,
        background_scene: Asset,
        embodiment: EmbodimentBase,
        minimum_height_to_lift: float = 0.04,
        episode_length_s: float = 5.0,
    ):
        super().__init__(
            lift_object=lift_object,
            background_scene=background_scene,
            minimum_height_to_lift=minimum_height_to_lift,
            episode_length_s=episode_length_s,
        )
        self.embodiment = embodiment

    def get_observation_cfg(self):
        return LiftObjectObservationsCfg(lift_object=self.lift_object)

    def get_rewards_cfg(self):
        return LiftObjectRewardCfg(lift_object=self.lift_object, minimum_height_to_lift=self.minimum_height_to_lift)

    def get_commands_cfg(self):
        return LiftObjectCommandsCfg(
            body_name=self.embodiment.get_rl_information()["body_name"], lift_object=self.lift_object
        )


@configclass
class LiftObjectObservationsCfg:
    """Observation specifications for the Lift Object task."""

    task_obs: ObsGroup = MISSING

    def __init__(self, lift_object: Asset):
        @configclass
        class TaskObsCfg(ObsGroup):
            """Observations for the Lift Object task."""

            target_object_position = ObsTerm(
                func=mdp_isaac_lab.generated_commands, params={"command_name": "object_pose"}
            )
            object_position = ObsTerm(
                func=general_observations.object_position_in_robot_root_frame,
                params={"object_cfg": SceneEntityCfg(lift_object.name)},
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        self.task_obs = TaskObsCfg()


@configclass
class LiftObjectCommandsCfg:
    """Commands for the Lift Object task."""

    object_pose: CommandTermCfg = MISSING

    def __init__(self, body_name: str, lift_object: Asset):
        initial_pose = lift_object.get_initial_pose()
        self.object_pose = mdp_isaac_lab.UniformPoseCommandCfg(
            asset_name="robot",
            body_name=body_name,
            resampling_time_range=(5.0, 5.0),
            debug_vis=True,
            ranges=mdp_isaac_lab.UniformPoseCommandCfg.Ranges(
                pos_x=(initial_pose.position_xyz[0] - 0.1, initial_pose.position_xyz[0] + 0.1),
                pos_y=(initial_pose.position_xyz[1] - 0.25, initial_pose.position_xyz[1] + 0.25),
                pos_z=(initial_pose.position_xyz[2] + 0.2, initial_pose.position_xyz[2] + 0.4),
                roll=(0.0, 0.0),
                pitch=(0.0, 0.0),
                yaw=(0.0, 0.0),
            ),
        )


@configclass
class LiftObjectRewardCfg:
    """Reward terms for the Lift Object task."""

    reaching_object: RewardTermCfg = MISSING
    lifting_object: RewardTermCfg = MISSING
    object_goal_tracking: RewardTermCfg = MISSING
    object_goal_tracking_fine_grained: RewardTermCfg = MISSING

    def __init__(self, lift_object: Asset, minimum_height_to_lift: float):
        self.reaching_object = RewardTermCfg(
            func=general_rewards.object_ee_distance,
            params={
                "std": 0.1,
                "object_cfg": SceneEntityCfg(lift_object.name),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
            weight=1.0,
        )
        self.lifting_object = RewardTermCfg(
            func=general_rewards.object_is_lifted,
            params={
                "object_cfg": SceneEntityCfg(lift_object.name),
                "minimal_height": minimum_height_to_lift,
            },
            weight=15.0,
        )
        self.object_goal_tracking = RewardTermCfg(
            func=general_rewards.object_goal_distance,
            params={
                "std": 0.3,
                "minimal_height": minimum_height_to_lift,
                "command_name": "object_pose",
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg(lift_object.name),
            },
            weight=16.0,
        )
        self.object_goal_tracking_fine_grained = RewardTermCfg(
            func=general_rewards.object_goal_distance,
            params={
                "std": 0.05,
                "minimal_height": minimum_height_to_lift,
                "command_name": "object_pose",
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg(lift_object.name),
            },
            weight=5.0,
        )
