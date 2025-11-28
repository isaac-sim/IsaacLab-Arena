# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from dataclasses import MISSING
from typing import Any

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import CommandTermCfg, EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.observations.general_observation import object_position_in_world_frame
from isaaclab_arena.tasks.rewards import general_rewards
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.terminations import object_lifted
from isaaclab_arena.terms.events import set_object_pose
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object


class LiftObjectTask(TaskBase):
    def __init__(
        self,
        lift_object: Asset,
        background_scene: Asset,
        minimum_height_to_lift: float = 1.5,
        maximum_height_to_lift: float = 2,
        episode_length_s: float | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s)
        self.lift_object = lift_object
        self.background_scene = background_scene
        self.minimum_height_to_lift = minimum_height_to_lift
        self.maximum_height_to_lift = maximum_height_to_lift
        self.scene_config = None
        self.events_cfg = LiftObjectEventsCfg(lift_object=self.lift_object)
        self.termination_cfg = self.make_termination_cfg()
        self.observation_config = LiftObjectObservationsCfg(lift_object_name=self.lift_object.name)

    def get_scene_cfg(self):
        return self.scene_config

    def get_observation_cfg(self):
        return self.observation_config

    def get_termination_cfg(self):
        return self.termination_cfg

    def make_termination_cfg(self):
        success = TerminationTermCfg(
            func=object_lifted,
            params={
                "object_cfg": SceneEntityCfg(self.lift_object.name),
                "maximum_height": self.maximum_height_to_lift,
            },
        )
        object_dropped = TerminationTermCfg(
            func=mdp_isaac_lab.root_height_below_minimum,
            params={
                "minimum_height": self.background_scene.object_min_z,
                "asset_cfg": SceneEntityCfg(self.lift_object.name),
            },
        )
        return LiftObjectTerminationsCfg(success=success, object_dropped=object_dropped)

    def get_events_cfg(self):
        return self.events_cfg

    def get_mimic_env_cfg(self, embodiment_name: str):
        raise NotImplementedError("Function not implemented yet.")

    def get_prompt(self):
        raise NotImplementedError("Function not implemented yet.")

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

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
        initial_pose = lift_object.get_initial_pose()
        if initial_pose is not None:
            self.reset_lift_object_pose = EventTermCfg(
                func=set_object_pose,
                mode="reset",
                params={
                    "pose": initial_pose,
                    "asset_cfg": SceneEntityCfg(lift_object.name),
                },
            )
        else:
            print(f"Lift object {lift_object.name} has no initial pose. Not setting reset lift object pose event.")
            self.reset_lift_object_pose = None


@configclass
class LiftObjectTerminationsCfg:
    """Termination terms for the Lift Object task."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING
    object_dropped: TerminationTermCfg = MISSING


@configclass
class LiftObjectObservationsCfg:
    """Observation specifications for the Lift Object task."""

    task_obs: ObsGroup = MISSING

    def __init__(self, lift_object_name: str):

        class TaskObsCfg(ObsGroup):
            """Observations for the Lift Object task."""

            object_position = ObsTerm(
                func=object_position_in_world_frame, params={"asset_cfg": SceneEntityCfg(lift_object_name)}
            )

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = False

        self.task_obs = TaskObsCfg()


class LiftObjectTaskRL(LiftObjectTask):
    def __init__(
        self,
        lift_object: Asset,
        background_scene: Asset,
        embodiment_information: dict[str, Any],
        minimum_height_to_lift: float = 0.04,
        maximum_height_to_lift: float = 0.1,
        episode_length_s: float | None = None,
    ):
        super().__init__(
            lift_object=lift_object,
            background_scene=background_scene,
            minimum_height_to_lift=minimum_height_to_lift,
            maximum_height_to_lift=maximum_height_to_lift,
            episode_length_s=episode_length_s,
        )
        self.embodiment_information = embodiment_information

    def get_scene_cfg(self):
        return LiftObjectSceneCfg(embodiment_information=self.embodiment_information)

    def get_rewards_cfg(self):
        return LiftObjectRewardCfg(lift_object=self.lift_object, minimum_height_to_lift=self.minimum_height_to_lift)

    def get_commands_cfg(self):
        return LiftObjectCommandsCfg(embodiment_information=self.embodiment_information)


@configclass
class LiftObjectCommandsCfg:
    """Commands for the Lift Object task."""

    object_pose: CommandTermCfg = MISSING

    def __init__(self, embodiment_information: dict[str, Any]):

        self.object_pose = mdp_isaac_lab.UniformPoseCommandCfg(
            asset_name="robot",
            body_name=embodiment_information["body_name"],
            resampling_time_range=(5.0, 5.0),
            debug_vis=True,
            ranges=mdp_isaac_lab.UniformPoseCommandCfg.Ranges(
                pos_x=(0.4, 0.6),
                pos_y=(-0.25, 0.25),
                pos_z=(0.25, 0.5),
                roll=(0.0, 0.0),
                pitch=(0.0, 0.0),
                yaw=(0.0, 0.0),
            ),
        )


@configclass
class LiftObjectSceneCfg:
    """Configuration for Lift Object."""

    embodiment_end_effector_frame: SceneEntityCfg = MISSING

    def __init__(self, embodiment_information: dict[str, Any]):

        self.embodiment_end_effector_frame = FrameTransformerCfg(
            prim_path=embodiment_information["eef_prim_path"],
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=embodiment_information["target_prim_path"],
                    name=embodiment_information["target_frame_name"],
                    offset=OffsetCfg(
                        pos=embodiment_information["target_offset"],
                    ),
                ),
            ],
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
                "ee_frame_cfg": SceneEntityCfg("embodiment_end_effector_frame"),
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
