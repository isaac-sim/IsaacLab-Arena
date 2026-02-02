# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import CommandTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.observations import observations
from isaaclab_arena.tasks.rewards import lift_object_rewards, rewards
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.terminations import lift_object_il_success, lift_object_rl_success
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object
from isaaclab_arena.utils.pose import PoseRange


class LiftObjectTask(TaskBase):
    def __init__(
        self,
        lift_object: Asset,
        background_scene: Asset,
        episode_length_s: float = 5.0,
        goal_position_delta_xyz: tuple[float, float, float] = (0.0, 0.0, 0.3),
        goal_position_tolerance: float = 0.05,
    ):
        """Initialize the Lift Object task.

        Args:
            lift_object: The object to lift.
            background_scene: The background scene (table, etc.).
            episode_length_s: Episode length in seconds.
            goal_position_delta_xyz: Goal position delta [dx, dy, dz] relative to initial pose (m).
                Default: (0, 0, 0.3) = 30cm above initial position.
            goal_position_tolerance: Position tolerance for success (m).
        """
        super().__init__(episode_length_s=episode_length_s)
        self.lift_object = lift_object
        self.background_scene = background_scene

        # Compute goal position from object's initial pose + delta
        initial_pose = lift_object.get_initial_pose()
        if isinstance(initial_pose, PoseRange):
            initial_pose = initial_pose.get_midpoint()

        # Store goal pose for success termination (IL/teleoperation uses fixed goal)
        self.goal_position_xyz = (
            initial_pose.position_xyz[0] + goal_position_delta_xyz[0],
            initial_pose.position_xyz[1] + goal_position_delta_xyz[1],
            initial_pose.position_xyz[2] + goal_position_delta_xyz[2],
        )
        self.goal_position_tolerance = goal_position_tolerance

        self.scene_config = None
        self.events_cfg = None
        self.termination_cfg = self.make_il_termination_cfg()

    def get_scene_cfg(self):
        return self.scene_config

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_events_cfg(self):
        return self.events_cfg

    def make_il_termination_cfg(self):
        """Create termination configuration.

        Args:
            rl_training: If True, disables success termination (for RL training).
            use_command_goal: If True, uses goal from command manager (for RL evaluation).
        """
        object_dropped = TerminationTermCfg(
            func=mdp_isaac_lab.root_height_below_minimum,
            params={
                "minimum_height": self.background_scene.object_min_z,
                "asset_cfg": SceneEntityCfg(self.lift_object.name),
            },
        )

        # Use dynamic success termination
        success = TerminationTermCfg(
            func=lift_object_il_success,
            params={
                "object_cfg": SceneEntityCfg(self.lift_object.name),
                "goal_position": self.goal_position_xyz,
                "position_tolerance": self.goal_position_tolerance,
            },
        )

        return LiftObjectTerminationsCfg(object_dropped=object_dropped, success=success)

    def get_mimic_env_cfg(self, embodiment_name: str):
        raise NotImplementedError("Function not implemented yet.")

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.lift_object,
            offset=np.array([-1.5, -1.5, 1.5]),
        )


@configclass
class LiftObjectTerminationsCfg:
    """Termination terms for the Lift Object task.

    Note: success is optional and can be None for RL tasks where early
    termination on success is not desired.
    """

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    object_dropped: TerminationTermCfg = MISSING
    success: TerminationTermCfg = MISSING


class LiftObjectTaskRL(LiftObjectTask):
    def __init__(
        self,
        lift_object: Asset,
        background_scene: Asset,
        embodiment: EmbodimentBase,
        minimum_height_to_lift: float = 0.04,
        episode_length_s: float = 5.0,
        rl_training_mode: bool = True,
        target_x_delta: tuple[float, float] = (-0.1, 0.1),
        target_y_delta: tuple[float, float] = (-0.25, 0.25),
        target_z_delta: tuple[float, float] = (0.2, 0.4),
    ):
        """Initialize the Lift Object RL task.

        Args:
            lift_object: The object to lift.
            background_scene: The background scene (table, etc.).
            embodiment: The robot embodiment.
            minimum_height_to_lift: Minimum height to consider the object lifted (m).
            episode_length_s: Episode length in seconds.
            target_x_delta: Target range deltas for x [min_delta, max_delta] relative to initial pose (m).
            target_y_delta: Target range deltas for y [min_delta, max_delta] relative to initial pose (m).
            target_z_delta: Target range deltas for z [min_delta, max_delta] relative to initial pose (m).
            rl_training_mode: If True, disables success termination. Set to False for evaluation.
        """
        self.rl_training_mode = rl_training_mode

        self.minimum_height_to_lift = minimum_height_to_lift
        # Get object's initial pose to compute absolute target ranges
        initial_pose = lift_object.get_initial_pose()
        if isinstance(initial_pose, PoseRange):
            initial_pose = initial_pose.get_midpoint()

        # Compute absolute target ranges from deltas
        self.target_x_range = (
            initial_pose.position_xyz[0] + target_x_delta[0],
            initial_pose.position_xyz[0] + target_x_delta[1],
        )
        self.target_y_range = (
            initial_pose.position_xyz[1] + target_y_delta[0],
            initial_pose.position_xyz[1] + target_y_delta[1],
        )
        self.target_z_range = (
            initial_pose.position_xyz[2] + target_z_delta[0],
            initial_pose.position_xyz[2] + target_z_delta[1],
        )

        # Call parent with dummy delta (will be overridden by termination config anyway)
        super().__init__(
            lift_object=lift_object,
            background_scene=background_scene,
            episode_length_s=episode_length_s,
            goal_position_delta_xyz=(0, 0, 0),  # Dummy, termination will be overridden
            goal_position_tolerance=0.05,  # Dummy, termination will be overridden
        )

        self.embodiment = embodiment
        self.observation_cfg = LiftObjectObservationsCfg(
            lift_object=self.lift_object, robot_name=self.embodiment.get_embodiment_name_in_scene()
        )
        self.commands_cfg = LiftObjectCommandsCfg(
            asset_name=self.embodiment.get_embodiment_name_in_scene(),
            body_name=self.embodiment.get_command_body_name(),
            lift_object=self.lift_object,
            target_x_range=self.target_x_range,
            target_y_range=self.target_y_range,
            target_z_range=self.target_z_range,
        )
        self.rewards_cfg = LiftObjectRewardCfg(
            lift_object=self.lift_object,
            minimum_height_to_lift=self.minimum_height_to_lift,
            robot_name=self.embodiment.get_embodiment_name_in_scene(),
            ee_frame_name=self.embodiment.get_ee_frame_name(self.embodiment.get_arm_mode()),
        )

        # Override termination config with RL training mode
        self.termination_cfg = self.make_rl_termination_cfg()

    def make_rl_termination_cfg(self):
        """Create termination configuration for RL training mode."""
        object_dropped = TerminationTermCfg(
            func=mdp_isaac_lab.root_height_below_minimum,
            params={
                "minimum_height": self.background_scene.object_min_z,
                "asset_cfg": SceneEntityCfg(self.lift_object.name),
            },
        )

        # Use dynamic success termination
        success = TerminationTermCfg(
            func=lift_object_rl_success,
            params={
                "object_cfg": SceneEntityCfg(self.lift_object.name),
                "rl_training": self.rl_training_mode,
                "command_name": "object_pose",
                "position_tolerance": self.goal_position_tolerance,
            },
        )

        return LiftObjectTerminationsCfg(object_dropped=object_dropped, success=success)

    def get_observation_cfg(self):
        return self.observation_cfg

    def get_rewards_cfg(self):
        return self.rewards_cfg

    def get_commands_cfg(self):
        return self.commands_cfg

    def get_termination_cfg(self):
        return self.termination_cfg


@configclass
class LiftObjectObservationsCfg:
    """Observation specifications for the Lift Object task."""

    task_obs: ObsGroup = MISSING

    def __init__(self, lift_object: Asset, robot_name: str):
        @configclass
        class TaskObsCfg(ObsGroup):
            """Observations for the Lift Object task."""

            target_object_position = ObsTerm(
                func=mdp_isaac_lab.generated_commands, params={"command_name": "object_pose"}
            )
            object_position = ObsTerm(
                func=observations.object_position_in_frame,
                params={"root_frame_cfg": SceneEntityCfg(robot_name), "object_cfg": SceneEntityCfg(lift_object.name)},
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        self.task_obs = TaskObsCfg()


@configclass
class LiftObjectCommandsCfg:
    """Commands for the Lift Object task."""

    object_pose: CommandTermCfg = MISSING

    def __init__(
        self,
        asset_name: str,
        body_name: str,
        lift_object: Asset,
        target_x_range: tuple[float, float],
        target_y_range: tuple[float, float],
        target_z_range: tuple[float, float],
    ):
        self.object_pose = mdp_isaac_lab.UniformPoseCommandCfg(
            asset_name=asset_name,
            body_name=body_name,
            resampling_time_range=(5.0, 5.0),
            debug_vis=True,
            ranges=mdp_isaac_lab.UniformPoseCommandCfg.Ranges(
                pos_x=target_x_range,
                pos_y=target_y_range,
                pos_z=target_z_range,
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

    def __init__(self, lift_object: Asset, minimum_height_to_lift: float, robot_name: str, ee_frame_name: str):
        self.reaching_object = RewardTermCfg(
            func=rewards.object_ee_distance,
            params={
                "std": 0.1,
                "object_cfg": SceneEntityCfg(lift_object.name),
                "ee_frame_cfg": SceneEntityCfg(ee_frame_name),
            },
            weight=1.0,
        )
        self.lifting_object = RewardTermCfg(
            func=lift_object_rewards.object_is_lifted,
            params={
                "object_cfg": SceneEntityCfg(lift_object.name),
                "minimal_height": minimum_height_to_lift,
            },
            weight=15.0,
        )
        self.object_goal_tracking = RewardTermCfg(
            func=lift_object_rewards.object_goal_distance,
            params={
                "std": 0.3,
                "minimal_height": minimum_height_to_lift,
                "command_name": "object_pose",
                "robot_cfg": SceneEntityCfg(robot_name),
                "object_cfg": SceneEntityCfg(lift_object.name),
            },
            weight=16.0,
        )
        self.object_goal_tracking_fine_grained = RewardTermCfg(
            func=lift_object_rewards.object_goal_distance,
            params={
                "std": 0.05,
                "minimal_height": minimum_height_to_lift,
                "command_name": "object_pose",
                "robot_cfg": SceneEntityCfg(robot_name),
                "object_cfg": SceneEntityCfg(lift_object.name),
            },
            weight=5.0,
        )
