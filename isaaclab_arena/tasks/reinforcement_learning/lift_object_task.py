# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from dataclasses import MISSING
from functools import partial
from typing import TYPE_CHECKING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import CommandTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import combine_frame_transforms

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.observations import observations
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object
from isaaclab_arena.utils.pose import PoseRange

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv


@register_task
class LiftObjectTaskRL(TaskBase):
    """Lift an object to a target position sampled each episode, for RL training and evaluation."""

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
        goal_position_tolerance: float = 0.05,
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
            goal_position_tolerance: Distance from the commanded goal position within which the lift succeeds (m).
        """
        super().__init__(episode_length_s=episode_length_s)
        self.lift_object = lift_object
        self.background_scene = background_scene
        self.embodiment = embodiment
        self.rl_training_mode = rl_training_mode
        self.minimum_height_to_lift = minimum_height_to_lift
        self.goal_position_tolerance = goal_position_tolerance
        self.scene_config = None
        self.events_cfg = None

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

        self.observation_cfg = LiftObjectObservationsCfg(
            lift_object=self.lift_object, robot_name=self.embodiment.get_scene_key()
        )
        self.commands_cfg = LiftObjectCommandsCfg(
            asset_name=self.embodiment.get_scene_key(),
            body_name=self.embodiment.get_command_body_name(),
            lift_object=self.lift_object,
            target_x_range=self.target_x_range,
            target_y_range=self.target_y_range,
            target_z_range=self.target_z_range,
        )
        self.rewards_cfg = LiftObjectRewardCfg(
            lift_object=self.lift_object,
            minimum_height_to_lift=self.minimum_height_to_lift,
            robot_name=self.embodiment.get_scene_key(),
            ee_frame_name=self.embodiment.get_ee_frame_name(self.embodiment.get_arm_mode()),
        )

    def get_scene_cfg(self):
        return self.scene_config

    def get_events_cfg(self):
        return self.events_cfg

    def get_termination_cfg(self) -> TaskTerminationCfg:
        object_dropped = TerminationTermCfg(
            func=mdp_isaac_lab.root_height_below_minimum,
            params={
                "minimum_height": self.background_scene.object_min_z,
                "asset_cfg": SceneEntityCfg(self.lift_object.name),
            },
        )
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name="lift_object",
                    predicate_sequence=[
                        partial(
                            object_at_commanded_position,
                            object_cfg=SceneEntityCfg(self.lift_object.name),
                            robot_cfg=SceneEntityCfg(self.embodiment.get_scene_key()),
                            rl_training=self.rl_training_mode,
                            command_name="object_pose",
                            position_tolerance=self.goal_position_tolerance,
                        )
                    ],
                )
            ],
            failures={"object_dropped": object_dropped},
        )

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError("Function not implemented yet.")

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.lift_object,
            offset=np.array([-1.5, -1.5, 1.5]),
        )

    def get_observation_cfg(self):
        return self.observation_cfg

    def get_rewards_cfg(self):
        return self.rewards_cfg

    def get_commands_cfg(self):
        return self.commands_cfg


@configclass
class LiftObjectObservationsCfg:
    """Observation specifications for the Lift Object task."""

    task_obs: ObsGroup = MISSING

    def __init__(self, lift_object: Asset, robot_name: str):
        @configclass
        class TaskObsCfg(ObsGroup):
            target_object_position = ObsTerm(
                func=mdp_isaac_lab.generated_commands, params={"command_name": "object_pose"}
            )
            object_position = ObsTerm(
                func=observations.object_position_in_frame,
                params={
                    "root_frame_cfg": SceneEntityCfg(robot_name),
                    "object_cfg": SceneEntityCfg(lift_object.name),
                },
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
            func=object_ee_distance,
            params={
                "std": 0.1,
                "object_cfg": SceneEntityCfg(lift_object.name),
                "ee_frame_cfg": SceneEntityCfg(ee_frame_name),
            },
            weight=1.0,
        )
        self.lifting_object = RewardTermCfg(
            func=object_is_lifted,
            params={
                "object_cfg": SceneEntityCfg(lift_object.name),
                "minimal_height": minimum_height_to_lift,
            },
            weight=15.0,
        )
        self.object_goal_tracking = RewardTermCfg(
            func=object_goal_distance,
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
            func=object_goal_distance,
            params={
                "std": 0.05,
                "minimal_height": minimum_height_to_lift,
                "command_name": "object_pose",
                "robot_cfg": SceneEntityCfg(robot_name),
                "object_cfg": SceneEntityCfg(lift_object.name),
            },
            weight=5.0,
        )


def _commanded_position_w(
    env: "IsaacLabArenaManagerBasedRLEnv", robot_cfg: SceneEntityCfg, command_name: str
) -> torch.Tensor:
    """Return the goal position of a pose command, transformed from the robot base frame into the world frame.

    Args:
        env: The RL environment instance.
        robot_cfg: The robot whose base frame the command is expressed in.
        command_name: The name of the pose command.

    Returns:
        The goal positions t_W_G, shape (num_envs, 3).
    """
    T_W_B = env.arena_world.get_pose_w(robot_cfg.name)
    t_B_G = env.command_manager.get_command(command_name)[:, :3]
    t_W_G, _ = combine_frame_transforms(T_W_B[:, :3], T_W_B[:, 3:], t_B_G)
    return t_W_G


def object_at_commanded_position(
    env: "IsaacLabArenaManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    rl_training: bool = False,
    command_name: str = "object_pose",
    position_tolerance: float = 0.05,
) -> torch.Tensor:
    """Check whether the object is within a distance tolerance of the commanded goal position.

    The goal is read from the command manager in the robot base frame. During RL training, success
    never terminates the episode, so this returns False for every environment.

    Args:
        env: The RL environment instance.
        object_cfg: The configuration of the object to track.
        robot_cfg: The robot configuration (needed to transform the goal to world frame).
        rl_training: If True, always returns False (disables success termination for RL training).
        command_name: The name of the command that holds the goal position.
        position_tolerance: Distance tolerance for success (m).

    Returns:
        A boolean tensor of shape (num_envs,) indicating success.
    """
    if rl_training:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    t_W_O = env.arena_world.get_position_w(object_cfg.name)
    t_W_G = _commanded_position_w(env, robot_cfg, command_name)
    distance = torch.linalg.norm(t_W_G - t_W_O, dim=1)
    return distance < position_tolerance


def object_ee_distance(
    env: "IsaacLabArenaManagerBasedRLEnv",
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    t_W_O = env.arena_world.get_position_w(object_cfg.name)

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    t_W_E = ee_frame.data.target_pos_w.torch[..., 0, :]
    distance_to_object = torch.norm(t_W_O - t_W_E, dim=1)

    return 1 - torch.tanh(distance_to_object / std)


def object_is_lifted(
    env: "IsaacLabArenaManagerBasedRLEnv",
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object_height_w = env.arena_world.get_position_w(object_cfg.name)[:, 2]
    return torch.where(object_height_w > minimal_height, 1.0, 0.0)


def object_goal_distance(
    env: "IsaacLabArenaManagerBasedRLEnv",
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    t_W_O = env.arena_world.get_position_w(object_cfg.name)
    t_W_G = _commanded_position_w(env, robot_cfg, command_name)
    distance_to_goal = torch.norm(t_W_G - t_W_O, dim=1)
    object_is_above_minimal_height = t_W_O[:, 2] > minimal_height
    return object_is_above_minimal_height * (1 - torch.tanh(distance_to_goal / std))
