# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Franka volume-deformable lift task for policy evaluation."""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import CommandTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.lift_franka_soft import mdp as soft_lift_mdp
from isaaclab_tasks.manager_based.manipulation.lift_franka_soft.mdp.observations import (
    DeformableSampledPointsInRobotRootFrame,
)

from isaaclab_arena.assets.deformable_object import DeformableObject
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.metrics.deformable_goal_reached_rate import DeformableGoalReachedRateMetric
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.tasks.task_base import TaskBase


@configclass
class FrankaSoftLiftCommandsCfg:
    """Commanded deformable goal pose."""

    deformable_pose: CommandTermCfg = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6),
            pos_y=(-0.25, 0.25),
            pos_z=(0.25, 0.5),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        goal_pose_visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/Command/goal_pose",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.03,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.1, 0.9, 0.2),
                        opacity=0.4,
                    ),
                ),
            },
        ),
    )


@configclass
class FrankaSoftLiftObservationsCfg:
    """Policy observations for the Franka soft-lift MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        deformable_sampled_points = ObsTerm(
            func=DeformableSampledPointsInRobotRootFrame,
            params={"asset_cfg": SceneEntityCfg("deformable"), "num_points": 20},
        )
        target_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "deformable_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class FrankaSoftLiftEventsCfg:
    """Reset events for the soft-lift task."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.9, 1.1), "velocity_range": (0.0, 0.0)},
    )

    reset_deformable = EventTerm(
        func=mdp.reset_nodal_state_uniform,
        mode="reset",
        params={
            "position_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("deformable"),
        },
    )


@configclass
class FrankaSoftLiftRewardsCfg:
    """Source-parity reward diagnostics for the deformable lift task."""

    reaching_deformable = RewTerm(
        func=soft_lift_mdp.deformable_ee_distance,
        params={"std": 0.1, "asset_cfg": SceneEntityCfg("deformable")},
        weight=5.0,
    )
    lifting_deformable = RewTerm(
        func=soft_lift_mdp.deformable_lifted,
        params={"minimal_height": 0.04, "asset_cfg": SceneEntityCfg("deformable")},
        weight=5.0,
    )
    deformable_goal_tracking = RewTerm(
        func=soft_lift_mdp.deformable_com_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.075,
            "command_name": "deformable_pose",
            "asset_cfg": SceneEntityCfg("deformable"),
        },
        weight=16.0,
    )
    deformable_goal_tracking_fine_grained = RewTerm(
        func=soft_lift_mdp.deformable_com_goal_distance,
        params={
            "std": 0.05,
            "minimal_height": 0.075,
            "command_name": "deformable_pose",
            "asset_cfg": SceneEntityCfg("deformable"),
        },
        weight=5.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-2)
    gripper_close = RewTerm(
        func=soft_lift_mdp.gripper_close_action,
        params={"action_name": "gripper_action"},
        weight=-1.0,
    )
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-2)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-4)


@configclass
class FrankaSoftLiftTerminationsCfg:
    """Time-out and source safety terminations; no success termination."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    deformable_outside_table = DoneTerm(
        func=soft_lift_mdp.deformable_outside_table_bounds,
        params={
            "x_bounds": (0.0, 1.0),
            "y_bounds": (-0.5, 0.5),
            "asset_cfg": SceneEntityCfg("deformable"),
        },
    )
    deformable_dropped = DoneTerm(
        func=soft_lift_mdp.deformable_com_below_minimum,
        params={"minimum_height": -0.1, "asset_cfg": SceneEntityCfg("deformable")},
    )
    ee_below_table = DoneTerm(
        func=soft_lift_mdp.ee_below_minimum,
        params={"minimum_height": 0.0, "ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )


@register_task
class FrankaSoftLiftTask(TaskBase):
    """Evaluation task for Franka lifting a volume deformable block."""

    def __init__(
        self,
        deformable: DeformableObject,
        table: Object,
        episode_length_s: float = 5.0,
    ):
        super().__init__(
            episode_length_s=episode_length_s,
            task_description="Lift the deformable block to the commanded target pose.",
        )
        self.deformable = deformable
        self.table = table
        self.commands_cfg = FrankaSoftLiftCommandsCfg()
        self.observations_cfg = FrankaSoftLiftObservationsCfg()
        self.events_cfg = FrankaSoftLiftEventsCfg()
        self.rewards_cfg = FrankaSoftLiftRewardsCfg()
        self.terminations_cfg = FrankaSoftLiftTerminationsCfg()

    def get_scene_cfg(self):
        return None

    def get_observation_cfg(self):
        return self.observations_cfg

    def get_commands_cfg(self):
        return self.commands_cfg

    def get_events_cfg(self):
        return self.events_cfg

    def get_rewards_cfg(self):
        return self.rewards_cfg

    def get_termination_cfg(self):
        return self.terminations_cfg

    def get_metrics(self) -> list[MetricBase]:
        return [DeformableGoalReachedRateMetric()]

    def get_mimic_env_cfg(self, arm_mode):
        raise NotImplementedError("Franka soft lift does not define a mimic workflow.")

    def get_viewer_cfg(self) -> ViewerCfg:
        viewer = ViewerCfg()
        viewer.origin_type = "asset_root"
        viewer.asset_name = "robot"
        viewer.env_index = 0
        viewer.eye = (1.25, -1.5, 0.75)
        viewer.resolution = (1920, 1080)
        return viewer
