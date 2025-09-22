# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import MISSING

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place.mdp as mdp
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLMimicEnv  # noqa: F401
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.sensors import CameraCfg  # noqa: F401
from isaaclab.utils import configclass

import isaac_arena.terms.transforms as transforms_terms
from isaac_arena.assets.register import register_asset
from isaac_arena.embodiments.embodiment_base import EmbodimentBase
from isaac_arena.embodiments.g1.mdp import wbc_events as wbc_events_mdp
from isaac_arena.embodiments.g1.mdp.actions.g1_decoupled_wbc_action_cfg import G1DecoupledWBCActionCfg
from isaac_arena.geometry.pose import Pose
from isaac_arena.isaaclab_utils.resets import reset_all_articulation_joints


@register_asset
class G1Embodiment(EmbodimentBase):
    """Embodiment for the G1 robot."""

    name = "g1"

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        # Configuration structs
        self.scene_config = G1SceneCfg()
        self.action_config = G1WBCActionCfg()
        self.observation_config = G1ObservationsCfg()
        self.event_config = G1EventCfg()
        self.mimic_env = MISSING

        # XR settings
        # This unfortunately works wrt to global coordinates, so its ideal if the robot is at the origin.
        # NOTE(xinjie.yao, 2025.09.09): Copied from GR1T2.py
        self.xr: XrCfg = XrCfg(
            anchor_pos=(0.0, 0.0, -1.0),
            anchor_rot=(0.70711, 0.0, 0.0, -0.70711),
        )


@configclass
class G1SceneCfg:

    # Gear'WBC G1 config, used in WBC training
    # TODO(xinjie.yao, 2025.09.15): Add G1 USD to isaac arena assets
    robot: ArticulationCfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path="omniverse://isaac-dev.ov.nvidia.com/Isaac/Samples/Groot/Robots/g1_29dof_with_hand_rev_1_0.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
        ),
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.8, -1.38, 0.78),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                # target angles [rad]
                "left_hip_pitch_joint": -0.1,
                "left_hip_roll_joint": 0.0,
                "left_hip_yaw_joint": 0.0,
                "left_knee_joint": 0.3,
                "left_ankle_pitch_joint": -0.2,
                "left_ankle_roll_joint": 0.0,
                "right_hip_pitch_joint": -0.1,
                "right_hip_roll_joint": 0.0,
                "right_hip_yaw_joint": 0.0,
                "right_knee_joint": 0.3,
                "right_ankle_pitch_joint": -0.2,
                "right_ankle_roll_joint": 0.0,
                "waist_yaw_joint": 0.0,
                "waist_roll_joint": 0.0,
                "waist_pitch_joint": 0.0,
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": 0.0,
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "legs": IdealPDActuatorCfg(
                joint_names_expr=[
                    ".*_hip_yaw_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_pitch_joint",
                    ".*_knee_joint",
                ],
                effort_limit={
                    ".*_hip_yaw_joint": 88.0,
                    ".*_hip_roll_joint": 88.0,
                    ".*_hip_pitch_joint": 88.0,
                    ".*_knee_joint": 139.0,
                },
                velocity_limit={
                    ".*_hip_yaw_joint": 32.0,
                    ".*_hip_roll_joint": 32.0,
                    ".*_hip_pitch_joint": 32.0,
                    ".*_knee_joint": 20.0,
                },
                stiffness={
                    ".*_hip_yaw_joint": 150.0,
                    ".*_hip_roll_joint": 150.0,
                    ".*_hip_pitch_joint": 150.0,
                    ".*_knee_joint": 300.0,
                },
                damping={
                    ".*_hip_yaw_joint": 2.0,
                    ".*_hip_roll_joint": 2.0,
                    ".*_hip_pitch_joint": 2.0,
                    ".*_knee_joint": 4.0,
                },
                armature={
                    ".*_hip_.*": 0.03,
                    ".*_knee_joint": 0.03,
                },
            ),
            "feet": IdealPDActuatorCfg(
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                stiffness={
                    ".*_ankle_pitch_joint": 40.0,
                    ".*_ankle_roll_joint": 40.0,
                },
                damping={
                    ".*_ankle_pitch_joint": 2,
                    ".*_ankle_roll_joint": 2,
                },
                effort_limit={
                    ".*_ankle_pitch_joint": 50.0,
                    ".*_ankle_roll_joint": 50.0,
                },
                velocity_limit={
                    ".*_ankle_pitch_joint": 37.0,
                    ".*_ankle_roll_joint": 37.0,
                },
                armature=0.03,
                friction=0.03,
            ),
            "waist": IdealPDActuatorCfg(
                joint_names_expr=[
                    "waist_.*_joint",
                ],
                effort_limit={
                    "waist_yaw_joint": 88.0,
                    "waist_roll_joint": 50.0,
                    "waist_pitch_joint": 50.0,
                },
                velocity_limit={
                    "waist_yaw_joint": 32.0,
                    "waist_roll_joint": 37.0,
                    "waist_pitch_joint": 37.0,
                },
                stiffness={
                    "waist_yaw_joint": 250.0,
                    "waist_roll_joint": 250.0,
                    "waist_pitch_joint": 250.0,
                },
                damping={
                    "waist_yaw_joint": 5.0,
                    "waist_roll_joint": 5.0,
                    "waist_pitch_joint": 5.0,
                },
                armature=0.03,
                friction=0.03,
            ),
            "arms": IdealPDActuatorCfg(
                joint_names_expr=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
                effort_limit={
                    ".*_shoulder_pitch_joint": 25.0,
                    ".*_shoulder_roll_joint": 25.0,
                    ".*_shoulder_yaw_joint": 25.0,
                    ".*_elbow_joint": 25.0,
                    ".*_wrist_roll_joint": 25.0,
                    ".*_wrist_pitch_joint": 5.0,
                    ".*_wrist_yaw_joint": 5.0,
                },
                velocity_limit={
                    ".*_shoulder_pitch_joint": 37.0,
                    ".*_shoulder_roll_joint": 37.0,
                    ".*_shoulder_yaw_joint": 37.0,
                    ".*_elbow_joint": 37.0,
                    ".*_wrist_roll_joint": 37.0,
                    ".*_wrist_pitch_joint": 22.0,
                    ".*_wrist_yaw_joint": 22.0,
                },
                stiffness={
                    ".*_shoulder_pitch_joint": 100.0,
                    ".*_shoulder_roll_joint": 100.0,
                    ".*_shoulder_yaw_joint": 40.0,
                    ".*_elbow_joint": 40.0,
                    ".*_wrist_.*_joint": 20.0,
                },
                damping={
                    ".*_shoulder_pitch_joint": 5.0,
                    ".*_shoulder_roll_joint": 5.0,
                    ".*_shoulder_yaw_joint": 2.0,
                    ".*_elbow_joint": 2.0,
                    ".*_wrist_.*_joint": 2.0,
                },
                armature={".*_shoulder_.*": 0.03, ".*_elbow_.*": 0.03, ".*_wrist_.*_joint": 0.03},
                friction=0.03,
            ),
            # TODO: check with teleop
            "hands": IdealPDActuatorCfg(
                joint_names_expr=[
                    ".*_hand_.*",
                ],
                effort_limit=2.0,
                velocity_limit=10.0,
                stiffness=4.0,
                damping=0.2,
                armature=0.03,
                friction=0.03,
            ),
        },
    )

    # TODO(vik: Fix camera and xr issues)
    # robot_pov_cam: CameraCfg = CameraCfg(
    #         prim_path="{ENV_REGEX_NS}/Robot/head_yaw_link/RobotPOVCam",
    #         update_period=0.0,
    #         height=512,
    #         width=512,
    #         data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
    #         spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.01, 1.0e5)),
    #         offset=CameraCfg.OffsetCfg(
    #             pos=(0.12515, 0.0, 0.06776),
    #             rot=(0.62, 0.32, -0.32, -0.63),
    #             convention="opengl",
    #         ),
    #     )


@configclass
class G1ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        # TODO(xinjie.yao, 2025.09.09): Add robot pov camera
        robot_joint_vel = ObsTerm(
            func=base_mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        right_wrist_pose_pelvis_frame = ObsTerm(
            func=transforms_terms.transform_pose_from_world_to_target_frame,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "target_link_name": "right_wrist_yaw_link",
                "target_frame_name": "pelvis",
            },
        )
        left_wrist_pose_pelvis_frame = ObsTerm(
            func=transforms_terms.transform_pose_from_world_to_target_frame,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "target_link_name": "left_wrist_yaw_link",
                "target_frame_name": "pelvis",
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class WBCObsCfg(ObsGroup):
        """Observations for WBC policy group with state values."""

        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_joint_vel = ObsTerm(
            func=base_mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        right_wrist_pose_pelvis_frame = ObsTerm(
            func=transforms_terms.transform_pose_from_world_to_target_frame,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "target_link_name": "right_wrist_yaw_link",
                "target_frame_name": "pelvis",
            },
        )
        left_wrist_pose_pelvis_frame = ObsTerm(
            func=transforms_terms.transform_pose_from_world_to_target_frame,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "target_link_name": "left_wrist_yaw_link",
                "target_frame_name": "pelvis",
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    wbc: WBCObsCfg = WBCObsCfg()


@configclass
class G1WBCActionCfg:
    """Action specifications for the MDP, for G1 WBC action."""

    g1_action: ActionTermCfg = G1DecoupledWBCActionCfg(asset_name="robot", joint_names=[".*"])


@configclass
class G1EventCfg:
    """Configuration for events."""

    # NOTE(xinjieyao, 2025-09-15): This will reset all the articulation joints to the initial state,
    # e.g. the robot will go to the initial pose, the microwave will return to init state, etc.
    reset_all = EventTerm(func=reset_all_articulation_joints, mode="reset")

    reset_wbc_policy = EventTerm(func=wbc_events_mdp.reset_decoupled_wbc_policy, mode="reset")
