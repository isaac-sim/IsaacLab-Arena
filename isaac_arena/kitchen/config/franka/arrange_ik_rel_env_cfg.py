# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from mindmap.tasks.mimic_task_definitions.kitchen import mdp
from mindmap.tasks.mimic_task_definitions.kitchen.arrange_env_cfg import ArrangeEnvCfg
from mindmap.tasks.mimic_task_definitions.kitchen.mdp import franka_arrange_events
from mindmap.tasks.mimic_task_definitions.stack.mdp import franka_stack_events

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for events."""

    ### RANDOMIZE FRANKA ARM POSE ###

    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        # We changed the mode from startup to reset as the default pose got reset after it was
        # set by the startup event.
        # TODO(remos): find out why this happened and fix it
        mode="reset",
        params={
            "default_pose": [0.0, -0.785, -0.1107, -1.1775, 0.0, 0.785, 0.785, 0.0400, 0.0400],
        },
    )
    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    ### RANDOMIZE TABLE OBJECT POSITIONS ###

    randomize_table_object_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.35, 0.6),
                "y": (-0.3, 0.3),
                "z": (0.03, 0.03),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (3.14, 3.14),
            },
            "min_separation": 0.2,
            "asset_cfgs": [
                SceneEntityCfg("target_mug"),
                SceneEntityCfg("mac_n_cheese_on_table"),
                SceneEntityCfg("tomato_soup_on_table"),
            ],
        },
    )

    ### RANDOMIZE DRAWER OBJECT POSITIONS ###

    permute_drawers = EventTerm(
        func=franka_arrange_events.permute_object_poses,
        mode="reset",
        params={
            "pose_selection_list": [
                (0.06, -0.55, -0.16, 0.0, 0.0, 0.0),
                (0.06, 0.55, -0.16, 0.0, 0.0, 0.0),
            ],
            "asset_cfgs": [
                SceneEntityCfg("bottom_of_drawer_with_mugs"),
                SceneEntityCfg("bottom_of_drawer_with_boxes"),
            ],
        },
    )
    permute_objects_poses_in_mug_drawer = EventTerm(
        func=franka_arrange_events.permute_object_poses_relative_to_parent,
        mode="reset",
        params={
            "parent_asset_cfg": SceneEntityCfg("bottom_of_drawer_with_mugs"),
            "asset_cfgs": [SceneEntityCfg("mug1_in_drawer"), SceneEntityCfg("mug2_in_drawer")],
            "relative_object_poses": [
                (-0.05, -0.25, 0.01, 0.0, 0.0, 0.0),
                (-0.05, 0.25, 0.01, 0.0, 0.0, 0.0),
            ],
        },
    )
    permute_objects_poses_in_box_drawer = EventTerm(
        func=franka_arrange_events.permute_object_poses_relative_to_parent,
        mode="reset",
        params={
            "parent_asset_cfg": SceneEntityCfg("bottom_of_drawer_with_boxes"),
            "asset_cfgs": [
                SceneEntityCfg("sugar_box_in_drawer"),
                SceneEntityCfg("pudding_box_in_drawer"),
                SceneEntityCfg("gelatin_box_in_drawer"),
            ],
            "relative_object_poses": [
                (-0.05, -0.3, 0.01, 0.0, 0.0, 0.0),
                (-0.05, -0.2, 0.01, 0.0, 0.0, 0.0),
                (-0.05, 0.2, 0.01, 0.0, 0.0, 0.0),
                (-0.05, 0.3, 0.01, 0.0, 0.0, 0.0),
            ],
        },
    )


@configclass
class ArrangeKitchenObjectEnvCfg(ArrangeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Add semantics
        self.scene.robot.spawn.semantic_tags = [("class", "robot_arm")]

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        self.scene.target_mug = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/target_mug",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.35, 0.0, 0.094], rot=[0.0, 0.0, 0.0, 1.0]
            ),
            spawn=UsdFileCfg(
                usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/mug_physics.usd",
                scale=(0.0125, 0.0125, 0.0125),
                activate_contact_sensors=True,
            ),
        )

        self.scene.contact_forces_target_mug = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/target_mug", history_length=3, track_air_time=True
        )

        # Add the cams
        # Set wrist camera
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0333,
            height=256,
            width=256,
            data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=[0.13, 0.0, -0.15], rot=[-0.70614, 0.03701, 0.03701, -0.70614], convention="ros"
            ),
        )

        # Set table view camera
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0333,
            height=256,
            width=256,
            data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=[-1.0, 0.0, 1.6], rot=[0.64, 0.30, -0.30, -0.64], convention="opengl"
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )
