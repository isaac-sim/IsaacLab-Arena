# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp


##
# Scene definition
##
@configclass
class KitchenTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    ### ROBOT + KITCHEN SCENE ###

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # cameras: will be populated by agent env cfg
    wrist_cam: CameraCfg = MISSING
    table_cam: CameraCfg = MISSING

    # Add the kitchen scene here
    kitchen = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Kitchen",
        # These positions are hardcoded for the kitchen scene. Its important to keep them.
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.772, 3.39, -0.895], rot=[0.70711, 0, 0, -0.70711]
        ),
        spawn=UsdFileCfg(
            usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/kitchen_scene_teleop_v3.usd"
        ),
    )

    ### HELPER OBJECTS ###

    # Add a plate right below the bottom of the drawer were the mugs are placed.
    # This will be useful to have a fixed reference to the mugs drawer in mimicgen
    bottom_of_drawer_with_mugs = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/bottom_of_drawer_with_mugs",
        spawn=sim_utils.CuboidCfg(
            size=[0.4, 0.65, 0.01],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
        ),
    )
    # Add a plate right below the bottom of the drawer were the boxes are placed.
    # This will be useful to have a fixed reference to the boxes drawer in mimicgen
    bottom_of_drawer_with_boxes = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/bottom_of_drawer_with_boxes",
        spawn=sim_utils.CuboidCfg(
            size=[0.4, 0.65, 0.01],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
        ),
    )

    ### OBJECTS ON TABLE ###

    mac_n_cheese_on_table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/mac_n_cheese_on_table",
        spawn=UsdFileCfg(
            usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/mac_n_cheese_physics.usd",
            scale=(1.0, 1.0, 1.0),
        ),
    )
    tomato_soup_on_table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/tomato_soup_on_table",
        spawn=UsdFileCfg(
            usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/tomato_soup_physics.usd",
            scale=(1.0, 1.0, 1.0),
        ),
    )

    ### OBJECTS IN DRAWERS ###

    # To have a fixed reference frame for mimicgen
    mug1_in_drawer = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/mug1_in_drawer",
        spawn=UsdFileCfg(
            # usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Isaac/Props/Mugs/SM_Mug_A2.usd",
            usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/mug2_physics.usd",
            scale=(0.0125, 0.0125, 0.0125),
            activate_contact_sensors=True,
        ),
    )
    mug2_in_drawer = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/mug2_in_drawer",
        spawn=UsdFileCfg(
            usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/mug3_physics.usd",
            # usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Isaac/Props/Mugs/SM_Mug_D1.usd",
            scale=(0.0125, 0.0125, 0.0125),
        ),
    )
    sugar_box_in_drawer = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/sugar_box_in_drawer",
        spawn=UsdFileCfg(
            usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/sugar_box_physics.usd",
            scale=(1.0, 1.0, 1.0),
        ),
    )
    pudding_box_in_drawer = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/pudding_box_in_drawer",
        spawn=UsdFileCfg(
            usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/pudding_box_physics.usd",
            scale=(1.0, 1.0, 1.0),
        ),
    )
    gelatin_box_in_drawer = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/gelatin_box_in_drawer",
        spawn=UsdFileCfg(
            usd_path=f"omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/Collected_kitchen_scene/gelatin_box_physics.usd",
            scale=(1.0, 1.0, 1.0),
        ),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp_1 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("target_mug"),
                "contact_sensor_cfg": SceneEntityCfg("contact_forces_target_mug"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.2, "asset_cfg": SceneEntityCfg("target_mug")},
    )

    success = DoneTerm(func=mdp.object_in_drawer)


@configclass
class ArrangeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stacking environment."""

    # Scene settings
    scene: KitchenTableSceneCfg = KitchenTableSceneCfg(
        num_envs=4096, env_spacing=30, replicate_physics=False
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers
    commands = None
    rewards = None
    events = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = 2

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
