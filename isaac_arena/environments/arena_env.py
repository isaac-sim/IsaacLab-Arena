from abc import ABC
from dataclasses import MISSING

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.rewards import is_alive
from isaaclab.envs.mdp.terminations import time_out
from isaaclab.managers import RewardTermCfg, TerminationTermCfg

from isaac_arena.embodiments.embodiments import EmbodimentBase
from isaac_arena.metrics.metrics import MetricsBase
from isaac_arena.scene.scene import SceneBase
from isaac_arena.tasks.task import TaskBase
from isaac_arena.embodiments.franka.franka import Franka
from isaac_arena.scene.instances.kitchen_mug_pick_and_place_scene import KitchenPickAndPlaceScene
from isaac_arena.scene.pick_and_place_scene import KitchenSceneCfg

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

#from . import mdp
from isaac_arena.environments import mdp

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

from isaac_arena.environments.mdp import franka_arrange_events, franka_stack_events
#from mindmap.tasks.mimic_task_definitions.kitchen import mdp
#from mindmap.tasks.mimic_task_definitions.kitchen.arrange_env_cfg import ArrangeEnvCfg
#from mindmap.tasks.mimic_task_definitions.kitchen.mdp import franka_arrange_events
#from mindmap.tasks.mimic_task_definitions.stack.mdp import franka_stack_events

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

#class ArenaEnv(ABC):
#    isaac_lab_env_cfg: ManagerBasedRLEnvCfg = MISSING
#
#    metrics_cfg: MISSING
#
#
#def compile_env(
#    scene: SceneBase,
#    embodiment: EmbodimentBase,
#    task: TaskBase,
#    metrics: MetricsBase,
#) -> ArenaEnv:
#    # Compose embodiment and scene observation cfg
#    class ObservationCfg:
#        embodiment_observation = embodiment.get_observation_cfg()
#        # scene_observation = scene.get_observation_cfg()
#
#    # Compose embodiment and scene events cfg
#    # class EventsCfg:
#    #     embodiment_events = embodiment.get_events_cfg()
#    #     scene_events = scene.get_events_cfg()
#
#    class IsaacLabEnvCfg(ManagerBasedRLEnvCfg):
#        scene_cfg = scene.get_scene_cfg()
#        observations_cfg = ObservationCfg()
#        actions_cfg = embodiment.get_action_cfg()
#        terminations_cfg = None
#        events_cfg = None
#
#        def __post_init__(self):
#            self._add_robot_to_scene_cfg()
#
#        def _add_robot_to_scene_cfg(self):
#            self.scene_cfg.robot = embodiment.get_robot_cfg()
#
#    return ArenaEnv(
#        isaac_lab_env_cfg=IsaacLabEnvCfg(),
#        metrics_cfg=metrics.get_metrics_cfg(),
#    )
#
#
#franka_global = Franka()
#scene_global = KitchenPickAndPlaceScene()
## Compose embodiment and scene observation cfg7
#
#class ObservationsCfg:
#    embodiment_observation = franka_global.get_observation_cfg()
#    # scene_observation = scene.get_observation_cfg()
#
# Compose embodiment and scene events cfg
# class EventsCfg:
#     embodiment_events = embodiment.get_events_cfg()
#     scene_events = scene.get_events_cfg()

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

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING

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
