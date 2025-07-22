from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

import isaaclab.envs.mdp as mdp_isaac_lab
from isaac_arena.embodiments.embodiment_base import ActionsCfg, EmbodimentBase, EventCfg, ObservationsCfg
from isaac_arena.embodiments.mdp import franka_stack_events
from isaac_arena.embodiments.mdp.observations import ee_frame_pos, ee_frame_quat, gripper_pos
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


class FrankaEmbodiment(EmbodimentBase):
    def __init__(self):
        self.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.action_config = FrankaActionsCfg()
        self.observation_config = FrankaObservationsCfg()
        self.event_config = FrankaEventCfg()
        self.robot.spawn.semantic_tags = [("class", "robot_arm")]


@configclass
class FrankaActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""

    arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    )

    gripper_action: ActionTermCfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class FrankaObservationsCfg(ObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp_isaac_lab.last_action)
        joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel_rel)
        eef_pos = ObsTerm(func=ee_frame_pos)
        eef_quat = ObsTerm(func=ee_frame_quat)
        gripper_pos = ObsTerm(func=gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class FrankaEventCfg(EventCfg):
    """Configuration for Franek."""

    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
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
