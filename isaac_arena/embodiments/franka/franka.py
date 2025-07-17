from dataclasses import MISSING

from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
)
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG 

from isaac_arena.embodiments.embodiments import EmbodimentBase
from . import mdp


class Franka(EmbodimentBase):
    def __init__(self):
        self.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    @configclass
    class ActionsCfg:
        """Action specifications for the MDP."""

        # will be set by agent env cfg
        # arm_action: mdp.JointPositionActionCfg = MISSING
        # gripper_action: mdp.BinaryJointPositionActionCfg = MISSING
        pass
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        # actions = ObsTerm(func=mdp.last_action)
        # joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        # eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        # gripper_pos = ObsTerm(func=mdp.gripper_pos)
        pass

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
    
    def get_robot_cfg(self) -> ArticulationCfg:
        return self.robot

    def get_action_cfg(self) -> ActionsCfg:
        return self.ActionsCfg()

    def get_observation_cfg(self) -> PolicyCfg:
        return self.PolicyCfg()

