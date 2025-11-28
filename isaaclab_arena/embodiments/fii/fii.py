from collections.abc import Sequence
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.assets.register import register_asset

from isaaclab_arena.utils.pose import Pose
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLMimicEnv 
import torch
import isaaclab.utils.math as PoseUtils
import numpy as np
import torch
import os
import math
import tempfile

from .swerve_ik import swerve_isosceles_ik
import isaaclab.controllers.utils as ControllerUtils

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as base_mdp

from isaaclab.controllers.pink_ik.local_frame_task import LocalFrameTask
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.assets.articulation import Articulation
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp as manip_mdp

from isaaclab.controllers.pink_ik.pink_ik_cfg import PinkIKControllerCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.envs import ManagerBasedEnv

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

@register_asset
class FiiEmbodiment(EmbodimentBase):
    """
    Embodiment for the FII robot.
    """

    name = "fii"

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.scene_config = FiiSceneCfg()
        self.action_config = FiiActionsCfg()
        self.observation_config = FiiObservationsCfg()
        
        # Convert USD to URDF for Pink IK controller
        self.temp_urdf_dir = tempfile.gettempdir()
        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene_config.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=False
        )
        
        # Set the URDF and mesh paths for the IK controller
        self.action_config.upper_body_ik.controller.urdf_path = temp_urdf_output_path
        self.action_config.upper_body_ik.controller.mesh_path = temp_urdf_meshes_output_path

#=======================================================================
#   SCENE
#=======================================================================
@configclass
class FiiSceneCfg:
    """Scene configuration for the FII embodiment."""

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), 
            rot=(0.7071068, 0.0, 0.0, 0.7071068),
            joint_pos={
                "jack_joint": 0.7,
                "left_1_joint": 0.0,
                "left_2_joint": 0.785398,
                "left_3_joint": 0.0,
                "left_4_joint": 1.570796,
                "left_5_joint": 0.0,
                "left_6_joint": -0.785398,
                "left_7_joint": 0.0,
                "right_1_joint": 0.0,
                "right_2_joint": 0.785398,
                "right_3_joint": 0.0,
                "right_4_joint": 1.570796,
                "right_5_joint": 0.0,
                "right_6_joint": -0.785398,
                "right_7_joint": 0.0,
            }
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspaces/isaaclab_arena/isaaclab_arena/embodiments/embodiment_library/Fiibot_W_1_V2_251016_Modified.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            )
        ),
        actuators={
            "actuators": ImplicitActuatorCfg(
                joint_names_expr=[".*"], 
                damping=None, 
                stiffness=None
            ),
            "jack_joint": ImplicitActuatorCfg(
                joint_names_expr=["jack_joint"], 
                damping=5000., 
                stiffness=500000.
            ),
        },
    )
#=======================================================================
#   ACTIONS
#=======================================================================


class FiibotLowerBodyAction(ActionTerm):
    """Action term that is based on Agile lower body RL policy."""

    cfg: "FiibotLowerBodyActionCfg"
    """The configuration of the action term."""

    _asset: Articulation
    """The articulation asset to which the action term is applied."""

    def __init__(self, cfg: "FiibotLowerBodyActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        self._env = env

        self._joint_names = [
            "walk_mid_top_joint",
            "walk_left_bottom_joint",
            "walk_right_bottom_joint",
            "jack_joint",
            "front_wheel_joint",
            "left_wheel_joint",
            "right_wheel_joint"
        ]

        self._joint_ids = [
            self._asset.data.joint_names.index(joint_name)
            for joint_name in self._joint_names
        ]

        self._joint_pos_target = torch.zeros(self.num_envs, 7, device=self.device)
        self._joint_vel_target = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def action_dim(self) -> int:
        """Lower Body Action: [vx, vy, wz, jack_joint_height]"""
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._joint_pos_target

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._joint_pos_target
    
    def process_actions(self, actions: torch.Tensor):

        ik_out = swerve_isosceles_ik(
            vx=float(actions[0, 0]),
            vy=float(actions[0, 1]),
            wz=float(actions[0, 2]),
            L1=0.30438,
            d=0.17362,
            w=0.25,
            R=0.06
        )

        self._joint_pos_target[:, 0] = ik_out['wheel1']['angle_rad']
        self._joint_pos_target[:, 1] = ik_out['wheel2']['angle_rad']
        self._joint_pos_target[:, 2] = ik_out['wheel3']['angle_rad']
        self._joint_pos_target[:, 3] = float(actions[0, 3])

        self._joint_vel_target[:, 0] = ik_out['wheel1']['omega']
        self._joint_vel_target[:, 1] = ik_out['wheel2']['omega']
        self._joint_vel_target[:, 2] = ik_out['wheel3']['omega']

    def apply_actions(self):

        self._joint_pos_target[:, 4:] = self._joint_pos_target[:, 4:] + self._env.physics_dt * self._joint_vel_target

        self._asset.set_joint_position_target(
            target=self._joint_pos_target,
            joint_ids=self._joint_ids
        )



@configclass
class FiibotLowerBodyActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = FiibotLowerBodyAction


@configclass
class FiiActionsCfg:
    """Action specifications for the MDP."""

    upper_body_ik = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=[
            # "waist_joint",
            "left_1_joint",
            "left_2_joint",
            "left_3_joint",
            "left_4_joint",
            "left_5_joint",
            "left_6_joint",
            "left_7_joint",
            "right_1_joint",
            "right_2_joint",
            "right_3_joint",
            "right_4_joint",
            "right_5_joint",
            "right_6_joint",
            "right_7_joint"
        ],
        hand_joint_names=[
            "left_hand_grip1_joint",
            "left_hand_grip2_joint",
            "right_hand_grip1_joint",
            "right_hand_grip2_joint"
        ],
        target_eef_link_names={
            "left_wrist": "Fiibot_W_2_V2_left_7_Link",
            "right_wrist": "Fiibot_W_2_V2_right_7_Link",
        },
        asset_name="robot",
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="base_link",
            num_hand_joints=4,
            show_ik_warnings=True,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                LocalFrameTask(
                    "Fiibot_W_2_V2_left_7_Link",
                    base_link_frame_name="Root",
                    position_cost=1.0,
                    orientation_cost=1.0,
                    lm_damping=10,
                    gain=0.1,
                ),
                LocalFrameTask(
                    "Fiibot_W_2_V2_right_7_Link",
                    base_link_frame_name="Root",
                    position_cost=1.0,
                    orientation_cost=1.0,
                    lm_damping=10,
                    gain=0.1,
                )
            ],
            fixed_input_tasks=[],
        )
    )

    lower_body_ik = FiibotLowerBodyActionCfg(
        asset_name="robot"
    )


#=======================================================================
#   OBSERVATIONS
#=======================================================================
@configclass
class FiiObservationsCfg:
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=manip_mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        object_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("io_board")})
        object_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("io_board")})
        robot_links_state = ObsTerm(func=manip_mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(func=manip_mdp.get_eef_pos, params={"link_name": "left_7_Link"})
        left_eef_quat = ObsTerm(func=manip_mdp.get_eef_quat, params={"link_name": "left_7_Link"})
        right_eef_pos = ObsTerm(func=manip_mdp.get_eef_pos, params={"link_name": "right_7_Link"})
        right_eef_quat = ObsTerm(func=manip_mdp.get_eef_quat, params={"link_name": "right_7_Link"})

        hand_joint_state = ObsTerm(func=manip_mdp.get_robot_joint_state, params={"joint_names": [
            "left_hand_grip1_joint",
            "left_hand_grip2_joint",
            "right_hand_grip1_joint",
            "right_hand_grip2_joint"
        ]})

        # Note: object_obs function hardcodes env.scene["object"], which doesn't exist in our scene
        # We already have object_pos and object_rot observations above, so this is redundant
        # object = ObsTerm(
        #     func=manip_mdp.object_obs,
        #     params={"left_eef_link_name": "left_7_Link", "right_eef_link_name": "right_7_Link"},
        # )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False


    policy: PolicyCfg = PolicyCfg()

