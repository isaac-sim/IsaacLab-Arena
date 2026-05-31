# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Franka OSC embodiment for the NIST gear insertion task."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.franka.franka import _DEFAULT_CAMERA_OFFSET, FrankaEmbodimentBase
from isaaclab_arena.embodiments.franka.nist_gear_insertion.gear_grasp import get_franka_nist_gear_insertion_grasp_config
from isaaclab_arena.tasks.nist_gear_insertion.events import GraspCfg
from isaaclab_arena.utils.pose import Pose

__all__ = [
    "FrankaNistGearInsertionOscEmbodiment",
]

_FRANKA_MIMIC_OSC_USD_PATH = f"{ISAACLAB_NUCLEUS_DIR}/Factory/franka_mimic.usd"

# Mirrors Isaac Lab Factory's franka_mimic OSC setup in
# isaaclab_tasks.direct.factory.factory_env_cfg.
_FRANKA_MIMIC_OSC_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=True,
    max_depenetration_velocity=5.0,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=3666.0,
    enable_gyroscopic_forces=True,
    solver_position_iteration_count=192,
    solver_velocity_iteration_count=1,
    max_contact_impulse=1e32,
)

_FRANKA_MIMIC_OSC_ARTICULATION_PROPS = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=False,
    solver_position_iteration_count=192,
    solver_velocity_iteration_count=1,
)

_FRANKA_MIMIC_OSC_COLLISION_PROPS = sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0)

# Default pose from the Factory-style Franka mimic asset.
_FRANKA_MIMIC_OSC_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
    "panda_finger_joint2": 0.04,
}

# OSC writes joint torques directly for the arm; the hand remains position controlled.
_FRANKA_MIMIC_OSC_ACTUATORS = {
    "panda_arm1": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-4]"],
        stiffness=0.0,
        damping=0.0,
        friction=0.0,
        armature=0.0,
        effort_limit_sim=87,
        velocity_limit_sim=124.6,
    ),
    "panda_arm2": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[5-7]"],
        stiffness=0.0,
        damping=0.0,
        friction=0.0,
        armature=0.0,
        effort_limit_sim=12,
        velocity_limit_sim=149.5,
    ),
    "panda_hand": ImplicitActuatorCfg(
        joint_names_expr=["panda_finger_joint[1-2]"],
        effort_limit_sim=40.0,
        velocity_limit_sim=0.04,
        stiffness=7500.0,
        damping=173.0,
        friction=0.1,
        armature=0.0,
    ),
}

# Franka mimic USD with contact sensors enabled for the wrist force body.
_FRANKA_MIMIC_OSC_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_FRANKA_MIMIC_OSC_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=_FRANKA_MIMIC_OSC_RIGID_PROPS,
        articulation_props=_FRANKA_MIMIC_OSC_ARTICULATION_PROPS,
        collision_props=_FRANKA_MIMIC_OSC_COLLISION_PROPS,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=_FRANKA_MIMIC_OSC_JOINT_POS,
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
    ),
    actuators=_FRANKA_MIMIC_OSC_ACTUATORS,
)

_GEAR_INSERTION_INITIAL_JOINT_POSE = [
    0.561824,
    0.287201,
    -0.543103,
    -2.410188,
    0.507908,
    2.847644,
    0.454298,
    0.04,
    0.04,
]


def _gear_insertion_ee_frame_cfg() -> FrameTransformerCfg:
    """Return Franka frames used by the gear insertion OSC policy.

    The policy commands the centered fingertip frame, while the finger frames
    are exposed for grasp/reset diagnostics.
    """
    return FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_fingertip_centered",
                name="end_effector",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                name="tool_rightfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                name="tool_leftfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
        ],
    )


@register_asset
class FrankaNistGearInsertionOscEmbodiment(FrankaEmbodimentBase):
    """Franka embodiment for NIST gear insertion with OSC torque control.

    The embodiment owns the Franka mimic robot setup, OSC command frame, and
    grasp metadata. Environments wire scene-specific asset names and insertion
    geometry into the action, observation, and reward configs.
    """

    name = "franka_nist_gear_insertion_osc"

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        initial_joint_pose: list[float] | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        camera_offset: Pose | None = _DEFAULT_CAMERA_OFFSET,
        is_tiled_camera: bool = False,
    ):
        super().__init__(
            enable_cameras=enable_cameras,
            initial_pose=initial_pose,
            initial_joint_pose=initial_joint_pose,
            concatenate_observation_terms=concatenate_observation_terms,
            arm_mode=arm_mode,
            camera_offset=camera_offset,
            is_tiled_camera=is_tiled_camera,
        )
        self.scene_config.robot = _FRANKA_MIMIC_OSC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene_config.ee_frame = _gear_insertion_ee_frame_cfg()
        self.set_initial_joint_pose(initial_joint_pose or _GEAR_INSERTION_INITIAL_JOINT_POSE)

    def get_command_body_name(self) -> str:
        return "panda_fingertip_centered"

    def get_gear_insertion_grasp_config(self) -> GraspCfg:
        return get_franka_nist_gear_insertion_grasp_config()
