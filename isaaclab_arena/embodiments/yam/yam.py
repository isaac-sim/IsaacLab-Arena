# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fixed-base I2RT YAM embodiment for the Newton physics backend (MJWarp solver)."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import mdp
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.nucleus import ARENA_NUCLEUS_DIR
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.embodiments.yam.cameras import YamCameraCfg
from isaaclab_arena.utils.physics_backend import PhysicsBackend
from isaaclab_arena.utils.pose import Pose

YAM_USD_PATH = f"{ARENA_NUCLEUS_DIR}/Arena/assets/robot_library/yam/i2rt_yam_default.usda"
"""Newton/MJWarp YAM: MuJoCo-converted USD with Newton contact and MuJoCo joint properties."""

ARM_JOINT_NAMES = [f"joint{index}" for index in range(1, 7)]
GRIPPER_JOINT_NAME = "left_finger"
"""Driven finger joint."""
PASSIVE_GRIPPER_JOINT_NAME = "right_finger"
"""Finger joint the USD's Newton mimic couples to the driven finger with a -1 gain."""
END_EFFECTOR_BODY_NAME = "link_6"
GRIPPER_OPEN_POSITION = 0.037524
GRIPPER_CLOSED_POSITION = 0.0
NEWTON_CONTACT_GAP = 0.001
"""Newton collision contact gap in meters."""

_DEFAULT_ARM_JOINT_POSITIONS = (0.0, 0.85, 0.60, 0.0, 0.0, 0.0)


def make_yam_articulation_cfg(prim_path: str, usd_path: str = YAM_USD_PATH) -> ArticulationCfg:
    """Build one fixed-base YAM articulation with position-driven joints.

    Args:
        prim_path: Prim path of the articulation, e.g. ``{ENV_REGEX_NS}/Robot``.
        usd_path: YAM USD to spawn.
    """
    joint_pos = dict(zip(ARM_JOINT_NAMES, _DEFAULT_ARM_JOINT_POSITIONS, strict=True))
    joint_pos[GRIPPER_JOINT_NAME] = GRIPPER_OPEN_POSITION
    joint_pos[PASSIVE_GRIPPER_JOINT_NAME] = -GRIPPER_OPEN_POSITION
    return ArticulationCfg(
        prim_path=prim_path,
        articulation_root_prim_path="/Geometry/arm",
        spawn=sim_utils.UsdFileCfg(usd_path=usd_path, copy_from_source=False, activate_contact_sensors=True),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos=joint_pos, joint_vel={".*": 0.0}),
        soft_joint_pos_limit_factor=0.95,
        actuators={
            "arm_joints_1_3": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-3]"], stiffness=80.0, damping=6.0, joint_effort_limit=28.0
            ),
            "arm_joint_4": ImplicitActuatorCfg(
                joint_names_expr=["joint4"], stiffness=30.0, damping=2.0, joint_effort_limit=10.0
            ),
            "arm_joints_5_6": ImplicitActuatorCfg(
                joint_names_expr=["joint[5-6]"], stiffness=30.0, damping=2.0, joint_effort_limit=10.0
            ),
            "gripper": ImplicitActuatorCfg(joint_names_expr=[GRIPPER_JOINT_NAME], stiffness=1000.0, damping=100.0),
            # The Newton mimic joint drives this finger from the driven one, so it carries no drive of its own.
            "gripper_passive": ImplicitActuatorCfg(
                joint_names_expr=[PASSIVE_GRIPPER_JOINT_NAME], stiffness=0.0, damping=0.0
            ),
        },
    )


@configclass
class YamSceneCfg:
    """The YAM articulation."""

    robot: ArticulationCfg | None = None


@configclass
class YamActionsCfg:
    """Absolute arm joint targets and a gripper command in [0, 1] (0 open, 1 closed)."""

    arm_action: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot", joint_names=ARM_JOINT_NAMES, preserve_order=True, use_default_offset=False
    )
    gripper_action: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[GRIPPER_JOINT_NAME],
        scale=GRIPPER_CLOSED_POSITION - GRIPPER_OPEN_POSITION,
        offset=GRIPPER_OPEN_POSITION,
        preserve_order=True,
        use_default_offset=False,
    )


@configclass
class YamObservationsCfg:
    """Joint state and previous-action policy observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Concatenated proprioceptive observations."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@register_asset
class YamEmbodiment(EmbodimentBase):
    """One fixed-base YAM arm with absolute joint-position control. Implemented for Newton (MJWarp solver) only."""

    registry_name = "yam"
    tags = ["embodiment", "yam"]
    default_arm_mode = ArmMode.SINGLE_ARM

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None, usd_path: str = YAM_USD_PATH):
        """Initialize the embodiment; ``enable_cameras`` adds the wrist and top-down cameras."""
        super().__init__(enable_cameras=enable_cameras, initial_pose=initial_pose, concatenate_observation_terms=True)
        self.scene_config = YamSceneCfg(robot=make_yam_articulation_cfg("{ENV_REGEX_NS}/Robot", usd_path))
        self.action_config = YamActionsCfg()
        self.observation_config = YamObservationsCfg()
        self.camera_config = YamCameraCfg()
        self.add_camera_variations(self.camera_config)

    def _configure_physics_backend(self, backend: PhysicsBackend) -> None:
        """Apply the Newton spawn overrides; PhysX is not implemented."""
        if backend is PhysicsBackend.NEWTON:
            self._configure_newton_spawn()
        else:
            raise NotImplementedError(f"{self.registry_name} does not implement the {backend} physics backend.")

    def _configure_newton_spawn(self) -> None:
        """Spawn the YAM with gravity compensation, no self-collision and a small contact gap."""
        from isaaclab.sim.schemas.schemas_cfg import JointDriveBaseCfg
        from isaaclab_newton.sim.schemas import (
            MujocoRigidBodyPropertiesCfg,
            NewtonArticulationRootPropertiesCfg,
            NewtonCollisionPropertiesCfg,
        )

        spawn = self.scene_config.robot.spawn
        spawn.rigid_props = MujocoRigidBodyPropertiesCfg(gravcomp=1.0)
        spawn.articulation_props = NewtonArticulationRootPropertiesCfg(self_collision_enabled=False)
        spawn.collision_props = NewtonCollisionPropertiesCfg(contact_margin=0.0, contact_gap=NEWTON_CONTACT_GAP)
        spawn.joint_drive_props = JointDriveBaseCfg(ensure_drives_exist=True)

    def get_command_body_name(self) -> str:
        return END_EFFECTOR_BODY_NAME

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return END_EFFECTOR_BODY_NAME
