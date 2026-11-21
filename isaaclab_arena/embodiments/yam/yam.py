# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fixed-base I2RT YAM embodiment, spawned from AUTOLab's MuJoCo model under both PhysX and Newton (MJWarp)."""

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
from pxr import Usd, UsdPhysics

from isaaclab_arena.assets.nucleus import ARENA_NUCLEUS_DIR
from isaaclab_arena.assets.physics_config import UsdPrimSpawnPhysicsCfg
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.embodiments.yam.cameras import YamCameraCfg
from isaaclab_arena.utils.physics_backend import PhysicsBackend
from isaaclab_arena.utils.pose import Pose

YAM_USD_PATH = f"{ARENA_NUCLEUS_DIR}/Arena/assets/robot_library/yam/i2rt_yam_default.usda"
"""AUTOLab's MuJoCo YAM model converted to USD, with its contact, joint and gravity-compensation settings.

Both PhysX and Newton (MJWarp) spawn this USD. Isaac Sim's ``Robots_Multiphysics/I2RT/yam`` asset is not used:
it has no colliders.
"""

ARM_JOINT_NAMES = [f"joint{index}" for index in range(1, 7)]
GRIPPER_JOINT_NAME = "left_finger"
"""Driven finger joint."""
PASSIVE_GRIPPER_JOINT_NAME = "right_finger"
"""Finger joint the USD's mimic joint couples to the driven finger with a -1 gain."""
END_EFFECTOR_BODY_NAME = "link_6"
GRIPPER_OPEN_POSITION = 0.037524
GRIPPER_CLOSED_POSITION = 0.0

_DEFAULT_ARM_JOINT_POSITIONS = (0.0, 0.85, 0.60, 0.0, 0.0, 0.0)
_ETL_PHYSX_ARM_DRIVES = {
    "arm_joints_1_3": (40.0, 2.5, 28.0),
    "arm_joint_4": (20.0, 0.5, 10.0),
    "arm_joints_5_6": (10.0, 1.0, 10.0),
}
"""ETL's PhysX arm drives per actuator group: stiffness (N m/rad), damping (N m s/rad) and effort limit (N m)."""
_ETL_PHYSX_GRIPPER_EFFORT_LIMIT = 40.0
"""ETL's PhysX force limit on the driven finger, in newtons."""
_USD_BASE_JOINT_PATH = "Geometry/arm/PhysicsFixedJoint"
"""The USD's fixed joint from its static root Xform to the robot base, relative to the asset root."""


@configclass
class YamPhysxFixedBaseCfg(UsdPrimSpawnPhysicsCfg):
    """Replace the USD's base joint to its static root Xform with Isaac Lab's world-fixed root joint.

    PhysX anchors the USD's joint where the robot spawns, so root pose writes such as placement resets leave the base
    in place. Isaac Lab's fixed root joint follows them. This is ETL's PhysX fixed-base setup.
    """

    def validate_target(self, prim: Usd.Prim, root: Usd.Prim) -> None:
        joint = UsdPhysics.FixedJoint(prim)
        assert joint and joint.GetBody0Rel().GetTargets() == [root.GetPath()], "Expected the USD's YAM base joint"

    def apply(self, prim: Usd.Prim, root: Usd.Prim) -> None:
        from isaaclab.sim.schemas import apply_articulation_root_properties

        # Disable the USD's joint first, or Isaac Lab would take it as the existing world joint and keep it.
        UsdPhysics.FixedJoint(prim).GetJointEnabledAttr().Set(False)
        (base_path,) = UsdPhysics.FixedJoint(prim).GetBody1Rel().GetTargets()
        fixed = apply_articulation_root_properties(str(base_path), (), stage=root.GetStage(), fix_root_link=True)
        assert fixed, "Isaac Lab could not fix the YAM base to the world."


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
            # The USD's mimic joint moves this finger with the driven one, so it carries no drive of its own.
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
    """One fixed-base YAM arm with absolute joint-position control.

    PhysX and Newton (MJWarp) both spawn AUTOLab's MuJoCo model, ``YAM_USD_PATH``. Newton uses the USD as authored;
    PhysX applies ETL's PhysX setup on top of it.
    """

    registry_name = "yam"
    tags = ["embodiment", "yam"]
    default_arm_mode = ArmMode.SINGLE_ARM
    """SINGLE_ARM fits this class because it spawns one articulation; a bimanual YAM rig needs its own embodiment."""

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None, usd_path: str = YAM_USD_PATH):
        """Initialize the embodiment; ``enable_cameras`` adds the wrist and top-down cameras."""
        super().__init__(enable_cameras=enable_cameras, initial_pose=initial_pose, concatenate_observation_terms=True)
        self.scene_config = YamSceneCfg(robot=make_yam_articulation_cfg("{ENV_REGEX_NS}/Robot", usd_path))
        self.action_config = YamActionsCfg()
        self.observation_config = YamObservationsCfg()
        self.camera_config = YamCameraCfg()
        self.add_camera_variations(self.camera_config)

    def _configure_physics_backend(self, backend: PhysicsBackend) -> None:
        """Apply ETL's PhysX setup to a PhysX spawn; Newton uses the USD as authored.

        ETL's setup is a world-fixed base that follows root pose writes, no gravity on the robot links (PhysX ignores
        the USD's MuJoCo gravity compensation), and ETL's arm drives and finger force limit.
        """
        if backend is PhysicsBackend.PHYSX:
            from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

            prim_physics = self.spawn_cfg_addon.setdefault("robot", {}).setdefault("prim_physics", {})
            prim_physics[_USD_BASE_JOINT_PATH] = YamPhysxFixedBaseCfg()
            robot = self.scene_config.robot
            robot.spawn.rigid_props = PhysxRigidBodyCfg(disable_gravity=True)
            for group, (stiffness, damping, effort_limit) in _ETL_PHYSX_ARM_DRIVES.items():
                robot.actuators[group] = robot.actuators[group].replace(
                    stiffness=stiffness, damping=damping, joint_effort_limit=effort_limit
                )
            robot.actuators["gripper"] = robot.actuators["gripper"].replace(
                joint_effort_limit=_ETL_PHYSX_GRIPPER_EFFORT_LIMIT
            )

    def get_command_body_name(self) -> str:
        return END_EFFECTOR_BODY_NAME

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return END_EFFECTOR_BODY_NAME
