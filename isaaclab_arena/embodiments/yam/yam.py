# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bimanual I2RT YAM embodiment, spawned from AUTOLab's MuJoCo model under both PhysX and Newton (MJWarp)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

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
from isaaclab_arena.embodiments.embodiment_base import ArticulationGeometrySpec, EmbodimentBase
from isaaclab_arena.embodiments.yam.cameras import YamBimanualCameraCfg
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.physics_backend import PhysicsBackend
from isaaclab_arena.utils.pose import Pose, PosePerEnv

if TYPE_CHECKING:
    import trimesh

YAM_USD_PATH = f"{ARENA_NUCLEUS_DIR}/Arena/assets/robot_library/yam/i2rt_yam_default.usda"
"""AUTOLab's MuJoCo YAM model converted to USD, with its contact, joint and gravity-compensation settings.

Both PhysX and Newton (MJWarp) spawn this USD. Isaac Sim's ``Robots_Multiphysics/I2RT/yam`` asset is not used:
it has no colliders.
"""

LEFT_ROBOT_KEY = "left_robot"
RIGHT_ROBOT_KEY = "right_robot"
t_M_B_BY_ARM = {LEFT_ROBOT_KEY: (0.0, 0.31, 0.0), RIGHT_ROBOT_KEY: (0.0, -0.31, 0.0)}
"""Each arm's base position in the rig frame M, which sits midway between the bases with +X forward and +Z up.

The bases are 0.62 m apart, as in AUTOLab's bimanual scene.
"""

ARM_JOINT_NAMES = [f"joint{index}" for index in range(1, 7)]
GRIPPER_JOINT_NAME = "left_finger"
"""Driven finger joint."""
PASSIVE_GRIPPER_JOINT_NAME = "right_finger"
"""Right finger joint, which follows the driven finger through the USD's mimic joint and has no drive of its own.

The USD couples the fingers so the left finger stays at minus the right finger's position. Both MJWarp and PhysX
read this mimic joint, so no separate PhysX mimic is added: it would constrain the same finger pair twice.
"""
END_EFFECTOR_BODY_NAME = "link_6"
GRIPPER_OPEN_POSITION = 0.037524
GRIPPER_CLOSED_POSITION = 0.0

_DEFAULT_ARM_JOINT_POSITIONS = (0.0, 0.85, 0.60, 0.0, 0.0, 0.0)
_PHYSX_ARM_DRIVES = {
    "arm_joints_1_3": (40.0, 2.5, 28.0),
    "arm_joint_4": (20.0, 0.5, 10.0),
    "arm_joints_5_6": (10.0, 1.0, 10.0),
}
"""PhysX arm drives per actuator group: stiffness (N m/rad), damping (N m s/rad) and effort limit (N m)."""
_PHYSX_GRIPPER_EFFORT_LIMIT = 40.0
"""PhysX force limit on the driven finger, in newtons."""
_USD_BASE_JOINT_PATH = "Geometry/arm/PhysicsFixedJoint"
"""The USD's fixed joint from its static root Xform to the robot base, relative to the asset root."""


@configclass
class YamPhysxFixedBaseCfg(UsdPrimSpawnPhysicsCfg):
    """Replace the USD's base joint to its static root Xform with Isaac Lab's world-fixed root joint.

    PhysX anchors the USD's joint where the robot spawns, so root pose writes such as placement resets leave the base
    in place. Isaac Lab's fixed root joint follows them.
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
        prim_path: Prim path of the articulation, e.g. ``{ENV_REGEX_NS}/LeftRobot``.
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


def _make_arm_action_cfg(asset_name: str) -> JointPositionActionCfg:
    """Build absolute joint-position targets for one arm's six joints."""
    return JointPositionActionCfg(
        asset_name=asset_name, joint_names=ARM_JOINT_NAMES, preserve_order=True, use_default_offset=False
    )


def _make_gripper_action_cfg(asset_name: str) -> JointPositionActionCfg:
    """Build one arm's gripper command in [0, 1] (0 open, 1 closed) on its driven finger."""
    return JointPositionActionCfg(
        asset_name=asset_name,
        joint_names=[GRIPPER_JOINT_NAME],
        scale=GRIPPER_CLOSED_POSITION - GRIPPER_OPEN_POSITION,
        offset=GRIPPER_OPEN_POSITION,
        preserve_order=True,
        use_default_offset=False,
    )


@configclass
class YamBimanualSceneCfg:
    """The left and right YAM articulations."""

    left_robot: ArticulationCfg | None = None
    right_robot: ArticulationCfg | None = None


@configclass
class YamBimanualAbsoluteJointPositionActionsCfg:
    """Left arm, left gripper, right arm and right gripper commands, in that policy order."""

    left_arm_action: JointPositionActionCfg = _make_arm_action_cfg(LEFT_ROBOT_KEY)
    left_gripper_action: JointPositionActionCfg = _make_gripper_action_cfg(LEFT_ROBOT_KEY)
    right_arm_action: JointPositionActionCfg = _make_arm_action_cfg(RIGHT_ROBOT_KEY)
    right_gripper_action: JointPositionActionCfg = _make_gripper_action_cfg(RIGHT_ROBOT_KEY)


@configclass
class YamBimanualObservationsCfg:
    """Both arms' joint states and the previous action."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Concatenated proprioceptive observations."""

        left_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg(LEFT_ROBOT_KEY)})
        left_joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg(LEFT_ROBOT_KEY)})
        right_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg(RIGHT_ROBOT_KEY)})
        right_joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg(RIGHT_ROBOT_KEY)})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@register_asset
class YamBimanualAbsoluteJointPositionEmbodiment(EmbodimentBase):
    """Two fixed-base YAM arms side by side, each with absolute joint-position control.

    The embodiment pose is the rig frame midway between the two bases (see ``t_M_B_BY_ARM``). PhysX and Newton
    (MJWarp) both spawn AUTOLab's MuJoCo model, ``YAM_USD_PATH``, for each arm. Newton uses the USD as authored;
    PhysX adds the overrides in ``_configure_physics_backend``.

    NOTE(xinjieyao, 2026-11-21): The right finger has no drive. The USD's mimic joint makes it mirror the driven left
    finger under both backends (see ``PASSIVE_GRIPPER_JOINT_NAME``).

    NOTE(xinjieyao, 2026-11-21): The USD sets MuJoCo's ``gravcomp`` to 1 on every robot link, which MJWarp applies and
    PhysX ignores. Under PhysX the robot links have gravity disabled instead; without it the arms sag under their
    drives.
    """

    registry_name = "yam_bimanual_abs_joint_pos"
    tags = ["embodiment", "yam", "bimanual"]
    default_arm_mode = ArmMode.DUAL_ARM
    """DUAL_ARM fits this class because it spawns two arms; LEFT or RIGHT marks a task done with one of them."""

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None, usd_path: str = YAM_USD_PATH):
        """Initialize the embodiment; ``enable_cameras`` adds both wrist cameras and the top-down camera."""
        super().__init__(enable_cameras=enable_cameras, initial_pose=initial_pose, concatenate_observation_terms=True)
        self.scene_config = YamBimanualSceneCfg(
            left_robot=make_yam_articulation_cfg("{ENV_REGEX_NS}/LeftRobot", usd_path),
            right_robot=make_yam_articulation_cfg("{ENV_REGEX_NS}/RightRobot", usd_path),
        )
        for name, t_M_B in t_M_B_BY_ARM.items():
            getattr(self.scene_config, name).init_state.pos = t_M_B
        self.action_config = YamBimanualAbsoluteJointPositionActionsCfg()
        self.observation_config = YamBimanualObservationsCfg()
        self.camera_config = YamBimanualCameraCfg()
        self.camera_config.mount_top_camera_on_left_base(t_M_B_BY_ARM[LEFT_ROBOT_KEY])
        self.add_camera_variations(self.camera_config)

    def _configure_physics_backend(self, backend: PhysicsBackend) -> None:
        """Apply the PhysX overrides to both arms' PhysX spawns; Newton uses the USD as authored.

        PhysX gets a world-fixed base that follows root pose writes, no gravity on the robot links (PhysX ignores the
        USD's MuJoCo gravity compensation), softer arm drives and a finger force limit.
        """
        if backend is not PhysicsBackend.PHYSX:
            return
        from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

        for name in t_M_B_BY_ARM:
            prim_physics = self.spawn_cfg_addon.setdefault(name, {}).setdefault("prim_physics", {})
            prim_physics[_USD_BASE_JOINT_PATH] = YamPhysxFixedBaseCfg()
            robot = getattr(self.scene_config, name)
            robot.spawn.rigid_props = PhysxRigidBodyCfg(disable_gravity=True)
            for group, (stiffness, damping, effort_limit) in _PHYSX_ARM_DRIVES.items():
                robot.actuators[group] = robot.actuators[group].replace(
                    stiffness=stiffness, damping=damping, joint_effort_limit=effort_limit
                )
            robot.actuators["gripper"] = robot.actuators["gripper"].replace(
                joint_effort_limit=_PHYSX_GRIPPER_EFFORT_LIMIT
            )

    def get_scene_key(self) -> str:
        """Return the left arm as the primary scene entity."""
        return LEFT_ROBOT_KEY

    def layout_pose_to_scene_writes(self, layout_pose: Pose) -> list[tuple[str, Pose]]:
        """Return each arm's base pose for a rig pose ``layout_pose`` (T_E_M, rig frame M in the env frame E)."""
        return [(name, layout_pose.multiply(Pose(position_xyz=t_M_B))) for name, t_M_B in t_M_B_BY_ARM.items()]

    def _update_scene_cfg_with_robot_initial_pose(self, scene_config: Any, pose: Pose) -> Any:
        for name, T_E_B in self.layout_pose_to_scene_writes(pose):
            robot = getattr(scene_config, name)
            robot.init_state.pos = T_E_B.position_xyz
            robot.init_state.rot = T_E_B.rotation_xyzw
        return scene_config

    def get_initial_pose(self) -> Pose | PosePerEnv:
        """Env-local rig pose: the explicit ``initial_pose`` if set, otherwise derived from the left arm's base."""
        if self.initial_pose is not None:
            return self.initial_pose
        left_state = self.scene_config.left_robot.init_state
        T_E_L = Pose(position_xyz=tuple(left_state.pos), rotation_xyzw=tuple(left_state.rot))
        t_L_M = tuple(-value for value in t_M_B_BY_ARM[LEFT_ROBOT_KEY])
        return T_E_L.multiply(Pose(position_xyz=t_L_M))

    def set_joint_initial_pos(self, joint_pos: Mapping[str, float]) -> None:
        """Update both arms' initial joint positions by joint name."""
        for name in t_M_B_BY_ARM:
            getattr(self.scene_config, name).init_state.joint_pos.update(joint_pos)

    def get_placement_geometry_source(self) -> ArticulationGeometrySpec:
        """Return one arm's USD articulation state; both arms share it."""
        robot = self.scene_config.left_robot
        return ArticulationGeometrySpec(
            usd_path=robot.spawn.usd_path, scale=(1.0, 1.0, 1.0), joint_pos=dict(robot.init_state.joint_pos)
        )

    def get_bounding_box(self, prim_path: str | None = None) -> AxisAlignedBoundingBox:
        """Return rig-relative bounds covering both arms at their configured joint positions."""
        import torch

        arm_bbox = super().get_bounding_box(prim_path=prim_path)
        arm_bboxes = [arm_bbox.translated(t_M_B) for t_M_B in t_M_B_BY_ARM.values()]
        return AxisAlignedBoundingBox(
            min_point=torch.minimum(arm_bboxes[0].min_point, arm_bboxes[1].min_point),
            max_point=torch.maximum(arm_bboxes[0].max_point, arm_bboxes[1].max_point),
        )

    def get_collision_mesh(self) -> trimesh.Trimesh:
        """Return both arms' meshes in the rig frame."""
        import trimesh

        arm_mesh = super().get_collision_mesh()
        return trimesh.util.concatenate([arm_mesh.copy().apply_translation(t) for t in t_M_B_BY_ARM.values()])

    def get_command_body_name(self) -> str:
        return END_EFFECTOR_BODY_NAME

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return END_EFFECTOR_BODY_NAME
