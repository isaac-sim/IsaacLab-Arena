# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Static scene and observation configurations for the selected FR3 asset."""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.physics import PhysicsEvent, PhysicsManager
from isaaclab.sim.schemas.schemas_cfg import JointDriveBaseCfg
from isaaclab.sim.spawners.from_files import from_files
from isaaclab.sim.utils import clone
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics.newton_manager import NewtonManager
from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg
from isaaclab_newton.sim.schemas.schemas_cfg import MujocoJointDrivePropertiesCfg, MujocoRigidBodyPropertiesCfg
from newton.solvers import SolverMuJoCo
from pxr import UsdPhysics

ROBOT_USD_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "industrial_tool_sort"
    / "industrial__fr3_robotiq_2f85"
    / "franka_fr3_robotiq_2f85.usda"
)

ARM_JOINT_NAMES = [f"fr3_joint{index}" for index in range(1, 8)]
GRIPPER_JOINT_NAME = "left_driver_joint"
END_EFFECTOR_BODY_NAME = "robotiq_base"
# The authored driver upper limit is 51.5662 degrees.
GRIPPER_CLOSED_ANGLE = math.radians(51.5662)
_DROID_WORKING_HEIGHT_M = 1.35
_NEWTON_GRAVCOMP_EXCLUDED_BODY_NAMES = {"base", "fr3_link0"}


def _newton_native_actuators_in_use() -> bool:
    """Whether this spawn must seed Newton's position target mode."""
    from isaaclab.sim import SimulationContext

    sim = SimulationContext.instance()
    return sim is not None and bool(getattr(sim.cfg, "use_newton_actuators", False))


def _deactivate_mjc_actuators(root_prim) -> None:
    """Let Isaac Lab own FR3 commands on every physics backend.

    The source USD contains MuJoCo position actuators for standalone MuJoCo
    use. Newton imports them in addition to the Isaac Lab articulation
    actuators, so their zero-valued controls fight the IK joint targets.
    Tendons and equality constraints remain active because they describe the
    Robotiq linkage rather than a second command source.
    """

    pending = list(root_prim.GetChildren())
    while pending:
        prim = pending.pop()
        pending.extend(prim.GetChildren())
        if prim.GetTypeName() == "MjcActuator":
            prim.SetActive(False)


def _ensure_newton_position_drives(root_prim) -> None:
    """Make commanded FR3 drives position-controlled before Newton imports them.

    The asset authors the arm drives with zero stiffness, and Newton must also
    discover the Robotiq driver in position target mode before Isaac Lab applies
    the configured implicit-actuator gains. A small positive import-time
    stiffness establishes that mode; the actuator model replaces it with the
    configured gains during initialization.
    """

    drive_cfg = JointDriveBaseCfg(stiffness=1.0e-3)
    pending = list(reversed(root_prim.GetChildren()))
    while pending:
        prim = pending.pop()
        pending.extend(reversed(prim.GetChildren()))
        if str(prim.GetName()) in (*ARM_JOINT_NAMES, GRIPPER_JOINT_NAME):
            sim_utils.modify_joint_drive_properties(prim.GetPath(), drive_cfg)


def _configure_newton_gravity_compensation(root_prim) -> None:
    """Match Newton's FR3 IK gravity routing without touching gripper joints."""

    body_cfg = MujocoRigidBodyPropertiesCfg(gravcomp=1.0)
    joint_cfg = MujocoJointDrivePropertiesCfg(actuatorgravcomp=True)
    pending = list(root_prim.GetChildren())
    while pending:
        prim = pending.pop()
        pending.extend(prim.GetChildren())
        name = str(prim.GetName())
        has_api = getattr(prim, "HasAPI", None)
        is_rigid_body = prim.GetTypeName() == "PhysicsRigidBody" or (
            has_api is not None and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        )
        if is_rigid_body and name not in _NEWTON_GRAVCOMP_EXCLUDED_BODY_NAMES:
            sim_utils.modify_rigid_body_properties(prim.GetPath(), body_cfg)
        if name in ARM_JOINT_NAMES:
            sim_utils.modify_joint_drive_properties(prim.GetPath(), joint_cfg)


def _configure_newton_builder_gravity(builder) -> None:
    """Route FR3 gravity compensation into Newton's solver custom arrays."""

    required = {"mujoco:gravcomp", "mujoco:jnt_actgravcomp"}
    if not required.issubset(builder.custom_attributes):
        SolverMuJoCo.register_custom_attributes(builder)

    gravcomp = builder.custom_attributes["mujoco:gravcomp"]
    if gravcomp.values is None:
        gravcomp.values = {}
    for body_index, label in enumerate(builder.body_label):
        path = str(label)
        name = path.rsplit("/", 1)[-1]
        if "/Robot/" in path and name not in _NEWTON_GRAVCOMP_EXCLUDED_BODY_NAMES:
            gravcomp.values[body_index] = 1.0

    actuator_gravcomp = builder.custom_attributes["mujoco:jnt_actgravcomp"]
    if actuator_gravcomp.values is None:
        actuator_gravcomp.values = {}
    for joint_index, label in enumerate(builder.joint_label):
        if str(label).rsplit("/", 1)[-1] in ARM_JOINT_NAMES:
            dof_index = int(builder.joint_qd_start[joint_index])
            actuator_gravcomp.values[dof_index] = True


def _on_newton_model_init(_payload) -> None:
    builder = NewtonManager._builder
    if builder is not None:
        # A prebuilt one-environment model can bypass Newton's cloner callback,
        # leaving the actuator adapter without the divisor for its flat layout.
        # Avoid forcing the much slower replication path for this num_envs=1 repo.
        if NewtonManager._num_envs is None:
            NewtonManager._num_envs = max(1, int(builder.world_count))
        _configure_newton_builder_gravity(builder)


def _register_newton_gravity_callback() -> None:
    """Install the builder hook only when the active backend is Newton."""

    if not isinstance(PhysicsManager._cfg, NewtonCfg):
        return
    NewtonManager.register_callback(
        _on_newton_model_init,
        PhysicsEvent.MODEL_INIT,
        name="industrial_fr3_newton_gravity",
        wrap_weak_ref=False,
    )


@clone
def spawn_fr3_without_mjc_actuators(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the FR3 while suppressing duplicate authored command sources."""

    root_prim = from_files._spawn_from_usd_file(
        prim_path,
        cfg.usd_path,
        cfg,
        translation,
        orientation,
        **kwargs,
    )
    _deactivate_mjc_actuators(root_prim)
    if _newton_native_actuators_in_use():
        _ensure_newton_position_drives(root_prim)
    _configure_newton_gravity_compensation(root_prim)
    _register_newton_gravity_callback()
    return root_prim


@configclass
class IndustrialFr3RobotiqSceneCfg:
    """A fixed FR3 in the air, with no stand or auxiliary mount asset."""

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            func=spawn_fr3_without_mjc_actuators,
            usd_path=str(ROBOT_USD_PATH),
            activate_contact_sensors=True,
            # Root fixation is the only generic physics override; Newton gravity
            # routing is applied selectively by the custom spawner above.
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=False),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, _DROID_WORKING_HEIGHT_M),
            joint_pos={
                "fr3_joint1": 0.0,
                "fr3_joint2": -math.pi / 5,
                "fr3_joint3": 0.0,
                "fr3_joint4": -4 * math.pi / 5,
                "fr3_joint5": 0.0,
                "fr3_joint6": 3 * math.pi / 5,
                "fr3_joint7": 0.0,
                GRIPPER_JOINT_NAME: 0.0,
            },
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            "fr3_joint_1_2": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint[1-2]"],
                effort_limit_sim=87.0,
                velocity_limit_sim=2.175,
                stiffness=650.0,
                damping=100.0,
            ),
            "fr3_joint_3_4": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint[3-4]"],
                effort_limit_sim=87.0,
                velocity_limit_sim=2.175,
                stiffness=650.0,
                damping=100.0,
            ),
            "fr3_joint_5_7": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint[5-7]"],
                effort_limit_sim=12.0,
                velocity_limit_sim=2.61,
                stiffness=650.0,
                damping=100.0,
            ),
            "robotiq_driver": ImplicitActuatorCfg(
                joint_names_expr=[GRIPPER_JOINT_NAME],
                stiffness=100.0,
                effort_limit_sim=5.0,
                velocity_limit_sim=2.0,
                damping=10.0,
            ),
        },
    )


@configclass
class IndustrialFr3RobotiqEventCfg:
    """Restore the robot's configured working pose on every reset."""

    reset_robot_joints: EventTermCfg = EventTermCfg(
        func=mdp_isaac_lab.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class IndustrialFr3RobotiqObservationsCfg:
    """DROID-compatible keys for arm, gripper, and Robotiq base state."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp_isaac_lab.last_action)
        robot_joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            from .observations import arm_joint_pos, ee_pos, ee_quat, gripper_pos

            self.joint_pos = ObsTerm(func=arm_joint_pos)
            self.gripper_pos = ObsTerm(func=gripper_pos)
            self.eef_pos = ObsTerm(func=ee_pos)
            self.eef_quat = ObsTerm(func=ee_quat)
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
