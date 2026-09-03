# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Flexiv Rizon embodiments and teleoperation configuration."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from copy import deepcopy

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab_assets import FLEXIV_RIZON4S_GRAV_GRIPPER_CFG

from isaaclab_arena.assets.device_library import KeyboardCfg, SpaceMouseCfg
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.relations.collision_mode import CollisionMode
from isaaclab_arena.utils.pose import Pose

RIZON_ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7")
"""Ordered arm joints controlled by differential inverse kinematics."""

RIZON_GRAV_GRIPPER_MIMIC_SIGNS = {
    "finger_joint": 1.0,
    "left_inner_knuckle_joint": 1.0,
    "right_inner_knuckle_joint": 1.0,
    "right_outer_knuckle_joint": 1.0,
    "left_outer_finger_joint": -1.0,
    "right_outer_finger_joint": -1.0,
}
"""Joint directions for the Grav gripper's mechanically coupled joints."""

RIZON_GRIPPER_OPEN_POSITION = 0.5
RIZON_GRIPPER_CLOSE_POSITION = -0.1
RIZON_GRIPPER_MAX_TARGET_STEP = 0.03


def get_rizon_gripper_command(position: float) -> dict[str, float]:
    """Return a complete Grav gripper command for one drive-joint position."""
    return {name: sign * position for name, sign in RIZON_GRAV_GRIPPER_MIMIC_SIGNS.items()}


class PersistentTargetDifferentialIKAction(DifferentialInverseKinematicsAction):
    """Accumulate relative commands from the previous target so zero input holds pose."""

    def __init__(self, cfg: DifferentialInverseKinematicsActionCfg, env) -> None:
        super().__init__(cfg, env)
        self._target_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def process_actions(self, actions: torch.Tensor) -> None:
        """Convert a delta-pose action into a persistent end-effector target."""
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

        current_position, current_orientation = self._compute_frame_pose()
        use_target = self._target_initialized.unsqueeze(-1)
        base_position = torch.where(use_target, self._ik_controller.ee_pos_des, current_position)
        base_orientation = torch.where(use_target, self._ik_controller.ee_quat_des, current_orientation)
        self._ik_controller.set_command(self._processed_actions, base_position, base_orientation)
        self._target_initialized[:] = True

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear persistent targets for the reset environments."""
        super().reset(env_ids)
        if env_ids is None:
            self._target_initialized[:] = False
        else:
            self._target_initialized[env_ids] = False


class RateLimitedBinaryJointPositionAction(BinaryJointPositionAction):
    """Rate-limit binary gripper targets and interpret zero input as hold."""

    def __init__(self, cfg: BinaryJointPositionActionCfg, env) -> None:
        super().__init__(cfg, env)
        self._smoothed_target = self._close_command.unsqueeze(0).repeat(self.num_envs, 1)

    def process_actions(self, actions: torch.Tensor) -> None:
        """Select an open/close target, preserving the current target for zero input."""
        self._raw_actions[:] = actions
        if actions.dtype == torch.bool:
            target = torch.where(actions, self._open_command, self._close_command)
        else:
            target = torch.where(
                actions > 0.0,
                self._open_command,
                torch.where(actions < 0.0, self._close_command, self._smoothed_target),
            )
        if self.cfg.clip is not None:
            target = torch.clamp(target, min=self._clip[:, :, 0], max=self._clip[:, :, 1])
        target_delta = torch.clamp(
            target - self._smoothed_target,
            min=-RIZON_GRIPPER_MAX_TARGET_STEP,
            max=RIZON_GRIPPER_MAX_TARGET_STEP,
        )
        self._smoothed_target += target_delta
        self._processed_actions[:] = self._smoothed_target

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Restore the closed target for the reset environments."""
        super().reset(env_ids)
        if env_ids is None:
            self._smoothed_target[:] = self._close_command
        else:
            self._smoothed_target[env_ids] = self._close_command


class _InitiallyClosedSe3Keyboard(Se3Keyboard):
    """SE(3) keyboard whose binary gripper state begins closed."""

    def __init__(self, cfg: Se3KeyboardCfg) -> None:
        super().__init__(cfg)
        self._close_gripper = True

    def reset(self) -> None:
        """Clear motion input and restore the closed gripper state."""
        super().reset()
        self._close_gripper = True


class _InitiallyClosedSe3SpaceMouse(Se3SpaceMouse):
    """SE(3) SpaceMouse whose binary gripper state begins closed."""

    def __init__(self, cfg: Se3SpaceMouseCfg) -> None:
        super().__init__(cfg)
        self._close_gripper = True

    def reset(self) -> None:
        """Clear motion input and restore the closed gripper state."""
        super().reset()
        self._close_gripper = True


class InitiallyClosedKeyboardCfg(KeyboardCfg):
    """Arena keyboard configuration for an object that starts grasped."""

    def get_device_cfg(self, pipeline_builder=None, embodiment=None) -> Se3KeyboardCfg:
        """Return a keyboard device whose first gripper command remains closed."""
        return Se3KeyboardCfg(
            pos_sensitivity=self.pos_sensitivity,
            rot_sensitivity=self.rot_sensitivity,
            gripper_term=True,
            sim_device=self.sim_device,
            class_type=_InitiallyClosedSe3Keyboard,
        )


class InitiallyClosedSpaceMouseCfg(SpaceMouseCfg):
    """Arena SpaceMouse configuration for an object that starts grasped."""

    def get_device_cfg(self, pipeline_builder=None, embodiment=None) -> Se3SpaceMouseCfg:
        """Return a SpaceMouse device whose first gripper command remains closed."""
        return Se3SpaceMouseCfg(
            pos_sensitivity=self.pos_sensitivity,
            rot_sensitivity=self.rot_sensitivity,
            gripper_term=True,
            sim_device=self.sim_device,
            class_type=_InitiallyClosedSe3SpaceMouse,
        )


@register_asset
class Rizon4sGravDifferentialIKNewtonEmbodiment(EmbodimentBase):
    """Newton-tuned Flexiv Rizon 4S with relative differential-IK control."""

    name = "rizon4s_grav_differential_ik_newton"
    tags = ["embodiment", "rizon", "newton"]
    default_arm_mode = ArmMode.SINGLE_ARM

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        collision_mode: CollisionMode | str | None = None,
    ) -> None:
        super().__init__(
            enable_cameras=enable_cameras,
            initial_pose=initial_pose,
            concatenate_observation_terms=concatenate_observation_terms,
            arm_mode=arm_mode,
            collision_mode=collision_mode,
        )
        self.scene_config = RizonSceneCfg(robot=_make_rizon_newton_robot_cfg())
        self.action_config = RizonDifferentialIKActionsCfg()
        self.observation_config = RizonObservationsCfg()
        self.observation_config.policy.concatenate_terms = concatenate_observation_terms

    def get_command_body_name(self) -> str:
        return "flange"

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return "flange"


@configclass
class RizonSceneCfg:
    """Scene entries supplied by a Rizon embodiment."""

    robot: ArticulationCfg | None = None


@configclass
class RizonDifferentialIKActionsCfg:
    """Relative flange-pose and binary Grav-gripper actions."""

    arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
        class_type=PersistentTargetDifferentialIKAction,
        asset_name="robot",
        joint_names=list(RIZON_ARM_JOINT_NAMES),
        body_name="flange",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),
        scale=0.01,
    )
    gripper_action: ActionTermCfg = BinaryJointPositionActionCfg(
        class_type=RateLimitedBinaryJointPositionAction,
        asset_name="robot",
        joint_names=list(RIZON_GRAV_GRIPPER_MIMIC_SIGNS),
        open_command_expr=get_rizon_gripper_command(RIZON_GRIPPER_OPEN_POSITION),
        close_command_expr=get_rizon_gripper_command(RIZON_GRIPPER_CLOSE_POSITION),
    )


@configclass
class RizonObservationsCfg:
    """Robot state observations exposed by the Rizon embodiment."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


def _make_rizon_newton_robot_cfg() -> ArticulationCfg:
    """Return the upstream Rizon asset with Newton-specific drive tuning."""
    from isaaclab_newton.sim.schemas import MujocoJointDrivePropertiesCfg

    robot_cfg = deepcopy(FLEXIV_RIZON4S_GRAV_GRIPPER_CFG)
    robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    robot_cfg.spawn.make_uninstanceable = True
    robot_cfg.spawn.joint_drive_props = MujocoJointDrivePropertiesCfg(actuatorgravcomp=True)
    robot_cfg.spawn.collision_props = sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0)

    robot_cfg.init_state.joint_pos.update(get_rizon_gripper_command(RIZON_GRIPPER_CLOSE_POSITION))
    robot_cfg.actuators["shoulder"].joint_effort_limit = 123.0
    robot_cfg.actuators["shoulder"].stiffness = 6000.0
    robot_cfg.actuators["shoulder"].damping = 108.5
    robot_cfg.actuators["elbow"].joint_effort_limit = 64.0
    robot_cfg.actuators["elbow"].stiffness = 4200.0
    robot_cfg.actuators["elbow"].damping = 90.7
    robot_cfg.actuators["wrist"].joint_effort_limit = 39.0
    robot_cfg.actuators["wrist"].stiffness = 1500.0
    robot_cfg.actuators["wrist"].damping = 54.2
    robot_cfg.actuators["gripper_drive"] = ImplicitActuatorCfg(
        joint_names_expr=["finger_joint"],
        joint_effort_limit=200.0,
        joint_velocity_limit=0.75,
        stiffness=2000.0,
        damping=50.0,
        friction=0.0,
        armature=0.1,
    )
    robot_cfg.actuators["gripper_passive"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_knuckle_joint", ".*_outer_finger_joint"],
        joint_effort_limit=20.0,
        joint_velocity_limit=0.75,
        stiffness=2000.0,
        damping=50.0,
        friction=0.0,
        armature=0.05,
    )
    return robot_cfg
