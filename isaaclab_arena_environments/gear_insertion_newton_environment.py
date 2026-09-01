# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Self-contained DROID gear-insertion environment using Newton.

This is an Arena-local bring-up of Isaac-cap PR 24 (merge commit
``35c2a7a9acb3604953b4ec110e2ff0c4469f5d55``). It intentionally keeps the
task-specific implementation in one module so the environment can be exercised
without installing ``industrial_benchmark``.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.droid.droid import DroidEmbodimentBase
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg


GEAR_TYPES = ("gear_small", "gear_medium", "gear_large")
GEAR_ASSET_NAMES = {
    "gear_small": "factory_gear_small",
    "gear_medium": "factory_gear_medium",
    "gear_large": "factory_gear_large",
}
GEAR_PRIM_NAMES = {
    "gear_small": "FactoryGearSmall",
    "gear_medium": "FactoryGearMedium",
    "gear_large": "FactoryGearLarge",
}

DROID_ARM_JOINT_NAMES = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
)
DROID_IK_SEED_JOINT_POSITIONS = (0.98, -0.47, -1.73, -1.42, -1.28, 2.71, 1.35)
DROID_GRIPPER_MIMIC_SIGNS = {
    "finger_joint": 1.0,
    "left_inner_finger_joint": -1.0,
    "left_inner_finger_knuckle_joint": -1.0,
    "right_outer_knuckle_joint": 1.0,
    "right_inner_finger_joint": 1.0,
    "right_inner_finger_knuckle_joint": -1.0,
}
DROID_GRIPPER_JOINT_NAMES = tuple(DROID_GRIPPER_MIMIC_SIGNS)
DROID_GRIPPER_OPEN_COMMAND = dict.fromkeys(DROID_GRIPPER_JOINT_NAMES, 0.0)
DROID_GRIPPER_CLOSE_COMMAND = {name: sign * 0.7 for name, sign in DROID_GRIPPER_MIMIC_SIGNS.items()}

DROID_BASE_GEAR_POSITION = (0.48099045, -0.0667565, 0.08975)
DROID_BASE_GEAR_ROTATION_XYZW = (0.0, 0.0, 0.70711, -0.70711)
NEWTON_GEAR_OFFSETS = {
    "gear_small": (0.0823685, 0.0, 0.0),
    "gear_medium": (0.0366185, 0.0, 0.0),
    "gear_large": (-0.0391315, 0.0, 0.0),
}
NEWTON_GEAR_INITIAL_POSITIONS = {
    gear_type: (
        DROID_BASE_GEAR_POSITION[0],
        DROID_BASE_GEAR_POSITION[1] - offset[0],
        DROID_BASE_GEAR_POSITION[2] + 0.0075,
    )
    for gear_type, offset in NEWTON_GEAR_OFFSETS.items()
}

MAPLE_TABLE_TOP_Z = 0.003000684082508087
MAPLE_TABLE_POSITION = (0.0, 0.0, 0.071 - MAPLE_TABLE_TOP_Z)
MAPLE_TABLE_TOP_COLLISION_SIZE = (0.7, 1.0)
MAPLE_TABLE_TOP_COLLISION_THICKNESS = 0.02
MAPLE_TABLE_TOP_COLLISION_POSITION = (0.5485909044742584, 0.02206302247941494, 0.071)
NEWTON_GEAR_CONTACT_OFFSET = 1.0e-4
NEWTON_GEAR_TABLETOP_Z = MAPLE_TABLE_TOP_COLLISION_POSITION[2] + 0.01875 + 2.0 * NEWTON_GEAR_CONTACT_OFFSET
NEWTON_GEAR_TABLETOP_POSITIONS = {
    "gear_small": (
        MAPLE_TABLE_TOP_COLLISION_POSITION[0] - 0.15,
        MAPLE_TABLE_TOP_COLLISION_POSITION[1] + 0.12,
        NEWTON_GEAR_TABLETOP_Z,
    ),
    "gear_medium": (
        MAPLE_TABLE_TOP_COLLISION_POSITION[0] + 0.15,
        MAPLE_TABLE_TOP_COLLISION_POSITION[1] + 0.12,
        NEWTON_GEAR_TABLETOP_Z,
    ),
    "gear_large": (
        MAPLE_TABLE_TOP_COLLISION_POSITION[0],
        MAPLE_TABLE_TOP_COLLISION_POSITION[1] + 0.32,
        NEWTON_GEAR_TABLETOP_Z,
    ),
}
NEWTON_GEAR_TABLETOP_ROTATION_XYZW = (0.0, 0.0, 0.0, 1.0)

GEAR_GRASP_OFFSETS = {
    "gear_small": (-0.16245, 0.0, 0.0),
    "gear_medium": (-0.16085, 0.0, 0.0),
    "gear_large": (-0.15985, 0.0, 0.0),
}
GEAR_GRASP_WIDTHS = {"gear_small": 0.50, "gear_medium": 0.30, "gear_large": 0.24}
GEAR_CLOSE_WIDTHS = {"gear_small": 0.65, "gear_medium": 0.461, "gear_large": 0.412}
_HALF_GEAR_SEGMENT = math.pi / 6.0
GEAR_GRASP_ROTATION_XYZW = (
    -math.sin(_HALF_GEAR_SEGMENT / 2.0) / math.sqrt(2.0),
    math.cos(_HALF_GEAR_SEGMENT / 2.0) / math.sqrt(2.0),
    math.sin(_HALF_GEAR_SEGMENT / 2.0) / math.sqrt(2.0),
    math.cos(_HALF_GEAR_SEGMENT / 2.0) / math.sqrt(2.0),
)

GEAR_GREEN_DIFFUSE_COLOR = (0.0, 0.8, 0.2)
GEAR_GREEN_VISUAL_MATERIAL_PATH = "green_material"
MAPLE_TABLE_TOP_COLLISION_COLOR = (0.43, 0.28, 0.15)
MAPLE_TABLE_LEG_RENDER_COLOR = (0.2, 0.22, 0.24)
NEWTON_GEAR_MAX_DEPENETRATION_VELOCITY = 1.0
GEAR_STATIC_FRICTION = 0.75
FINGER_STATIC_FRICTION = 2.0

_DROID_FINGERTIP_COLLISION_BOUNDS = {
    "left_inner_finger": ((0.1111, 0.0425, -0.011), (0.1491, 0.06166, 0.011)),
    "right_inner_finger": ((0.1111, -0.06166, -0.011), (0.1491, -0.0425, 0.011)),
}


@dataclass
class GearInsertionNewtonEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the self-contained Newton gear-insertion environment."""

    gear_type: Literal["gear_small", "gear_medium", "gear_large"] = "gear_small"
    """Gear variant grasped at reset: gear_small, gear_medium, or gear_large."""

    episode_length_s: float = 66.6
    """Episode timeout in seconds."""

    def __post_init__(self) -> None:
        assert self.gear_type in GEAR_TYPES, f"Unsupported gear_type: {self.gear_type!r}"
        assert self.episode_length_s > 0.0, "episode_length_s must be positive."


@register_environment
class GearInsertionNewtonEnvironment(ArenaEnvironmentFactory[GearInsertionNewtonEnvironmentCfg]):
    """Compose PR 24's DROID, Factory gears, maple table, task, and Newton tuning."""

    name = "gear_insertion_newton"
    _legacy_argparse_cfg_type = GearInsertionNewtonEnvironmentCfg

    def build(self, cfg: GearInsertionNewtonEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the Arena-local gear-insertion prototype."""
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

        embodiment = self.asset_registry.get_asset_by_name("droid_differential_ik")(enable_cameras=cfg.enable_cameras)
        _configure_newton_droid_embodiment(embodiment)
        scene = _build_gear_insertion_scene(self.asset_registry)
        task = _make_gear_insertion_task(cfg)
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            env_cfg_callback=_make_env_cfg_callback(cfg),
        )


_NEWTON_DROID_IK_ACTION_TYPE = None
_GEAR_GRIPPER_ACTION_TYPE = None
_FINITE_DIFFERENCE_GRASP_RESET_TYPE = None


def _get_newton_droid_ik_action_type():
    """Return the lazily defined Newton-specific DROID differential-IK action."""
    global _NEWTON_DROID_IK_ACTION_TYPE
    if _NEWTON_DROID_IK_ACTION_TYPE is None:
        from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

        class NewtonDroidDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
            """Use the DROID arm columns from Newton's fixed-base Jacobian layout."""

            def __init__(self, cfg, env):
                super().__init__(cfg, env)
                self._jacobi_joint_ids = self._joint_ids

        _NEWTON_DROID_IK_ACTION_TYPE = NewtonDroidDifferentialInverseKinematicsAction
    return _NEWTON_DROID_IK_ACTION_TYPE


def _get_gear_gripper_action_type():
    """Return the lazily defined gear-aware, slew-limited gripper action."""
    global _GEAR_GRIPPER_ACTION_TYPE
    if _GEAR_GRIPPER_ACTION_TYPE is None:
        import torch

        from isaaclab_arena.embodiments.droid.actions import BinaryJointPositionZeroToOneAction

        class GearInsertionBinaryJointPositionAction(BinaryJointPositionZeroToOneAction):
            """Use standard teleop gripper commands while holding neutral actions."""

            def __init__(self, cfg, env):
                super().__init__(cfg, env)
                gear_type = env.cfg.gear_type
                command_sign = torch.sign(self._close_command)
                self._gear_open_command = GEAR_GRASP_WIDTHS[gear_type] * command_sign
                self._gear_close_command = GEAR_CLOSE_WIDTHS[gear_type] * command_sign
                self._max_command_step = 0.5 * env.step_dt

            def process_actions(self, actions: torch.Tensor) -> None:
                previous_commands = self._processed_actions.clone()
                self._raw_actions[:] = actions
                if actions.dtype == torch.bool:
                    desired_commands = torch.where(
                        actions,
                        self._gear_close_command,
                        self._gear_open_command,
                    )
                else:
                    close_mask = actions < -0.5
                    open_mask = actions > 0.5
                    desired_commands = torch.where(
                        close_mask,
                        self._gear_close_command,
                        torch.where(open_mask, self._gear_open_command, previous_commands),
                    )
                command_delta = torch.clamp(
                    desired_commands - previous_commands,
                    min=-self._max_command_step,
                    max=self._max_command_step,
                )
                self._processed_actions = previous_commands + command_delta
                if self.cfg.clip is not None:
                    self._processed_actions = torch.clamp(
                        self._processed_actions,
                        min=self._clip[:, :, 0],
                        max=self._clip[:, :, 1],
                    )

            def reset(self, env_ids=None) -> None:
                import warp as wp

                super().reset(env_ids)
                if env_ids is None:
                    env_ids = slice(None)
                joint_position = wp.to_torch(self._asset.data.joint_pos)
                joint_ids = wp.to_torch(self._joint_ids).long()
                self._processed_actions[env_ids] = joint_position[env_ids].index_select(1, joint_ids)

        _GEAR_GRIPPER_ACTION_TYPE = GearInsertionBinaryJointPositionAction
    return _GEAR_GRIPPER_ACTION_TYPE


def _configure_newton_droid_embodiment(embodiment: DroidEmbodimentBase) -> None:
    """Apply the PR's Newton settings to one new DROID embodiment instance."""
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab_newton.sim.schemas import NewtonMaterialPropertiesCfg

    robot_cfg = deepcopy(embodiment.scene_config.robot)
    embodiment.scene_config.robot = robot_cfg
    robot_cfg.actuators["gripper"] = ImplicitActuatorCfg(
        joint_names_expr=list(DROID_GRIPPER_JOINT_NAMES),
        joint_effort_limit=5.0,
        joint_velocity_limit=1.0,
        stiffness=20.0,
        damping=5.0,
        armature=0.1,
    )
    robot_cfg.spawn.usd_path = _ensure_newton_compatible_droid_usd(robot_cfg.spawn.usd_path)
    robot_cfg.spawn.rigid_props.disable_gravity = False
    robot_cfg.spawn.physics_material = NewtonMaterialPropertiesCfg(
        static_friction=FINGER_STATIC_FRICTION,
        dynamic_friction=FINGER_STATIC_FRICTION,
        restitution=0.0,
    )
    robot_cfg.init_state.joint_pos.update(dict(zip(DROID_ARM_JOINT_NAMES, DROID_IK_SEED_JOINT_POSITIONS, strict=True)))

    embodiment.scene_config.ee_frame = deepcopy(embodiment.scene_config.ee_frame)
    embodiment.scene_config.ee_frame.target_frames[0].prim_path = "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link"
    embodiment.action_config.arm_action = deepcopy(embodiment.action_config.arm_action)
    embodiment.action_config.arm_action.body_name = "base_link"
    embodiment.action_config.arm_action.class_type = _get_newton_droid_ik_action_type()


def _make_env_cfg_callback(cfg: GearInsertionNewtonEnvironmentCfg):
    """Return the environment-specific Newton configuration callback."""

    def configure_gear_insertion_newton(
        env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
    ) -> IsaacLabArenaManagerBasedRLEnvCfg:
        from isaaclab.devices.device_base import DevicesCfg
        from isaaclab.devices.spacemouse import Se3SpaceMouseCfg

        from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import ArenaPhysicsCfg

        env_cfg.viewer.eye = (1.6, 1.2, 1.0)
        env_cfg.viewer.lookat = (0.55, 0.05, 0.08)
        env_cfg.decimation = 4
        env_cfg.sim.render_interval = 4
        env_cfg.sim.dt = 1.0 / 120.0
        env_cfg.sim.physics = ArenaPhysicsCfg().newton
        env_cfg.sim.physics.num_substeps = 12
        env_cfg.sim.physics.default_shape_cfg.gap = 0.0
        env_cfg.sim.physics.default_shape_cfg.mu = GEAR_STATIC_FRICTION
        env_cfg.sim.physics.solver_cfg.ccd_iterations = 35
        env_cfg.sim.physics.solver_cfg.use_mujoco_contacts = False
        env_cfg.sim.physics.solver_cfg.njmax = 512
        env_cfg.scene.env_spacing = 1.5
        env_cfg.scene.replicate_physics = True
        env_cfg.events.randomize_franka_joint_state = None

        env_cfg.actions.gripper_action.class_type = _get_gear_gripper_action_type()
        env_cfg.actions.gripper_action.joint_names = list(DROID_GRIPPER_JOINT_NAMES)
        env_cfg.actions.gripper_action.open_command_expr = DROID_GRIPPER_OPEN_COMMAND
        env_cfg.actions.gripper_action.close_command_expr = DROID_GRIPPER_CLOSE_COMMAND
        env_cfg.teleop_devices = DevicesCfg(
            devices={
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=env_cfg.sim.device,
                )
            }
        )
        env_cfg.gear_type = cfg.gear_type
        return env_cfg

    return configure_gear_insertion_newton


def _reset_gear_layout(env, env_ids, gear_type: str) -> None:
    """Place all gear variants on the table in the PR's deterministic play layout."""
    import torch

    orientation = torch.tensor(
        NEWTON_GEAR_TABLETOP_ROTATION_XYZW,
        device=env.device,
        dtype=torch.float32,
    ).repeat(len(env_ids), 1)
    zero_velocity = torch.zeros((len(env_ids), 6), device=env.device)
    for candidate in GEAR_TYPES:
        asset = env.scene[GEAR_ASSET_NAMES[candidate]]
        position = torch.tensor(
            NEWTON_GEAR_TABLETOP_POSITIONS[candidate],
            device=env.device,
            dtype=torch.float32,
        ).repeat(len(env_ids), 1)
        position += env.scene.env_origins[env_ids]
        asset.write_root_pose_to_sim_index(
            root_pose=torch.cat((position, orientation), dim=-1),
            env_ids=env_ids,
        )
        asset.write_root_velocity_to_sim_index(root_velocity=zero_velocity, env_ids=env_ids)


def _get_finite_difference_grasp_reset_type():
    """Return the lazily defined deterministic DROID grasp-reset event."""
    global _FINITE_DIFFERENCE_GRASP_RESET_TYPE
    if _FINITE_DIFFERENCE_GRASP_RESET_TYPE is None:
        import torch

        import isaaclab.utils.math as math_utils
        from isaaclab.managers import EventTermCfg, ManagerTermBase

        class FiniteDifferenceGearGraspReset(ManagerTermBase):
            """Place DROID at the selected gear using finite-difference IK."""

            def __init__(self, cfg: EventTermCfg, env):
                super().__init__(cfg, env)
                self.robot = env.scene["robot"]
                self.gear_type = cfg.params["gear_type"]
                self.gear = env.scene[GEAR_ASSET_NAMES[self.gear_type]]
                body_ids, _ = self.robot.find_bodies(["base_link"])
                assert len(body_ids) == 1, "DROID base_link body was not found."
                self.ee_body_id = body_ids[0]
                self.arm_joint_ids, _ = self.robot.find_joints(list(DROID_ARM_JOINT_NAMES), preserve_order=True)
                self.gripper_joint_ids, _ = self.robot.find_joints(list(DROID_GRIPPER_JOINT_NAMES), preserve_order=True)
                assert len(self.arm_joint_ids) == len(DROID_ARM_JOINT_NAMES)
                assert len(self.gripper_joint_ids) == len(DROID_GRIPPER_JOINT_NAMES)
                self.grasp_offset = torch.tensor(GEAR_GRASP_OFFSETS[self.gear_type], device=env.device)
                self.grasp_rotation = torch.tensor(GEAR_GRASP_ROTATION_XYZW, device=env.device)
                self.gripper_signs = torch.tensor(tuple(DROID_GRIPPER_MIMIC_SIGNS.values()), device=env.device)

            @staticmethod
            def _write_joint_state(env, robot, env_ids, joint_position) -> None:
                robot.write_joint_position_to_sim_index(position=joint_position, env_ids=env_ids)
                robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_position), env_ids=env_ids)
                env.sim.forward()

            def __call__(
                self,
                env,
                env_ids,
                gear_type: str,
                max_iterations: int = 50,
                position_threshold: float = 1.0e-3,
                rotation_threshold: float = 1.0e-3,
            ) -> None:
                import warp as wp

                assert gear_type == self.gear_type
                gear_position = wp.to_torch(self.gear.data.root_link_pos_w)[env_ids]
                gear_rotation = wp.to_torch(self.gear.data.root_link_quat_w)[env_ids]
                target_rotation = math_utils.quat_mul(
                    gear_rotation,
                    self.grasp_rotation.repeat(len(env_ids), 1),
                )
                target_position = gear_position + math_utils.quat_apply(
                    target_rotation,
                    self.grasp_offset.repeat(len(env_ids), 1),
                )

                joint_position = wp.to_torch(self.robot.data.joint_pos)[env_ids].clone()
                self._write_joint_state(env, self.robot, env_ids, joint_position)
                for _ in range(max_iterations):
                    ee_position = wp.to_torch(self.robot.data.body_pos_w)[env_ids, self.ee_body_id].clone()
                    ee_rotation = wp.to_torch(self.robot.data.body_quat_w)[env_ids, self.ee_body_id].clone()
                    position_error, rotation_error = math_utils.compute_pose_error(
                        ee_position,
                        ee_rotation,
                        target_position,
                        target_rotation,
                    )
                    if torch.all(torch.linalg.vector_norm(position_error, dim=-1) < position_threshold) and torch.all(
                        torch.linalg.vector_norm(rotation_error, dim=-1) < rotation_threshold
                    ):
                        break

                    perturbation = 1.0e-3
                    jacobian = torch.empty(
                        len(env_ids),
                        6,
                        len(self.arm_joint_ids),
                        device=env.device,
                        dtype=joint_position.dtype,
                    )
                    for column, joint_id in enumerate(self.arm_joint_ids):
                        perturbed_position = joint_position.clone()
                        perturbed_position[:, joint_id] += perturbation
                        self._write_joint_state(env, self.robot, env_ids, perturbed_position)
                        jacobian[:, :3, column] = (
                            wp.to_torch(self.robot.data.body_pos_w)[env_ids, self.ee_body_id] - ee_position
                        ) / perturbation
                        _, orientation_delta = math_utils.compute_pose_error(
                            ee_position,
                            ee_rotation,
                            wp.to_torch(self.robot.data.body_pos_w)[env_ids, self.ee_body_id],
                            wp.to_torch(self.robot.data.body_quat_w)[env_ids, self.ee_body_id],
                        )
                        jacobian[:, 3:, column] = orientation_delta / perturbation

                    self._write_joint_state(env, self.robot, env_ids, joint_position)
                    jacobian_transpose = jacobian.transpose(1, 2)
                    damping = 0.01 * torch.eye(6, device=env.device, dtype=joint_position.dtype)
                    pose_error = torch.cat((position_error, rotation_error), dim=-1).unsqueeze(-1)
                    joint_delta = (
                        jacobian_transpose
                        @ torch.linalg.solve(
                            jacobian @ jacobian_transpose + damping,
                            pose_error,
                        )
                    ).squeeze(-1)
                    limits = wp.to_torch(self.robot.data.joint_pos_limits)[env_ids][:, self.arm_joint_ids]
                    current_arm_position = joint_position[:, self.arm_joint_ids]
                    joint_position[:, self.arm_joint_ids] = torch.clamp(
                        current_arm_position + torch.clamp(joint_delta, -0.2, 0.2),
                        min=limits[:, :, 0],
                        max=limits[:, :, 1],
                    )
                    self._write_joint_state(env, self.robot, env_ids, joint_position)

                grasp_width = GEAR_GRASP_WIDTHS[self.gear_type]
                joint_position[:, self.gripper_joint_ids] = grasp_width * self.gripper_signs
                self._write_joint_state(env, self.robot, env_ids, joint_position)
                self.robot.set_joint_position_target_index(target=joint_position, env_ids=env_ids)
                self.robot.set_joint_velocity_target_index(target=torch.zeros_like(joint_position), env_ids=env_ids)

        _FINITE_DIFFERENCE_GRASP_RESET_TYPE = FiniteDifferenceGearGraspReset
    return _FINITE_DIFFERENCE_GRASP_RESET_TYPE


def _selected_gear_inserted(env, gear_type: str) -> object:
    """Return whether the selected gear is aligned, upright, and settled on its peg."""
    import torch

    import isaaclab.utils.math as math_utils
    import warp as wp

    gear = env.scene[GEAR_ASSET_NAMES[gear_type]]
    base = env.scene["factory_gear_base"]
    gear_position = wp.to_torch(gear.data.root_link_pos_w)
    gear_rotation = wp.to_torch(gear.data.root_link_quat_w)
    gear_velocity = wp.to_torch(gear.data.root_com_vel_w)
    base_position = wp.to_torch(base.data.root_link_pos_w)
    base_rotation = wp.to_torch(base.data.root_link_quat_w)
    offset = torch.tensor(NEWTON_GEAR_OFFSETS[gear_type], device=env.device).repeat(env.num_envs, 1)
    target_position = base_position + math_utils.quat_apply(base_rotation, offset)
    xy_error = torch.linalg.vector_norm(gear_position[:, :2] - target_position[:, :2], dim=-1)
    z_error = torch.abs((gear_position[:, 2] - base_position[:, 2]) - 0.0075)
    up_axis = torch.tensor((0.0, 0.0, 1.0), device=env.device).repeat(env.num_envs, 1)
    upright = (math_utils.quat_apply(gear_rotation, up_axis) * math_utils.quat_apply(base_rotation, up_axis)).sum(
        dim=-1
    ) >= math.cos(math.radians(15.0))
    return (
        (xy_error <= 0.015)
        & (z_error <= 0.01)
        & upright
        & (torch.linalg.vector_norm(gear_velocity[:, :3], dim=-1) <= 0.05)
        & (torch.linalg.vector_norm(gear_velocity[:, 3:], dim=-1) <= 0.5)
    )


def _make_gear_insertion_task(cfg: GearInsertionNewtonEnvironmentCfg):
    """Create the self-contained reset, termination, and viewer task."""
    from dataclasses import MISSING

    import isaaclab.envs.mdp as mdp
    from isaaclab.envs.common import ViewerCfg
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    from isaaclab.utils.configclass import configclass

    from isaaclab_arena.embodiments.common.arm_mode import ArmMode
    from isaaclab_arena.metrics.metric_base import MetricBase
    from isaaclab_arena.metrics.success_rate import SuccessRateMetric
    from isaaclab_arena.tasks.task_base import TaskBase

    @configclass
    class GearInsertionEventsCfg:
        reset_scene: EventTerm = EventTerm(
            func=mdp.reset_scene_to_default,
            mode="reset",
            params={"reset_joint_targets": True},
        )
        reset_gear_layout: EventTerm = EventTerm(
            func=_reset_gear_layout,
            mode="reset",
            params={"gear_type": cfg.gear_type},
        )
        reset_robot_to_grasp: EventTerm = EventTerm(
            func=_get_finite_difference_grasp_reset_type(),
            mode="reset",
            params={"gear_type": cfg.gear_type},
        )

    @configclass
    class GearInsertionTerminationsCfg:
        time_out: DoneTerm = DoneTerm(func=mdp.time_out, time_out=True)
        success: DoneTerm = MISSING

    class GearInsertionTask(TaskBase):
        """Insert the selected Factory gear onto its matching base peg."""

        def __init__(self) -> None:
            super().__init__(
                episode_length_s=cfg.episode_length_s,
                task_description=f"Insert the selected {cfg.gear_type} onto its matching peg.",
            )
            self.events_cfg = GearInsertionEventsCfg()
            self.terminations_cfg = GearInsertionTerminationsCfg(
                success=DoneTerm(
                    func=_selected_gear_inserted,
                    params={"gear_type": cfg.gear_type},
                )
            )

        def get_scene_cfg(self):
            return None

        def get_termination_cfg(self):
            return self.terminations_cfg

        def get_events_cfg(self):
            return self.events_cfg

        def get_mimic_env_cfg(self, arm_mode: ArmMode):
            return None

        def get_metrics(self) -> list[MetricBase]:
            return [SuccessRateMetric()]

        def get_viewer_cfg(self) -> ViewerCfg:
            return ViewerCfg(eye=(1.6, 1.2, 1.0), lookat=(0.55, 0.05, 0.08))

    return GearInsertionTask()


def _build_gear_insertion_scene(asset_registry):
    """Build the PR's maple-table scene with three Factory gear variants."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path
    from isaaclab_newton.sim.schemas import NewtonMaterialPropertiesCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.object_library import DomeLight
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.relations.relations import IsAnchor
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    class GearInsertionRigidObject(Object):
        """Rigid object wrapper preserving the source task's sensor setting."""

        def _generate_rigid_cfg(self) -> RigidObjectCfg:
            assert self.object_type == ObjectType.RIGID
            object_cfg = RigidObjectCfg(
                prim_path=self.prim_path,
                spawn=self._get_spawn_cfg(activate_contact_sensors=False),
                **self.asset_cfg_addon,
            )
            return self._add_initial_pose_to_cfg(object_cfg)

    gear_material = NewtonMaterialPropertiesCfg(
        static_friction=GEAR_STATIC_FRICTION,
        dynamic_friction=GEAR_STATIC_FRICTION,
        restitution=0.0,
    )

    def make_gear(
        gear_type: str,
        pose: Pose,
        *,
        kinematic_enabled: bool = False,
        visual_diffuse_color: tuple[float, float, float] | None = None,
    ) -> Object:
        usd_leaf = "factory_gear_base" if gear_type == "gear_base" else GEAR_ASSET_NAMES[gear_type]
        prim_name = "FactoryGearBase" if gear_type == "gear_base" else GEAR_PRIM_NAMES[gear_type]
        instance_name = "factory_gear_base" if gear_type == "gear_base" else GEAR_ASSET_NAMES[gear_type]
        spawn_cfg_addon = {
            "func": _spawn_newton_mesh_collision_usd,
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                kinematic_enabled=kinematic_enabled,
                max_depenetration_velocity=NEWTON_GEAR_MAX_DEPENETRATION_VELOCITY,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            "mass_props": sim_utils.MassPropertiesCfg(mass=None),
            "collision_props": sim_utils.CollisionPropertiesCfg(
                contact_offset=NEWTON_GEAR_CONTACT_OFFSET,
                rest_offset=0.0,
            ),
            "physics_material": gear_material,
        }
        if visual_diffuse_color is not None:
            spawn_cfg_addon["visual_material"] = sim_utils.PreviewSurfaceCfg(
                diffuse_color=visual_diffuse_color,
                roughness=0.55,
            )
            spawn_cfg_addon["visual_material_path"] = GEAR_GREEN_VISUAL_MATERIAL_PATH
        remote_usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Factory/gear_assets/{usd_leaf}/{usd_leaf}.usd"
        gear = GearInsertionRigidObject(
            name=instance_name,
            prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
            object_type=ObjectType.RIGID,
            usd_path=retrieve_file_path(remote_usd_path, force_download=False),
            initial_pose=pose,
            spawn_cfg_addon=spawn_cfg_addon,
        )
        gear.disable_reset_pose()
        return gear

    maple_table = asset_registry.get_asset_by_name("maple_table_robolab")()
    maple_table.set_initial_pose(Pose(position_xyz=MAPLE_TABLE_POSITION))
    maple_table.object_cfg.spawn = deepcopy(maple_table.object_cfg.spawn)
    maple_table.object_cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
    maple_table.object_cfg.spawn.func = _spawn_newton_maple_table_usd
    maple_table.object_cfg.spawn.physics_material = gear_material

    table_reference = ObjectReference(
        name="table",
        prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
        parent_asset=maple_table,
        object_type=ObjectType.RIGID,
    )
    table_reference.add_relation(IsAnchor())

    tabletop_collider = Object(
        name="maple_table_top_collision",
        prim_path="{ENV_REGEX_NS}/maple_table_top_collision",
        object_type=ObjectType.RIGID,
        spawner_cfg=sim_utils.CuboidCfg(
            func=_spawn_maple_table_top_collision,
            size=(
                *MAPLE_TABLE_TOP_COLLISION_SIZE,
                MAPLE_TABLE_TOP_COLLISION_THICKNESS,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=NEWTON_GEAR_CONTACT_OFFSET),
            physics_material=gear_material,
            visible=True,
        ),
        initial_pose=Pose(
            position_xyz=(
                MAPLE_TABLE_TOP_COLLISION_POSITION[0],
                MAPLE_TABLE_TOP_COLLISION_POSITION[1],
                MAPLE_TABLE_TOP_COLLISION_POSITION[2] - MAPLE_TABLE_TOP_COLLISION_THICKNESS / 2.0,
            )
        ),
        tags=["background", "collision"],
    )

    base = make_gear(
        "gear_base",
        Pose(
            position_xyz=DROID_BASE_GEAR_POSITION,
            rotation_xyzw=DROID_BASE_GEAR_ROTATION_XYZW,
        ),
        kinematic_enabled=True,
    )
    gears = [
        make_gear(
            gear_type,
            Pose(
                position_xyz=NEWTON_GEAR_INITIAL_POSITIONS[gear_type],
                rotation_xyzw=DROID_BASE_GEAR_ROTATION_XYZW,
            ),
            visual_diffuse_color=GEAR_GREEN_DIFFUSE_COLOR,
        )
        for gear_type in GEAR_TYPES
    ]
    ground = Object(
        name="ground",
        prim_path="/World/ground",
        object_type=ObjectType.BASE,
        spawner_cfg=sim_utils.GroundPlaneCfg(),
        initial_pose=Pose(position_xyz=(0.0, 0.0, -1.05)),
    )
    light = DomeLight(
        instance_name="light",
        prim_path="/World/light",
        spawner_cfg=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=2500.0,
        ),
    )
    return Scene(
        assets=[
            ground,
            maple_table,
            tabletop_collider,
            base,
            *gears,
            table_reference,
            light,
        ]
    )


def _spawn_newton_mesh_collision_usd(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **_kwargs,
):
    """Spawn a Factory USD and replace its source collision meshes for Newton."""
    from isaaclab.sim import schemas
    from isaaclab.sim.utils import (
        bind_visual_material,
        create_prim,
        get_current_stage,
        make_uninstanceable,
        select_usd_variants,
    )
    from isaaclab.utils.assets import check_file_path, retrieve_file_path
    from isaaclab.utils.version import has_kit

    usd_path = cfg.usd_path
    file_status = check_file_path(usd_path)
    if file_status == 0:
        raise FileNotFoundError(f"USD file not found at path: {usd_path}")
    if file_status == 2:
        usd_path = retrieve_file_path(usd_path, force_download=False)

    stage = get_current_stage()
    if not stage.GetPrimAtPath(prim_path).IsValid():
        create_prim(
            prim_path,
            usd_path=usd_path,
            translation=translation,
            orientation=orientation,
            scale=cfg.scale,
            stage=stage,
        )
    if cfg.variants is not None:
        select_usd_variants(prim_path, cfg.variants)
    make_uninstanceable(prim_path, stage=stage)
    _author_newton_mesh_collision_leaves(stage, prim_path)

    if cfg.rigid_props is not None:
        schemas.modify_rigid_body_properties(prim_path, cfg.rigid_props)
    if cfg.collision_props is not None:
        schemas.modify_collision_properties(prim_path, cfg.collision_props)
    if cfg.mass_props is not None:
        schemas.modify_mass_properties(prim_path, cfg.mass_props)

    if cfg.visual_material is not None and has_kit():
        material_path = (
            f"{prim_path}/{cfg.visual_material_path}"
            if not cfg.visual_material_path.startswith("/")
            else cfg.visual_material_path
        )
        cfg.visual_material.func(material_path, cfg.visual_material)
        bind_visual_material(prim_path, material_path, stage=stage)
    return stage.GetPrimAtPath(prim_path)


def _author_newton_mesh_collision_leaves(stage, prim_path: str) -> None:
    """Author PR 24's convex gear/base collision decomposition."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(prim_path)
    rigid_body = next(
        (prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)),
        None,
    )
    assert rigid_body is not None, f"No rigid body found below {prim_path}"
    source_meshes = [
        UsdGeom.Mesh(prim)
        for prim in Usd.PrimRange(root)
        if "/collisions" in str(prim.GetPath()) and prim.IsA(UsdGeom.Mesh)
    ]
    assert source_meshes, f"No collision meshes found below {prim_path}"

    xform_cache = UsdGeom.XformCache()
    body_to_world = xform_cache.GetLocalToWorldTransform(rigid_body)
    body_points = []
    for source_mesh in source_meshes:
        mesh_to_body = xform_cache.GetLocalToWorldTransform(source_mesh.GetPrim()) * body_to_world.GetInverse()
        body_points.extend(mesh_to_body.Transform(point) for point in source_mesh.GetPointsAttr().Get())
    body_center = Gf.Vec3d(*(
        (min(point[axis] for point in body_points) + max(point[axis] for point in body_points)) / 2.0
        for axis in range(3)
    ))
    for child_name in ("visuals", "collisions"):
        child = rigid_body.GetChild(child_name)
        if child.IsValid():
            UsdGeom.Xformable(child).AddTranslateOp().Set(-body_center)

    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)

    xform_cache.Clear()
    body_to_world = xform_cache.GetLocalToWorldTransform(rigid_body)
    collision_root = UsdGeom.Xform.Define(stage, rigid_body.GetPath().AppendChild("newton_collisions"))
    collision_points = []
    for source_mesh in source_meshes:
        mesh_to_body = xform_cache.GetLocalToWorldTransform(source_mesh.GetPrim()) * body_to_world.GetInverse()
        collision_points.extend(Gf.Vec3f(mesh_to_body.Transform(point)) for point in source_mesh.GetPointsAttr().Get())
    if "FactoryGearBase" in prim_path:
        _author_newton_base_collision(collision_root, collision_points)
    else:
        _author_newton_gear_collision(collision_root, collision_points)


def _author_newton_base_collision(collision_root, points) -> None:
    """Approximate the base as one platform and three convex pegs."""
    z_levels = sorted({round(float(point[2]), 6) for point in points})
    assert len(z_levels) >= 3, "Gear base must contain a platform and raised pegs."
    z_min, platform_top, z_max = z_levels[0], z_levels[1], z_levels[-1]
    _author_box(
        collision_root,
        "platform",
        min(float(point[0]) for point in points),
        max(float(point[0]) for point in points),
        min(float(point[1]) for point in points),
        max(float(point[1]) for point in points),
        z_min,
        platform_top,
    )
    for gear_name, offset in NEWTON_GEAR_OFFSETS.items():
        center_x, center_y = offset[:2]
        peg_points = [
            point
            for point in points
            if float(point[2]) > platform_top + 1.0e-5 and abs(float(point[0]) - center_x) < 0.02
        ]
        assert peg_points, f"Could not find the {gear_name} base peg."
        radius = max(math.hypot(float(point[0]) - center_x, float(point[1]) - center_y) for point in peg_points)
        _author_convex_cylinder(
            collision_root,
            f"{gear_name}_peg",
            (center_x, center_y),
            radius,
            platform_top,
            z_max,
        )


def _author_newton_gear_collision(collision_root, points) -> None:
    """Approximate a gear plate and hub as convex annular leaves."""
    z_min = min(float(point[2]) for point in points)
    z_max = max(float(point[2]) for point in points)
    radial_points = [(float(point[0]) ** 2 + float(point[1]) ** 2) ** 0.5 for point in points]
    bore_radius = min(radial_points)
    hub_radius = max(
        radius for point, radius in zip(points, radial_points, strict=True) if float(point[2]) > z_max - 1.0e-5
    )
    plate_top = max(
        float(point[2]) for point, radius in zip(points, radial_points, strict=True) if radius > hub_radius + 1.0e-4
    )
    segments = 6
    inner_radius = (bore_radius + 2.0 * NEWTON_GEAR_CONTACT_OFFSET) / math.cos(math.pi / segments)
    plate_radius = max(radial_points)
    _author_convex_annulus(
        collision_root,
        "plate",
        inner_radius,
        plate_radius,
        z_min,
        plate_top,
        segments,
    )
    _author_convex_annulus(
        collision_root,
        "hub",
        inner_radius,
        hub_radius,
        plate_top,
        z_max,
        segments,
    )


def _author_box(
    collision_root,
    name: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> None:
    from pxr import Gf

    points = [
        Gf.Vec3f(x, y, z)
        for z in (z_min, z_max)
        for x, y in ((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max))
    ]
    _author_convex_mesh(
        collision_root,
        name,
        points,
        [4] * 6,
        [3, 2, 1, 0, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7],
    )


def _author_convex_cylinder(
    collision_root,
    name: str,
    center,
    radius: float,
    z_min: float,
    z_max: float,
    segments: int = 12,
) -> None:
    from pxr import Gf

    points = [
        Gf.Vec3f(
            center[0] + radius * math.cos(2.0 * math.pi * index / segments),
            center[1] + radius * math.sin(2.0 * math.pi * index / segments),
            z,
        )
        for z in (z_min, z_max)
        for index in range(segments)
    ]
    face_counts = [segments, segments, *([4] * segments)]
    face_indices = list(reversed(range(segments))) + list(range(segments, 2 * segments))
    for index in range(segments):
        next_index = (index + 1) % segments
        face_indices.extend((index, next_index, segments + next_index, segments + index))
    _author_convex_mesh(collision_root, name, points, face_counts, face_indices)


def _author_convex_annulus(
    collision_root,
    name: str,
    inner_radius: float,
    outer_radius: float,
    z_min: float,
    z_max: float,
    segments: int = 6,
) -> None:
    from pxr import Gf

    face_counts = [4] * 6
    face_indices = [
        3,
        2,
        1,
        0,
        4,
        5,
        6,
        7,
        0,
        1,
        5,
        4,
        1,
        2,
        6,
        5,
        2,
        3,
        7,
        6,
        3,
        0,
        4,
        7,
    ]
    angle_offset = math.pi / segments
    for index in range(segments):
        angles = (
            angle_offset + 2.0 * math.pi * index / segments,
            angle_offset + 2.0 * math.pi * (index + 1) / segments,
        )
        xy_points = [
            (inner_radius * math.cos(angles[0]), inner_radius * math.sin(angles[0])),
            (outer_radius * math.cos(angles[0]), outer_radius * math.sin(angles[0])),
            (outer_radius * math.cos(angles[1]), outer_radius * math.sin(angles[1])),
            (inner_radius * math.cos(angles[1]), inner_radius * math.sin(angles[1])),
        ]
        points = [Gf.Vec3f(x, y, z) for z in (z_min, z_max) for x, y in xy_points]
        _author_convex_mesh(
            collision_root,
            f"{name}_{index}",
            points,
            face_counts,
            face_indices,
        )


def _author_convex_mesh(collision_root, name: str, points, face_counts, face_indices) -> None:
    """Author one Newton-readable convex mesh leaf."""
    from pxr import UsdGeom, UsdPhysics

    stage = collision_root.GetPrim().GetStage()
    mesh = UsdGeom.Mesh.Define(stage, collision_root.GetPath().AppendChild(name))
    mesh.CreatePointsAttr().Set(points)
    mesh.CreateFaceVertexCountsAttr().Set(face_counts)
    mesh.CreateFaceVertexIndicesAttr().Set(face_indices)
    mesh.CreateExtentAttr().Set(UsdGeom.PointBased.ComputeExtent(points))
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr().Set(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("convexHull")


def _spawn_newton_maple_table_usd(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the maple table with Newton-readable colors and no tabletop collision."""
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    from pxr import Gf, Sdf, Usd, UsdPhysics, UsdShade

    prim = spawn_from_usd(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    stage = prim.GetStage()
    for descendant in Usd.PrimRange(prim):
        if descendant.GetName() == "top" and descendant.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(descendant).CreateCollisionEnabledAttr().Set(False)
    _bind_newton_omnipbr_color(
        stage,
        f"{prim_path}/table/table_01/top",
        f"{prim_path}/Looks/newton_maple_top",
        MAPLE_TABLE_TOP_COLLISION_COLOR,
        Gf,
        Sdf,
        UsdShade,
    )
    for leg_index in range(4):
        _bind_newton_omnipbr_color(
            stage,
            f"{prim_path}/table/table_01/leg_{leg_index}",
            f"{prim_path}/Looks/newton_table_legs",
            MAPLE_TABLE_LEG_RENDER_COLOR,
            Gf,
            Sdf,
            UsdShade,
        )
    return prim


def _bind_newton_omnipbr_color(
    stage,
    shape_path: str,
    material_path: str,
    color: tuple[float, float, float],
    Gf,
    Sdf,
    UsdShade,
) -> None:
    shape_prim = stage.GetPrimAtPath(shape_path)
    if not shape_prim.IsValid():
        return
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/OmniPBRShader")
    shader_prim = shader.GetPrim()
    shader_prim.CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("OmniPBR.mdl"))
    shader_prim.CreateAttribute("info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token).Set("OmniPBR")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))
    UsdShade.MaterialBindingAPI.Apply(shape_prim)
    UsdShade.MaterialBindingAPI(shape_prim).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )


def _author_display_color(stage, prim_path: str, color: tuple[float, float, float]) -> None:
    from pxr import Gf, Sdf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    display_color = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
        "displayColor",
        Sdf.ValueTypeNames.Color3fArray,
        UsdGeom.Tokens.constant,
        1,
    )
    display_color.Set([Gf.Vec3f(*color)])


def _spawn_maple_table_top_collision(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the finite tabletop proxy with a Newton-visible maple color."""
    from isaaclab.sim.spawners.shapes.shapes import spawn_cuboid

    prim = spawn_cuboid(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    stage = prim.GetStage()
    _author_display_color(stage, prim_path, MAPLE_TABLE_TOP_COLLISION_COLOR)
    _author_display_color(
        stage,
        f"{prim_path}/geometry/mesh",
        MAPLE_TABLE_TOP_COLLISION_COLOR,
    )
    return prim


def _ensure_newton_compatible_droid_usd(
    usd_path: str,
    min_mass: float = 0.02,
    min_diagonal_inertia: float = 1.0e-5,
    gravity_compensation: float = 1.0,
) -> str:
    """Return a cached DROID USD suitable for Newton's MuJoCo solver."""
    import shutil
    from pathlib import Path

    from isaaclab.utils.assets import retrieve_file_path

    source = Path(retrieve_file_path(usd_path, force_download=False))
    assert source.is_file(), f"USD path must resolve to a local file: {usd_path}"

    target = source.with_name(f"{source.stem}_gear_insertion_newton{source.suffix}")
    if (
        target.exists()
        and target.stat().st_mtime >= source.stat().st_mtime
        and _is_newton_compatible(target, gravity_compensation)
    ):
        return str(target)

    shutil.copy2(source, target)
    _author_minimum_rigid_body_inertias(
        target,
        min_mass=min_mass,
        min_diagonal_inertia=min_diagonal_inertia,
    )
    _author_gravity_compensation(target, gravity_compensation)
    _author_collision_mesh_leaves(target)
    _author_droid_fingertip_collisions(target)
    return str(target)


def _is_newton_compatible(usd_path, gravity_compensation: float) -> bool:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    assert stage is not None, f"Could not open USD: {usd_path}"
    if any(prim.HasAPI(UsdPhysics.RigidBodyAPI) and _needs_minimum_inertia(prim) for prim in stage.Traverse()):
        return False
    if any(
        prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetAttribute("mjc:gravcomp").Get() != gravity_compensation
        for prim in stage.Traverse()
    ):
        return False
    mimic_references = [
        relationship
        for prim in stage.Traverse()
        if "Robotiq_2F_85" in str(prim.GetPath())
        for relationship in prim.GetRelationships()
        if relationship.GetName().endswith(":referenceJoint")
    ]
    if len(mimic_references) != 5 or any(not relationship.GetTargets() for relationship in mimic_references):
        return False
    droid_collision_meshes = [
        prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh) and "Robotiq_2F_85" in str(prim.GetPath())
    ]
    if not _has_expected_droid_fingertip_collisions(droid_collision_meshes):
        return False
    if any(
        prim.HasAPI(UsdPhysics.CollisionAPI)
        and prim.GetName() != "newton_pad_collision"
        and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
        for prim in droid_collision_meshes
    ):
        return False
    return all(
        prim.IsA(UsdGeom.Boundable) or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    )


def _has_expected_droid_fingertip_collisions(collision_meshes) -> bool:
    from pxr import UsdGeom

    proxies = {
        prim.GetParent().GetName(): prim for prim in collision_meshes if prim.GetName() == "newton_pad_collision"
    }
    if set(proxies) != set(_DROID_FINGERTIP_COLLISION_BOUNDS):
        return False
    for body_name, (
        expected_minimum,
        expected_maximum,
    ) in _DROID_FINGERTIP_COLLISION_BOUNDS.items():
        proxy = UsdGeom.Mesh(proxies[body_name])
        if proxy.ComputePurpose() != UsdGeom.Tokens.guide:
            return False
        points = proxy.GetPointsAttr().Get()
        minimum = tuple(min(float(point[axis]) for point in points) for axis in range(3))
        maximum = tuple(max(float(point[axis]) for point in points) for axis in range(3))
        if any(
            not math.isclose(actual, expected, abs_tol=1.0e-6)
            for actual, expected in zip(
                (*minimum, *maximum),
                (*expected_minimum, *expected_maximum),
                strict=True,
            )
        ):
            return False
    return True


def _author_minimum_rigid_body_inertias(
    usd_path,
    min_mass: float,
    min_diagonal_inertia: float,
) -> None:
    from pxr import Gf, Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    assert stage is not None, f"Could not open USD: {usd_path}"

    diagonal_inertia = Gf.Vec3f(
        min_diagonal_inertia,
        min_diagonal_inertia,
        min_diagonal_inertia,
    )
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI) or not _needs_minimum_inertia(prim):
            continue

        mass_api = UsdPhysics.MassAPI(prim)
        if not prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI.Apply(prim)

        if _invalid_positive_value(mass_api.GetMassAttr().Get()):
            mass_api.CreateMassAttr().Set(min_mass)
        if _invalid_diagonal_inertia(mass_api.GetDiagonalInertiaAttr().Get()):
            mass_api.CreateDiagonalInertiaAttr().Set(diagonal_inertia)
        if _invalid_center_of_mass(mass_api.GetCenterOfMassAttr().Get()):
            mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0))
        if _invalid_principal_axes(mass_api.GetPrincipalAxesAttr().Get()):
            mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf.GetIdentity())

    stage.GetRootLayer().Save()


def _author_gravity_compensation(usd_path, gravity_compensation: float) -> None:
    """Compensate articulation gravity without disabling gravity for task objects."""
    from pxr import Sdf, Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    assert stage is not None, f"Could not open USD: {usd_path}"
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.CreateAttribute("mjc:gravcomp", Sdf.ValueTypeNames.Float).Set(gravity_compensation)
    stage.GetRootLayer().Save()


def _author_collision_mesh_leaves(usd_path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    assert stage is not None, f"Could not open USD: {usd_path}"

    collision_root_paths = [
        prim.GetPath()
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.IsA(UsdGeom.Boundable)
    ]
    for path in collision_root_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsInstance():
            prim.SetInstanceable(False)
    stage.GetRootLayer().Save()
    stage.Reload()

    for path in collision_root_paths:
        root = stage.GetPrimAtPath(path)
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            if "Robotiq_2F_85" in str(prim.GetPath()):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
                continue
            mesh_api = UsdPhysics.MeshCollisionAPI(prim)
            if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_api.CreateApproximationAttr().Set("convexHull")
            UsdGeom.Imageable(prim).CreatePurposeAttr().Set(UsdGeom.Tokens.default_)
        UsdPhysics.CollisionAPI(root).CreateCollisionEnabledAttr().Set(False)

    stage.GetRootLayer().Save()


def _author_droid_fingertip_collisions(usd_path) -> None:
    """Author link-local convex pads that Newton imports without transform drift."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    assert stage is not None, f"Could not open USD: {usd_path}"

    for body_name, (minimum, maximum) in _DROID_FINGERTIP_COLLISION_BOUNDS.items():
        body = next(
            prim for prim in stage.Traverse() if prim.GetName() == body_name and "Robotiq_2F_85" in str(prim.GetPath())
        )
        mesh = UsdGeom.Mesh.Define(
            stage,
            body.GetPath().AppendChild("newton_pad_collision"),
        )
        points = [
            Gf.Vec3f(x, y, z)
            for z in (minimum[2], maximum[2])
            for x, y in (
                (minimum[0], minimum[1]),
                (maximum[0], minimum[1]),
                (maximum[0], maximum[1]),
                (minimum[0], maximum[1]),
            )
        ]
        mesh.CreatePointsAttr().Set(points)
        mesh.CreateFaceVertexCountsAttr().Set([4] * 6)
        mesh.CreateFaceVertexIndicesAttr().Set([
            3,
            2,
            1,
            0,
            4,
            5,
            6,
            7,
            0,
            1,
            5,
            4,
            1,
            2,
            6,
            5,
            2,
            3,
            7,
            6,
            3,
            0,
            4,
            7,
        ])
        mesh.CreateExtentAttr().Set(UsdGeom.PointBased.ComputeExtent(points))
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        mesh.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr().Set(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("convexHull")

    stage.GetRootLayer().Save()


def _needs_minimum_inertia(prim) -> bool:
    from pxr import UsdPhysics

    if not prim.HasAPI(UsdPhysics.MassAPI):
        return True
    mass_api = UsdPhysics.MassAPI(prim)
    return (
        _invalid_positive_value(mass_api.GetMassAttr().Get())
        or _invalid_diagonal_inertia(mass_api.GetDiagonalInertiaAttr().Get())
        or _invalid_center_of_mass(mass_api.GetCenterOfMassAttr().Get())
        or _invalid_principal_axes(mass_api.GetPrincipalAxesAttr().Get())
    )


def _invalid_positive_value(value) -> bool:
    return value is None or float(value) <= 0.0


def _invalid_diagonal_inertia(value) -> bool:
    return value is None or any(float(component) <= 0.0 for component in value)


def _invalid_center_of_mass(value) -> bool:
    return value is None or any(not math.isfinite(float(component)) for component in value)


def _invalid_principal_axes(value) -> bool:
    if value is None:
        return True
    components = (value.GetReal(), *value.GetImaginary())
    return any(not math.isfinite(float(component)) for component in components) or math.isclose(
        sum(float(component) ** 2 for component in components),
        0.0,
    )
