# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Gear Assembly scene using Arena's existing Droid embodiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory
from isaaclab_arena.tasks.gear_assembly.specs import (
    DROID_ARM_JOINT_NAMES,
    DROID_BASE_GEAR_POSE,
    DROID_GEAR_ASSEMBLY_EMBODIMENTS,
    DROID_IK_SEED_JOINT_POSITIONS,
    MAPLE_TABLE_POSE,
    MAPLE_TABLE_TOP_COLLISION_POSE,
    NEWTON_DROID_BASE_GEAR_POSE,
    NEWTON_DROID_GEAR_POSES,
    NEWTON_GEAR_TABLETOP_ORIENTATION_XYZW,
    NEWTON_GEAR_TABLETOP_PARKING_POSITIONS,
    get_droid_robot_spec,
)

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg
    from isaaclab_arena.tasks.gear_assembly.task import GearAssemblyTask


@dataclass
class GearAssemblyEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the Arena Droid Gear Assembly environment."""

    embodiment: str = "droid_abs_joint_pos"
    mode: str = "play"
    physics_backend: str = "newton"

    def __post_init__(self) -> None:
        assert (
            self.embodiment in DROID_GEAR_ASSEMBLY_EMBODIMENTS
        ), f"Gear Assembly is Droid-only; got embodiment={self.embodiment!r}"
        assert self.mode in {"play", "randomized"}, f"Unsupported Gear Assembly mode: {self.mode!r}"
        assert self.physics_backend in {
            "newton",
            "physx",
        }, f"Unsupported Gear Assembly physics backend: {self.physics_backend!r}"


@register_environment
class GearAssemblyEnvironment(ArenaEnvironmentFactory[GearAssemblyEnvironmentCfg]):
    """Arena Gear Assembly factory using the existing Droid robot."""

    name = "gear_assembly"
    _legacy_argparse_cfg_type = GearAssemblyEnvironmentCfg

    def build(self, cfg: GearAssemblyEnvironmentCfg) -> IsaacLabArenaEnvironment:
        import isaaclab.sim as sim_utils

        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_library import DomeLight
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.gear_assembly.assets import (
            make_factory_gear_base,
            make_factory_gear_large,
            make_factory_gear_medium,
            make_factory_gear_small,
            make_ground,
            make_maple_table_top_collision,
            spawn_newton_maple_table_usd,
        )
        from isaaclab_arena.tasks.gear_assembly.task import GearAssemblyTask

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)
        if cfg.physics_backend == "newton":
            from isaaclab_arena.embodiments.droid.droid import configure_droid_robot_for_newton
            from isaaclab_arena.utils.usd.newton import ensure_newton_compatible_droid_usd

            configure_droid_robot_for_newton(embodiment.scene_config.robot)
            embodiment.scene_config.robot.spawn.usd_path = ensure_newton_compatible_droid_usd(
                embodiment.scene_config.robot.spawn.usd_path
            )
            # Newton's pinned importer treats disableGravity as scene-wide. Keep
            # world gravity active and use per-body MuJoCo gravity compensation.
            embodiment.scene_config.robot.spawn.rigid_props.disable_gravity = False
        embodiment.observation_config = None
        newton_mesh_collisions = cfg.physics_backend == "newton"
        robot_spec = get_droid_robot_spec(newton_backend=newton_mesh_collisions)
        base_pose = NEWTON_DROID_BASE_GEAR_POSE if newton_mesh_collisions else DROID_BASE_GEAR_POSE
        gear_poses = (
            NEWTON_DROID_GEAR_POSES
            if newton_mesh_collisions
            else {gear_type: DROID_BASE_GEAR_POSE for gear_type in ("gear_small", "gear_medium", "gear_large")}
        )
        maple_table = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        maple_table.set_initial_pose(MAPLE_TABLE_POSE)
        if newton_mesh_collisions:
            maple_table.object_cfg.spawn.func = spawn_newton_maple_table_usd
        table_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
            parent_asset=maple_table,
            object_type=ObjectType.RIGID,
        )
        table_reference.add_relation(IsAnchor())

        assets = [
            make_ground(),
            maple_table,
            *([make_maple_table_top_collision(MAPLE_TABLE_TOP_COLLISION_POSE)] if newton_mesh_collisions else []),
            make_factory_gear_base(base_pose, newton_mesh_collisions=newton_mesh_collisions),
            make_factory_gear_small(gear_poses["gear_small"], newton_mesh_collisions=newton_mesh_collisions),
            make_factory_gear_medium(gear_poses["gear_medium"], newton_mesh_collisions=newton_mesh_collisions),
            make_factory_gear_large(gear_poses["gear_large"], newton_mesh_collisions=newton_mesh_collisions),
            table_reference,
            DomeLight(
                instance_name="light",
                prim_path="/World/light",
                spawner_cfg=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
            ),
        ]

        task = GearAssemblyTask(robot_spec=robot_spec, mode=cfg.mode, newton_backend=newton_mesh_collisions)
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=Scene(assets=assets),
            task=task,
            env_cfg_callback=_make_env_cfg_callback(cfg, task),
        )


def _make_env_cfg_callback(cfg: GearAssemblyEnvironmentCfg, task: GearAssemblyTask):
    def gear_assembly_env_cfg_callback(
        env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
    ) -> IsaacLabArenaManagerBasedRLEnvCfg:
        from isaaclab_physx.physics import PhysxCfg

        from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import ArenaPhysicsCfg

        env_cfg.episode_length_s = 6.66
        env_cfg.viewer.eye = (1.6, -1.2, 1.0)
        env_cfg.viewer.lookat = (0.55, 0.05, 0.08)
        env_cfg.decimation = 4
        env_cfg.sim.render_interval = 4
        env_cfg.sim.dt = 1.0 / 120.0

        if cfg.physics_backend == "newton":
            from isaaclab_arena.embodiments.droid.actions import (
                DROID_GRIPPER_CLOSE_COMMAND,
                DROID_GRIPPER_JOINT_NAMES,
                DROID_GRIPPER_OPEN_COMMAND,
            )
            from isaaclab_arena.tasks.gear_assembly.actions import GearAssemblyBinaryJointPositionAction
            from isaaclab_arena.tasks.gear_assembly.events import set_robot_to_grasp_pose_with_finite_difference_ik

            env_cfg.sim.physics = ArenaPhysicsCfg().newton
            # Show the hub-aligned parallel jaws side-on in viewport recordings.
            env_cfg.viewer.eye = (1.6, 1.2, 1.0)
            # Fine gear contacts need a higher integration rate than Arena's default.
            env_cfg.sim.physics.num_substeps = 12
            env_cfg.sim.physics.default_shape_cfg.gap = 0.0
            env_cfg.sim.physics.solver_cfg.ccd_iterations = 35
            env_cfg.sim.physics.solver_cfg.use_mujoco_contacts = False
            # Nine simultaneous grasps exceed Newton's default constraint capacity (300).
            env_cfg.sim.physics.solver_cfg.njmax = 512
            # Keep millimeter-scale contacts near the origin for single-precision Newton dynamics.
            env_cfg.scene.env_spacing = 1.5
            env_cfg.scene.replicate_physics = True
            # Select the collision-free Franka IK branch shared by all three tabletop grasps.
            env_cfg.scene.robot.init_state.joint_pos.update(
                dict(zip(DROID_ARM_JOINT_NAMES, DROID_IK_SEED_JOINT_POSITIONS, strict=True))
            )
            env_cfg.events.init_franka_arm_pose.params["default_pose"][
                : len(DROID_ARM_JOINT_NAMES)
            ] = DROID_IK_SEED_JOINT_POSITIONS
            env_cfg.events.randomize_franka_joint_state = None
            env_cfg.events.set_robot_to_grasp_pose.func = set_robot_to_grasp_pose_with_finite_difference_ik
            env_cfg.actions.gripper_action.class_type = GearAssemblyBinaryJointPositionAction
            env_cfg.actions.gripper_action.joint_names = list(DROID_GRIPPER_JOINT_NAMES)
            env_cfg.actions.gripper_action.open_command_expr = DROID_GRIPPER_OPEN_COMMAND
            env_cfg.actions.gripper_action.close_command_expr = DROID_GRIPPER_CLOSE_COMMAND
        elif cfg.physics_backend == "physx":
            env_cfg.sim.physics = PhysxCfg(
                gpu_collision_stack_size=2**30,
                gpu_max_rigid_contact_count=2**23,
                gpu_max_rigid_patch_count=2**23,
            )
            env_cfg.scene.replicate_physics = False
        else:
            raise ValueError(f"Unsupported Gear Assembly physics backend: {cfg.physics_backend}")

        for attr_name, value in task.runtime_env_attrs().items():
            setattr(env_cfg, attr_name, value)
        if cfg.physics_backend == "newton":
            env_cfg.events.randomize_gears_and_base_pose.params["selected_parking_positions"] = (
                NEWTON_GEAR_TABLETOP_PARKING_POSITIONS
            )
            env_cfg.events.randomize_gears_and_base_pose.params["selected_orientation_xyzw"] = (
                NEWTON_GEAR_TABLETOP_ORIENTATION_XYZW
            )
            if cfg.mode == "play":
                # Evaluation starts the selected gear on the table. The source failure terms are
                # gripper-relative, so play mode terminates only on the assembled success term.
                env_cfg.terminations.gear_dropped = None
                env_cfg.terminations.gear_orientation_exceeded = None
                env_cfg.terminations.time_out = None
        return env_cfg

    return gear_assembly_env_cfg_callback
