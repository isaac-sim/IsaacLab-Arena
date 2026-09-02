# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Self-contained Newton prototype for DROID gear assembly.

The prototype deliberately keeps the environment, scene assets, task, and
Newton compatibility settings together.  The Factory USD collision meshes are
used directly; this module does not generate replacement collision geometry.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.embodiments.droid.droid import DroidEmbodimentBase
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg


TABLE_TOP_Z = 0.071
"""World-space height of the maple tabletop."""

MAPLE_TABLE_TOP_Z = 0.003000684082508087
"""Top of the table geometry in the maple-table USD's local frame."""

GEAR_ASSET_DIR = Path(__file__).with_name("gear_assembly_assets")
GEAR_SCALE = (1.0, 1.0, 1.0)
GEAR_HALF_HEIGHT = 0.01875
GEAR_INITIAL_POSITION = (0.41, 0.17, TABLE_TOP_Z + GEAR_HALF_HEIGHT + 0.001)
GEAR_BASE_POSITION = (0.55, -0.08, TABLE_TOP_Z + GEAR_HALF_HEIGHT)
MEDIUM_GEAR_TARGET_OFFSET = (0.0366185, 0.0, 0.0075)

SUCCESS_XY_THRESHOLD = 0.015
SUCCESS_Z_THRESHOLD = 0.010
SUCCESS_UPRIGHT_THRESHOLD_DEG = 15.0
SUCCESS_LINEAR_VELOCITY_THRESHOLD = 0.05
SUCCESS_ANGULAR_VELOCITY_THRESHOLD = 0.5
SUCCESS_SUPPORT_Z_THRESHOLD = 0.005
SUCCESS_CONSECUTIVE_STEPS = 10
_SUCCESS_COUNTER_ATTR = "_gear_assembly_consecutive_success_count"

DROID_ARM_JOINT_NAMES = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
)
DROID_GEAR_WORKSPACE_JOINT_POSITIONS = (0.98, -0.47, -1.73, -1.42, -1.28, 2.71, 1.35)

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
DROID_GRIPPER_CLOSE_COMMAND = {joint_name: sign * 0.461 for joint_name, sign in DROID_GRIPPER_MIMIC_SIGNS.items()}


@dataclass
class GearAssemblyEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the self-contained Newton gear-assembly prototype."""

    episode_length_s: float = 70.0
    """Maximum episode duration."""

    sdf_max_resolution: int = 256
    """Maximum Newton SDF resolution for the Factory gear collision meshes."""

    def __post_init__(self) -> None:
        assert self.episode_length_s > 0.0, "episode_length_s must be positive"
        assert self.sdf_max_resolution > 0, "sdf_max_resolution must be positive"
        assert self.sdf_max_resolution % 8 == 0, "Newton SDF resolution must be divisible by 8"


@register_environment
class GearAssemblyEnvironment(ArenaEnvironmentFactory[GearAssemblyEnvironmentCfg]):
    """Build a DROID task for picking up and inserting the medium Factory gear."""

    name = "gear_assembly"
    _legacy_argparse_cfg_type = GearAssemblyEnvironmentCfg

    def build(self, cfg: GearAssemblyEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the monolithic prototype through Arena's normal abstractions."""
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

        embodiment = self.asset_registry.get_asset_by_name("droid_differential_ik")(enable_cameras=cfg.enable_cameras)
        _configure_newton_droid(embodiment)

        scene, gear_base, medium_gear = _build_scene(self.asset_registry, cfg)
        task = _make_gear_assembly_task(cfg, gear_base, medium_gear)
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            env_cfg_callback=_configure_environment_for_newton,
        )

    @staticmethod
    def _add_legacy_cli_only_args(parser) -> None:
        """Expose the flag expected by Isaac Lab's teleoperation scripts."""
        parser.add_argument("--teleop_device", type=str, default=None)


def _build_scene(asset_registry, cfg: GearAssemblyEnvironmentCfg):
    """Create the table, source-USD gear pair, and lighting."""
    import isaaclab.sim as sim_utils
    from isaaclab_newton.sim.schemas import NewtonMaterialPropertiesCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    gear_material = NewtonMaterialPropertiesCfg(
        static_friction=3.0,
        dynamic_friction=3.0,
        restitution=0.0,
    )
    base_material = NewtonMaterialPropertiesCfg(static_friction=0.0, dynamic_friction=0.0, restitution=0.0)
    factory_spawn = _make_factory_asset_spawn(cfg.sdf_max_resolution)

    def make_factory_object(
        *,
        name: str,
        usd_name: str,
        prim_name: str,
        pose: Pose,
        mass: float,
        kinematic: bool,
        physics_material,
        object_type: ObjectType = ObjectType.RIGID,
    ) -> Object:
        obj = Object(
            name=name,
            prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
            object_type=object_type,
            usd_path=str(GEAR_ASSET_DIR / f"{usd_name}.usda"),
            scale=GEAR_SCALE,
            initial_pose=pose,
            spawn_cfg_addon={
                "func": factory_spawn,
                "make_uninstanceable": True,
                "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    kinematic_enabled=kinematic,
                    linear_damping=0.05,
                    angular_damping=0.10,
                    max_depenetration_velocity=1.0,
                ),
                "mass_props": sim_utils.MassPropertiesCfg(mass=mass),
                "collision_props": sim_utils.CollisionPropertiesCfg(
                    contact_offset=1.0e-4,
                    rest_offset=0.0,
                ),
                "physics_material": physics_material,
            },
        )
        # The task-level reset restores the complete scene, so separate object
        # pose terms would only write the same state twice.
        obj.disable_reset_pose()
        return obj

    background = asset_registry.get_asset_by_name("maple_table_robolab")()
    background.set_initial_pose(Pose(position_xyz=(0.0, 0.0, TABLE_TOP_Z - MAPLE_TABLE_TOP_Z)))

    gear_base = make_factory_object(
        name="gear_base",
        usd_name="factory_gear_base",
        prim_name="GearBase",
        pose=Pose(position_xyz=GEAR_BASE_POSITION),
        mass=0.05,
        kinematic=True,
        physics_material=base_material,
        object_type=ObjectType.RIGID,
    )
    medium_gear = make_factory_object(
        name="medium_gear",
        usd_name="factory_gear_medium",
        prim_name="MediumGear",
        pose=Pose(position_xyz=GEAR_INITIAL_POSITION),
        mass=0.019,
        kinematic=False,
        physics_material=gear_material,
    )

    light = asset_registry.get_asset_by_name("light")()
    light.set_intensity(1800.0)
    directional_light = asset_registry.get_asset_by_name("directional_light")()
    return Scene(assets=[background, gear_base, medium_gear, light, directional_light]), gear_base, medium_gear


def _make_factory_asset_spawn(sdf_max_resolution: int):
    """Return a USD spawner that adds Newton SDF metadata to source colliders."""
    from isaaclab.sim import schemas
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    from isaaclab.sim.utils import clone
    from isaaclab_newton.sim.schemas import NewtonCollisionCfg, NewtonSDFCollisionCfg
    from pxr import Usd, UsdGeom, UsdPhysics

    @clone
    def spawn_factory_asset(
        prim_path: str,
        spawner_cfg,
        translation: tuple[float, float, float] | None = None,
        orientation: tuple[float, float, float, float] | None = None,
        **kwargs,
    ):
        prim = spawn_from_usd(
            prim_path,
            spawner_cfg,
            translation=translation,
            orientation=orientation,
            **kwargs,
        )
        stage = prim.GetStage()
        collision_meshes = [
            candidate
            for candidate in Usd.PrimRange(prim)
            if candidate.IsA(UsdGeom.Mesh) and candidate.HasAPI(UsdPhysics.CollisionAPI)
        ]
        assert collision_meshes, f"Factory asset at '{prim_path}' has no authored collision mesh"
        for collision_mesh in collision_meshes:
            schemas.apply_collision_properties(
                str(collision_mesh.GetPath()),
                [NewtonCollisionCfg(contact_margin=0.0, contact_gap=1.0e-4)],
                stage,
            )
            schemas.apply_mesh_collision_properties(
                str(collision_mesh.GetPath()),
                [
                    NewtonSDFCollisionCfg(
                        sdf_max_resolution=sdf_max_resolution,
                        sdf_narrow_band_inner=-0.005,
                        sdf_narrow_band_outer=0.005,
                        hydroelastic_enabled=True,
                        hydroelastic_stiffness=1.0e8,
                    )
                ],
                stage,
            )
        return prim

    return spawn_factory_asset


_NEWTON_DROID_SPAWN = None


def _get_newton_droid_spawn():
    """Return a spawner that exposes the DROID USD's existing gripper colliders."""
    global _NEWTON_DROID_SPAWN
    if _NEWTON_DROID_SPAWN is None:
        from isaaclab.sim import schemas
        from isaaclab.sim.spawners.from_files import spawn_from_usd
        from isaaclab.sim.utils import clone
        from isaaclab_newton.sim.schemas import MujocoRigidBodyPropertiesCfg

        @clone
        def spawn_newton_droid(
            prim_path: str,
            spawner_cfg,
            translation: tuple[float, float, float] | None = None,
            orientation: tuple[float, float, float, float] | None = None,
            **kwargs,
        ):
            prim = spawn_from_usd(
                prim_path,
                spawner_cfg,
                translation=translation,
                orientation=orientation,
                **kwargs,
            )
            _promote_source_collision_meshes(prim)
            schemas.modify_rigid_body_properties(
                prim_path,
                MujocoRigidBodyPropertiesCfg(gravcomp=1.0),
                prim.GetStage(),
            )
            return prim

        _NEWTON_DROID_SPAWN = spawn_newton_droid
    return _NEWTON_DROID_SPAWN


def _promote_source_collision_meshes(root_prim) -> None:
    """Move collision schemas from unsupported Xforms onto their source meshes.

    Some Robotiq collision groups carry ``CollisionAPI`` on an Xform above an
    existing mesh. Newton consumes the mesh-level schema, so copy the authored
    enable/approximation settings to those meshes and disable the invalid Xform
    collider. Mesh points, topology, and transforms remain untouched.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    collision_groups = [
        candidate
        for candidate in Usd.PrimRange(root_prim)
        if candidate.HasAPI(UsdPhysics.CollisionAPI) and not candidate.IsA(UsdGeom.Gprim)
    ]
    for collision_group in collision_groups:
        collision_api = UsdPhysics.CollisionAPI(collision_group)
        collision_enabled = collision_api.GetCollisionEnabledAttr().Get()
        collision_enabled = True if collision_enabled is None else collision_enabled
        source_approximation = None
        if collision_group.HasAPI(UsdPhysics.MeshCollisionAPI):
            source_approximation = UsdPhysics.MeshCollisionAPI(collision_group).GetApproximationAttr().Get()

        meshes = [candidate for candidate in Usd.PrimRange(collision_group) if candidate.IsA(UsdGeom.Mesh)]
        assert meshes, f"Collision group '{collision_group.GetPath()}' has no source mesh"
        for mesh in meshes:
            mesh_collision_api = (
                UsdPhysics.CollisionAPI(mesh)
                if mesh.HasAPI(UsdPhysics.CollisionAPI)
                else UsdPhysics.CollisionAPI.Apply(mesh)
            )
            mesh_collision_api.CreateCollisionEnabledAttr().Set(collision_enabled)
            mesh_approximation_api = (
                UsdPhysics.MeshCollisionAPI(mesh)
                if mesh.HasAPI(UsdPhysics.MeshCollisionAPI)
                else UsdPhysics.MeshCollisionAPI.Apply(mesh)
            )
            mesh_approximation_api.CreateApproximationAttr().Set(source_approximation or "convexHull")

        collision_api.CreateCollisionEnabledAttr().Set(False)


def _configure_newton_droid(embodiment: DroidEmbodimentBase) -> None:
    """Apply scoped Newton spawn, actuator, and end-effector settings to DROID."""
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
    from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
    from isaaclab_newton.sim.schemas import NewtonMaterialPropertiesCfg

    robot_cfg = deepcopy(embodiment.scene_config.robot)
    embodiment.scene_config.robot = robot_cfg
    robot_cfg.spawn.func = _get_newton_droid_spawn()
    robot_cfg.spawn.make_uninstanceable = True
    robot_cfg.spawn.rigid_props.disable_gravity = False
    robot_cfg.spawn.physics_material = NewtonMaterialPropertiesCfg(
        static_friction=3.0,
        dynamic_friction=3.0,
        restitution=0.0,
    )
    robot_cfg.init_state.joint_pos.update(
        dict(zip(DROID_ARM_JOINT_NAMES, DROID_GEAR_WORKSPACE_JOINT_POSITIONS, strict=True))
    )
    robot_cfg.actuators["gripper"] = ImplicitActuatorCfg(
        joint_names_expr=list(DROID_GRIPPER_JOINT_NAMES),
        effort_limit=20.0,
        velocity_limit=1.2,
        stiffness=40.0,
        damping=8.0,
        armature=0.05,
    )

    embodiment.action_config.arm_action = deepcopy(embodiment.action_config.arm_action)
    embodiment.action_config.arm_action.body_name = "base_link"
    embodiment.action_config.arm_action.body_offset = None
    # Large Cartesian transfers with a downward-facing wrist can pass close to
    # a DROID singularity. Newton exposes that rank loss more sharply than the
    # PhysX-tuned default fixed-DLS controller, so use the controller's
    # manipulability-aware damping and keep the redundant arm off joint limits.
    arm_controller = embodiment.action_config.arm_action.controller
    arm_controller.ik_method = "adaptive_dls"
    arm_controller.ik_params = {
        "lambda_min": 0.05,
        "lambda_max": 0.20,
        "sigma_thresh": 0.02,
    }
    arm_controller.joint_limit_avoidance_gain = 0.10
    arm_controller.joint_limit_avoidance_margin = 0.35
    # Native keyboard and SpaceMouse devices use positive=open and
    # negative=close; zero remains open for zero-action rollouts.
    embodiment.action_config.gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(DROID_GRIPPER_JOINT_NAMES),
        open_command_expr=DROID_GRIPPER_OPEN_COMMAND,
        close_command_expr=DROID_GRIPPER_CLOSE_COMMAND,
    )

    embodiment.scene_config.ee_frame = deepcopy(embodiment.scene_config.ee_frame)
    target_frame = embodiment.scene_config.ee_frame.target_frames[0]
    target_frame.prim_path = "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link"
    target_frame.offset = OffsetCfg()


def _configure_environment_for_newton(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Select Newton and configure a stable 30 Hz manipulation loop."""
    from isaaclab.devices.device_base import DevicesCfg
    from isaaclab.devices.keyboard import Se3KeyboardCfg
    from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
    from isaaclab_newton.physics import HydroelasticSDFCfg, NewtonCollisionPipelineCfg

    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import ArenaPhysicsCfg

    env_cfg.sim.dt = 1.0 / 240.0
    env_cfg.decimation = 8
    env_cfg.sim.render_interval = 8
    env_cfg.sim.physics = ArenaPhysicsCfg().newton
    env_cfg.sim.physics.num_substeps = 4
    env_cfg.sim.physics.default_shape_cfg.gap = 0.0
    env_cfg.sim.physics.solver_cfg.ls_iterations = 50
    env_cfg.sim.physics.solver_cfg.ccd_iterations = 35
    env_cfg.sim.physics.solver_cfg.njmax = 4096
    env_cfg.sim.physics.solver_cfg.nconmax = 4096
    env_cfg.sim.physics.collision_cfg = NewtonCollisionPipelineCfg(
        sdf_hydroelastic_config=HydroelasticSDFCfg(reduce_contacts=True, normal_matching=True)
    )
    env_cfg.scene.env_spacing = 1.5
    env_cfg.scene.replicate_physics = True
    env_cfg.events.randomize_franka_joint_state = None
    env_cfg.teleop_devices = DevicesCfg(
        devices={
            "keyboard": Se3KeyboardCfg(
                pos_sensitivity=0.05,
                rot_sensitivity=0.05,
                sim_device=env_cfg.sim.device,
            ),
            "spacemouse": Se3SpaceMouseCfg(
                pos_sensitivity=0.05,
                rot_sensitivity=0.05,
                sim_device=env_cfg.sim.device,
            ),
        }
    )
    return env_cfg


def _gear_is_inserted_now(
    env,
    base_asset_cfg,
    gear_asset_cfg,
    target_offset_xyz,
    xy_threshold: float,
    z_threshold: float,
    upright_axis_threshold_deg: float,
    linear_velocity_threshold: float,
    angular_velocity_threshold: float,
    support_z_threshold: float,
):
    """Return whether the gear is currently aligned, seated, upright, and still."""
    import torch

    import isaaclab.utils.math as math_utils
    import warp as wp

    base = env.scene[base_asset_cfg.name]
    gear = env.scene[gear_asset_cfg.name]
    base_position = wp.to_torch(base.data.root_link_pos_w)
    base_orientation = wp.to_torch(base.data.root_link_quat_w)
    gear_position = wp.to_torch(gear.data.root_link_pos_w)
    gear_orientation = wp.to_torch(gear.data.root_link_quat_w)
    gear_velocity = wp.to_torch(gear.data.root_com_vel_w)
    target_offset = torch.as_tensor(target_offset_xyz, device=env.device, dtype=gear_position.dtype).expand(
        env.num_envs, -1
    )
    target_position = base_position + math_utils.quat_apply(base_orientation, target_offset)
    position_error = gear_position - target_position

    xy_error = torch.linalg.vector_norm(position_error[:, :2], dim=-1)
    z_error = torch.abs(position_error[:, 2])
    support_error = torch.clamp(position_error[:, 2], min=0.0)
    local_up = torch.tensor((0.0, 0.0, 1.0), device=env.device).repeat(env.num_envs, 1)
    gear_up = math_utils.quat_apply(gear_orientation, local_up)
    base_up = math_utils.quat_apply(base_orientation, local_up)
    upright = (gear_up * base_up).sum(dim=-1) >= math.cos(math.radians(upright_axis_threshold_deg))
    settled = (torch.linalg.vector_norm(gear_velocity[:, :3], dim=-1) <= linear_velocity_threshold) & (
        torch.linalg.vector_norm(gear_velocity[:, 3:], dim=-1) <= angular_velocity_threshold
    )
    return (
        (xy_error <= xy_threshold)
        & (z_error <= z_threshold)
        & (support_error <= support_z_threshold)
        & upright
        & settled
    )


def _reset_gear_success_counter(env, env_ids) -> None:
    """Clear the persistent insertion counter for reset environments."""
    import torch

    success_count = getattr(env, _SUCCESS_COUNTER_ATTR, None)
    if success_count is None:
        success_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
        setattr(env, _SUCCESS_COUNTER_ATTR, success_count)
    if env_ids is None:
        success_count.zero_()
    else:
        success_count[env_ids] = 0


def _gear_is_inserted_with_hold(
    env,
    base_asset_cfg,
    gear_asset_cfg,
    target_offset_xyz,
    xy_threshold: float,
    z_threshold: float,
    upright_axis_threshold_deg: float,
    linear_velocity_threshold: float,
    angular_velocity_threshold: float,
    support_z_threshold: float,
    consecutive_success_steps: int,
):
    """Return true after the insertion condition persists for enough steps."""
    import torch

    success_count = getattr(env, _SUCCESS_COUNTER_ATTR, None)
    if success_count is None:
        _reset_gear_success_counter(env, None)
        success_count = getattr(env, _SUCCESS_COUNTER_ATTR)
    success_now = _gear_is_inserted_now(
        env,
        base_asset_cfg,
        gear_asset_cfg,
        target_offset_xyz,
        xy_threshold,
        z_threshold,
        upright_axis_threshold_deg,
        linear_velocity_threshold,
        angular_velocity_threshold,
        support_z_threshold,
    )
    success_count.copy_(torch.where(success_now, success_count + 1, torch.zeros_like(success_count)))
    return success_count >= consecutive_success_steps


def _make_gear_assembly_task(cfg: GearAssemblyEnvironmentCfg, base_asset: Object, held_asset: Object):
    """Define the reset, success, failure, metrics, and language task."""
    import isaaclab.envs.mdp as mdp
    from isaaclab.envs.common import ViewerCfg
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    from isaaclab.utils.configclass import configclass

    from isaaclab_arena.embodiments.common.arm_mode import ArmMode
    from isaaclab_arena.metrics.metric_base import MetricBase
    from isaaclab_arena.metrics.object_moved import ObjectMovedRateMetric
    from isaaclab_arena.metrics.success_rate import SuccessRateMetric
    from isaaclab_arena.tasks.task_base import TaskBase

    @configclass
    class GearAssemblyEventsCfg:
        reset_scene: EventTerm = EventTerm(
            func=mdp.reset_scene_to_default,
            mode="reset",
            params={"reset_joint_targets": True},
        )
        reset_success_counter: EventTerm = EventTerm(func=_reset_gear_success_counter, mode="reset")

    @configclass
    class GearAssemblyTerminationsCfg:
        time_out: DoneTerm = DoneTerm(func=mdp.time_out, time_out=True)
        success: DoneTerm = DoneTerm(
            func=_gear_is_inserted_with_hold,
            params={
                "base_asset_cfg": SceneEntityCfg(base_asset.name),
                "gear_asset_cfg": SceneEntityCfg(held_asset.name),
                "target_offset_xyz": MEDIUM_GEAR_TARGET_OFFSET,
                "xy_threshold": SUCCESS_XY_THRESHOLD,
                "z_threshold": SUCCESS_Z_THRESHOLD,
                "upright_axis_threshold_deg": SUCCESS_UPRIGHT_THRESHOLD_DEG,
                "linear_velocity_threshold": SUCCESS_LINEAR_VELOCITY_THRESHOLD,
                "angular_velocity_threshold": SUCCESS_ANGULAR_VELOCITY_THRESHOLD,
                "support_z_threshold": SUCCESS_SUPPORT_Z_THRESHOLD,
                "consecutive_success_steps": SUCCESS_CONSECUTIVE_STEPS,
            },
        )
        gear_dropped: DoneTerm = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={
                "minimum_height": TABLE_TOP_Z - 0.12,
                "asset_cfg": SceneEntityCfg(held_asset.name),
            },
        )

    class GearAssemblyTask(TaskBase):
        """Pick up the medium gear and insert it onto the matching base peg."""

        def __init__(self) -> None:
            super().__init__(
                episode_length_s=cfg.episode_length_s,
                task_description="Pick up the medium gear and insert it onto the matching peg in the gear base.",
            )
            self.events_cfg = GearAssemblyEventsCfg()
            self.terminations_cfg = GearAssemblyTerminationsCfg()

        def get_scene_cfg(self):
            return None

        def get_termination_cfg(self):
            return self.terminations_cfg

        def get_events_cfg(self):
            return self.events_cfg

        def get_mimic_env_cfg(self, arm_mode: ArmMode):
            return None

        def get_metrics(self) -> list[MetricBase]:
            return [SuccessRateMetric(), ObjectMovedRateMetric(held_asset)]

        def get_viewer_cfg(self) -> ViewerCfg:
            return ViewerCfg(eye=(1.45, 1.10, 0.90), lookat=(0.48, 0.02, 0.08))

    return GearAssemblyTask()
