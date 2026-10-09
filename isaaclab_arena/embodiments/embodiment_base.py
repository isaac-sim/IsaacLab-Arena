# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import keyword
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.managers import EventTermCfg

from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.gripper import Gripper
from isaaclab_arena.relations.collision_mode import CollisionMode
from isaaclab_arena.relations.placement_asset import PlaceableAsset
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.cameras import ArenaCameraCfg, make_camera_observation_cfg
from isaaclab_arena.utils.configclass import combine_configclass_instances, transform_configclass_instance
from isaaclab_arena.utils.physics_backend import PhysicsBackend
from isaaclab_arena.utils.pose import Pose, PosePerEnv, PoseRange

if TYPE_CHECKING:
    import trimesh


@dataclass(frozen=True)
class ArticulationGeometrySpec:
    """USD articulation state used to compute embodiment geometry."""

    usd_path: str
    """Robot USD, as spawned."""

    scale: tuple[float, float, float]
    """Per-axis spawn scale."""

    joint_pos: Mapping[str, float]
    """Joint positions to pose the geometry at, revolute in radians, keyed by name or Isaac Lab regex."""


class EmbodimentBase(PlaceableAsset):

    name: str | None = None
    tags: list[str] = ["embodiment"]
    default_arm_mode: ArmMode | None = None
    instance_key: str | None
    """Key that names this robot instance's scene entities and terms, or None for the unkeyed names."""
    embodiment_type: str
    """Registered embodiment type; the asset name is the instance key when one is given."""
    gripper: Gripper | None
    """Gripper attached to the robot body, when the embodiment defines one."""
    spawn_cfg_addon: dict[str, dict[str, Any]] = {}
    """Define how embodiment USD/geometry is spawned and which schemas/properties are set."""

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        collision_mode: CollisionMode | str | None = None,
        spawn_cfg_addon: dict[str, dict[str, Any]] | None = None,
        instance_key: str | None = None,
    ):
        assert self.name is not None, "Embodiment name is required"
        assert instance_key is None or (
            instance_key.isidentifier()
            and instance_key.isascii()
            and instance_key.islower()
            and not keyword.iskeyword(instance_key)
            and instance_key != "robot"
        ), (
            f"Instance key {instance_key!r} must be a lowercase ASCII identifier that is neither a Python keyword"
            " nor 'robot'"
        )
        self.embodiment_type = self.name
        self.instance_key = instance_key
        super().__init__(name=instance_key or self.name, tags=self.tags, collision_mode=collision_mode)
        if "embodiment" not in self.tags:
            self.tags.append("embodiment")
        self.enable_cameras = enable_cameras
        self.initial_pose = initial_pose
        self.concatenate_observation_terms = concatenate_observation_terms
        self.arm_mode = arm_mode or self.default_arm_mode
        self.gripper = None
        # Give each robot its own copy so changes don't affect other robots.
        self.spawn_cfg_addon = deepcopy(self.spawn_cfg_addon if spawn_cfg_addon is None else spawn_cfg_addon)
        # These should be filled by the subclass
        self.scene_config: Any | None = None
        self.camera_config: Any | None = None
        self.action_config: Any | None = None
        self.observation_config: Any | None = None
        self.event_config: Any | None = None
        self.reward_config: Any | None = None
        self.curriculum_config: Any | None = None
        self.command_config: Any | None = None
        self.mimic_env: Any | None = None
        self.xr: Any | None = None
        self._configured_physics_backend: PhysicsBackend | None = None

    def get_placement_geometry_source(self) -> ArticulationGeometrySpec:
        """Return the USD articulation state used to compute embodiment geometry."""
        robot = self.get_robot_cfg()
        spawn = robot.spawn
        assert spawn.usd_path is not None, "The robot articulation must use a USD spawn for placement"
        scale_x, scale_y, scale_z = spawn.scale or (1.0, 1.0, 1.0)
        return ArticulationGeometrySpec(
            usd_path=spawn.usd_path,
            scale=(scale_x, scale_y, scale_z),
            joint_pos=dict(robot.init_state.joint_pos or {}),
        )

    def get_bounding_box(self, prim_path: str | None = None) -> AxisAlignedBoundingBox:
        """Return root-relative bounds of the articulation posed at its configured joint positions.

        Args:
            prim_path: Optional sub-prim to bound (e.g. stand only). When None, bounds the
                full default prim.
        """
        # Import locally because USD/pxr is available only after simulation initialization.
        from isaaclab_arena.utils.usd.helpers import compute_local_bounding_box_from_usd_at_joint_pos

        source = self.get_placement_geometry_source()
        return compute_local_bounding_box_from_usd_at_joint_pos(
            source.usd_path, source.joint_pos, source.scale, prim_path=prim_path
        )

    def get_collision_mesh(self) -> trimesh.Trimesh | None:
        """Return the robot mesh from its USD default prim."""
        # Import locally because USD/pxr is available only after simulation initialization.
        from isaaclab_arena.utils.usd.helpers import extract_trimesh_from_usd_path

        source = self.get_placement_geometry_source()
        return extract_trimesh_from_usd_path(source.usd_path, source.scale)

    def _set_initial_pose(self, pose: Pose | PoseRange | PosePerEnv) -> None:
        """Store the configured pose; the construction pose is applied in ``get_scene_cfg``."""
        assert isinstance(pose, (Pose, PosePerEnv)), "Embodiments support a fixed Pose or PosePerEnv only"
        self.initial_pose = pose

    def _get_initial_pose_as_pose(self) -> Pose | None:
        # Read the explicit override, not get_initial_pose()'s scene_config fallback: an unset pose must
        # collapse to None so get_scene_cfg leaves the init-configured robot/stand states untouched.
        return self._collapse_pose_to_single(self.initial_pose)

    def _build_reset_event(self) -> EventTermCfg | None:
        """Build the reset event that restores this embodiment's root pose (and auxiliary prims)."""
        # NOTE(zihaox): This is nearly a duplicate of ObjectBase._build_reset_event. The two diverge only
        # because an embodiment routes its writes through layout_pose_to_scene_writes so a compound asset
        # can also move its auxiliary prims (e.g. Droid's stand). Unify with ObjectBase once auxiliary
        # prims travel with their parent on reset and the two paths converge.
        from isaaclab_arena.terms.events import reset_placement_asset_pose, reset_placement_asset_pose_per_env

        initial_pose = self.initial_pose
        if initial_pose is None:
            return None
        if isinstance(initial_pose, PosePerEnv):
            return EventTermCfg(
                func=reset_placement_asset_pose_per_env,
                mode="reset",
                params={"write_pose_list": [self.layout_pose_to_scene_writes(pose) for pose in initial_pose.poses]},
            )
        assert isinstance(initial_pose, Pose), "Embodiments support a fixed Pose or PosePerEnv only"
        return EventTermCfg(
            func=reset_placement_asset_pose,
            mode="reset",
            params={"scene_writes": self.layout_pose_to_scene_writes(initial_pose)},
        )

    def set_joint_initial_pos(self, joint_pos: Mapping[str, float]) -> None:
        """Update the robot's initial joint positions by joint name."""
        self.get_robot_cfg().init_state.joint_pos.update(joint_pos)

    def get_initial_pose(self) -> Pose | PosePerEnv:
        """Env-local robot base pose, resolved in order: the explicit ``initial_pose`` override if set,
        otherwise the ``scene_config`` robot ``init_state`` default."""
        if self.initial_pose is not None:
            return self.initial_pose

        init_state = self.get_robot_cfg().init_state
        return Pose(
            position_xyz=tuple(float(v) for v in init_state.pos),
            rotation_xyzw=tuple(float(v) for v in init_state.rot),
        )

    def configure_physics_backend(self, backend: PhysicsBackend) -> None:
        """Apply physics-backend-specific overrides before the env cfg is composed."""
        if self._configured_physics_backend == backend:
            return
        assert self._configured_physics_backend is None, (
            f"Embodiment '{self.name}' is already configured for physics backend "
            f"'{self._configured_physics_backend.value}' and cannot be reconfigured for '{backend}'."
        )
        self._configure_physics_backend(backend)
        self._configured_physics_backend = backend

    def _configure_physics_backend(self, backend: PhysicsBackend) -> None:
        """Apply subclass-specific physics-backend overrides."""

    def _apply_spawn_cfg_addons(self) -> None:
        """Apply this embodiment's named spawn addons after backend-specific defaults."""
        from isaaclab_arena.assets.physics_spawner import make_usd_spawn_cfg_with_addons

        replacements = {}
        for name, addons in self.spawn_cfg_addon.items():
            robot_cfg = getattr(self.scene_config, name, None)
            assert robot_cfg is not None, f"Embodiment spawn addon references unknown scene entry {name!r}"
            assert getattr(robot_cfg, "spawn", None) is not None, f"Embodiment scene entry {name!r} has no spawn config"
            replacements[name] = make_usd_spawn_cfg_with_addons(robot_cfg.spawn, addons)
        # Publish only after every embodiment entry validates, avoiding half-applied bimanual settings.
        for name, spawn_cfg in replacements.items():
            getattr(self.scene_config, name).spawn = spawn_cfg

    def get_scene_cfg(self) -> Any:
        # Apply task settings whenever the scene is collected, even without a backend hook.
        self._apply_spawn_cfg_addons()
        construction_pose = self._get_initial_pose_as_pose()
        if construction_pose is not None:
            self.scene_config = self._update_scene_cfg_with_robot_initial_pose(self.scene_config, construction_pose)
        if self.enable_cameras:
            if self.camera_config is not None:
                return combine_configclass_instances(
                    "SceneCfg",
                    self.scene_config,
                    self.get_camera_cfg(),
                )
        return self.scene_config

    def get_action_cfg(self) -> Any:
        return self.action_config

    def get_observation_cfg(self) -> Any:
        if self.enable_cameras:
            if self.camera_config is not None:
                camera_observation_config = make_camera_observation_cfg(self.camera_config)
                return combine_configclass_instances(
                    "ObservationCfg",
                    self.observation_config,
                    camera_observation_config,
                )
        return self.observation_config

    def get_rewards_cfg(self) -> Any:
        return self.reward_config

    def get_curriculum_cfg(self) -> Any:
        return self.curriculum_config

    def get_commands_cfg(self) -> Any:
        return self.command_config

    def get_events_cfg(self) -> Any:
        if self._pose_event_cfg is None:
            return self.event_config
        from isaaclab_arena.utils.configclass import make_configclass

        pose_reset_cfg = make_configclass(
            "EmbodimentPoseResetCfg",
            [(self.get_instance_name("robot_reset_pose"), EventTermCfg, self._pose_event_cfg)],
        )()
        # Merge the pose reset last so it runs after joint/root resets in ``event_config``.
        return combine_configclass_instances("EventsCfg", self.event_config, pose_reset_cfg)

    def get_mimic_env(self) -> ManagerBasedRLMimicEnv:
        return self.mimic_env

    def get_xr_cfg(self) -> Any:
        return self.xr

    def get_teleop_target_frame_prim_path(self) -> str | None:
        """Optional USD prim path for rebasing teleop poses (e.g. robot base link). Returns None if not set."""

    def get_camera_cfg(self) -> Any:
        if self.camera_config is None:
            return None
        # In Arena we expect camera configs to inherit from ArenaCameraCfg.
        assert isinstance(
            self.camera_config, ArenaCameraCfg
        ), f"Expected camera_config to inherit from ArenaCameraCfg; got {type(self.camera_config).__name__}."
        return self.camera_config.get_cfg()

    def add_camera_variations(self, camera_rig: ArenaCameraCfg) -> None:
        """Register extrinsics and intrinsics variations for every camera in ``camera_rig``."""
        from isaaclab_arena.variations.camera_extrinsics_variation import CameraExtrinsicsVariation
        from isaaclab_arena.variations.camera_intrinsics_variation import CameraIntrinsicsVariation

        for camera_name in camera_rig.camera_names():
            self.add_variation(CameraExtrinsicsVariation(camera_name=camera_name))
            self.add_variation(CameraIntrinsicsVariation(camera_name=camera_name, camera_rig=camera_rig))

    def _update_scene_cfg_with_robot_initial_pose(self, scene_config: Any, pose: Pose) -> Any:
        robot = getattr(scene_config, self.get_scene_key(), None)
        assert robot is not None, f"scene_config.{self.get_scene_key()} must be populated before setting the root pose"
        robot.init_state.pos = pose.position_xyz
        robot.init_state.rot = pose.rotation_xyzw
        return scene_config

    def get_recorder_term_cfg(self, record_trajectories: bool = False) -> Any:
        """Return this embodiment's recorder terms, or None if it defines none.

        Args:
            record_trajectories: Whether to also include the per-step trajectory recorder terms,
                built with this embodiment's own frame transformers and scene key.
        """
        if not record_trajectories:
            return None
        from isaaclab_arena.terms.recorders import make_trajectory_recorder_terms_cfg

        return make_trajectory_recorder_terms_cfg(
            frame_transformer_names=self.get_ee_frame_transformer_names(),
            asset_name=self.get_scene_key(),
            term_name=self.get_instance_name,
        )

    def get_scene_key(self) -> str:
        """Return the embodiment's Isaac Lab scene key: the instance key, or "robot" when unkeyed."""
        return self.get_instance_name("robot")

    def get_instance_name(self, name: str) -> str:
        """Return the name this robot instance gives a scene entity, frame, or manager term.

        Unkeyed robots keep every name. A keyed robot names its "robot" articulation by the key and
        prefixes every other name with "<key>_", so several robots can share one environment.

        Args:
            name: The unkeyed name, for example "robot", "ee_frame", or "arm_action".
        """
        if self.instance_key is None:
            return name
        return self.instance_key if name == "robot" else f"{self.instance_key}_{name}"

    def get_robot_prim_path(self) -> str:
        """Return the robot's root prim path: "{ENV_REGEX_NS}/<key>", or "{ENV_REGEX_NS}/Robot" when unkeyed."""
        return "{ENV_REGEX_NS}/" + (self.instance_key or "Robot")

    def with_instance_names(self, cfg: Any, bases: tuple[type, ...] = ()) -> Any:
        """Return a configclass with each top-level field of ``cfg`` named by ``get_instance_name``.

        Field values and order are unchanged, and unkeyed robots get ``cfg`` itself. Call it once
        while constructing the configurations, after their values carry this robot's names.

        When keyed, the returned class keeps only the fields of ``cfg`` and ``bases``: methods,
        ``__post_init__``, and non-field attributes of ``cfg``'s class are not carried over. Pass
        non-configclass bases such as ``ArenaCameraCfg`` in ``bases``, and set non-field state such
        as camera tiling after naming.

        Args:
            cfg: Configclass instance whose field names are the unkeyed names.
            bases: Base classes of the returned configclass, for example the camera rig base class.
        """
        if self.instance_key is None:
            return cfg

        def name_fields(fields: list[tuple[str, type, Any]]) -> list[tuple[str, type, Any]]:
            return [(self.get_instance_name(name), field_type, value) for name, field_type, value in fields]

        return transform_configclass_instance(cfg, name_fields, bases=bases)

    def get_robot_cfg(self) -> Any:
        """Return the robot articulation configuration stored in ``scene_config`` under the scene key."""
        robot = getattr(self.scene_config, self.get_scene_key(), None)
        assert robot is not None, f"scene_config.{self.get_scene_key()} must be populated"
        return robot

    def get_scene_root_keys(self) -> tuple[str, ...]:
        """Return every rigid-object and articulation root configured by this embodiment."""
        if self.scene_config is None:
            return ()
        return tuple(
            name for name, cfg in vars(self.scene_config).items() if isinstance(cfg, (ArticulationCfg, RigidObjectCfg))
        )

    def set_initial_scene_root_poses(self, poses: dict[str, PosePerEnv]) -> None:
        """Seed each measured embodiment root, keeping independent roots independently positioned."""
        assert set(poses) == set(self.get_scene_root_keys()), "Provide every embodiment scene root"
        if tuple(poses) == (self.get_scene_key(),):
            super().set_initial_scene_root_poses(poses)
            return
        # Independent articulation roots have no single logical placement pose.
        self.initial_pose = None
        for name, per_env_poses in poses.items():
            root_cfg = getattr(self.scene_config, name)
            pose = per_env_poses.poses[0]
            root_cfg.init_state.pos = pose.position_xyz
            root_cfg.init_state.rot = pose.rotation_xyzw

    def get_ee_frame_transformer_names(self) -> list[str]:
        """Names of the scene's end-effector frame transformer sensors.

        Override for embodiments with more than one tracked end-effector (e.g. bi-manual robots),
        or whose single frame transformer is not named "ee_frame".
        """
        return [self.get_instance_name("ee_frame")]

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        # In case of multiple ee frames one can use self.mimic_arm_mode to get the correct ee frame name
        return ""

    def get_command_body_name(self) -> str:
        return ""

    def get_gripper(self) -> Gripper:
        """Return this embodiment's supported width-reporting gripper."""
        assert self.gripper is not None, f"Embodiment '{self.name}' has no supported gripper."
        return self.gripper

    def get_arm_mode(self) -> ArmMode:
        return self.arm_mode
