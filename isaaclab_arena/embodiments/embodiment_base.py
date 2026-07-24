# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.managers import EventTermCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg

from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.relations.placement_asset import PlacementAsset
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.cameras import ArenaCameraCfg, make_camera_observation_cfg
from isaaclab_arena.utils.configclass import combine_configclass_instances
from isaaclab_arena.utils.pose import Pose, PosePerEnv, PoseRange

if TYPE_CHECKING:
    import trimesh


class EmbodimentBase(PlacementAsset):

    name: str | None = None
    tags: list[str] = ["embodiment"]
    default_arm_mode: ArmMode | None = None

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
    ):
        assert self.name is not None, "Embodiment name is required"
        super().__init__(name=self.name, tags=self.tags)
        if "embodiment" not in self.tags:
            self.tags.append("embodiment")
        self.enable_cameras = enable_cameras
        self.initial_pose = initial_pose
        self.concatenate_observation_terms = concatenate_observation_terms
        self.arm_mode = arm_mode or self.default_arm_mode
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
        self.termination_cfg: Any | None = None
        self._collision_mesh: trimesh.Trimesh | None = None
        """Lazily-extracted robot collision mesh, cached so the USD is opened once."""
        self.pose_event_cfg: EventTermCfg | None = None
        """Root-pose reset event applied on environment reset; ``None`` when the embodiment has no configured pose."""

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return root-relative bounds computed from the articulation's USD geometry."""
        # Import locally because USD/pxr is available only after simulation initialization.
        from isaaclab_arena.utils.usd_helpers import compute_local_bounding_box_from_usd

        assert self.scene_config is not None, "scene_config must be populated before placement"
        robot = self.scene_config.robot
        assert robot is not None, "scene_config.robot must be populated before placement"
        spawn = robot.spawn
        assert spawn.usd_path is not None, "scene_config.robot must use a USD spawn for placement"
        scale = tuple(spawn.scale or (1.0, 1.0, 1.0))
        # TODO(zihaox): Account for configured initial joint positions in bounds and collision meshes.
        return compute_local_bounding_box_from_usd(spawn.usd_path, scale)

    def get_collision_mesh(self) -> trimesh.Trimesh | None:
        """Return the robot's collision mesh from its USD default prim, in the default joint pose."""
        if self._collision_mesh is None:
            # Import locally because USD/pxr is available only after simulation initialization.
            from pxr import Usd

            from isaaclab_arena.utils.usd_helpers import extract_trimesh_from_prim

            assert self.scene_config is not None, "scene_config must be populated before placement"
            robot = self.scene_config.robot
            assert robot is not None, "scene_config.robot must be populated before placement"
            spawn = robot.spawn
            assert spawn.usd_path is not None, "scene_config.robot must use a USD spawn for placement"
            scale = tuple(spawn.scale or (1.0, 1.0, 1.0))
            stage = Usd.Stage.Open(spawn.usd_path)
            assert stage is not None, f"could not open robot USD: {spawn.usd_path}"
            default_prim = stage.GetDefaultPrim() or stage.GetPseudoRoot()
            # The default prim scopes extraction to the robot, excluding sibling scene props
            # (ground planes, stray objects) baked into some flattened articulation USDs.
            self._collision_mesh = extract_trimesh_from_prim(stage, default_prim.GetPath().pathString, scale)
        return self._collision_mesh

    def _get_initial_pose_as_pose(self) -> Pose | None:
        """Return a single ``Pose`` for scene construction; ``PosePerEnv`` collapses to env 0."""
        initial_pose = self.initial_pose
        if initial_pose is None:
            return None
        if isinstance(initial_pose, PosePerEnv):
            return initial_pose.poses[0]
        assert isinstance(initial_pose, Pose), "Embodiments support a fixed Pose or PosePerEnv only"
        return initial_pose

    def _set_pose_state(self, pose: Pose | PosePerEnv) -> None:
        """Store the configured pose; the construction pose is materialized in ``get_scene_cfg``."""
        self.initial_pose = pose

    def set_spawn_pose(self, pose: Pose) -> None:
        """Set the scene-construction pose without rebuilding the reset event."""
        self._set_pose_state(pose)

    def set_initial_pose(self, pose: Pose | PoseRange | PosePerEnv) -> None:
        """Set the embodiment root pose and rebuild its pose reset event."""
        assert isinstance(pose, (Pose, PosePerEnv)), "Embodiments support a fixed Pose or PosePerEnv only"
        self._set_pose_state(pose)
        self.pose_event_cfg = self._init_pose_event_cfg()

    def _init_pose_event_cfg(self) -> EventTermCfg | None:
        """Build the reset event that restores this embodiment's root pose (and auxiliary prims)."""
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

    def has_pose_reset_event(self) -> bool:
        """Return whether the embodiment owns a root-pose reset event."""
        return self.pose_event_cfg is not None

    def set_joint_initial_pos(self, joint_pos: Mapping[str, float]) -> None:
        """Update the robot's initial joint positions by joint name."""
        assert self.scene_config is not None, "scene_config must be populated before setting joint positions"
        robot = self.scene_config.robot
        assert robot is not None, "scene_config.robot must be populated before setting joint positions"
        robot.init_state.joint_pos.update(joint_pos)

    def get_initial_pose(self) -> Pose | PosePerEnv:
        """Env-local robot base pose, resolved in order: the explicit ``initial_pose`` override if set,
        otherwise the ``scene_config`` robot ``init_state`` default."""
        if self.initial_pose is not None:
            return self.initial_pose

        assert hasattr(self.scene_config, "robot"), "scene_config must be populated with a `robot`."
        init_state = self.scene_config.robot.init_state
        return Pose(
            position_xyz=tuple(float(v) for v in init_state.pos),
            rotation_xyzw=tuple(float(v) for v in init_state.rot),
        )

    def get_scene_cfg(self) -> Any:
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
        if self.pose_event_cfg is None:
            return self.event_config
        from isaaclab_arena.utils.configclass import make_configclass

        pose_reset_cfg = make_configclass(
            "EmbodimentPoseResetCfg",
            [("robot_reset_pose", EventTermCfg, self.pose_event_cfg)],
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
        assert scene_config is not None, "scene_config must be populated before setting the root pose"
        robot = scene_config.robot
        assert robot is not None, "scene_config.robot must be populated before setting the root pose"
        robot.init_state.pos = pose.position_xyz
        robot.init_state.rot = pose.rotation_xyzw
        return scene_config

    def get_recorder_term_cfg(self) -> RecorderManagerBaseCfg:
        return None

    def get_termination_cfg(self) -> Any:
        return self.termination_cfg

    def get_scene_name(self) -> str:
        """Return the embodiment's Isaac Lab scene key."""
        return "robot"

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        # In case of multiple ee frames one can use self.mimic_arm_mode to get the correct ee frame name
        return ""

    def get_command_body_name(self) -> str:
        return ""

    def get_arm_mode(self) -> ArmMode:
        return self.arm_mode
