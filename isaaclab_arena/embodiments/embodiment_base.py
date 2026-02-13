# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.relations.relations import AtPosition, Relation, RelationBase
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox, quaternion_to_90_deg_z_quarters
from isaaclab_arena.utils.cameras import make_camera_observation_cfg
from isaaclab_arena.utils.configclass import combine_configclass_instances
from isaaclab_arena.utils.pose import Pose, PoseRange
from isaaclab_arena.utils.usd_helpers import compute_local_bounding_box_from_usd


class EmbodimentBase(Asset):

    name: str | None = None
    tags: list[str] = ["embodiment"]
    default_arm_mode: ArmMode | None = None

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        bounding_box: AxisAlignedBoundingBox | None = None,
    ):
        self.enable_cameras = enable_cameras
        self.initial_pose = initial_pose
        self.concatenate_observation_terms = concatenate_observation_terms
        self.arm_mode = arm_mode or self.default_arm_mode
        self.bounding_box = bounding_box
        self.relations: list[RelationBase] = []
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

    def set_initial_pose(self, pose: Pose | PoseRange) -> None:
        self.initial_pose = pose

    def get_initial_pose(self) -> Pose | PoseRange | None:
        """Get the initial pose of the embodiment.

        Returns:
            The initial pose, or None if not set.
        """
        return self.initial_pose

    def add_relation(self, relation: RelationBase) -> None:
        """Add a spatial relation or marker to this embodiment.

        Args:
            relation: The relation to add (e.g. NextTo, On, IsAnchor).
        """
        self.relations.append(relation)

    def get_relations(self) -> list[RelationBase]:
        """Get all relations for this embodiment."""
        return self.relations

    def get_spatial_relations(self) -> list[RelationBase]:
        """Get only spatial relations (On, NextTo, AtPosition, etc.), excluding markers like IsAnchor."""
        return [r for r in self.relations if isinstance(r, (Relation, AtPosition))]

    def set_bounding_box(self, bounding_box: AxisAlignedBoundingBox) -> None:
        """Set the bounding box representing the embodiment's footprint.

        Args:
            bounding_box: Local axis-aligned bounding box for the embodiment.
        """
        self.bounding_box = bounding_box

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Get local bounding box (relative to embodiment origin).

        If no bounding box has been set manually (via constructor or set_bounding_box()),
        attempts to compute it automatically from the robot USD in the scene config.

        Returns:
            The embodiment's local bounding box.

        Raises:
            RuntimeError: If no bounding box is set and it cannot be computed from
                the scene config (e.g. scene_config is None or has no robot field).
        """
        if self.bounding_box is None:
            usd_path = self._get_robot_usd_path()
            if usd_path is None:
                raise RuntimeError(
                    f"Cannot compute bounding box for embodiment '{self.name}': "
                    "could not find a robot USD path at scene_config.robot.spawn.usd_path. "
                    "Either set a bounding box manually via set_bounding_box(), "
                    "or ensure the embodiment's scene_config has a robot with a USD spawn path."
                )
            self.bounding_box = compute_local_bounding_box_from_usd(usd_path)
        return self.bounding_box

    def _get_robot_usd_path(self) -> str | None:
        """Extract the robot USD path from the scene config, if available.

        Returns:
            The USD path string, or None if not available.
        """
        if self.scene_config is None:
            return None
        robot_cfg = getattr(self.scene_config, "robot", None)
        if robot_cfg is None:
            return None
        spawn_cfg = getattr(robot_cfg, "spawn", None)
        if spawn_cfg is None:
            return None
        usd_path = getattr(spawn_cfg, "usd_path", None)
        return usd_path

    def get_world_bounding_box(self) -> AxisAlignedBoundingBox:
        """Get bounding box in world coordinates (local bbox rotated and translated).

        Only 90 degree rotations around Z axis are supported. If initial_pose is
        not set or is a PoseRange, returns the local bounding box without transformation.

        Returns:
            The embodiment's bounding box in world coordinates.
        """
        local_bbox = self.get_bounding_box()
        if self.initial_pose is None or not isinstance(self.initial_pose, Pose):
            return local_bbox
        quarters = quaternion_to_90_deg_z_quarters(self.initial_pose.rotation_wxyz)
        return local_bbox.rotated_90_around_z(quarters).translated(self.initial_pose.position_xyz)

    def get_scene_cfg(self) -> Any:
        if self.initial_pose is not None:
            # _update_scene_cfg_with_robot_initial_pose expects a fixed Pose.
            # If initial_pose is a PoseRange, use its midpoint for the scene config.
            pose = self.initial_pose if isinstance(self.initial_pose, Pose) else self.initial_pose.get_midpoint()
            self.scene_config = self._update_scene_cfg_with_robot_initial_pose(self.scene_config, pose)
        if self.enable_cameras:
            if self.camera_config is not None:
                return combine_configclass_instances(
                    "SceneCfg",
                    self.scene_config,
                    self.camera_config,
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
        return self.event_config

    def get_mimic_env(self) -> ManagerBasedRLMimicEnv:
        return self.mimic_env

    def get_xr_cfg(self) -> Any:
        return self.xr

    def get_camera_cfg(self) -> Any:
        return self.camera_config

    def _update_scene_cfg_with_robot_initial_pose(self, scene_config: Any, pose: Pose) -> Any:
        if scene_config is None or not hasattr(scene_config, "robot"):
            raise RuntimeError("scene_config must be populated with a `robot` before calling `set_robot_initial_pose`.")
        scene_config.robot.init_state.pos = pose.position_xyz
        scene_config.robot.init_state.rot = pose.rotation_wxyz
        return scene_config

    def get_recorder_term_cfg(self) -> RecorderManagerBaseCfg:
        return None

    def get_termination_cfg(self) -> Any:
        return self.termination_cfg

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
        return env_cfg

    def get_embodiment_name_in_scene(self) -> str:
        return "robot"

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        # In case of multiple ee frames one can use self.mimic_arm_mode to get the correct ee frame name
        return ""

    def get_command_body_name(self) -> str:
        return ""

    def get_arm_mode(self) -> ArmMode:
        return self.arm_mode
