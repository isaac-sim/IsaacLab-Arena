# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Preserve Franka composition from upstream commit 489f3a755 for comparison.

Configuration classes are unchanged by the instance-key contribution. This fixture
reconstructs their original initialization and getters independently of the changed
Franka constructor and getter implementations.
"""

from typing import Any

from isaaclab.managers import EventTermCfg

from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.embodiments.franka.franka import (
    FRANKA_PANDA_HIGH_PD_CFG,
    FrankaCameraCfg,
    FrankaEventCfg,
    FrankaIKActionCfg,
    FrankaIKEmbodiment,
    FrankaMimicEnv,
    FrankaObservationsCfg,
    FrankaRewardsCfg,
    FrankaSceneCfg,
    _franka_robot_cfg_on_stand,
)
from isaaclab_arena.utils.cameras import ArenaCameraCfg, make_camera_observation_cfg
from isaaclab_arena.utils.configclass import combine_configclass_instances


class LegacyFrankaIKEmbodiment(FrankaIKEmbodiment):
    """Build the unkeyed Franka using its pre-change initialization and getters."""

    def __init__(self, enable_cameras=False):
        EmbodimentBase.__init__(self, enable_cameras=enable_cameras)
        self.event_config = FrankaEventCfg()
        self.reward_config = FrankaRewardsCfg()
        self.mimic_env = FrankaMimicEnv
        self.camera_config = FrankaCameraCfg()
        self.scene_config = FrankaSceneCfg()
        self.observation_config = FrankaObservationsCfg()
        self.observation_config.policy.concatenate_terms = self.concatenate_observation_terms
        self.add_camera_variations(self.camera_config)
        self.scene_config.robot = _franka_robot_cfg_on_stand(FRANKA_PANDA_HIGH_PD_CFG.copy())
        self.action_config = FrankaIKActionCfg()

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
        if self._pose_event_cfg is None:
            return self.event_config
        from isaaclab_arena.utils.configclass import make_configclass

        pose_reset_cfg = make_configclass(
            "EmbodimentPoseResetCfg",
            [("robot_reset_pose", EventTermCfg, self._pose_event_cfg)],
        )()
        # Merge the pose reset last so it runs after joint/root resets in ``event_config``.
        return combine_configclass_instances("EventsCfg", self.event_config, pose_reset_cfg)

    def get_camera_cfg(self) -> Any:
        if self.camera_config is None:
            return None
        # In Arena we expect camera configs to inherit from ArenaCameraCfg.
        assert isinstance(
            self.camera_config, ArenaCameraCfg
        ), f"Expected camera_config to inherit from ArenaCameraCfg; got {type(self.camera_config).__name__}."
        return self.camera_config.get_cfg()

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
            frame_transformer_names=self.get_ee_frame_transformer_names(), asset_name=self.get_scene_key()
        )

    def get_termination_cfg(self) -> Any:
        return self.termination_cfg

    def get_scene_key(self) -> str:
        """Return the embodiment's Isaac Lab scene key."""
        return "robot"

    def get_ee_frame_transformer_names(self) -> list[str]:
        """Names of the scene's end-effector frame transformer sensors.

        Override for embodiments with more than one tracked end-effector (e.g. bi-manual robots),
        or whose single frame transformer is not named "ee_frame".
        """
        return ["ee_frame"]
