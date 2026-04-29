# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mdp.recorders.recorders_cfg import (
    InitialStateRecorderCfg,
    PostStepProcessedActionsRecorderCfg,
    PostStepStatesRecorderCfg,
    PreStepFlatPolicyObservationsRecorderCfg,
)
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.action_constants import (
    NAVIGATE_CMD_END_IDX,
    NAVIGATE_CMD_START_IDX,
)


class PostStepG1ObservationsActionRecorder(RecorderTerm):
    def record_post_step(self):
        return "action", self._env.obs_buf["action"]


@configclass
class PostStepG1ObservationsActionRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PostStepG1ObservationsActionRecorder


class PreStepG1LocomanipActionRecorder(RecorderTerm):
    def record_pre_step(self):
        actions = self._env.action_manager.action
        for term_name in self._env.action_manager.active_terms:
            if hasattr(self._env.action_manager.get_term(term_name), "navigate_cmd"):
                actions[:, NAVIGATE_CMD_START_IDX:NAVIGATE_CMD_END_IDX] = self._env.action_manager.get_term(
                    term_name
                ).navigate_cmd
        return "actions", actions


@configclass
class PreStepG1LocomanipActionRecorderCfg(RecorderTermCfg):
    """Configuration for the step action recorder term."""

    class_type: type[RecorderTerm] = PreStepG1LocomanipActionRecorder


class PreStepFlatCameraObservationsRecorder(RecorderTerm):
    """Recorder term that records the camera observations in each step."""

    def record_pre_step(self):
        return "camera_obs", self._env.obs_buf["camera_obs"]


@configclass
class PreStepFlatCameraObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the camera observation recorder term."""

    class_type: type[RecorderTerm] = PreStepFlatCameraObservationsRecorder


@configclass
class G1LocomanipRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder configurations for recording G1 locomanipulation actions and states."""

    record_initial_state = InitialStateRecorderCfg()
    record_post_step_states = PostStepStatesRecorderCfg()
    record_pre_step_actions = PreStepG1LocomanipActionRecorderCfg()
    record_pre_step_flat_policy_observations = PreStepFlatPolicyObservationsRecorderCfg()
    record_post_step_processed_actions = PostStepProcessedActionsRecorderCfg()
    record_post_step_flat_policy_observations = PostStepG1ObservationsActionRecorderCfg()
    record_pre_step_flat_camera_observations = PreStepFlatCameraObservationsRecorderCfg()
