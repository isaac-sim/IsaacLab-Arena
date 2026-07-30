# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass


class PreStepFlatCameraObservationsRecorder(RecorderTerm):
    """Recorder term that records the camera observations in each step."""

    def record_pre_step(self):
        return "camera_obs", self._env.obs_buf["camera_obs"]


@configclass
class PreStepFlatCameraObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the camera observation recorder term."""

    class_type: type[RecorderTerm] = PreStepFlatCameraObservationsRecorder


class PostStepFlatPolicyActionObservationRecorder(RecorderTerm):
    """Recorder term that records the ``action`` observation group at the end of each step.

    Mirrors the locomanip mimic patch's post-step action recorder, but no-ops on envs
    whose policy does not expose an ``action`` observation group so it can be safely
    enabled for any task.
    """

    def record_post_step(self):
        obs_buf = getattr(self._env, "obs_buf", None)
        if not isinstance(obs_buf, dict) or "action" not in obs_buf:
            return None, None
        return "action", obs_buf["action"]


@configclass
class PostStepFlatPolicyActionObservationRecorderCfg(RecorderTermCfg):
    """Configuration for the post-step ``action`` observation recorder term."""

    class_type: type[RecorderTerm] = PostStepFlatPolicyActionObservationRecorder


@configclass
class ArenaEnvRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Action/state recorder manager extended with arena-specific recorder terms."""

    record_pre_step_flat_camera_observations = PreStepFlatCameraObservationsRecorderCfg()
    record_post_step_flat_policy_action_observations = PostStepFlatPolicyActionObservationRecorderCfg()
