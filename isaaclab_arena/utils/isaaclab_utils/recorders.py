# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.envs.mdp.recorders.recorders_cfg import (
    ActionStateRecorderManagerCfg,
    InitialStateRecorderCfg,
    PostStepProcessedActionsRecorderCfg,
    PostStepStatesRecorderCfg,
    PreStepActionsRecorderCfg,
)
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.utils.configclass import combine_configclass_instances


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


class EpisodeIdentityRecorder(RecorderTerm):
    """Recorder term that stamps each exported demo with the episode it came from.

    Demos are named ``demo_0``, ``demo_1``, ... in export order, which carries no reference back to
    the ``(env_id, episode_in_env)`` pair the episode recorder writes to its JSONL. Recording the
    pair alongside the trajectory makes that join explicit.
    """

    def __init__(self, cfg: RecorderTermCfg, env) -> None:
        super().__init__(cfg, env)
        self._first_reset = True

    def record_pre_reset(self, env_ids: Sequence[int] | None):
        # The initial reset touches every env before any episode has run; there is nothing to stamp.
        if self._first_reset:
            self._first_reset = False
            return None, None
        env_ids = list(range(self._env.num_envs)) if env_ids is None else [int(env_id) for env_id in env_ids]
        # Runs before the env advances its counters, so this is still the finishing episode's index.
        episode_indices = [self._env.get_episode_index(env_id) for env_id in env_ids]
        return "episode_id", {
            "env_id": torch.tensor(env_ids, dtype=torch.int64, device=self._env.device),
            "episode_in_env": torch.tensor(episode_indices, dtype=torch.int64, device=self._env.device),
        }


@configclass
class EpisodeIdentityRecorderCfg(RecorderTermCfg):
    """Configuration for the episode identity recorder term."""

    class_type: type[RecorderTerm] = EpisodeIdentityRecorder


@configclass
class TrajectoryRecorderTermsCfg:
    """Recorder terms capturing per-step robot and object trajectories.

    Camera observations are deliberately excluded: at Arena's default resolutions they outweigh the
    state and action terms by three orders of magnitude. Use the imitation-learning scripts when
    image data is wanted.
    """

    record_initial_state = InitialStateRecorderCfg()
    record_post_step_states = PostStepStatesRecorderCfg()
    record_pre_step_actions = PreStepActionsRecorderCfg()
    record_post_step_processed_actions = PostStepProcessedActionsRecorderCfg()
    record_episode_id = EpisodeIdentityRecorderCfg()


def with_trajectory_recorder_terms(recorder_cfg: RecorderManagerBaseCfg) -> RecorderManagerBaseCfg:
    """Return ``recorder_cfg`` extended with the per-step trajectory recorder terms.

    The metric terms already on ``recorder_cfg`` are preserved, because metrics are computed by
    reading their terms back out of the same exported dataset.

    Args:
        recorder_cfg: The composed recorder manager config to extend.
    """
    return combine_configclass_instances(
        "TrajectoryRecorderManagerCfg",
        recorder_cfg,
        TrajectoryRecorderTermsCfg(),
        bases=(RecorderManagerBaseCfg,),
    )
