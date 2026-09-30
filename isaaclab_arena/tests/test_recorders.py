# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from types import SimpleNamespace

from isaaclab_arena.metrics.success_rate import SuccessRecorder, SuccessRecorderCfg
from isaaclab_arena.terms.recorders import (
    EpisodeIdentityRecorder,
    EpisodeIdentityRecorderCfg,
    make_trajectory_recorder_terms_cfg,
)

TRAJECTORY_TERM_NAMES = (
    "record_initial_state",
    "record_post_step_states",
    "record_pre_step_actions",
    "record_post_step_processed_actions",
    "record_episode_id",
    "record_gripper_state",
)


def test_trajectory_terms_cfg_has_all_terms():
    terms_cfg = make_trajectory_recorder_terms_cfg()

    for term_name in TRAJECTORY_TERM_NAMES:
        assert getattr(terms_cfg, term_name, None) is not None, f"missing trajectory term {term_name}"


def test_single_frame_transformer_adds_one_end_effector_poses_term():
    terms_cfg = make_trajectory_recorder_terms_cfg(frame_transformer_names=("left_ee_frame",), asset_name="left_arm")

    assert terms_cfg.record_end_effector_poses_0.frame_transformer_name == "left_ee_frame"
    assert terms_cfg.record_end_effector_poses_0.asset_name == "left_arm"
    assert not hasattr(terms_cfg, "record_end_effector_poses_1")


def test_multiple_frame_transformers_add_one_end_effector_poses_term_each():
    terms_cfg = make_trajectory_recorder_terms_cfg(
        frame_transformer_names=("left_ee_frame", "right_ee_frame"), asset_name="robot"
    )

    assert terms_cfg.record_end_effector_poses_0.frame_transformer_name == "left_ee_frame"
    assert terms_cfg.record_end_effector_poses_1.frame_transformer_name == "right_ee_frame"
    assert terms_cfg.record_end_effector_poses_0.asset_name == "robot"
    assert terms_cfg.record_end_effector_poses_1.asset_name == "robot"


def test_first_finished_episodes_have_trajectory_identity():
    """The first completion callback stamps each demo with its environment ID and episode index within that environment."""
    env = SimpleNamespace(num_envs=3, device="cpu", get_episode_index=lambda env_id: 0)
    recorder = EpisodeIdentityRecorder(EpisodeIdentityRecorderCfg(), env)
    key, identity = recorder.record_pre_reset([2, 0])
    assert key == "episode_id"
    assert identity["env_id"].tolist() == [2, 0]
    assert identity["episode_in_env"].tolist() == [0, 0]


def test_first_partial_completion_records_success():
    """Success is recorded when the first completed batch contains only one environment."""
    success = torch.tensor([False, True, True])
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        termination_manager=SimpleNamespace(active_terms=["success"], get_term=lambda name: success),
    )
    recorder = SuccessRecorder(SuccessRecorderCfg(), env)
    name, values = recorder.record_pre_reset([2])
    assert name == "success"
    assert values.tolist() == [True]
    name, values = recorder.record_pre_reset([0, 1])
    assert name == "success"
    assert values.tolist() == [False, True]
