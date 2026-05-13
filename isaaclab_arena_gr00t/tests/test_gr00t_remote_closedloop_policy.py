# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit test for Gr00tRemoteClosedloopPolicy.

Exercises the real obs/action translation pipeline and mocks only ``Gr00tPolicyClient`` -- the wire
boundary to a remote GR00T server -- so no server is required.

Verifies:
- the obs dict sent to the server has the expected language/video/state structure;
- the action dict returned by the server is converted to a tensor of the
  expected (num_envs, action_chunk_length, action_dim) shape;
- ``reset`` propagates to the client;
- a failing ping during construction raises ``ConnectionError``.
"""
# TODO(xinjieyao, 2026-04-27):will be moved to base container test when removing deprecated local policy

from __future__ import annotations

import numpy as np
import torch
import yaml
from typing import Any

import pytest

from isaaclab_arena_gr00t.tests.utils.constants import TestConstants as Gr00tTestConstants

NUM_ENVS = 2
ORIGINAL_HEIGHT = 480
ORIGINAL_WIDTH = 640
NUM_CHANNELS = 3
NUM_SIM_JOINTS = 43  # G1 43-DoF
ACTION_HORIZON = 50
ACTION_CHUNK_LENGTH = 50
EXPECTED_ACTION_DIM = 50  # 43 joints + 3 nav + 1 base height + 3 torso rpy

# Joint group sizes from gr00t_43dof_joint_space.yaml (must match the YAML).
POLICY_GROUP_SIZES = {
    "left_leg": 6,
    "right_leg": 6,
    "waist": 3,
    "left_arm": 7,
    "left_hand": 7,
    "right_arm": 7,
    "right_hand": 7,
}


# ----------------------------- fixtures ------------------------------ #


@pytest.fixture
def policy_config_yaml(tmp_path):
    """Write a copy of the g1 locomanip test config with model_path set to a
    HuggingFace repo id so ``Gr00tClosedloopPolicyConfig.__post_init__`` passes
    without a real checkpoint on disk (the remote policy never loads it)."""
    src = Gr00tTestConstants.test_data_dir + "/test_g1_locomanip_lerobot/test_g1_locomanip_gr00t_closedloop_config.yaml"
    with open(src) as f:
        cfg = yaml.safe_load(f)
    cfg["model_path"] = "nvidia/GR00T-N1.6-3B"  # HF id passes _is_huggingface_model_id
    dst = tmp_path / "remote_closedloop_config.yaml"
    with open(dst, "w") as f:
        yaml.dump(cfg, f)
    return str(dst)


@pytest.fixture
def synthetic_observation():
    """Env-shaped torch obs: one ego camera + 43 joint positions per env."""
    rgb = torch.randint(0, 255, (NUM_ENVS, ORIGINAL_HEIGHT, ORIGINAL_WIDTH, NUM_CHANNELS), dtype=torch.uint8)
    joint_pos = torch.randn(NUM_ENVS, NUM_SIM_JOINTS, dtype=torch.float32)
    return {
        "camera_obs": {"robot_head_cam_rgb": rgb},
        "policy": {"robot_joint_pos": joint_pos},
    }


def _make_action_response(num_envs: int, horizon: int) -> dict[str, np.ndarray]:
    """Synthetic GR00T-server response in the format ``build_gr00t_action_np`` expects."""
    response: dict[str, np.ndarray] = {}
    for group, size in POLICY_GROUP_SIZES.items():
        response[group] = np.random.randn(num_envs, horizon, size).astype(np.float32)
    response["navigate_command"] = np.random.randn(num_envs, horizon, 3).astype(np.float32)
    response["base_height_command"] = np.random.randn(num_envs, horizon, 1).astype(np.float32)
    return response


class _FakePolicyClient:
    """Minimal stand-in for ``gr00t.policy.server_client.PolicyClient``."""

    def __init__(self, *args, ping_ok: bool = True, **kwargs):
        self.init_kwargs = kwargs
        self._ping_ok = ping_ok
        self.last_observation: dict[str, Any] | None = None
        self.get_action_calls = 0
        self.reset_called = False

    def ping(self) -> bool:
        return self._ping_ok

    def get_action(self, observation: dict[str, Any]):
        self.last_observation = observation
        self.get_action_calls += 1
        # Match the real PolicyClient return signature: (action_dict, latency_or_meta).
        return _make_action_response(NUM_ENVS, ACTION_HORIZON), None

    def reset(self):
        self.reset_called = True


@pytest.fixture
def fake_client_factory(monkeypatch):
    """Patch ``Gr00tPolicyClient`` in the policy module's namespace and return
    a factory that gives access to the most recently constructed fake."""
    from isaaclab_arena_gr00t.policy import gr00t_remote_closedloop_policy as mod

    created: list[_FakePolicyClient] = []

    def factory(ping_ok: bool = True):
        def _ctor(*args, **kwargs):
            client = _FakePolicyClient(*args, ping_ok=ping_ok, **kwargs)
            created.append(client)
            return client

        monkeypatch.setattr(mod, "Gr00tPolicyClient", _ctor)
        return created

    return factory


def _build_policy(policy_config_yaml: str):
    from isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy import (
        Gr00tRemoteClosedloopPolicy,
        Gr00tRemoteClosedloopPolicyArgs,
    )

    args = Gr00tRemoteClosedloopPolicyArgs(
        policy_config_yaml_path=policy_config_yaml,
        policy_device="cpu",  # keep the test portable; remote policy does no GPU compute
        num_envs=NUM_ENVS,
        remote_host="unused",
        remote_port=0,
        remote_api_token=None,
    )
    return Gr00tRemoteClosedloopPolicy(args)


# ------------------------------- tests ------------------------------- #


def test_observation_sent_to_server_has_expected_structure(
    policy_config_yaml, synthetic_observation, fake_client_factory
):
    """The dict passed to ``PolicyClient.get_action`` matches GR00T's expected
    language/video/state layout for the g1 sim WBC modality config."""
    clients = fake_client_factory(ping_ok=True)
    policy = _build_policy(policy_config_yaml)
    policy.set_task_description("pick up the brown box")

    policy._get_action_chunk(synthetic_observation, ["robot_head_cam_rgb"])

    assert len(clients) == 1
    sent = clients[0].last_observation
    assert sent is not None and clients[0].get_action_calls == 1

    # language: one [task] entry per env
    assert "language" in sent
    lang = sent["language"]["annotation.human.task_description"]
    assert lang == [["pick up the brown box"]] * NUM_ENVS

    # video: one ego view, shaped (N, 1, H, W, C) and uint8
    assert "video" in sent and "ego_view" in sent["video"]
    ego = sent["video"]["ego_view"]
    assert ego.shape == (NUM_ENVS, 1, ORIGINAL_HEIGHT, ORIGINAL_WIDTH, NUM_CHANNELS)
    assert ego.dtype == np.uint8

    # state: one (N, 1, dof_in_group) entry per modality state key
    expected_state_keys = {"left_arm", "right_arm", "left_hand", "right_hand", "waist"}
    assert set(sent["state"].keys()) == expected_state_keys
    for key in expected_state_keys:
        arr = sent["state"][key]
        assert arr.ndim == 3
        assert arr.shape[0] == NUM_ENVS
        assert arr.shape[1] == 1
        assert arr.shape[2] == POLICY_GROUP_SIZES[key]


def test_action_response_is_translated_to_correct_tensor_shape(
    policy_config_yaml, synthetic_observation, fake_client_factory
):
    """Server's grouped joint dict is reassembled into a (N, chunk, action_dim) tensor."""
    fake_client_factory(ping_ok=True)
    policy = _build_policy(policy_config_yaml)
    policy.set_task_description("pick up the brown box")

    action = policy._get_action_chunk(synthetic_observation, ["robot_head_cam_rgb"])

    assert isinstance(action, torch.Tensor)
    # _get_action_chunk asserts action_tensor.shape[1] >= action_chunk_length, but
    # the full horizon is returned; pin both dims explicitly.
    assert action.shape == (NUM_ENVS, ACTION_HORIZON, EXPECTED_ACTION_DIM)
    assert action.device.type == "cpu"


def test_reset_propagates_to_client(policy_config_yaml, fake_client_factory):
    clients = fake_client_factory(ping_ok=True)
    policy = _build_policy(policy_config_yaml)
    assert clients[0].reset_called is False
    policy.reset()
    assert clients[0].reset_called is True


def test_construction_fails_when_server_unreachable(policy_config_yaml, fake_client_factory):
    fake_client_factory(ping_ok=False)
    with pytest.raises(ConnectionError):
        _build_policy(policy_config_yaml)
