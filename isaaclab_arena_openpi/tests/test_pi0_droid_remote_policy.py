# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the openpi DROID remote policy.

These tests import the real ``openpi_client`` package (installed per
``README.md``) so we exercise the actual ``image_tools.resize_with_pad``,
the real ``WebsocketClientPolicy`` class structure, and the real
``websockets.exceptions`` types. The only things stubbed are the
network-touching pieces: ``WebsocketClientPolicy._wait_for_server``
(skip the connect handshake) and ``WebsocketClientPolicy.infer`` (return
a synthetic action chunk instead of doing a round-trip).

If ``openpi_client`` isn't installed, the tests fail at import — that's
the intended signal that the package's runtime dependency is missing.
"""

from __future__ import annotations

import numpy as np
import torch
import types

import pytest
import websockets.exceptions
from openpi_client import websocket_client_policy

from isaaclab_arena_openpi.policy.pi0_droid_config import MAX_RECONNECT_ATTEMPTS
from isaaclab_arena_openpi.policy.pi0_droid_remote_policy import Pi0DroidRemotePolicy, Pi0DroidRemotePolicyArgs


def _fake_env(num_envs: int = 1):
    """Minimal stand-in for the gym env passed to get_action."""
    return types.SimpleNamespace(unwrapped=types.SimpleNamespace(num_envs=num_envs))


def _fake_observation(num_envs: int = 1) -> dict:
    """Mimic the structure of the DROID gym observation dict."""
    return {
        "camera_obs": {
            "external_camera_rgb": torch.zeros((num_envs, 720, 1280, 3), dtype=torch.uint8),
            "wrist_camera_rgb": torch.zeros((num_envs, 720, 1280, 3), dtype=torch.uint8),
        },
        "policy": {
            "joint_pos": torch.zeros((num_envs, 7), dtype=torch.float32),
            "gripper_pos": torch.zeros((num_envs, 1), dtype=torch.float32),
        },
    }


def _synthetic_chunk() -> np.ndarray:
    """A horizon=15, action_dim=8 chunk shaped like pi05_droid_jointpos output."""
    actions = np.tile(
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32),
        (15, 1),
    )
    # Make the gripper values flip at indices 0 and 1 so binarisation is testable.
    actions[0, -1] = 0.2
    actions[1, -1] = 0.7
    return actions


def _patch_websocket_client(monkeypatch, infer_impl=None) -> None:
    """Block the real openpi WebsocketClientPolicy from touching the network.

    Patches ``_wait_for_server`` to skip the handshake and ``infer`` to
    return synthetic chunks. Both attributes are restored after each test
    by pytest's ``monkeypatch``.
    """
    monkeypatch.setattr(
        websocket_client_policy.WebsocketClientPolicy,
        "_wait_for_server",
        lambda self: (None, {}),
    )
    if infer_impl is None:
        infer_impl = lambda self, request: {"actions": _synthetic_chunk()}  # noqa: E731
    monkeypatch.setattr(
        websocket_client_policy.WebsocketClientPolicy,
        "infer",
        infer_impl,
    )


@pytest.fixture
def make_policy(monkeypatch):
    """Factory: build a Pi0DroidRemotePolicy with the network layer patched."""
    _patch_websocket_client(monkeypatch)

    def _factory(policy_variant: str = "pi05"):
        return Pi0DroidRemotePolicy(Pi0DroidRemotePolicyArgs(policy_variant=policy_variant, policy_device="cpu"))

    return _factory


def test_unknown_variant_raises(monkeypatch):
    _patch_websocket_client(monkeypatch)
    with pytest.raises(ValueError, match="Unknown policy_variant"):
        Pi0DroidRemotePolicy(Pi0DroidRemotePolicyArgs(policy_variant="not_a_real_model"))


def test_pack_request_uses_pi0_wire_keys(make_policy):
    policy = make_policy()
    policy.set_task_description("pick up the block")
    droid_obs = policy._extract_droid_observation(_fake_observation())
    server_request = policy._pack_pi0_request(droid_obs, "pick up the block")

    assert set(server_request.keys()) == {
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
        "observation/joint_position",
        "observation/gripper_position",
        "prompt",
    }
    # Real image_tools.resize_with_pad: actually produces (224, 224, 3) uint8.
    assert server_request["observation/exterior_image_1_left"].shape == (224, 224, 3)
    assert server_request["observation/exterior_image_1_left"].dtype == np.uint8
    assert server_request["observation/wrist_image_left"].shape == (224, 224, 3)
    assert server_request["observation/wrist_image_left"].dtype == np.uint8
    assert server_request["observation/joint_position"].shape == (7,)
    assert server_request["observation/gripper_position"].shape == (1,)
    assert server_request["prompt"] == "pick up the block"


def test_extract_observation_reads_droid_keys(make_policy):
    policy = make_policy()
    droid_obs = policy._extract_droid_observation(_fake_observation())

    assert droid_obs["exterior_image"].shape == (720, 1280, 3)
    assert droid_obs["wrist_image"].shape == (720, 1280, 3)
    assert droid_obs["joint_position"].shape == (7,)
    assert droid_obs["gripper_position"].shape == (1,)


def test_pi05_open_loop_horizon_is_15(make_policy):
    policy = make_policy(policy_variant="pi05")
    assert policy.open_loop_horizon == 15


def test_pi0_fast_open_loop_horizon_is_10(make_policy):
    policy = make_policy(policy_variant="pi0_fast")
    assert policy.open_loop_horizon == 10


def test_get_action_caches_chunk_and_advances_index(make_policy):
    policy = make_policy()
    policy.set_task_description("pick up the block")
    env = _fake_env(num_envs=1)
    obs = _fake_observation()

    first_action = policy.get_action(env, obs)
    second_action = policy.get_action(env, obs)

    assert first_action.shape == (1, 8) and second_action.shape == (1, 8)
    assert first_action.dtype == torch.float32

    # Server returned gripper values 0.2 (row 0) and 0.7 (row 1). The
    # policy passes them through unchanged — arena's BinaryJointPositionZeroToOneAction
    # applies the >0.5 threshold itself. Confirms the cache is advancing
    # rather than refetching every step.
    assert first_action[0, -1].item() == pytest.approx(0.2)
    assert second_action[0, -1].item() == pytest.approx(0.7)


def test_get_action_rejects_multi_env(make_policy):
    policy = make_policy()
    policy.set_task_description("pick up the block")
    with pytest.raises(AssertionError, match="num_envs=1"):
        policy.get_action(_fake_env(num_envs=2), _fake_observation())


def test_reset_drops_cached_chunk(make_policy):
    policy = make_policy()
    policy.set_task_description("pick up the block")
    env = _fake_env()

    policy.get_action(env, _fake_observation())
    assert policy._cached_action_chunk is not None and policy._next_chunk_step == 1

    policy.reset()
    assert policy._cached_action_chunk is None and policy._next_chunk_step == 0


def test_short_server_response_raises(monkeypatch):
    _patch_websocket_client(
        monkeypatch,
        infer_impl=lambda self, request: {"actions": np.zeros((5, 8), dtype=np.float32)},  # < open_loop_horizon=15
    )
    policy = Pi0DroidRemotePolicy(Pi0DroidRemotePolicyArgs(policy_device="cpu"))
    policy.set_task_description("pick up the block")
    with pytest.raises(AssertionError, match="open_loop_horizon"):
        policy._fetch_action_chunk(_fake_observation())


def test_call_server_with_retry_reconnects_on_drop(monkeypatch):
    """Drop the first connection; the second call should succeed and the cached chunk should be flushed."""
    call_count = {"n": 0}
    successful_response = {"actions": np.zeros((15, 8), dtype=np.float32)}

    def flaky_infer(self, request):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise websockets.exceptions.ConnectionClosedError(None, None)
        return successful_response

    _patch_websocket_client(monkeypatch, infer_impl=flaky_infer)
    policy = Pi0DroidRemotePolicy(Pi0DroidRemotePolicyArgs(policy_device="cpu"))
    policy.set_task_description("pick up the block")
    policy._cached_action_chunk = np.zeros((15, 8), dtype=np.float32)
    policy._next_chunk_step = 5

    response = policy._call_server_with_retry({"prompt": "x"})

    assert response is successful_response
    assert call_count["n"] == 2
    # Cache flushed so the next get_action re-queries with fresh obs.
    assert policy._cached_action_chunk is None
    assert policy._next_chunk_step == 0


def test_call_server_with_retry_gives_up_after_max_attempts(monkeypatch):
    def always_drops(self, request):
        raise websockets.exceptions.ConnectionClosedError(None, None)

    _patch_websocket_client(monkeypatch, infer_impl=always_drops)
    policy = Pi0DroidRemotePolicy(Pi0DroidRemotePolicyArgs(policy_device="cpu"))
    policy.set_task_description("pick up the block")

    with pytest.raises(websockets.exceptions.ConnectionClosedError):
        policy._call_server_with_retry({"prompt": "x"})

    assert MAX_RECONNECT_ATTEMPTS >= 2
