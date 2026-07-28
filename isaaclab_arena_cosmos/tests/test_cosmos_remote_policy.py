# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch
import types

import pytest
import websockets.exceptions

from isaaclab_arena.assets.registries import PolicyRegistry
from isaaclab_arena_cosmos.policy.cosmos_remote_config import CosmosRemotePolicyCfg
from isaaclab_arena_cosmos.policy.cosmos_remote_policy import CosmosRemotePolicy
from isaaclab_arena_cosmos.policy.droid_adapter import CosmosDroidAdapter
from isaaclab_arena_cosmos.policy.websocket_client import WebsocketClientPolicy

_DEFAULT_HORIZON = CosmosRemotePolicyCfg.open_loop_horizon


def _fake_env(num_envs: int = 1):
    return types.SimpleNamespace(unwrapped=types.SimpleNamespace(num_envs=num_envs))


def _fake_observation(num_envs: int = 1) -> dict:
    return {
        "camera_obs": {
            "wrist_camera_rgb": torch.zeros((num_envs, 720, 1280, 3), dtype=torch.uint8),
            "external_camera_rgb": torch.zeros((num_envs, 720, 1280, 3), dtype=torch.uint8),
            "external_camera_2_rgb": torch.zeros((num_envs, 720, 1280, 3), dtype=torch.uint8),
        },
        "policy": {
            "joint_pos": torch.zeros((num_envs, 7), dtype=torch.float32),
            "gripper_pos": torch.zeros((num_envs, 1), dtype=torch.float32),
        },
    }


def _synthetic_chunk(num_steps: int = 32) -> np.ndarray:
    """A (num_steps, action_dim=8) chunk shaped like the RoboLab joint_pos output."""
    actions = np.tile(
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32),
        (num_steps, 1),
    )
    # Distinguish row 0 from row 1 so chunk advancement is observable.
    actions[0, -1] = 0.2
    actions[1, -1] = 0.7
    return actions


def _patch_websocket_client(monkeypatch, infer_impl=None) -> None:
    monkeypatch.setattr(
        WebsocketClientPolicy,
        "_wait_for_server",
        lambda self: (None, {}),
    )
    if infer_impl is None:
        infer_impl = lambda self, request: {"action": _synthetic_chunk()}  # noqa: E731
    monkeypatch.setattr(
        WebsocketClientPolicy,
        "infer",
        infer_impl,
    )


@pytest.fixture
def make_policy(monkeypatch):
    _patch_websocket_client(monkeypatch)

    def _factory(**cfg_overrides):
        cfg_overrides.setdefault("policy_device", "cpu")
        return CosmosRemotePolicy(CosmosRemotePolicyCfg(**cfg_overrides))

    return _factory


def test_droid_adapter_uses_cosmos_wire_keys():
    """The wire-format contract between CosmosDroidAdapter and the RoboLab server."""
    adapter = CosmosDroidAdapter()
    extracted = adapter.extract(_fake_observation(), env_id=0)
    server_request = adapter.pack_request(extracted, "pick up the banana and place it in the bowl")

    assert set(server_request.keys()) == {
        "observation/wrist_image_left",
        "observation/exterior_image_1_left",
        "observation/exterior_image_2_left",
        "observation/joint_position",
        "observation/gripper_position",
        "prompt",
    }
    for image_key in (
        "observation/wrist_image_left",
        "observation/exterior_image_1_left",
        "observation/exterior_image_2_left",
    ):
        assert server_request[image_key].shape == (720, 1280, 3)
        assert server_request[image_key].dtype == np.uint8
    assert server_request["observation/joint_position"].shape == (7,)
    assert server_request["observation/gripper_position"].shape == (1,)
    assert server_request["prompt"] == "pick up the banana and place it in the bowl"


def test_registration_builds_typed_config_and_resolves_adapter(monkeypatch):
    """The registered typed config retains embodiment-adapter selection."""
    _patch_websocket_client(monkeypatch)
    assert PolicyRegistry().get_policy_cfg_type(CosmosRemotePolicy) is CosmosRemotePolicyCfg
    policy = CosmosRemotePolicy(
        CosmosRemotePolicyCfg(
            policy_device="cpu",
            remote_host="localhost",
            remote_port=8000,
            cosmos_embodiment_adapter="droid",
        )
    )
    assert isinstance(policy._cosmos_embodiment_adapter, CosmosDroidAdapter)


def test_unknown_embodiment_adapter_raises(monkeypatch):
    _patch_websocket_client(monkeypatch)
    with pytest.raises(ValueError, match="cosmos_embodiment_adapter"):
        CosmosRemotePolicy(CosmosRemotePolicyCfg(policy_device="cpu", cosmos_embodiment_adapter="nope"))


def test_close_is_idempotent(make_policy):
    """close() drops the client and tolerates a second call."""
    policy = make_policy()
    assert policy._websocket_client is not None
    policy.close()
    assert policy._websocket_client is None
    policy.close()  # second call must not raise


def test_get_action_caches_chunk_and_advances_index(make_policy):
    """Two consecutive get_action calls replay rows 0 and 1 from one fetched chunk."""
    policy = make_policy()
    policy.set_task_description("pick up the banana and place it in the bowl")
    env = _fake_env(num_envs=1)
    obs = _fake_observation()

    first_action = policy.get_action(env, obs)
    second_action = policy.get_action(env, obs)

    assert first_action.shape == (1, 8) and second_action.shape == (1, 8)
    assert first_action.dtype == torch.float32
    assert first_action[0, -1].item() == pytest.approx(0.2)
    assert second_action[0, -1].item() == pytest.approx(0.7)


def test_get_action_parallel_envs_loops_per_env(monkeypatch):
    """num_envs>1: one infer per env per chunk refill, batched into (num_envs, action_dim)."""
    call_count = {"n": 0}

    def counting_infer(self, request):
        call_count["n"] += 1
        return {"action": _synthetic_chunk()}

    _patch_websocket_client(monkeypatch, infer_impl=counting_infer)
    policy = CosmosRemotePolicy(CosmosRemotePolicyCfg(policy_device="cpu"))
    policy.set_task_description("pick up the banana and place it in the bowl")

    num_envs = 3
    env = _fake_env(num_envs=num_envs)
    obs = _fake_observation(num_envs=num_envs)

    first_action = policy.get_action(env, obs)
    second_action = policy.get_action(env, obs)

    # One infer call per env on the first get_action (cache miss); none on the
    # second (chunk row 1 is still cached for each env).
    assert call_count["n"] == num_envs
    assert first_action.shape == (num_envs, 8)
    assert second_action.shape == (num_envs, 8)
    for env_id in range(num_envs):
        assert first_action[env_id, -1].item() == pytest.approx(0.2)
        assert second_action[env_id, -1].item() == pytest.approx(0.7)


def test_short_server_chunk_is_rejected(monkeypatch):
    """A server chunk shorter than open_loop_horizon fails loudly rather than under-replaying."""

    def short_infer(self, request):
        return {"action": _synthetic_chunk(num_steps=_DEFAULT_HORIZON - 1)}

    _patch_websocket_client(monkeypatch, infer_impl=short_infer)
    policy = CosmosRemotePolicy(CosmosRemotePolicyCfg(policy_device="cpu"))
    policy.set_task_description("pick up the banana and place it in the bowl")

    with pytest.raises(AssertionError, match="open_loop_horizon"):
        policy.get_action(_fake_env(), _fake_observation())


def test_reset_honors_env_ids(monkeypatch):
    """reset(env_ids) clears only those envs' caches; others keep replaying."""
    _patch_websocket_client(monkeypatch)
    policy = CosmosRemotePolicy(CosmosRemotePolicyCfg(policy_device="cpu"))
    policy.set_task_description("pick up the banana and place it in the bowl")
    env = _fake_env(num_envs=3)
    obs = _fake_observation(num_envs=3)

    policy.get_action(env, obs)  # populates caches for all 3 envs

    policy.reset(env_ids=torch.tensor([0, 2]))

    assert policy._cached_action_chunks[0] is None
    assert policy._cached_action_chunks[1] is not None  # untouched
    assert policy._cached_action_chunks[2] is None
    assert policy._next_chunk_steps == [0, 1, 0]


def test_call_server_with_retry_reconnects_on_drop(monkeypatch):
    """Drop the first connection; second call succeeds and cache is flushed."""
    call_count = {"n": 0}
    successful_response = {"action": np.zeros((32, 8), dtype=np.float32)}

    def flaky_infer(self, request):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise websockets.exceptions.ConnectionClosedError(None, None)
        return successful_response

    _patch_websocket_client(monkeypatch, infer_impl=flaky_infer)
    policy = CosmosRemotePolicy(CosmosRemotePolicyCfg(policy_device="cpu"))
    policy.set_task_description("pick up the banana and place it in the bowl")
    policy._cached_action_chunks = [np.zeros((32, 8), dtype=np.float32)]
    policy._next_chunk_steps = [5]

    response = policy._call_server_with_retry({"prompt": "x"})

    assert response is successful_response
    assert call_count["n"] == 2
    assert policy._cached_action_chunks == [None]
    assert policy._next_chunk_steps == [0]


def test_call_server_with_retry_gives_up_after_max_attempts(monkeypatch):
    def always_drops(self, request):
        raise websockets.exceptions.ConnectionClosedError(None, None)

    _patch_websocket_client(monkeypatch, infer_impl=always_drops)
    policy = CosmosRemotePolicy(CosmosRemotePolicyCfg(policy_device="cpu"))
    policy.set_task_description("pick up the banana and place it in the bowl")

    with pytest.raises(websockets.exceptions.ConnectionClosedError):
        policy._call_server_with_retry({"prompt": "x"})
