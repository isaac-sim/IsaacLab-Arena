# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch
import types

import pytest
import websockets.exceptions
from openpi_client import websocket_client_policy

from isaaclab_arena_cosmos3.policy.cosmos3_remote_config import Cosmos3RemotePolicyArgs
from isaaclab_arena_cosmos3.policy.cosmos3_remote_policy import Cosmos3RemotePolicy
from isaaclab_arena_cosmos3.policy.droid_adapter import Cosmos3DroidAdapter


def _fake_env(num_envs: int = 1):
    return types.SimpleNamespace(unwrapped=types.SimpleNamespace(num_envs=num_envs, device=torch.device("cpu")))


def _fake_observation(num_envs: int = 1) -> dict:
    return {
        "camera_obs": {
            "external_camera_rgb": torch.zeros((num_envs, 720, 1280, 3), dtype=torch.uint8),
            "external_camera_2_rgb": torch.zeros((num_envs, 720, 1280, 3), dtype=torch.uint8),
            "wrist_camera_rgb": torch.zeros((num_envs, 720, 1280, 3), dtype=torch.uint8),
        },
        "policy": {
            "joint_pos": torch.zeros((num_envs, 7), dtype=torch.float32),
            "gripper_pos": torch.zeros((num_envs, 1), dtype=torch.float32),
        },
    }


def _synthetic_chunk() -> np.ndarray:
    """A horizon=32, action_dim=8 chunk shaped like cosmos3 droid output."""
    actions = np.tile(
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32),
        (32, 1),
    )
    # Distinguish row 0 from row 1 so chunk advancement is observable.
    actions[0, -1] = 0.2
    actions[1, -1] = 0.7
    return actions


def _patch_websocket_client(monkeypatch, infer_impl=None) -> None:
    monkeypatch.setattr(
        websocket_client_policy.WebsocketClientPolicy,
        "_wait_for_server",
        lambda self: (None, {}),
    )
    if infer_impl is None:
        infer_impl = lambda self, request: {"action": _synthetic_chunk()}  # noqa: E731
    monkeypatch.setattr(
        websocket_client_policy.WebsocketClientPolicy,
        "infer",
        infer_impl,
    )


@pytest.fixture
def make_policy(monkeypatch):
    _patch_websocket_client(monkeypatch)

    def _factory():
        return Cosmos3RemotePolicy(
            Cosmos3RemotePolicyArgs(),
            cosmos3_embodiment_adapter=Cosmos3DroidAdapter(),
        )

    return _factory


def test_droid_adapter_uses_cosmos3_wire_keys():
    """The wire-format contract between Cosmos3DroidAdapter and the cosmos3 server."""
    adapter = Cosmos3DroidAdapter()
    extracted = adapter.extract(_fake_observation(), env_id=0)
    server_request = adapter.pack_request(extracted, "pick up the block")

    assert set(server_request.keys()) == {
        "observation/image",
        "observation/joint_position",
        "observation/gripper_position",
        "prompt",
    }
    # Cosmos3 composes three cameras into a single 540×640 image.
    assert server_request["observation/image"].shape == (540, 640, 3)
    assert server_request["observation/image"].dtype == np.uint8
    assert server_request["observation/joint_position"].shape == (7,)
    assert server_request["observation/gripper_position"].shape == (1,)
    assert server_request["prompt"] == "pick up the block"


def test_droid_adapter_extract_reads_three_cameras():
    """extract() pulls left, right, and wrist cameras from the arena obs dict."""
    adapter = Cosmos3DroidAdapter()
    obs = _fake_observation(num_envs=2)

    # Set distinguishable values on the wrist camera.
    obs["camera_obs"]["wrist_camera_rgb"][0] = torch.full((720, 1280, 3), 42, dtype=torch.uint8)

    extracted = adapter.extract(obs, env_id=0)
    assert extracted.wrist_image.shape == (720, 1280, 3)
    assert (extracted.wrist_image == 42).all()
    assert extracted.left_image.shape == (720, 1280, 3)
    assert extracted.right_image.shape == (720, 1280, 3)
    assert extracted.joint_position.shape == (7,)
    assert extracted.gripper_position.shape == (1,)


def test_from_dict_resolves_adapter(monkeypatch):
    """eval_runner path: JSON dict -> Cosmos3RemotePolicy with adapter resolved from the dict."""
    _patch_websocket_client(monkeypatch)
    policy = Cosmos3RemotePolicy.from_dict({
        "remote_host": "localhost",
        "remote_port": 8000,
        "cosmos3_embodiment_adapter": "droid",
    })
    assert isinstance(policy._cosmos3_embodiment_adapter, Cosmos3DroidAdapter)
    assert policy._open_loop_horizon == 32


def test_from_dict_defaults_adapter_to_droid(monkeypatch):
    """When cosmos3_embodiment_adapter is omitted from the dict, it defaults to 'droid'."""
    _patch_websocket_client(monkeypatch)
    policy = Cosmos3RemotePolicy.from_dict({
        "remote_host": "localhost",
        "remote_port": 8000,
    })
    assert isinstance(policy._cosmos3_embodiment_adapter, Cosmos3DroidAdapter)


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
    policy.set_task_description("pick up the block")
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
    policy = Cosmos3RemotePolicy(Cosmos3RemotePolicyArgs(), cosmos3_embodiment_adapter=Cosmos3DroidAdapter())
    policy.set_task_description("pick up the block")

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


def test_reset_honors_env_ids(monkeypatch):
    """reset(env_ids) clears only those envs' caches; others keep replaying."""
    _patch_websocket_client(monkeypatch)
    policy = Cosmos3RemotePolicy(Cosmos3RemotePolicyArgs(), cosmos3_embodiment_adapter=Cosmos3DroidAdapter())
    policy.set_task_description("pick up the block")
    env = _fake_env(num_envs=3)
    obs = _fake_observation(num_envs=3)

    policy.get_action(env, obs)  # populates caches for all 3 envs

    policy.reset(env_ids=torch.tensor([0, 2]))

    assert policy._cached_action_chunks[0] is None
    assert policy._cached_action_chunks[1] is not None  # untouched
    assert policy._cached_action_chunks[2] is None
    assert policy._next_chunk_steps == [0, 1, 0]


def test_gripper_binarization(monkeypatch):
    """Gripper values > 0.5 are binarized to 1.0; others stay 0.0."""
    # Raw chunk with mixed gripper values.
    chunk = np.zeros((32, 8), dtype=np.float32)
    chunk[:, -1] = np.array([0.0, 0.3, 0.5, 0.51, 0.7, 1.0, 0.49, 0.8] + [0.0] * 24, dtype=np.float32)

    def custom_infer(self, request):
        return {"action": chunk.copy()}

    _patch_websocket_client(monkeypatch, infer_impl=custom_infer)
    policy = Cosmos3RemotePolicy(Cosmos3RemotePolicyArgs(), cosmos3_embodiment_adapter=Cosmos3DroidAdapter())
    policy.set_task_description("pick up the block")

    result = policy._fetch_action_chunk(_fake_observation(), env_id=0)

    # Check gripper dimension binarization.
    expected_gripper = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0] + [0.0] * 24, dtype=np.float32)
    np.testing.assert_array_equal(result[:, -1], expected_gripper)
    # Non-gripper dimensions unchanged.
    assert result[3, 0] == 0.0  # joint values preserved


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
    policy = Cosmos3RemotePolicy(Cosmos3RemotePolicyArgs(), cosmos3_embodiment_adapter=Cosmos3DroidAdapter())
    policy.set_task_description("pick up the block")
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
    policy = Cosmos3RemotePolicy(Cosmos3RemotePolicyArgs(), cosmos3_embodiment_adapter=Cosmos3DroidAdapter())
    policy.set_task_description("pick up the block")

    with pytest.raises(websockets.exceptions.ConnectionClosedError):
        policy._call_server_with_retry({"prompt": "x"})


def test_get_action_asserts_when_no_task_description(make_policy):
    """get_action raises AssertionError if set_task_description was never called."""
    policy = make_policy()
    env = _fake_env()
    obs = _fake_observation()

    with pytest.raises(AssertionError, match="language instruction"):
        policy.get_action(env, obs)


def test_fetch_action_chunk_asserts_on_shape_mismatch(monkeypatch):
    """_fetch_action_chunk asserts when response action has wrong action_dim."""

    def bad_shape_infer(self, request):
        return {"action": np.zeros((32, 6), dtype=np.float32)}  # expects 8

    _patch_websocket_client(monkeypatch, infer_impl=bad_shape_infer)
    policy = Cosmos3RemotePolicy(Cosmos3RemotePolicyArgs(), cosmos3_embodiment_adapter=Cosmos3DroidAdapter())
    policy.set_task_description("pick up the block")

    with pytest.raises(AssertionError, match="Expected action of shape"):
        policy._fetch_action_chunk(_fake_observation(), env_id=0)


def test_fetch_action_chunk_truncates_chunk_to_horizon(monkeypatch):
    """When the server returns more than horizon steps, only horizon are kept."""

    def oversized_infer(self, request):
        return {"action": np.ones((40, 8), dtype=np.float32)}

    _patch_websocket_client(monkeypatch, infer_impl=oversized_infer)
    policy = Cosmos3RemotePolicy(Cosmos3RemotePolicyArgs(), cosmos3_embodiment_adapter=Cosmos3DroidAdapter())
    policy.set_task_description("pick up the block")

    chunk = policy._fetch_action_chunk(_fake_observation(), env_id=0)
    assert chunk.shape == (32, 8)
