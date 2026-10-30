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

from isaaclab_arena.assets.registries import PolicyRegistry
from isaaclab_arena.policy.websocket_client import WebsocketClientPolicy
from isaaclab_arena_openpi.policy.droid_adapter import Pi0DroidAdapter
from isaaclab_arena_openpi.policy.pi0_remote_config import Pi0RemotePolicyCfg
from isaaclab_arena_openpi.policy.pi0_remote_policy import Pi0RemotePolicy

CONTROL_STEP_DT = 1.0 / 15.0
OPEN_LOOP_HORIZON = 15


def _fake_env(num_envs: int = 1):
    return types.SimpleNamespace(unwrapped=types.SimpleNamespace(num_envs=num_envs, step_dt=CONTROL_STEP_DT))


def _fake_observation(num_envs: int = 1) -> dict:
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
        infer_impl = lambda self, request: {"actions": _synthetic_chunk()}  # noqa: E731
    monkeypatch.setattr(
        WebsocketClientPolicy,
        "infer",
        infer_impl,
    )


@pytest.fixture
def make_policy(monkeypatch):
    _patch_websocket_client(monkeypatch)

    def _factory(policy_variant: str = "pi05"):
        return Pi0RemotePolicy(
            Pi0RemotePolicyCfg(policy_variant=policy_variant, policy_device="cpu"),
        )

    return _factory


def test_droid_adapter_uses_pi0_wire_keys():
    """The wire-format contract between Pi0DroidAdapter and the openpi droid server."""
    adapter = Pi0DroidAdapter()
    extracted = adapter.extract(_fake_observation(), env_id=0)
    server_request = adapter.pack_request(extracted, "pick up the block")

    assert set(server_request.keys()) == {
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
        "observation/joint_position",
        "observation/gripper_position",
        "prompt",
    }
    assert server_request["observation/exterior_image_1_left"].shape == (224, 224, 3)
    assert server_request["observation/exterior_image_1_left"].dtype == np.uint8
    assert server_request["observation/wrist_image_left"].shape == (224, 224, 3)
    assert server_request["observation/wrist_image_left"].dtype == np.uint8
    assert server_request["observation/joint_position"].shape == (7,)
    assert server_request["observation/gripper_position"].shape == (1,)
    assert server_request["prompt"] == "pick up the block"


def test_registration_builds_typed_config_and_resolves_adapter(monkeypatch):
    """The registered typed config retains embodiment-adapter selection."""
    _patch_websocket_client(monkeypatch)
    assert PolicyRegistry().get_policy_cfg_type(Pi0RemotePolicy) is Pi0RemotePolicyCfg
    policy = Pi0RemotePolicy(
        Pi0RemotePolicyCfg(
            policy_variant="pi05",
            policy_device="cpu",
            remote_host="localhost",
            remote_port=8000,
            openpi_embodiment_adapter="droid",
        )
    )
    assert isinstance(policy._adapter, Pi0DroidAdapter)
    assert policy._open_loop_horizon == 15


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
        return {"actions": _synthetic_chunk()}

    _patch_websocket_client(monkeypatch, infer_impl=counting_infer)
    policy = Pi0RemotePolicy(Pi0RemotePolicyCfg(policy_device="cpu"))
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
    policy = Pi0RemotePolicy(Pi0RemotePolicyCfg(policy_device="cpu"))
    policy.set_task_description("pick up the block")
    env = _fake_env(num_envs=3)
    obs = _fake_observation(num_envs=3)

    policy.get_action(env, obs)  # populates caches for all 3 envs

    policy.reset(env_ids=torch.tensor([0, 2]))

    # -1 marks an env without a chunk; env 1 keeps replaying from row 1.
    assert policy._scheduler.current_action_index.tolist() == [-1, 1, -1]


def test_call_server_with_retry_reconnects_on_drop(monkeypatch):
    """Drop the first connection; second call succeeds and cache is flushed."""
    call_count = {"n": 0}
    successful_response = {"actions": np.zeros((15, 8), dtype=np.float32)}

    def flaky_infer(self, request):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise websockets.exceptions.ConnectionClosedError(None, None)
        return successful_response

    _patch_websocket_client(monkeypatch, infer_impl=flaky_infer)
    policy = Pi0RemotePolicy(Pi0RemotePolicyCfg(policy_device="cpu"))
    policy.set_task_description("pick up the block")
    env = _fake_env(num_envs=2)
    obs = _fake_observation(num_envs=2)
    policy.get_action(env, obs)  # both envs fetch; env 0's first attempt drops the connection
    assert call_count["n"] == 3
    assert policy._scheduler.current_action_index.tolist() == [1, 1]

    # The reconnect flushes every env's chunk on the next step, so both fetch again with a fresh observation.
    policy.get_action(env, obs)

    assert call_count["n"] == 5
    assert policy._scheduler.current_action_index.tolist() == [1, 1]


def test_call_server_with_retry_gives_up_after_max_attempts(monkeypatch):
    def always_drops(self, request):
        raise websockets.exceptions.ConnectionClosedError(None, None)

    _patch_websocket_client(monkeypatch, infer_impl=always_drops)
    policy = Pi0RemotePolicy(Pi0RemotePolicyCfg(policy_device="cpu"))
    policy.set_task_description("pick up the block")

    with pytest.raises(websockets.exceptions.ConnectionClosedError):
        policy._call_server_with_retry({"prompt": "x"})


def test_pi0_remote_policy_declares_the_adapter_image_size():
    """Report the 224x224 box the DROID adapter resizes camera frames into, for camera_resolution_mode policy_default."""
    assert Pi0RemotePolicy.default_image_size(Pi0RemotePolicyCfg(openpi_embodiment_adapter="droid")) == (224, 224)
    assert Pi0DroidAdapter.target_image_size == (224, 224)


def _hold_action(num_envs: int) -> torch.Tensor:
    return torch.full((num_envs, 8), -1.0)


def test_latency_holds_then_replays_the_chunk_requested_two_steps_earlier(monkeypatch):
    """With 85 ms at 15 Hz the chunk requested on step 0 is applied from step 2, and the arm holds until then."""
    from isaaclab_arena.policy import remote_policy_base

    call_count = {"n": 0}

    def counting_infer(self, request):
        call_count["n"] += 1
        return {"actions": _synthetic_chunk()}

    _patch_websocket_client(monkeypatch, infer_impl=counting_infer)
    monkeypatch.setattr(remote_policy_base, "hold_action_from_action_manager", lambda env: _hold_action(1))
    policy = Pi0RemotePolicy(Pi0RemotePolicyCfg(policy_device="cpu", inference_latency_ms=85.0))
    policy.set_task_description("pick up the block")
    env = _fake_env()
    obs = _fake_observation()

    actions = [policy.get_action(env, obs)[0, -1].item() for _ in range(OPEN_LOOP_HORIZON + 3)]

    assert policy._scheduler.latency_steps == 2
    assert actions[:2] == [-1.0, -1.0]  # holding while the first request is in flight
    assert actions[2] == pytest.approx(0.2)  # row 0 of the chunk requested on step 0
    assert actions[3] == pytest.approx(0.7)  # row 1
    assert actions[OPEN_LOOP_HORIZON + 1] == pytest.approx(0.8)  # row 14, last action of the first chunk
    # The second request was issued on step 15, so its chunk is usable on step 17 without a hold.
    assert actions[OPEN_LOOP_HORIZON + 2] == pytest.approx(0.2)
    assert call_count["n"] == 2
    assert policy._scheduler.hold_steps.tolist() == [2]
    assert policy._scheduler.stall_steps.tolist() == [0]


def test_latency_reset_discards_the_in_flight_request_and_holds_again(monkeypatch):
    """A reset env drops its pending chunk, requests a new one on its next step and holds for the latency."""
    from isaaclab_arena.policy import remote_policy_base

    _patch_websocket_client(monkeypatch)
    monkeypatch.setattr(remote_policy_base, "hold_action_from_action_manager", lambda env: _hold_action(2))
    policy = Pi0RemotePolicy(Pi0RemotePolicyCfg(policy_device="cpu", inference_latency_ms=70.0))
    policy.set_task_description("pick up the block")
    env = _fake_env(num_envs=2)
    obs = _fake_observation(num_envs=2)

    for _ in range(3):
        policy.get_action(env, obs)  # steps 0-2: both envs hold twice, then replay row 0
    policy.reset(env_ids=torch.tensor([1]))
    action = policy.get_action(env, obs)  # step 3: env 0 replays row 1, env 1 holds again

    assert action[0, -1].item() == pytest.approx(0.7)
    assert action[1, -1].item() == -1.0
    assert policy._scheduler.requests_issued == 3


def test_zero_latency_policy_never_needs_a_hold_action(make_policy):
    """Without latency the chunk is fetched on the step it is needed and no hold action is read from the env."""
    policy = make_policy()
    policy.set_task_description("pick up the block")

    action = policy.get_action(_fake_env(), _fake_observation())

    assert action[0, -1].item() == pytest.approx(0.2)
    assert policy._scheduler.latency_steps == 0
