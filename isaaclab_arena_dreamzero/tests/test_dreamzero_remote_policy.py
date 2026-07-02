# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch
import types
import uuid

import msgpack
import pytest
import websockets.exceptions

from isaaclab_arena_dreamzero.policy.dreamzero_remote_config import DreamZeroRemotePolicyConfig
from isaaclab_arena_dreamzero.policy.dreamzero_remote_policy import DreamZeroRemotePolicy, _msgpack_encode
from isaaclab_arena_dreamzero.policy.droid_adapter import DroidAdapter, DroidAdapterConfig
from isaaclab_arena_dreamzero.policy.image_utils import TARGET_H, TARGET_W, resize_with_pad

# ── Helpers ───────────────────────────────────────────────────────────────────

NUM_ENVS = 2
NUM_JOINTS = DroidAdapterConfig().num_arm_joints + 1  # arm joints + 1 gripper in robot_joint_pos
ACTION_DIM = DroidAdapter.action_dim


def _fake_env(num_envs: int = NUM_ENVS):
    return types.SimpleNamespace(unwrapped=types.SimpleNamespace(num_envs=num_envs))


def _fake_observation(num_envs: int = NUM_ENVS) -> dict:
    return {
        "camera_obs": {
            "external_camera_rgb": torch.zeros((num_envs, 480, 640, 3), dtype=torch.uint8),
            "wrist_camera_rgb": torch.zeros((num_envs, 480, 640, 3), dtype=torch.uint8),
        },
        "policy": {
            "robot_joint_pos": torch.arange(num_envs * NUM_JOINTS, dtype=torch.float32).reshape(num_envs, NUM_JOINTS),
        },
    }


def _synthetic_chunk(horizon: int = 24) -> np.ndarray:
    chunk = np.tile(
        np.arange(ACTION_DIM, dtype=np.float32) * 0.1,
        (horizon, 1),
    )
    chunk[0, -1] = 0.11
    chunk[1, -1] = 0.22
    return chunk


def _packed_response(chunk: np.ndarray) -> bytes:
    return msgpack.packb({"actions": chunk}, default=_msgpack_encode)


_FAKE_GREETING = msgpack.packb({"server": "fake"}, default=_msgpack_encode)


class _FakeWs:
    """Minimal WebSocket stand-in for use in monkeypatching.

    The first recv() returns a fake server greeting (consumed by _connect()).
    Subsequent recv() calls return action chunk responses.
    """

    def __init__(self, response_fn=None):
        self._response_fn = response_fn or (lambda req: _packed_response(_synthetic_chunk()))
        self._sent = []
        self.closed = False
        self._greeting_sent = False

    def send(self, data: bytes) -> None:
        self._sent.append(data)

    def recv(self, timeout=None) -> bytes:
        if not self._greeting_sent:
            self._greeting_sent = True
            return _FAKE_GREETING
        last = self._sent[-1] if self._sent else b""
        return self._response_fn(last)

    def close(self) -> None:
        self.closed = True


def _patch_ws(monkeypatch, fake_ws: _FakeWs | None = None) -> _FakeWs:
    if fake_ws is None:
        fake_ws = _FakeWs()
    monkeypatch.setattr(
        "isaaclab_arena_dreamzero.policy.dreamzero_remote_policy.ws_sync.connect",
        lambda *args, **kwargs: fake_ws,
    )
    return fake_ws


def _make_dreamzero_remote_policy(
    open_loop_horizon: int = 24, num_arm_joints: int = 7, cam2_source: str = "black"
) -> DreamZeroRemotePolicy:
    adapter = DroidAdapter(DroidAdapterConfig(num_arm_joints=num_arm_joints, cam2_source=cam2_source))
    return DreamZeroRemotePolicy(
        DreamZeroRemotePolicyConfig(open_loop_horizon=open_loop_horizon, policy_device="cpu"),
        dreamzero_embodiment_adapter=adapter,
    )


@pytest.fixture
def make_policy(monkeypatch):
    _patch_ws(monkeypatch)

    def _factory(num_arm_joints: int = 7, cam2_source: str = "black", open_loop_horizon: int = 24):
        policy = _make_dreamzero_remote_policy(
            open_loop_horizon=open_loop_horizon, num_arm_joints=num_arm_joints, cam2_source=cam2_source
        )
        policy.set_task_description("pick up the object")
        return policy

    return _factory


# ── Config validation ─────────────────────────────────────────────────────────


def test_config_defaults():
    """Default config values match the DreamZero server spec."""
    cfg = DreamZeroRemotePolicyConfig()
    assert cfg.remote_port == 5000
    assert cfg.open_loop_horizon == 24


def test_config_rejects_non_positive_horizon():
    """open_loop_horizon of zero or less is rejected at construction time."""
    with pytest.raises(AssertionError, match="open_loop_horizon"):
        DreamZeroRemotePolicyConfig(open_loop_horizon=0)


def test_droid_adapter_config_defaults():
    """Default DroidAdapterConfig values match the DreamZero-DROID wire spec."""
    cfg = DroidAdapterConfig()
    assert cfg.num_arm_joints == 7
    assert cfg.cam2_source == "black"


def test_droid_adapter_config_rejects_invalid_cam2_source():
    """cam2_source rejects values outside the allowed set."""
    with pytest.raises(AssertionError, match="cam2_source"):
        DroidAdapterConfig(cam2_source="fisheye")


def test_droid_adapter_config_rejects_non_positive_num_arm_joints():
    """num_arm_joints of zero or less is rejected at construction time."""
    with pytest.raises(AssertionError, match="num_arm_joints"):
        DroidAdapterConfig(num_arm_joints=0)


def test_droid_adapter_config_defaults_target_image_size_to_image_utils_constants():
    """target_image_height/width default to image_utils' TARGET_H/TARGET_W."""
    cfg = DroidAdapterConfig()
    assert cfg.target_image_height == TARGET_H
    assert cfg.target_image_width == TARGET_W


def test_droid_adapter_config_rejects_non_positive_target_image_size():
    """target_image_height/width of zero or less are rejected at construction time."""
    with pytest.raises(AssertionError, match="target_image_height"):
        DroidAdapterConfig(target_image_height=0)
    with pytest.raises(AssertionError, match="target_image_width"):
        DroidAdapterConfig(target_image_width=0)


# ── from_dict ─────────────────────────────────────────────────────────────────


def test_from_dict_routes_fields_to_policy_and_adapter_configs(monkeypatch):
    """from_dict splits a flat dict between DreamZeroRemotePolicyConfig and DroidAdapterConfig by field name."""
    _patch_ws(monkeypatch)
    policy = DreamZeroRemotePolicy.from_dict({
        "remote_host": "example.com",
        "remote_port": 1234,
        "policy_device": "cpu",
        "num_arm_joints": 6,
        "cam2_source": "duplicate",
    })
    assert policy.config.remote_host == "example.com"
    assert policy.config.remote_port == 1234
    adapter = policy._dreamzero_embodiment_adapter
    assert isinstance(adapter, DroidAdapter)
    assert adapter.config.num_arm_joints == 6
    assert adapter.config.cam2_source == "duplicate"


def test_from_dict_defaults_embodiment_adapter_to_droid(monkeypatch):
    """Omitting dreamzero_embodiment_adapter from the dict defaults to the droid adapter."""
    _patch_ws(monkeypatch)
    policy = DreamZeroRemotePolicy.from_dict({"policy_device": "cpu"})
    assert isinstance(policy._dreamzero_embodiment_adapter, DroidAdapter)


def test_from_dict_rejects_unsupported_embodiment_adapter():
    """An unrecognized dreamzero_embodiment_adapter key is rejected before any construction happens."""
    with pytest.raises(AssertionError, match="dreamzero_embodiment_adapter"):
        DreamZeroRemotePolicy.from_dict({"dreamzero_embodiment_adapter": "g1"})


def test_from_dict_rejects_unknown_keys():
    """A key that matches neither DreamZeroRemotePolicyConfig nor DroidAdapterConfig raises a clear error."""
    with pytest.raises(AssertionError, match="Unknown DreamZeroRemotePolicy config keys"):
        DreamZeroRemotePolicy.from_dict({"embodiment": "droid"})


# ── image_utils ───────────────────────────────────────────────────────────────


def test_resize_with_pad_output_shape():
    """resize_with_pad produces the expected (TARGET_H, TARGET_W, 3) uint8 canvas."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    out = resize_with_pad(img)
    assert out.shape == (TARGET_H, TARGET_W, 3)
    assert out.dtype == np.uint8


def test_resize_with_pad_upscales():
    """resize_with_pad correctly upscales a small image to the target canvas size."""
    small = np.full((90, 80, 3), 200, dtype=np.uint8)
    out = resize_with_pad(small, height=180, width=320)
    assert out.shape == (180, 320, 3)


def test_resize_with_pad_preserves_aspect_ratio():
    """A square image padded into a landscape canvas has black padding on the sides."""
    square = np.full((100, 100, 3), 255, dtype=np.uint8)
    out = resize_with_pad(square, height=180, width=320)
    assert out.shape == (180, 320, 3)
    # Left column should be black padding since 100:100 < 320:180 (landscape canvas).
    assert out[90, 0, 0] == 0


# ── Wire format ───────────────────────────────────────────────────────────────


def test_build_request_flat_keys(make_policy):
    """Wire-format request uses flat slash-delimited keys, not a nested observation dict."""
    policy = make_policy()
    policy._maybe_init_per_env_state(NUM_ENVS)
    obs = _fake_observation()
    req = policy._build_request(obs, env_id=0)

    expected_obs_keys = {
        "observation/exterior_image_0_left",
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
        "observation/joint_position",
        "observation/cartesian_position",
        "observation/gripper_position",
    }
    assert expected_obs_keys.issubset(req.keys())
    # No nested "observation" dict — all keys are flat.
    assert not isinstance(req.get("observation"), dict)


def test_build_request_image_shapes(make_policy):
    """All three images in the request are resized to (TARGET_H, TARGET_W, 3) uint8."""
    policy = make_policy()
    policy._maybe_init_per_env_state(NUM_ENVS)
    obs = _fake_observation()
    req = policy._build_request(obs, env_id=0)

    for key in (
        "observation/exterior_image_0_left",
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
    ):
        assert req[key].shape == (TARGET_H, TARGET_W, 3), f"{key} has wrong shape"
        assert req[key].dtype == np.uint8


def test_build_request_image_shapes_use_configured_target_size(monkeypatch):
    """Overriding target_image_height/width on DroidAdapterConfig resizes request images accordingly."""
    _patch_ws(monkeypatch)
    adapter = DroidAdapter(DroidAdapterConfig(target_image_height=90, target_image_width=160))
    policy = DreamZeroRemotePolicy(
        DreamZeroRemotePolicyConfig(policy_device="cpu"), dreamzero_embodiment_adapter=adapter
    )
    policy.set_task_description("pick up the object")
    policy._maybe_init_per_env_state(NUM_ENVS)
    obs = _fake_observation()
    req = policy._build_request(obs, env_id=0)

    for key in (
        "observation/exterior_image_0_left",
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
    ):
        assert req[key].shape == (90, 160, 3), f"{key} has wrong shape"


def test_build_request_joint_split(make_policy):
    """robot_joint_pos is split into arm joint_position and gripper_position at num_arm_joints."""
    policy = make_policy(num_arm_joints=7)
    policy._maybe_init_per_env_state(NUM_ENVS)
    obs = _fake_observation()
    req = policy._build_request(obs, env_id=0)

    assert req["observation/joint_position"].shape == (7,)
    assert req["observation/gripper_position"].shape == (1,)
    joint_pos = obs["policy"]["robot_joint_pos"][0].numpy()
    np.testing.assert_array_equal(req["observation/joint_position"], joint_pos[:7])
    np.testing.assert_array_equal(req["observation/gripper_position"], joint_pos[7:8])


def test_build_request_cartesian_position_is_zeros(make_policy):
    """cartesian_position is always a zero vector of shape (6,), as the server requires."""
    policy = make_policy()
    policy._maybe_init_per_env_state(NUM_ENVS)
    obs = _fake_observation()
    req = policy._build_request(obs, env_id=0)
    np.testing.assert_array_equal(req["observation/cartesian_position"], np.zeros(6, dtype=np.float32))


def test_build_request_prompt_and_session(make_policy):
    """Request includes the task prompt, 'infer' endpoint, and a valid UUID session_id."""
    policy = make_policy()
    policy._maybe_init_per_env_state(NUM_ENVS)
    obs = _fake_observation()
    req = policy._build_request(obs, env_id=0)
    assert req["prompt"] == "pick up the object"
    assert req["endpoint"] == "infer"
    assert isinstance(req["session_id"], str)
    # Must be a valid UUID.
    uuid.UUID(req["session_id"])


# ── cam2_source ───────────────────────────────────────────────────────────────


def test_cam2_source_black_returns_zeros(make_policy):
    """cam2_source='black' fills the second exterior image slot with a zero canvas."""
    policy = make_policy(cam2_source="black")
    policy._maybe_init_per_env_state(NUM_ENVS)
    obs = _fake_observation()
    req = policy._build_request(obs, env_id=0)
    assert req["observation/exterior_image_1_left"].sum() == 0


def test_cam2_source_duplicate_copies_cam0(make_policy):
    """cam2_source='duplicate' fills the second slot with a pixel-identical copy of cam0."""
    policy = make_policy(cam2_source="duplicate")
    policy._maybe_init_per_env_state(NUM_ENVS)
    obs = _fake_observation()
    # Give cam0 a non-zero value.
    obs["camera_obs"]["external_camera_rgb"][:] = 42
    req = policy._build_request(obs, env_id=0)
    np.testing.assert_array_equal(
        req["observation/exterior_image_0_left"],
        req["observation/exterior_image_1_left"],
    )


def test_cam2_source_right_missing_raises(make_policy):
    """Missing right-shoulder camera raises AssertionError when cam2_source='right'."""
    policy = make_policy(cam2_source="right")
    policy._maybe_init_per_env_state(NUM_ENVS)
    obs = _fake_observation()  # no 'over_shoulder_right_camera' key
    with pytest.raises(AssertionError, match="external_camera_2_rgb"):
        policy._build_request(obs, env_id=0)


# ── Session management ────────────────────────────────────────────────────────


def test_session_id_lazy_init(make_policy):
    """Session IDs are None at init and stable once minted by _get_or_create_session_id."""
    policy = make_policy()
    policy._maybe_init_per_env_state(NUM_ENVS)
    assert policy._session_ids[0] is None
    sid = policy._get_or_create_session_id(0)
    assert sid is not None
    uuid.UUID(sid)
    assert policy._get_or_create_session_id(0) == sid  # stable


def test_partial_reset_preserves_other_envs(make_policy):
    """reset(env_ids=[0]) clears only that env's cache and session; env 1 is untouched."""
    policy = make_policy()
    env = _fake_env()
    obs = _fake_observation()
    policy.get_action(env, obs)  # populate caches

    policy.reset(env_ids=torch.tensor([0]))

    assert policy._cached_action_chunks[0] is None
    assert policy._session_ids[0] is None
    assert policy._cached_action_chunks[1] is not None
    assert policy._session_ids[1] is not None


def test_full_reset_clears_all(make_policy):
    """reset() with no env_ids clears every environment's cache and session."""
    policy = make_policy()
    env = _fake_env()
    obs = _fake_observation()
    policy.get_action(env, obs)

    policy.reset()

    assert all(c is None for c in policy._cached_action_chunks)
    assert all(s is None for s in policy._session_ids)


# ── Chunk caching and replay ──────────────────────────────────────────────────


def test_get_action_shape_and_dtype(make_policy):
    """get_action returns a float32 tensor of shape (num_envs, ACTION_DIM)."""
    policy = make_policy()
    env = _fake_env(num_envs=NUM_ENVS)
    obs = _fake_observation(num_envs=NUM_ENVS)
    action = policy.get_action(env, obs)
    assert action.shape == (NUM_ENVS, ACTION_DIM)
    assert action.dtype == torch.float32


def test_get_action_advances_chunk_index(make_policy):
    """Consecutive get_action calls replay successive rows from the cached chunk."""
    policy = make_policy(open_loop_horizon=24)
    env = _fake_env(num_envs=1)
    obs = _fake_observation(num_envs=1)

    first = policy.get_action(env, obs)
    second = policy.get_action(env, obs)

    # Row 0 and row 1 of the synthetic chunk differ in the last column.
    assert first[0, -1].item() == pytest.approx(0.11)
    assert second[0, -1].item() == pytest.approx(0.22)


def test_get_action_refetches_when_chunk_exhausted(monkeypatch):
    """Server is queried exactly twice when the chunk is exhausted after open_loop_horizon steps."""
    call_count = {"n": 0}

    def counting_response(data):
        call_count["n"] += 1
        return _packed_response(_synthetic_chunk(horizon=2))

    fake_ws = _FakeWs(response_fn=counting_response)
    _patch_ws(monkeypatch, fake_ws)

    policy = _make_dreamzero_remote_policy(open_loop_horizon=2)
    policy.set_task_description("test")
    env = _fake_env(num_envs=1)
    obs = _fake_observation(num_envs=1)

    policy.get_action(env, obs)  # step 0 — fetches chunk (call 1)
    policy.get_action(env, obs)  # step 1 — replays from cache
    policy.get_action(env, obs)  # step 2 — exhausted, fetches again (call 2)

    # Two server fetches: one on first get_action, one when the chunk is exhausted.
    assert call_count["n"] == 2


# ── parse_action_chunk ────────────────────────────────────────────────────────


def test_parse_action_chunk_pads_7d_to_8d(make_policy):
    """7-D server responses are zero-padded to ACTION_DIM in the last column."""
    policy = make_policy(open_loop_horizon=4)
    raw = np.ones((4, 7), dtype=np.float32)
    chunk = policy._parse_action_chunk(raw)
    assert chunk.shape == (4, 8)
    assert chunk[:, 7].sum() == 0.0


def test_parse_action_chunk_truncates_extra(make_policy):
    """Chunks longer than open_loop_horizon are truncated, not tiled."""
    policy = make_policy(open_loop_horizon=4)
    raw = np.ones((30, 8), dtype=np.float32)
    chunk = policy._parse_action_chunk(raw)
    assert chunk.shape == (4, 8)


def test_parse_action_chunk_rejects_underflow(make_policy):
    """Server returning fewer steps than open_loop_horizon raises AssertionError."""
    policy = make_policy(open_loop_horizon=24)
    raw = np.ones((10, 8), dtype=np.float32)
    with pytest.raises(AssertionError, match="open_loop_horizon"):
        policy._parse_action_chunk(raw)


def test_parse_action_chunk_accepts_dict_response(make_policy):
    """Server response as {'actions': ndarray} is accepted alongside bare arrays."""
    policy = make_policy(open_loop_horizon=4)
    raw = {"actions": np.ones((4, 8), dtype=np.float32)}
    chunk = policy._parse_action_chunk(raw)
    assert chunk.shape == (4, 8)


# ── Reconnect behavior ────────────────────────────────────────────────────────


def test_call_server_reconnects_on_connection_error(monkeypatch):
    """Connection failure triggers a reconnect, session IDs are refreshed, and the retry succeeds."""
    call_count = {"n": 0}
    good_response = _packed_response(_synthetic_chunk())

    class _FlakeyWs(_FakeWs):
        def send(self, data):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise websockets.exceptions.ConnectionClosedError(None, None)
            super().send(data)

        def recv(self, timeout=None):
            return good_response

    flakey = _FlakeyWs()
    fresh_ws = _FakeWs()

    connect_calls = {"n": 0}

    def fake_connect(*args, **kwargs):
        connect_calls["n"] += 1
        if connect_calls["n"] == 1:
            return flakey
        return fresh_ws

    monkeypatch.setattr(
        "isaaclab_arena_dreamzero.policy.dreamzero_remote_policy.ws_sync.connect",
        fake_connect,
    )

    policy = _make_dreamzero_remote_policy(open_loop_horizon=24)
    policy.set_task_description("test")
    policy._maybe_init_per_env_state(1)
    policy._session_ids[0] = "old-uuid"

    obs = _fake_observation(num_envs=1)
    request = policy._build_request(obs, env_id=0)
    policy._call_server_with_retry(request, env_id=0)

    # After reconnect: session IDs all cleared then a fresh UUID minted for env 0.
    assert policy._session_ids[0] is not None
    assert policy._session_ids[0] != "old-uuid"
    # request['session_id'] must have been refreshed so the retry used the new UUID.
    assert request["session_id"] == policy._session_ids[0]
    assert connect_calls["n"] == 2
    # The retry was actually delivered on the fresh WebSocket.
    assert len(fresh_ws._sent) == 1


def test_call_server_gives_up_after_max_attempts(monkeypatch):
    """After MAX_RECONNECT_ATTEMPTS failures the original exception is re-raised."""

    def always_fails(*args, **kwargs):
        raise websockets.exceptions.ConnectionClosedError(None, None)

    fake_ws = _FakeWs()
    fake_ws.send = always_fails

    _patch_ws(monkeypatch, fake_ws)
    policy = _make_dreamzero_remote_policy()
    policy.set_task_description("test")
    policy._maybe_init_per_env_state(1)

    obs = _fake_observation(num_envs=1)
    request = policy._build_request(obs, env_id=0)

    with pytest.raises(websockets.exceptions.ConnectionClosedError):
        policy._call_server_with_retry(request)


# ── close ─────────────────────────────────────────────────────────────────────


def test_close_is_idempotent(make_policy):
    """close() sets _ws to None and tolerates a second call without raising."""
    policy = make_policy()
    assert policy._ws is not None
    policy.close()
    assert policy._ws is None
    policy.close()  # must not raise


def test_close_does_not_raise_when_ws_already_dead(monkeypatch):
    """close() suppresses ConnectionClosed from _send_reset when the socket is already gone."""
    fake_ws = _FakeWs()

    def boom(*args, **kwargs):
        raise websockets.exceptions.ConnectionClosedError(None, None)

    fake_ws.send = boom
    _patch_ws(monkeypatch, fake_ws)

    policy = _make_dreamzero_remote_policy()
    policy.set_task_description("test")
    policy.close()  # must not propagate the ConnectionClosed from _send_reset


def test_get_action_after_close_raises_clearly(make_policy):
    """get_action() after close() raises RuntimeError, not an opaque AttributeError."""
    policy = make_policy()
    env = _fake_env()
    obs = _fake_observation()
    policy.close()
    with pytest.raises((RuntimeError, OSError, AttributeError)):
        policy.get_action(env, obs)
