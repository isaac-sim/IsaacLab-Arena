# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit test for Gr00tRemoteClosedloopPolicy.

Exercises the real obs/action translation pipeline and mocks only the native PolicyClient -- the
wire boundary to a remote GR00T server -- so no server is required.

Verifies:
- the obs dict sent to the server has the expected language/video/state structure;
- the action dict returned by the server is converted to a tensor of the
  expected (num_envs, action_chunk_length, action_dim) shape;
- the DROID adapter sends the embodiment's gripper closure fraction (0 open, 1 closed)
  and returns arm targets followed by the gripper command;
- the GR1 adapter maps state, action, and hold slots by joint name;
- ``reset`` propagates to the client;
- a failing ping during construction raises ``ConnectionError``.
"""

from __future__ import annotations

import numpy as np
import sys
import torch
import yaml
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from isaaclab_arena.assets.registries import PolicyRegistry
from isaaclab_arena.policy import action_scheduling
from isaaclab_arena_gr00t.policy import gr00t_remote_closedloop_policy as gr00t_policy
from isaaclab_arena_gr00t.policy.embodiment_adapters.droid_adapter import Gr00tDroidAdapter
from isaaclab_arena_gr00t.policy.embodiment_adapters.g1_wbc_adapter import Gr00tG1WbcAdapter
from isaaclab_arena_gr00t.policy.embodiment_adapters.gr1_arms_adapter import Gr00tGr1ArmsAdapter

pytestmark = pytest.mark.gr00t_policy

NUM_ENVS = 2
ORIGINAL_HEIGHT = 480
ORIGINAL_WIDTH = 640
NUM_CHANNELS = 3
NUM_SIM_JOINTS = 43  # G1 43-DoF
ACTION_HORIZON = 50  # action delta_indices in g1_sim_wbc_data_config.py
EXPECTED_ACTION_DIM = 50  # 43 joints + 3 nav + 1 base height + 3 torso rpy

DROID_CAMERA_SHAPE = (NUM_ENVS, 360, 640, NUM_CHANNELS)
DROID_ACTION_HORIZON = 32  # action delta_indices in GR00T's OXE_DROID modality config
DROID_NUM_ARM_JOINTS = 7
DROID_ACTION_DIM = 8  # 7 arm joint targets + 1 gripper command

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
def synthetic_observation():
    """Env-shaped torch obs: one ego camera + 43 joint positions per env."""
    rgb = torch.randint(0, 255, (NUM_ENVS, ORIGINAL_HEIGHT, ORIGINAL_WIDTH, NUM_CHANNELS), dtype=torch.uint8)
    joint_pos = torch.randn(NUM_ENVS, NUM_SIM_JOINTS, dtype=torch.float32)
    return {
        "camera_obs": {"robot_head_cam_rgb": rgb},
        "policy": {"robot_joint_pos": joint_pos},
    }


@pytest.fixture
def droid_observation():
    """DROID-shaped torch obs: two cameras plus the embodiment's arm, gripper, and last-action terms."""
    return {
        "camera_obs": {
            "external_camera_rgb": torch.zeros(DROID_CAMERA_SHAPE, dtype=torch.uint8),
            "wrist_camera_rgb": torch.zeros(DROID_CAMERA_SHAPE, dtype=torch.uint8),
        },
        "policy": {
            "joint_pos": torch.arange(NUM_ENVS * DROID_NUM_ARM_JOINTS, dtype=torch.float32).reshape(
                NUM_ENVS, DROID_NUM_ARM_JOINTS
            ),
            # Gripper closure fraction (0 open, 1 closed): fully closed and half closed.
            "gripper_pos": torch.tensor([[1.0], [0.5]]),
            # Last env actions; the final column is the 0-to-1 gripper command. Env 1 is still closing, so its last
            # command (1.0) differs from both its measured closure (0.5) and a zero fill.
            "actions": torch.tensor([[0.0] * DROID_NUM_ARM_JOINTS + [0.0], [0.0] * DROID_NUM_ARM_JOINTS + [1.0]]),
        },
    }


def _make_action_response(num_envs: int, horizon: int = ACTION_HORIZON) -> dict[str, np.ndarray]:
    """Synthetic G1 GR00T-server response in the format the G1 joint-space adapter expects."""
    response: dict[str, np.ndarray] = {}
    for group, size in POLICY_GROUP_SIZES.items():
        response[group] = np.random.randn(num_envs, horizon, size).astype(np.float32)
    response["navigate_command"] = np.random.randn(num_envs, horizon, 3).astype(np.float32)
    response["base_height_command"] = np.random.randn(num_envs, horizon, 1).astype(np.float32)
    return response


def _make_droid_action_response(num_envs: int, horizon: int = DROID_ACTION_HORIZON) -> dict[str, np.ndarray]:
    """Synthetic GR00T-N1.6-DROID response with distinct arm and gripper values."""
    return {
        "joint_position": np.random.randn(num_envs, horizon, DROID_NUM_ARM_JOINTS).astype(np.float32),
        "gripper_position": np.random.rand(num_envs, horizon, 1).astype(np.float32),
    }


class _FakePolicyClient:
    """Minimal stand-in for ``gr00t.policy.server_client.PolicyClient``."""

    def __init__(self, *args, ping_ok: bool = True, action_response_factory=_make_action_response, **kwargs):
        self.init_kwargs = kwargs
        self._ping_ok = ping_ok
        self._action_response_factory = action_response_factory
        self.last_action_response: dict[str, np.ndarray] | None = None
        self.last_observation: dict[str, Any] | None = None
        self.get_action_calls = 0
        self.reset_called = False

    def ping(self) -> bool:
        return self._ping_ok

    def get_action(self, observation: dict[str, Any]):
        self.last_observation = observation
        self.get_action_calls += 1
        # Match the real PolicyClient return signature: (action_dict, latency_or_meta).
        self.last_action_response = self._action_response_factory(NUM_ENVS)
        return self.last_action_response, None

    def reset(self):
        self.reset_called = True


@pytest.fixture
def fake_client_factory(monkeypatch):
    """Install a fake native PolicyClient and expose each constructed client."""
    created: list[_FakePolicyClient] = []

    def factory(ping_ok: bool = True, action_response_factory=_make_action_response):
        def _ctor(*args, **kwargs):
            client = _FakePolicyClient(
                *args, ping_ok=ping_ok, action_response_factory=action_response_factory, **kwargs
            )
            created.append(client)
            return client

        server_client_module = ModuleType("gr00t.policy.server_client")
        server_client_module.PolicyClient = _ctor
        monkeypatch.setitem(sys.modules, "gr00t.policy.server_client", server_client_module)
        return created

    return factory


def _build_policy(gr00t_embodiment_adapter: str = "g1_wbc", scheduler: str = "chunk", **cfg_fields):
    cfg = gr00t_policy.Gr00tRemoteClosedloopPolicyCfg(
        gr00t_embodiment_adapter=gr00t_embodiment_adapter,
        policy_device="cpu",  # keep the test portable; remote policy does no GPU compute
        num_envs=NUM_ENVS,
        remote_host="unused",
        remote_port=0,
        remote_api_token=None,
        scheduler=scheduler,
        **cfg_fields,
    )
    return gr00t_policy.Gr00tRemoteClosedloopPolicy(cfg)


# ------------------------------- tests ------------------------------- #


def test_gr00t_client_imports():
    """Import the real lightweight GR00T client API."""
    from gr00t.policy.server_client import MsgSerializer, PolicyClient

    assert MsgSerializer is not None
    assert PolicyClient is not None


def test_observation_sent_to_server_has_expected_structure(synthetic_observation, fake_client_factory):
    """The dict passed to ``PolicyClient.get_action`` matches GR00T's expected
    language/video/state layout for the g1 sim WBC modality config."""
    clients = fake_client_factory(ping_ok=True)
    policy = _build_policy()
    policy.set_task_description("pick up the brown box")

    policy._get_action_chunk(synthetic_observation)

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


def test_action_response_is_translated_to_correct_tensor_shape(synthetic_observation, fake_client_factory):
    """Server's grouped joint dict is reassembled into a (N, chunk, action_dim) tensor."""
    fake_client_factory(ping_ok=True)
    policy = _build_policy()
    policy.set_task_description("pick up the brown box")

    action = policy._get_action_chunk(synthetic_observation)

    assert isinstance(action, torch.Tensor)
    # _get_action_chunk asserts action_tensor.shape[1] >= action_chunk_length, but
    # the full horizon is returned; pin both dims explicitly.
    assert action.shape == (NUM_ENVS, ACTION_HORIZON, EXPECTED_ACTION_DIM)
    assert action.device.type == "cpu"


def test_reset_propagates_to_client(fake_client_factory):
    clients = fake_client_factory(ping_ok=True)
    policy = _build_policy()
    assert clients[0].reset_called is False
    policy.reset()
    assert clients[0].reset_called is True


def test_construction_fails_when_server_unreachable(fake_client_factory):
    fake_client_factory(ping_ok=False)
    with pytest.raises(ConnectionError):
        _build_policy()


def test_registration_builds_typed_config_and_scheduler(fake_client_factory):
    """The registered typed config retains scheduler selection."""
    fake_client_factory(ping_ok=True)
    assert (
        PolicyRegistry().get_policy_cfg_type(gr00t_policy.Gr00tRemoteClosedloopPolicy)
        is gr00t_policy.Gr00tRemoteClosedloopPolicyCfg
    )
    policy = gr00t_policy.Gr00tRemoteClosedloopPolicy(
        gr00t_policy.Gr00tRemoteClosedloopPolicyCfg(
            gr00t_embodiment_adapter="g1_wbc",
            policy_device="cpu",
            num_envs=NUM_ENVS,
            remote_host="unused",
            remote_port=0,
            scheduler="synced_batch",
        )
    )

    assert isinstance(policy.config, gr00t_policy.Gr00tRemoteClosedloopPolicyCfg)
    assert isinstance(policy._chunking_state, action_scheduling.SyncedBatchActionScheduler)


# ---------------- scheduler-switch tests ---------------- #


@pytest.mark.parametrize("scheduler_cls_name", ["ActionChunkScheduler", "SyncedBatchActionScheduler"])
def test_get_action_returns_correct_shape_for_each_scheduler(
    synthetic_observation, fake_client_factory, scheduler_cls_name
):
    """Both ActionChunkScheduler and SyncedBatchActionScheduler should drive the policy
    end-to-end and produce one (num_envs, action_dim) action per get_action call."""
    scheduler_cls = getattr(action_scheduling, scheduler_cls_name)
    scheduler = "synced_batch" if scheduler_cls_name == "SyncedBatchActionScheduler" else "chunk"

    fake_client_factory(ping_ok=True)
    policy = _build_policy(scheduler=scheduler)
    assert isinstance(policy._chunking_state, scheduler_cls)
    policy.set_task_description("pick up the brown box")

    action = policy.get_action(env=None, observation=synthetic_observation)

    assert isinstance(action, torch.Tensor)
    assert action.shape == (NUM_ENVS, EXPECTED_ACTION_DIM)
    assert action.device.type == "cpu"


def test_synced_batch_holds_joint_position_for_env_after_partial_reset(synthetic_observation, fake_client_factory):
    """After resetting a single env, SyncedBatchActionScheduler should hold that env on
    the joint-position-derived hold_action while the others continue stepping their chunk."""
    clients = fake_client_factory(ping_ok=True)
    policy = _build_policy(scheduler="synced_batch")
    policy.set_task_description("pick up the brown box")

    # Step 1: every env needs a chunk → exactly one fetch.
    policy.get_action(env=None, observation=synthetic_observation)
    assert clients[0].get_action_calls == 1

    # Reset env 1 only; env 0 still has its chunk.
    policy.reset(env_ids=torch.tensor([1]))

    action = policy.get_action(env=None, observation=synthetic_observation)
    # No new chunk fetched (env 0 not yet exhausted, so .all() is False).
    assert clients[0].get_action_calls == 1
    expected_hold = torch.zeros(EXPECTED_ACTION_DIM)
    expected_hold[:NUM_SIM_JOINTS] = synthetic_observation["policy"]["robot_joint_pos"][1]
    torch.testing.assert_close(action[1], expected_hold)


def test_default_image_size_keeps_gr00t_padding_in_the_loop():
    """Declare twice the DROID target size so GR00T's pad-and-resize still runs on fitted camera frames."""
    from isaaclab_arena_gr00t.policy.gr00t_core import resize_rgb_for_policy

    cfg = gr00t_policy.Gr00tRemoteClosedloopPolicyCfg(
        gr00t_embodiment_adapter="droid", remote_host="unused", remote_port=0, remote_api_token=None
    )

    image_size = gr00t_policy.Gr00tRemoteClosedloopPolicy.default_image_size(cfg)

    assert image_size == (360, 640)
    fitted_frame = np.zeros((1, *image_size, 3), dtype=np.uint8)
    fitted_frame[:, : image_size[0] // 2] = 255  # scene content in the top half of the frame
    (processed,) = resize_rgb_for_policy([fitted_frame], target_image_size=(180, 320, 3))
    assert processed.shape == (1, 180, 320, 3)
    # Padding 360x640 to a 640x640 square puts the frame in rows 140-500, so its top half lands in rows
    # 39-90 of the 180-row output, below a black band, exactly as it does for a 720x1280 camera frame.
    assert processed[0, :35].max() == 0
    assert processed[0, 45:85].min() == 255
    assert processed[0, 95:].max() == 0


# ---------------- embodiment adapter tests ---------------- #


def test_config_selects_embodiment_adapter(fake_client_factory):
    """gr00t_embodiment_adapter selects the adapter by built-in name or by class path."""
    fake_client_factory(ping_ok=True)

    assert isinstance(_build_policy("droid").embodiment_adapter, Gr00tDroidAdapter)
    gr1_adapter = _build_policy("gr1_arms").embodiment_adapter
    assert isinstance(gr1_adapter, Gr00tGr1ArmsAdapter)
    assert gr1_adapter.action_dim == 36  # 14 arm joints and 22 hand joints
    assert isinstance(_build_policy("g1_wbc").embodiment_adapter, Gr00tG1WbcAdapter)

    class_path = f"{Gr00tDroidAdapter.__module__}.{Gr00tDroidAdapter.__qualname__}"
    assert isinstance(_build_policy(class_path).embodiment_adapter, Gr00tDroidAdapter)


def test_action_chunk_length_defaults_to_checkpoint_action_horizon(fake_client_factory):
    """Without action_chunk_length, each chunk runs the modality config's full action horizon."""
    fake_client_factory(ping_ok=True)

    assert _build_policy("g1_wbc").action_chunk_length == ACTION_HORIZON
    assert _build_policy("droid").action_chunk_length == DROID_ACTION_HORIZON
    assert _build_policy("g1_wbc", action_chunk_length=10).action_chunk_length == 10
    with pytest.raises(AssertionError, match="action_chunk_length must be between 1 and"):
        _build_policy("g1_wbc", action_chunk_length=ACTION_HORIZON + 1)


def test_droid_state_sent_to_server_uses_gripper_closure_fraction(droid_observation, fake_client_factory):
    """DROID state comes from the embodiment's arm joints and gripper closure fraction, not the raw finger angle."""
    clients = fake_client_factory(ping_ok=True, action_response_factory=_make_droid_action_response)
    policy = _build_policy("droid")
    policy.set_task_description("put the lemon in the bowl")

    policy._get_action_chunk(droid_observation)

    sent = clients[0].last_observation
    assert set(sent["video"]) == {"exterior_image_1_left", "wrist_image_left"}
    for frames in sent["video"].values():
        assert frames.shape == (NUM_ENVS, 1, 180, 320, NUM_CHANNELS)
    assert set(sent["state"]) == {"joint_position", "gripper_position"}
    np.testing.assert_array_equal(
        sent["state"]["joint_position"][:, 0, :], droid_observation["policy"]["joint_pos"].numpy()
    )
    np.testing.assert_array_equal(
        sent["state"]["gripper_position"][:, 0, :], droid_observation["policy"]["gripper_pos"].numpy()
    )


def test_droid_action_is_arm_targets_then_gripper_command(droid_observation, fake_client_factory):
    """GR00T's joint_position and gripper_position actions fill the 7 arm slots and the gripper slot in order."""
    clients = fake_client_factory(ping_ok=True, action_response_factory=_make_droid_action_response)
    policy = _build_policy("droid")
    policy.set_task_description("put the lemon in the bowl")

    action = policy._get_action_chunk(droid_observation)

    response = clients[0].last_action_response
    assert action.shape == (NUM_ENVS, DROID_ACTION_HORIZON, DROID_ACTION_DIM)
    np.testing.assert_allclose(action[..., :DROID_NUM_ARM_JOINTS].numpy(), response["joint_position"])
    np.testing.assert_allclose(action[..., DROID_NUM_ARM_JOINTS:].numpy(), response["gripper_position"])


def test_droid_synced_batch_holds_arm_joints_and_last_gripper_command(droid_observation, fake_client_factory):
    """A waiting DROID env holds its measured arm joints and its last 0-to-1 gripper command."""
    clients = fake_client_factory(ping_ok=True, action_response_factory=_make_droid_action_response)
    policy = _build_policy("droid", scheduler="synced_batch")
    policy.set_task_description("put the lemon in the bowl")

    policy.get_action(env=None, observation=droid_observation)
    policy.reset(env_ids=torch.tensor([1]))
    action = policy.get_action(env=None, observation=droid_observation)

    assert clients[0].get_action_calls == 1
    proprio = droid_observation["policy"]
    expected_hold = torch.cat([proprio["joint_pos"][1], proprio["actions"][1, DROID_NUM_ARM_JOINTS:]])
    torch.testing.assert_close(action[1], expected_hold)


@pytest.mark.parametrize(
    "gr00t_embodiment_adapter, message",
    [
        (
            "droid_arm",
            r"must be one of \['droid', 'g1_wbc', 'gr1_arms'\] or a dotted class path, got 'droid_arm'",
        ),
        ("pathlib.Path", "is not a Gr00tEmbodimentAdapter subclass"),
        (
            "isaaclab_arena_gr00t.policy.embodiment_adapters.droid_adapter.Gr00tDroid",
            "droid_adapter has no Gr00tDroid",
        ),
        (
            "isaaclab_arena_gr00t.policy.embodiment_adapters.gr00t_embodiment_adapter.Gr00tEmbodimentAdapter",
            r"does not implement \['build_action_np', 'build_hold_action', 'extract_state_np'\]",
        ),
    ],
)
def test_embodiment_adapter_config_errors(fake_client_factory, gr00t_embodiment_adapter, message):
    """An invalid adapter name or class path fails when the config is built, before connecting."""
    clients = fake_client_factory(ping_ok=True)

    with pytest.raises(AssertionError, match=message):
        _build_policy(gr00t_embodiment_adapter)
    assert not clients


def test_adapter_subclass_loads_joint_maps_from_str_paths():
    """A subclass may set its joint-map paths as str instead of Path."""

    class Gr1ArmsAdapterWithStrPaths(Gr00tGr1ArmsAdapter):
        policy_joints_config_path = str(Gr00tGr1ArmsAdapter.policy_joints_config_path)
        state_joints_config_path = str(Gr00tGr1ArmsAdapter.state_joints_config_path)
        action_joints_config_path = str(Gr00tGr1ArmsAdapter.action_joints_config_path)

    adapter = Gr1ArmsAdapterWithStrPaths()
    reference = Gr00tGr1ArmsAdapter()
    assert adapter.policy_joints_config == reference.policy_joints_config
    assert adapter.state_joints_config == reference.state_joints_config
    assert adapter.action_joints_config == reference.action_joints_config


def test_gr1_adapter_maps_joints_by_name():
    """GR1 state, action, and hold slots follow the joint names in the adapter's joint YAMLs."""
    adapter = Gr00tGr1ArmsAdapter()
    policy_groups = yaml.safe_load(Path(adapter.policy_joints_config_path).read_text())["joints"]
    state_columns = yaml.safe_load(Path(adapter.state_joints_config_path).read_text())["joints"]
    action_columns = yaml.safe_load(Path(adapter.action_joints_config_path).read_text())["joints"]
    # Each sim joint's position equals its state column, so every value identifies its joint.
    joint_pos = torch.arange(len(state_columns), dtype=torch.float32).repeat(NUM_ENVS, 1)
    observation = {"policy": {"robot_joint_pos": joint_pos}}

    state = adapter.extract_state_np(observation)
    for group, joint_names in policy_groups.items():
        expected = [state_columns[name] for name in joint_names]
        np.testing.assert_array_equal(state[group], np.tile(expected, (NUM_ENVS, 1)))

    # Each GR00T action value encodes its group and its position within the group.
    horizon = 4
    response = {}
    for group_idx, (group, joint_names) in enumerate(policy_groups.items()):
        values = 100 * group_idx + np.arange(len(joint_names), dtype=np.float32)
        response[group] = np.broadcast_to(values, (NUM_ENVS, horizon, len(joint_names))).copy()
    action = adapter.build_action_np(response)
    assert action.shape == (NUM_ENVS, horizon, len(action_columns))
    for group_idx, (group, joint_names) in enumerate(policy_groups.items()):
        for joint_idx, name in enumerate(joint_names):
            np.testing.assert_array_equal(action[..., action_columns[name]], 100 * group_idx + joint_idx)

    hold = adapter.build_hold_action(observation)
    for name, column in action_columns.items():
        expected_value = float(state_columns.get(name, 0.0))
        torch.testing.assert_close(hold[:, column], torch.full((NUM_ENVS,), expected_value))
