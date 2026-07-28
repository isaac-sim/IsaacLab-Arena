# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch
import types

import pytest

from isaaclab_arena_cosmos.policy.cosmos_remote_config import CosmosRemotePolicyCfg
from isaaclab_arena_cosmos.policy.cosmos_remote_policy import CosmosRemotePolicy
from isaaclab_arena_cosmos.policy.droid_adapter import CosmosDroidAdapter
from isaaclab_arena_openpi.policy.websocket_client import WebsocketClientPolicy

# The closed-loop machinery (chunk caching, replay horizon, reconnect, parallel envs) is
# shared with openpi via RemoteChunkReplayPolicy and is exercised by the tests in the
# isaaclab_arena_openpi package. These tests cover only what is specific to Cosmos: the
# DROID observation wire format and the singular "action" response key.


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


def test_droid_adapter_uses_cosmos_wire_keys():
    """The wire-format contract between CosmosDroidAdapter and the Cosmos server."""
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


def test_policy_reads_singular_action_response_key(monkeypatch):
    """The Cosmos server returns its chunk under "action" (singular), unlike openpi's "actions"."""
    assert CosmosRemotePolicy.server_response_actions_key == "action"

    chunk = np.tile(np.arange(8, dtype=np.float32), (32, 1))
    chunk[0, -1] = 0.123  # mark row 0 so we can assert it is what get_action returns
    monkeypatch.setattr(WebsocketClientPolicy, "_wait_for_server", lambda self: (None, {}))
    monkeypatch.setattr(WebsocketClientPolicy, "infer", lambda self, request: {"action": chunk})

    policy = CosmosRemotePolicy(CosmosRemotePolicyCfg(policy_device="cpu"))
    policy.set_task_description("pick up the banana and place it in the bowl")

    action = policy.get_action(_fake_env(), _fake_observation())

    assert action.shape == (1, 8)
    assert action.dtype == torch.float32
    assert action[0, -1].item() == pytest.approx(0.123)
