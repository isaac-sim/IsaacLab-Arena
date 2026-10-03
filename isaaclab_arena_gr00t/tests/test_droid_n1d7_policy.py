# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check N1.7's wire contract, pose convention, and absolute action handling."""

import numpy as np
import torch
from copy import deepcopy
from scipy.spatial.transform import Rotation
from types import SimpleNamespace

import msgpack
import pytest
import warp as wp
from gr00t.data.types import ModalityConfig
from gr00t.data.utils import to_json_serializable
from gr00t.policy.server_client import MsgSerializer

from isaaclab_arena_gr00t.embodiments.droid.n1d7_observations import droid_pose_to_eef9d
from isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy import (
    Gr00tRemoteClosedloopPolicy,
    Gr00tRemoteClosedloopPolicyCfg,
)
from isaaclab_arena_gr00t.utils.n1d7_wire import decode_n1d7_response

pytestmark = pytest.mark.gr00t_policy


def n1d7_wire_response(value):
    """Emit N1.7 envelopes, then receive them through the real N1.6 decoder."""

    def encode(item):
        if isinstance(item, ModalityConfig):
            return {"__ModalityConfig__": True, "as_json": to_json_serializable(item)}
        if isinstance(item, np.ndarray):
            return {
                b"nd": True,
                b"type": item.dtype.str,
                b"kind": b"",
                b"shape": item.shape,
                b"data": item.tobytes(),
            }
        raise TypeError(type(item))

    return MsgSerializer.from_bytes(msgpack.packb(value, default=encode))


def test_wire_adapter_rejects_object_arrays():
    payload = {b"nd": True, b"type": "O", b"data": b"not numeric", b"shape": [1]}
    with pytest.raises(ValueError, match="Unsupported GR00T response dtype"):
        decode_n1d7_response(payload)


class DroidClient:
    """Model the checkpoint's native server response without performing inference."""

    def __init__(self, **kwargs):
        self.observation = None

    def ping(self):
        return True

    def get_modality_config(self):
        modalities = {
            "video": ModalityConfig([0], ["exterior_image_1_left", "wrist_image_left"]),
            "state": ModalityConfig([0], ["eef_9d", "gripper_position", "joint_position"]),
            "action": ModalityConfig(
                list(range(40)),
                ["eef_9d", "gripper_position", "joint_position"],
                action_configs=[
                    {"rep": "RELATIVE", "type": "EEF", "format": "XYZ_ROT6D", "state_key": "eef_9d"},
                    {"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT", "state_key": "gripper_position"},
                    {"rep": "RELATIVE", "type": "NON_EEF", "format": "DEFAULT", "state_key": "joint_position"},
                ],
            ),
            "language": ModalityConfig([0], ["annotation.language.language_instruction"]),
        }
        return n1d7_wire_response(modalities)

    def get_action(self, observation):
        self.observation = observation
        batch = observation["state"]["joint_position"].shape[0]
        # Upstream decode_action already converts relative predictions to absolute positions.
        return (
            n1d7_wire_response({
                "eef_9d": np.zeros((batch, 40, 9), dtype=np.float32),
                "joint_position": np.full((batch, 40, 7), 0.25, dtype=np.float32),
                "gripper_position": np.full((batch, 40, 1), 0.9, dtype=np.float32),
            }),
            {},
        )


@pytest.fixture
def droid_policy(monkeypatch):
    monkeypatch.setattr("gr00t.policy.server_client.PolicyClient", DroidClient)
    policy = Gr00tRemoteClosedloopPolicy(
        Gr00tRemoteClosedloopPolicyCfg(
            policy_config_yaml_path="isaaclab_arena_gr00t/policy/config/droid_manip_gr00t_n1d7_closedloop_config.yaml",
            policy_device="cpu",
            num_envs=2,
        )
    )
    policy.set_task_description("pick up the spring clamp and place it in the right bin")
    return policy


@pytest.fixture
def droid_observation():
    return {
        "policy": {
            "robot_joint_pos": torch.ones(2, 13),
            "gripper_pos": torch.tensor([[0.0], [1.0]]),
            "droid_eef_pose_base": torch.tensor([
                [0.4, 0.1, 0.3, 1.0, 0.0, 0.0, 0.0],
                [0.5, 0.2, 0.4, 1.0, 0.0, 0.0, 0.0],
            ]),
        },
        "camera_obs": {
            "external_camera_rgb": torch.zeros(2, 180, 320, 3, dtype=torch.uint8),
            "wrist_camera_rgb": torch.full((2, 180, 320, 3), 255, dtype=torch.uint8),
        },
    }


def test_n1d7_observation_and_absolute_actions(droid_policy, droid_observation):
    actions = droid_policy._get_action_chunk(droid_observation, ["external_camera_rgb", "wrist_camera_rgb"])
    sent = droid_policy._client.observation
    assert set(sent["state"]) == {"eef_9d", "gripper_position", "joint_position"}
    assert sent["state"]["eef_9d"].shape == (2, 1, 9)
    np.testing.assert_allclose(sent["state"]["eef_9d"][0, 0], [0.4, 0.1, 0.3, 0, 0, -1, -1, 0, 0])
    np.testing.assert_array_equal(sent["state"]["gripper_position"][:, 0, 0], [0, 1])
    assert sent["video"]["exterior_image_1_left"].shape == (2, 1, 180, 320, 3)
    assert sent["video"]["wrist_image_left"].min() == 255
    assert actions.shape == (2, 40, 8)
    torch.testing.assert_close(actions[..., :7], torch.full((2, 40, 7), 0.25))
    torch.testing.assert_close(actions[..., 7], torch.full((2, 40), 0.9))


def test_n1d7_replans_after_eight_actions(droid_policy, droid_observation):
    for _ in range(8):
        action = droid_policy.get_action(None, droid_observation)
        assert action.shape == (2, 8)
    assert droid_policy._chunking_state.env_requires_new_chunk.all()
    held = droid_policy._extract_hold_action(droid_observation)
    torch.testing.assert_close(held[:, 7], torch.tensor([0.0, 1.0]))


def test_n1d7_requires_end_effector_observation(droid_policy, droid_observation):
    observation = deepcopy(droid_observation)
    del observation["policy"]["droid_eef_pose_base"]
    with pytest.raises(KeyError, match="droid_eef_pose_base"):
        droid_policy._get_action_chunk(observation, ["external_camera_rgb", "wrist_camera_rgb"])


def test_pose_matches_droid_euler_convention():
    # Independent, explicit Rx @ Ry @ Rz construction checks the TFG convention,
    # row-major rot6d layout, and egocentric correction for a multi-axis rotation.
    angles = [0.3, -0.4, 0.7]
    quat = Rotation.from_euler("xyz", angles).as_quat()
    pose = np.array([[0.4, 0.1, 0.3, quat[3], *quat[:3]]])
    x, y, z = angles
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    correction = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    expected = (rx @ ry @ rz @ correction)[:2].reshape(6)
    np.testing.assert_allclose(droid_pose_to_eef9d(pose)[0, 3:], expected, atol=1e-6)


def test_panda_link8_pose_is_invariant_to_robot_placement():
    from isaaclab_arena.embodiments.droid.observations import droid_eef_pose_base

    # The second robot is translated and rotated 90 degrees around world Z.
    # Its wrist has exactly the same local pose as the first robot's wrist.
    s = 2**-0.5
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.4, 0.1, 0.3]], [[10.0, 20.0, 1.0], [9.9, 20.4, 1.3]]])
    # Articulation data and Isaac Lab's frame transforms use xyzw quaternions.
    quaternions = torch.tensor([[[0.0, 0.0, 0.0, 1.0], [s, 0.0, 0.0, s]], [[0.0, 0.0, s, s], [0.5, 0.5, 0.5, 0.5]]])
    data = SimpleNamespace(
        body_names=["panda_link0", "panda_link7"],
        body_pos_w=wp.from_torch(positions),
        body_quat_w=wp.from_torch(quaternions),
    )
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})
    actual = droid_eef_pose_base(env)
    expected = torch.tensor([[0.4, -0.007, 0.3, s, s, 0.0, 0.0]]).repeat(2, 1)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


def test_n1d6_droid_retains_its_local_modalities(monkeypatch, droid_observation):
    class N1d6Client(DroidClient):
        def get_modality_config(self):
            raise AssertionError("N1.6 should use its existing local modality config")

        def get_action(self, observation):
            actions, info = super().get_action(observation)
            return {k: v[:, :32] for k, v in decode_n1d7_response(actions).items() if k != "eef_9d"}, info

    monkeypatch.setattr("gr00t.policy.server_client.PolicyClient", N1d6Client)
    policy = Gr00tRemoteClosedloopPolicy(
        Gr00tRemoteClosedloopPolicyCfg(
            policy_config_yaml_path="isaaclab_arena_gr00t/policy/config/droid_manip_gr00t_closedloop_config.yaml",
            policy_device="cpu",
            num_envs=2,
        )
    )
    policy.set_task_description("pick up the clamp")
    del droid_observation["policy"]["droid_eef_pose_base"]
    del droid_observation["policy"]["gripper_pos"]
    action = policy._get_action_chunk(droid_observation, ["external_camera_rgb", "wrist_camera_rgb"])
    assert action.shape == (2, 32, 8)
    assert set(policy._client.observation["state"]) == {"joint_position", "gripper_position"}


@pytest.mark.parametrize(
    "state_keys,horizon",
    [(["joint_position", "gripper_position"], 40), (["eef_9d", "joint_position", "gripper_position"], 32)],
)
def test_rejects_mismatched_checkpoint(monkeypatch, state_keys, horizon):
    class WrongCheckpointClient(DroidClient):
        def get_modality_config(self):
            config = decode_n1d7_response(super().get_modality_config())
            config["state"].modality_keys = state_keys
            config["action"].delta_indices = list(range(horizon))
            return config

    monkeypatch.setattr("gr00t.policy.server_client.PolicyClient", WrongCheckpointClient)
    with pytest.raises(AssertionError, match="server|checkpoint"):
        Gr00tRemoteClosedloopPolicy(
            Gr00tRemoteClosedloopPolicyCfg(
                policy_config_yaml_path=(
                    "isaaclab_arena_gr00t/policy/config/droid_manip_gr00t_n1d7_closedloop_config.yaml"
                ),
                policy_device="cpu",
            )
        )
