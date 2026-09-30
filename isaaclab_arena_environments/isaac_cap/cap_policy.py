# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CAP's RGB-D socket client; all graph and planning code runs in CAP."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import socket
import struct
import time
import torch
from dataclasses import dataclass, field
from itertools import product
from typing import Any

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import matrix_from_quat

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg


def cap_episode_finished(env) -> torch.Tensor:
    """End a disconnected CAP episode after its policy-requested settling period."""
    return torch.full((env.num_envs,), getattr(env, "cap_episode_finished", False), device=env.device, dtype=torch.bool)


_CAP_ROBOT_FR3 = "fr3"
_CAP_ROBOT_YAM_BIMANUAL = "yam_bimanual"


@dataclass
class CapPolicyCfg(PolicyCfg):
    """Connect a supported CAP robot profile to a Arena environment."""

    host: str = "127.0.0.1"
    port: int = 9000
    connect_timeout_s: float = 180.0
    io_timeout_s: float = 180.0
    settle_s: float = 2.0
    robot_profile: str = _CAP_ROBOT_FR3
    camera_mapping: dict[str, str] = field(default_factory=dict)


@register_policy
class CapPolicy(PolicyBase[CapPolicyCfg]):
    """Exchange calibrated observations and absolute joint targets with CAP."""

    name = "cap_remote"
    _max_frame_bytes = 64 * 1024 * 1024

    def __init__(self, config: CapPolicyCfg) -> None:
        super().__init__(config)
        if config.robot_profile not in (_CAP_ROBOT_FR3, _CAP_ROBOT_YAM_BIMANUAL):
            raise ValueError(f"Unsupported CAP robot profile: {config.robot_profile}")
        # Optional client dependencies must not prevent environment-only use.
        import msgpack
        import msgpack_numpy

        self._msgpack = msgpack
        self._numpy_codec = msgpack_numpy
        self._socket: socket.socket | None = None
        self._finished = False
        self._settle_steps = 0
        self._last_gripper: torch.Tensor | None = None
        self._env: ManagerBasedRLEnv | None = None

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.close()
        self._finished = False
        self._settle_steps = 0
        self._last_gripper = None
        if self._env is not None:
            self._env.cap_episode_finished = False

    def _connect(self) -> None:
        deadline = time.monotonic() + self.config.connect_timeout_s
        while True:
            try:
                self._socket = socket.create_connection((self.config.host, self.config.port), timeout=2)
                self._socket.settimeout(self.config.io_timeout_s)
                return
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.5)

    def _receive(self, size: int) -> bytes:
        assert self._socket is not None
        data = bytearray()
        while len(data) < size:
            chunk = self._socket.recv(size - len(data))
            if not chunk:
                raise EOFError("CAP graph closed the connection")
            data.extend(chunk)
        return bytes(data)

    def _exchange(self, frame: dict[str, Any]) -> dict[str, Any]:
        assert self._socket is not None
        payload = self._msgpack.packb(frame, default=self._numpy_codec.encode, use_bin_type=True)
        assert len(payload) <= self._max_frame_bytes
        self._socket.sendall(struct.pack("!I", len(payload)) + payload)
        size = struct.unpack("!I", self._receive(4))[0]
        assert 0 < size <= self._max_frame_bytes
        reply = self._msgpack.unpackb(self._receive(size), object_hook=self._numpy_codec.decode, raw=False)
        assert isinstance(reply, dict)
        return reply

    @staticmethod
    def _camera(env: ManagerBasedRLEnv, name: str) -> dict[str, Any]:
        data = env.scene[name].data
        pose = torch.eye(4, device=env.device)
        pose[:3, :3] = matrix_from_quat(data.quat_w_ros.torch[0])
        pose[:3, 3] = data.pos_w.torch[0] - env.scene.env_origins[0]
        rgb = data.output["rgb"].torch[0, ..., :3].cpu().numpy()
        depth = data.output["distance_to_image_plane"].torch[0].squeeze(-1).cpu().numpy()
        return {
            "images": {"rgb": np.ascontiguousarray(rgb, dtype=np.uint8)},
            "depth_data": np.ascontiguousarray(np.nan_to_num(depth, nan=0, posinf=0, neginf=0), dtype=np.float32),
            "intrinsics": {"left": {"intrinsics_matrix": data.intrinsic_matrices.torch[0].cpu().numpy()}},
            "pose_mat": pose.cpu().numpy(),
        }

    @staticmethod
    def _tip_reach(robot: Articulation) -> float:
        names = list(robot.body_names)
        base = names.index("robotiq_base")
        pads = [names.index(name) for name in ("robotiq_left_pad", "robotiq_right_pad")]
        pos = robot.data.body_pos_w.torch[0]
        rot = matrix_from_quat(robot.data.body_quat_w.torch[0])
        approach = rot[base, :, 2]
        # Shared FR3/Robotiq USD: TCP is 0.157 m along robotiq_base's local Z.
        # The two pad collision boxes below use pad-link coordinates in metres.
        tcp = pos[base] + approach * 0.157
        centers = pos.new_tensor(((0.043258, 0, 0.12), (0.043258, 0, 0.13875)))
        corners = centers[:, None] + pos.new_tensor(list(product((-1, 1), repeat=3)))[None] * pos.new_tensor(
            (0.002, 0.011, 0.009375)
        )
        world = (rot[pads] @ corners.reshape(-1, 3).T).transpose(1, 2) + pos[pads, None]
        return float(((world - tcp) * approach).sum(-1).max().cpu())

    @staticmethod
    def _valid(reply: dict[str, Any], channel: str, arm: str = "left") -> bool:
        if channel in reply:
            value = reply[channel]
        elif "command_valid" in reply:
            value = reply["command_valid"]
        else:
            block = reply.get(arm, {})
            value = block.get(channel, block.get("command_valid", False))
        if isinstance(value, dict):
            value = value.get(arm, False)
        assert isinstance(value, (bool, np.bool_)), f"Invalid CAP validity flag: {channel}={value!r}"
        return bool(value)

    @staticmethod
    def _fr3_hold_action(env: ManagerBasedRLEnv) -> torch.Tensor:
        robot = env.scene["robot"]
        names = list(robot.joint_names)
        joints = robot.data.joint_pos.torch[0]
        arm = joints[[names.index(f"fr3_joint{i}") for i in range(1, 8)]]
        scale = env.action_manager.get_term("gripper_action").cfg.scale
        closed = torch.clamp(joints[names.index("left_driver_joint")] / scale, 0, 1)
        return torch.cat((arm, closed.reshape(1))).clone()

    @staticmethod
    def _yam_hold_action(env: ManagerBasedRLEnv) -> torch.Tensor:
        blocks = []
        for side in ("left", "right"):
            robot = env.scene[f"{side}_robot"]
            names = list(robot.joint_names)
            joints = robot.data.joint_pos.torch[0]
            arm = joints[[names.index(f"joint{i}") for i in range(1, 7)]]
            gripper_cfg = env.action_manager.get_term(f"{side}_gripper_action").cfg
            driver = joints[names.index("left_finger")]
            closed = torch.clamp((driver - gripper_cfg.offset) / gripper_cfg.scale, 0, 1)
            blocks.extend((arm, closed.reshape(1)))
        return torch.cat(blocks).clone()

    def _hold_action(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        if self.config.robot_profile == _CAP_ROBOT_YAM_BIMANUAL:
            return self._yam_hold_action(env)
        elif self.config.robot_profile == _CAP_ROBOT_FR3:
            return self._fr3_hold_action(env)
        else:
            raise ValueError(f"Unsupported CAP robot profile: {self.config.robot_profile}")

    def _camera_pairs(self) -> tuple[tuple[str, str], ...]:
        if self.config.camera_mapping:
            return tuple(self.config.camera_mapping.items())
        return (
            ("top_camera", "overhead"),
            ("wrist_camera", "eye_in_hand"),
            ("exterior_left_camera", "agentview"),
        )

    def _observation_frame(self, env: ManagerBasedRLEnv, action: torch.Tensor) -> dict[str, Any]:
        if self.config.robot_profile == _CAP_ROBOT_YAM_BIMANUAL:
            assert action.shape == (14,), f"Bimanual YAM action must have shape (14,), got {action.shape}."
            frame = {
                "timestamp": time.time(),
                "left": {"joint_pos": [*action[:6].cpu().tolist(), 1 - float(action[6])]},
                "right": {"joint_pos": [*action[7:13].cpu().tolist(), 1 - float(action[13])]},
            }
            for name, alias in self._camera_pairs():
                frame[alias] = self._camera(env, name)
            return frame

        elif self.config.robot_profile == _CAP_ROBOT_FR3:
            frame = {
                "timestamp": time.time(),
                "left": {"joint_pos": [*action[:7].cpu().tolist(), 1 - float(action[7])]},
                "_isaac_cap": {
                    "workspace": {
                        "surface_z": 0.780,
                        "transport_z": 1.102,
                        "align_clearance_m": 0.12,
                        "pregrasp_standoff_m": 0.12,
                    },
                    "gripper": {"tip_reach_m": self._tip_reach(env.scene["robot"])},
                },
            }
            for name, alias in self._camera_pairs():
                frame[alias] = self._camera(env, name)
            return frame
        else:
            raise ValueError(f"Unsupported CAP robot profile: {self.config.robot_profile}")

    def _apply_reply(self, reply: dict[str, Any], action: torch.Tensor) -> None:
        if self.config.robot_profile == _CAP_ROBOT_YAM_BIMANUAL:
            assert action.shape == (14,), f"Bimanual YAM action must have shape (14,), got {action.shape}."
            for arm, offset in (("left", 0), ("right", 7)):
                block = reply.get(arm, {})
                if self._valid(reply, "arm_valid", arm):
                    target = np.asarray(block["joint_pos"], dtype=np.float32)
                    assert target.shape == (6,) and np.isfinite(target).all()
                    action[offset : offset + 6] = torch.as_tensor(target, device=action.device)
                if self._valid(reply, "gripper_valid", arm):
                    opened = float(block["gripper"])
                    assert np.isfinite(opened) and 0 <= opened <= 1
                    action[offset + 6] = 1 - opened
            self._last_gripper = action[[6, 13]].clone()
            return

        elif self.config.robot_profile == _CAP_ROBOT_FR3:
            block = reply.get("left", {})
            if self._valid(reply, "arm_valid"):
                target = np.asarray(block["joint_pos"], dtype=np.float32)
                assert target.shape == (7,) and np.isfinite(target).all()
                action[:7] = torch.as_tensor(target, device=action.device)
            if self._valid(reply, "gripper_valid"):
                opened = float(block["gripper"])
                assert np.isfinite(opened) and 0 <= opened <= 1
                action[7] = 1 - opened
            self._last_gripper = action[7].clone()
        else:
            raise ValueError(f"Unsupported CAP robot profile: {self.config.robot_profile}")

    def _settle(self, env: ManagerBasedRLEnv, action: torch.Tensor) -> None:
        if self.config.robot_profile == _CAP_ROBOT_YAM_BIMANUAL:
            if self._last_gripper is not None:
                action[[6, 13]] = self._last_gripper
        elif self.config.robot_profile == _CAP_ROBOT_FR3:
            if self._last_gripper is not None:
                action[7] = self._last_gripper
        else:
            raise ValueError(f"Unsupported CAP robot profile: {self.config.robot_profile}")
        self._settle_steps += 1
        env.cap_episode_finished = self._settle_steps * env.step_dt >= self.config.settle_s

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        env = env.unwrapped
        assert env.num_envs == 1, "CAP client requires one graph per environment"
        self._env = env
        action = self._hold_action(env)
        if not self._finished:
            if self._socket is None:
                # Refresh RTX buffers after the episode reset before CAP sees them.
                for _ in range(5):
                    env.sim.render()
                env.scene.update(0.0)
                print(
                    f"[CapPolicy] Environment ready; waiting for GaP at {self.config.host}:{self.config.port}",
                    flush=True,
                )
                self._connect()
            try:
                reply = self._exchange(self._observation_frame(env, action))
            except (EOFError, ConnectionResetError, BrokenPipeError) as error:
                print(f"[CapPolicy] {error}; settling for {self.config.settle_s}s", flush=True)
                self._finished = True
                self.close()
            else:
                self._apply_reply(reply, action)
        if self._finished:
            self._settle(env, action)
        return action[None]
