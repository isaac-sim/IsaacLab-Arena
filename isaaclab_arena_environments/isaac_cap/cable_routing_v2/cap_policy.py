# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CAP socket adapter for the cable-routing YAM-I2RT contract."""

from __future__ import annotations

import numpy as np
import socket
import torch
from dataclasses import dataclass, field
from typing import Any

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_environments.isaac_cap.cap_policy import CapPolicy, CapPolicyCfg


@dataclass
class CapYamI2rtPolicyCfg(CapPolicyCfg):
    """Connect Arena's cable-routing YAM-I2RT embodiment to its CAP graph."""

    port: int = 19000
    robot_profile: str = "yam_bimanual"
    gripper_open_position: float = 0.0475
    wire_arm_order: list[str] = field(default_factory=lambda: ["right", "left"])
    workspace: dict[str, float] = field(default_factory=dict)


@register_policy
class CapYamI2rtPolicy(PolicyBase[CapYamI2rtPolicyCfg]):
    """Translate the cable YAM-I2RT schema over the shared CAP socket transport."""

    name = "cap_yam_i2rt_remote"
    _max_frame_bytes = CapPolicy._max_frame_bytes
    _camera = staticmethod(CapPolicy._camera)
    close = CapPolicy.close
    _connect = CapPolicy._connect
    _receive = CapPolicy._receive
    _exchange = CapPolicy._exchange
    reset = CapPolicy.reset
    _settle = CapPolicy._settle
    get_action = CapPolicy.get_action

    def __init__(self, config: CapYamI2rtPolicyCfg) -> None:
        super().__init__(config)
        assert config.gripper_open_position > 0.0, "gripper_open_position must be positive"
        assert sorted(config.wire_arm_order) == [
            "left",
            "right",
        ], "wire_arm_order must map the GaP left/right channels to both physical YAM arms"
        assert config.camera_mapping, "camera_mapping must select at least one cable-routing camera"
        assert all(
            name and alias for name, alias in config.camera_mapping.items()
        ), "camera_mapping must contain non-empty Arena camera names and GaP aliases"
        assert config.workspace, "workspace must provide the cable-routing graph geometry"
        assert all(np.isfinite(value) for value in config.workspace.values()), "workspace values must be finite"
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

    def _arm_state(self, env: ManagerBasedRLEnv, side: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Read one prefixed six-joint arm and its paired I2RT fingers."""
        robot = env.scene[f"{side}_robot"]
        names = list(robot.joint_names)
        positions = robot.data.joint_pos.torch[0]
        arm = positions[[names.index(f"{side}_joint{index}") for index in range(1, 7)]]
        finger = positions[[names.index(f"{side}_joint{index}") for index in (7, 8)]].mean()
        opened = torch.clamp(finger / self.config.gripper_open_position, 0.0, 1.0)
        return arm, opened

    def _hold_action(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Hold both physical I2RT arms at their observed joint state."""
        blocks = []
        for side in ("left", "right"):
            arm, opened = self._arm_state(env, side)
            blocks.append(torch.cat((arm, (1.0 - opened).reshape(1))))
        action = torch.cat(blocks)
        assert action.shape == (14,), f"Bimanual YAM action must have shape (14,), got {action.shape}."
        return action

    def _observation_frame(self, env: ManagerBasedRLEnv, action: torch.Tensor) -> dict[str, Any]:
        """Build the arm, camera, and workspace payload expected by cable CAP."""
        import time

        assert action.shape == (14,), f"Bimanual YAM action must have shape (14,), got {action.shape}."
        frame: dict[str, Any] = {"timestamp": time.time()}
        for wire_side, physical_side in zip(("left", "right"), self.config.wire_arm_order, strict=True):
            offset = 0 if physical_side == "left" else 7
            frame[wire_side] = {
                "joint_pos": [*action[offset : offset + 6].cpu().tolist(), 1.0 - float(action[offset + 6])]
            }
        for name, alias in self.config.camera_mapping.items():
            frame[alias] = self._camera(env, name)
        frame["_isaac_cap"] = {"workspace": dict(self.config.workspace)}
        return frame

    @staticmethod
    def _valid(reply: dict[str, Any], block: dict[str, Any], channel: str, wire_side: str) -> bool:
        """Read cable CAP's top-level per-arm validity flags."""
        value = block.get(channel, reply.get(channel, False))
        if isinstance(value, dict):
            value = value.get(wire_side, False)
        assert isinstance(value, (bool, np.bool_)), f"Invalid CAP validity flag: {channel}={value!r}"
        return bool(value)

    def _apply_reply(self, reply: dict[str, Any], action: torch.Tensor) -> None:
        """Map CAP wire arms back onto Arena's physical action-vector slots."""
        for wire_side, physical_side in zip(("left", "right"), self.config.wire_arm_order, strict=True):
            block = reply.get(wire_side, {})
            assert isinstance(block, dict), f"Invalid CAP arm reply: {wire_side}={block!r}"
            offset = 0 if physical_side == "left" else 7
            if self._valid(reply, block, "arm_valid", wire_side):
                target = np.asarray(block["joint_pos"], dtype=np.float32)
                assert target.shape == (6,) and np.isfinite(target).all()
                action[offset : offset + 6] = torch.as_tensor(target, device=action.device)
            if self._valid(reply, block, "gripper_valid", wire_side):
                opened = float(block["gripper"])
                assert np.isfinite(opened) and 0.0 <= opened <= 1.0
                action[offset + 6] = 1.0 - opened
        self._last_gripper = action[[6, 13]].clone()


__all__ = ["CapYamI2rtPolicy", "CapYamI2rtPolicyCfg"]
