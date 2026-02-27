# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pi0 client policy for DROID environments.


# TODO(cvolk)
/isaac-sim/python.sh -m isaaclab_arena.evaluation.policy_runner   --policy_type isaaclab_arena.policy.pi0_droid_client.Pi0DroidClient   --enable_cameras   --num_episodes 10   --remote_host localhost   --remote_port 8000   droid_pick_and_place_srl   --embodiment droid_rel_joint_pos


Thin client that connects directly to an openpi WebSocket server.
No custom server wrapper needed — uses the upstream pi0 server as-is.

pi0_fast_droid outputs joint *velocities* (7) + gripper position (1),
clipped to [-1, 1]. Use with ``droid_rel_joint_pos`` embodiment which
applies the first 7 dims as deltas from the current joint position.
"""

from __future__ import annotations

import argparse
import gymnasium as gym
import numpy as np
import torch
from dataclasses import dataclass
from gymnasium.spaces.dict import Dict as GymSpacesDict
from typing import Any

from openpi_client import image_tools, websocket_client_policy

from isaaclab_arena.policy.policy_base import PolicyBase


@dataclass
class Pi0DroidClientConfig:
    remote_host: str = "localhost"
    remote_port: int = 8000
    open_loop_horizon: int = 8
    image_size: int = 224
    external_camera_key: str = "external_camera_rgb"
    wrist_camera_key: str = "wrist_camera_rgb"


class Pi0DroidClient(PolicyBase):
    """Client-side policy that talks directly to an openpi WebSocket server.

    Designed for DROID embodiment with joint-position action space.
    Handles action chunking, image resizing, and gripper binarization locally.
    """

    config_class = Pi0DroidClientConfig

    def __init__(self, config: Pi0DroidClientConfig) -> None:
        super().__init__(config=config)
        self.cfg = config

        print(f"[Pi0DroidClient] Connecting to openpi server at {config.remote_host}:{config.remote_port}...")
        self.client = websocket_client_policy.WebsocketClientPolicy(config.remote_host, config.remote_port)
        print("[Pi0DroidClient] Server ready.")

        self._chunk: np.ndarray | None = None
        self._chunk_idx: int = 0
        self.task_description: str | None = None

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        if self._chunk is None or self._chunk_idx >= self.cfg.open_loop_horizon:
            self._chunk_idx = 0
            self._chunk = self._request_chunk(observation)

        action_np = self._chunk[self._chunk_idx].copy()
        self._chunk_idx += 1

        # pi0 outputs joint velocities clipped to [-1, 1] for the arm,
        # and a gripper position for the last dim.
        action_np[:7] = np.clip(action_np[:7], -1.0, 1.0)
        action_np[-1] = 1.0 if action_np[-1] > 0.5 else 0.0
        return torch.tensor(action_np, dtype=torch.float32, device="cuda").unsqueeze(0)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._chunk = None
        self._chunk_idx = 0

    def set_task_description(self, task_description: str | None) -> str:
        self.task_description = task_description
        return self.task_description or ""

    def _request_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        obs = self._extract_observation(observation)
        sz = self.cfg.image_size

        request = {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(obs["external_image"], sz, sz),
            "observation/wrist_image_left": image_tools.resize_with_pad(obs["wrist_image"], sz, sz),
            "observation/joint_position": obs["joint_position"],
            "observation/gripper_position": obs["gripper_position"],
            "prompt": self.task_description or "",
        }

        response = self.client.infer(request)
        return response["actions"]

    def _extract_observation(self, obs_dict: dict[str, Any], env_id: int = 0) -> dict[str, np.ndarray]:
        ext_image = obs_dict["camera_obs"][self.cfg.external_camera_key][env_id].detach().cpu().numpy()
        wrist_image = obs_dict["camera_obs"][self.cfg.wrist_camera_key][env_id].detach().cpu().numpy()
        joint_pos = obs_dict["policy"]["joint_pos"][env_id].detach().cpu().numpy()
        gripper_pos = obs_dict["policy"]["gripper_pos"][env_id].detach().cpu().numpy()

        return {
            "external_image": ext_image,
            "wrist_image": wrist_image,
            "joint_position": joint_pos,
            "gripper_position": gripper_pos,
        }

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group("Pi0 DROID Client")
        group.add_argument("--remote_host", type=str, default="localhost")
        group.add_argument("--remote_port", type=int, default=8000)
        group.add_argument("--open_loop_horizon", type=int, default=8)
        group.add_argument("--image_size", type=int, default=224)
        group.add_argument("--external_camera_key", type=str, default="external_camera_rgb")
        group.add_argument("--wrist_camera_key", type=str, default="wrist_camera_rgb")
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> Pi0DroidClient:
        config = Pi0DroidClientConfig(
            remote_host=args.remote_host,
            remote_port=args.remote_port,
            open_loop_horizon=args.open_loop_horizon,
            image_size=args.image_size,
            external_camera_key=args.external_camera_key,
            wrist_camera_key=args.wrist_camera_key,
        )
        return Pi0DroidClient(config)
