# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import io
import numpy as np
from typing import Any

import msgpack
import zmq
from PIL import Image

from isaaclab_arena.policy.inference_client import InferenceClient

# GR00T policy resolution
RESOLUTION = (180, 320)


def quat_to_euler_xyz(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw) in XYZ convention.

    Args:
        quat: Quaternion array of shape (..., 4) in (w, x, y, z) format.

    Returns:
        Euler angles array of shape (..., 3) in (roll, pitch, yaw) format.
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack([roll, pitch, yaw], axis=-1)


# ==============================================================================
# Minimal GR00T Policy Client (ZMQ-based)
# ==============================================================================


class _MsgSerializer:
    """Msgpack serializer with numpy array support."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=_MsgSerializer._encode)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=_MsgSerializer._decode)

    @staticmethod
    def _decode(obj):
        if isinstance(obj, dict) and "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def _encode(obj):
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


class GR00TPolicyClient:
    """Minimal ZMQ client for GR00T policy server."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        api_token: str | None = None,
    ):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.api_token = api_token
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def get_action(self, observation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Get action from the policy server."""
        request = {
            "endpoint": "get_action",
            "data": {"observation": observation, "options": None},
        }
        if self.api_token:
            request["api_token"] = self.api_token

        self.socket.send(_MsgSerializer.to_bytes(request))
        message = self.socket.recv()

        if message == b"ERROR":
            raise RuntimeError("Server error. Make sure the GR00T policy server is running.")

        response = _MsgSerializer.from_bytes(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")

        return tuple(response)  # (action_dict, info_dict)

    def ping(self) -> bool:
        """Check if the server is reachable."""
        try:
            request = {"endpoint": "ping"}
            if self.api_token:
                request["api_token"] = self.api_token
            self.socket.send(_MsgSerializer.to_bytes(request))
            self.socket.recv()
            return True
        except zmq.error.ZMQError:
            return False

    def __del__(self):
        self.socket.close()
        self.context.term()


# ==============================================================================
# Image utilities
# ==============================================================================


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Resizes images to target size with padding to preserve aspect ratio."""
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> np.ndarray:
    """Resize single image with padding."""
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return np.array(image)

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    return np.array(zero_image)


# ==============================================================================
# GR00T Franka Client (InferenceClient for IsaacLab Arena)
# ==============================================================================


class Gr00tFrankaClient(InferenceClient):
    """Inference client for GR00T policy on DROID (Franka + Robotiq) with joint position action space.

    Connects to a remote GR00T policy server via ZMQ, sends observations
    (images + proprioception), and returns joint-position actions with
    open-loop action chunking.
    """

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 5555,
        open_loop_horizon: int = 10,
        api_token: str | None = None,
    ) -> None:
        print(f"[{self.__class__.__name__}] Connecting to GR00T policy server at {remote_host}:{remote_port}...")
        self.client = GR00TPolicyClient(
            host=remote_host,
            port=remote_port,
            api_token=api_token,
        )
        print(f"[{self.__class__.__name__}] Connected to GR00T policy server.")

        self.open_loop_horizon = open_loop_horizon
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None

    def reset(self):
        """Reset the client state for a new episode."""
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None

    def infer(self, obs: dict, instruction: str) -> dict:
        """Infer the next action from the GR00T policy.

        Args:
            obs: Observation dictionary from IsaacLab Arena environment containing
                 camera_obs and policy groups.
            instruction: Language instruction for the task.

        Returns:
            Dictionary with 'action' (np.ndarray) and 'viz' (np.ndarray) keys.
        """
        curr_obs = self._extract_observation(obs)

        # Query the policy server if we need a new action chunk
        if self.actions_from_chunk_completed == 0 or self.actions_from_chunk_completed >= self.open_loop_horizon:
            self.actions_from_chunk_completed = 0

            # Resize images to the resolution expected by GR00T
            ext_image = resize_with_pad(curr_obs["external_image"], RESOLUTION[0], RESOLUTION[1])
            wrist_image = resize_with_pad(curr_obs["wrist_image"], RESOLUTION[0], RESOLUTION[1])

            # Prepare request data in GR00T format
            # GR00T expects: [B, T, H, W, C] for video, [B, T, D] for state
            request_data = {
                "video.exterior_image_1_left": ext_image[None, None, ...],  # [1, 1, H, W, C]
                "video.wrist_image_left": wrist_image[None, None, ...],  # [1, 1, H, W, C]
                "state.eef_position": curr_obs["eef_position"][None, None, ...],  # [1, 1, 3]
                "state.eef_rotation": curr_obs["eef_euler"][None, None, ...],  # [1, 1, 3]
                "state.joint_position": curr_obs["joint_position"][None, None, ...].astype(np.float32),
                "state.gripper_position": curr_obs["gripper_position"][None, None, ...].astype(np.float32),
                "annotation.language.language_instruction": [instruction],
                "annotation.language.language_instruction_2": [instruction],
                "annotation.language.language_instruction_3": [instruction],
            }

            # Get action from policy server
            response = self.client.get_action(request_data)
            # Response: (action_dict, info_dict)
            action_dict = response[0]
            joint_action = action_dict["action.joint_position"][0]  # [N, 7]
            gripper_action = action_dict["action.gripper_position"][0]  # [N, 1]
            self.pred_action_chunk = np.concatenate([joint_action, gripper_action], axis=1)  # [N, 8]

        # Select current action from chunk
        action = self.pred_action_chunk[self.actions_from_chunk_completed]
        self.actions_from_chunk_completed += 1

        # Binarize gripper action
        if action[-1].item() > 0.5:
            action = np.concatenate([action[:-1], np.ones((1,))])
        else:
            action = np.concatenate([action[:-1], np.zeros((1,))])

        # Create visualization
        ext_img = resize_with_pad(curr_obs["external_image"], RESOLUTION[0], RESOLUTION[1])
        wrist_img = resize_with_pad(curr_obs["wrist_image"], RESOLUTION[0], RESOLUTION[1])
        viz = np.concatenate([ext_img, wrist_img], axis=1)

        return {"action": action, "viz": viz}

    def _extract_observation(self, obs_dict: dict, *, env_id: int = 0) -> dict:
        """Extract and format observation from the IsaacLab Arena environment.

        Maps IsaacLab Arena observation structure to the internal format:
          - camera_obs/external_camera_rgb  → external_image
          - camera_obs/wrist_camera_rgb     → wrist_image
          - policy/joint_pos               → joint_position
          - policy/gripper_pos             → gripper_position
          - policy/eef_pos                 → eef_position
          - policy/eef_quat                → eef_euler (converted)
        """
        # Extract images
        external_image = obs_dict["camera_obs"]["external_camera_rgb"][env_id].clone().detach().cpu().numpy()
        wrist_image = obs_dict["camera_obs"]["wrist_camera_rgb"][env_id].clone().detach().cpu().numpy()

        # Extract proprioceptive state
        proprio = obs_dict["policy"]
        joint_position = proprio["joint_pos"][env_id].clone().detach().cpu().numpy()
        gripper_position = proprio["gripper_pos"][env_id].clone().detach().cpu().numpy()

        # Extract EEF pose and convert quat to euler
        eef_position = proprio["eef_pos"][env_id].clone().detach().cpu().numpy()
        eef_quat = proprio["eef_quat"][env_id].clone().detach().cpu().numpy()
        eef_euler = quat_to_euler_xyz(eef_quat)

        return {
            "external_image": external_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
            "eef_position": eef_position,
            "eef_euler": eef_euler,
        }
