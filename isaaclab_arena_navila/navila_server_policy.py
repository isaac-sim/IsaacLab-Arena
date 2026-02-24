# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""NaVILA server-side policy for VLN navigation.

This policy runs on the **server** side (typically a GPU machine with the
NaVILA / LLaVA model loaded).  It receives RGB image observations from the
client, runs VLM inference, parses the text output into a velocity command,
and returns it.

To add a different VLM backend (e.g. GR00T), create a new server policy
in ``isaaclab_arena_navila/`` following this pattern.

Launch via the standard Arena remote policy server runner::

    python -m isaaclab_arena.remote_policy.remote_policy_server_runner \\
        --policy_type isaaclab_arena_navila.navila_server_policy.NaVilaServerPolicy \\
        --model_path /path/to/navila/checkpoint \\
        --port 5555
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_image,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model

from isaaclab_arena.remote_policy.action_protocol import ActionProtocol, VlnVelocityActionProtocol
from isaaclab_arena.remote_policy.server_side_policy import ServerSidePolicy


# ========================================================================== #
# VLM command parsing                                                        #
# ========================================================================== #


def parse_vlm_output_to_velocity(text: str) -> tuple[np.ndarray, float]:
    """Parse VLM text output into a velocity command and duration.

    Recognized commands (case-insensitive):
      - "turn left [15|30|45]"  -> [0, 0, +pi/6], duration 0.5-1.5s
      - "turn right [15|30|45]" -> [0, 0, -pi/6], duration 0.5-1.5s
      - "move forward [25|50|75]" -> [0.5, 0, 0], duration 0.5-1.5s
      - "stop" -> [0, 0, 0], duration 0.0

    Returns:
        (vel_cmd, duration) where vel_cmd is [vx, vy, yaw_rate].
    """
    t = text.lower().strip()

    if "turn left" in t:
        vx, vy, wz = 0.0, 0.0, np.pi / 6.0
        if "45" in t:
            dur = 1.5
        elif "30" in t:
            dur = 1.0
        else:
            dur = 0.5
        return np.array([vx, vy, wz], dtype=np.float32), dur

    if "turn right" in t:
        vx, vy, wz = 0.0, 0.0, -np.pi / 6.0
        if "45" in t:
            dur = 1.5
        elif "30" in t:
            dur = 1.0
        else:
            dur = 0.5
        return np.array([vx, vy, wz], dtype=np.float32), dur

    if "move forward" in t or "move" in t:
        vx, vy, wz = 0.5, 0.0, 0.0
        if "75" in t:
            dur = 1.5
        elif "50" in t:
            dur = 1.0
        elif "25" in t:
            dur = 0.5
        else:
            dur = 0.5
        return np.array([vx, vy, wz], dtype=np.float32), dur

    if "stop" in t:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.0

    return np.array([0.5, 0.0, 0.0], dtype=np.float32), 0.5


# ========================================================================== #
# Server-side policy                                                         #
# ========================================================================== #


@dataclass
class NaVilaServerPolicyConfig:
    """Configuration for :class:`NaVilaServerPolicy`."""

    model_path: str = ""
    device: str = "cuda"
    num_video_frames: int = 8
    conv_mode: str = "llama_3"
    history_padding_mode: str = "repeat_first"
    max_history_frames: int = 200
    max_new_tokens: int = 80


class NaVilaServerPolicy(ServerSidePolicy):
    """NaVILA VLM server policy for VLN navigation.

    Wraps the NaVILA (LLaVA-based) model to generate navigation
    velocity commands from RGB image observations.  Maintains a
    full episode image history and uses uniform sampling (matching
    the NaVILA-Bench evaluation protocol) so the VLM can determine
    task completion and output "stop".
    """

    config_class = NaVilaServerPolicyConfig

    def __init__(self, config: NaVilaServerPolicyConfig) -> None:
        super().__init__(config)
        self._instruction: str = "Navigate to the target."
        self._image_history: list[Image.Image] = []
        self._num_video_frames: int = config.num_video_frames
        self._device = config.device
        self._conv_mode = config.conv_mode
        self._model_path = config.model_path
        self._history_padding_mode = config.history_padding_mode
        self._max_history_frames = config.max_history_frames
        self._max_new_tokens = config.max_new_tokens
        if self._history_padding_mode not in {"repeat_first", "black"}:
            raise ValueError(
                "history_padding_mode must be one of: repeat_first, black. "
                f"Got: {self._history_padding_mode!r}"
            )
        if self._max_history_frames <= 0:
            raise ValueError(f"max_history_frames must be > 0. Got: {self._max_history_frames}")
        if self._max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be > 0. Got: {self._max_new_tokens}")

        print(f"[NaVilaServerPolicy] Loading VLM model from: {self._model_path}")
        model_name = get_model_name_from_path(self._model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            self._model_path, model_name, None
        )
        self._tokenizer = tokenizer
        self._model = model.to(self._device)
        self._image_processor = image_processor
        print("[NaVilaServerPolicy] VLM model loaded successfully.")

    # ------------------------------------------------------------------ #
    # Protocol                                                            #
    # ------------------------------------------------------------------ #

    def _build_protocol(self) -> ActionProtocol:
        return VlnVelocityActionProtocol(
            action_dim=3,
            observation_keys=["camera_obs.robot_head_cam_rgb"],
            default_duration=0.5,
        )

    # ------------------------------------------------------------------ #
    # RPC methods                                                         #
    # ------------------------------------------------------------------ #

    def set_task_description(self, task_description: str | None) -> dict[str, Any]:
        self._instruction = task_description or "Navigate to the target."
        print(f"[NaVilaServerPolicy] Instruction: {self._instruction[:80]}...")
        return {"task_description": self._instruction}

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        nested = self.unpack_observation(observation)
        rgb_np = None
        if "camera_obs" in nested:
            cam_obs = nested["camera_obs"]
            for cam_key in cam_obs:
                cam_data = cam_obs[cam_key]
                if isinstance(cam_data, dict) and "rgb" in cam_data:
                    rgb_np = cam_data["rgb"]
                elif isinstance(cam_data, np.ndarray):
                    rgb_np = cam_data
                break

        if rgb_np is not None:
            if rgb_np.ndim == 4:
                rgb_np = rgb_np[0]
            img = Image.fromarray(rgb_np[:, :, :3].astype(np.uint8))
            self._image_history.append(img)
            if len(self._image_history) > self._max_history_frames:
                self._image_history = self._image_history[-self._max_history_frames:]

        vlm_text = self._run_vlm_inference()
        vel_cmd, duration = parse_vlm_output_to_velocity(vlm_text)

        action = {"action": vel_cmd, "duration": duration}
        info = {"vlm_text": vlm_text}
        return action, info

    def reset(self, env_ids=None, reset_options=None) -> dict[str, Any]:
        self._image_history.clear()
        return {"status": "reset_success"}

    # ------------------------------------------------------------------ #
    # VLM inference                                                       #
    # ------------------------------------------------------------------ #

    def _run_vlm_inference(self) -> str:
        """Run VLM forward pass on the image history + instruction.

        Matches NaVILA-Bench: uniformly sample ``num_video_frames - 1``
        frames from the *full* episode history, plus the latest frame.
        This gives the VLM global temporal context so it can determine
        when the navigation task is complete and output "stop".
        """
        if not self._image_history:
            return "move forward 50"

        n = self._num_video_frames
        num_images = len(self._image_history)

        if num_images < n:
            if self._history_padding_mode == "black":
                # Match NaVILA-Bench exactly for apples-to-apples diagnosis.
                pad_img = Image.new("RGB", self._image_history[-1].size, (0, 0, 0))
            else:
                # Repeating the first real image often gives a steadier early
                # rollout than feeding synthetic black frames.
                pad_img = self._image_history[0]
            frames = [pad_img] * (n - num_images) + list(self._image_history)
        else:
            # Uniform sample n-1 from full history + latest (matching NaVILA)
            n_hist = n - 1
            indices = [int(i * (num_images - 1) / n_hist) for i in range(n_hist)]
            frames = [self._image_history[i] for i in indices]
            frames.append(self._image_history[-1])

        self._model.config.image_processor = self._image_processor
        processed = []
        for img in frames:
            p = process_image(img, self._model.config, None)
            processed.append(p)
        if all(x.shape == processed[0].shape for x in processed):
            image_tensor = torch.stack(processed, dim=0).to(self._device, dtype=torch.float16)
        else:
            image_tensor = processed[0].unsqueeze(0).to(self._device, dtype=torch.float16)

        conv = conv_templates[self._conv_mode].copy()
        image_token = "<image>\n"
        n_hist = self._num_video_frames - 1
        qs = (
            f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
            f'of historical observations {image_token * n_hist}, and current observation <image>\n. '
            f'Your assigned task is: "{self._instruction}" '
            f"Analyze this series of images to decide your next action, which could be turning left "
            f"or right by a specific degree, moving forward a certain distance, or stop if the task "
            f"is completed."
        )
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self._device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self._tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                images=[image_tensor],
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=self._max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        output_text = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(f"[VLM] history={len(self._image_history)} output=\"{output_text[-120:]}\"")
        return output_text

    # ------------------------------------------------------------------ #
    # CLI helpers                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group("VLN Server-Side Policy")
        group.add_argument(
            "--model_path", type=str, required=True,
            help="Path to the NaVILA / LLaVA model checkpoint.",
        )
        group.add_argument(
            "--policy_device", type=str, default="cuda",
            help="Device for VLM inference (default: cuda).",
        )
        group.add_argument(
            "--num_video_frames", type=int, default=8,
            help="Number of video frames to send to the VLM (default: 8).",
        )
        group.add_argument(
            "--conv_mode", type=str, default="llama_3",
            help="Conversation template mode for LLaVA (default: llama_3).",
        )
        group.add_argument(
            "--history_padding_mode",
            type=str,
            default="repeat_first",
            choices=("repeat_first", "black"),
            help="How to pad early frame history before enough frames are collected.",
        )
        group.add_argument(
            "--max_history_frames",
            type=int,
            default=200,
            help="Maximum number of RGB frames kept in server-side history.",
        )
        group.add_argument(
            "--max_new_tokens",
            type=int,
            default=80,
            help="Maximum generated tokens per VLM query.",
        )
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> NaVilaServerPolicy:
        config = NaVilaServerPolicyConfig(
            model_path=args.model_path,
            device=getattr(args, "policy_device", "cuda"),
            num_video_frames=getattr(args, "num_video_frames", 8),
            conv_mode=getattr(args, "conv_mode", "llama_3"),
            history_padding_mode=getattr(args, "history_padding_mode", "repeat_first"),
            max_history_frames=getattr(args, "max_history_frames", 200),
            max_new_tokens=getattr(args, "max_new_tokens", 80),
        )
        return NaVilaServerPolicy(config)
