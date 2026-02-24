# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""VLN hierarchical policy: remote VLM + local RSL-RL locomotion.

This composite policy has two layers:
  - **High-level**: A remote VLM server (e.g. NaVILA, GR00T) that
    produces velocity commands ``[vx, vy, yaw_rate]`` from RGB images.
  - **Low-level**: A local RSL-RL locomotion policy that converts
    velocity commands into joint-position actions.

Works with Arena's ``policy_runner.py`` like any other policy.

Usage::

    python -m isaaclab_arena.evaluation.policy_runner \\
        --policy_type isaaclab_arena.policy.vln.vln_vlm_locomotion_policy.VlnVlmLocomotionPolicy \\
        --remote_host localhost --remote_port 5555 \\
        --ll_checkpoint_path /path/to/rsl_rl/model.pt \\
        --ll_agent_cfg /path/to/agent.yaml \\
        --num_episodes 10 \\
        VLN_Benchmark \\
        --usd_path /path/to/scene.usd \\
        --r2r_dataset_path /path/to/dataset.json.gz
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces.dict import Dict as GymSpacesDict
from PIL import Image as PILImage
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import load_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_arena.policy.client_side_policy import ClientSidePolicy
from isaaclab_arena.remote_policy.action_protocol import VlnVelocityActionProtocol
from isaaclab_arena.remote_policy.remote_policy_config import RemotePolicyConfig


class VlnVlmLocomotionPolicy(ClientSidePolicy):
    """VLN hierarchical policy: remote VLM + local RSL-RL locomotion.

    High-level layer queries a remote VLM server for velocity commands.
    Low-level layer runs a pre-trained RSL-RL locomotion policy to
    convert velocity commands into joint-position actions.

    Inherits from ``ClientSidePolicy`` to reuse the ZeroMQ handshake,
    observation packing, and remote lifecycle management.
    """

    def __init__(
        self,
        remote_config: RemotePolicyConfig,
        ll_checkpoint_path: str,
        ll_agent_cfg: str,
        device: str = "cuda",
        vel_cmd_obs_indices: tuple[int, int] = (9, 12),
        warmup_steps: int = 200,
        debug: bool = False,
        save_interval: int = 0,
        save_dir: str = "",
        ignore_vlm_stop: bool = False,
        min_vlm_stop_distance: float = -1.0,
    ):
        super().__init__(
            config=None,
            remote_config=remote_config,
            protocol_cls=VlnVelocityActionProtocol,
        )
        self._device = device
        self._vel_cmd_indices = vel_cmd_obs_indices
        self._warmup_steps = warmup_steps
        self._debug = debug
        self._save_interval = save_interval
        self._save_dir = save_dir
        self._ignore_vlm_stop = ignore_vlm_stop
        self._min_vlm_stop_distance = min_vlm_stop_distance

        # RSL-RL low-level policy (loaded lazily in first get_action)
        self._ll_checkpoint_path = ll_checkpoint_path
        self._ll_agent_cfg = ll_agent_cfg
        self._ll_policy = None
        self._ll_obs_td = None
        self._ll_vec_env = None

        # VLM scheduling state
        self._step_count: int = 0
        self._target_step: int = 0
        self._last_vel_cmd = np.zeros(self.action_dim, dtype=np.float32)
        self._env_dt: float | None = None

        # Track current instruction to detect episode changes
        self._current_instruction: str | None = None

        self._vlm_query_count: int = 0

    # ------------------------------------------------------------------ #
    # PolicyBase interface                                                #
    # ------------------------------------------------------------------ #

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        """Return joint-position actions for the environment.

        Internally: VLM query → velocity command → inject into obs →
        RSL-RL forward pass → joint actions.
        """
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
        self._last_unwrapped = unwrapped

        # Lazy init
        if self._ll_policy is None:
            self._load_low_level_policy(env)
            if self._debug:
                self._debug_save_dir = "/tmp/vln_debug_frames"
                os.makedirs(self._debug_save_dir, exist_ok=True)
            if self._save_interval > 0 and self._save_dir:
                ts = time.strftime("%Y%m%d_%H%M%S")
                self._save_dir = os.path.join(self._save_dir, ts)
                os.makedirs(self._save_dir, exist_ok=True)
                print(f"[VlnPolicy] Saving frames every {self._save_interval} steps to {self._save_dir}")

        if self._env_dt is None:
            try:
                self._env_dt = float(unwrapped.cfg.sim.dt * unwrapped.cfg.decimation)
            except Exception:
                self._env_dt = 0.02

        # Detect per-episode instruction changes from env.extras
        self._check_instruction_update(unwrapped)

        goal_dist = self._get_distance_to_goal(unwrapped)

        # Query VLM if scheduling says it's time
        if self._step_count >= self._target_step:
            packed_obs = self.pack_observation_for_server(observation)
            resp = self.remote_client.get_action(observation=packed_obs)

            vel_cmd = np.asarray(
                resp.get("action", np.zeros(self.action_dim)), dtype=np.float32
            )
            duration = float(resp.get("duration", self.protocol.default_duration))
            vlm_text = resp.get("vlm_text", "")
            self._last_vel_cmd = vel_cmd

            self._vlm_query_count += 1
            if self._debug:
                dist_str = f" dist={goal_dist:.2f}" if goal_dist is not None else ""
                print(
                    f"[VlnPolicy VLM #{self._vlm_query_count}] step={self._step_count}"
                    f" cmd={vel_cmd} dur={duration:.1f}s{dist_str}"
                    f" vlm=\"{vlm_text[-80:] if vlm_text else ''}\""
                )

            if self._env_dt > 0.0 and duration > 0.0:
                steps_to_hold = max(1, int(duration / self._env_dt))
            else:
                steps_to_hold = 1
            self._target_step = self._step_count + steps_to_hold

            # STOP: VLM returns zero velocity + zero duration
            if np.allclose(vel_cmd, 0.0) and duration <= 0.0:
                accept_stop, stop_reason = self._should_accept_vlm_stop(goal_dist)
                if accept_stop:
                    extras = getattr(unwrapped, "extras", {})
                    if "vln_stop_called" in extras:
                        extras["vln_stop_called"][:] = True
                elif self._debug:
                    print(
                        f"[VlnPolicy STOP ignored] step={self._step_count} "
                        f"dist={goal_dist if goal_dist is not None else 'unknown'} "
                        f"reason={stop_reason}"
                    )

                if self._debug:
                    self._save_debug_frame(observation, self._step_count)
                if self._save_dir:
                    self._save_frame_to_dir(observation, unwrapped)

            if self._debug and self._step_count % 200 == 0:
                self._save_debug_frame(observation, self._step_count)

        # Get the latest proprioceptive observation from the RSL-RL wrapper.
        self._ll_obs_td = self._ll_vec_env.get_observations()

        # Inject velocity command into the policy observation in-place
        i, j = self._vel_cmd_indices
        cmd_tensor = torch.tensor(self._last_vel_cmd, device=self._device, dtype=torch.float32)
        self._ll_obs_td["policy"][:, i:j] = cmd_tensor

        # Run low-level policy (pass full TensorDict, RSL-RL indexes by group)
        with torch.inference_mode():
            joint_actions = self._ll_policy(self._ll_obs_td)

        if self._debug and self._step_count % 500 == 0:
            try:
                pos = unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
                goal = unwrapped.extras.get("current_goal_pos")
                dist_str = ""
                if goal is not None:
                    goal_np = np.asarray(goal)
                    if goal_np.ndim == 2:
                        goal_np = goal_np[0]
                    dist_str = f" dist_to_goal={np.linalg.norm(pos - goal_np):.2f}"
                ep_id = ""
                ep_ids = unwrapped.extras.get("current_episode_id")
                if ep_ids is not None:
                    ep_id = f" ep={ep_ids[0]}"
                print(f"[VlnPolicy] step={self._step_count} robot=[{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.2f}] cmd={self._last_vel_cmd}{dist_str}{ep_id}")
            except Exception:
                pass

        if self._save_interval > 0 and self._save_dir and self._step_count % self._save_interval == 0:
            self._save_frame_to_dir(observation, unwrapped)

        self._step_count += 1
        return joint_actions

    def _save_debug_frame(self, observation, step):
        """Save camera images with distance and instruction overlay."""
        try:
            cam_dict = self._get_camera_obs_dict(observation, getattr(self, "_last_unwrapped", None))
            if cam_dict is None:
                return

            goal_dist = self._get_distance_to_goal(
                getattr(self, "_last_unwrapped", None)
            )
            instruction = self._current_instruction or ""

            for cam_name, cam_data in cam_dict.items():
                if "rgb" not in cam_name:
                    continue
                img_tensor = cam_data
                if hasattr(img_tensor, 'cpu'):
                    img_np = img_tensor[0].cpu().numpy() if img_tensor.ndim == 4 else img_tensor.cpu().numpy()
                else:
                    img_np = np.asarray(img_tensor)
                    if img_np.ndim == 4:
                        img_np = img_np[0]
                if img_np.dtype in (np.float32, np.float64):
                    img_np = np.clip(img_np * 255.0 if img_np.max() <= 1.5 else img_np, 0, 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)

                img = PILImage.fromarray(img_np)
                self._draw_overlay(img, goal_dist, instruction, step)

                tag = "head" if "head" in cam_name else "follow"
                path = f"{self._debug_save_dir}/{tag}_{step:06d}.png"
                img.save(path)
        except Exception as e:
            print(f"[VlnPolicy DEBUG] Failed to save frame: {e}")

    @staticmethod
    def _draw_overlay(img, goal_dist, instruction, step):
        """Draw distance and instruction text on a PIL image."""
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
            font_sm = font

        # Distance in red
        dist_text = f"Dist: {goal_dist:.2f}m" if goal_dist is not None else "Dist: --"
        draw.text((10, 10), dist_text, fill=(255, 50, 50), font=font)

        # Step counter
        draw.text((10, 35), f"Step: {step}", fill=(255, 255, 255), font=font_sm)

        # Instruction in red (wrap to fit image width)
        if instruction:
            max_chars = img.width // 8
            lines = [instruction[i:i + max_chars] for i in range(0, min(len(instruction), max_chars * 3), max_chars)]
            y = img.height - 20 * len(lines) - 5
            for line in lines:
                draw.text((10, y), line, fill=(255, 80, 80), font=font_sm)
                y += 18

    def _save_frame_to_dir(self, observation, unwrapped) -> None:
        """Save camera frames to the configured save directory."""
        try:
            cam_dict = self._get_camera_obs_dict(observation, unwrapped)
            if cam_dict is None:
                return
            goal_dist = self._get_distance_to_goal(unwrapped)
            instruction = self._current_instruction or ""
            try:
                pos = unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
                pos_str = f"[{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.2f}]"
            except Exception:
                pos_str = "?"
            ep_id = 0
            ep_ids = getattr(unwrapped, "extras", {}).get("current_episode_id")
            if ep_ids is not None:
                ep_id = ep_ids[0]
            for cam_name, cam_data in cam_dict.items():
                if "rgb" not in cam_name:
                    continue
                img_tensor = cam_data
                if hasattr(img_tensor, 'cpu'):
                    img_np = img_tensor[0].cpu().numpy() if img_tensor.ndim == 4 else img_tensor.cpu().numpy()
                else:
                    img_np = np.asarray(img_tensor)
                    if img_np.ndim == 4:
                        img_np = img_np[0]
                if img_np.dtype in (np.float32, np.float64):
                    img_np = np.clip(img_np * 255.0 if img_np.max() <= 1.5 else img_np, 0, 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)
                img = PILImage.fromarray(img_np)
                self._draw_overlay(img, goal_dist, instruction, self._step_count)
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(img)
                try:
                    font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
                except OSError:
                    font_sm = ImageFont.load_default()
                draw.text((10, 55), f"Pos: {pos_str}", fill=(255, 255, 255), font=font_sm)
                tag = "head" if "head" in cam_name else "follow"
                path = os.path.join(self._save_dir, f"ep{ep_id}_{tag}_{self._step_count:06d}.png")
                img.save(path)
        except Exception as e:
            print(f"[VlnPolicy] frame save error: {e}")

    def _get_camera_obs_dict(self, observation, unwrapped) -> dict[str, Any] | None:
        """Return a camera_obs dict from either the step observation or the env."""
        if isinstance(observation, dict):
            cam_dict = observation.get("camera_obs")
            if isinstance(cam_dict, dict):
                return cam_dict

        obs_manager = getattr(unwrapped, "observation_manager", None)
        if obs_manager is None:
            return None

        try:
            cam_dict = obs_manager.compute_group("camera_obs", update_history=False)
        except Exception:
            return None
        return cam_dict if isinstance(cam_dict, dict) else None

    def _should_accept_vlm_stop(self, goal_dist: float | None) -> tuple[bool, str]:
        """Return whether a zero-duration stop should terminate the episode."""
        reasons: list[str] = []
        if self._ignore_vlm_stop:
            reasons.append("ignore_vlm_stop")
        if (
            self._min_vlm_stop_distance >= 0.0
            and goal_dist is not None
            and goal_dist > self._min_vlm_stop_distance
        ):
            reasons.append(
                f"goal_dist={goal_dist:.2f} > min_vlm_stop_distance={self._min_vlm_stop_distance:.2f}"
            )
        return (not reasons, ", ".join(reasons) if reasons else "accepted")

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset VLM scheduling state and notify server."""
        super().reset(env_ids)
        self._step_count = 0
        self._target_step = 0
        self._last_vel_cmd[:] = 0.0
        self._current_instruction = None

    def set_task_description(self, task_description: str | None) -> str:
        """Forward task description to the VLM server."""
        self.task_description = task_description
        if task_description is not None:
            self.remote_client.call_endpoint(
                "set_task_description",
                data={"task_description": task_description},
                requires_input=True,
            )
        return self.task_description or ""

    # ------------------------------------------------------------------ #
    # Low-level policy loading                                            #
    # ------------------------------------------------------------------ #

    def _load_low_level_policy(self, env) -> None:
        """Load the RSL-RL locomotion policy and do warmup."""
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env

        if isinstance(env, RslRlVecEnvWrapper):
            vec_env = env
        else:
            vec_env = RslRlVecEnvWrapper(unwrapped)
        self._ll_vec_env = vec_env

        agent_cfg_dict = load_yaml(self._ll_agent_cfg)
        device = agent_cfg_dict.get("device", "cuda")

        runner = OnPolicyRunner(vec_env, agent_cfg_dict, log_dir=None, device=device)
        runner.load(self._ll_checkpoint_path)
        self._ll_policy = runner.get_inference_policy(device=vec_env.unwrapped.device)

        # Warmup
        self._ll_obs_td = vec_env.get_observations()
        zero_cmd = torch.zeros(self.action_dim, device=self._device)
        i, j = self._vel_cmd_indices

        print(f"[VlnPolicy] Warming up ({self._warmup_steps} steps)...")
        for step in range(self._warmup_steps):
            self._ll_obs_td["policy"][:, i:j] = zero_cmd
            with torch.inference_mode():
                actions = self._ll_policy(self._ll_obs_td)
            self._ll_obs_td = vec_env.step(actions)[0]
        print("[VlnPolicy] Warmup complete.")

    # ------------------------------------------------------------------ #
    # Distance helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_distance_to_goal(self, unwrapped) -> float | None:
        """Return Euclidean distance from robot to goal, or None."""
        try:
            pos = unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
            goal = unwrapped.extras.get("current_goal_pos")
            if goal is None:
                return None
            goal_np = np.asarray(goal)
            if goal_np.ndim == 2:
                goal_np = goal_np[0]
            return float(np.linalg.norm(pos - goal_np))
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Instruction tracking                                                #
    # ------------------------------------------------------------------ #

    def _check_instruction_update(self, unwrapped) -> None:
        """Detect per-episode instruction changes from env.extras."""
        extras = getattr(unwrapped, "extras", {})
        instruction = extras.get("current_instruction")
        if instruction is None:
            return
        if isinstance(instruction, list):
            instruction = instruction[0]

        if instruction != self._current_instruction:
            self._current_instruction = instruction
            self.remote_client.call_endpoint(
                "set_task_description",
                data={"task_description": instruction},
                requires_input=True,
            )
            self._step_count = 0
            self._target_step = 0
            self._last_vel_cmd[:] = 0.0

    # ------------------------------------------------------------------ #
    # CLI helpers                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = ClientSidePolicy.add_remote_args_to_parser(parser)

        ll = parser.add_argument_group("Low-Level Locomotion Policy")
        ll.add_argument(
            "--ll_checkpoint_path", type=str, required=True,
            help="Path to the RSL-RL checkpoint (e.g. model_0.pt).",
        )
        ll.add_argument(
            "--ll_agent_cfg", type=str, required=True,
            help="Path to the RSL-RL agent config YAML.",
        )
        ll.add_argument(
            "--warmup_steps", type=int, default=200,
            help="Low-level policy warmup steps (default: 200).",
        )
        ll.add_argument(
            "--policy_device", type=str, default="cuda",
            help="Device for policy inference (default: cuda).",
        )
        ll.add_argument(
            "--debug_vln", action="store_true", default=False,
            help="Enable VLN debug logging and camera frame saving (default: off).",
        )
        ll.add_argument(
            "--save_interval", type=int, default=0,
            help="Save camera frames every N steps to --save_dir (0=disabled).",
        )
        ll.add_argument(
            "--save_dir", type=str, default="",
            help="Directory for saved camera frames.",
        )
        ll.add_argument(
            "--ignore_vlm_stop",
            action="store_true",
            default=False,
            help="Diagnostic mode: never let VLM STOP terminate the episode.",
        )
        ll.add_argument(
            "--min_vlm_stop_distance",
            type=float,
            default=-1.0,
            help="Diagnostic mode: ignore STOP while distance-to-goal is above this threshold; negative disables.",
        )
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> VlnVlmLocomotionPolicy:
        remote_config = ClientSidePolicy.build_remote_config_from_args(args)
        return VlnVlmLocomotionPolicy(
            remote_config=remote_config,
            ll_checkpoint_path=args.ll_checkpoint_path,
            ll_agent_cfg=args.ll_agent_cfg,
            device=getattr(args, "policy_device", "cuda"),
            warmup_steps=getattr(args, "warmup_steps", 200),
            debug=getattr(args, "debug_vln", False),
            save_interval=getattr(args, "save_interval", 0),
            save_dir=getattr(args, "save_dir", ""),
            ignore_vlm_stop=getattr(args, "ignore_vlm_stop", False),
            min_vlm_stop_distance=getattr(args, "min_vlm_stop_distance", -1.0),
        )
