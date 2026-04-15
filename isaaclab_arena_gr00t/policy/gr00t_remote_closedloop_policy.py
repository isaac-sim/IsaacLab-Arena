# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T remote closed-loop policy using GR00T's native PolicyClient.

This policy connects to a GR00T policy server (launched via
``gr00t/eval/run_gr00t_server.py``) and reuses the same observation/action
translation pipeline as the local ``Gr00tClosedloopPolicy``.
"""

from __future__ import annotations

import argparse
import gymnasium as gym
import torch
from dataclasses import dataclass, field
from typing import Any

from gr00t.policy.server_client import PolicyClient as Gr00tPolicyClient

from isaaclab_arena.policy.action_chunking import ActionChunkScheduler
from isaaclab_arena.policy.action_scheduler import ActionScheduler
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena.utils.multiprocess import get_local_rank, get_world_size
from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import Gr00tClosedloopPolicyConfig, TaskMode
from isaaclab_arena_gr00t.policy.gr00t_core import (
    Gr00tBasePolicyArgs,
    build_gr00t_action_tensor,
    build_gr00t_policy_observations,
    compute_action_dim,
    extract_obs_numpy_from_torch,
    load_gr00t_joint_configs,
)
from isaaclab_arena_gr00t.utils.io_utils import (
    create_config_from_yaml,
    load_gr00t_modality_config_from_file,
)


@dataclass
class Gr00tRemoteClosedloopPolicyArgs(Gr00tBasePolicyArgs):
    """Configuration for Gr00tRemoteClosedloopPolicy.

    Inherits policy_config_yaml_path and policy_device from Gr00tBasePolicyArgs,
    and adds remote server connection parameters and num_envs.
    """

    num_envs: int = field(default=1, metadata={"help": "Number of environments to simulate"})
    remote_host: str = field(default="localhost", metadata={"help": "GR00T policy server hostname"})
    remote_port: int = field(default=5555, metadata={"help": "GR00T policy server port"})
    remote_api_token: str | None = field(default=None, metadata={"help": "API token for the policy server"})

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> Gr00tRemoteClosedloopPolicyArgs:
        """Create configuration from parsed CLI arguments."""
        return cls(
            policy_config_yaml_path=args.policy_config_yaml_path,
            policy_device=args.policy_device,
            num_envs=args.num_envs,
            remote_host=args.remote_host,
            remote_port=args.remote_port,
            remote_api_token=getattr(args, "remote_api_token", None),
        )


class Gr00tRemoteClosedloopPolicy(PolicyBase):
    """GR00T closed-loop policy that delegates inference to a remote GR00T server.

    Uses GR00T's native ``PolicyClient`` (from ``gr00t.policy.server_client``)
    to communicate with a GR00T policy server. The observation/action translation
    pipeline is identical to the local ``Gr00tClosedloopPolicy``.

    Server side (run independently):
        python gr00t/eval/run_gr00t_server.py \\
          --model_path nvidia/GR00T-N1.6-DROID \\
          --embodiment_tag OXE_DROID --device cuda --host 0.0.0.0 --port 5555

    Client side (Arena evaluation):
        python policy_runner.py \\
          --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \\
          --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/droid_manip_gr00t_closedloop_config.yaml \\
          --remote_host 10.0.0.1 --remote_port 5555 \\
          --enable_cameras --num_episodes 5 \\
          pick_and_place_maple_table --embodiment droid_abs_joint_pos
    """

    name = "gr00t_remote_closedloop"
    config_class = Gr00tRemoteClosedloopPolicyArgs

    def __init__(self, config: Gr00tRemoteClosedloopPolicyArgs, action_scheduler: ActionScheduler | None = None):
        super().__init__(config)

        # Policy config (for obs/action translation — no model loading)
        self.policy_config: Gr00tClosedloopPolicyConfig = create_config_from_yaml(
            config.policy_config_yaml_path, Gr00tClosedloopPolicyConfig
        )
        self.num_envs = config.num_envs
        self.device = config.policy_device
        if get_world_size() > 1 and "cuda" in self.device:
            self.device = f"cuda:{get_local_rank()}"
        self.task_mode = TaskMode(self.policy_config.task_mode_name)

        # Joint configs (for sim↔policy joint remapping)
        (
            self.policy_joints_config,
            self.robot_action_joints_config,
            self.robot_state_joints_config,
        ) = load_gr00t_joint_configs(self.policy_config)

        # Modality config (for building GR00T observation dicts)
        self.modality_configs = load_gr00t_modality_config_from_file(
            self.policy_config.modality_config_path,
            self.policy_config.embodiment_tag,
        )

        # Action / chunk shapes
        self.action_dim = compute_action_dim(self.task_mode, self.robot_action_joints_config)
        self.action_chunk_length = self.policy_config.action_chunk_length

        if action_scheduler is None:
            action_scheduler = ActionChunkScheduler(
                num_envs=self.num_envs,
                action_chunk_length=self.action_chunk_length,
                action_horizon=self.policy_config.action_horizon,
                action_dim=self.action_dim,
                device=self.device,
                dtype=torch.float,
            )
        self._action_scheduler = action_scheduler

        # Connect to GR00T's native policy server
        self._client = Gr00tPolicyClient(
            host=config.remote_host,
            port=config.remote_port,
            api_token=config.remote_api_token,
            strict=False,
        )
        if not self._client.ping():
            raise ConnectionError(
                f"Cannot reach GR00T policy server at {config.remote_host}:{config.remote_port}"
            )

        self.task_description: str | None = None

    # ---------------------- CLI helpers -------------------

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            "Gr00t Remote Closedloop Policy",
            "Arguments for GR00T remote closed-loop policy evaluation.",
        )
        group.add_argument(
            "--policy_config_yaml_path",
            type=str,
            required=True,
            help="Path to the Gr00t closedloop policy config YAML file",
        )
        group.add_argument(
            "--policy_device",
            type=str,
            default="cuda",
            help="Device for Arena-side tensor operations (default: cuda)",
        )
        group.add_argument("--remote_host", type=str, default="localhost", help="GR00T policy server hostname")
        group.add_argument("--remote_port", type=int, default=5555, help="GR00T policy server port")
        group.add_argument("--remote_api_token", type=str, default=None, help="API token for the policy server")
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> Gr00tRemoteClosedloopPolicy:
        config = Gr00tRemoteClosedloopPolicyArgs.from_cli_args(args)
        return Gr00tRemoteClosedloopPolicy(config)

    # ---------------------- Policy interface -------------------

    def set_task_description(self, task_description: str | None) -> str:
        if task_description is None:
            task_description = self.policy_config.language_instruction
        if not task_description:
            raise ValueError(
                "No language instruction provided. Set 'language_instruction' in the job config, "
                "pass --language_instruction on the CLI, or define 'task_description' on the task class."
            )
        self.task_description = task_description
        return self.task_description

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        def fetch_chunk() -> torch.Tensor:
            return self._get_action_chunk(observation, self.policy_config.pov_cam_name_sim)

        return self._action_scheduler.get_action(fetch_chunk)

    def _get_action_chunk(
        self, observation: dict[str, Any], camera_names: list[str] | str = "robot_head_cam_rgb"
    ) -> torch.Tensor:
        """Get an action chunk from the remote GR00T server.

        Same pipeline as Gr00tClosedloopPolicy.get_action_chunk(), but calls
        GR00T's PolicyClient instead of a local Gr00tPolicy.
        """
        if isinstance(camera_names, str):
            camera_names = [camera_names]

        # 1. Reuse the same obs translation as local policy
        assert self.task_description is not None, "Task description is not set"
        rgb_list_np, joint_pos_sim_np = extract_obs_numpy_from_torch(nested_obs=observation, camera_names=camera_names)
        policy_observations = build_gr00t_policy_observations(
            rgb_list_np=rgb_list_np,
            joint_pos_sim_np=joint_pos_sim_np,
            task_description=self.task_description,
            policy_config=self.policy_config,
            robot_state_joints_config=self.robot_state_joints_config,
            policy_joints_config=self.policy_joints_config,
            modality_configs=self.modality_configs,
        )

        # 2. Call GR00T's own client
        robot_action_policy, _ = self._client.get_action(policy_observations)

        # 3. Reuse the same action translation as local policy
        action_tensor = build_gr00t_action_tensor(
            robot_action_policy=robot_action_policy,
            task_mode=self.task_mode,
            policy_joints_config=self.policy_joints_config,
            robot_action_joints_config=self.robot_action_joints_config,
            device=self.device,
            embodiment_tag=self.policy_config.embodiment_tag,
        )

        assert action_tensor.shape[0] == self.num_envs and action_tensor.shape[1] >= self.action_chunk_length
        return action_tensor

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._client.reset()
        self._action_scheduler.reset(env_ids)
