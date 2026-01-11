# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from gr00t.experiment.data_config import DATA_CONFIG_MAP, load_data_config
from gr00t.model.policy import Gr00tPolicy

from isaaclab_arena.remote_policy.server_side_policy import ServerSidePolicy
from isaaclab_arena.remote_policy.action_protocol import ActionMode, ChunkingActionProtocol
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.policy_constants import (
    NUM_BASE_HEIGHT_CMD,
    NUM_NAVIGATE_CMD,
    NUM_TORSO_ORIENTATION_RPY_CMD,
)

from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import Gr00tClosedloopPolicyConfig, TaskMode
from isaaclab_arena_gr00t.utils.image_conversion import resize_frames_with_padding
from isaaclab_arena_gr00t.utils.io_utils import create_config_from_yaml, load_robot_joints_config_from_yaml
from isaaclab_arena_gr00t.utils.joints_conversion import (
    remap_policy_joints_to_sim_joints,
    remap_sim_joints_to_policy_joints,
)
from isaaclab_arena_gr00t.utils.robot_joints import JointsAbsPosition


class Gr00tRemoteServerSidePolicy(ServerSidePolicy):
    """Server-side wrapper around Gr00tPolicy."""

    def __init__(self, policy_config_yaml_path: Path) -> None:
        super().__init__()

        print(f"[Gr00tRemoteServerSidePolicy] loading config from: {policy_config_yaml_path}")
        self.policy_config = create_config_from_yaml(policy_config_yaml_path, Gr00tClosedloopPolicyConfig)
        print(
            "[Gr00tRemoteServerSidePolicy] config:\n"
            f"  model_path        = {self.policy_config.model_path}\n"
            f"  embodiment_tag    = {self.policy_config.embodiment_tag}\n"
            f"  task_mode_name    = {self.policy_config.task_mode_name}\n"
            f"  data_config       = {self.policy_config.data_config}\n"
            f"  action_horizon    = {self.policy_config.action_horizon}\n"
            f"  action_chunk_len  = {self.policy_config.action_chunk_length}\n"
            f"  pov_cam_name_sim  = {self.policy_config.pov_cam_name_sim}\n"
            f"  policy_device     = {self.policy_config.policy_device}\n"
        )

        self.device = self.policy_config.policy_device
        self.task_mode = TaskMode(self.policy_config.task_mode_name)

        # Joints configs
        self.policy_joints_config = load_robot_joints_config_from_yaml(
            self.policy_config.policy_joints_config_path
        )
        self.robot_action_joints_config = load_robot_joints_config_from_yaml(
            self.policy_config.action_joints_config_path
        )
        self.robot_state_joints_config = load_robot_joints_config_from_yaml(
            self.policy_config.state_joints_config_path
        )

        # Action dimension
        self.action_dim = self._compute_action_dim()
        # Chunk length
        self.action_chunk_length = self.policy_config.action_chunk_length
        self.required_observation_keys: List[str] = [
            f"camera_obs.{self.policy_config.pov_cam_name_sim}",
            "policy.robot_joint_pos",
        ]

        # Underlying GR00T policy
        self.policy = self._load_local_policy()
        print("[Gr00tRemoteServerSidePolicy] Gr00tPolicy loaded successfully")

        # Task description will be set via set_task_description RPC
        self._task_description: str | None = None

    # ------------ protocol ------------

    def _build_protocol(self) -> ChunkingActionProtocol:
        obs_keys = [
            f"camera_obs.{self.policy_config.pov_cam_name_sim}",
            "policy.robot_joint_pos",
        ]
        proto = ChunkingActionProtocol(
            action_dim=self.action_dim,
            observation_keys=obs_keys,
            action_chunk_length=self.action_chunk_length,
        )
        print(f"[Gr00tRemoteServerSidePolicy] protocol mode = {proto.mode.value}")
        return proto

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #

    def _compute_action_dim(self) -> int:
        action_dim = len(self.robot_action_joints_config)
        if self.task_mode == TaskMode.G1_LOCOMANIPULATION:
            action_dim += NUM_NAVIGATE_CMD + NUM_BASE_HEIGHT_CMD + NUM_TORSO_ORIENTATION_RPY_CMD
        return action_dim

    def _load_local_policy(self) -> Gr00tPolicy:
        print(f"[Gr00tRemoteServerSidePolicy] loading data_config={self.policy_config.data_config}")
        if self.policy_config.data_config in DATA_CONFIG_MAP:
            data_config = DATA_CONFIG_MAP[self.policy_config.data_config]
        elif self.policy_config.data_config == "unitree_g1_sim_wbc":
            data_config = load_data_config("isaaclab_arena_gr00t.data_config:UnitreeG1SimWBCDataConfig")
        else:
            raise ValueError(f"Invalid data config: {self.policy_config.data_config}")

        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        model_path = Path(self.policy_config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        print(f"[Gr00tRemoteServerSidePolicy] loading checkpoint from: {model_path}")

        policy = Gr00tPolicy(
            model_path=str(model_path),
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=self.policy_config.embodiment_tag,
            denoising_steps=self.policy_config.denoising_steps,
            device=self.policy_config.policy_device,
        )
        return policy

    def _build_policy_observations(
        self,
        observation: Dict[str, Any],
        camera_name: str,
    ) -> Dict[str, Any]:
        # Expect observation to contain:
        # - "camera_obs"[camera_name]: (num_envs, H, W, C) as torch.Tensor on device
        # - "policy"]["robot_joint_pos"]: (num_envs, num_joints) as torch.Tensor on device
        observation = self.unpack_observation(observation)
        rgb = observation["camera_obs"][camera_name]
        num_envs = rgb.shape[0]
        if rgb.shape[1:3] != self.policy_config.target_image_size[:2]:
            rgb = resize_frames_with_padding(
                rgb,
                target_image_size=self.policy_config.target_image_size,
                bgr_conversion=False,
                pad_img=True,
            )

        joint_pos_sim = observation["policy"]["robot_joint_pos"]
        joint_pos_state_sim = JointsAbsPosition(joint_pos_sim, self.robot_state_joints_config)
        joint_pos_state_policy = remap_sim_joints_to_policy_joints(
            joint_pos_state_sim,
            self.policy_joints_config,
        )

        assert self._task_description is not None, "Task description is not set"
        policy_observations: Dict[str, Any] = {
            "annotation.human.task_description": [self._task_description] * num_envs,
            "video.ego_view": rgb.reshape(
                num_envs,
                1,
                self.policy_config.target_image_size[0],
                self.policy_config.target_image_size[1],
                self.policy_config.target_image_size[2],
            ),
            "state.left_arm": joint_pos_state_policy["left_arm"].reshape(num_envs, 1, -1),
            "state.right_arm": joint_pos_state_policy["right_arm"].reshape(num_envs, 1, -1),
            "state.left_hand": joint_pos_state_policy["left_hand"].reshape(num_envs, 1, -1),
            "state.right_hand": joint_pos_state_policy["right_hand"].reshape(num_envs, 1, -1),
        }
        if self.task_mode == TaskMode.G1_LOCOMANIPULATION:
            policy_observations["state.waist"] = joint_pos_state_policy["waist"].reshape(num_envs, 1, -1)
        return policy_observations

    # ------------------------------------------------------------------ #
    # ServerSidePolicy interface
    # ------------------------------------------------------------------ #

    def set_task_description(self, task_description: str | None) -> Dict[str, Any]:
        if task_description is None:
            task_description = self.policy_config.language_instruction
        self._task_description = task_description
        return {"status": "ok"}

    def get_action(
        self,
        observation: Dict[str, Any],
        options: dict[str, Any] | None = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        camera_name = self.policy_config.pov_cam_name_sim

        policy_observations = self._build_policy_observations(observation, camera_name)
        robot_action_policy = self.policy.get_action(policy_observations)

        robot_action_sim = remap_policy_joints_to_sim_joints(
            robot_action_policy,
            self.policy_joints_config,
            self.robot_action_joints_config,
            self.device,
        )

        if self.task_mode == TaskMode.G1_LOCOMANIPULATION:
            torso_orientation_rpy_command = torch.zeros(
                robot_action_policy["action.navigate_command"].shape,
                dtype=torch.float,
                device=self.device,
            )
            action_tensor = torch.cat(
                [
                    robot_action_sim.get_joints_pos(),
                    torch.tensor(
                        robot_action_policy["action.navigate_command"],
                        dtype=torch.float,
                        device=self.device,
                    ),
                    torch.tensor(
                        robot_action_policy["action.base_height_command"],
                        dtype=torch.float,
                        device=self.device,
                    ),
                    torso_orientation_rpy_command,
                ],
                axis=2,
            )
        elif self.task_mode == TaskMode.GR1_TABLETOP_MANIPULATION:
            action_tensor = robot_action_sim.get_joints_pos()
        else:
            raise ValueError(f"Unsupported task mode: {self.task_mode}")

        if action_tensor.shape[1] < self.action_chunk_length:
            raise ValueError(
                f"Returned action horizon {action_tensor.shape[1]} "
                f"is shorter than action_chunk_length {self.action_chunk_length}"
            )

        action_chunk = action_tensor[:, : self.action_chunk_length, :].cpu().numpy()
        action: Dict[str, Any] = {"action": action_chunk}
        info: Dict[str, Any] = {}
        return action, info

    def reset(self, env_ids: list[int] | None = None, reset_options: Dict[str, Any] | None = None) -> Dict[str, Any]:
        # GR00T policy is stateless for this closed-loop usage; nothing to reset
        return {"status": "reset_success"}
