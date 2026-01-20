# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from pathlib import Path

from gr00t.data.embodiment_tags import EmbodimentTag

from isaaclab_arena_gr00t.policy.config.task_mode import TaskMode


@dataclass
class LerobotReplayActionPolicyConfig:
    # model specific parameters
    dataset_path: str = field(default="", metadata={"description": "Full path to the dataset directory."})
    action_horizon: int = field(
        default=16, metadata={"description": "Number of actions in the policy's predictionhorizon."}
    )
    embodiment_tag: str = field(
        default=EmbodimentTag.NEW_EMBODIMENT.name,  # "NEW_EMBODIMENT"
        metadata={
            "description": (
                "Identifier for the robot embodiment used in the policy inference. "
                f"Must be one of: {', '.join([tag.name for tag in EmbodimentTag])} (case-insensitive). "
                "Common values: 'GR1', 'NEW_EMBODIMENT', 'UNITREE_G1'."
            )
        },
    )
    video_backend: str = field(default="decord", metadata={"description": "Video backend to use for the policy."})
    modality_config_path: str = field(
        default="isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_config.py",
        metadata={"description": "Path to the modality configuration to use for the policy."},
    )
    policy_joints_config_path: Path = field(
        default=Path(__file__).parent.parent.resolve() / "config" / "g1" / "gr00t_43dof_joint_space.yaml",
        metadata={"description": "Path to the YAML file specifying the joint ordering configuration for GR00T policy."},
    )
    task_mode_name: str = field(
        default=TaskMode.G1_LOCOMANIPULATION.value,
        metadata={"description": "Task option name of the policy inference."},
    )
    # robot simulation specific parameters
    # Only replay action and set it as targets
    action_joints_config_path: Path = field(
        default=Path(__file__).parent.parent.resolve() / "config" / "g1" / "43dof_joint_space.yaml",
        metadata={
            "description": (
                "Path to the YAML file specifying the joint ordering configuration for GR1 action space in Lab."
            )
        },
    )
    # action chunking specific parameters
    action_chunk_length: int = field(
        default=1,  # Replay actions from every recorded timestamp in the dataset
        metadata={
            "description": "Number of actions to execute per inference rollout (can be less than action_horizon)."
        },
    )

    def __post_init__(self):
        assert (
            self.action_chunk_length <= self.action_horizon
        ), "action_chunk_length must be less than or equal to action_horizon"
        # assert all paths exist
        assert Path(
            self.policy_joints_config_path
        ).exists(), f"policy_joints_config_path does not exist: {self.policy_joints_config_path}"
        assert Path(
            self.action_joints_config_path
        ).exists(), f"action_joints_config_path does not exist: {self.action_joints_config_path}"
        # LeRobotSingleDataset does not take relative path
        self.dataset_path = Path(self.dataset_path).resolve()
        assert Path(self.dataset_path).exists(), f"dataset_path does not exist: {self.dataset_path}"

        # Validate embodiment_tag using EmbodimentTag enum (case-insensitive)
        # Normalize to uppercase for enum name lookup
        embodiment_tag_upper = self.embodiment_tag.upper()
        valid_tag_names = [tag.name for tag in EmbodimentTag]

        if embodiment_tag_upper not in valid_tag_names:
            raise ValueError(
                f"Invalid embodiment_tag '{self.embodiment_tag}'. "
                f"Must be one of: {', '.join(valid_tag_names)} (case-insensitive)"
            )

        # Normalize to uppercase for consistency
        self.embodiment_tag = embodiment_tag_upper

        # Validate task mode compatibility with embodiment tag
        if self.task_mode_name == TaskMode.G1_LOCOMANIPULATION.value:
            assert self.embodiment_tag == EmbodimentTag.NEW_EMBODIMENT.name, (
                f"embodiment_tag must be {EmbodimentTag.NEW_EMBODIMENT.name} for G1 locomanipulation, got"
                f" {self.embodiment_tag}"
            )
        elif self.task_mode_name == TaskMode.GR1_TABLETOP_MANIPULATION.value:
            assert self.embodiment_tag == EmbodimentTag.GR1.name, (
                f"embodiment_tag must be {EmbodimentTag.GR1.name} for GR1 tabletop manipulation, got"
                f" {self.embodiment_tag}"
            )
        else:
            raise ValueError(f"Invalid inference mode: {self.task_mode_name}")
