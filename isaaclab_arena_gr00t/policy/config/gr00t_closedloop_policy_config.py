# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from pathlib import Path
from isaaclab_arena_gr00t.policy.config.task_mode import TaskMode

@dataclass
class Gr00tClosedloopPolicyConfig:

    language_instruction: str = field(
        default="", metadata={"description": "Instruction given to the policy in natural language."}
    )
    model_path: str = field(
        default=None, metadata={"description": "Full path to the tuned model checkpoint directory."}
    )
    action_horizon: int = field(
        default=16, metadata={"description": "Number of actions in the policy's predictionhorizon."}
    )
    embodiment_tag: str = field(
        default="NEW_EMBODIMENT",
        metadata={
            "description": (
                "Identifier for the robot embodiment used in the policy inference (e.g., 'gr1' or 'new_embodiment')."
            )
        },
    )
    denoising_steps: int = field(
        default=4, metadata={"description": "Number of denoising steps used in the policy inference."}
    )
    modality_config_path: str = field(
        default=None, metadata={"description": "Path to the modality configuration file."}
    )
    original_image_size: tuple[int, int, int] = field(
        default=(480, 640, 3), metadata={"description": "Original size of input images as (height, width, channels)."}
    )
    target_image_size: tuple[int, int, int] = field(
        default=(480, 640, 3),
        metadata={"description": "Target size for images after resizing and padding as (height, width, channels)."},
    )
    policy_joints_config_path: Path = field(
        default=Path(__file__).parent.resolve() / "config" / "g1" / "gr00t_43dof_joint_space.yaml",
        metadata={"description": "Path to the YAML file specifying the joint ordering configuration for GR00T policy."},
    )
    task_mode_name: str = field(
        default=TaskMode.G1_LOCOMANIPULATION.value,
        metadata={"description": "Task option name of the policy inference."},
    )
    # robot simulation specific parameters
    action_joints_config_path: Path = field(
        default=Path(__file__).parent.parent.resolve() / "config" / "g1" / "43dof_joint_space.yaml",
        metadata={
            "description": (
                "Path to the YAML file specifying the joint ordering configuration for GR1 action space in Lab."
            )
        },
    )
    state_joints_config_path: Path = field(
        default=Path(__file__).parent.parent.resolve() / "config" / "g1" / "43dof_joint_space.yaml",
        metadata={
            "description": (
                "Path to the YAML file specifying the joint ordering configuration for GR1 state space in Lab."
            )
        },
    )
    # Default to GPU policy and CPU physics simulation
    policy_device: str = field(
        default="cuda", metadata={"description": "Device to run the policy model on (e.g., 'cuda' or 'cpu')."}
    )
    video_backend: str = field(default="decord", metadata={"description": "Video backend to use for evaluation."})
    pov_cam_name_sim: str = field(
        default="robot_head_cam_rgb", metadata={"description": "Name of the POV camera of the robot in simulation."}
    )
    # Closed loop specific parameters
    action_chunk_length: int = field(
        default=16,
        metadata={
            "description": "Number of actions to execute per inference rollout (can be less than action_horizon)."
        },
    )
    seed: int = field(default=10, metadata={"description": "Random seed for reproducibility."})

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
        assert Path(
            self.state_joints_config_path
        ).exists(), f"state_joints_config_path does not exist: {self.state_joints_config_path}"
        assert Path(self.model_path).exists(), f"model_path does not exist: {self.model_path}"
        assert Path(
            self.modality_config_path
        ).exists(), f"modality_config_path does not exist: {self.modality_config_path}"
        # embodiment_tag
        assert self.embodiment_tag in [
            "GR1",
            "NEW_EMBODIMENT",
        ], "embodiment_tag must be one of the following: " + ", ".join(["GR1", "NEW_EMBODIMENT"])
        if self.task_mode_name == TaskMode.G1_LOCOMANIPULATION.value:
            assert (
                self.embodiment_tag == "NEW_EMBODIMENT"
            ), "embodiment_tag must be new_embodiment for G1 locomanipulation"
        elif self.task_mode_name == TaskMode.GR1_TABLETOP_MANIPULATION.value:
            assert self.embodiment_tag == "GR1", "embodiment_tag must be GR1 for GR1 tabletop manipulation"
        else:
            raise ValueError(f"Invalid inference mode: {self.task_mode}")
