# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


# Policy data loader and architecture configuration depend on which task to choose
class TaskMode(Enum):
    G1_LOCOMANIPULATION = "g1_locomanipulation"
    GR1_TABLETOP_MANIPULATION = "gr1_tabletop_manipulation"


@dataclass
class LerobotReplayActionPolicyConfig:
    # model specific parameters
    dataset_path: str = field(default="", metadata={"description": "Full path to the dataset directory."})
    action_horizon: int = field(
        default=16, metadata={"description": "Number of actions in the policy's predictionhorizon."}
    )
    embodiment_tag: str = field(
        default="new_embodiment",
        metadata={
            "description": (
                "Identifier for the robot embodiment used in the policy inference (e.g., 'gr1' or 'new_embodiment')."
            )
        },
    )
    video_backend: str = field(default="decord", metadata={"description": "Video backend to use for the policy."})
    data_config: str = field(
        default="unitree_g1_sim_wbc", metadata={"description": "Name of the data configuration to use for the policy."}
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
    num_feedback_actions: int = field(
        default=1,  # Replay actions from every recorded timestamp in the dataset
        metadata={
            "description": "Number of feedback actions to execute per rollout (can be less than action_horizon)."
        },
    )

    def __post_init__(self):
        assert (
            self.num_feedback_actions <= self.action_horizon
        ), "num_feedback_actions must be less than or equal to action_horizon"
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
        # embodiment_tag
        assert self.embodiment_tag in [
            "gr1",
            "new_embodiment",
        ], "embodiment_tag must be one of the following: " + ", ".join(["gr1", "new_embodiment"])
        if self.task_mode_name == TaskMode.G1_LOCOMANIPULATION.value:
            assert (
                self.embodiment_tag == "new_embodiment"
            ), "embodiment_tag must be new_embodiment for G1 locomanipulation"
        elif self.task_mode_name == TaskMode.GR1_TABLETOP_MANIPULATION.value:
            assert (
                self.embodiment_tag == "gr1"
            ), "embodiment_tag must be gr1 for GR1 tabletop manipulation. Is {self.embodiment_tag}"
        else:
            raise ValueError(f"Invalid inference mode: {self.task_mode}")


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
        default="new_embodiment",
        metadata={
            "description": (
                "Identifier for the robot embodiment used in the policy inference (e.g., 'gr1' or 'new_embodiment')."
            )
        },
    )
    denoising_steps: int = field(
        default=4, metadata={"description": "Number of denoising steps used in the policy inference."}
    )
    data_config: str = field(
        default="unitree_g1_sim_wbc", metadata={"description": "Name of the data configuration to use for the policy."}
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
    num_feedback_actions: int = field(
        default=16,
        metadata={
            "description": "Number of feedback actions to execute per rollout (can be less than action_horizon)."
        },
    )
    seed: int = field(default=10, metadata={"description": "Random seed for reproducibility."})

    def __post_init__(self):
        assert (
            self.num_feedback_actions <= self.action_horizon
        ), "num_feedback_actions must be less than or equal to action_horizon"
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
        # embodiment_tag
        assert self.embodiment_tag in [
            "gr1",
            "new_embodiment",
        ], "embodiment_tag must be one of the following: " + ", ".join(["gr1", "new_embodiment"])
        if self.task_mode_name == TaskMode.G1_LOCOMANIPULATION.value:
            assert (
                self.embodiment_tag == "new_embodiment"
            ), "embodiment_tag must be new_embodiment for G1 locomanipulation"
        elif self.task_mode_name == TaskMode.GR1_TABLETOP_MANIPULATION.value:
            assert self.embodiment_tag == "gr1", "embodiment_tag must be gr1 for GR1 tabletop manipulation"
        else:
            raise ValueError(f"Invalid inference mode: {self.task_mode}")
