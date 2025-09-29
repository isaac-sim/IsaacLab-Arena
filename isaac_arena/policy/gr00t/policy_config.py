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
from pathlib import Path


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
    video_backend: str = field(
        default="decord", metadata={"description": "Video backend to use for the policy."}
    )
    data_config: str = field(
        default="unitree_g1_sim_wbc", metadata={"description": "Name of the data configuration to use for the policy."}
    )
    gr00t_joints_config_path: Path = field(
        default=Path(__file__).parent.parent.resolve() / "config" / "g1" / "gr00t_43dof_joint_space.yaml",
        metadata={"description": "Path to the YAML file specifying the joint ordering configuration for GR00T policy."},
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


    # Closed loop specific parameters
    num_feedback_actions: int = field(
        default=16,
        metadata={
            "description": "Number of feedback actions to execute per rollout (can be less than action_horizon)."
        },
    )

    def __post_init__(self):
        assert (
            self.num_feedback_actions <= self.action_horizon
        ), "num_feedback_actions must be less than or equal to action_horizon"
        # assert all paths exist
        assert Path(self.gr00t_joints_config_path).exists(), "gr00t_joints_config_path does not exist"
        assert Path(self.action_joints_config_path).exists(), "action_joints_config_path does not exist"
        assert Path(self.state_joints_config_path).exists(), "state_joints_config_path does not exist"
        assert Path(self.dataset_path).exists(), "dataset_path does not exist. Do not use relative paths."
        # embodiment_tag
        assert self.embodiment_tag in [
            "gr1",
            "new_embodiment",
        ], "embodiment_tag must be one of the following: " + ", ".join(["gr1", "new_embodiment"])
