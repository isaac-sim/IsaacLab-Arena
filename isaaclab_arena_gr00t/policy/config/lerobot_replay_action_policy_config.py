# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from pathlib import Path

from isaaclab_arena_gr00t.policy.config.task_mode import TaskMode


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
