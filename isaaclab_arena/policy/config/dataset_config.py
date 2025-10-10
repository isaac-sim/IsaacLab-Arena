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

import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Gr00tDatasetConfig:
    # Datasets & task specific parameters
    data_root: Path = field(
        default=Path("/datasets/"),
        metadata={"description": "Root directory for all data storage."},
    )
    language_instruction: str = field(
        default=None, metadata={"description": "Instruction given to the policy in natural language."}
    )
    hdf5_name: str = field(default=None, metadata={"description": "Name of the HDF5 file to use for the dataset."})

    # Mimic-generated HDF5 datafield
    # NOTE(xinjieyao, 2025-09-25): robot joint position must exist in the HDF5 file
    state_name_sim: str = field(
        default="robot_joint_pos", metadata={"description": "Name of the state in the HDF5 file."}
    )
    left_eef_pos_name_sim: str = field(
        default=None, metadata={"description": "Name of the left eef position in the HDF5 file(optional)."}
    )
    left_eef_quat_name_sim: str = field(
        default=None, metadata={"description": "Name of the left eef quaternion in the HDF5 file(optional)."}
    )
    right_eef_pos_name_sim: str = field(
        default=None, metadata={"description": "Name of the right eef position in the HDF5 file(optional)."}
    )
    right_eef_quat_name_sim: str = field(
        default=None, metadata={"description": "Name of the right eef quaternion in the HDF5 file(optional)."}
    )
    teleop_base_height_command_name_sim: str = field(
        default=None, metadata={"description": "Name of the teleop base height command in the HDF5 file."}
    )
    teleop_navigate_command_name_sim: str = field(
        default=None, metadata={"description": "Name of the teleop navigate command in the HDF5 file."}
    )
    teleop_torso_orientation_rpy_command_name_sim: str = field(
        default=None, metadata={"description": "Name of the teleop waist roll pitch yaw command in the HDF5 file."}
    )
    action_name_sim: str = field(
        default="processed_actions", metadata={"description": "Name of the action in the HDF5 file."}
    )
    pov_cam_name_sim: str = field(
        default="robot_head_cam", metadata={"description": "Name of the POV camera in the HDF5 file."}
    )
    # Gr00t-LeRobot datafield
    state_name_lerobot: str = field(
        default="observation.state", metadata={"description": "Name of the state in the LeRobot file."}
    )
    action_name_lerobot: str = field(
        default="action", metadata={"description": "Name of the action in the LeRobot file."}
    )
    action_eef_name_sim: str = field(
        default="action.eef", metadata={"description": "Name of the eef action in the HDF5 file."}
    )

    video_name_lerobot: str = field(
        default="observation.images.ego_view", metadata={"description": "Name of the video in the LeRobot file."}
    )
    task_description_lerobot: str = field(
        default="annotation.human.action.task_description",
        metadata={"description": "Name of the task description in the LeRobot file."},
    )
    valid_lerobot: str = field(
        default="annotation.human.action.valid", metadata={"description": "Name of the validity in the LeRobot file."}
    )

    # Parquet
    chunks_size: int = field(default=1000, metadata={"description": "Number of episodes per data chunk."})
    # mp4 video
    fps: int = field(default=50, metadata={"description": "Frames per second for video recording."})
    # Metadata files
    data_path: str = field(
        default="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        metadata={"description": "Template path for storing episode data files."},
    )
    video_path: str = field(
        default="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        metadata={"description": "Template path for storing episode video files."},
    )
    modality_template_path: Path = field(
        default=Path(__file__).parent.resolve() / "g1" / "modality.json",
        metadata={"description": "Path to the modality template JSON file."},
    )
    modality_fname: str = field(
        default="modality.json", metadata={"description": "Filename for the modality JSON file."}
    )
    episodes_fname: str = field(
        default="episodes.jsonl", metadata={"description": "Filename for the episodes JSONL file."}
    )
    tasks_fname: str = field(default="tasks.jsonl", metadata={"description": "Filename for the tasks JSONL file."})
    info_template_path: Path = field(
        default=Path(__file__).parent.resolve() / "g1" / "info.json",
        metadata={"description": "Path to the info template JSON file."},
    )
    info_fname: str = field(default="info.json", metadata={"description": "Filename for the info JSON file."})
    # policy specific parameters
    policy_joints_config_path: Path = field(
        default=Path(__file__).parent.resolve() / "g1" / "gr00t_43dof_joint_space.yaml",
        metadata={"description": "Path to the YAML file specifying the joint ordering configuration used in dataset."},
    )
    robot_type: str = field(
        default="null", metadata={"description": "Type of robot embodiment used in the policy fine-tuning."}
    )
    # robot simulation specific parameters
    action_joints_config_path: Path = field(
        default=Path(__file__).parent.resolve() / "g1" / "43dof_joint_space.yaml",
        metadata={
            "description": (
                "Path to the YAML file specifying the joint ordering configuration for robot action space in Lab."
            )
        },
    )
    state_joints_config_path: Path = field(
        default=Path(__file__).parent.resolve() / "g1" / "43dof_joint_space.yaml",
        metadata={
            "description": (
                "Path to the YAML file specifying the joint ordering configuration for robot state space in Lab."
            )
        },
    )
    original_image_size: tuple[int, int, int] = field(
        default=(480, 640, 3), metadata={"description": "Original size of input images as (height, width, channels)."}
    )
    target_image_size: tuple[int, int, int] = field(
        default=(480, 640, 3), metadata={"description": "Target size for images after resizing and padding."}
    )

    hdf5_file_path: Path = field(init=False)
    lerobot_data_dir: Path = field(init=False)
    task_index: int = field(default=0, metadata={"description": "Task index for the task description in LeRobot file."})

    def __post_init__(self):

        self.hdf5_file_path = self.data_root / self.hdf5_name
        self.lerobot_data_dir = self.data_root / self.hdf5_name.replace(".hdf5", "") / "lerobot"

        assert self.hdf5_file_path.exists(), f"{self.hdf5_file_path} does not exist"
        assert Path(self.policy_joints_config_path).exists(), f"{self.policy_joints_config_path} does not exist"
        assert Path(self.action_joints_config_path).exists(), f"{self.action_joints_config_path} does not exist"
        assert Path(self.state_joints_config_path).exists(), f"{self.state_joints_config_path} does not exist"
        assert Path(self.info_template_path).exists(), f"{self.info_template_path} does not exist"
        assert Path(self.modality_template_path).exists(), f"{self.modality_template_path} does not exist"
        # in case lerobot_data_dir already exists, may be left over from previous runs, ask for user confirmation before removing
        if self.lerobot_data_dir.exists():
            print(f"Warning: lerobot_data_dir {self.lerobot_data_dir} already exists.")
            if input(f"Are you sure you want to remove {self.lerobot_data_dir}? (y/n): ") == "y":
                shutil.rmtree(self.lerobot_data_dir)
            else:
                print(f"Skipping removal of {self.lerobot_data_dir}")
        # Prepare data keys for mimic-generated hdf5 file
        # Minimum set of keys are state & action
        self.hdf5_keys = {"state": self.state_name_sim, "action": self.action_name_sim}
        # Optional keys if provided
        if self.left_eef_pos_name_sim:
            self.hdf5_keys["left_eef_pos"] = self.left_eef_pos_name_sim
        if self.left_eef_quat_name_sim:
            self.hdf5_keys["left_eef_quat"] = self.left_eef_quat_name_sim
        if self.right_eef_pos_name_sim:
            self.hdf5_keys["right_eef_pos"] = self.right_eef_pos_name_sim
        if self.right_eef_quat_name_sim:
            self.hdf5_keys["right_eef_quat"] = self.right_eef_quat_name_sim
        if self.teleop_base_height_command_name_sim:
            self.hdf5_keys["teleop_base_height_command"] = self.teleop_base_height_command_name_sim
        if self.teleop_navigate_command_name_sim:
            self.hdf5_keys["teleop_navigate_command"] = self.teleop_navigate_command_name_sim
        if self.teleop_torso_orientation_rpy_command_name_sim:
            self.hdf5_keys["teleop_torso_orientation_rpy_command"] = self.teleop_torso_orientation_rpy_command_name_sim
        if self.action_eef_name_sim:
            self.hdf5_keys["action_eef_pose"] = self.action_eef_name_sim

        # Prepare data keys for LeRobot file
        self.lerobot_keys = {
            "state": self.state_name_lerobot,
            "action": self.action_name_lerobot,
            "video": self.video_name_lerobot,
            "annotation": (self.task_description_lerobot,),
        }
        if "left_eef_pos" in self.hdf5_keys:
            self.lerobot_keys["obs_eef_pose"] = "observation.eef_pose"
            self.lerobot_keys["action_eef_pose"] = "action.eef_pose"
        if "teleop_base_height_command" in self.hdf5_keys:
            self.lerobot_keys["teleop_base_height_command"] = "teleop.base_height_command"
        if "teleop_navigate_command" in self.hdf5_keys:
            self.lerobot_keys["teleop_navigate_command"] = "teleop.navigate_command"
        if "teleop_torso_orientation_rpy_command" in self.hdf5_keys:
            self.lerobot_keys["teleop_torso_orientation_rpy_command"] = "teleop.torso_orientation_rpy_command"
