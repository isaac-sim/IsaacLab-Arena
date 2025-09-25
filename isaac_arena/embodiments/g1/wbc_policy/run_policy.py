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

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationData


def convert_sim_joint_to_wbc_joint(
    sim_joint_data: np.ndarray, sim_joint_names: list[str], wbc_joints_order: dict[str, int]
) -> np.ndarray:
    """Convert sim joint observations to WBC joint observations.

    Args:
        sim_joint_data: Sim joint data in Lab's order
        sim_joint_names: Sim joint names in Lab's order
        wbc_joints_order: WBC joint order in policy config yaml

    Returns:
        WBC joint data in WBC joint order
    """
    num_joints = len(wbc_joints_order)
    num_envs = sim_joint_data.shape[0]
    wbc_joint_data = np.zeros((num_envs, num_joints))

    # Check if sim_joint_data is a numpy array, if not, convert from torch tensor to numpy
    if not isinstance(sim_joint_data, np.ndarray):
        sim_joint_data = sim_joint_data.cpu().numpy()

    for sim_joint_name in sim_joint_names:
        sim_joint_index = sim_joint_names.index(sim_joint_name)
        assert sim_joint_name in wbc_joints_order, f"Joint {sim_joint_name} not found in loco_manip_g1_joints_order"
        wbc_joint_index = wbc_joints_order[sim_joint_name]
        wbc_joint_data[:, wbc_joint_index] = sim_joint_data[:, sim_joint_index]
    return wbc_joint_data


def prepare_observations(
    num_envs: int, robot_data: ArticulationData, wbc_joints_order: dict[str, int]
) -> dict[str, np.ndarray]:
    """Prepare observations for the policy.

    Args:
        num_envs: Number of environments
        robot_data: Robot data
        wbc_joints_order: WBC joint order in policy config yaml

    Returns:
        Observations for the policy
        - q: Joint positions
        - dq: Joint velocities
        - ddq: Joint accelerations
        - floating_base_pose: Floating base pose
        - floating_base_vel: Floating base velocity
        - floating_base_acc: Floating base acceleration
        - torso_quat: Torso quaternion
        - torso_ang_vel: Torso angular velocity
    """
    # Get robot joint observations
    sim_joint_pos = robot_data.joint_pos.cpu().numpy()
    sim_joint_vel = robot_data.joint_vel.cpu().numpy()
    num_joints = len(robot_data.joint_names)

    # Convert joints data from Lab's order to GR00T's order saved in config yaml
    wbc_joint_pos = np.zeros((num_envs, num_joints))
    wbc_joint_vel = np.zeros((num_envs, num_joints))
    wbc_joint_acc = np.zeros((num_envs, num_joints))
    wbc_joint_pos = convert_sim_joint_to_wbc_joint(sim_joint_pos, robot_data.joint_names, wbc_joints_order)
    wbc_joint_vel = convert_sim_joint_to_wbc_joint(sim_joint_vel, robot_data.joint_names, wbc_joints_order)

    # Prepare obs dict for WBC policy input to G1DecoupledWholeBodyPolicy class
    assert wbc_joint_pos.shape == wbc_joint_vel.shape == wbc_joint_acc.shape == (num_envs, num_joints)

    root_link_pos_w = robot_data.root_link_pos_w.cpu().numpy()
    root_link_quat_w = robot_data.root_link_quat_w.cpu().numpy()
    base_pose_w = np.concatenate((root_link_pos_w, root_link_quat_w), axis=1)
    base_lin_vel_b = robot_data.root_link_lin_vel_b.cpu().numpy()
    base_ang_vel_b = robot_data.root_link_ang_vel_b.cpu().numpy()

    base_vel_b = np.concatenate((base_lin_vel_b, base_ang_vel_b), axis=1)
    # torso link in world frame
    torso_link_pose_w = robot_data.body_link_state_w[:, robot_data.body_names.index("torso_link"), :]
    torso_link_quat_w = torso_link_pose_w[:, 3:7]  # w, x, y, z
    torso_link_ang_vel_w = torso_link_pose_w[:, -3:]

    torso_link_ang_vel_b = math_utils.quat_apply_inverse(torso_link_quat_w, torso_link_ang_vel_w)

    # Prepare obs tmers
    wbc_obs = {
        "q": wbc_joint_pos,
        "dq": wbc_joint_vel,
        "ddq": np.zeros((num_envs, num_joints)),  # Not used by Standing Waist Height Policy
        "tau_est": np.zeros((num_envs, num_joints)),  # Not used by Standing Waist Height Policy
        "floating_base_pose": base_pose_w,  # wrt world frame, used to project gravity vector to local frame
        "floating_base_vel": base_vel_b,  # wrt body frame
        "floating_base_acc": np.zeros((num_envs, 6)),  # Not used by Standing Waist Height Policy
        "torso_quat": torso_link_quat_w.cpu().numpy(),
        "torso_ang_vel": torso_link_ang_vel_b.cpu().numpy(),
    }
    return wbc_obs


def postprocess_actions(
    wbc_action: dict[str, np.ndarray],
    robot_data: ArticulationData,
    wbc_g1_joints_order: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    """Postprocess actions for the policy."""
    num_envs = wbc_action["q"].shape[0]
    num_joints = len(robot_data.joint_names)
    processed_actions = torch.zeros((num_envs, num_joints), device=device)
    wbc_joints_pos_action = torch.from_numpy(wbc_action["q"])
    # Convert wbc gr00t joints order to Lab joints order
    for wbc_joint_name, wbc_joint_index in wbc_g1_joints_order.items():
        if wbc_joint_name not in robot_data.joint_names:
            print(f"Joint {wbc_joint_name} not found in asset")
            continue
        sim_joint_index = robot_data.joint_names.index(wbc_joint_name)
        processed_actions[:, sim_joint_index] = wbc_joints_pos_action[:, wbc_joint_index]
    return processed_actions
