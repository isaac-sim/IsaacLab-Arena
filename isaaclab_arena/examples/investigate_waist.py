# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# %%

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pathlib

# hdf5_path = "/datasets/2026_02_02_gtc_dli/potato_into_fridge_generated_100.hdf5"
# hdf5_path = "/datasets/2026_02_02_gtc_dli/potato_into_fridge.hdf5"
# hdf5_path = "/datasets/2026_02_02_gtc_dli/jug_into_fridge_generated_100.hdf5"
hdf5_path = "/datasets/2026_02_02_gtc_dli/gr1_jug_into_fridge_no_randomization/jug_into_fridge_generated_100_peterd_v2_no_randomization.hdf5"

right_arm_joint_names = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
]
right_arm_joint_indices = [13, 18, 23, 25]


waist_yaw_trajectories_deg = []
right_arm_joint_positions_trajectories_deg = []
with h5py.File(hdf5_path, "r") as f:
    data = f["data"]
    for idx, demo_id in enumerate(data):
        # demo_id = "demo_0"
        # print(data.keys())
        demo_data = data[demo_id]
        joint_positions = demo_data["states"]["articulation"]["robot"]["joint_position"]

        # Waist
        waist_yaw_trajectory_rad = joint_positions[:, 2]
        waist_yaw_trajectory_deg = waist_yaw_trajectory_rad * 180 / np.pi
        waist_yaw_trajectories_deg.append(waist_yaw_trajectory_deg)

        # Right Arm
        right_arm_joint_positions = joint_positions[:, right_arm_joint_indices]
        right_arm_joint_positions_deg = right_arm_joint_positions * 180 / np.pi
        right_arm_joint_positions_trajectories_deg.append(right_arm_joint_positions_deg)

        if idx == 20:
            break


DEMO_IDX = 2

# for demo_id, waist_yaw_trajectory in enumerate(waist_yaw_trajectories_deg):
#     plt.plot(waist_yaw_trajectory, label=f"Demo {demo_id}")
# plt.plot(waist_yaw_trajectories_deg[DEMO_IDX], label=f"Demo {DEMO_IDX}")
# plt.title("Waist Yaw Trajectory across original demos")
# # plt.legend(loc="lower left")
# plt.xlabel("Sample Index")
# plt.ylabel("Waist Yaw (deg)")
# output_dir = pathlib.Path("/workspaces/isaaclab_arena/isaaclab_arena/examples/output")
# output_path = output_dir / "waist_yaw_trajectory.png"
# plt.savefig(output_path)
# plt.show()

# for demo_id, right_arm_joint_positions_trajectory in enumerate(right_arm_joint_positions_trajectories_deg):
#     plt.plot(right_arm_joint_positions_trajectory, label=f"Demo {demo_id}")
plt.plot(right_arm_joint_positions_trajectories_deg[DEMO_IDX], label=f"Demo {DEMO_IDX}")
plt.plot(waist_yaw_trajectories_deg[DEMO_IDX], label=f"Demo {DEMO_IDX}")
plt.title("Right Arm Joint Positions Trajectory across original demos")
# plt.legend(loc="lower left")
plt.xlabel("Sample Index")
plt.ylabel("Joint Positions (deg)")
output_dir = pathlib.Path("/workspaces/isaaclab_arena/isaaclab_arena/examples/output")
output_path = output_dir / "joint_positions_trajectory.png"
plt.legend([*right_arm_joint_names, "Waist Yaw"])
plt.savefig(output_path)

plt.show()

# %%

rollout_joint_positions = np.load(
    "/workspaces/isaaclab_arena/isaaclab_arena/examples/output/joint_positions_trajectory.npy"
)
rollout_right_arm_joint_positions = rollout_joint_positions[:, right_arm_joint_indices]
rollout_right_arm_joint_positions_deg = rollout_right_arm_joint_positions * 180 / np.pi

plt.plot(rollout_right_arm_joint_positions_deg, label=f"Rollout", marker=".")
plt.plot(right_arm_joint_positions_trajectories_deg[DEMO_IDX], label=f"Demo {DEMO_IDX}")
plt.title("Right Arm Joint Positions Trajectory across rollout")
plt.xlabel("Sample Index")
plt.ylabel("Joint Positions (deg)")
legend_labels = []
for joint_name in right_arm_joint_names:
    legend_labels.append(f"rollout {joint_name} (deg)")
# for joint_name in right_arm_joint_names:
#     legend_labels.append(f"demo {joint_name} (deg)")
plt.legend(legend_labels)
plt.savefig(output_path)
plt.show()


# %%

rollout_waist_yaw_trajectory = rollout_joint_positions[:, 2]
rollout_waist_yaw_trajectory_deg = rollout_waist_yaw_trajectory * 180 / np.pi
plt.plot(rollout_waist_yaw_trajectory_deg, label=f"Rollout", marker=".")
plt.plot(waist_yaw_trajectories_deg[DEMO_IDX], label=f"Demo {DEMO_IDX}")
plt.title("Waist Yaw Trajectory across rollout")
plt.xlabel("Sample Index")
plt.ylabel("Waist Yaw (deg)")
plt.legend(["Rollout", "Demo"])
plt.savefig(output_path)

# %%

joint_names = [
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_yaw_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_roll_joint",
    "left_knee_pitch_joint",
    "right_knee_pitch_joint",
    "head_roll_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "head_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "head_yaw_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_thumb_proximal_yaw_joint",
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_thumb_proximal_yaw_joint",
    "L_index_intermediate_joint",
    "L_middle_intermediate_joint",
    "L_pinky_intermediate_joint",
    "L_ring_intermediate_joint",
    "L_thumb_proximal_pitch_joint",
    "R_index_intermediate_joint",
    "R_middle_intermediate_joint",
    "R_pinky_intermediate_joint",
    "R_ring_intermediate_joint",
    "R_thumb_proximal_pitch_joint",
    "L_thumb_distal_joint",
    "R_thumb_distal_joint",
]

right_arm_joint_names = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
]

right_arm_joint_indices = [joint_names.index(joint_name) for joint_name in right_arm_joint_names]
print(f"Right arm joint indices: {right_arm_joint_indices}")


# %%

# Calculation to determine the induced hand offset by the waist yaw joint
arm_length = 0.5
yaw_angle_deg = 5.0
yaw_angle_rad = yaw_angle_deg * np.pi / 180
hand_offset = yaw_angle_rad * arm_length
print(f"Hand offset: {hand_offset} meters")

# %%
