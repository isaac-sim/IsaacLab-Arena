# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# %%

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pathlib

hdf5_path = "/datasets/2026_02_02_gtc_dli/potato_into_fridge_generated_100.hdf5"
hdf5_path = "/datasets/2026_02_02_gtc_dli/potato_into_fridge.hdf5"

waist_yaw_trajectories_deg = []
with h5py.File(hdf5_path, "r") as f:
    data = f["data"]
    for demo_id in data:
        # demo_id = "demo_0"
        # print(data.keys())
        demo_data = data[demo_id]
        joint_positions = demo_data["states"]["articulation"]["robot"]["joint_position"]
        waist_yaw_trajectory_rad = joint_positions[:, 2]
        waist_yaw_trajectory_deg = waist_yaw_trajectory_rad * 180 / np.pi
        waist_yaw_trajectories_deg.append(waist_yaw_trajectory_deg)


for demo_id, waist_yaw_trajectory in enumerate(waist_yaw_trajectories_deg):
    plt.plot(waist_yaw_trajectory, label=f"Demo {demo_id}")
plt.title("Waist Yaw Trajectory across original demos")
plt.legend(loc="lower left")
plt.xlabel("Sample Index")
plt.ylabel("Waist Yaw (deg)")
output_dir = pathlib.Path("/workspaces/isaaclab_arena/isaaclab_arena/examples/output")
output_path = output_dir / "waist_yaw_trajectory.png"
plt.savefig(output_path)
plt.show()


# %%

# Calculation to determine the induced hand offset by the waist yaw joint
arm_length = 0.5
yaw_angle_deg = 5.0
yaw_angle_rad = yaw_angle_deg * np.pi / 180
hand_offset = yaw_angle_rad * arm_length
print(f"Hand offset: {hand_offset} meters")

# %%
