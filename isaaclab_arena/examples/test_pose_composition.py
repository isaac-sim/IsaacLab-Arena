# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# %%


from isaaclab_arena.utils.pose import Pose

T_B_A = Pose(position_xyz=(1.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
T_C_B = Pose(position_xyz=(2.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))

T_C_A = T_C_B.multiply(T_B_A)

assert T_C_A.position_xyz == (3.0, 0.0, 0.0)
assert T_C_A.rotation_wxyz == (1.0, 0.0, 0.0, 0.0)

T_C_B = Pose(position_xyz=(2.0, 0.0, 0.0), rotation_wxyz=(0.7071068, 0.0, 0.0, 0.7071068))


# %%

import torch

from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

B_R_A = matrix_from_quat(torch.tensor(B_T_A.rotation_wxyz))
C_R_B = matrix_from_quat(torch.tensor(C_T_B.rotation_wxyz))

C_R_A = C_R_B @ B_R_A
print(C_R_A)

C_q_A = quat_from_matrix(C_R_A)
print(C_q_A)

C_t_A = C_R_A @ torch.tensor(B_T_A.position_xyz) + torch.tensor(C_T_B.position_xyz)
print(C_t_A)

C_T_A = Pose(position_xyz=tuple(C_t_A.tolist()), rotation_wxyz=tuple(C_q_A.tolist()))

print(C_T_A)


# %%
