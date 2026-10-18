# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""World-frame TCP measurements and absolute pose actions for the bimanual YAM."""

import torch

from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

TCP_OFFSET = (0.0, -0.044, 0.13)


def tcp_pose_w(env, side):
    """Return T_W_TCP as batched position and xyzw quaternion."""
    robot = env.scene[f"{side}_robot"]
    index = robot.body_names.index("link_6")
    T_W_L = robot.data.body_pose_w.torch[:, index]
    offset = torch.tensor(TCP_OFFSET, device=env.device).expand(env.num_envs, -1)
    p, q = combine_frame_transforms(T_W_L[:, :3], T_W_L[:, 3:], offset)
    return torch.cat((p, q), dim=-1)


def absolute_pose_actions(env, targets_w, grippers):
    """Convert left/right T_W_TCP targets into the 16-value base-frame action layout."""
    commands = []
    for side in ("left", "right"):
        robot = env.scene[f"{side}_robot"]
        target = torch.as_tensor(targets_w[side], dtype=torch.float32, device=env.device).reshape(-1, 7)
        target = target.expand(env.num_envs, -1)
        T_W_B = robot.data.root_pose_w.torch
        p, q = subtract_frame_transforms(T_W_B[:, :3], T_W_B[:, 3:], target[:, :3], target[:, 3:])
        grip = torch.as_tensor(grippers[side], device=env.device).reshape(-1, 1).expand(env.num_envs, 1)
        commands.extend((p, q, grip))
    return torch.cat(commands, dim=-1)
