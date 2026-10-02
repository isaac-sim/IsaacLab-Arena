# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Physical fixture helpers shared by return-to-service simulation tests."""

from pathlib import Path

import pytest


def _require_assets() -> None:
    manifest = Path.home() / ".cache/isaaclab_arena/return_to_service/assets/manifest.json"
    if not manifest.is_file():
        pytest.skip("Generate the return-to-service Blender asset bundle in the default cache before simulation tests.")


def _write_pose(base, name: str, env_id: int, T_W_O, velocity=None) -> None:
    """Set a test fixture pose and velocity through the simulation API."""
    import torch

    indices = torch.tensor([env_id], device=base.device, dtype=torch.int32)
    body = base.scene[name]
    body.write_root_pose_to_sim_index(root_pose=T_W_O.reshape(1, 7), env_ids=indices)
    root_velocity = T_W_O.new_zeros((1, 6)) if velocity is None else T_W_O.new_tensor(velocity).reshape(1, 6)
    body.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=indices)


def _write_joint(base, name: str, joint_name: str, env_id: int, value: float) -> None:
    """Inject a mechanism position to check its measured release and interlock behavior."""
    import torch

    articulation = base.scene[name]
    indices = torch.tensor([env_id], device=base.device, dtype=torch.int32)
    position = articulation.data.joint_pos.torch[indices].clone()
    position[:, articulation.data.joint_names.index(joint_name)] = value
    articulation.write_joint_state_to_sim_index(position=position, velocity=torch.zeros_like(position), env_ids=indices)


def _pose_in_parent(base, parent_name: str, pose, env_id: int):
    import torch

    from isaaclab.utils.math import quat_apply, quat_mul

    T_W_P = base.arena_world.get_pose_w(parent_name)[env_id]
    t_W_O = T_W_P[:3] + quat_apply(T_W_P[3:], T_W_P.new_tensor(pose.position_xyz))
    q_W_O = quat_mul(T_W_P[3:], T_W_P.new_tensor(pose.rotation_xyzw))
    return torch.cat((t_W_O, q_W_O))
