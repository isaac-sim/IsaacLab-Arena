# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import torch
from typing import Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.affordances.affordance_base import AffordanceBase


class Placeable(AffordanceBase):
    """Interface for placeable objects.
    Placeable objects are objects that can be placed upright in a scene.
    They are characterized by their upright axis and orientation threshold, which are used to determine if the object is placed upright.
    - The upright axis is the axis in the object's local frame that is used to determine if the object is placed upright.
    - The orientation threshold is the threshold for the angle between the upright axis and the world +Z direction, in radians.
    """

    def __init__(
        self, upright_axis_name: Literal["x", "y", "z"] = "z", orientation_threshold: float = math.pi / 18.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.upright_axis_name = upright_axis_name
        assert upright_axis_name in ["x", "y", "z"], "Upright axis must be one of x, y, or z"
        self.orientation_threshold = orientation_threshold

    def is_placed_upright(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg | None = None,
        orientation_threshold: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns a boolean tensor of whether the object is placeable.

        Args:
            env: The environment instance.
            asset_cfg: Asset configuration. If None, uses the object's name.
            orientation_threshold: Orientation threshold in radians. If None, uses the object's orientation threshold.
            If a tensor, must have shape [num_envs].

        Returns:
            A boolean tensor of whether the object is placed upright. Shape: [num_envs].
        """
        if asset_cfg is None:
            asset_cfg = SceneEntityCfg(self.name)
        # We allow for overriding the object-level threshold by passing an argument to this
        # function explicitly. Otherwise we use the object-level threshold.
        if orientation_threshold is None:
            orientation_threshold = self.orientation_threshold
        object_entity: RigidObject = env.scene[asset_cfg.name]
        object_quat = object_entity.data.root_quat_w

        upright_axis_world = get_object_axis_in_world_frame(object_quat, self.upright_axis_name)

        world_up = torch.zeros_like(upright_axis_world)
        world_up[..., 2] = 1.0

        cos_angle = torch.sum(upright_axis_world * world_up, dim=-1).clamp(-1.0, 1.0)
        angle_error = torch.acos(cos_angle)

        orientation_threshold_tensor = torch.as_tensor(
            orientation_threshold, device=object_quat.device, dtype=object_quat.dtype
        )

        success = angle_error < orientation_threshold_tensor

        return success

    def place_upright(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg | None = None,
        upright_percentage: float | torch.Tensor = 1.0,
    ):
        """Place the object upright (in all the environments).

        This function places the object upright by rotating its upright axis towards the world +Z direction.
        The upright percentage is a value in [0, 1] that describes the absolute target angle of the object_upright_axis:
        0.0 results in a perpendicular orientation (90 degrees) in the plane spanned by the object_upright_axis and world +Z,
        and 1.0 fully aligns the axis with +Z (0 degrees). Other target angle in (0, 1) rotates in the plane spanned by the object_upright_axis and world +Z as well.

        Args:
            env: The environment instance.
            env_ids: Environment indices to apply the operation to. If None, applies to all environments.
            asset_cfg: Asset configuration. If None, uses the object's name.
            upright_percentage: Target upright percentage(s) in [0, 1]. Can be:
                - A scalar float applied to all specified environments
                - A tensor with shape matching len(env_ids), with per-environment values

        Raises:
            ValueError: If upright_percentage is a tensor with size not matching len(env_ids).
        """
        if asset_cfg is None:
            asset_cfg = SceneEntityCfg(self.name)

        set_normalized_object_pose(
            env=env,
            asset_cfg=asset_cfg,
            upright_percentage=upright_percentage,
            env_ids=env_ids,
            upright_axis_name=self.upright_axis_name,
        )


def get_object_axis_in_world_frame(object_quat: torch.tensor, upright_axis_name: str) -> torch.tensor:
    """Get the object axis in the world frame.
    Args:
        object_quat: The quaternion of the object. Shape: [num_envs, 4] (w, x, y, z).
        upright_axis_name: The name of the object axis to get in the world frame. Can be "x", "y", or "z".

    Returns:
        The object axis in the world frame. Shape: [num_envs, 3].
    """
    axis_index = {"x": 0, "y": 1, "z": 2}[upright_axis_name]
    rotation_mats = math_utils.matrix_from_quat(object_quat)
    object_axis_world = rotation_mats[:, :, axis_index]
    return object_axis_world


def set_normalized_object_pose(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    upright_percentage: float | torch.Tensor = 1.0,
    env_ids: torch.Tensor | None = None,
    upright_axis_name: str = "z",
) -> None:
    """Rotate a rigid object towards the upright orientation directly through its root pose.

    Values of ``upright_percentage`` in [0, 1] describe the absolute target orientation of the object_upright_axis.

    Target orientation diagram (only rotates in the plane spanned by the object_upright_axis and world +Z):

                        +Z (world up)
                        ↑   target upright axis (upright_percentage=1.0)
                        |
                        |
                        |
                        |
                        |
                        |    / object_upright_axis
                        |   /
                        |  /
                        | / θ
                        ●------→ target upright axis (upright_percentage=0.0, 90° to +Z)
                    object                           (the direction with the smallest angle to the object_upright_axis)

    Target orientation: only rotates in the plane spanned by the object_upright_axis and world +Z.
    - upright_percentage = 1.0: the target upright axis aligns with world +Z (0°)
    - upright_percentage = 0.0: the target upright axis is perpendicular to world +Z (90°), and the direction with the smallest angle to the object_upright_axis

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the rigid object asset.
        upright_percentage: Target upright percentage(s) in [0, 1]. Can be:
            - A scalar float applied to all specified environments
            - A tensor with shape matching len(env_ids), with per-environment values
        env_ids: Environment indices to apply the operation to. If None, applies to all environments.
        upright_axis_name: Name of the object's local axis to align ("x", "y", or "z").

    Raises:
        ValueError: If upright_percentage is a tensor with size not matching len(env_ids).
    """
    object_entity: RigidObject = env.scene[asset_cfg.name]
    device = env.device
    dtype = object_entity.data.root_quat_w.dtype

    if env_ids is not None:
        env_ids = env_ids.to(env.device)
    else:
        env_ids = torch.arange(object_entity.data.root_quat_w.shape[0], device=env.device)

    # Validate upright_percentage shape if it's a tensor
    if isinstance(upright_percentage, torch.Tensor):
        num_envs = len(env_ids)
        if upright_percentage.numel() != num_envs:
            raise ValueError(
                f"upright_percentage tensor must have {num_envs} elements to match env_ids, "
                f"but got {upright_percentage.numel()} elements"
            )

    object_quat = object_entity.data.root_quat_w[env_ids]
    object_pos = object_entity.data.root_pos_w[env_ids]

    target_quat = _compute_target_quaternions(
        object_quat=object_quat,
        upright_percentage=upright_percentage,
        upright_axis_name=upright_axis_name,
    )

    pose_tensor = torch.cat([object_pos, target_quat], dim=-1)
    object_entity.write_root_pose_to_sim(pose_tensor, env_ids=env_ids)
    zero_velocity = torch.zeros((env_ids.numel(), 6), device=device, dtype=dtype)
    object_entity.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)


def _compute_target_quaternions(
    object_quat: torch.Tensor,
    upright_percentage: float | torch.Tensor,
    upright_axis_name: str,
) -> torch.Tensor:
    """Compute the target quaternions for the object given the current orientation and the upright percentage.

    Args:
        object_quat: Current quaternion orientations of the objects. Shape: [num_envs, 4] (w, x, y, z).
        upright_percentage: Target upright percentage(s) in [0, 1]. Can be:
            - A scalar float applied to all environments
            - A tensor with num_envs elements (shape [num_envs] or [num_envs, 1])
            Value of 0.0 creates perpendicular orientation (90° to world +Z) in the plane spanned by the object_upright_axis and world +Z,
            value of 1.0 fully aligns with world +Z (0°).
        upright_axis_name: Name of the object's local axis to align ("x", "y", or "z").

    Returns:
        Target quaternions for the object. Shape: [num_envs, 4] (w, x, y, z).
    """

    upright_percentage_t = torch.as_tensor(upright_percentage, device=object_quat.device, dtype=object_quat.dtype)
    if upright_percentage_t.dim() == 0:
        upright_percentage_t = upright_percentage_t.expand(object_quat.shape[0])

    current_axis = get_object_axis_in_world_frame(object_quat, upright_axis_name)

    world_up = torch.zeros_like(current_axis)
    world_up[:, 2] = 1.0

    horizontal = current_axis - torch.sum(current_axis * world_up, dim=-1, keepdim=True) * world_up
    horizontal_norm = horizontal.norm(dim=-1, keepdim=True)
    fallback_axis = torch.zeros_like(horizontal)
    fallback_axis[:, 0] = 1.0

    # if current_axis is parallel to world up, use the fallback axis to compute the rotation axis
    needs_fallback = horizontal_norm.squeeze(-1) < 1e-6
    if needs_fallback.any():
        horizontal[needs_fallback] = fallback_axis[needs_fallback]
        horizontal_norm[needs_fallback] = horizontal[needs_fallback].norm(dim=-1, keepdim=True)
    horizontal_dir = horizontal / horizontal_norm.clamp(min=1e-6)

    target_angles = (1.0 - upright_percentage_t).view(-1, 1) * (math.pi / 2.0)
    target_axis = torch.sin(target_angles) * horizontal_dir + torch.cos(target_angles) * world_up
    target_axis = torch.nn.functional.normalize(target_axis, dim=-1)

    dot_product = torch.sum(current_axis * target_axis, dim=-1).clamp(-1.0, 1.0)
    rotation_angle = torch.acos(dot_product)

    rotation_axis = torch.cross(current_axis, target_axis, dim=-1)
    axis_norm = rotation_axis.norm(dim=-1, keepdim=True)

    # if current_axis is parallel to target_axis, use the fallback axis to compute the rotation axis
    needs_fallback = axis_norm.squeeze(-1) < 1e-6
    if needs_fallback.any():
        current_axis_fb = current_axis[needs_fallback]
        fallback_axis_fb = fallback_axis[needs_fallback]
        near_parallel_x = torch.abs(current_axis_fb[:, 0]) > 0.9
        if near_parallel_x.any():
            fallback_axis_fb[near_parallel_x] = torch.tensor(
                [0.0, 1.0, 0.0], device=object_quat.device, dtype=object_quat.dtype
            )
        rotation_axis[needs_fallback] = torch.cross(current_axis_fb, fallback_axis_fb, dim=-1)
        axis_norm[needs_fallback] = rotation_axis[needs_fallback].norm(dim=-1, keepdim=True)

    axis_norm = axis_norm.clamp(min=1e-6)
    rotation_axis_unit = rotation_axis / axis_norm

    align_quat = math_utils.quat_from_angle_axis(rotation_angle, rotation_axis_unit)

    new_quat = math_utils.quat_mul(align_quat, object_quat)
    new_quat = torch.nn.functional.normalize(new_quat, dim=-1)
    return new_quat
