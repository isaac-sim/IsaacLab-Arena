# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""VLM observations for the workbench's articulated contact tasks."""

import torch
from dataclasses import dataclass

from isaaclab.utils.math import quat_apply, quat_apply_inverse, subtract_frame_transforms

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_vlm_agent.object_pose_adapter import ObjectPoseObservationAdapter
from isaaclab_arena_vlm_agent.observation_adapter import ImageStateObservationAdapter
from isaaclab_arena_vlm_agent.vlm_agent_policy import VLMGoalPolicy, VLMGoalPolicyCfg


class WorkbenchObservationAdapter(ObjectPoseObservationAdapter):
    """Add selected articulation joints and lid contact geometry to image and object-pose observations."""

    def __init__(self, scene_keys):
        super().__init__(ImageStateObservationAdapter(), scene_keys)

    def extract_vlm_input_from_observation(self, env, observation, env_id):
        images, state = super().extract_vlm_input_from_observation(env, observation, env_id)
        base = env.unwrapped
        joints = {}
        for name in self.scene_keys:
            if name in base.scene.articulations:
                asset = base.scene[name]
                joints[name] = dict(zip(asset.joint_names, asset.data.joint_pos.torch[env_id].tolist()))
        assistance = {"joint_positions_m_or_rad": joints}
        if "socket_set_lid" in self.scene_keys:
            assistance["lid"] = self._lid_contact_geometry(base, env_id)
        state["simulator_ground_truth_assistance"] = assistance
        return images, state

    def _lid_contact_geometry(self, base, env_id):
        world = base.arena_world
        selected = slice(env_id, env_id + 1)
        T_W_R = world.get_pose_w("robot")[selected]
        T_W_C = world.get_body_pose_w("socket_set", "Case_base")[selected]
        T_W_L = world.get_pose_w("socket_set_lid")[selected]
        t_C_hinge = T_W_C.new_tensor([[0.0, 0.074, 0.03]])
        axis_C = T_W_C.new_tensor([[-1.0, 0.0, 0.0]])
        t_W_hinge = T_W_C[:, :3] + quat_apply(T_W_C[:, 3:], t_C_hinge)
        axis_W = quat_apply(T_W_C[:, 3:], axis_C)
        bounds_L = world.get_aabb_in_local_frame("socket_set_lid")
        t_L_contact = bounds_L.center[selected].clone()
        t_L_contact[:, 1] = bounds_L.min_point[selected, 1] + 0.012
        t_L_contact[:, 2] = bounds_L.max_point[selected, 2] + 0.004
        t_W_contact = T_W_L[:, :3] + quat_apply(T_W_L[:, 3:], t_L_contact)
        tangent_W = -torch.linalg.cross(axis_W, t_W_contact - t_W_hinge)
        tangent_W /= torch.linalg.vector_norm(tangent_W, dim=-1, keepdim=True).clamp_min(1e-9)
        tangent_R = quat_apply_inverse(T_W_R[:, 3:], tangent_W)
        t_R_contact, _ = subtract_frame_transforms(T_W_R[:, :3], T_W_R[:, 3:], t_W_contact)
        normal_W = quat_apply(T_W_L[:, 3:], T_W_L.new_tensor([[0.0, 0.0, 1.0]]))
        normal_R = quat_apply_inverse(T_W_R[:, 3:], normal_W)
        return {
            "outer_face_contact_root_m": t_R_contact[0].tolist(),
            "outward_normal_root": normal_R[0].tolist(),
            "closing_tangent_root": tangent_R[0].tolist(),
            "note": (
                "This is a fingertip contact point, not a wrist goal. Approach from outside the outer face. "
                "Subtract the rotated closed-tip offset to locate base_link, then sweep along the closing tangent."
            ),
        }


@dataclass
class WorkbenchVLMGoalPolicyCfg(VLMGoalPolicyCfg):
    """Register the workbench observation adapter with the standard VLM goal configuration."""


@register_policy
class WorkbenchVLMGoalPolicy(VLMGoalPolicy, PolicyBase[WorkbenchVLMGoalPolicyCfg]):
    """Use the standard VLM goal controller with privileged workbench contact observations."""

    name = "workbench_vlm_goal"

    def __init__(self, config: WorkbenchVLMGoalPolicyCfg):
        assert config.vlm_embodiment_adapter == "droid"
        assert config.observation_modality.value == "image_state"
        assert config.object_poses
        super().__init__(config, observation_adapter=WorkbenchObservationAdapter(config.object_poses))
