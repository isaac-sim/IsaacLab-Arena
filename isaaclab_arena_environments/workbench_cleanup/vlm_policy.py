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
from isaaclab_arena_vlm_agent.joint_position_adapter import JointPositionObservationAdapter
from isaaclab_arena_vlm_agent.object_pose_adapter import ObjectPoseObservationAdapter
from isaaclab_arena_vlm_agent.observation_adapter import ImageStateObservationAdapter, VLMObservationAdapter
from isaaclab_arena_vlm_agent.vlm_agent_policy import ObservationModality, VLMGoalPolicy, VLMGoalPolicyCfg

# Derived from the simready_central_socket_set asset's authored hinge and lid geometry.
_SOCKET_SET_CASE_BODY = "Case_base"
_SOCKET_SET_HINGE_POSITION_CASE_M = (0.0, 0.074, 0.03)
_SOCKET_SET_HINGE_AXIS_CASE = (-1.0, 0.0, 0.0)
_SOCKET_SET_CONTACT_INSET_FROM_LID_EDGE_M = 0.012
_SOCKET_SET_CONTACT_OFFSET_ABOVE_LID_M = 0.004


class WorkbenchObservationAdapter(VLMObservationAdapter):
    """Add lid contact geometry to image, state, object-pose, and joint-position observations."""

    def __init__(self, object_poses, joint_positions):
        observation_adapter = ImageStateObservationAdapter()
        observation_adapter = ObjectPoseObservationAdapter(observation_adapter, object_poses)
        observation_adapter = JointPositionObservationAdapter(observation_adapter, joint_positions)
        self.observation_adapter = observation_adapter

    def extract_vlm_input_from_observation(self, env, observation, env_id):
        images, state = self.observation_adapter.extract_vlm_input_from_observation(env, observation, env_id)
        base = env.unwrapped
        assistance = dict(state["simulator_ground_truth_assistance"])
        assistance["lid"] = self._lid_contact_geometry(base, env_id)
        state["simulator_ground_truth_assistance"] = assistance
        return images, state

    def _lid_contact_geometry(self, base, env_id):
        """Return the lid contact point, outward normal, and closing direction in the robot root frame."""
        world = base.arena_world
        selected = slice(env_id, env_id + 1)
        T_W_R = world.get_pose_w("robot")[selected]
        T_W_C = world.get_body_pose_w("socket_set", _SOCKET_SET_CASE_BODY)[selected]
        T_W_L = world.get_pose_w("socket_set_lid")[selected]
        t_C_hinge = T_W_C.new_tensor([_SOCKET_SET_HINGE_POSITION_CASE_M])
        axis_C = T_W_C.new_tensor([_SOCKET_SET_HINGE_AXIS_CASE])
        t_W_hinge = T_W_C[:, :3] + quat_apply(T_W_C[:, 3:], t_C_hinge)
        axis_W = quat_apply(T_W_C[:, 3:], axis_C)
        bounds_L = world.get_aabb_in_local_frame("socket_set_lid")
        t_L_contact = bounds_L.center[selected].clone()
        t_L_contact[:, 1] = bounds_L.min_point[selected, 1] + _SOCKET_SET_CONTACT_INSET_FROM_LID_EDGE_M
        t_L_contact[:, 2] = bounds_L.max_point[selected, 2] + _SOCKET_SET_CONTACT_OFFSET_ABOVE_LID_M
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
    """Configure WorkbenchVLMGoalPolicy with the standard VLM goal settings."""


@register_policy
class WorkbenchVLMGoalPolicy(VLMGoalPolicy, PolicyBase[WorkbenchVLMGoalPolicyCfg]):
    """Use the standard VLM goal controller with privileged workbench contact observations."""

    name = "workbench_vlm_goal"

    def __init__(self, config: WorkbenchVLMGoalPolicyCfg):
        assert (
            config.observation_modality == ObservationModality.image_state
        ), "WorkbenchVLMGoalPolicy always sends images and robot state; set observation_modality: image_state"
        super().__init__(
            config,
            observation_adapter=WorkbenchObservationAdapter(config.object_poses, config.joint_positions),
        )
