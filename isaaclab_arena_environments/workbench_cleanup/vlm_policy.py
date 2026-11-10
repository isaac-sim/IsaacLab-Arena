# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Workbench VLM goals with measured geometry and gradual DROID pose commands."""

import json
import torch
from collections import deque
from dataclasses import dataclass

from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_apply_inverse,
    quat_slerp,
    subtract_frame_transforms,
)

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena.tasks.predicates.spatial import object_bounds_center_over_destination
from isaaclab_arena_vlm_agent.object_pose_adapter import ObjectPoseObservationAdapter
from isaaclab_arena_vlm_agent.observation_adapter import ImageStateObservationAdapter
from isaaclab_arena_vlm_agent.vlm_agent_policy import VLMGoalPolicy, VLMGoalPolicyCfg


class WorkbenchObservationAdapter(ObjectPoseObservationAdapter):
    """Add privileged joint positions and local bounds to camera, robot, and object observations."""

    def __init__(self, scene_keys):
        super().__init__(ImageStateObservationAdapter(), scene_keys)
        self._previous_grasps = {}

    def reset(self, env_ids=None):
        """Discard motion comparisons for new episodes."""
        if env_ids is None:
            self._previous_grasps.clear()
        else:
            for env_id in env_ids.reshape(-1).tolist():
                self._previous_grasps.pop(env_id, None)

    def extract_vlm_input_from_observation(self, env, observation, env_id):
        images, state = super().extract_vlm_input_from_observation(env, observation, env_id)
        base = env.unwrapped
        bounds_by_object = {}
        for name in self.scene_keys:
            if name not in base.scene.articulations:
                bounds = base.arena_world.get_aabb_in_local_frame(name)
                bounds_by_object[name] = {
                    "min_object_frame_m": bounds.min_point[env_id].tolist(),
                    "max_object_frame_m": bounds.max_point[env_id].tolist(),
                }
        joints = {}
        for name in ("top_drawer", "socket_set", "oscilloscope"):
            asset = base.scene[name]
            joints[name] = dict(zip(asset.joint_names, asset.data.joint_pos.torch[env_id].tolist()))
        state["simulator_ground_truth_assistance"] = {
            "local_object_bounds": bounds_by_object,
            "joint_positions_m_or_rad": joints,
            "task_complete": bool(base.progress_tracker.is_complete()[env_id]),
            "note": "Privileged simulator measurements. Bounds use each object's local frame, not the robot frame.",
        }
        if "white_bin" in self.scene_keys:
            state["simulator_ground_truth_assistance"]["bin_grasp"] = self._bin_grasp_measurements(base, env_id)
        if "socket_set_lid" in self.scene_keys:
            state["simulator_ground_truth_assistance"]["lid"] = self._lid_contact_geometry(base, env_id)
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

    def _bin_grasp_measurements(self, base, env_id):
        world = base.arena_world
        selected = slice(env_id, env_id + 1)
        T_W_R = world.get_pose_w("robot")[selected]
        T_W_B = world.get_pose_w("white_bin")[selected]
        t_R_B, q_R_B = subtract_frame_transforms(T_W_R[:, :3], T_W_R[:, 3:], T_W_B[:, :3], T_W_B[:, 3:])
        T_R_B = torch.cat((t_R_B, q_R_B), dim=-1)
        T_R_G = world.get_body_pose_in_root("robot", "base_link")[selected]
        t_G_B, _ = subtract_frame_transforms(T_R_G[:, :3], T_R_G[:, 3:], t_R_B, q_R_B)
        measurements = {}
        previous = self._previous_grasps.get(env_id)
        if previous is not None:
            previous_T_R_G, previous_T_R_B, previous_t_G_B = previous
            measurements.update({
                "gripper_motion_since_last_request_m": torch.linalg.vector_norm(
                    T_R_G[:, :3] - previous_T_R_G[:, :3]
                ).item(),
                "bin_motion_since_last_request_m": torch.linalg.vector_norm(t_R_B - previous_T_R_B[:, :3]).item(),
                "bin_to_gripper_translation_drift_m": torch.linalg.vector_norm(t_G_B - previous_t_G_B).item(),
            })
        self._previous_grasps[env_id] = (T_R_G.clone(), T_R_B.clone(), t_G_B.clone())
        for label, frame in (("left_pad", "tool_leftfinger"), ("right_pad", "tool_rightfinger")):
            t_W_pad = world.get_frame_position_w("ee_frame", frame)[selected]
            t_B_pad, _ = subtract_frame_transforms(T_W_B[:, :3], T_W_B[:, 3:], t_W_pad)
            measurements[f"{label}_in_bin_frame_m"] = t_B_pad[0].tolist()
        measurements["note"] = (
            "After a test lift, the bin should follow the gripper with little relative drift. "
            "A closed gripper or reached wrist goal alone does not prove a grasp."
        )
        bounds_B = world.get_aabb_in_local_frame("white_bin")
        candidates = bin_side_wall_gripper_poses(T_R_B, bounds_B.min_point[selected], bounds_B.max_point[selected])
        measurements["side_wall_contact_estimates_root_xyz_xyzw"] = {
            name: pose[0].tolist() for name, pose in candidates.items()
        }
        bounds_W_bin = world.get_aabb_w("white_bin")
        tabletop_top_W = world.get_aabb_w("tabletop").max_point[env_id, 2]
        front_edges_W = [world.get_aabb_w("overhead_shelf").max_point[env_id, 0]]
        for index in range(1, 5):
            front_edges_W.append(world.get_aabb_w(f"blue_container_{index:02d}").max_point[env_id, 0])
        measurements["back_edge_clearance_from_overhead_front_m"] = (
            bounds_W_bin.min_point[env_id, 0] - torch.stack(front_edges_W).amax()
        ).item()
        measurements["lowest_point_above_tabletop_m"] = (bounds_W_bin.min_point[env_id, 2] - tabletop_top_W).item()
        measurements["bounds_center_over_shelf"] = bool(
            object_bounds_center_over_destination(
                world.get_centroid_w("white_bin"),
                world.get_pose_w("overhead_shelf"),
                world.get_aabb_in_local_frame("overhead_shelf"),
            )[env_id]
        )
        # This is the seating region used by the existing force recipe, expressed in the shelf's frame.
        T_W_S = world.get_pose_w("overhead_shelf")[selected]
        t_S_target = T_W_S.new_tensor([[-0.05, 0.23, 1.22]])
        q_S_target = T_W_S.new_tensor([[0.0, 0.13052619, 0.0, 0.99144486]])
        t_W_target, q_W_target = combine_frame_transforms(T_W_S[:, :3], T_W_S[:, 3:], t_S_target, q_S_target)
        t_R_target, q_R_target = subtract_frame_transforms(T_W_R[:, :3], T_W_R[:, 3:], t_W_target, q_W_target)
        T_R_target = torch.cat((t_R_target, q_R_target), dim=-1)
        measurements["bin_seating_estimate_root_xyz_xyzw"] = T_R_target[0].tolist()
        if T_W_B[0, 2] > tabletop_top_W + 0.05:
            measurements["gripper_seating_estimate_if_grasp_preserved_root_xyz_xyzw"] = gripper_pose_for_held_object(
                T_R_B, T_R_G, T_R_target
            )[0].tolist()
            t_S_front = t_S_target + t_S_target.new_tensor([[0.065, 0.0, -0.0174167]])
            t_W_front, q_W_front = combine_frame_transforms(T_W_S[:, :3], T_W_S[:, 3:], t_S_front, q_S_target)
            t_R_front, q_R_front = subtract_frame_transforms(T_W_R[:, :3], T_W_R[:, 3:], t_W_front, q_W_front)
            T_R_front = torch.cat((t_R_front, q_R_front), dim=-1)
            measurements["front_bay_gripper_seating_estimate_root_xyz_xyzw"] = gripper_pose_for_held_object(
                T_R_B, T_R_G, T_R_front
            )[0].tolist()
        measurements["seating_note"] = (
            "An approximate destination, not a path or proof of support. Establish a grasp, pull the bin forward "
            "to clear the overhead structures, lift above the shelf edge, then transfer and lower. "
            "The gripper estimate assumes the current bin-to-gripper transform stays fixed; recheck after slip."
        )
        return measurements


@dataclass
class WorkbenchVLMGoalPolicyCfg(VLMGoalPolicyCfg):
    """Configure gradual pose tracking and recent-goal observations for the workbench."""

    goal_history_size: int = 12
    """Number of completed goals included in the next inference request."""

    translation_speed_m_s: float = 0.15
    """Maximum translation speed of the commanded pose trajectory."""

    rotation_speed_rad_s: float = 0.8
    """Maximum angular speed of the commanded pose trajectory."""

    max_tracking_translation_m: float = 0.08
    """Maximum target displacement from the measured gripper during contact."""

    max_tracking_rotation_rad: float = 0.35
    """Maximum target rotation error from the measured gripper during contact."""

    def __post_init__(self):
        super().__post_init__()
        assert self.vlm_embodiment_adapter == "droid"
        assert self.observation_modality.value == "image_state"
        assert self.object_poses
        assert type(self.goal_history_size) is int and self.goal_history_size > 0
        assert self.translation_speed_m_s > 0 and self.rotation_speed_rad_s > 0
        assert self.max_tracking_translation_m > 0 and 0 < self.max_tracking_rotation_rad <= 3.141593


@register_policy
class WorkbenchVLMGoalPolicy(VLMGoalPolicy, PolicyBase[WorkbenchVLMGoalPolicyCfg]):
    """Track VLM-selected DROID goals gradually, with joint feedback and recent history."""

    name = "workbench_vlm_goal"

    def __init__(self, config: WorkbenchVLMGoalPolicyCfg):
        super().__init__(config, observation_adapter=WorkbenchObservationAdapter(config.object_poses))
        self._goal_history = {}
        self._pose_targets = {}
        self._pose_goals = {}

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self.observation_adapter.reset(env_ids)
        for values in (self._goal_history, self._pose_targets, self._pose_goals):
            if env_ids is None:
                values.clear()
            else:
                for env_id in env_ids.reshape(-1).tolist():
                    values.pop(env_id, None)

    def _get_request_content(self, env, observation, env_id):
        content = super()._get_request_content(env, observation, env_id)
        history = self._goal_history.setdefault(env_id, deque(maxlen=self.config.goal_history_size))
        previous = self._previous_goals.get(env_id)
        if previous is not None:
            history.append(dict(previous))
        content.insert(1, {"type": "text", "text": json.dumps({"recent_goals": list(history)})})
        return content

    def get_action(self, env, observation):
        actions = super().get_action(env, observation)
        base = env.unwrapped
        T_R_G = base.arena_world.get_body_pose_in_root("robot", "base_link")
        active_mask = getattr(base, "active_episode_mask", None)
        for env_id in range(base.num_envs):
            if active_mask is not None and not active_mask[env_id] and env_id in self._pose_targets:
                actions[env_id, :7] = self._pose_targets[env_id]
                continue
            T_R_goal = actions[env_id, :7].clone()
            previous_goal = self._pose_goals.get(env_id)
            if previous_goal is None or not torch.equal(previous_goal, T_R_goal):
                self._pose_targets[env_id] = T_R_G[env_id].clone()
                self._pose_goals[env_id] = T_R_goal.clone()
            T_R_target = limit_pose_increment(
                self._pose_targets[env_id],
                T_R_goal,
                self.config.translation_speed_m_s * base.step_dt,
                self.config.rotation_speed_rad_s * base.step_dt,
            )
            T_R_target = limit_pose_increment(
                T_R_G[env_id],
                T_R_target,
                self.config.max_tracking_translation_m,
                self.config.max_tracking_rotation_rad,
            )
            self._pose_targets[env_id] = T_R_target.clone()
            actions[env_id, :7] = T_R_target
        return actions


def limit_pose_increment(T_R_G, T_R_goal, translation_step_m, rotation_step_rad):
    """Step toward a pose target with bounded translation and shortest-arc rotation."""
    T_R_command = T_R_goal.clone()
    delta_R = T_R_goal[:3] - T_R_G[:3]
    distance = torch.linalg.vector_norm(delta_R).item()
    T_R_command[:3] = T_R_G[:3] + delta_R * min(1.0, translation_step_m / max(distance, 1e-9))
    dot = torch.dot(T_R_G[3:], T_R_goal[3:]).abs().clamp(0, 1)
    angle = (2 * torch.acos(dot)).item()
    fraction = min(1.0, rotation_step_rad / max(angle, 1e-9))
    T_R_command[3:] = quat_slerp(T_R_G[3:], T_R_goal[3:], fraction)
    return T_R_command


def gripper_pose_for_held_object(T_R_B, T_R_G, T_R_target):
    """Return a gripper pose that preserves its measured transform relative to the bin at the target."""
    t_B_G, q_B_G = subtract_frame_transforms(T_R_B[:, :3], T_R_B[:, 3:], T_R_G[:, :3], T_R_G[:, 3:])
    t_R_G, q_R_G = combine_frame_transforms(T_R_target[:, :3], T_R_target[:, 3:], t_B_G, q_B_G)
    return torch.cat((t_R_G, q_R_G), dim=-1)


def bin_side_wall_gripper_poses(T_R_B, minimum_B, maximum_B):
    """Estimate oblique gripper contact poses on the two side walls from live bin bounds."""
    size_B = maximum_B - minimum_B
    center_B = (minimum_B + maximum_B) / 2
    q_B_G = T_R_B.new_tensor([[-0.5, 0.0, 0.866025404, 0.0]]).expand(len(T_R_B), -1)
    t_G_pad = T_R_B.new_tensor([[0.1446, 0.0, 0.0]]).expand(len(T_R_B), -1)
    candidates = {}
    for name, side in (("positive_y", 1), ("negative_y", -1)):
        t_B_pad = center_B.clone()
        t_B_pad[:, 0] += (0.070 / 0.15) * (size_B[:, 0] / 2 - 0.004)
        t_B_pad[:, 1] += side * (0.102 / 0.11) * (size_B[:, 1] / 2 - 0.004)
        t_B_pad[:, 2] = maximum_B[:, 2] - torch.minimum(size_B[:, 2] * 0.3, size_B.new_tensor(0.025))
        t_B_G = t_B_pad - quat_apply(q_B_G, t_G_pad)
        t_R_G, q_R_G = combine_frame_transforms(T_R_B[:, :3], T_R_B[:, 3:], t_B_G, q_B_G)
        candidates[name] = torch.cat((t_R_G, q_R_G), dim=-1)
    return candidates
