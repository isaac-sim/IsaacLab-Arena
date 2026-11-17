# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task 10: the cart someone moved.

The scene of task 1 (a mug on the counter, a wall cabinet with an open door) plus one movable element: a tall cart
(a post standing on the counter) that is either in the lane between the mug and the cabinet opening or set aside. The
robot sees the scene during an initial look. While it grasps the mug, the cart may be moved to the other spot. The
wrist image is withheld after the look, except for a short re-observation window that opens when the cart is first
back in view after the move. The robot must carry the mug over the cart if it is in the lane, or straight to the
cabinet if it is not, and put the mug on the lower shelf.

Hidden: the cart's new pose, apart from the re-observation window (the frames are withheld).
Randomised: the scene of task 1 (lower shelf only), the cart's old spot, whether it moves (probability 0.8).
Difficulty knob: length of the re-observation window in seconds (shorter is harder).
Success: as task 1, with no collision at the new pose and no detour around the old one: the tool path from the grasp
until the end is at most 10% (plus 4 cm) longer than the oracle route for the new pose.
"""

from __future__ import annotations

import math
import torch

from isaaclab_arena_environments.spatial_memory.controller import GRIPPER_CLOSE, GRIPPER_OPEN, Waypoint, pitched_quat
from isaaclab_arena_environments.spatial_memory.geometry import ALL_FACES
from isaaclab_arena_environments.spatial_memory.layout_utils import pose, uniform
from isaaclab_arena_environments.spatial_memory.parts import PartSpec, board
from isaaclab_arena_environments.spatial_memory.runtime import NOT_SET
from isaaclab_arena_environments.spatial_memory.task_logic import LayoutSample, PlanMode
from isaaclab_arena_environments.spatial_memory.tasks.mug_past_door import (
    APPROACH_PITCH_RAD,
    GRASP_HEIGHT_M,
    INSERT_DEPTH_M,
    MUG_HEIGHT_M,
    REST_JOINT_POSE,
    TRANSIT_HEIGHT_M,
    MugPastDoorLogic,
)
from isaaclab_arena_environments.spatial_memory.visibility import VisibilityTarget

CART_SIZE_M = (0.06, 0.06, 0.32)
"""The cart is a tall post, higher than the straight carry route."""
CART_ASIDE_XY = (0.60, 0.50)
"""Spot on the counter away from every route."""
CART_LANE_FRACTION = 0.55
"""Position of the lane spot along the straight carry route, from the opening (0) to the grasp (1)."""
MOVE_PROBABILITY = 0.8
LOOK_S = 2.0
"""Seconds of camera frames at the start of the episode."""
OVER_CLEARANCE_M = 0.08
"""Clearance of the mug's base above the cart when the route goes over it."""
PATH_TOLERANCE = 0.10
PATH_MARGIN_M = 0.04
PATH_STEP_M = 0.01
"""The carry path is measured in steps of at least this length, so that jitter while holding still does not add up."""

__all__ = ["MovedCartLogic", "REST_JOINT_POSE"]


class MovedCartLogic(MugPastDoorLogic):
    """Logic of task 10 (see module docstring)."""

    name = "spatial_memory_moved_cart"
    difficulty_name = "reobservation_window_s"
    default_difficulty = 1.0
    episode_length_s = 40.0
    task_description = "Pick up the mug and put it on the lower shelf inside the open cabinet."
    withholds_camera_frames = True

    def door_angle_deg(self) -> float:
        return 90.0

    def build_part_specs(self) -> list[PartSpec]:
        return super().build_part_specs() + [board("cart", CART_SIZE_M, colour="blue")]

    def sample_layout(self, generator: torch.Generator) -> LayoutSample:
        sample = super().sample_layout(generator)
        poses = sample.part_poses_e
        params = sample.params
        names = [spec.name for spec in self.part_specs()]
        # Lower shelf only, and the mug further left than in task 1 so that the cart fits between mug and cabinet.
        params["support_z"] = torch.tensor(self._lower_support_z(poses, names))
        params["target_compartment"] = torch.tensor(0)
        mug_x = uniform(generator, 0.34, 0.40)
        mug_y = uniform(generator, 0.38, 0.44)
        poses[names.index("mug")] = torch.tensor(pose(mug_x, mug_y, 0.5 * MUG_HEIGHT_M + 0.001))
        params["mug_start_xy"] = torch.tensor([mug_x, mug_y])
        # The lane spot is on the straight carry route, a little nearer the grasp than the opening, where the arm
        # comes down.
        start_xy, end_xy = self._carry_ends_xy(params)
        lane_xy = tuple(end + CART_LANE_FRACTION * (start - end) for start, end in zip(start_xy, end_xy))
        old_in_lane = float(torch.rand(1, generator=generator)) < 0.5
        moves = float(torch.rand(1, generator=generator)) < MOVE_PROBABILITY
        new_in_lane = old_in_lane != moves
        old_xy = lane_xy if old_in_lane else CART_ASIDE_XY
        new_xy = lane_xy if new_in_lane else CART_ASIDE_XY
        poses[names.index("cart")] = torch.tensor(pose(old_xy[0], old_xy[1], 0.5 * CART_SIZE_M[2]))
        params["cart_old_xy"] = torch.tensor(old_xy)
        params["cart_new_xy"] = torch.tensor(new_xy)
        params["cart_old_in_lane"] = torch.tensor(old_in_lane)
        params["cart_new_in_lane"] = torch.tensor(new_in_lane)
        params["cart_moves"] = torch.tensor(moves)
        params["reference_path_m"] = torch.tensor(self._carry_path_length(params, new_in_lane))
        return LayoutSample(part_poses_e=poses, params=params)

    def _lower_support_z(self, poses: torch.Tensor, names: list[str]) -> float:
        """Return the top of the cabinet's bottom board."""
        from isaaclab_arena_environments.spatial_memory.layout_utils import BOARD_T_M

        return float(poses[names.index("cab_bottom")][2]) + 0.5 * BOARD_T_M

    # ----------------------------------------------------------------------------------------------------------------

    def visibility_targets(self) -> list[VisibilityTarget]:
        return [VisibilityTarget("cart", "cart", ALL_FACES, spacing_m=0.03)]

    def hidden_target_names(self) -> list[str]:
        return ["cart"]

    def on_reset(self, env, runtime, env_ids: torch.Tensor) -> None:
        state = runtime.task_state
        if "moved" not in state:
            state["moved"] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            state["window_start_step"] = torch.full((env.num_envs,), NOT_SET, dtype=torch.long, device=env.device)
            state["carry_path_m"] = torch.zeros(env.num_envs, device=env.device)
            state["path_anchor_w"] = torch.zeros(env.num_envs, 3, device=env.device)
        state["moved"][env_ids] = False
        state["window_start_step"][env_ids] = NOT_SET
        state["carry_path_m"][env_ids] = 0.0

    def update_frame_withholding(self, env, runtime, visible: torch.Tensor, active: torch.Tensor) -> None:
        # The re-observation window opens the first time the cart is in view after the move.
        state = runtime.task_state
        cart_visible = visible[:, runtime.target_names.index("cart")]
        opening = state["moved"] & (state["window_start_step"] == NOT_SET) & cart_visible & active
        state["window_start_step"][opening] = env.episode_length_buf[opening]

    def _window_open(self, env, runtime) -> torch.Tensor:
        start = runtime.task_state["window_start_step"]
        window_steps = int(round(self.difficulty / runtime.step_dt))
        elapsed = env.episode_length_buf - start
        return (start != NOT_SET) & (elapsed < window_steps)

    def camera_blocked(self, env, runtime) -> torch.Tensor:
        if "moved" not in runtime.task_state:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        look = env.episode_length_buf < int(round(LOOK_S / runtime.step_dt))
        return ~look & ~self._window_open(env, runtime)

    def on_step(self, env, runtime, visible: torch.Tensor, active: torch.Tensor) -> None:
        # The cart is moved (or left where it is) when the robot first grasps, i.e. while it looks at the mug.
        state = runtime.task_state
        tool_w = self.tool_pose_w(env)[:, :3]
        step = (tool_w - state["path_anchor_w"]).norm(dim=-1)
        advance = state["moved"] & active & (step >= PATH_STEP_M)
        state["carry_path_m"] += torch.where(advance, step, torch.zeros_like(step))
        state["path_anchor_w"][advance] = tool_w[advance]
        grasped = (runtime.first_close_step != NOT_SET) & ~state["moved"] & active
        if grasped.any():
            env_ids = grasped.nonzero().squeeze(-1)
            cart = env.scene["cart"]
            new_xy = runtime.params["cart_new_xy"][env_ids]
            pose_w = cart.data.root_pose_w.torch[env_ids].clone()
            pose_w[:, :2] = new_xy + env.scene.env_origins[env_ids, :2]
            cart.write_root_pose_to_sim(pose_w, env_ids=env_ids)
            cart.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)
            state["moved"] |= grasped
            state["path_anchor_w"][env_ids] = tool_w[env_ids]

    def action_phase(self, env, runtime) -> torch.Tensor:
        return runtime.task_state["moved"]

    def hidden_phase_mask(self, env, runtime, in_action: torch.Tensor) -> torch.Tensor:
        hidden = in_action & ~self._window_open(env, runtime)
        return hidden.unsqueeze(-1).expand(-1, len(runtime.target_names))

    def static_part_mask(self, env, runtime) -> torch.Tensor:
        mask = super().static_part_mask(env, runtime).clone()
        mask[:, runtime.part_index["cart"]] = False
        return mask

    def success_terms(self, env, runtime) -> dict[str, torch.Tensor]:
        terms = super().success_terms(env, runtime)
        carry_path = runtime.task_state["carry_path_m"]
        limit = (1.0 + PATH_TOLERANCE) * runtime.params["reference_path_m"] + PATH_MARGIN_M
        terms["no_detour"] = runtime.task_state["moved"] & (carry_path <= limit)
        return terms

    # ----------------------------------------------------------------------------------------------------------------

    def _carry_ends_xy(self, params) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the start (above the mug) and end (in front of the opening) of the carry route."""
        mug_x, mug_y = (float(v) for v in params["mug_start_xy"])
        front_x = float(params["front_x"])
        lane_y = float(params["center_y"]) + 0.06
        return (mug_x, mug_y), (front_x - 0.14, lane_y)

    def _carry_route(self, params, over_cart: bool) -> list[tuple[float, float, float]]:
        """Return the tool positions of the carry route from the grasp to the withdrawn pose."""
        (mug_x, mug_y), (end_x, lane_y) = self._carry_ends_xy(params)
        front_x = float(params["front_x"])
        support_z = float(params["support_z"])
        place_z = support_z + GRASP_HEIGHT_M + 0.004
        carry_z = place_z + 0.015
        place_x = front_x + INSERT_DEPTH_M
        transit_z = max(carry_z, TRANSIT_HEIGHT_M)
        if over_cart:
            transit_z = max(transit_z, CART_SIZE_M[2] + OVER_CLEARANCE_M + GRASP_HEIGHT_M)
        return [
            (mug_x, mug_y, GRASP_HEIGHT_M),
            (mug_x, mug_y, transit_z),
            (end_x, lane_y, transit_z),
            (front_x - 0.12, lane_y, carry_z),
            (place_x, lane_y, carry_z),
            (place_x, lane_y, place_z),
            (front_x - 0.12, lane_y, place_z),
        ]

    def _carry_path_length(self, params, over_cart: bool) -> float:
        route = self._carry_route(params, over_cart)
        return sum(math.dist(a, b) for a, b in zip(route[:-1], route[1:]))

    def plan(self, env, runtime, env_id: int, mode: str) -> list[Waypoint]:
        params = {key: value[env_id] for key, value in runtime.params.items()}
        # The oracle knows where the cart will be; the blind controller plans for the pose it saw at the start.
        key = "cart_new_in_lane" if mode == PlanMode.ORACLE else "cart_old_in_lane"
        route = self._carry_route(params, bool(params[key]))
        forward = pitched_quat(0.0, APPROACH_PITCH_RAD)
        grasp, lift, transit, preinsert, insert, lower, withdraw = route
        plan = super().plan(env, runtime, env_id, mode)
        close_index = next(index for index, waypoint in enumerate(plan) if waypoint.label == "close")
        return plan[: close_index + 1] + [
            Waypoint(lift, forward, GRIPPER_CLOSE, label="lift", max_speed_mps=0.1),
            Waypoint(transit, forward, GRIPPER_CLOSE, label="transit", max_speed_mps=0.2),
            Waypoint(preinsert, forward, GRIPPER_CLOSE, label="preinsert", max_speed_mps=0.15),
            Waypoint(insert, forward, GRIPPER_CLOSE, label="insert", max_speed_mps=0.1),
            Waypoint(lower, forward, GRIPPER_CLOSE, label="lower", max_speed_mps=0.05),
            Waypoint(lower, forward, GRIPPER_OPEN, hold_steps=8, label="release"),
            Waypoint(withdraw, forward, label="withdraw", max_speed_mps=0.12),
            Waypoint(withdraw, forward, label="done", max_steps=10_000),
        ]
