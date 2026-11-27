# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The cart someone moved.

The scene of the mug-past-the-door task (a mug on the counter, a wall cabinet with an open door) plus one movable element: a tall cart
(a post standing on the counter) that is either in the lane between the mug and the cabinet opening or set aside. The
robot sees the scene during an initial look. While it grasps the mug, the cart may be moved to the other spot. The
wrist image is withheld after the look, except for a short re-observation window that opens when the cart is first
back in view after the move. The robot must carry the mug over the cart if it is in the lane, or straight to the
cabinet if it is not, and put the mug on the lower shelf.

Hidden: the cart's new pose, apart from the re-observation window (the frames are withheld).
Randomized: the scene of the mug-past-the-door task (lower shelf only), the cart's old spot, whether it moves (probability 0.8).
Difficulty knob: length of the re-observation window in seconds (shorter is harder).
Success: as in the mug-past-the-door task, with no collision at the new pose and no detour around the old one: the tool path from the grasp
until the end is at most 10% (plus 4 cm) longer than the oracle route for the new pose.
"""

from __future__ import annotations

import math
import torch

from isaaclab_arena_environments.spatial_memory.episode_state import NOT_SET
from isaaclab_arena_environments.spatial_memory.franka_controller import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    HOLD_FOREVER_STEPS,
    Waypoint,
    pitched_quat,
)
from isaaclab_arena_environments.spatial_memory.geometry import ALL_FACES
from isaaclab_arena_environments.spatial_memory.layout_sampling import pose, uniform
from isaaclab_arena_environments.spatial_memory.parts import PartSpec, board
from isaaclab_arena_environments.spatial_memory.task_logic import LayoutSample, PlanMode
from isaaclab_arena_environments.spatial_memory.tasks.mug_past_door import (
    APPROACH_PITCH_RAD,
    GRASP_HEIGHT_M,
    INSERT_DEPTH_M,
    MUG_HEIGHT_M,
    TRANSIT_HEIGHT_M,
    MugPastDoorLogic,
)
from isaaclab_arena_environments.spatial_memory.visibility import VisibilityTarget

CART_SIZE_M = (0.06, 0.06, 0.32)
"""Size of the cart, a tall post higher than the straight carry route."""
CART_ASIDE_XY = (0.60, 0.50)
"""Environment-frame spot on the counter, away from every route, where the cart stands when not in the lane."""
CART_LANE_FRACTION = 0.55
"""Position of the lane spot along the straight carry route, from the opening (0) to the grasp (1)."""
MOVE_PROBABILITY = 0.8
"""Probability that the cart is moved to its other spot when the robot grasps the mug."""
LOOK_S = 2.0
"""Seconds of camera frames at the start of the episode."""
OVER_CLEARANCE_M = 0.08
"""Clearance of the mug's base above the cart when the route goes over it."""
PATH_TOLERANCE = 0.10
"""Fraction by which the carry path may exceed the oracle route for the cart's new pose."""
PATH_MARGIN_M = 0.04
"""Length by which the carry path may exceed the oracle route on top of ``PATH_TOLERANCE``."""
PATH_STEP_M = 0.01
"""Minimum step length in which the carry path is measured, so that jitter while holding still does not add up."""


class MovedCartLogic(MugPastDoorLogic):
    """Logic of the moved-cart task (see the module docstring)."""

    name = "spatial_memory_moved_cart"
    difficulty_name = "reobservation_window_s"
    default_difficulty = 1.0
    episode_length_s = 40.0
    task_description = "Pick up the mug and put it on the lower shelf inside the open cabinet."
    withholds_camera_frames = True

    def door_angle_deg(self) -> float:
        """Return the nominal door opening angle in degrees: perpendicular to the cabinet front."""
        return 90.0

    def build_part_specs(self) -> list[PartSpec]:
        return super().build_part_specs() + [board("cart", CART_SIZE_M, color="blue")]

    def sample_layout(self, generator: torch.Generator) -> LayoutSample:
        sample = super().sample_layout(generator)
        poses = sample.part_poses_e
        params = sample.layout_params
        names = [spec.name for spec in self.part_specs()]
        # Lower shelf only, and the mug further left than in the mug-past-the-door task so that the cart fits between mug and cabinet.
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
        return LayoutSample(part_poses_e=poses, layout_params=params)

    def _lower_support_z(self, poses: torch.Tensor, names: list[str]) -> float:
        """Return the top of the cabinet's bottom board."""
        from isaaclab_arena_environments.spatial_memory.layout_sampling import BOARD_THICKNESS_M

        return float(poses[names.index("cab_bottom")][2]) + 0.5 * BOARD_THICKNESS_M

    # ----------------------------------------------------------------------------------------------------------------

    def visibility_targets(self) -> list[VisibilityTarget]:
        return [VisibilityTarget("cart", "cart", ALL_FACES, spacing_m=0.03)]

    def hidden_target_names(self) -> list[str]:
        return ["cart"]

    def remembered_target_names(self) -> list[str]:
        return ["cart"]

    def on_reset(self, env, episode, env_ids: torch.Tensor) -> None:
        episode.task_state_buffer("moved", False, torch.bool)
        episode.task_state_buffer("window_start_step", NOT_SET, torch.long)
        episode.task_state_buffer("carry_path_m", 0.0, torch.float32)
        episode.task_state_buffer("path_anchor_w", 0.0, torch.float32, (3,))

    def update_frame_withholding(self, env, episode, visible: torch.Tensor, active: torch.Tensor) -> None:
        # The re-observation window opens the first time the cart is in view after the move.
        state = episode.task_state
        cart_visible = visible[:, episode.target_names.index("cart")]
        opening = state["moved"] & (state["window_start_step"] == NOT_SET) & cart_visible & active
        state["window_start_step"][opening] = env.episode_length_buf[opening]

    def _window_open(self, env, episode) -> torch.Tensor:
        """Return whether the re-observation window is open (N,)."""
        start = episode.task_state["window_start_step"]
        window_steps = int(round(self.difficulty / episode.step_dt))
        elapsed = env.episode_length_buf - start
        return (start != NOT_SET) & (elapsed < window_steps)

    def camera_blocked(self, env, episode) -> torch.Tensor:
        if "moved" not in episode.task_state:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        look = env.episode_length_buf < int(round(LOOK_S / episode.step_dt))
        return ~look & ~self._window_open(env, episode)

    def on_step(self, env, episode, visible: torch.Tensor, active: torch.Tensor) -> None:
        # The cart is moved (or left where it is) when the robot first grasps, i.e. while it looks at the mug.
        state = episode.task_state
        t_W_tcp = self.tool_pose_w(env)[:, :3]
        step = (t_W_tcp - state["path_anchor_w"]).norm(dim=-1)
        advance = state["moved"] & active & (step >= PATH_STEP_M)
        state["carry_path_m"] += torch.where(advance, step, torch.zeros_like(step))
        state["path_anchor_w"][advance] = t_W_tcp[advance]
        grasped = (episode.first_close_step != NOT_SET) & ~state["moved"] & active
        if grasped.any():
            env_ids = grasped.nonzero().squeeze(-1)
            cart = env.scene["cart"]
            t_E_cart_new_xy = episode.layout_params["cart_new_xy"][env_ids]
            T_W_cart = cart.data.root_pose_w.torch[env_ids].clone()
            T_W_cart[:, :2] = t_E_cart_new_xy + env.scene.env_origins[env_ids, :2]
            cart.write_root_pose_to_sim(T_W_cart, env_ids=env_ids)
            cart.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)
            state["moved"] |= grasped
            state["path_anchor_w"][env_ids] = t_W_tcp[env_ids]

    def hidden_phase(self, env, episode) -> torch.Tensor:
        return episode.task_state["moved"]

    def hidden_target_mask(self, env, episode, in_hidden_phase: torch.Tensor) -> torch.Tensor:
        hidden = in_hidden_phase & ~self._window_open(env, episode)
        return hidden.unsqueeze(-1).expand(-1, len(episode.target_names))

    def static_part_mask(self, env, episode) -> torch.Tensor:
        mask = super().static_part_mask(env, episode).clone()
        mask[:, episode.part_index["cart"]] = False
        return mask

    def success_terms(self, env, episode) -> dict[str, torch.Tensor]:
        terms = super().success_terms(env, episode)
        carry_path = episode.task_state["carry_path_m"]
        limit = (1.0 + PATH_TOLERANCE) * episode.layout_params["reference_path_m"] + PATH_MARGIN_M
        terms["no_detour"] = episode.task_state["moved"] & (carry_path <= limit)
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
        """Return the length of the carry route."""
        route = self._carry_route(params, over_cart)
        return sum(math.dist(a, b) for a, b in zip(route[:-1], route[1:]))

    def plan(self, env, episode, env_id: int, mode: str) -> list[Waypoint]:
        params = {key: value[env_id] for key, value in episode.layout_params.items()}
        # The oracle knows where the cart will be; the blind controller plans for the pose it saw at the start.
        key = "cart_new_in_lane" if mode == PlanMode.ORACLE else "cart_old_in_lane"
        route = self._carry_route(params, bool(params[key]))
        q_E_tcp = pitched_quat(0.0, APPROACH_PITCH_RAD)
        grasp, lift, transit, preinsert, insert, lower, withdraw = route
        plan = super().plan(env, episode, env_id, mode)
        labels = [waypoint.label for waypoint in plan]
        close_index = labels.index("close")
        return plan[: close_index + 1] + [
            Waypoint(lift, q_E_tcp, GRIPPER_CLOSE, label="lift", max_speed_mps=0.1),
            Waypoint(transit, q_E_tcp, GRIPPER_CLOSE, label="transit", max_speed_mps=0.2),
            Waypoint(preinsert, q_E_tcp, GRIPPER_CLOSE, label="preinsert", max_speed_mps=0.15),
            Waypoint(insert, q_E_tcp, GRIPPER_CLOSE, label="insert", max_speed_mps=0.1),
            Waypoint(lower, q_E_tcp, GRIPPER_CLOSE, label="lower", max_speed_mps=0.05),
            Waypoint(lower, q_E_tcp, GRIPPER_OPEN, hold_steps=8, label="release"),
            Waypoint(withdraw, q_E_tcp, label="withdraw", max_speed_mps=0.12),
            Waypoint(withdraw, q_E_tcp, label="done", max_steps=HOLD_FOREVER_STEPS),
        ]
