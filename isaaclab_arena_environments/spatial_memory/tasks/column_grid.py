# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Carry through a surveyed column grid to a dock specified after pickup."""

from __future__ import annotations

import itertools
import math
import torch

from isaaclab_arena_environments.spatial_memory.g1_adapter import ARMS_WIDE, G1_FLOOR_Z_M, G1Waypoint
from isaaclab_arena_environments.spatial_memory.layout_sampling import pose, uniform
from isaaclab_arena_environments.spatial_memory.parts import board
from isaaclab_arena_environments.spatial_memory.predicates import part_upright
from isaaclab_arena_environments.spatial_memory.task_logic import LayoutSample, PlanMode
from isaaclab_arena_environments.spatial_memory.tasks.box_carry import (
    LOAD_AHEAD_M,
    LOAD_CENTER_Z_M,
    TABLE_AHEAD_M,
    TABLE_SIZE_XY_M,
    BoxCarryLogic,
)
from isaaclab_arena_environments.spatial_memory.tasks.loading_lane import LoadingLaneLogic

ROW_X_E = (-1.52, -3.60, -5.68)
"""Longitudinal positions of the column rows."""

DOCK_X_E = -7.36
"""Longitudinal position of the pelvis during unloading, facing -x."""

LANE_Y_E = (-1.80, -0.60, 0.60, 1.80)
"""Lateral positions of the four receiving docks."""


CHECKPOINT_X_E = (-2.56, -4.64)
"""Longitudinal positions of the inspection gantries between the column rows."""

CROSSING_X_E = (ROW_X_E[0], CHECKPOINT_X_E[0], ROW_X_E[1], CHECKPOINT_X_E[1], ROW_X_E[2])
"""Crossing planes in outbound travel order."""


DISTANCE_MARGIN_M = 0.35
"""Allowed excess polyline distance over the best sequence of nominal openings."""

OPENING_TOLERANCE_M = 0.39
"""Maximum pelvis offset from a free lane centre at a row crossing."""


def crossing_distance_m(openings_y_E, request_y_E):
    """Measure the polyline through the openings, inspection points and receiving dock."""
    a, b, c = openings_y_E
    first, second, dock = request_y_E
    xs = (-0.4, *CROSSING_X_E, DOCK_X_E)
    ys = (0.0, a, first, b, second, c, dock)
    return sum(math.hypot(x1 - x0, y1 - y0) for x0, x1, y0, y1 in zip(xs[:-1], xs[1:], ys[:-1], ys[1:], strict=True))


def shortest_crossings(open_y_E, request_y_E):
    """Choose one opening in each row for the requested inspection points and dock."""
    return min(itertools.product(*open_y_E), key=lambda route: crossing_distance_m(route, request_y_E))


class ColumnGridLogic(LoadingLaneLogic):
    """Inspect three rows of columns and retain their openings for a delayed dispatch request."""

    name = "spatial_memory_column_grid"
    task_description = (
        "Inspect the column rows, pick up the carton and pass the requested inspection points and deliver to the"
        " requested dock."
    )
    minimum_env_spacing_m = 14.0
    episode_length_s = 255.0

    def obstacle_names(self):
        names = []
        for row in range(3):
            names.extend([f"column_{row}_left", f"column_{row}_right"])
        return names

    def item_support_names(self):
        return ["pick_table", *(f"dock_{i}" for i in range(4))]

    def hand_touchable_part_names(self):
        return self.item_support_names()

    def build_part_specs(self):
        parts = self.pickup_parts()
        for name in self.obstacle_names():
            parts.append(board(name, (0.30, 1.12, 0.78), "gray"))
        for i in range(4):
            parts.append(board(f"dock_{i}", (*TABLE_SIZE_XY_M, self.table_height())))
        for row in range(2):
            parts.append(board(f"gantry_{row}", (0.08, 5.0, 0.12), "gray"))
            for side in ("left", "right"):
                parts.append(board(f"gantry_{row}_{side}", (0.10, 0.10, 2.0), "gray"))
        return parts

    def sample_layout(self, generator):
        table_z_E = G1_FLOOR_Z_M + self.table_height() / 2
        poses = {
            "pick_table": pose(TABLE_AHEAD_M, 0, table_z_E),
            "load": pose(LOAD_AHEAD_M, 0, LOAD_CENTER_Z_M + 0.002),
        }
        for i, y_E in enumerate(LANE_Y_E):
            poses[f"dock_{i}"] = pose(DOCK_X_E - TABLE_AHEAD_M, y_E, table_z_E, math.pi)
        for row, x_E in enumerate(CHECKPOINT_X_E):
            poses[f"gantry_{row}"] = pose(x_E, 0, G1_FLOOR_Z_M + 2.06)
            for side, y_E in (("left", -2.45), ("right", 2.45)):
                poses[f"gantry_{row}_{side}"] = pose(x_E, y_E, G1_FLOOR_Z_M + 1.0)
        openings = []
        for row, x_E in enumerate(ROW_X_E):
            offset = uniform(generator, -0.13, 0.13)
            lanes = torch.randperm(4, generator=generator).tolist()
            for name, lane in zip(("left", "right"), sorted(lanes[:2]), strict=True):
                poses[f"column_{row}_{name}"] = pose(x_E, LANE_Y_E[lane] + offset, G1_FLOOR_Z_M + 0.39)
            openings.append([LANE_Y_E[i] + offset for i in sorted(lanes[2:])])
        request = torch.randint(4, (3,), generator=generator)
        request_y_E = [LANE_Y_E[i] for i in request.tolist()]
        best = shortest_crossings(openings, request_y_E)
        return LayoutSample(
            self.part_poses_tensor(poses),
            {
                "open_y_E": torch.tensor(openings),
                "request": request,
                "request_y_E": torch.tensor(request_y_E),
                "distance_budget_m": torch.tensor(crossing_distance_m(best, request_y_E) + DISTANCE_MARGIN_M),
                "t_E_start_xy": torch.tensor([-4.13, -2.82]) + 0.04 * torch.rand(2, generator=generator),
            },
        )

    def instruction_dim(self):
        return 12

    def instruction(self, env, episode):
        request = torch.nn.functional.one_hot(episode.layout_params["request"], 4).flatten(1)
        return request.float() * episode.instruction_revealed[:, None]

    def commitment_made(self, env, episode, active):
        """Keep monitoring future rows until the loaded robot crosses the final row."""
        return episode.task_state_buffer("next_row", 0, torch.long) == len(CROSSING_X_E)

    def hidden_target_mask(self, env, episode, in_hidden_phase):
        next_crossing = episode.task_state_buffer("next_row", 0, torch.long)
        # The two columns in a row stop carrying hidden information once the loaded robot passes them.
        row_crossings = next_crossing.new_tensor((0, 0, 2, 2, 4, 4))
        return in_hidden_phase[:, None] & (next_crossing[:, None] <= row_crossings)

    def on_step(self, env, episode, visible, active):
        t_E = self._pelvis_xy_e(env)
        ready = (t_E.norm(dim=1) < 0.12) & episode.visible_before_hidden_phase.all(dim=1) & episode.gripper_closed
        still_hidden = self.hidden_target_mask(env, episode, episode.instruction_revealed)
        relevant = visible & (~episode.instruction_revealed[:, None] | still_hidden)
        self.update_memory_probe(env, episode, relevant, active, reveal_allowed=ready)
        carrying = active & episode.instruction_revealed & self._carrying(env)
        next_row = episode.task_state_buffer("next_row", 0, torch.long)
        crossings = episode.task_state_buffer("crossing_y_E", 0.0, torch.float32, (5,))
        for i, x_E in enumerate(CROSSING_X_E):
            crossing = carrying & (next_row == i) & (t_E[:, 0] < x_E)
            crossings[crossing, i] = t_E[crossing, 1]
            next_row[crossing] += 1

    def extra_failures(self, env, episode):
        # The inbound pedestrian gallery is open to an unloaded robot only.
        t_E = self._pelvis_xy_e(env)
        in_grid = (t_E[:, 0] < -0.86) & (t_E[:, 0] > ROW_X_E[-1] - 0.5)
        escaped = episode.instruction_revealed & in_grid & (t_E[:, 1].abs() > 2.48)
        return BoxCarryLogic.extra_failures(self, env, episode) | escaped

    def success_terms(self, env, episode):
        params = episode.layout_params
        crossing_y = episode.task_state_buffer("crossing_y_E", 0.0, torch.float32, (5,))
        within_opening = (crossing_y[:, ::2, None] - params["open_y_E"]).abs().amin(dim=2) < OPENING_TOLERANCE_M
        ys = torch.cat((torch.zeros_like(crossing_y[:, :1]), crossing_y, params["request_y_E"][:, 2:]), dim=1)
        dx = ys.new_tensor((-0.4, *CROSSING_X_E, DOCK_X_E)).diff()
        distance = torch.hypot(ys.diff(dim=1), dx).sum(dim=1)
        load = env.arena_world.get_pose_e("load")
        on_dock = ((load[:, 0] - (DOCK_X_E - TABLE_AHEAD_M)).abs() < 0.18) & (
            (load[:, 1] - params["request_y_E"][:, 2]).abs() < 0.24
        )
        return {
            "crossed_grid": (episode.task_state_buffer("next_row", 0, torch.long) == 5) & within_opening.all(dim=1),
            "inspection_points": ((crossing_y[:, 1::2] - params["request_y_E"][:, :2]).abs() < 0.30).all(dim=1),
            "short_route": distance <= params["distance_budget_m"],
            "on_requested_dock": on_dock & ((load[:, 2] - LOAD_CENTER_Z_M).abs() < 0.03),
            "upright": part_upright(env, "load"),
            "released": ~self.gripper_closed(env) & (self._pelvis_xy_e(env)[:, 0] > DOCK_X_E + 0.22),
        }

    def plan(self, env, episode, env_id, mode):
        params = episode.layout_params
        request_y_E = params["request_y_E"][env_id].tolist()
        first, second, dock_y_E = request_y_E
        # Even without the layout, the controller can interpolate between requested stations.
        route = (first * 0.5, (first + second) * 0.5, (second + dock_y_E) * 0.5)
        if mode == PlanMode.ORACLE:
            route = shortest_crossings(params["open_y_E"][env_id].tolist(), request_y_E)
        return [*self.survey_and_pickup(), *self.deliver_through_openings(route, request_y_E)]

    def survey_and_pickup(self):
        """Inspect each column row and collect the carton before beginning the requested route."""
        plan = []
        for x_E in reversed(ROW_X_E):
            for yaw in (2.54, 1.90):
                plan.append(
                    G1Waypoint(
                        (x_E + 1.55, -2.80, yaw),
                        ARMS_WIDE,
                        tolerance_m=0.14,
                        tolerance_rad=0.15,
                        max_steps=850,
                        hold_steps=15,
                    )
                )
        plan.extend(self.inspection_and_pickup())
        return plan

    def deliver_through_openings(self, route, request_y_E):
        """Carry through the selected openings and requested inspection points to the dock."""
        dock_y_E = request_y_E[2]
        plan = []
        for row, (x_E, y_E) in enumerate(zip(ROW_X_E, route, strict=True)):
            plan.extend((self.carry_to(x_E + 0.35, y_E, 0), self.carry_to(x_E - 0.80, y_E, 0)))
            if row < len(CHECKPOINT_X_E):
                checkpoint = CHECKPOINT_X_E[row]
                plan.extend((
                    self.carry_to(checkpoint + 0.18, request_y_E[row], 0),
                    self.carry_to(checkpoint - 0.18, request_y_E[row], 0),
                ))
        plan.extend((self.carry_to(DOCK_X_E + 0.45, dock_y_E, 0), self.carry_to(DOCK_X_E + 0.45, dock_y_E, math.pi)))
        plan.extend(self.unload_at(DOCK_X_E, dock_y_E, math.pi, (DOCK_X_E + 0.46, dock_y_E)))
        return plan
