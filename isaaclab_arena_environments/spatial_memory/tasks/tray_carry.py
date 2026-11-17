# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task 3: carry the box past the chair and the counter corner.

A G1 humanoid stands at a table with a box (the load) on it. Behind it, a route of about 3 m leads to a destination
table; a chair stands on the straight line and a counter corner juts into it. The robot sees the route, squeezes the
box between its hands, lifts it, walks back, turns and carries it to the destination table, where it sets the box
down and steps back. While the robot passes an obstacle, the obstacle is out of the head camera's view.

Hidden: each obstacle while the robot carries the box past it (level with it or just past it): the obstacle is then
out of the head camera's view. Measured: the squeezed box is too narrow to hide an obstacle the robot walks
around while it is still ahead, so the claim covers the pass, where knees, arms and box could clip it.
Randomised: chair position and side, counter corner position, side and how far it juts in, destination distance.
Difficulty knob: height of the box in metres (taller blocks more of the view).
Success: the box rests upright on the destination table, the hands are open and the robot stepped back, no collision
by body or box.
"""

from __future__ import annotations

import math
import torch

from isaaclab_arena_environments.spatial_memory.g1_adapter import (
    ARMS_CARRY,
    ARMS_PLACE,
    ARMS_SQUEEZE,
    ARMS_WIDE,
    G1_FLOOR_Z_M,
    BaseWaypoint,
)
from isaaclab_arena_environments.spatial_memory.geometry import ALL_FACES
from isaaclab_arena_environments.spatial_memory.layout_utils import pose, uniform
from isaaclab_arena_environments.spatial_memory.parts import PartSpec, board, box_item
from isaaclab_arena_environments.spatial_memory.predicates import item_pose_e, item_upright
from isaaclab_arena_environments.spatial_memory.task_logic import LayoutSample, PlanMode, SpatialMemoryTaskLogic
from isaaclab_arena_environments.spatial_memory.visibility import VisibilityTarget

LOAD_DEPTH_M = 0.20
LOAD_WIDTH_M = 0.26
"""The box is squeezed across its width."""
LOAD_CENTER_Z_M = 0.10
"""Height of the box's centre above the pelvis when it rests on a table, level with the squeezing hands."""
LOAD_AHEAD_M = 0.39
"""Distance of the box's centre in front of the pelvis on the pick table."""
TABLE_SIZE_XY_M = (0.40, 0.70)
TABLE_AHEAD_M = 0.51
"""Distance of a table's centre in front of the pelvis when the robot stands at it."""
CHAIR_SIZE_M = (0.45, 0.45, 0.45)
COUNTER_SIZE_M = (0.60, 0.60, 0.90)
ROUTE_X_M = -0.40
"""The robot walks back to this x, turns to face +y and follows the route along this line."""
CLEARANCE_M = 0.45
"""Lateral distance the oracle keeps between the pelvis and an obstacle's edge."""
PASSING_M = 0.6
PASSING_LATERAL_M = 1.3
"""The robot is passing an obstacle from when its centre is level with the pelvis until its far edge is ``PASSING_M``
behind, within ``PASSING_LATERAL_M`` to the side."""
STEP_BACK_M = 0.45


class TrayCarryLogic(SpatialMemoryTaskLogic):
    """Logic of task 3 (see module docstring)."""

    name = "spatial_memory_tray_carry"
    difficulty_name = "load_height_m"
    default_difficulty = 0.25
    episode_length_s = 60.0
    task_description = "Carry the box to the other table without bumping into anything."

    @property
    def load_height(self) -> float:
        height = float(self.difficulty)
        assert 0.10 <= height <= 0.40, f"load_height_m must be between 0.10 and 0.40, got {height}"
        return height

    def table_height(self) -> float:
        """Return the height of both tables above the floor."""
        return LOAD_CENTER_Z_M - 0.5 * self.load_height - G1_FLOOR_Z_M

    def build_part_specs(self) -> list[PartSpec]:
        table_size = (*TABLE_SIZE_XY_M, self.table_height())
        return [
            board("pick_table", table_size, colour="wood"),
            board("dest_table", table_size, colour="wood"),
            board("chair", CHAIR_SIZE_M, colour="blue"),
            board("counter", COUNTER_SIZE_M, colour="grey"),
            box_item("load", (LOAD_DEPTH_M, LOAD_WIDTH_M, self.load_height), colour="orange", mass_kg=0.25),
        ]

    def sample_layout(self, generator: torch.Generator) -> LayoutSample:
        table = self.new_pose_table()
        table_z = G1_FLOOR_Z_M + 0.5 * self.table_height()
        table["pick_table"] = pose(TABLE_AHEAD_M, 0.0, table_z)
        table["load"] = pose(LOAD_AHEAD_M, 0.0, LOAD_CENTER_Z_M + 0.002)
        chair_y = uniform(generator, 0.95, 1.1)
        counter_y = chair_y + uniform(generator, 1.5, 1.6)
        dest_y = counter_y + uniform(generator, 0.9, 1.0)
        table["dest_table"] = pose(ROUTE_X_M, dest_y + TABLE_AHEAD_M, table_z, 0.5 * math.pi)
        chair_side = 1.0 if float(torch.rand(1, generator=generator)) < 0.5 else -1.0
        chair_x = ROUTE_X_M + chair_side * uniform(generator, 0.05, 0.15)
        table["chair"] = pose(chair_x, chair_y, G1_FLOOR_Z_M + 0.5 * CHAIR_SIZE_M[2])
        counter_side = 1.0 if float(torch.rand(1, generator=generator)) < 0.5 else -1.0
        jut = uniform(generator, 0.10, 0.20)
        counter_x = ROUTE_X_M + counter_side * (0.5 * COUNTER_SIZE_M[0] - jut)
        table["counter"] = pose(counter_x, counter_y, G1_FLOOR_Z_M + 0.5 * COUNTER_SIZE_M[2])
        params = {
            "dest_y": torch.tensor(dest_y),
            "chair_xy": torch.tensor([chair_x, chair_y]),
            "chair_side": torch.tensor(chair_side),
            "counter_xy": torch.tensor([counter_x, counter_y]),
            "counter_side": torch.tensor(counter_side),
            "dest_table_xy": torch.tensor([ROUTE_X_M, dest_y + TABLE_AHEAD_M]),
        }
        return LayoutSample(part_poses_e=self.pose_table_to_tensor(table), params=params)

    # ----------------------------------------------------------------------------------------------------------------

    def obstacle_names(self) -> list[str]:
        return ["chair", "counter"]

    def visibility_targets(self) -> list[VisibilityTarget]:
        return [VisibilityTarget(name, name, ALL_FACES, spacing_m=0.05) for name in self.obstacle_names()]

    def hidden_target_names(self) -> list[str]:
        return self.obstacle_names()

    def manipulated_item_names(self) -> list[str]:
        return ["load"]

    def item_support_names(self) -> list[str]:
        return ["pick_table", "dest_table"]

    def allowed_contacts(self, env, runtime) -> torch.Tensor:
        """The default contacts, and the hands may touch the tables while picking and placing."""
        from isaaclab_arena_environments.spatial_memory.collision import filtered_sensor_labels

        allowed = super().allowed_contacts(env, runtime)[0].clone()
        labels = filtered_sensor_labels(self)
        table_ids = runtime.part_ids(["pick_table", "dest_table"])
        for name in self.embodiment.gripper_bodies:
            allowed[labels.index(name), table_ids] = True
        return allowed.unsqueeze(0).expand(env.num_envs, -1, -1)

    def _passing(self, env, runtime) -> torch.Tensor:
        """Return (N, 2) whether the robot is passing each obstacle along the route (+y): the obstacle's centre is
        level with or behind the pelvis, its far edge less than ``PASSING_M`` behind, and it is within
        ``PASSING_LATERAL_M`` to the side."""
        position_xy = self.tool_pose_w(env)[:, :2] - env.scene.env_origins[:, :2]
        passing = []
        for name, size in (("chair", CHAIR_SIZE_M), ("counter", COUNTER_SIZE_M)):
            offset = item_pose_e(env, name)[:, :2] - position_xy
            ahead = offset[:, 1]
            behind_edge = -ahead - 0.5 * size[1]
            passing.append((ahead < -0.1) & (behind_edge < PASSING_M) & (offset[:, 0].abs() < PASSING_LATERAL_M))
        return torch.stack(passing, dim=-1)

    def _carrying(self, env) -> torch.Tensor:
        return self.gripper_closed(env) & (item_pose_e(env, "load")[:, 2] > LOAD_CENTER_Z_M + 0.01)

    def action_phase(self, env, runtime) -> torch.Tensor:
        return self._carrying(env) & self._passing(env, runtime).any(dim=-1)

    def hidden_phase_mask(self, env, runtime, in_action: torch.Tensor) -> torch.Tensor:
        return in_action.unsqueeze(-1) & self._passing(env, runtime)

    def success_terms(self, env, runtime) -> dict[str, torch.Tensor]:
        params = runtime.params
        load = item_pose_e(env, "load")
        table_xy = params["dest_table_xy"]
        # The destination table is turned by 90 degrees: its x extent runs along y.
        inside = ((load[:, 0] - table_xy[:, 0]).abs() < 0.5 * TABLE_SIZE_XY_M[1] - 0.05) & (
            (load[:, 1] - table_xy[:, 1]).abs() < 0.5 * TABLE_SIZE_XY_M[0] - 0.03
        )
        table_top = G1_FLOOR_Z_M + self.table_height()
        resting = (load[:, 2] - 0.5 * self.load_height - table_top).abs() < 0.03
        pelvis_y = self.tool_pose_w(env)[:, 1] - env.scene.env_origins[:, 1]
        stepped_back = pelvis_y < params["dest_y"] - 0.5 * STEP_BACK_M
        return {
            "load_on_table": inside & resting,
            "load_upright": item_upright(env, "load"),
            "released_and_stepped_back": ~self.gripper_closed(env) & stepped_back,
        }

    def extra_failures(self, env, runtime) -> torch.Tensor:
        fallen = self.tool_pose_w(env)[:, 2] - env.scene.env_origins[:, 2] < -0.35
        dropped = item_pose_e(env, "load")[:, 2] < G1_FLOOR_Z_M + self.table_height() - 0.3
        return fallen | dropped

    # ----------------------------------------------------------------------------------------------------------------

    def _route(self, params, avoid: bool) -> list[tuple[float, float]]:
        """Return the pelvis route from the turn spot to the destination spot."""
        dest_y = float(params["dest_y"])
        start = (ROUTE_X_M, 0.0)
        end = (ROUTE_X_M, dest_y)
        if not avoid:
            return [start, end]
        chair_x, chair_y = (float(v) for v in params["chair_xy"])
        chair_side = float(params["chair_side"])
        counter_x, counter_y = (float(v) for v in params["counter_xy"])
        counter_side = float(params["counter_side"])
        # Pass each obstacle on the side away from it, keeping CLEARANCE_M from its edge.
        chair_pass_x = chair_x - chair_side * (0.5 * CHAIR_SIZE_M[0] + CLEARANCE_M)
        counter_pass_x = counter_x - counter_side * (0.5 * COUNTER_SIZE_M[0] + CLEARANCE_M)
        half_chair = 0.5 * CHAIR_SIZE_M[1] + 0.25
        half_counter = 0.5 * COUNTER_SIZE_M[1] + 0.2
        # Past the chair, step sideways to the counter's passing line before the box reaches the counter.
        return [
            start,
            (chair_pass_x, chair_y - half_chair),
            (chair_pass_x, chair_y + half_chair),
            (counter_pass_x, chair_y + half_chair),
            (counter_pass_x, counter_y - half_counter),
            (counter_pass_x, counter_y + half_counter),
            (ROUTE_X_M, dest_y - 0.3),
            end,
        ]

    def plan(self, env, runtime, env_id: int, mode: str) -> list[BaseWaypoint]:
        params = {key: value[env_id] for key, value in runtime.params.items()}
        dest_y = float(params["dest_y"])
        facing_y = 0.5 * math.pi
        plan = [
            BaseWaypoint((0.0, 0.0, 0.0), ARMS_WIDE, label="arms_wide", tolerance_m=0.1, hold_steps=10),
            BaseWaypoint((0.0, 0.0, 0.0), ARMS_SQUEEZE, label="squeeze", tolerance_m=0.1, hold_steps=15),
            BaseWaypoint((0.0, 0.0, 0.0), ARMS_CARRY, label="lift", tolerance_m=0.1, hold_steps=10),
            BaseWaypoint((ROUTE_X_M, 0.0, 0.0), ARMS_CARRY, label="back", tolerance_m=0.08, arm_time_s=0.1),
            BaseWaypoint((ROUTE_X_M, 0.0, facing_y), ARMS_CARRY, label="turn", tolerance_m=0.1, arm_time_s=0.1),
        ]
        # The oracle walks around the obstacles; the blind controller walks the straight line through them.
        route = self._route(params, avoid=mode == PlanMode.ORACLE)
        # The robot keeps facing +y and walks the route forwards and sideways, so the box never swings.
        for point in route[1:-1]:
            plan.append(
                BaseWaypoint(
                    (point[0], point[1], facing_y),
                    ARMS_CARRY,
                    label="route",
                    tolerance_m=0.15,
                    tolerance_rad=0.4,
                    arm_time_s=0.1,
                )
            )
        plan += [
            BaseWaypoint(
                (ROUTE_X_M, dest_y, facing_y),
                ARMS_CARRY,
                label="arrive",
                tolerance_m=0.06,
                tolerance_rad=0.08,
                arm_time_s=0.1,
                hold_steps=20,
                max_steps=400,
                max_forward_mps=0.25,
            ),
            BaseWaypoint((ROUTE_X_M, dest_y, facing_y), ARMS_PLACE, label="lower", tolerance_m=0.1, hold_steps=20),
            BaseWaypoint(
                (ROUTE_X_M, dest_y, facing_y),
                ARMS_WIDE,
                label="release",
                tolerance_m=0.1,
                arm_time_s=2.5,
                hold_steps=20,
            ),
            BaseWaypoint((ROUTE_X_M, dest_y - STEP_BACK_M, facing_y), ARMS_WIDE, label="step_back", tolerance_m=0.08),
            BaseWaypoint((ROUTE_X_M, dest_y - STEP_BACK_M, facing_y), ARMS_WIDE, label="done", max_steps=100_000),
        ]
        return plan
