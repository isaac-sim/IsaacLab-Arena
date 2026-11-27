# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Carry the box past the chair and the counter corner.

A G1 humanoid stands at a table with a box (the load) on it. Behind it, a route of about 3 m leads to a destination
table; a chair stands on the straight line and a counter corner juts into it. The robot sees the route, squeezes the
box between its hands, lifts it, walks back, turns and carries it to the destination table, where it sets the box
down and steps back. While the robot passes an obstacle, the obstacle is out of the head camera's view.

Hidden: each obstacle while the robot carries the box past it (level with it or just past it): the obstacle is then
out of the head camera's view. Measured: the squeezed box is too narrow to hide an obstacle the robot walks
around while it is still ahead, so the claim covers the pass, where knees, arms and box could clip it.
Randomized: chair position and side, counter corner position, side and how far it juts in, destination distance.
Difficulty knob: height of the box in meters (taller blocks more of the view).
Success: the box rests upright on the destination table, the hands are open and the robot stepped back, no collision
by body or box.
"""

from __future__ import annotations

import math
import torch

from isaaclab_arena_environments.spatial_memory.franka_controller import HOLD_FOREVER_STEPS
from isaaclab_arena_environments.spatial_memory.g1_adapter import (
    ARMS_CARRY,
    ARMS_PLACE,
    ARMS_SQUEEZE,
    ARMS_WIDE,
    G1_FLOOR_Z_M,
    G1Waypoint,
)
from isaaclab_arena_environments.spatial_memory.geometry import ALL_FACES
from isaaclab_arena_environments.spatial_memory.layout_sampling import pose, uniform
from isaaclab_arena_environments.spatial_memory.parts import PartSpec, board, box_item
from isaaclab_arena_environments.spatial_memory.predicates import part_upright
from isaaclab_arena_environments.spatial_memory.task_logic import LayoutSample, PlanMode, SpatialMemoryTaskLogic
from isaaclab_arena_environments.spatial_memory.visibility import VisibilityTarget

LOAD_DEPTH_M = 0.20
"""Extent of the box along the robot's facing direction at the pick table."""
LOAD_WIDTH_M = 0.26
"""Width of the box, across which the hands squeeze it."""
LOAD_CENTER_Z_M = 0.10
"""Height of the box's center above the pelvis when it rests on a table, level with the squeezing hands."""
LOAD_AHEAD_M = 0.39
"""Distance of the box's center in front of the pelvis on the pick table."""
LOAD_LIFTED_M = 0.01
"""Height of the box above its resting height on the pick table above which it counts as lifted."""
TABLE_SIZE_XY_M = (0.40, 0.70)
"""Footprint (depth, width) of the pick and the destination table."""
TABLE_AHEAD_M = 0.51
"""Distance of a table's center in front of the pelvis when the robot stands at it."""
CHAIR_SIZE_M = (0.45, 0.45, 0.45)
"""Size of the chair standing on the route."""
COUNTER_SIZE_M = (0.60, 0.60, 0.90)
"""Size of the counter whose corner juts into the route."""
ROUTE_X_M = -0.40
"""Environment-frame x of the route line, which the robot walks back to before turning to face +y."""
CLEARANCE_M = 0.45
"""Lateral distance between the pelvis and an obstacle's edge on the oracle route."""
PASSING_START_M = 0.1
"""Distance of an obstacle's center behind the pelvis from which the robot is passing it."""
PASSING_M = 0.6
"""Distance of an obstacle's far edge behind the pelvis until which the robot is passing it."""
PASSING_LATERAL_M = 1.3
"""Maximum lateral distance between the pelvis and an obstacle's center while the robot is passing it."""
PLACED_EDGE_MARGIN_X_M = 0.05
"""Minimum x distance between the placed box's center and the destination table's edge."""
PLACED_EDGE_MARGIN_Y_M = 0.03
"""Minimum y distance between the placed box's center and the destination table's edge."""
RESTING_TOLERANCE_M = 0.03
"""Maximum vertical distance between the box's bottom and the destination table top for the box to rest on it."""
STEP_BACK_M = 0.45
"""Distance the robot steps back from the destination table after setting the box down."""
FALLEN_PELVIS_Z_M = -0.35
"""The robot has fallen once its pelvis is this far below its standing height."""
DROP_DEPTH_M = 0.3
"""The box was dropped once it is this far below the table tops."""


class BoxCarryLogic(SpatialMemoryTaskLogic):
    """Logic of the box-carry task (see the module docstring)."""

    name = "spatial_memory_box_carry"
    robot = "g1"
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
            board("pick_table", table_size, color="wood"),
            board("dest_table", table_size, color="wood"),
            board("chair", CHAIR_SIZE_M, color="blue"),
            board("counter", COUNTER_SIZE_M, color="gray"),
            box_item("load", (LOAD_DEPTH_M, LOAD_WIDTH_M, self.load_height), color="orange", mass_kg=0.25),
        ]

    def sample_layout(self, generator: torch.Generator) -> LayoutSample:
        poses = self.parked_part_poses()
        table_z = G1_FLOOR_Z_M + 0.5 * self.table_height()
        poses["pick_table"] = pose(TABLE_AHEAD_M, 0.0, table_z)
        poses["load"] = pose(LOAD_AHEAD_M, 0.0, LOAD_CENTER_Z_M + 0.002)
        chair_y = uniform(generator, 0.95, 1.1)
        counter_y = chair_y + uniform(generator, 1.5, 1.6)
        dest_y = counter_y + uniform(generator, 0.9, 1.0)
        poses["dest_table"] = pose(ROUTE_X_M, dest_y + TABLE_AHEAD_M, table_z, 0.5 * math.pi)
        chair_side = 1.0 if float(torch.rand(1, generator=generator)) < 0.5 else -1.0
        chair_x = ROUTE_X_M + chair_side * uniform(generator, 0.05, 0.15)
        poses["chair"] = pose(chair_x, chair_y, G1_FLOOR_Z_M + 0.5 * CHAIR_SIZE_M[2])
        counter_side = 1.0 if float(torch.rand(1, generator=generator)) < 0.5 else -1.0
        jut = uniform(generator, 0.10, 0.20)
        counter_x = ROUTE_X_M + counter_side * (0.5 * COUNTER_SIZE_M[0] - jut)
        poses["counter"] = pose(counter_x, counter_y, G1_FLOOR_Z_M + 0.5 * COUNTER_SIZE_M[2])
        params = {
            "dest_y": torch.tensor(dest_y),
            "chair_xy": torch.tensor([chair_x, chair_y]),
            "chair_side": torch.tensor(chair_side),
            "counter_xy": torch.tensor([counter_x, counter_y]),
            "counter_side": torch.tensor(counter_side),
            "dest_table_xy": torch.tensor([ROUTE_X_M, dest_y + TABLE_AHEAD_M]),
        }
        return LayoutSample(part_poses_e=self.part_poses_tensor(poses), layout_params=params)

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

    def hand_touchable_part_names(self) -> list[str]:
        return ["pick_table", "dest_table"]

    def _passing(self, env, episode) -> torch.Tensor:
        """Return (N, 2) whether the robot is passing each obstacle along the route (+y).

        The obstacle's center is more than ``PASSING_START_M`` behind the pelvis, its far edge less than ``PASSING_M``
        behind, and it is within ``PASSING_LATERAL_M`` to the side.
        """
        t_E_pelvis_xy = self.embodiment.tracked_pose_w(env)[:, :2] - env.scene.env_origins[:, :2]
        passing = []
        for name, size in (("chair", CHAIR_SIZE_M), ("counter", COUNTER_SIZE_M)):
            t_pelvis_obstacle_xy = env.arena_world.get_pose_e(name)[:, :2] - t_E_pelvis_xy
            ahead = t_pelvis_obstacle_xy[:, 1]
            behind_edge = -ahead - 0.5 * size[1]
            lateral = t_pelvis_obstacle_xy[:, 0].abs()
            passing.append((ahead < -PASSING_START_M) & (behind_edge < PASSING_M) & (lateral < PASSING_LATERAL_M))
        return torch.stack(passing, dim=-1)

    def _carrying(self, env) -> torch.Tensor:
        return self.gripper_closed(env) & (env.arena_world.get_pose_e("load")[:, 2] > LOAD_CENTER_Z_M + LOAD_LIFTED_M)

    def hidden_phase(self, env, episode) -> torch.Tensor:
        return self._carrying(env) & self._passing(env, episode).any(dim=-1)

    def hidden_target_mask(self, env, episode, in_hidden_phase: torch.Tensor) -> torch.Tensor:
        return in_hidden_phase.unsqueeze(-1) & self._passing(env, episode)

    def success_terms(self, env, episode) -> dict[str, torch.Tensor]:
        params = episode.layout_params
        t_E_load = env.arena_world.get_pose_e("load")[:, :3]
        t_E_table_xy = params["dest_table_xy"]
        # The destination table is turned by 90 degrees: its x extent runs along y.
        inside = ((t_E_load[:, 0] - t_E_table_xy[:, 0]).abs() < 0.5 * TABLE_SIZE_XY_M[1] - PLACED_EDGE_MARGIN_X_M) & (
            (t_E_load[:, 1] - t_E_table_xy[:, 1]).abs() < 0.5 * TABLE_SIZE_XY_M[0] - PLACED_EDGE_MARGIN_Y_M
        )
        table_top_z = G1_FLOOR_Z_M + self.table_height()
        resting = (t_E_load[:, 2] - 0.5 * self.load_height - table_top_z).abs() < RESTING_TOLERANCE_M
        t_E_pelvis_y = self.embodiment.tracked_pose_w(env)[:, 1] - env.scene.env_origins[:, 1]
        stepped_back = t_E_pelvis_y < params["dest_y"] - 0.5 * STEP_BACK_M
        return {
            "load_on_table": inside & resting,
            "load_upright": part_upright(env, "load"),
            "released_and_stepped_back": ~self.gripper_closed(env) & stepped_back,
        }

    def extra_failures(self, env, episode) -> torch.Tensor:
        fallen = self.embodiment.tracked_pose_w(env)[:, 2] - env.scene.env_origins[:, 2] < FALLEN_PELVIS_Z_M
        dropped = env.arena_world.get_pose_e("load")[:, 2] < G1_FLOOR_Z_M + self.table_height() - DROP_DEPTH_M
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

    def plan(self, env, episode, env_id: int, mode: PlanMode) -> list[G1Waypoint]:
        params = {key: value[env_id] for key, value in episode.layout_params.items()}
        dest_y = float(params["dest_y"])
        facing_y = 0.5 * math.pi
        plan = [
            G1Waypoint((0.0, 0.0, 0.0), ARMS_WIDE, label="arms_wide", tolerance_m=0.1, hold_steps=10),
            G1Waypoint((0.0, 0.0, 0.0), ARMS_SQUEEZE, label="squeeze", tolerance_m=0.1, hold_steps=15),
            G1Waypoint((0.0, 0.0, 0.0), ARMS_CARRY, label="lift", tolerance_m=0.1, hold_steps=10),
            G1Waypoint((ROUTE_X_M, 0.0, 0.0), ARMS_CARRY, label="back", tolerance_m=0.08, arm_time_s=0.1),
            G1Waypoint((ROUTE_X_M, 0.0, facing_y), ARMS_CARRY, label="turn", tolerance_m=0.1, arm_time_s=0.1),
        ]
        # The oracle walks around the obstacles; the blind controller walks the straight line through them.
        route = self._route(params, avoid=mode == PlanMode.ORACLE)
        # The robot keeps facing +y and walks the route forwards and sideways, so the box never swings.
        for point in route[1:-1]:
            plan.append(
                G1Waypoint(
                    (point[0], point[1], facing_y),
                    ARMS_CARRY,
                    label="route",
                    tolerance_m=0.15,
                    tolerance_rad=0.4,
                    arm_time_s=0.1,
                )
            )
        plan += [
            G1Waypoint(
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
            G1Waypoint((ROUTE_X_M, dest_y, facing_y), ARMS_PLACE, label="lower", tolerance_m=0.1, hold_steps=20),
            G1Waypoint(
                (ROUTE_X_M, dest_y, facing_y),
                ARMS_WIDE,
                label="release",
                tolerance_m=0.1,
                arm_time_s=2.5,
                hold_steps=20,
            ),
            G1Waypoint((ROUTE_X_M, dest_y - STEP_BACK_M, facing_y), ARMS_WIDE, label="step_back", tolerance_m=0.08),
            G1Waypoint(
                (ROUTE_X_M, dest_y - STEP_BACK_M, facing_y), ARMS_WIDE, label="done", max_steps=HOLD_FOREVER_STEPS
            ),
        ]
        return plan
