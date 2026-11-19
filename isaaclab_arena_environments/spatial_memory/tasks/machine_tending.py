# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Machine tending through an opening.

A box enclosure (a stand-in for a machine tool) sits on a plinth, with a framed opening in its front wall. Inside, a
cradle fixture holds the part between a front and a back block, and two clamp blocks stand just behind the front wall,
on either side of the corridor the hand must use. The Franka picks the part from the table outside, carries it through
the opening, seats it in the cradle, and withdraws. While the hand is inside, the wrist camera looks at the part and
the fixture; the opening's frame and the clamps are behind the camera, around the forearm and wrist.

Phases: grasp the part (the enclosure interior is visible through the opening), carry it to the opening, insert
(hidden phase: tool 10 cm past the clamps), lower into the cradle, release and withdraw.
Randomized: enclosure position and bed height, fixture position, clamp positions, part position.
Difficulty knob: opening width in meters (smaller is harder).
Success: part upright and seated in the cradle, gripper open and withdrawn through the opening, no collision with the
enclosure, the clamps or the fixture.
"""

from __future__ import annotations

import torch

from isaaclab_arena.tasks.predicates.spatial import velocity_below_threshold
from isaaclab_arena_environments.spatial_memory.franka_controller import GRIPPER_CLOSE, Waypoint, pitched_quat
from isaaclab_arena_environments.spatial_memory.franka_waypoints import PITCH_45_RAD, pitched_pick, release_and_hold
from isaaclab_arena_environments.spatial_memory.geometry import ALL_FACES
from isaaclab_arena_environments.spatial_memory.layout_sampling import BOARD_THICKNESS_M, pose, uniform
from isaaclab_arena_environments.spatial_memory.parts import PartSpec, board, box_item
from isaaclab_arena_environments.spatial_memory.predicates import AT_REST_SPEED_MPS, part_upright, tool_position_e
from isaaclab_arena_environments.spatial_memory.task_logic import LayoutSample, PlanMode, SpatialMemoryTaskLogic
from isaaclab_arena_environments.spatial_memory.visibility import VisibilityTarget

ENCLOSURE_DEPTH_M = 0.36
"""Outer depth of the enclosure along x."""
ENCLOSURE_WIDTH_M = 0.64
"""Outer width of the enclosure along y."""
ENCLOSURE_HEIGHT_M = 0.50
"""Height of the enclosure's walls above the machine bed."""
OPENING_HEIGHT_M = 0.40
"""Height of the opening above the machine bed."""
PLINTH_HEIGHT_M = 0.30
"""Height of the plinth the enclosure sits on."""
PART_SIZE_M = (0.045, 0.045, 0.06)
"""Size of the part the robot loads into the machine."""
GRASP_HEIGHT_M = 0.035
"""Height of the tool center point above the part's base when grasping."""
CRADLE_BLOCK_M = (0.012, 0.06, 0.035)
"""Size of each of the cradle's front and back blocks."""
CRADLE_GAP_M = 0.065
"""Distance between the cradle's front and back blocks (part width plus clearance)."""
FIXTURE_DEPTH_M = 0.17
"""Distance of the cradle center behind the front wall."""
CLAMP_SIZE_M = (0.05, 0.05, 0.08)
"""Size of each clamp block."""
REST_JOINT_POSE = (-0.12, 0.4, 0.0, -2.0, 0.0, 2.9, 0.785)
"""Franka arm posture the scripted controller pulls towards in the inverse-kinematics null space."""
HIDDEN_PHASE_MARGIN_M = 0.10
"""Distance of the tool center point past the clamps from which the hidden phase starts."""
SEATED_SLACK_M = 0.003
"""Extra x distance beyond the cradle's free play within which the part still counts as seated."""
SEATED_Y_TOLERANCE_M = 0.03
"""Maximum y distance between the part's center and the cradle's center for the part to be seated."""
ON_BED_TOLERANCE_M = 0.015
"""Maximum vertical distance between the part's base and the machine bed for the part to rest on the bed."""
SEATED_MAX_TILT_DEG = 25.0
"""Maximum tilt of a seated part: a part leaning against a cradle block still counts as seated."""
WITHDRAWN_MARGIN_M = 0.04
"""Distance in front of the enclosure's front the tool center point must be for the arm to count as withdrawn."""
DROPPED_Z_M = -0.05
"""Environment-frame height below which the part counts as dropped."""


class MachineTendingLogic(SpatialMemoryTaskLogic):
    """Logic of the machine-tending task (see the module docstring)."""

    name = "spatial_memory_machine_tending"
    rest_joint_pose = REST_JOINT_POSE
    difficulty_name = "opening_width_m"
    default_difficulty = 0.44
    episode_length_s = 50.0
    task_description = "Pick up the part, place it in the fixture inside the machine, and withdraw the arm."

    def build_part_specs(self) -> list[PartSpec]:
        t = BOARD_THICKNESS_M
        side_width = 0.5 * (ENCLOSURE_WIDTH_M - self.difficulty)
        assert side_width > 0.02, "The opening is wider than the enclosure."
        return [
            board("table", (0.80, 1.40, 0.04), color="white"),
            board("plinth", (ENCLOSURE_DEPTH_M, ENCLOSURE_WIDTH_M, PLINTH_HEIGHT_M), color="dark"),
            board("bed", (ENCLOSURE_DEPTH_M, ENCLOSURE_WIDTH_M, t), color="gray"),
            board("encl_back", (t, ENCLOSURE_WIDTH_M, ENCLOSURE_HEIGHT_M), color="gray"),
            board("encl_left", (ENCLOSURE_DEPTH_M, t, ENCLOSURE_HEIGHT_M), color="gray"),
            board("encl_right", (ENCLOSURE_DEPTH_M, t, ENCLOSURE_HEIGHT_M), color="gray"),
            board("encl_roof", (ENCLOSURE_DEPTH_M, ENCLOSURE_WIDTH_M, t), color="gray"),
            board("front_left", (t, side_width, ENCLOSURE_HEIGHT_M), color="yellow"),
            board("front_right", (t, side_width, ENCLOSURE_HEIGHT_M), color="yellow"),
            board("lintel", (t, self.difficulty, ENCLOSURE_HEIGHT_M - OPENING_HEIGHT_M), color="yellow"),
            board("cradle_front", CRADLE_BLOCK_M, color="dark"),
            board("cradle_back", CRADLE_BLOCK_M, color="dark"),
            board("clamp_left", CLAMP_SIZE_M, color="orange"),
            board("clamp_right", CLAMP_SIZE_M, color="orange"),
            box_item("part", PART_SIZE_M, color="cyan", mass_kg=0.1),
        ]

    def sample_layout(self, generator: torch.Generator) -> LayoutSample:
        t = BOARD_THICKNESS_M
        poses = self.parked_part_poses()
        front_x = uniform(generator, 0.47, 0.51)
        center_y = uniform(generator, -0.10, -0.04)
        bed_z = uniform(generator, 0.22, 0.26)  # top face of the machine bed
        fixture_y = center_y + uniform(generator, -0.03, 0.03)
        fixture_x = front_x + FIXTURE_DEPTH_M
        clamp_x = front_x + uniform(generator, 0.035, 0.055)
        clamp_left_y = fixture_y + uniform(generator, 0.16, 0.19)
        clamp_right_y = fixture_y - uniform(generator, 0.16, 0.19)
        center_x = front_x + 0.5 * ENCLOSURE_DEPTH_M
        wall_z = bed_z + 0.5 * ENCLOSURE_HEIGHT_M
        side_width = 0.5 * (ENCLOSURE_WIDTH_M - self.difficulty)
        poses["table"] = pose(0.60, 0.0, -0.02)
        poses["plinth"] = pose(center_x, center_y, bed_z - t - 0.5 * PLINTH_HEIGHT_M)
        poses["bed"] = pose(center_x, center_y, bed_z - 0.5 * t)
        poses["encl_back"] = pose(front_x + ENCLOSURE_DEPTH_M - 0.5 * t, center_y, wall_z)
        poses["encl_left"] = pose(center_x, center_y + 0.5 * ENCLOSURE_WIDTH_M - 0.5 * t, wall_z)
        poses["encl_right"] = pose(center_x, center_y - 0.5 * ENCLOSURE_WIDTH_M + 0.5 * t, wall_z)
        poses["encl_roof"] = pose(center_x, center_y, bed_z + ENCLOSURE_HEIGHT_M + 0.5 * t)
        front_wall_x = front_x - 0.5 * t
        poses["front_left"] = pose(front_wall_x, center_y + 0.5 * self.difficulty + 0.5 * side_width, wall_z)
        poses["front_right"] = pose(front_wall_x, center_y - 0.5 * self.difficulty - 0.5 * side_width, wall_z)
        poses["lintel"] = pose(front_wall_x, center_y, bed_z + 0.5 * (OPENING_HEIGHT_M + ENCLOSURE_HEIGHT_M))
        cradle_z = bed_z + 0.5 * CRADLE_BLOCK_M[2]
        cradle_offset = 0.5 * (CRADLE_GAP_M + CRADLE_BLOCK_M[0])
        poses["cradle_front"] = pose(fixture_x - cradle_offset, fixture_y, cradle_z)
        poses["cradle_back"] = pose(fixture_x + cradle_offset, fixture_y, cradle_z)
        clamp_z = bed_z + 0.5 * CLAMP_SIZE_M[2]
        poses["clamp_left"] = pose(clamp_x, clamp_left_y, clamp_z)
        poses["clamp_right"] = pose(clamp_x, clamp_right_y, clamp_z)
        part_x = uniform(generator, 0.36, 0.42)
        part_y = uniform(generator, 0.30, 0.38)
        poses["part"] = pose(part_x, part_y, 0.5 * PART_SIZE_M[2] + 0.001)
        params = {
            "front_x": torch.tensor(front_x),
            "center_y": torch.tensor(center_y),
            "bed_z": torch.tensor(bed_z),
            "fixture_xy": torch.tensor([fixture_x, fixture_y]),
            "clamp_ys": torch.tensor([clamp_left_y, clamp_right_y]),
            "clamp_x": torch.tensor(clamp_x),
            "part_start_xy": torch.tensor([part_x, part_y]),
        }
        return LayoutSample(part_poses_e=self.part_poses_tensor(poses), layout_params=params)

    # ----------------------------------------------------------------------------------------------------------------

    def visibility_targets(self) -> list[VisibilityTarget]:
        return [
            VisibilityTarget("opening_left", "front_left", ALL_FACES),
            VisibilityTarget("opening_right", "front_right", ALL_FACES),
            VisibilityTarget("opening_top", "lintel", ALL_FACES),
            VisibilityTarget("clamp_left", "clamp_left", ALL_FACES, spacing_m=0.02),
            VisibilityTarget("clamp_right", "clamp_right", ALL_FACES, spacing_m=0.02),
        ]

    def hidden_target_names(self) -> list[str]:
        return ["opening_left", "opening_right", "opening_top", "clamp_left", "clamp_right"]

    def manipulated_item_names(self) -> list[str]:
        return ["part"]

    def item_support_names(self) -> list[str]:
        return ["table", "bed", "cradle_front", "cradle_back"]

    def hidden_phase(self, env, episode) -> torch.Tensor:
        # Inside: the tool is deeper than the clamps, so the hand and wrist are between the clamps and the frame.
        t_E_tcp = tool_position_e(env, self)
        return t_E_tcp[:, 0] > episode.layout_params["clamp_x"] + HIDDEN_PHASE_MARGIN_M

    def success_terms(self, env, episode) -> dict[str, torch.Tensor]:
        params = episode.layout_params
        t_E_part = env.arena_world.get_pose_e("part")[:, :3]
        t_E_fixture_xy = params["fixture_xy"]
        seated_x = (t_E_part[:, 0] - t_E_fixture_xy[:, 0]).abs() < 0.5 * (
            CRADLE_GAP_M - PART_SIZE_M[0]
        ) + SEATED_SLACK_M
        seated_y = (t_E_part[:, 1] - t_E_fixture_xy[:, 1]).abs() < SEATED_Y_TOLERANCE_M
        on_bed = (t_E_part[:, 2] - 0.5 * PART_SIZE_M[2] - params["bed_z"]).abs() < ON_BED_TOLERANCE_M
        withdrawn = tool_position_e(env, self)[:, 0] < params["front_x"] - WITHDRAWN_MARGIN_M
        return {
            "part_in_cradle": seated_x & seated_y,
            "part_on_bed": on_bed,
            "part_upright": part_upright(env, "part", max_tilt_deg=SEATED_MAX_TILT_DEG),
            "part_at_rest": velocity_below_threshold(env, "part", AT_REST_SPEED_MPS),
            "released_and_withdrawn": ~self.gripper_closed(env) & withdrawn,
        }

    def extra_failures(self, env, episode) -> torch.Tensor:
        return env.arena_world.get_pose_e("part")[:, 2] < DROPPED_Z_M

    # ----------------------------------------------------------------------------------------------------------------

    def plan(self, env, episode, env_id: int, mode: str) -> list[Waypoint]:
        params = {key: value[env_id] for key, value in episode.layout_params.items()}
        part_x, part_y = (float(v) for v in params["part_start_xy"])
        fixture_x, fixture_y = (float(v) for v in params["fixture_xy"])
        front_x = float(params["front_x"])
        bed_z = float(params["bed_z"])
        q_E_tcp = pitched_quat(0.0, PITCH_45_RAD)
        place_z = bed_z + GRASP_HEIGHT_M + 0.003
        carry_z = bed_z + GRASP_HEIGHT_M + CRADLE_BLOCK_M[2] + 0.04
        plan = pitched_pick((part_x, part_y), GRASP_HEIGHT_M)
        if mode == PlanMode.ORACLE:
            # Line up with the corridor between the clamps outside the opening, then insert straight in.
            lane_y = 0.5 * float(params["clamp_ys"][0] + params["clamp_ys"][1])
            plan += [
                Waypoint((part_x, part_y, carry_z + 0.03), q_E_tcp, GRIPPER_CLOSE, label="lift", max_speed_mps=0.15),
                Waypoint(
                    (front_x - 0.18, lane_y, carry_z), q_E_tcp, GRIPPER_CLOSE, label="preinsert", max_speed_mps=0.25
                ),
                Waypoint((fixture_x, lane_y, carry_z), q_E_tcp, GRIPPER_CLOSE, label="insert", max_speed_mps=0.15),
                Waypoint(
                    (fixture_x, fixture_y, carry_z),
                    q_E_tcp,
                    GRIPPER_CLOSE,
                    label="align",
                    max_speed_mps=0.08,
                    align_item="part",
                ),
                Waypoint(
                    (fixture_x, fixture_y, place_z),
                    q_E_tcp,
                    GRIPPER_CLOSE,
                    label="lower",
                    max_speed_mps=0.05,
                    align_item="part",
                ),
            ]
            exit_y = lane_y
        else:
            # Straight from the lift pose to the fixture, ignoring the frame and the clamps.
            plan += [
                Waypoint((part_x, part_y, carry_z + 0.03), q_E_tcp, GRIPPER_CLOSE, label="lift", max_speed_mps=0.1),
                Waypoint((fixture_x, fixture_y, place_z), q_E_tcp, GRIPPER_CLOSE, label="straight", max_speed_mps=0.15),
            ]
            exit_y = fixture_y
        plan += [
            Waypoint((fixture_x, fixture_y, place_z), q_E_tcp, label="release", hold_steps=8),
            Waypoint((fixture_x, fixture_y, carry_z), q_E_tcp, label="raise", max_speed_mps=0.08),
            Waypoint((fixture_x, exit_y, carry_z), q_E_tcp, label="to_lane", max_speed_mps=0.08),
        ]
        plan += release_and_hold((fixture_x, exit_y, carry_z), q_E_tcp, (front_x - 0.12, exit_y, carry_z + 0.02))
        return plan
