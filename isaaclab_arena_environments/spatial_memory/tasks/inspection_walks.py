# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Inspection and navigation courses scored from the G1 body pose."""

import math
import torch

from isaaclab_arena_environments.spatial_memory.tasks.workcell import Fixture, Milestone, SurveyedWorkcell


class ServiceTrench(SurveyedWorkcell):
    name = "spatial_memory_service_trench"
    task_description = "Walk headfirst to the end of the narrow service trench. Back out to the mouth facing the end."
    example = "A technician backs out of a server aisle after inspecting a rear connector."
    survey_targets = ("left_rack", "end_wall")
    approach = ("left_rack", 1.39, -0.6)
    fixtures = (
        Fixture("left_rack", (-2.55, 0.60, 0.79), (2.9, 0.16, 1.58)),
        Fixture("right_rack", (-2.55, -0.60, 0.79), (2.9, 0.16, 1.58)),
        Fixture("end_wall", (-4.04, 0, 0.55), (0.12, 1.35, 1.10), "blue"),
    )
    stages = (
        Milestone("inspected_end", "pelvis", "end_wall", (0.50, 0, 0), yaw=math.pi, hold_s=0.40),
        Milestone("reversed_halfway", "pelvis", "end_wall", (1.65, 0, 0), yaw=math.pi, retreat_m=0.80),
        Milestone("backed_clear", "pelvis", "end_wall", (3.10, 0, 0), yaw=math.pi, retreat_m=1.05),
    )


class DockEdge(SurveyedWorkcell):
    name = "spatial_memory_dock_edge"
    task_description = "Back toward an inspected dock sill, stop at the standoff line facing away, then walk clear."
    example = "A dock spotter backs up to a loading-bay edge while watching the approaching vehicle."
    survey_targets = ("sill", "wheel_guide")
    approach = ("sill", 1.34, 0)
    fixtures = (
        Fixture("sill", (-3.25, 0, 0.16), (0.20, 1.85, 0.32), "yellow"),
        Fixture("wheel_guide", (-2.55, 0.97, 0.21), (1.55, 0.15, 0.42)),
    )
    stages = (
        Milestone("staged", "pelvis", "sill", (1.40, 0, 0), yaw=0),
        Milestone("stopped_short", "pelvis", "sill", (0.48, 0, 0), radius=0.12, yaw=0, hold_s=1, retreat_m=0.65),
        Milestone("cleared_dock", "pelvis", "sill", (1.70, 0, 0), yaw=0),
    )


class SidewaysStacks(SurveyedWorkcell):
    name = "spatial_memory_sideways_stacks"
    task_description = "Face the shelving and sidestep between the stacks, maintaining that heading at three stations."
    example = "An archivist shuffles sideways between compact shelves while reading the spines."
    survey_targets = ("low_stack", "high_stack")
    approach = ("low_stack", 1.44, -0.63)
    fixtures = (
        Fixture("low_stack", (-2.65, 0.63, 0.56), (2.8, 0.18, 1.12), "wood"),
        Fixture("high_stack", (-2.65, -0.63, 0.82), (2.8, 0.18, 1.64)),
    )
    stages = tuple(
        Milestone(label, "pelvis", "low_stack", (x, -0.63, 0), yaw=math.pi / 2)
        for label, x in (("entered_sideways", 1.10), ("passed_midpoint", 0), ("cleared_stack", -1.25))
    )


class LowHeader(SurveyedWorkcell):
    name = "spatial_memory_low_header"
    task_description = "Lower the pelvis beneath the inspected lintel and emerge on its far side."
    example = "A maintenance worker ducks under ductwork in a mechanical room."
    survey_targets = ("lintel", "jamb_a")
    approach = ("lintel", 0.96, 0)
    fixtures = (
        Fixture("lintel", (-2.65, 0, 1.30), (0.55, 1.60, 0.10), "orange"),
        Fixture("jamb_a", (-2.65, 0.78, 0.63), (0.55, 0.12, 1.26)),
        Fixture("jamb_b", (-2.65, -0.78, 0.63), (0.55, 0.12, 1.26)),
    )
    stages = (
        Milestone("approached_header", "pelvis", "lintel", (0.72, 0, 0)),
        Milestone("lowered_under_header", "pelvis", "lintel", (0, 0, 0), max_pelvis_height=0.63),
        Milestone("emerged", "pelvis", "lintel", (-0.75, 0, 0)),
    )


class TankPerimeter(SurveyedWorkcell):
    approach_by_bearing = True
    name = "spatial_memory_tank_perimeter"
    task_description = "Inspect all four faces of an enclosed tank in order and return to the first face."
    example = "A plant operator checks a vessel from each side before signing off a walkaround."
    survey_targets = ("tank", "pipe_chase")
    approach = ("tank", 1.41, 0.45)
    fixtures = (
        Fixture("tank", (-3.10, 0, 0.84), (1.38, 1.42, 1.68), "blue"),
        Fixture("pipe_chase", (-4.63, 0.15, 0.42), (0.12, 2.8, 0.84)),
    )
    stages = tuple(
        Milestone(label, "pelvis", "tank", (x, y, 0), yaw=yaw, hold_s=0.35)
        for label, x, y, yaw in (
            ("front_face", 1.17, 0, math.pi),
            ("side_face", 0, 1.18, -math.pi / 2),
            ("rear_face", -1.13, 0, 0),
            ("return_face", 0, -1.18, math.pi / 2),
            ("closed_loop", 1.17, 0, math.pi),
        )
    )


class Sightline(SurveyedWorkcell):
    name = "spatial_memory_sightline"
    task_description = "Inspect a pair of survey posts, then occupy their extended sightline and face along it."
    example = "A surveyor reoccupies a line of stakes after walking back to the instrument station."
    survey_targets = ("near_post", "far_post")
    approach = ("near_post", 1.78, 0.0)
    fixtures = (
        Fixture("near_post", (-3.0, 0.10, 0.60), (0.12, 0.12, 1.2), "yellow"),
        Fixture("far_post", (-3.90, 0.55, 0.78), (0.12, 0.12, 1.56), "blue"),
    )
    stages = (
        Milestone("sighted_line", "pelvis", "near_post", (1.00, 0, 0), radius=0.12, yaw=math.pi, hold_s=0.80),
        Milestone(
            "resighted_from_baseline", "pelvis", "near_post", (1.73, 0, 0), radius=0.12, yaw=math.pi, hold_s=0.80
        ),
    )

    def sample_layout(self, generator):
        layout = super().sample_layout(generator)
        offset = torch.rand((), generator=generator) * 0.56 - 0.28
        angle = layout.layout_params["cell_yaw"]
        lateral = torch.stack((-angle.sin(), angle.cos()))
        layout.part_poses_e[self.part_names().index("far_post"), :2] += offset * lateral
        layout.layout_params["far_post_offset"] = offset
        return layout

    def reference_pose(self, env, name):
        T_E_ref = super().reference_pose(env, name)
        if name == "near_post":
            far = env.arena_world.get_pose_e("far_post")
            baseline = T_E_ref[:, :2] - far[:, :2]
            half_yaw = torch.atan2(baseline[:, 1], baseline[:, 0]) / 2
            T_E_ref[:, 3:5] = 0
            T_E_ref[:, 5] = half_yaw.sin()
            T_E_ref[:, 6] = half_yaw.cos()
        return T_E_ref
