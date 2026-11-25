# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""G1 traffic tasks combining surveyed fixed geometry with a live moving obstacle or support."""

from isaaclab_arena_environments.spatial_memory.tasks.workcell import Fixture, Milestone, Motion, SurveyedWorkcell


class CartRefuge(SurveyedWorkcell):
    approach_by_bearing = True
    name = "spatial_memory_cart_refuge"
    task_description = "Yield inside an inspected passing recess as the cart goes by; then rejoin the aisle."
    example = "A porter steps into a corridor recess to let a linen cart pass."
    survey_targets = ("alcove_back", "alcove_end")
    approach = ("alcove_back", 1.46, -0.52)
    fixtures = (
        Fixture("alcove_back", (-2.85, 1.21, 0.64), (1.35, 0.12, 1.28), "wood"),
        Fixture("alcove_end", (-3.53, 0.83, 0.64), (0.12, 0.85, 1.28)),
        Fixture("cart", (-2.78, 0, 0.45), (0.68, 0.57, 0.90), "blue"),
    )
    motions = (Motion("cart", "x", 1.12, 7.0),)
    stages = (
        Milestone(
            "took_refuge", "pelvis", "alcove_back", (0, -0.49, 0), hold_s=0.40, motion_window=("cart", "x", -0.24, 0.24)
        ),
        Milestone("rejoined_beyond", "pelvis", "alcove_back", (-1.10, -1.21, 0)),
    )


class SwingingBoom(SurveyedWorkcell):
    approach_by_bearing = True
    name = "spatial_memory_swinging_boom"
    task_description = "Round the outside of a sweeping boom and reach the service side of its pedestal."
    example = "A ground-crew member walks around the sweep of a pivoting baggage arm."
    survey_targets = ("pivot", "service_box")
    approach = ("pivot", 1.45, 1.63)
    fixtures = (
        Fixture("pivot", (-3.10, -0.56, 0.63), (0.27, 0.27, 1.26)),
        Fixture("boom", (-2.40, -0.56, 1.05), (1.40, 0.12, 0.12), "orange"),
        Fixture("service_box", (-3.98, -0.58, 0.62), (0.34, 0.56, 1.24), "wood"),
    )
    motions = (Motion("boom", "yaw", 1.16, 6.2, 0.70),)
    stages = (
        Milestone("rounded_sweep", "pelvis", "pivot", (0.10, 1.70, 0)),
        Milestone("reached_service_side", "pelvis", "service_box", (0, 0.58, 0)),
    )


class LiftGate(SurveyedWorkcell):
    name = "spatial_memory_lift_gate"
    task_description = "Stage at the remembered portal, cross beneath the raised shutter and clear its exit."
    example = "A depot worker crosses a loading portal when its overhead shutter is high enough."
    survey_targets = ("gate_post", "far_post")
    approach = ("gate_post", 0.95, -0.91)
    fixtures = (
        Fixture("gate_post", (-2.75, 0.91, 1.05), (0.28, 0.20, 2.10)),
        Fixture("far_post", (-2.75, -0.91, 1.05), (0.28, 0.20, 2.10)),
        Fixture("shutter", (-2.75, 0, 1.57), (0.21, 1.60, 0.24), "yellow"),
    )
    motions = (Motion("shutter", "z", 0.50, 9.5),)
    stages = (
        Milestone("staged_at_shutter", "pelvis", "gate_post", (0.76, -0.91, 0)),
        Milestone(
            "crossed_raised_shutter", "pelvis", "gate_post", (0, -0.91, 0), motion_window=("shutter", "z", 0.23, 0.51)
        ),
        Milestone("cleared_threshold", "pelvis", "gate_post", (-0.75, -0.91, 0)),
    )


class ShuttleEscort(SurveyedWorkcell):
    name = "spatial_memory_shuttle_escort"
    task_description = "Meet the shuttle beside its guide rail, escort it past the midpoint and leave at the far berth."
    example = "A factory attendant walks beside a slow inspection shuttle between two work bays."
    survey_targets = ("guide", "berth")
    approach = ("guide", 1.7, 1.08)
    fixtures = (
        Fixture("guide", (-3.10, -0.42, 0.14), (3.40, 0.12, 0.28)),
        Fixture("shuttle", (-3.10, 0, 0.44), (0.58, 0.45, 0.88), "blue"),
        Fixture("berth", (-4.48, 1.25, 0.44), (0.70, 0.12, 0.88), "green"),
    )
    motions = (Motion("shuttle", "x", 1.18, 22.0),)
    stages = tuple(
        Milestone(
            label,
            "pelvis",
            "shuttle",
            (0, 0.66, 0),
            radius=0.25,
            hold_s=0.35,
            motion_window=("shuttle", "x", low, high),
        )
        for label, low, high in (
            ("met_shuttle", 0.66, 1.20),
            ("escorted_midway", -0.22, 0.22),
            ("escorted_to_berth", -1.2, -0.68),
        )
    ) + (Milestone("left_at_berth", "pelvis", "berth", (0, -0.40, 0)),)


class MovingReceiver(SurveyedWorkcell):
    name = "spatial_memory_moving_receiver"
    task_description = (
        "Lift the parcel from the fixed bench and release it on the moving receiver near the dispatch end."
    )
    example = "A packer loads a parcel onto a slow transfer carriage."
    survey_targets = ("bench",)
    approach = ("bench", 0.84, 0)
    fixtures = (
        Fixture("bench", (-2.28, -0.70, 0.38), (0.67, 0.64, 0.76), "wood"),
        Fixture("parcel", (-2.28, -0.70, 0.87), (0.24, 0.24, 0.21), "orange", movable=True),
        Fixture("receiver", (-2.75, 0.54, 0.38), (0.58, 0.55, 0.76), "blue"),
    )
    supports = ("bench", "receiver")
    motions = (Motion("receiver", "x", 0.62, 18.0),)
    stages = (
        Milestone("lifted_from_bench", "parcel", "bench", (0, 0, 0.70), radius=0.16),
        Milestone(
            "released_on_receiver",
            "parcel",
            "receiver",
            (0, 0, 0.485),
            radius=0.12,
            released=True,
            hold_s=0.32,
            motion_window=("receiver", "x", -0.63, -0.52),
        ),
    )


class BollardChicane(SurveyedWorkcell):
    name = "spatial_memory_bollard_chicane"
    task_description = "Negotiate two offset retracting bollards. Cross each footprint only when that bollard is down."
    example = "A yard inspector crosses a staggered pair of retractable vehicle barriers."
    survey_targets = ("divider",)
    approach = ("divider", 1.55, 0.45)
    fixtures = (
        Fixture("divider", (-2.94, 0, 0.33), (0.34, 0.56, 0.66)),
        Fixture("bollard_a", (-2.22, 0.45, 0.05), (0.23, 0.23, 0.90), "orange"),
        Fixture("bollard_b", (-3.62, -0.45, 0.05), (0.23, 0.23, 0.90), "orange"),
    )
    motions = (Motion("bollard_a", "z", 0.56, 8.0), Motion("bollard_b", "z", 0.56, 11.0))
    stages = (
        Milestone(
            "passed_first_socket", "pelvis", "bollard_a", (0, 0, 0), motion_window=("bollard_a", "z", -0.57, -0.50)
        ),
        Milestone("rounded_divider", "pelvis", "divider", (0, 0.73, 0)),
        Milestone(
            "passed_second_socket", "pelvis", "bollard_b", (0, 0, 0), motion_window=("bollard_b", "z", -0.57, -0.50)
        ),
        Milestone("cleared_barriers", "pelvis", "divider", (-1.32, -0.45, 0)),
    )
