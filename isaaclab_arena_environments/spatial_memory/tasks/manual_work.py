# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Manual inspection, handling and placement operations with measured hand and object milestones."""

import math
import torch

from isaaclab_arena_environments.spatial_memory.g1_adapter import G1_FLOOR_Z_M
from isaaclab_arena_environments.spatial_memory.layout_sampling import pose
from isaaclab_arena_environments.spatial_memory.tasks.workcell import Fixture, Milestone, SurveyedWorkcell, heading


class RecessedReset(SurveyedWorkcell):
    preset_hand = ("left", "pad")
    name = "spatial_memory_recessed_reset"
    task_description = (
        "Preset the left palm to the surveyed pad height, touch the recessed control, then withdraw the hand."
    )
    example = "A technician reaches into a recessed reset station under a machine guard."
    survey_targets = ("pad", "hood")
    approach = ("pad", 0.78, 0)
    fixtures = (
        Fixture("pad", (-2.52, 0, 1.02), (0.08, 0.18, 0.18), "green"),
        Fixture("hood", (-2.39, 0, 1.25), (0.43, 0.60, 0.08)),
        Fixture("housing", (-2.62, 0, 0.67), (0.12, 0.67, 1.34)),
    )
    touchable = ("pad",)
    stages = (
        Milestone("reached_pad", "left", "pad", (0.075, 0, 0), radius=0.09, contact_parts=("pad",), hold_s=0.34),
        Milestone("withdrew_hand", "left", "pad", (0.42, 0, 0), radius=0.15),
    )

    def sample_layout(self, generator):
        layout = super().sample_layout(generator)
        height = (torch.randint(4, (), generator=generator) - 1.5) * 0.12
        for name in ("pad", "hood"):
            layout.part_poses_e[self.part_names().index(name), 2] += height
        layout.layout_params["pad_height_offset"] = height
        return layout


class TwoHandInterlock(SurveyedWorkcell):
    name = "spatial_memory_two_hand_interlock"
    task_description = "Hold both palms at the two separated interlock pads simultaneously."
    example = "A press operator uses both hands on separated controls to enable a cycle."
    survey_targets = ("pad_a", "pad_b")
    approach = ("paired_panel", 0.79, 0)
    fixtures = (
        Fixture("console", (-2.58, 0, 0.50), (0.42, 0.92, 1.0), "blue"),
        Fixture("paired_panel", (-2.49, 0, 1.03), (0.17, 0.78, 0.06), "wood"),
        Fixture("pad_a", (-2.49, 0.28, 1.08), (0.14, 0.12, 0.04), "green"),
        Fixture("pad_b", (-2.49, -0.28, 1.08), (0.14, 0.12, 0.04), "green"),
    )
    touchable = ("paired_panel", "pad_a", "pad_b")
    stages = (
        Milestone(
            "dual_enable",
            "left",
            "paired_panel",
            (0.02, -0.28, 0.095),
            radius=0.105,
            contact_parts=("pad_a", "pad_b"),
            hold_s=0.60,
        ),
    )


class SeamTrace(SurveyedWorkcell):
    name = "spatial_memory_seam_trace"
    task_description = "Follow a stepped seam with the right palm, passing the start, bend and raised end in order."
    example = "A maintenance inspector follows a weld bead around a step in a metal enclosure."
    survey_targets = ("low_seam", "high_seam")
    approach = ("low_seam", 0.8, -0.22)
    fixtures = (
        Fixture("seam_plate", (-2.67, 0, 0.70), (0.12, 1.25, 1.40)),
        Fixture("low_seam", (-2.58, -0.22, 0.89), (0.06, 0.50, 0.06), "yellow"),
        Fixture("riser", (-2.58, 0.02, 1.05), (0.06, 0.06, 0.35), "yellow"),
        Fixture("high_seam", (-2.58, 0.23, 1.21), (0.06, 0.47, 0.06), "yellow"),
    )
    touchable = ("low_seam", "riser", "high_seam")
    stages = tuple(
        Milestone(label, "right", "seam_plate", (0.16, y, z), radius=0.11)
        for label, y, z in (
            ("start_seam", -0.43, 0.19),
            ("lower_corner", 0, 0.19),
            ("upper_corner", 0, 0.51),
            ("end_seam", 0.43, 0.51),
        )
    )


class BlindReturnHook(SurveyedWorkcell):
    name = "spatial_memory_blind_return_hook"
    task_description = "Guide a palm around the free edge of a screen, behind its flange and back out."
    example = "A mechanic reaches around a splash shield to find a concealed inspection point."
    survey_targets = ("screen", "back_pad")
    approach = ("screen", 0.74, 0.45)
    fixtures = (
        Fixture("screen", (-2.55, 0, 0.96), (0.10, 0.62, 0.50), "wood"),
        Fixture("back_pad", (-2.73, 0.16, 0.99), (0.12, 0.14, 0.14), "green"),
    )
    touchable = ("back_pad",)
    stages = (
        Milestone("rounded_edge", "right", "screen", (0, 0.45, 0), radius=0.09),
        Milestone("found_hidden_pad", "right", "back_pad", (0, 0.09, 0), radius=0.09, contact_parts=("back_pad",)),
        Milestone("cleared_flange", "right", "screen", (0.09, 0.46, 0), radius=0.12),
    )


class FloorSled(SurveyedWorkcell):
    name = "spatial_memory_floor_sled"
    task_description = (
        "Set the floor sled moving toward the remembered bumper, park it without lifting it, then step clear."
    )
    example = "A warehouse worker nudges a low parts dolly into its marked parking stop."
    survey_targets = ("bumper", "side_guide")
    approach = ("bumper", 2.29, 0)
    fixtures = (
        Fixture("sled", (-1.70, 0, 0.12), (0.40, 0.44, 0.23), "orange", movable=True),
        Fixture("side_guide", (-2.46, 0.55, 0.16), (2, 0.13, 0.32)),
        Fixture("bumper", (-3.34, 0, 0.14), (0.14, 1.0, 0.28), "yellow"),
    )
    supports = ("bumper",)
    stages = (
        Milestone("sled_in_guide", "sled", "bumper", (0.88, 0, -0.02), radius=0.16),
        Milestone("sled_parked", "sled", "bumper", (0.31, 0, -0.02), radius=0.13, hold_s=0.6),
        Milestone("operator_clear", "pelvis", "bumper", (1.22, 0, 0)),
    )

    def sample_layout(self, generator):
        layout = super().sample_layout(generator)
        layout.part_poses_e[self.part_names().index("sled")] = torch.tensor(pose(-0.73, 0, G1_FLOOR_Z_M + 0.12))
        return layout

    def _check_approach(self, env, episode, active, pelvis):
        entered = self.commitment_made(env, episode, active)
        sled = env.arena_world.get_pose_e("sled")
        origin = episode.layout_poses_e[:, episode.part_index["sled"], :2]
        displacement = sled[:, :2] - origin
        crossing = active & ~entered & (displacement.norm(dim=1) >= 0.20)
        direction = torch.nn.functional.normalize(displacement, dim=1)
        to_bumper = env.arena_world.get_pose_e("bumper")[:, :2] - origin
        correct = (direction * torch.nn.functional.normalize(to_bumper, dim=1)).sum(dim=1) > math.cos(0.12)
        wrong = crossing & ~(episode.instruction_revealed & correct)
        episode.task_state_buffer("wrong_entrance", False, torch.bool).logical_or_(wrong)
        entered |= crossing

    def extra_failures(self, env, episode):
        raised = env.arena_world.get_pose_e("sled")[:, 2] - episode.layout_poses_e[:, episode.part_index["sled"], 2]
        return super().extra_failures(env, episode) | (raised > 0.09)


class CounterSlide(SurveyedWorkcell):
    name = "spatial_memory_counter_slide"
    task_description = "Slide a tray laterally under a low pass-through cover and leave it at the outlet."
    example = "A canteen worker slides a tray along a counter through a low serving hatch."
    survey_targets = ("cover", "counter")
    approach = ("counter", 0.88, -0.51)
    fixtures = (
        Fixture("counter", (-2.55, 0, 0.42), (0.75, 1.72, 0.84), "wood"),
        Fixture("cover", (-2.55, 0, 1.10), (0.77, 0.53, 0.09)),
        Fixture("tray", (-2.55, -0.51, 0.88), (0.31, 0.35, 0.075), "orange", movable=True),
    )
    supports = ("counter",)
    stages = (
        Milestone("through_hatch", "tray", "counter", (0, 0, 0.46), radius=0.12),
        Milestone("at_outlet", "tray", "counter", (0, 0.55, 0.46), radius=0.13, released=True, hold_s=0.50),
    )

    def extra_failures(self, env, episode):
        height_above_counter = env.arena_world.get_pose_e("tray")[:, 2] - env.arena_world.get_pose_e("counter")[:, 2]
        return super().extra_failures(env, episode) | ((height_above_counter - 0.46).abs() > 0.08)


class JigRotation(SurveyedWorkcell):
    name = "spatial_memory_jig_rotation"
    task_description = (
        "Align a long workpiece to the remembered jig before lifting it from staging, then settle it beside the"
        " locating fence."
    )
    example = "A machinist aligns a long part to a remote fixture before moving it off the staging bench."
    survey_targets = ("fence", "bench")
    approach = ("fence", 1.2, -0.15)
    fixtures = (
        Fixture("staging_bench", (0.48, 0, 0.40), (0.58, 0.84, 0.80), "wood"),
        Fixture("bench", (-2.51, 0, 0.40), (1.30, 1.32, 0.80), "wood"),
        Fixture("fence", (-2.65, 0.08, 0.88), (0.07, 0.70, 0.16)),
        Fixture("bar", (0.48, 0, 0.86), (0.49, 0.17, 0.115), "orange", movable=True),
    )
    supports = ("staging_bench", "bench", "fence")
    stages = (
        Milestone(
            "aligned_in_jig",
            "bar",
            "fence",
            (0.20, 0, -0.02),
            radius=0.12,
            yaw=math.pi / 2,
            symmetric_heading=True,
            support="bench",
            released=True,
            hold_s=0.7,
        ),
    )

    def sample_layout(self, generator):
        layout = super().sample_layout(generator)
        names = self.part_names()
        layout.part_poses_e[names.index("staging_bench")] = torch.tensor(pose(0.48, 0, G1_FLOOR_Z_M + 0.40))
        layout.part_poses_e[names.index("bar")] = torch.tensor(pose(0.48, 0, G1_FLOOR_Z_M + 0.86))
        fence = layout.part_poses_e[names.index("fence")]
        yaw = heading(fence[3:]) + torch.randint(4, (), generator=generator) * (math.pi / 4)
        fence[5], fence[6] = (yaw / 2).sin(), (yaw / 2).cos()
        layout.layout_params["fence_yaw"] = yaw
        return layout

    def _check_approach(self, env, episode, active, pelvis):
        entered = self.commitment_made(env, episode, active)
        bar = env.arena_world.get_pose_e("bar")
        home = episode.layout_poses_e[:, episode.part_index["bar"]]
        # Lift from the staging bench after inspecting the remote jig.
        lifted = bar[:, 2] - home[:, 2] > 0.09
        on_bench = (bar[:, :2] - home[:, :2]).norm(dim=1) < 0.32
        crossing = active & ~entered & lifted
        error = heading(bar[:, 3:]) - episode.layout_params["fence_yaw"] - math.pi / 2
        aligned = error.cos().abs() > math.cos(0.25)
        episode.task_state_buffer("wrong_entrance", False, torch.bool).logical_or_(
            crossing & ~(aligned & on_bench & episode.instruction_revealed)
        )
        entered |= crossing


class ToteExtraction(SurveyedWorkcell):
    name = "spatial_memory_tote_extraction"
    task_description = "Draw a tote out from under a shelf lip. Bring it beyond the apron before raising it."
    example = "A stockroom worker pulls a parts bin straight out before lifting it past the shelf above."
    survey_targets = ("shelf_lip", "shelf")
    approach = ("shelf", 0.92, 0)
    fixtures = (
        Fixture("shelf", (-2.56, 0, 0.39), (0.83, 0.95, 0.78), "wood"),
        Fixture("shelf_lip", (-2.70, 0, 1.18), (0.58, 0.95, 0.09)),
        Fixture("tote", (-2.59, 0, 0.89), (0.32, 0.29, 0.21), "orange", movable=True),
    )
    supports = ("shelf",)
    stages = (
        Milestone("withdrew_level", "tote", "shelf", (0.36, 0, 0.50), radius=0.12),
        Milestone("lifted_clear", "tote", "shelf", (0.70, 0, 0.71), radius=0.13),
    )


class ScaleTransfer(SurveyedWorkcell):
    name = "spatial_memory_scale_transfer"
    task_description = "Set the case down on the scale, release it for a reading, then remove it to the outbound pad."
    example = "A parcel clerk weighs a box before moving it to the dispatch counter."
    survey_targets = ("scale", "outbound")
    approach = ("scale", 0.92, -0.72)
    fixtures = (
        Fixture("bench", (-2.60, 0, 0.38), (0.70, 2.20, 0.76), "wood"),
        Fixture("scale", (-2.55, 0, 0.80), (0.47, 0.44, 0.08), "blue"),
        Fixture("outbound", (-2.55, 0.72, 0.78), (0.45, 0.44, 0.04), "green"),
        Fixture("case", (-2.55, -0.72, 0.87), (0.24, 0.25, 0.21), "orange", movable=True),
    )
    supports = ("bench", "scale", "outbound")
    stages = (
        Milestone("weighed", "case", "scale", (0, 0, 0.145), radius=0.085, released=True, hold_s=1.2),
        Milestone("dispatched", "case", "outbound", (0, 0, 0.125), radius=0.09, released=True, hold_s=0.6),
    )


class RaisedSill(SurveyedWorkcell):
    name = "spatial_memory_raised_sill"
    task_description = (
        "Lift the case above a raised sill, carry it across the divider and lower it onto the far support."
    )
    example = "A worker transfers a case over the raised lip separating two cleanroom counters."
    survey_targets = ("sill", "bench")
    approach = ("sill", 0.92, -0.51)
    fixtures = (
        Fixture("bench", (-2.48, 0, 0.37), (0.76, 1.72, 0.74), "wood"),
        Fixture("sill", (-2.48, 0, 0.89), (0.76, 0.09, 0.30), "yellow"),
        Fixture("case", (-2.48, -0.51, 0.85), (0.25, 0.25, 0.21), "orange", movable=True),
    )
    supports = ("bench",)
    stages = (
        Milestone("above_sill", "case", "sill", (0, 0, 0.34), radius=0.105),
        Milestone("lowered_beyond_sill", "case", "bench", (0, 0.51, 0.48), radius=0.12, released=True, hold_s=0.5),
    )
