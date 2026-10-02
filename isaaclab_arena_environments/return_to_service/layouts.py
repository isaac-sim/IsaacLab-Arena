# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Select coherent workcell placements and expose task-specific DROID reach targets."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

from isaaclab_arena.relations.placement_layouts import PlacementLayouts
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from .scene import ServiceLayout, ServiceScene


LAYOUT_NAMES = ("baseline", "translated_left", "rotated_right")
"""Named build-time layouts; the fixed robot and bench are never transformed."""

_FIXED_ROOTS = frozenset(("bench", "floor", "lighting"))


def _layout_transform(name: str) -> Pose:
    """Return one rigid transform applied to every workstation member."""
    assert name in LAYOUT_NAMES, f"Unknown service layout {name!r}; choose one of {LAYOUT_NAMES}"
    if name == "baseline":
        return Pose()
    if name == "translated_left":
        return Pose((0.0, 0.01, 0.0))
    # Rotate about the cradle's horizontal neighborhood rather than the robot base.
    yaw = math.radians(-1.0)
    pivot_x = 0.46
    return Pose(
        (pivot_x * (1.0 - math.cos(yaw)), -pivot_x * math.sin(yaw), 0.0),
        (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)),
    )


def apply_service_layout(workcell: ServiceScene, layout_name: str = "baseline") -> ServiceScene:
    """Apply a selected coherent layout before constructing the simulation.

    Args:
        workcell: Fresh baseline scene; its assets are updated in place before spawning.
        layout_name: Named bank member. Baseline returns the original scene unchanged.

    Returns:
        Scene with matching spawn poses and world-frame geometry metadata. All service
        fixtures, buttons, contents and coupled connectors share one rigid transform.
        Static roots select one layout per environment build; this is not a per-reset
        static-fixture randomizer or a certificate of collision-free robot trajectories.
    """
    layout = select_service_layout(workcell.layout, layout_name)
    if layout is workcell.layout:
        return workcell
    for name, pose in layout.initial_poses.items():
        if name not in _FIXED_ROOTS:
            workcell.assets[name].set_initial_pose(pose)
    return replace(workcell, layout=layout)


def select_service_layout(baseline: ServiceLayout, layout_name: str = "baseline") -> ServiceLayout:
    """Select geometry metadata independently of simulator or asset construction.

    Args:
        baseline: Unmodified baseline layout from build_service_layout.
        layout_name: Named coherent workcell transform.

    Returns:
        Selected geometry contract; the baseline input is never mutated.
    """
    transform = _layout_transform(layout_name)
    if layout_name == "baseline":
        return baseline
    poses = {}
    for name, pose in baseline.initial_poses.items():
        poses[name] = pose if name in _FIXED_ROOTS else transform.multiply(pose)
    regions = {}
    for name, region in baseline.regions.items():
        T_E_R = transform.multiply(Pose(region.center_xyz, region.rotation_xyzw))
        regions[name] = replace(region, center_xyz=T_E_R.position_xyz, rotation_xyzw=T_E_R.rotation_xyzw)
    buttons = {}
    for name, button in baseline.buttons.items():
        buttons[name] = replace(button, position_xyz=poses[name].position_xyz)
    return replace(baseline, initial_poses=poses, regions=regions, buttons=buttons)


def service_placement_layouts(workcell: ServiceScene) -> PlacementLayouts:
    """Return one complete cached row for the selected layout's writable physics roots.

    Static fixtures already have their selected build-time poses. Arena's shared
    PlacementLayouts reset restores every rigid component, kinematic fixture, button
    and case together; their joint reset events remain owned by their assets.

    Args:
        workcell: Scene after selecting its named layout.

    Returns:
        Native cached placement data suitable for IsaacLabArenaEnvironment.
    """
    from isaaclab_arena.assets.object_type import ObjectType

    poses = {}
    for name, asset in workcell.assets.items():
        if asset.object_type in (ObjectType.RIGID, ObjectType.ARTICULATION):
            poses[asset.get_scene_key()] = [workcell.layout.initial_poses[name]]
    return PlacementLayouts(poses)


def critical_interaction_poses(layout: ServiceLayout) -> dict[str, Pose]:
    """Return explicit DROID TCP targets for pointwise reachability screening.

    Args:
        layout: Selected layout containing Blender-authored interaction frames.

    Returns:
        Named TCP-to-environment poses for grasps, axial extraction endpoints,
        tester mating, packed placements, button presses, and sampled lid/latch
        angles. Loose-part and lid-bar frames are converted to DROID's +X approach;
        latch metadata already specifies TCP orientation. These discrete
        poses do not certify a continuous path, collision clearance, grasp stability,
        or successful execution. Solver results must be reported separately.
    """
    from .scene import _pose

    targets = {}
    T_G_TCP = Pose(rotation_xyzw=(0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)))

    def held_target(label: str, name: str, T_E_O: Pose) -> None:
        targets[label] = T_E_O.multiply(layout.grasp_poses[name]).multiply(T_G_TCP)
        if name == "obstruction":
            # Exchanging the parallel jaws preserves the opposed tab contacts.
            targets[label] = targets[label].multiply(Pose(rotation_xyzw=(1.0, 0.0, 0.0, 0.0)))

    for name, grasp in layout.grasp_poses.items():
        if name == "obstruction":
            continue  # The cup is removed and staged before accessing its inlet.
        T_E_TCP = layout.initial_poses[name].multiply(grasp).multiply(T_G_TCP)
        targets[f"initial/{name}/grasp"] = T_E_TCP
        targets[f"initial/{name}/approach"] = T_E_TCP.translate((0.0, 0.0, 0.10))

    for socket_name, socket in layout.sockets.items():
        if socket_name == "obstruction":
            continue  # Fault initialization is not a policy insertion requirement.
        T_E_target = layout.initial_poses[socket.parent_name].multiply(socket.pose_in_parent)
        for name in socket.candidate_names:
            if name in layout.grasp_poses:
                held_target(f"socket/{socket_name}/{name}", name, T_E_target)

    for socket_name, name, extraction in (
        ("battery", "battery_original", (-0.075, 0.0, 0.0)),
        ("filter", "filter_original", (0.075, 0.0, 0.0)),
        ("cup", "dust_cup", (0.10, 0.0, 0.0)),
        ("airflow_adapter", "airflow_adapter", (0.085, 0.0, 0.0)),
    ):
        socket = layout.sockets[socket_name]
        T_E_P = layout.initial_poses[socket.parent_name]
        target = T_E_P.multiply(socket.pose_in_parent.translate(extraction))
        held_target(f"extraction/{name}", name, target)

    for name, button in layout.buttons.items():
        root = layout.initial_poses[name]
        # The cap rests at +12 mm; a 4 mm depression is the measured actuation threshold.
        targets[f"button/{name}/press"] = root.multiply(Pose((0.0, 0.0, 0.012 + button.pressed_position_m))).multiply(
            T_G_TCP
        )

    for region_name, subject, rotation in (
        ("battery_service", "battery_original", (-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))),
        ("filter_service", "filter_original", (0.0, 0.0, 0.0, 1.0)),
        # The loose obstruction has no keyed disposal orientation. Turning it
        # half a turn places its grasp tab toward the robot above the aperture.
        ("waste", "obstruction", (0.0, 0.0, 1.0, 0.0)),
    ):
        lower, upper = layout.source_records[region_name]["affordances"]["interior_bounds"]
        record = layout.source_records[subject]
        object_center = tuple(
            (low + high) / 2 for low, high in zip(record["bounds_min"], record["bounds_max"], strict=True)
        )
        rotated_center = Pose(rotation_xyzw=rotation).multiply(Pose(object_center)).position_xyz
        center = (
            (lower[0] + upper[0]) / 2 - rotated_center[0],
            (lower[1] + upper[1]) / 2 - rotated_center[1],
            upper[2] + 0.075,
        )
        root = layout.initial_poses[region_name].multiply(Pose(center, rotation))
        held_target(f"disposal/{region_name}", subject, root)

    lower, upper = layout.source_records["waste"]["affordances"]["interior_bounds"]
    dump_center = ((lower[0] + upper[0]) / 2, (lower[1] + upper[1]) / 2, upper[2] + 0.065)
    cavity = layout.source_records["dust_cup"]["affordances"]["debris_cavity_cylinder"]
    mouth = (cavity["x_range"][0], *cavity["center_yz"])
    yaw = Pose(rotation_xyzw=(0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)))
    for angle in (0.0, -math.pi / 2):
        rotation = yaw.multiply(Pose(rotation_xyzw=(0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2))))
        rotated_mouth = rotation.multiply(Pose(mouth)).position_xyz
        origin = tuple(center - offset for center, offset in zip(dump_center, rotated_mouth, strict=True))
        target = layout.initial_poses["waste"].multiply(Pose(origin, rotation.rotation_xyzw))
        held_target(f"waste/cup_dump/{math.degrees(angle):g}", "dust_cup", target)

    T_E_C = layout.initial_poses["case"]
    case = layout.source_records["case"]["affordances"]
    floor = case["interior_bounds"][0][2]
    cup_bottom = layout.source_records["dust_cup"]["bounds_min"][2]
    staged_position = T_E_C.multiply(Pose((0.135, -0.041, floor - cup_bottom))).position_xyz
    staged_cup = Pose(staged_position, layout.initial_poses["dust_cup"].rotation_xyzw)
    held_target("staging/dust_cup", "dust_cup", staged_cup)
    obstruction = layout.sockets["obstruction"].pose_in_parent
    held_target("staging/obstruction/grasp", "obstruction", staged_cup.multiply(obstruction))
    targets["staging/obstruction/approach"] = targets["staging/obstruction/grasp"].multiply(Pose((-0.03, 0.0, 0.0)))
    held_target(
        "staging/obstruction/extracted", "obstruction", staged_cup.multiply(obstruction.translate((0.04, 0.0, 0.0)))
    )
    for name, local in layout.packing_poses.items():
        subject = "battery_original" if name == "battery" else name
        held_target(f"packing/{name}", subject, T_E_C.multiply(local))

    lid_grasp = _pose(case["lid_handle_grasp"]).multiply(Pose((0.0, 0.0, 0.025))).multiply(T_G_TCP)
    for angle in (-1.8, -0.9, 0.0):
        hinge = Pose(rotation_xyzw=(math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2)))
        T_E_lid = T_E_C.multiply(_pose(case["lid_closed_pose"])).multiply(hinge)
        targets[f"case/lid/{angle:g}"] = T_E_lid.multiply(lid_grasp)
    for angle in (1.4, 0.7, 0.0):
        rotation = Pose(rotation_xyzw=(math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2)))
        T_E_latch = T_E_C.multiply(Pose(tuple(case["latch"]["position_xyz"]))).multiply(rotation)
        targets[f"case/latch/{angle:g}"] = T_E_latch.multiply(_pose(case["latch_grasp"]))
    return targets
