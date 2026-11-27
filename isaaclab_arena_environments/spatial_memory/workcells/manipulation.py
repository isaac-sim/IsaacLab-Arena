# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bench-scale fixtures for manipulation with occluded support and clearance surfaces."""

import math


def cabinet_reach(scene, rng):
    scene.bench()
    scene.box("Back", (0, 0.30, 1.09), (0.92, 0.04, 0.53), "frame")
    scene.box("Shelf", (0, 0.05, 1.36), (0.92, 0.54, 0.035))
    scene.box("ShelfLip", (0, -0.21, 1.32), (0.92, 0.034, 0.09), "limit")
    hinge = scene.group("DoorHinge", (-0.45, -0.19, 1.07), (0, 0, rng.uniform(-65, -38)))
    scene.box("OpenDoor", (0, -0.25, 0), (0.045, 0.50, 0.5), "steel", parent=hinge)
    scene.cylinder("ResetPad", (0.14, 0.267, 1.06), 0.055, 0.032, "target", (90, 0, 0))


def tote_extraction(scene, rng):
    scene.bench()
    scene.box("Rear", (0, 0.32, 1.04), (0.94, 0.035, 0.44), "frame")
    scene.box("Shelf", (0, 0.07, 1.22), (0.94, 0.55, 0.035))
    scene.box("Lip", (0, -0.19, 1.19), (0.94, 0.035, 0.075), "limit")
    scene.box("SideStop", (0.40, rng.uniform(-0.37, -0.27), 0.91), (0.05, 0.13, 0.18), "limit")
    scene.tray("Tote", (0, -0.12, 0.83), (0.53, 0.46, 0.22))
    scene.box("Grip", (0, -0.37, 0.97), (0.22, 0.035, 0.042), "frame")


def hanging_pan(scene, rng):
    scene.bench()
    for i, x in enumerate((-0.36, 0.36)):
        scene.bar(f"Post{i}", (x, 0.20, 0.82), (x, 0.20, 1.46), 0.026)
    scene.bar("Rail", (-0.43, 0.2, 1.46), (0.43, 0.2, 1.46), 0.028)
    for i, x in enumerate((-0.11, rng.uniform(0.12, 0.19))):
        scene.bar(f"HookBack{i}", (x, 0.20, 1.45), (x, 0.16, 1.32), 0.014)
        scene.bar(f"HookSeat{i}", (x, 0.16, 1.32), (x, 0.08, 1.32), 0.014)
        scene.bar(f"HookReturn{i}", (x, 0.08, 1.32), (x, 0.08, 1.38), 0.014, "limit")
        # Three bars form a genuine opening at the top of the handle.
        for j, dx in enumerate((-0.025, 0.025)):
            scene.bar(f"HandleSide{i}{j}", (x + dx, 0.12, 1.30), (x + dx, 0.12, 1.35), 0.008, "work")
        scene.bar(f"HandleTop{i}", (x - 0.025, 0.12, 1.35), (x + 0.025, 0.12, 1.35), 0.008, "work")
        scene.bar(f"Handle{i}", (x, 0.12, 1.10), (x, 0.12, 1.30), 0.022, "work")
        scene.cylinder(f"Pan{i}", (x, 0.12, 1.04), 0.12, 0.045, "work" if i == 0 else "steel", (90, 0, 0))


def valve_turn(scene, rng):
    scene.bench()
    scene.box("Mount", (0, 0.19, 1.04), (0.62, 0.10, 0.43), "frame")
    scene.cylinder("Spindle", (0, 0.10, 1.08), 0.055, 0.12, "target", (90, 0, 0))
    scene.bar("Lever", (0, 0.038, 1.08), (0.23, 0.038, 1.13), 0.025, "work")
    scene.cylinder("Knob", (0.23, 0.005, 1.13), 0.037, 0.08, "work", (90, 0, 0))
    x = rng.uniform(-0.19, -0.14)
    scene.bar("Pipe", (x, -0.06, 0.82), (x, -0.06, 1.36), 0.042)
    scene.box("Guard", (0, 0.08, 1.31), (0.60, 0.32, 0.04), "limit")
    scene.box("Stop", (0.03, 0.075, 1.26), (0.045, 0.11, 0.055), "target")


def seam_inspection(scene, rng):
    scene.bench()
    scene.box("Housing", (0, 0.02, 1.01), (0.56, 0.38, 0.38), "frame")
    bend = rng.uniform(1.01, 1.09)
    points = [
        (-0.28, -0.182, 0.94),
        (-0.07, -0.182, 0.94),
        (0.05, -0.182, bend),
        (0.285, -0.182, bend),
        (0.285, 0.17, bend),
    ]
    for i, (a, b) in enumerate(zip(points, points[1:])):
        scene.bar(f"Weld{i}", a, b, 0.012, "target")
    scene.bar("ProbeTip", (-0.16, -0.26, 0.99), (-0.10, -0.19, 0.95), 0.009)
    scene.bar("ProbeGrip", (-0.28, -0.40, 1.07), (-0.16, -0.26, 0.99), 0.028, "work")


def panel_catches(scene, rng):
    scene.bench()
    scene.box("Panel", (0, 0.12, 1.06), (0.82, 0.10, 0.48), "frame")
    for i, x in enumerate((-0.25, 0.25)):
        depth = rng.uniform(0.12, 0.21)
        scene.box(f"Catch{i}", (x, 0.051, 1.06), (0.10, 0.035, 0.07), "target")
        scene.box(f"Hood{i}", (x, 0.06 - depth / 2, 1.16), (0.21, depth, 0.035))
        scene.box(f"Cheek{i}", (x + (0.09 if i else -0.09), 0.06 - depth / 2, 1.085), (0.03, depth, 0.16), "limit")


def dishwasher(scene, rng):
    scene.bench()
    for i, x in enumerate((-0.27, 0.27)):
        scene.bar(f"RackSide{i}", (x, -0.30, 0.87), (x, 0.30, 0.87), 0.017)
    for i, y in enumerate((-0.26, -0.13, 0, 0.13, 0.26)):
        scene.bar(f"RackCross{i}", (-0.27, y, 0.87), (0.27, y, 0.87), 0.012)
        for j, x in enumerate((-0.12, 0.12)):
            scene.bar(f"Tine{i}{j}", (x, y, 0.87), (x, y + 0.045, 1.04), 0.011, "target")
    scene.cylinder("Neighbor", (0, 0.13, 1.09), 0.20, 0.025, "steel", (rng.uniform(71, 83), 0, 0))
    scene.cylinder("HeldPlate", (0, -0.095, 1.20), 0.21, 0.025, "work", (87, 0, -5))


def billet_insertion(scene, rng):
    scene.bench()
    scene.box("MachineBed", (0, 0.02, 0.89), (0.78, 0.66, 0.14), "frame")
    for i, x in enumerate((-0.26, 0.26)):
        scene.box(f"Portal{i}", (x, -0.16, 1.12), (0.09, 0.12, 0.36))
    scene.box("Top", (0, -0.16, 1.31), (0.61, 0.12, 0.06))
    scene.box("Sill", (0, -0.16, 1.00), (0.50, 0.13, 0.08), "limit")
    scene.box("OffsetJaw", (0.13, rng.uniform(0.04, 0.15), 1.12), (0.12, 0.11, 0.20), "limit")
    scene.box("AxialStop", (0, 0.28, 1.09), (0.30, 0.05, 0.21), "target")
    scene.box("Billet", (-0.05, -0.38, 1.12), (0.17, 0.42, 0.11), "work", (-6, 0, 0))


def rail_support(scene, rng):
    scene.bench()
    for i, (x, d) in enumerate(((-0.23, 0.58), (0.23, rng.uniform(0.33, 0.43)))):
        scene.box(f"Rail{i}", (x, 0.03, 0.89), (0.073, d, 0.14), "target")
    scene.box("Crate", (0, -0.06, 1.13), (0.58, 0.44, 0.27), "work", (0, -5, 0))


def stack_bracing(scene, rng):
    scene.bench()
    scene.box("RearFoot", (-0.19, rng.uniform(0.04, 0.12), 0.92), (0.18, 0.24, 0.2), "target")
    scene.box("Spacer", (0.16, -0.03, 0.92), (0.25, 0.40, 0.20), "work")
    scene.box("UpperCrate", (0, 0, 1.17), (0.64, 0.41, 0.30), "wood")
    scene.box("PullTab", (0.16, -0.25, 0.92), (0.11, 0.055, 0.063), "frame")


def worktop_clamp(scene, rng):
    scene.bench()
    scene.box("Worktop", (0, -0.20, 0.86), (0.72, 0.68, 0.08), "wood")
    scene.box("BeveledUnderside", (0.20, -0.46, 0.80), (0.20, 0.14, 0.052), "limit", (rng.uniform(-24, -15), 0, 0))
    scene.box("Spine", (0, -0.60, 0.82), (0.055, 0.05, 0.29), "work")
    for i, z in enumerate((0.69, 0.96)):
        scene.box(f"ClampArm{i}", (0, -0.52, z), (0.055, 0.20, 0.035), "work")
    scene.cylinder("Screw", (0, -0.45, 0.74), 0.016, 0.12)
    scene.cylinder("LowerJaw", (0, -0.45, 0.805), 0.042, 0.02, "target")
    scene.cylinder("UpperJaw", (0, -0.45, 0.92), 0.038, 0.04, "target")


def loaded_drawer(scene, rng):
    scene.bench()
    scene.box("Back", (0, 0.32, 1.0), (0.85, 0.05, 0.36), "frame")
    for i, x in enumerate((-0.40, 0.40)):
        scene.box(f"CabinetSide{i}", (x, 0.05, 1.0), (0.05, 0.59, 0.36))
    scene.box("Crossbar", (rng.uniform(0.06, 0.15), -0.12, 1.13), (0.47, 0.07, 0.10), "limit")
    scene.tray("Drawer", (0, -0.24, 0.84), (0.72, 0.52, 0.14), "wood")
    scene.tray("InnerTray", (0.04, -0.23, 0.87), (0.46, 0.29, 0.05), "target")
    scene.bar("ToolHandle", (-0.10, -0.26, 0.93), (0.24, -0.14, 1.15), 0.023, "work")


def pipe_bracket(scene, rng):
    scene.bench()
    scene.box("Wall", (0, 0.20, 1.10), (0.80, 0.05, 0.55), "frame")
    y = rng.uniform(0.025, 0.07)
    for i, x in enumerate((-0.27, 0.27)):
        scene.bar(f"Standoff{i}", (x, 0.185, 1.08), (x, y, 1.08), 0.027)
    scene.bar("Pipe", (-0.38, y, 1.08), (0.38, y, 1.08), 0.064, "target")
    u = scene.group("Bracket", (0.02, y - 0.13, 1.12), (-18, 0, 0))
    scene.box("Bridge", (0, 0, 0.08), (0.09, 0.23, 0.025), "work", parent=u)
    for i, sign in enumerate((-1, 1)):
        scene.box(f"Leg{i}", (0, sign * 0.10, 0), (0.09, 0.025, 0.16), "work", parent=u)


def panel_removal(scene, rng):
    scene.bench()
    for i, x in enumerate((-0.43, 0.43)):
        scene.box(f"FrameSide{i}", (x, 0.03, 0.98), (0.055, 0.53, 0.32), "frame")
    scene.box("Back", (0, 0.27, 0.99), (0.88, 0.05, 0.34), "frame")
    scene.box("FrontLip", (0, -0.22, 0.89), (0.88, 0.06, 0.14), "limit")
    scene.box("Overhang", (0.18, 0.18, rng.uniform(1.17, 1.25)), (0.55, 0.26, 0.04), "limit")
    scene.box("Worktop", (0, -0.02, 0.99), (0.78, 0.42, 0.045), "work", (-11, 0, 0))


def hatch_transfer(scene, rng):
    scene.bench(1.6, 0.83)
    for i, y in enumerate((-0.30, 0.30)):
        scene.box(f"Post{i}", (0.16, y, 0.99), (0.09, 0.07, 0.34), "frame")
    scene.box("HatchRoof", (0.16, 0, 1.18), (0.38, 0.68, 0.05))
    for i, (x, y) in enumerate(((-0.08, 0.17), (0.39, rng.uniform(-0.23, -0.16)))):
        scene.box(f"Guide{i}", (x, y, 0.87), (0.10, 0.11, 0.10), "limit")
    tray = scene.tray("ServingTray", (-0.39, -0.005, 0.83), (0.55, 0.29, 0.044))
    for i, x in enumerate((-0.14, 0.07)):
        scene.cylinder(f"Cup{i}", (x, 0, 0.085), 0.04, 0.15, "target", parent=tray)


def probe_withdrawal(scene, rng):
    scene.bench()
    scene.box("HousingBack", (0, 0.25, 1.05), (0.70, 0.065, 0.46), "frame")
    scene.box("LeftBaffle", (-0.16, -0.03, 1.11), (0.28, 0.06, 0.25), "limit")
    scene.box("RightBaffle", (0.21, rng.uniform(0.04, 0.11), 1.04), (0.19, 0.06, 0.32), "limit")
    scene.bar("ProbeShaft", (0.04, -0.29, 1.07), (0.04, 0.15, 1.07), 0.014)
    scene.bar("BentTip", (0.04, 0.15, 1.07), (-0.056, 0.18, 1.13), 0.014, "target")
    scene.bar("ProbeHandle", (0.04, -0.43, 1.07), (0.04, -0.29, 1.07), 0.036, "work")


def display_cover(scene, rng):
    scene.bench()
    scene.box("DisplayBase", (0, 0, 0.87), (0.62, 0.43, 0.10), "frame")
    scene.box("Glass", (0, 0, 0.927), (0.57, 0.38, 0.014), "target")
    # A folded strip is a representative peel configuration, with its free edge raised.
    scene.box("AdheredCover", (0, 0.09, 0.939), (0.57, 0.17, 0.008), "work")
    tilt = rng.uniform(30, 46)
    a = math.radians(tilt)
    scene.box(
        "FreeCover",
        (0, 0.005 - 0.10 * math.cos(a), 0.939 + 0.10 * math.sin(a)),
        (0.57, 0.20, 0.008),
        "work",
        (-tilt, 0, 0),
    )
    scene.box("HoodBack", (0, 0.25, 1.04), (0.70, 0.04, 0.44), "frame")
    scene.box("OverheadLip", (0, 0.10, 1.25), (0.70, 0.34, 0.035), "limit")


def scoop_pour(scene, rng):
    scene.bench()
    # Four sloping plates form an open hopper around an unobstructed throat.
    for i, s in enumerate((-1, 1)):
        scene.box(f"HopperSide{i}", (s * 0.16, 0, 0.997), (0.032, 0.33, 0.36), "target", (0, s * 25, 0))
        scene.box(f"HopperEnd{i}", (0, s * 0.14, 0.997), (0.35, 0.032, 0.36), "target", (-s * 23, 0, 0))
    scene.box("HoodPost", (0.34, 0.24, 1.18), (0.05, 0.05, 0.72), "frame")
    scene.box("Hood", (0.17, 0.15, rng.uniform(1.51, 1.59)), (0.39, 0.36, 0.04), "limit")
    scoop = scene.tray("Scoop", (-0.12, -0.27, 1.29), (0.19, 0.23, 0.075), "work", (-12, 0, 12))
    scene.bar("Handle", (0, -0.10, 0.035), (0, -0.38, 0.10), 0.021, "frame", scoop)
    for i in range(6):
        scene.cylinder(
            f"Pellet{i}", ((i % 3 - 1) * 0.045, (i // 3 - 0.5) * 0.065, 0.032), 0.018, 0.046, "wood", parent=scoop
        )
