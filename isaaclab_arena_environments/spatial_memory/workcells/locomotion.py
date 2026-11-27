# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Local transport and moving fixtures for short whole-body skills."""


def panel_corner(scene, rng):
    scene.box("InnerWall", (-0.48, 0.40, 0.70), (0.14, 1.26, 1.4), "steel")
    scene.box("WallReturn", (-0.88, -0.16, 0.70), (0.83, 0.14, 1.4), "steel")
    scene.box("OuterWall", (0.75, 0.50, 0.70), (0.14, 1.10, 1.4), "frame")
    scene.box("BackWall", (0.13, 1.03, 0.70), (1.38, 0.12, 1.4), "frame")
    scene.box("HeldWorktop", (0.04, -0.43, 0.84), (rng.uniform(1.22, 1.48), 0.37, 0.09), "work", (0, 0, 13))


def aisle_retreat(scene, rng):
    for i, x in enumerate((-0.65, 0.65)):
        scene.box(f"Rack{i}", (x, 0, 0.70), (0.31, 1.62, 1.40), "frame")
        for j, z in enumerate((0.28, 0.55, 0.82, 1.09)):
            scene.box(f"DriveBay{i}{j}", (x, -0.825, z), (0.26, 0.035, 0.20), "steel")
        scene.box(
            f"Projection{i}",
            (x + (0.18 if i == 0 else -0.18), (-0.34 if i == 0 else 0.25), 0.94),
            (0.18, 0.28, 0.18),
            "limit",
        )
    scene.box("Chassis", (0, -0.12, 0.92), (rng.uniform(0.47, 0.58), 0.90, 0.17), "work")
    for i, x in enumerate((-0.15, 0.15)):
        scene.box(f"Grip{i}", (x, -0.59, 0.92), (0.08, 0.04, 0.09), "frame")


def platform_step(scene, rng):
    h = rng.uniform(0.16, 0.23)
    scene.box("Platform", (0, 0.30, h / 2), (0.94, 0.78, h), "steel")
    scene.box("Nosing", (0, -0.077, h - 0.005), (0.94, 0.027, 0.013), "limit")
    for i, y in enumerate((0.02, 0.61)):
        scene.bar(f"Post{i}", (0.43, y, h), (0.43, y, h + 0.80), 0.024)
    scene.bar("Handrail", (0.43, 0.02, h + 0.8), (0.43, 0.61, h + 0.8), 0.026, "target")
    scene.box("HeldCrate", (0, -0.49, 0.88), (0.55, 0.41, 0.31), "work")


def beam_clearance(scene, rng):
    h = rng.uniform(1.48, 1.62)
    for i, x in enumerate((-0.65, 0.65)):
        scene.box(f"BeamPost{i}", (x, 0, h / 2), (0.09, 0.19, h), "frame")
    scene.box("Crossbeam", (0, 0, h), (1.40, 0.30, 0.14), "limit")
    scene.box("DuctStep", (0.31, 0.10, h - 0.105), (0.46, 0.34, 0.08), "steel")
    scene.box("TallItem", (-0.09, -0.53, 1.03), (0.34, 0.30, 0.78), "work")


def trolley_dock(scene, rng):
    berth = scene.group("Berth", (0, 0.22, 0), (0, 0, rng.uniform(-13, 13)))
    for i, x in enumerate((-0.37, 0.37)):
        scene.box(f"Kerb{i}", (x, 0, 0.09), (0.075, 0.83, 0.18), "steel", parent=berth)
        scene.box(f"Stop{i}", (x * 0.76, 0.39, 0.13), (0.14, 0.10, 0.26), "target", parent=berth)
    scene.cart("Trolley", (0.065, -0.58, 0))


def pillar_pass(scene, rng):
    scene.box("Plinth", (0, 0, 0.13), (0.65, rng.uniform(0.56, 0.72), 0.26), "limit")
    scene.box("Pillar", (0, 0.04, 0.93), (0.34, 0.32, 1.6), "steel")
    scene.box("Shoulder", (0.16, 0.04, 1.20), (0.23, 0.43, 0.30), "frame")
    scene.box("InstrumentCase", (0.40, -0.46, 0.85), (0.87, 0.26, 0.37), "work")
    scene.bar("CaseGrip", (0.23, -0.46, 1.08), (0.55, -0.46, 1.08), 0.024, "frame")
    for i, x in enumerate((0.23, 0.55)):
        scene.bar(f"GripFoot{i}", (x, -0.46, 1.03), (x, -0.46, 1.08), 0.019, "frame")


def closing_door(scene, rng):
    for i, x in enumerate((-0.54, 0.54)):
        scene.box(f"Jamb{i}", (x, 0, 0.96), (0.10, 0.20, 1.92), "frame")
    scene.box("Lintel", (0, 0, 1.95), (1.18, 0.20, 0.09))
    scene.box("Return", (-0.72, 0.20, 0.55), (0.31, 0.55, 1.10), "limit")
    angle = rng.uniform(65, 82)
    hinge = scene.group("Door", (-0.49, 0, 0), (0, 0, angle))
    scene.box("Leaf", (0.48, 0, 0.93), (0.96, 0.058, 1.86), "target", parent=hinge)
    scene.bar("Handle", (0.75, -0.065, 1.00), (0.88, -0.065, 1.00), 0.02, "steel", hinge)
    scene.animate(hinge, "rotateXYZ", [(0, (0, 0, angle)), (72, (0, 0, 12))])
    scene.box("Parcel", (0.10, -0.53, 0.91), (0.64, 0.45, 0.32), "work")


def moving_carriage(scene, rng):
    scene.bench(1.55, 0.90)
    for i, y in enumerate((-0.12, 0.13)):
        scene.box(f"Rail{i}", (0, y, 0.855), (1.38, 0.047, 0.07), "steel")
    carriage = scene.group("Carriage", (-0.48, 0, 0.92))
    scene.box("Deck", (0, 0, 0), (0.36, 0.38, 0.07), "work", parent=carriage)
    scene.box("Part", (0, 0, 0.105), (0.20, 0.20, 0.14), "target", parent=carriage)
    for i, x in enumerate((-0.065, 0.065)):
        scene.bar(f"HandlePost{i}", (x, 0, 0.17), (x, 0, 0.24), 0.014, parent=carriage)
    scene.bar("Handle", (-0.065, 0, 0.24), (0.065, 0, 0.24), 0.015, parent=carriage)
    scene.box("WristGuard", (0.15, rng.uniform(0.21, 0.29), 1.02), (0.63, 0.06, 0.40), "limit")
    scene.animate(carriage, "translate", [(0, (-0.48, 0, 0.92)), (64, (0.48, 0, 0.92)), (128, (-0.48, 0, 0.92))])


def descending_hood(scene, rng):
    scene.bench()
    for i, x in enumerate((-0.36, 0.36)):
        scene.box(f"Upright{i}", (x, 0.20, 1.15), (0.05, 0.08, 0.66), "frame")
        scene.box(f"GuideStop{i}", (x * 0.78, (-0.14 if i else 0.12), 0.89), (0.09, 0.15, 0.14), "limit")
    scene.tray("Tray", (0, -0.19, 0.835), (0.46, 0.48, 0.075))
    hood = scene.group("Hood", (0, 0, 1.40))
    scene.box("Top", (0, 0, 0), (0.70, 0.49, 0.10), "steel", parent=hood)
    scene.box("Skirt", (0, -0.22, -0.067), (0.70, 0.045, 0.09), "target", parent=hood)
    low = rng.uniform(1.00, 1.08)
    scene.animate(hood, "translate", [(0, (0, 0, 1.40)), (48, (0, 0, low)), (80, (0, 0, low)), (128, (0, 0, 1.40))])
