# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Original, dimensioned manufacturing geometry for the vacuum service cell."""

from __future__ import annotations

import bpy
import math

from .case_geometry import build_case
from .geometry import Geometry

AXIS_X = (0, math.pi / 2, 0)


def fasteners(g: Geometry, positions):
    """Add recessed screw heads with machined cross slots."""
    for i, position in enumerate(positions):
        x, y, z = position
        g.cylinder(f"screw_{i}", position, 0.0026, 0.0012, "steel")
        g.box(f"slot_{i}", (x, y, z + 0.0007), (0.0035, 0.0005, 0.0002), "rubber", 0)


def grasp_tab(g: Geometry, position, width=0.032):
    """Add opposed textured pinch faces with a maximum 32 mm jaw span."""
    x, y, z = position
    g.box("grasp_tab", position, (width, 0.022, 0.016), "teal", 0.003, collision=True)
    for side in (-1, 1):
        for i in range(5):
            g.box(
                "grip_rib",
                (x - width * 0.32 + i * width * 0.16, y + side * 0.0113, z),
                (0.0015, 0.001, 0.011),
                "rubber",
                0.0003,
            )


def build_vacuum(g: Geometry):
    """Build the motor body, removable airflow components and physical releases."""
    g.asset(
        "vacuum_body",
        0.64,
        grasp={"position_xyz": [0.020, 0, 0.152], "jaw_span": 0.030},
        battery_socket=[-0.111, 0, 0.010],
        cup_socket=[0.123, 0, 0.025],
        filter_socket=[0.103, 0, 0.036],
        filter_capture_bounds=[[0.0705, -0.0099, 0.03545], [0.1395, 0.0099, 0.0392]],
        cup_latch={"position_xyz": [0.066, 0, 0.112], "axis": [1, 0, 0], "travel": 0.012},
        power_button={"position_xyz": [-0.030, 0, 0.113]},
    )
    g.cylinder("motor_shell", (0, 0, 0.064), 0.043, 0.142, "navy", AXIS_X, collision=True)
    g.cylinder("rear_end_cap", (-0.070, 0, 0.064), 0.0415, 0.009, "teal", AXIS_X)
    g.box("battery_support_shelf", (-0.106, 0, 0.007), (0.052, 0.048, 0.006), "steel", 0.001, collision=True)
    g.box("battery_terminal_stop", (-0.0765, 0, 0.029), (0.003, 0.048, 0.034), "steel", 0.0005, collision=True)
    # The shoulder's front plane meets the cup's rear lip at its nominal socket.
    # A radial overlap provides seating contact without pinching the cup bore.
    g.ring("service_socket", (0.0745, 0, 0.062), 0.043, 0.030, 0.007, "steel", AXIS_X, collision=True)
    # The body supports the filter when the cup is removed. The open upper side
    # gives the gripper access to its tab without trapping the cup on removal.
    g.box("filter_lower_guide", (0.105, 0, 0.0335), (0.070, 0.022, 0.004), "steel", 0.0005, collision=True)
    for side in (-1, 1):
        # A 20 mm channel guides the 18 mm skid without blocking the ring caps.
        g.box("filter_skid_kerb", (0.105, side * 0.0105, 0.036), (0.070, 0.001, 0.001), "steel", 0.0001, collision=True)
        # Short rails clear the surrounding cup bore at their upper corners.
        g.box("filter_side_guide", (0.105, side * 0.0305, 0.062), (0.070, 0.003, 0.012), "teal", 0.0005, collision=True)
    for side in (-1, 1):
        g.box("skid", (0, side * 0.026, 0.009), (0.12, 0.016, 0.018), "rubber", 0.005, collision=True)
        g.box("handle_pillar", (side * 0.046, 0, 0.110), (0.018, 0.026, 0.058), "navy", 0.006, collision=True)
        g.box("battery_guide", (-0.085, side * 0.0345, 0.026), (0.044, 0.008, 0.011), "steel", 0.001, collision=True)
    g.box("handle", (0, 0, 0.140), (0.105, 0.030, 0.019), "rubber", 0.007, collision=True)
    for i in range(7):
        g.box("handle_rib", (-0.03 + i * 0.010, 0, 0.150), (0.0015, 0.025, 0.001), "navy", 0.0003)
    for side in (-1, 1):
        for i in range(8):
            g.box("exhaust_vent", (-0.042 + i * 0.010, side * 0.040, 0.075), (0.005, 0.004, 0.018), "rubber", 0.002)
    g.box("nameplate", (0.013, -0.025, 0.098), (0.055, 0.018, 0.002), "teal", 0.002)
    g.label("SERVICE / A", (0.013, -0.025, 0.0995), 0.006)
    g.box("cup_release_rail", (0.063, 0, 0.108), (0.036, 0.028, 0.006), "steel", 0.001)

    g.asset("cup_latch", 0.015, axis=[1, 0, 0], travel=0.012, released_direction=-1)
    g.box("thumb_slider", (0, 0, 0.007), (0.029, 0.022, 0.014), "orange", 0.003, collision=True)
    g.box("retaining_tongue", (0.018, 0, -0.006), (0.026, 0.012, 0.009), "steel", 0.001, collision=True)
    for i in range(3):
        g.box("thumb_rib", (-0.008 + i * 0.008, 0, 0.0145), (0.002, 0.016, 0.0015), "rubber", 0.0004)

    cup_bore_radius = 0.037
    cup_wall_radius = cup_bore_radius + 0.004
    cup_lip_radius = cup_bore_radius + 0.006
    cup_center_z = 0.037
    airflow_port_inner_radius = 0.012
    g.asset(
        "dust_cup",
        0.13,
        grasp={"position_xyz": [-0.002, 0, 0.111], "jaw_span": 0.024},
        extraction_axis=[1, 0, 0],
        hollow_bore_radius=cup_bore_radius,
        filter_socket=[-0.020, 0, 0.011],
        obstruction_socket=[0.056, 0, 0.026],
        airflow_adapter_socket=[0.068, 0, 0.037],
        airflow_port_inner_radius=airflow_port_inner_radius,
        inlet_bounds=[[0.042, -0.012, 0.025], [0.068, 0.012, 0.049]],
        debris_cavity_bounds=[
            [-0.045, -cup_bore_radius, cup_center_z - cup_bore_radius],
            [0.042, cup_bore_radius, cup_center_z + cup_bore_radius],
        ],
        debris_cavity_cylinder={"x_range": [-0.045, 0.042], "center_yz": [0, cup_center_z], "radius": cup_bore_radius},
        debris_poses=[[0.022, -0.012, 0.018], [0.032, 0.012, 0.022], [0.032, 0, 0.040]],
    )
    g.ring(
        "collection_wall",
        (0, 0, cup_center_z),
        cup_wall_radius,
        cup_bore_radius,
        0.090,
        "ivory",
        AXIS_X,
        collision=True,
    )
    g.ring(
        "rear_sealing_lip",
        (-0.042, 0, cup_center_z),
        cup_lip_radius,
        cup_bore_radius,
        0.006,
        "teal",
        AXIS_X,
        collision=True,
    )
    g.ring(
        "front_bulkhead",
        (0.044, 0, cup_center_z),
        cup_wall_radius,
        airflow_port_inner_radius,
        0.004,
        "ivory",
        AXIS_X,
        collision=True,
    )
    g.ring("inlet", (0.055, 0, 0.037), 0.015, airflow_port_inner_radius, 0.026, "teal", AXIS_X, collision=True)
    g.box("tab_standoff", (-0.002, 0, 0.079), (0.023, 0.019, 0.012), "ivory", 0.002, collision=True)
    grasp_tab(g, (-0.002, 0, 0.091), 0.039)
    g.label("EMPTY", (-0.002, 0, 0.100), 0.007, "ivory")
    g.box("level_strip", (0, -cup_wall_radius + 0.0007, 0.043), (0.050, 0.002, 0.007), "teal", 0.0007)
    for i in range(5):
        g.box("level_mark", (-0.020 + i * 0.010, -cup_wall_radius - 0.0005, 0.043), (0.001, 0.001, 0.005), "ivory", 0)
    # A flat underside supports axial sliding without the cylinder rolling.
    # Keep its bottom level with the rear lip so the lip cannot catch the rail.
    g.box("belly_skid", (0.002, 0, -0.00375), (0.070, 0.018, 0.0045), "teal", 0.0005, collision=True)

    for model in ("A", "B"):
        name = "filter" if model == "A" else "filter_decoy"
        bottom = 0.0 if model == "A" else -0.004
        g.asset(
            name,
            0.036,
            model=model,
            grasp={"position_xyz": [0, 0, 0.080], "jaw_span": 0.022},
            extraction_axis=[1, 0, 0],
            support_footprint_bounds=[[-0.027, -0.009, bottom], [0.027, 0.009, bottom + 0.003]],
        )
        outer = 0.025 if model == "A" else 0.029
        vertices, faces = [], []
        for x in (-0.023, 0.023):
            for i in range(96):
                angle = math.tau * i / 96
                radius = outer if i % 2 == 0 else outer - 0.005
                vertices.append((x, radius * math.cos(angle), 0.026 + radius * math.sin(angle)))
        for i in range(96):
            j = (i + 1) % 96
            faces.append((i, j, 96 + j, 96 + i))
        mesh = bpy.data.meshes.new(f"pleated_filter_{model}")
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        obj = bpy.data.objects.new(f"pleated_filter_{model}", mesh)
        g.scene.collection.objects.link(obj)
        g.finish(obj, "filter_pleats", "filter_paper")
        for x in (-0.025, 0.025):
            g.ring(
                "end_cap",
                (x, 0, 0.026),
                outer + 0.001,
                0.014,
                0.005,
                "teal" if model == "A" else "orange",
                AXIS_X,
                collision=True,
            )
        g.ring("filter_support", (0, 0, 0.026), 0.020, 0.014, 0.046, "filter_paper", AXIS_X, collision=True)
        g.box(
            "anti_roll_skid",
            (0, 0, bottom + 0.0015),
            (0.054, 0.018, 0.003),
            "teal" if model == "A" else "orange",
            0.0005,
            collision=True,
        )
        grasp_tab(g, (0, 0, 0.049), 0.027)
        g.label(f"{model}", (0, 0, 0.058), 0.009)

    for model in ("A", "B"):
        name = "battery" if model == "A" else "battery_decoy"
        g.asset(
            name,
            0.18,
            model=model,
            grasp={"position_xyz": [-0.006, 0, 0.076], "jaw_span": 0.024},
            extraction_axis=[-1, 0, 0],
        )
        g.box("housing", (0, 0, 0.018), (0.060, 0.055, 0.036), "navy", 0.005, collision=True)
        g.box(
            "top_molding",
            (0, 0, 0.037),
            (0.057, 0.052, 0.006),
            "teal" if model == "A" else "orange",
            0.002,
            collision=True,
        )
        grasp_tab(g, (-0.006, 0, 0.052), 0.034)
        g.label(f"18V / {model}", (0.002, -0.016, 0.041), 0.006)
        for side in (-1, 1):
            g.box(
                "keyed_rail",
                (0.023, side * (0.019 if model == "A" else 0.015), 0.019),
                (0.020, 0.005, 0.008),
                "rubber",
                0.001,
                collision=True,
            )
        for i in range(3):
            g.box("terminal", (0.031, -0.012 + 0.012 * i, 0.029), (0.002, 0.006, 0.009), "steel", 0.0004)
        fasteners(g, [(-0.023, -0.020, 0.040), (0.023, 0.020, 0.040)])


def build_tools(g: Geometry):
    """Build hose-free accessories and rigid debris with graspable removal tabs."""
    for name, brush in (("nozzle", False), ("crevice_tool", False), ("brush_tool", True)):
        # The closing Robotiq fingers extend below their TCP; leave the tray rim clear.
        if brush:
            # The 22 mm finger width fits between the connector and wider brush head.
            grasp = {"position_xyz": [-0.006, 0, 0.044], "jaw_span": 0.028}
        else:
            grasp = {"position_xyz": [0, 0, 0.044], "jaw_span": 0.028}
        g.asset(name, 0.035, grasp=grasp)
        g.ring("connector", (-0.035, 0, 0.017), 0.017, 0.012, 0.034, "teal", AXIS_X, collision=True)
        g.box("neck", (0, 0, 0.017), (0.061, 0.028, 0.025), "navy", 0.004, collision=True)
        if brush:
            g.box("brush_head", (0.030, 0, 0.012), (0.043, 0.060, 0.018), "navy", 0.004, collision=True)
            for i in range(8):
                for side in (-1, 1):
                    g.cylinder(
                        "bristle_tuft",
                        (0.016 + i * 0.005, side * 0.021, 0.007),
                        0.0018,
                        0.014,
                        "rubber",
                        collision=True,
                    )
        else:
            # Keep the neck level with the connector on any flat support surface.
            g.box("neck_rest", (0.022, 0, 0.00225), (0.012, 0.016, 0.0045), "rubber", 0.001, collision=True)
            g.box("crevice_tip", (0.042, 0, 0.017), (0.050, 0.016, 0.017), "navy", 0.002, collision=True)
            g.box("intake", (0.0665, 0, 0.017), (0.001, 0.010, 0.009), "rubber", 0.001)
        g.label("BRUSH" if brush else "CREVICE", (0, 0, 0.031), 0.006)

    g.asset(
        "obstruction",
        0.009,
        grasp={
            "position_xyz": [0.054, 0, 0.039],
            "rotation_xyzw": [0.382683432, 0, 0.923879533, 0],
            "jaw_span": 0.020,
        },
    )
    # Extend the supported shank so the external tab cannot tip the plug out.
    # The tab begins 2 mm beyond the inlet; its corners must not enter the bore.
    g.cylinder("plug", (0.001, 0, 0.011), 0.0105, 0.030, "debris", AXIS_X, collision=True)
    g.box("pull_tab", (0.025, 0, 0.015), (0.022, 0.020, 0.008), "orange", 0.002, collision=True)
    g.label("PULL", (0.026, 0, 0.0195), 0.004, "navy")
    g.asset("debris", 0.003)
    g.box("crumb", (0, 0, 0.005), (0.012, 0.008, 0.009), "debris", 0.002, collision=True)
    for i in range(3):
        g.box("crumb_fleck", (-0.003 + i * 0.003, 0, 0.009), (0.002, 0.004, 0.001), "filter_paper", 0.0005)

    straight_stem_x_range = (-0.010, 0.020)
    tube_outer_radius = 0.0105
    g.asset(
        "airflow_adapter",
        0.055,
        grasp={"position_xyz": [0.042, -0.0325, 0.049], "jaw_span": 0.022},
        extraction_axis=[1, 0, 0],
        socket_separation=0.065,
        insertion_depth=0.010,
        straight_stem_x_range=list(straight_stem_x_range),
        tube_outer_radius=tube_outer_radius,
        cup_connector=[0, 0, 0],
        tester_connector=[0, -0.065, 0],
        parking_pose=[0.18, -0.12, 0.792],
    )
    stem_tip_x, stem_root_x = straight_stem_x_range
    centers = [(stem_tip_x, 0, 0), (stem_root_x, 0, 0)]
    for i in range(1, 24):
        angle = math.pi * i / 24
        centers.append((stem_root_x + 0.0325 * math.sin(angle), -0.0325 + 0.0325 * math.cos(angle), 0))
    centers.extend(((stem_root_x, -0.065, 0), (stem_tip_x, -0.065, 0)))
    g.swept_tube("airflow_bridge", centers, tube_outer_radius, 0.007, "teal")
    for y in (0, -0.065):
        g.ring("sealing_collar", (0.002, y, 0), 0.0125, 0.007, 0.004, "rubber", AXIS_X)
    g.box("tab_support", (0.042, -0.0325, 0.013), (0.021, 0.018, 0.018), "teal", 0.003, collision=True)
    grasp_tab(g, (0.042, -0.0325, 0.025), 0.027)
    g.label("AIR", (0.042, -0.0325, 0.034), 0.007)


def tray(g: Geometry, name, dimensions, color, text):
    """Create an open tray with physically separate floor and wall proxies."""
    x, y, z = dimensions
    floor_thickness = 0.006
    g.asset(
        name,
        0.16,
        interior_bounds=[[-x / 2 + 0.006, -y / 2 + 0.006, floor_thickness], [x / 2 - 0.006, y / 2 - 0.006, z]],
    )
    g.box("floor", (0, 0, floor_thickness / 2), (x, y, floor_thickness), color, 0.003, collision=True)
    for side in (-1, 1):
        g.box("wall_x", (side * (x / 2 - 0.003), 0, z / 2), (0.006, y, z), color, 0.002, collision=True)
        g.box("wall_y", (0, side * (y / 2 - 0.003), z / 2), (x - 0.012, 0.006, z), color, 0.002, collision=True)
    g.box("label_plate", (0, -y / 2 + 0.014, z + 0.001), (x * 0.83, 0.022, 0.003), "navy", 0.002)
    g.label(text, (0, -y / 2 + 0.014, z + 0.003), min(0.008, x / 18))


def build_fixtures(g: Geometry):
    """Build clamped service and diagnostic stations, bins and a spare-parts rack."""
    g.asset("cradle", 0.60, body_socket=[0, 0, 0.015])
    g.box("base", (0, 0, 0.007), (0.215, 0.145, 0.014), "steel", 0.006, collision=True)
    for x in (-0.050, 0.046):
        for side in (-1, 1):
            g.box("saddle", (x, side * 0.052, 0.036), (0.027, 0.019, 0.047), "teal", 0.005, collision=True)
            g.box("grip_liner", (x, side * 0.0428, 0.042), (0.023, 0.003, 0.031), "rubber", 0.001, collision=True)
    # Support the unlatched cup without constraints or retention forces. The
    # rail ends 8 mm before the cup at its 100 mm extraction endpoint.
    g.box("cup_rail_post", (0.094, 0, 0.0205), (0.030, 0.024, 0.013), "steel", 0.001, collision=True)
    g.box("cup_slide_rail", (0.124, 0, 0.0305), (0.092, 0.020, 0.007), "teal", 0.001, collision=True)
    g.label("SERVICE CRADLE", (0, -0.060, 0.0145), 0.009, "navy")
    fasteners(g, [(-0.090, -0.058, 0.015), (0.090, -0.058, 0.015), (-0.090, 0.058, 0.015), (0.090, 0.058, 0.015)])

    for name, label in (("battery_tester", "BATTERY LOAD"), ("airflow_tester", "AIRFLOW TEST")):
        if name == "battery_tester":
            indicator_y, bezel_radius, lens_radius = 0.009, 0.004, 0.0026
            display_y, display_depth, screen_shift, screen_depth = 0.039, 0.051, 0.0, 0.036
        else:
            indicator_y, bezel_radius, lens_radius = 0.001, 0.006, 0.004
            # Clear the cup's extraction corridor beside the airflow inlet.
            display_y, display_depth, screen_shift, screen_depth = 0.047, 0.034, 0.007, 0.032
        g.asset(
            name,
            0.7,
            test_button=[0.042, -0.048, 0.024],
            screen=[0, 0.037 + screen_shift, 0.046],
            indicator_positions={
                "pass": [-0.038, indicator_y, 0.030],
                "fail": [-0.012, indicator_y, 0.030],
                "idle": [0.014, indicator_y, 0.030],
            },
            measurement_zone=[0, -0.018, 0.025],
            battery_socket=[-0.017, -0.032, 0.025],
        )
        g.box("instrument_body", (0, 0, 0.012), (0.145, 0.135, 0.024), "ivory", 0.007, collision=True)
        g.box("display_housing", (0, display_y, 0.031), (0.133, display_depth, 0.026), "navy", 0.004, collision=True)
        g.box("screen_glass", (0, 0.040 + screen_shift, 0.045), (0.111, screen_depth, 0.002), "screen", 0.003)
        g.label(label, (0, 0.045 + screen_shift, 0.0465), 0.008, "cyan")
        status_names = {}
        readings = {
            "idle": "IDLE",
            "run": "TESTING",
            "invalid": "INVALID",
            "pass": "PASS",
            "fail": "FAIL",
        }
        for status, reading in readings.items():
            material = "amber" if status in ("fail", "invalid") else "cyan"
            obj = g.label(reading, (-0.028, 0.031 + screen_shift, 0.0465), 0.006, material)
            obj.name = f"{name}_status_{status}"
            status_names[status] = obj.name
        g.current.affordances["status_object_names"] = status_names
        reading_names = {}
        values = (
            (20.0, 13.5, 12.0) if name == "battery_tester" else (0.0, 22.0, 7.7, 9.9, 3.465, 17.6, 6.16, 7.92, 2.772)
        )
        unit = "V" if name == "battery_tester" else "L/s"
        for value in values:
            key = f"{value:g}"
            obj = g.label(f"{key} {unit}", (0.025, 0.031 + screen_shift, 0.0465), 0.006, "cyan")
            obj.name = f"{name}_reading_{key.replace('.', '_')}"
            reading_names[key] = obj.name
        g.current.affordances["reading_object_names"] = reading_names
        for x, text in ((-0.038, "PASS"), (-0.012, "FAIL"), (0.014, "IDLE")):
            g.ring("status_bezel", (x, indicator_y, 0.027), bezel_radius, lens_radius, 0.004, "steel")
            g.cylinder("status_lens", (x, indicator_y, 0.027), lens_radius, 0.003, "screen")
            if name != "battery_tester":
                g.label(text, (x, -0.010, 0.025), 0.0045, "navy")
        if name == "battery_tester":
            for side in (-1, 1):
                g.box(
                    "load_rail",
                    (-0.017, -0.032 + side * 0.034, 0.030),
                    (0.070, 0.005, 0.012),
                    "teal",
                    0.001,
                    collision=True,
                )
            g.label("A  /  18V", (-0.023, -0.032, 0.025), 0.007, "navy")
            g.label("LOAD TEST", (0.043, -0.024, 0.025), 0.0048, "navy")
        else:
            # Parallel +X-facing ports let both adapter stems insert and extract
            # together. Orthogonal or opposed sockets would trap a rigid bridge.
            g.box("intake_pedestal", (-0.029, -0.070, 0.041), (0.042, 0.025, 0.036), "ivory", 0.004, collision=True)
            airflow_port_inner_radius = 0.012
            g.ring(
                "intake_seal",
                (-0.029, -0.070, 0.077),
                0.018,
                airflow_port_inner_radius,
                0.020,
                "rubber",
                AXIS_X,
                collision=True,
            )
            g.current.affordances["airflow_port"] = [-0.019, -0.070, 0.077]
            g.current.affordances["airflow_port_inner_radius"] = airflow_port_inner_radius
            g.current.affordances["test_button"] = [-0.100, -0.135, 0.0]
            # Support the bridge after release without blocking the cup's +X
            # removal corridor. The pad and two inlet lips support its mass;
            # its top is 1 mm below the nominal bridge underside for insertion.
            g.box(
                "adapter_landing_post",
                (0.039, -0.056, 0.04275),
                (0.012, 0.010, 0.0375),
                "steel",
                0.001,
                collision=True,
            )
            g.box(
                "adapter_landing_pad",
                (0.040, -0.056, 0.0635),
                (0.016, 0.014, 0.004),
                "rubber",
                0.0002,
                collision=True,
            )
            for i in range(6):
                g.box("exhaust_grill", (-0.050 + i * 0.012, -0.063, 0.025), (0.006, 0.007, 0.001), "rubber", 0.0005)

    g.asset("button_base", 0.02)
    g.cylinder("bezel", (0, 0, 0.004), 0.014, 0.008, "steel", collision=True)
    g.asset("test_button", 0.012, travel=0.006, press_axis=[0, 0, -1])
    g.cylinder("cap", (0, 0, 0.004), 0.011, 0.008, "orange", collision=True)
    g.ring("top_rim", (0, 0, 0.008), 0.0085, 0.0075, 0.0008, "ivory")

    g.asset(
        "release_panel",
        0.065,
        button_poses={"battery": [-0.110, 0, 0.001], "cup": [0, 0, 0.001], "cradle": [0.110, 0, 0.001]},
    )
    g.box("label_plate", (0, 0, 0.0005), (0.345, 0.070, 0.001), "navy", 0.0004, collision=True)
    for x, label in ((-0.110, "BATTERY"), (0, "CUP"), (0.110, "CRADLE")):
        g.label(label, (x, 0.024, 0.0012), 0.008)
        g.label("RELEASE", (x, -0.025, 0.0012), 0.0055, "cyan")
        for side in (-1, 1):
            g.box("control_marker", (x + side * 0.030, 0, 0.0012), (0.004, 0.018, 0.0004), "teal", 0.0001)

    g.asset("airflow_test_panel", 0.012, button_pose=[0, 0, 0.001])
    g.box("label_plate", (0, 0, 0.0005), (0.087, 0.058, 0.001), "navy", 0.0004, collision=True)
    g.label("AIRFLOW TEST", (0, -0.022, 0.0012), 0.0055)
    g.label("CONNECT BRIDGE", (0, 0.023, 0.0012), 0.0042, "cyan")

    for name, label, color in (
        ("service_bin", "SERVICE", "teal"),
        ("battery_bin", "BATTERY SERVICE", "orange"),
        ("filter_bin", "FILTER SERVICE", "teal"),
        ("waste_bin", "DRY WASTE", "navy"),
    ):
        height = 0.075 if name == "battery_bin" else 0.065
        tray(g, name, (0.110, 0.115, height), color, label)
    tray(g, "parking_tray", (0.155, 0.170, 0.014), "teal", "PARTS PARKING")
    g.current.affordances["tool_poses"] = {
        "crevice_tool": [-0.005, -0.050, 0.008],
        "brush_tool": [-0.005, 0.035, 0.008],
    }
    tray(g, "spare_rack", (0.260, 0.140, 0.025), "ivory", "CLEAN PARTS / A    B")
    g.current.affordances["stock_poses"] = {
        "battery_spare": [-0.086, 0, 0.008],
        "battery_decoy": [0.086, 0, 0.008],
        "filter_spare": {
            "position_xyz": [0, -0.033, 0.008],
            "rotation_xyzw": [0, 0, math.sqrt(0.5), math.sqrt(0.5)],
        },
        "filter_decoy": [0, 0.033, 0.012],
    }
    for x in (-0.043, 0.043):
        g.box("divider", (x, 0.008, 0.018), (0.003, 0.090, 0.027), "teal", 0.001, collision=True)
    for x, label in ((-0.086, "A"), (0, "A"), (0.086, "B")):
        g.label(label, (x, 0.018, 0.007), 0.020, "teal")


def build_workcell(g: Geometry):
    """Build the workstation, floor and work-order placard from original geometry."""
    g.asset("bench", 55, tabletop_height=0, usable_bounds=[[-0.5, -0.5, 0], [0.5, 0.5, 0]])
    g.box("worktop", (0, 0, -0.025), (1.0, 1.0, 0.050), "worktop", 0.008, collision=True)
    g.box("front_apron", (0, -0.485, -0.075), (0.93, 0.026, 0.10), "navy", 0.006, collision=True)
    for x in (-0.43, 0.43):
        for y in (-0.43, 0.43):
            g.box("leg", (x, y, -0.400), (0.045, 0.045, 0.70), "steel", 0.003, collision=True)
            g.box("foot", (x, y, -0.764), (0.07, 0.07, 0.025), "rubber", 0.005, collision=True)
    g.label("ARENA  /  RETURN TO SERVICE", (0, -0.460, 0.001), 0.018, "navy")
    g.asset("floor", 1000)
    g.box("floor_tile", (0, 0, -0.010), (6, 6, 0.020), "floor", 0.005, collision=True)
    g.asset("work_order", 0.05)
    g.box("board", (0, 0, 0.004), (0.15, 0.10, 0.008), "navy", 0.004, collision=True)
    g.box("paper", (0, 0, 0.0085), (0.14, 0.09, 0.001), "ivory", 0.001)
    for text, y, size in (
        ("RETURN TO SERVICE", 0.030, 0.009),
        ("MODEL A / 18V", 0.014, 0.007),
        ("TEST / SERVICE / VERIFY", -0.005, 0.006),
        ("PRESERVE WORKING PARTS", -0.023, 0.006),
        ("PACK: BRUSH + CREVICE", -0.036, 0.006),
    ):
        g.label(text, (0, y, 0.0092), size, "navy")


def build_all(g: Geometry):
    """Construct the complete original asset library."""
    build_vacuum(g)
    build_tools(g)
    build_fixtures(g)
    build_case(g)
    build_workcell(g)
