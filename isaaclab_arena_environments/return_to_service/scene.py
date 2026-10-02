# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compose the compact service workstation and its shared geometric contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .assets import PreparedAssets, prepare_assets

if TYPE_CHECKING:
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    from .measurements import Bounds


BATTERY_NAMES = ("battery_original", "battery_spare", "battery_decoy")
FILTER_NAMES = ("filter_original", "filter_spare", "filter_decoy")
DEBRIS_NAMES = ("debris_0", "debris_1", "debris_2")


@dataclass(frozen=True)
class Region:
    """Describe an oriented interior volume relative to the local environment frame."""

    center_xyz: tuple[float, float, float]
    half_extents_xyz: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class SocketSpec:
    """Describe candidate component root poses relative to one parent body."""

    parent_name: str
    pose_in_parent: Pose
    candidate_names: tuple[str, ...]
    capture_bounds: Bounds | None = None
    """Optional parent-frame acceptance volume for the candidate's seating feature."""

    candidate_bounds: dict[str, Bounds] = field(default_factory=dict)
    """Candidate-local seating-feature geometry for containment-based alignment."""

    position_tolerance_m: float = 0.003
    """Maximum root-position error for seating and passive retention."""


@dataclass(frozen=True)
class ButtonSpec:
    """Describe a physically depressible button and its measured threshold."""

    scene_key: str
    position_xyz: tuple[float, float, float]
    joint_name: str = "press"
    cap_body_name: str = "cap"
    pressed_position_m: float = -0.004


@dataclass(frozen=True)
class ServiceLayout:
    """Share scene geometry with task predicates, reset events, and example policies."""

    table_height_m: float
    initial_poses: dict[str, Pose]
    """Object root poses relative to each environment origin."""

    source_names: dict[str, str]
    """Runtime scene names mapped to Blender manifest asset names."""

    sockets: dict[str, SocketSpec]
    regions: dict[str, Region]
    buttons: dict[str, ButtonSpec]
    packing_poses: dict[str, Pose]
    """Component root target poses in the case base frame."""

    grasp_poses: dict[str, Pose]
    """Suggested grasp frames in component roots: +Z points outward and Y spans the jaws."""

    source_records: dict[str, dict[str, Any]]
    """Blender-authored bounds, instrument visuals, and interaction geometry."""

    case_floor_contact_allowance_m: float = 50e-6
    """Permitted resting contact penetration at the case floor; side and ceiling bounds stay unchanged."""


@dataclass(frozen=True)
class ServiceScene:
    """Return the Arena scene and the geometry contract used to evaluate it."""

    scene: Scene
    assets: dict[str, Object]
    layout: ServiceLayout
    prepared: PreparedAssets


def build_service_scene(asset_root: str | Path | None = None, table_height_m: float = 0.78) -> ServiceScene:
    """Build the Blender workstation around a fixed robot at (0, 0, table_height_m).

    Args:
        asset_root: Blender manifest and exported asset directory.
        table_height_m: World height of the bench's authored top surface.

    Returns:
        The configured Arena scene, named assets, and socket/region contract.
    """
    from isaaclab.actuators import ImplicitActuatorCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    assert table_height_m > 0.0, "The workstation requires a positive table height"
    prepared = prepare_assets(asset_root)
    source_names = {
        "bench": "bench",
        "floor": "floor",
        "lighting": "lighting",
        "work_order": "work_order",
        "body": "vacuum_body",
        "dust_cup": "dust_cup",
        "battery_original": "battery",
        "battery_spare": "battery",
        "battery_decoy": "battery_decoy",
        "filter_original": "filter",
        "filter_spare": "filter",
        "filter_decoy": "filter_decoy",
        "crevice_tool": "crevice_tool",
        "brush_tool": "brush_tool",
        "obstruction": "obstruction",
        "cradle": "cradle",
        "battery_tester": "battery_tester",
        "airflow_tester": "airflow_tester",
        "airflow_adapter": "airflow_adapter",
        "battery_service": "battery_bin",
        "filter_service": "filter_bin",
        "waste": "waste_bin",
        "spare_rack": "spare_rack",
        "parking_tray": "parking_tray",
        "release_panel": "release_panel",
        "airflow_test_panel": "airflow_test_panel",
    }
    for name in DEBRIS_NAMES:
        source_names[name] = "debris"
    for source_name in set(source_names.values()):
        prepared.source_path(source_name)

    poses = {
        "bench": Pose((0.48, 0.0, table_height_m)),
        "floor": Pose((0.48, 0.0, 0.0)),
        "lighting": Pose(),
        "work_order": _on_bench(prepared, "work_order", 0.14, -0.25, table_height_m),
        "cradle": _on_bench(prepared, "cradle", 0.46, 0.15, table_height_m),
        "battery_tester": _on_bench(prepared, "battery_tester", 0.235, 0.315, table_height_m),
        "airflow_tester": _on_bench(prepared, "airflow_tester", 0.67, 0.155, table_height_m),
        "battery_service": _on_bench(prepared, "battery_bin", 0.715, -0.08, table_height_m),
        "filter_service": _on_bench(prepared, "filter_bin", 0.715, -0.235, table_height_m),
        "waste": _on_bench(prepared, "waste_bin", 0.645, 0.345, table_height_m),
        "spare_rack": _on_bench(prepared, "spare_rack", 0.450, 0.35, table_height_m),
        "parking_tray": _on_bench(prepared, "parking_tray", 0.310, 0.005, table_height_m),
        "case": _on_bench(prepared, "case_base", 0.445, -0.25, table_height_m),
        "airflow_adapter": _on_bench(prepared, "airflow_adapter", 0.585, -0.010, table_height_m),
        "release_panel": Pose((0.440, 0.435, table_height_m + 0.001)),
        "airflow_test_panel": Pose((0.570, 0.020, table_height_m + 0.001)),
    }
    # Open the lid away from the service area, leaving the tool approach clear.
    poses["case"] = Pose(poses["case"].position_xyz, (0.0, 0.0, 1.0, 0.0))
    filter_footprints = {}
    for name in FILTER_NAMES:
        filter_footprints[name] = prepared.affordance(source_names[name], "support_footprint_bounds")
    sockets = {
        "cradle": SocketSpec("cradle", _affordance_pose(prepared, "cradle", "body_socket"), ("body",)),
        "battery": SocketSpec("body", _affordance_pose(prepared, "vacuum_body", "battery_socket"), BATTERY_NAMES),
        "cup": SocketSpec(
            "body",
            _affordance_pose(prepared, "vacuum_body", "cup_socket"),
            ("dust_cup",),
            position_tolerance_m=0.001,
        ),
        "filter": SocketSpec(
            "body",
            _affordance_pose(prepared, "vacuum_body", "filter_socket"),
            FILTER_NAMES,
            capture_bounds=prepared.affordance("vacuum_body", "filter_capture_bounds"),
            candidate_bounds=filter_footprints,
        ),
        "airflow_adapter": SocketSpec(
            "dust_cup", _affordance_pose(prepared, "dust_cup", "airflow_adapter_socket"), ("airflow_adapter",)
        ),
        "battery_tester": SocketSpec(
            "battery_tester", _affordance_pose(prepared, "battery_tester", "battery_socket"), BATTERY_NAMES
        ),
        "obstruction": SocketSpec(
            "dust_cup", _affordance_pose(prepared, "dust_cup", "obstruction_socket"), ("obstruction",)
        ),
    }
    for socket_name, candidate in (
        ("cradle", "body"),
        ("battery", "battery_original"),
        ("cup", "dust_cup"),
        ("filter", "filter_original"),
        ("obstruction", "obstruction"),
    ):
        socket = sockets[socket_name]
        poses[candidate] = poses[socket.parent_name].multiply(socket.pose_in_parent)

    rack_positions = prepared.affordance("spare_rack", "stock_poses")
    for name in ("battery_spare", "battery_decoy", "filter_spare", "filter_decoy"):
        assert name in rack_positions, f"Spare rack does not define a {name} stock position"
        poses[name] = poses["spare_rack"].multiply(_pose(rack_positions[name]))
    tool_positions = prepared.affordance("parking_tray", "tool_poses")
    for name in ("crevice_tool", "brush_tool"):
        assert name in tool_positions, f"Parking tray does not define a {name} initial pose"
        poses[name] = poses["parking_tray"].multiply(_pose(tool_positions[name]))
    debris_positions = prepared.affordance("dust_cup", "debris_poses")
    assert len(debris_positions) == len(DEBRIS_NAMES), "The dust cup must define three initial debris poses"
    for name, local_pose in zip(DEBRIS_NAMES, debris_positions, strict=True):
        poses[name] = poses["dust_cup"].multiply(_pose(local_pose))

    button_positions = {
        "battery_test_button": (
            poses["battery_tester"].multiply(_affordance_pose(prepared, "battery_tester", "test_button")).position_xyz
        ),
        "airflow_test_button": (
            poses["airflow_tester"].multiply(_affordance_pose(prepared, "airflow_tester", "test_button")).position_xyz
        ),
        "battery_release": (0.330, 0.435, table_height_m + 0.002),
        "cup_release": (0.440, 0.435, table_height_m + 0.002),
        "cradle_release": (0.550, 0.435, table_height_m + 0.002),
    }
    buttons = {}
    for name, position in button_positions.items():
        buttons[name] = ButtonSpec(name, position)
        poses[name] = Pose(position)

    static_names = {
        "bench",
        "floor",
        "lighting",
        "work_order",
        "battery_service",
        "filter_service",
        "waste",
        "spare_rack",
        "parking_tray",
        "release_panel",
        "airflow_test_panel",
    }
    kinematic_names = {"cradle", "battery_tester", "airflow_tester"}
    assets = {}
    for name, source_name in source_names.items():
        if name in static_names:
            kind = "static"
            object_type = ObjectType.BASE
        else:
            kind = "kinematic" if name in kinematic_names else "rigid"
            object_type = ObjectType.RIGID
        assets[name] = Object(
            name=name,
            prim_path="/World/ServiceLighting" if name == "lighting" else None,
            object_type=object_type,
            usd_path=str(prepared.asset_usd(source_name, kind)),
            initial_pose=poses[name],
        )

    for name in buttons:
        button = Object(
            name=name,
            object_type=ObjectType.ARTICULATION,
            usd_path=str(prepared.button_usd()),
            initial_pose=poses[name],
        )
        button.object_cfg.actuators = {
            "spring": ImplicitActuatorCfg(
                joint_names_expr=["press"], stiffness=180.0, damping=1.5, joint_effort_limit=8.0
            )
        }
        button.object_cfg.init_state.joint_pos = {"press": 0.0}
        button.object_cfg.init_state.joint_vel = {"press": 0.0}
        assets[name] = button

    case = Object(
        name="case",
        object_type=ObjectType.ARTICULATION,
        usd_path=str(prepared.case_usd()),
        initial_pose=poses["case"],
    )
    case.object_cfg.actuators = {
        "passive": ImplicitActuatorCfg(
            joint_names_expr=["hinge", "latch"], stiffness=0.0, damping=0.08, joint_effort_limit=10.0
        )
    }
    case.object_cfg.init_state.joint_pos = {"hinge": -1.8, "latch": 1.4}
    case.object_cfg.init_state.joint_vel = {"hinge": 0.0, "latch": 0.0}
    assets["case"] = case

    regions = {}
    for name in ("battery_service", "filter_service", "waste", "parking_tray", "spare_rack"):
        source_name = source_names[name]
        regions[name] = _interior_region(prepared, source_name, poses[name])
    packing_records = prepared.affordance("case_base", "packing_poses")
    packing_poses = {}
    for name in ("body", "battery", "crevice_tool", "brush_tool"):
        assert name in packing_records, f"Case does not define its {name} packing pose"
        packing_poses[name] = _pose(packing_records[name])
    regions["case"] = _interior_region(prepared, "case_base", poses["case"])

    grasp_poses = {}
    for name, source_name in source_names.items():
        grasp = prepared.record(source_name).get("affordances", {}).get("grasp")
        if grasp is not None:
            grasp_poses[name] = _pose(grasp)
    source_records = {name: prepared.record(source) for name, source in source_names.items()}
    source_records["case"] = prepared.record("case_base")
    layout = ServiceLayout(
        table_height_m, poses, source_names, sockets, regions, buttons, packing_poses, grasp_poses, source_records
    )
    return ServiceScene(Scene(list(assets.values())), assets, layout, prepared)


def _pose(value: Any) -> Pose:
    from isaaclab_arena.utils.pose import Pose

    if isinstance(value, dict):
        assert "position_xyz" in value, "An asset pose must declare position_xyz"
        return Pose(
            tuple(value["position_xyz"]),
            tuple(value.get("rotation_xyzw", (0.0, 0.0, 0.0, 1.0))),
        )
    assert isinstance(value, (list, tuple)) and len(value) == 3, "An asset pose must be XYZ or a pose dictionary"
    return Pose(tuple(value))


def _affordance_pose(prepared: PreparedAssets, source_name: str, name: str) -> Pose:
    return _pose(prepared.affordance(source_name, name))


def _on_bench(prepared: PreparedAssets, source_name: str, x: float, y: float, table_height_m: float) -> Pose:
    from isaaclab_arena.utils.pose import Pose

    lower_z = float(prepared.record(source_name)["bounds_min"][2])
    return Pose((x, y, table_height_m - lower_z + 0.002))


def _interior_region(prepared: PreparedAssets, source_name: str, initial_pose: Pose) -> Region:
    from isaaclab_arena.utils.pose import Pose

    lower, upper = prepared.affordance(source_name, "interior_bounds")
    assert len(lower) == len(upper) == 3, "Interior bounds must contain two XYZ corners"
    center = tuple((lower[index] + upper[index]) / 2 for index in range(3))
    half_extents = tuple((upper[index] - lower[index]) / 2 for index in range(3))
    assert all(extent > 0 for extent in half_extents), "Interior bounds must have positive volume"
    T_E_R = initial_pose.multiply(Pose(center))
    return Region(T_E_R.position_xyz, half_extents, T_E_R.rotation_xyzw)
