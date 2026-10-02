# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check coherent layouts against authored physical geometry without starting simulation."""

import math
import torch
from types import SimpleNamespace

import pytest
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.geometry.collision_geometry import read_collision_primitives
from isaaclab_arena.geometry.containment import RegionContainment
from isaaclab_arena.tests.utils.return_to_service import _require_assets
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena_environments.return_to_service.assets import prepare_assets
from isaaclab_arena_environments.return_to_service.layouts import (
    LAYOUT_NAMES,
    apply_service_layout,
    critical_interaction_poses,
    service_placement_layouts,
)
from isaaclab_arena_environments.return_to_service.scene import (
    BATTERY_NAMES,
    DEBRIS_NAMES,
    FILTER_NAMES,
    ServiceScene,
    _pose,
    build_service_layout,
)


@pytest.fixture
def workcell():
    _require_assets()
    prepared = prepare_assets()
    layout = build_service_layout(prepared)
    rigid_names = {
        *BATTERY_NAMES,
        *FILTER_NAMES,
        *DEBRIS_NAMES,
        "body",
        "dust_cup",
        "crevice_tool",
        "brush_tool",
        "obstruction",
        "airflow_adapter",
        "cradle",
        "battery_tester",
        "airflow_tester",
    }
    assets = {}
    # Use real Arena object configuration with authored source geometry. The offline
    # contract check does not create PhysX overlays, a Scene, or a SimulationApp.
    for name, pose in layout.initial_poses.items():
        source = layout.source_names.get(name, "case_base" if name == "case" else "button_base")
        kind = ObjectType.RIGID if name in rigid_names else ObjectType.BASE
        if name == "case" or name in layout.buttons:
            kind = ObjectType.ARTICULATION
        assets[name] = Object(name, usd_path=str(prepared.source_path(source)), object_type=kind, initial_pose=pose)
    assets["case"].object_cfg.init_state.joint_pos = {"hinge": -1.8, "latch": 1.4}
    return ServiceScene(SimpleNamespace(assets=assets), assets, layout, prepared)


def _tensor(pose):
    return torch.tensor((*pose.position_xyz, *pose.rotation_xyzw), dtype=torch.float64)


def _relative(parent, child):
    parent, child = _tensor(parent), _tensor(child)
    inverse = quat_conjugate(parent[3:])
    return torch.cat((quat_apply(inverse, child[:3] - parent[:3]), quat_mul(inverse, child[3:])))


def _same_pose(actual, expected):
    actual, expected = _tensor(actual), _tensor(expected)
    torch.testing.assert_close(actual[:3], expected[:3], atol=5e-7, rtol=0)
    assert abs(float(torch.dot(actual[3:], expected[3:]))) == pytest.approx(1.0, abs=3e-7)


def _held_root(layout, subject, tcp):
    local = layout.grasp_poses[subject].multiply(Pose(rotation_xyzw=(0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))))
    if subject == "obstruction":
        local = local.multiply(Pose(rotation_xyzw=(1.0, 0.0, 0.0, 0.0)))
    tcp, local = _tensor(tcp), _tensor(local)
    rotation = quat_mul(tcp[3:], quat_conjugate(local[3:]))
    position = tcp[:3] - quat_apply(rotation, local[:3])
    return Pose(tuple(position.tolist()), tuple(rotation.tolist()))


def test_baseline_is_unchanged_and_invalid_selection_does_not_mutate(workcell):
    before = {name: asset.get_initial_pose() for name, asset in workcell.assets.items()}
    assert apply_service_layout(workcell, "baseline") is workcell
    with pytest.raises(AssertionError, match="Unknown service layout"):
        apply_service_layout(workcell, "unknown")
    assert {name: asset.get_initial_pose() for name, asset in workcell.assets.items()} == before


@pytest.mark.parametrize("layout_name", LAYOUT_NAMES)
def test_coupled_sockets_buttons_stock_and_region_frames_move_together(workcell, layout_name):
    original = workcell.layout
    selected = apply_service_layout(workcell, layout_name)
    poses = selected.layout.initial_poses
    for name in poses:
        _same_pose(selected.assets[name].get_initial_pose(), poses[name])
        if name in ("bench", "floor", "lighting"):
            assert poses[name] == original.initial_poses[name]
        else:
            torch.testing.assert_close(
                _relative(poses["cradle"], poses[name]),
                _relative(original.initial_poses["cradle"], original.initial_poses[name]),
                atol=5e-7,
                rtol=0,
            )
    for name, socket in selected.layout.sockets.items():
        if name in ("cradle", "battery", "cup", "filter", "obstruction"):
            child = socket.candidate_names[0]
            _same_pose(poses[child], poses[socket.parent_name].multiply(socket.pose_in_parent))
    for name, region in selected.layout.regions.items():
        old = original.regions[name]
        torch.testing.assert_close(
            _relative(poses[name], Pose(region.center_xyz, region.rotation_xyzw)),
            _relative(original.initial_poses[name], Pose(old.center_xyz, old.rotation_xyzw)),
            atol=5e-7,
            rtol=0,
        )
    for name, button in selected.layout.buttons.items():
        assert button.position_xyz == poses[name].position_xyz


@pytest.mark.parametrize("layout_name", LAYOUT_NAMES)
def test_cached_layout_covers_writable_roots_and_preserves_joint_configuration(workcell, layout_name, tmp_path):
    selected = apply_service_layout(workcell, layout_name)
    layouts = service_placement_layouts(selected)
    layouts.validate_assets(list(selected.assets.values()))
    expected = {
        name
        for name, asset in selected.assets.items()
        if asset.object_type in (ObjectType.RIGID, ObjectType.ARTICULATION)
    }
    assert set(layouts.poses) == expected
    assert layouts.num_layouts == 1
    for name in expected:
        assert layouts.poses[name] == [selected.layout.initial_poses[name]]
    assert selected.assets["case"].object_cfg.init_state.joint_pos == {"hinge": -1.8, "latch": 1.4}
    path = tmp_path / "layout.jsonl"
    layouts.write_episode_jsonl(path, source=f"return_to_service/{layout_name}")
    assert type(layouts).from_episode_jsonl(path).poses == layouts.poses


@pytest.mark.parametrize("layout_name", LAYOUT_NAMES)
def test_supported_roots_stay_over_bench_and_hinged_members_clear_its_surface(workcell, layout_name):
    selected = apply_service_layout(workcell, layout_name)
    prepared, layout = selected.prepared, selected.layout
    bench = read_collision_primitives(prepared.source_path("bench"), dtype=torch.float64)
    bench_bounds = bench.bounds_in_frame(_tensor(layout.initial_poses["bench"]))
    lower, upper = bench_bounds[0, :2], bench_bounds[1, :2]
    shapes = []
    for name, source in layout.source_names.items():
        if name not in ("bench", "floor", "lighting") and prepared.record(source)["collision_count"]:
            shapes.append((name, source, layout.initial_poses[name]))
    shapes.append(("case", "case_base", layout.initial_poses["case"]))
    case = prepared.record("case_base")["affordances"]
    T_E_C = layout.initial_poses["case"]
    hinge = Pose(rotation_xyzw=(math.sin(-0.9), 0.0, 0.0, math.cos(-0.9)))
    shapes.append(("case_lid", "case_lid", T_E_C.multiply(_pose(case["lid_closed_pose"])).multiply(hinge)))
    rotation = Pose(rotation_xyzw=(math.sin(0.7), 0.0, 0.0, math.cos(0.7)))
    shapes.append(
        ("case_latch", "case_latch", T_E_C.multiply(Pose(tuple(case["latch"]["position_xyz"]))).multiply(rotation))
    )
    for name, source, pose in shapes:
        geometry = read_collision_primitives(prepared.source_path(source), dtype=torch.float64)
        bounds = geometry.bounds_in_frame(_tensor(pose))
        if name in ("case_lid", "case_latch"):
            # These hinged members may overhang the edge; the unchanged baseline
            # deliberately opens the lid away from the service area. Their whole
            # collision volume must clear the tabletop at the initial joint pose.
            assert float(bounds[0, 2]) > float(bench_bounds[1, 2]), (layout_name, name, bounds)
            continue
        assert bool((bounds[0, :2] >= lower).all()), (layout_name, name, bounds, lower)
        assert bool((bounds[1, :2] <= upper).all()), (layout_name, name, bounds, upper)


@pytest.mark.parametrize("layout_name", LAYOUT_NAMES)
def test_nominal_packed_assembly_retains_full_shape_containment(workcell, layout_name):
    selected = apply_service_layout(workcell, layout_name)
    layout = selected.layout
    scene_cfg = SimpleNamespace(**{name: asset.object_cfg for name, asset in selected.assets.items()})
    checker = RegionContainment(scene_cfg)
    T_E_C = layout.initial_poses["case"]
    body = T_E_C.multiply(layout.packing_poses["body"])
    targets = {
        "body": body,
        "dust_cup": body.multiply(layout.sockets["cup"].pose_in_parent),
        "filter_original": body.multiply(layout.sockets["filter"].pose_in_parent),
        "battery_original": T_E_C.multiply(layout.packing_poses["battery"]),
        "crevice_tool": T_E_C.multiply(layout.packing_poses["crevice_tool"]),
        "brush_tool": T_E_C.multiply(layout.packing_poses["brush_tool"]),
    }
    bounds = layout.source_records["case"]["affordances"]["interior_bounds"]
    for name, target in targets.items():
        measured = checker.measure_in_frame(name, _tensor(target), _tensor(T_E_C), bounds)
        assert bool(measured.contained), (layout_name, name, measured.raw_face_margins_m)


@pytest.mark.parametrize("layout_name", LAYOUT_NAMES)
def test_critical_targets_are_tcp_poses_and_follow_the_selected_workstation(workcell, layout_name):
    original = workcell.layout
    expected = critical_interaction_poses(original)
    selected = apply_service_layout(workcell, layout_name)
    actual = critical_interaction_poses(selected.layout)
    assert expected.keys() == actual.keys()
    for label in expected:
        observed = _relative(selected.layout.initial_poses["cradle"], actual[label])
        target = _relative(original.initial_poses["cradle"], expected[label])
        torch.testing.assert_close(observed[:3], target[:3], atol=8e-7, rtol=0)
        assert abs(float(torch.dot(observed[3:], target[3:]))) == pytest.approx(1.0, abs=5e-7), label
    assert {
        "case/lid/-1.8",
        "case/latch/0",
        "extraction/dust_cup",
        "packing/body",
        "button/battery_release/press",
    } <= actual.keys()
    assert actual["initial/body/grasp"].position_xyz != selected.layout.initial_poses["body"].position_xyz


@pytest.mark.parametrize("layout_name", LAYOUT_NAMES)
def test_screened_service_poses_put_whole_parts_over_bins_and_keep_dump_mouth_centered(workcell, layout_name):
    selected = apply_service_layout(workcell, layout_name)
    layout = selected.layout
    targets = critical_interaction_poses(layout)
    for region_name, subject in (
        ("battery_service", "battery_original"),
        ("filter_service", "filter_original"),
        ("waste", "obstruction"),
    ):
        root = _held_root(layout, subject, targets[f"disposal/{region_name}"])
        relative = _relative(layout.initial_poses[region_name], root)
        source = layout.source_names[subject]
        geometry = read_collision_primitives(selected.prepared.source_path(source), dtype=torch.float64)
        bounds = geometry.bounds_in_frame(relative)
        interior = torch.tensor(
            layout.source_records[region_name]["affordances"]["interior_bounds"], dtype=torch.float64
        )
        assert bool((bounds[0, :2] >= interior[0, :2] + 0.003).all())
        assert bool((bounds[1, :2] <= interior[1, :2] - 0.003).all())
        assert float(bounds[0, 2]) > float(interior[1, 2])

    waste = layout.source_records["waste"]["affordances"]["interior_bounds"]
    cavity = layout.source_records["dust_cup"]["affordances"]["debris_cavity_cylinder"]
    mouth = Pose((cavity["x_range"][0], *cavity["center_yz"]))
    expected = torch.tensor(
        ((waste[0][0] + waste[1][0]) / 2, (waste[0][1] + waste[1][1]) / 2, waste[1][2] + 0.065), dtype=torch.float64
    )
    for label in ("waste/cup_dump/0", "waste/cup_dump/-90"):
        root = _held_root(layout, "dust_cup", targets[label])
        actual = _relative(layout.initial_poses["waste"], root.multiply(mouth))[:3]
        torch.testing.assert_close(actual, expected, atol=5e-7, rtol=0)

    staged = _held_root(layout, "dust_cup", targets["staging/dust_cup"])
    scene_cfg = SimpleNamespace(**{name: asset.object_cfg for name, asset in selected.assets.items()})
    measured = RegionContainment(scene_cfg).measure_in_frame(
        "dust_cup",
        _tensor(staged),
        _tensor(layout.initial_poses["case"]),
        layout.source_records["case"]["affordances"]["interior_bounds"],
    )
    assert bool(measured.contained), measured.raw_face_margins_m
