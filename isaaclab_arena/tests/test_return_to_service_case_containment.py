# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise measured case containment and conservative intrusion without simulation."""

import json
import math
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation
from types import SimpleNamespace

import pytest
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from isaaclab_arena_environments.return_to_service import containment
from isaaclab_arena_environments.return_to_service.collision_geometry import CollisionBoxes, CollisionPrimitives
from isaaclab_arena_environments.return_to_service.containment import RegionContainment
from isaaclab_arena_environments.return_to_service.measurements import box_contained_on_support
from isaaclab_arena_environments.return_to_service.runtime import ServiceRuntime


def pose(position=(0, 0, 0), quaternion=(0, 0, 0, 1), dtype=torch.float64):
    return torch.tensor((*position, *quaternion), dtype=dtype)


def compose(parent, local):
    return torch.cat((parent[:3] + quat_apply(parent[3:], local[:3]), quat_mul(parent[3:], local[3:])))


def configuration(path, scale=None):
    return SimpleNamespace(spawn=SimpleNamespace(usd_path=str(path), scale=scale))


@pytest.fixture
def configured(tmp_path):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    path = tmp_path / "compound.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    box = UsdGeom.Cube.Define(stage, "/Asset/Box")
    box.AddScaleOp().Set(Gf.Vec3d(0.02, 0.03, 0.04))
    cylinder = UsdGeom.Cylinder.Define(stage, "/Asset/Cylinder")
    cylinder.CreateRadiusAttr(0.015)
    cylinder.CreateHeightAttr(0.03)
    cylinder.CreateAxisAttr("Z")
    cylinder.AddTranslateOp().Set(Gf.Vec3d(0.03, 0, 0))
    for shape in (box, cylinder):
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        shape.GetPrim().SetCustomDataByKey("arena:feature", shape.GetPrim().GetName())
    stage.GetRootLayer().Save()
    scene = SimpleNamespace(part=configuration(path), case=configuration(tmp_path / "case.usda"))
    return scene, ((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1))


@pytest.mark.parametrize("batch_shape", ((), (3,), (2, 3)))
@pytest.mark.parametrize("scalar_region", (False, True))
def test_live_translated_rotated_regions_and_scalar_batch_agree(configured, batch_shape, scalar_region):
    scene, bounds = configured
    checker = RegionContainment(scene, {}, unit_scale_regions=("case",))
    local = pose((0.02, -0.01, 0.01), Rotation.from_euler("xyz", (0.1, -0.2, 0.3)).as_quat())
    expected = checker.measure_in_frame("part", local, pose(), bounds)
    for translation in ((0, 0, 0), (3.1, -2.4, 0.9)):
        region = pose(translation, Rotation.from_euler("xyz", (-0.3, 0.4, 1.1)).as_quat())
        world = compose(region, local).expand(*batch_shape, 7).clone()
        regions = region if scalar_region else region.expand(*batch_shape, 7)
        measured = checker.measure_in_frame("part", world, regions, bounds)
        assert measured.contained.shape == batch_shape
        assert bool(measured.contained.all())
        torch.testing.assert_close(measured.raw_face_margins_m, expected.raw_face_margins_m.expand(*batch_shape, 6))
        # Moving only the case after checker creation invalidates the same object pose.
        moved_region = regions.clone()
        moved_region[..., 0] += 0.5
        assert not bool(checker.measure_in_frame("part", world, moved_region, bounds).contained.any())


@pytest.mark.parametrize("face", range(6))
def test_six_face_contact_numerical_boundary_and_real_protrusion(configured, face):
    scene, bounds = configured
    checker = RegionContainment(scene, {})
    object_pose = pose()
    initial = checker.measure_in_frame("part", object_pose, pose(), bounds)
    side, axis = divmod(face, 3)
    direction = -1 if side == 0 else 1
    allowance = 50e-6 if face == 2 else 0.0
    object_pose[axis] = bounds[side][axis] - initial.occupied_bounds_R[side, axis] + direction * allowance
    touching = checker.measure_in_frame("part", object_pose, pose(), bounds, floor_allowance_m=50e-6)
    assert bool(touching.contained)
    assert float(touching.raw_face_margins_m[face]) == pytest.approx(-allowance, abs=1e-14)
    object_pose[axis] += direction * 0.9e-6
    assert bool(checker.measure_in_frame("part", object_pose, pose(), bounds, floor_allowance_m=50e-6).contained)
    object_pose[axis] += direction * 0.2e-6
    assert not bool(checker.measure_in_frame("part", object_pose, pose(), bounds, floor_allowance_m=50e-6).contained)


@pytest.mark.parametrize("bad", (0.0, 0.5, 2.0, float("nan"), float("inf")))
@pytest.mark.parametrize("frame", ("object", "region", "reciprocal"))
def test_individual_invalid_quaternions_cannot_cancel(configured, bad, frame):
    scene, bounds = configured
    checker = RegionContainment(scene, {})
    objects, regions = pose().repeat(2, 1), pose().repeat(2, 1)
    if frame in ("object", "reciprocal"):
        objects[1, 6] = bad
    if frame == "region":
        regions[1, 6] = bad
    elif frame == "reciprocal":
        regions[1, 6] = 1 / bad if bad else 0
    before_o, before_r = objects.clone(), regions.clone()
    measured = checker.measure_in_frame("part", objects, regions, bounds)
    assert measured.valid_pose.tolist() == [True, False]
    assert measured.contained.tolist() == [True, False]
    assert bool(torch.isnan(measured.occupied_bounds_R[1]).all())
    assert bool(torch.isnan(measured.effective_face_margins_m[1]).all())
    torch.testing.assert_close(objects, before_o, equal_nan=True)
    torch.testing.assert_close(regions, before_r, equal_nan=True)


@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize("bad", (float("nan"), float("inf"), -float("inf")))
@pytest.mark.parametrize("frame", ("object", "region"))
def test_nonfinite_translation_fails_closed(configured, index, bad, frame):
    scene, bounds = configured
    objects, regions = pose().repeat(2, 1), pose().repeat(2, 1)
    (objects if frame == "object" else regions)[1, index] = bad
    measured = RegionContainment(scene, {}).measure_in_frame("part", objects, regions, bounds)
    assert measured.valid_pose.tolist() == [True, False]
    assert measured.contained.tolist() == [True, False]


def test_small_quaternion_drift_is_normalized_without_mutating_observations(configured):
    scene, bounds = configured
    checker = RegionContainment(scene, {})
    region = pose((0.3, 0.2, -0.1), Rotation.from_euler("z", 0.3).as_quat())
    object_pose = compose(region, pose())
    expected = checker.measure_in_frame("part", object_pose, region, bounds)
    object_pose[3:] *= 1 + 0.9e-4
    region[3:] *= 1 - 0.9e-4
    original = object_pose.clone()
    measured = checker.measure_in_frame("part", object_pose, region, bounds)
    torch.testing.assert_close(measured.occupied_bounds_R, expected.occupied_bounds_R)
    torch.testing.assert_close(object_pose, original)
    object_pose[3:] *= 1.001
    assert not bool(checker.measure_in_frame("part", object_pose, region, bounds).contained)


@pytest.mark.parametrize(
    "bounds", (((0, 0, 0), (0, 1, 1)), ((0, 0, 0), (-1, 1, 1)), ((0, 0, 0), (1, math.inf, 1)), ((0, 0), (1, 1)))
)
def test_invalid_static_bounds_are_configuration_errors(configured, bounds):
    scene, _ = configured
    with pytest.raises(AssertionError):
        RegionContainment(scene, {}).measure_in_frame("part", pose(), pose(), bounds)


@pytest.mark.parametrize("allowance", (-1e-6, math.inf, math.nan))
def test_invalid_floor_allowance_is_configuration_error(configured, allowance):
    scene, bounds = configured
    with pytest.raises(AssertionError):
        RegionContainment(scene, {}).measure_in_frame("part", pose(), pose(), bounds, floor_allowance_m=allowance)


@pytest.mark.parametrize("scale", ((0, 1, 1), (-1, 1, 1), (math.inf, 1, 1), (math.nan, 1, 1), (1, 1)))
def test_invalid_component_and_case_scales_rejected(configured, scale):
    scene, bounds = configured
    scene.part.spawn.scale = scale
    with pytest.raises(AssertionError):
        RegionContainment(scene, {}).measure_in_frame("part", pose(), pose(), bounds)
    scene.case.spawn.scale = scale
    with pytest.raises(AssertionError):
        RegionContainment(scene, {}, unit_scale_regions=("case",))


def test_supported_component_scale_changes_shape_but_case_scale_is_not_silent(configured):
    scene, bounds = configured
    scene.part.spawn.scale = (2, 2, 3)
    checker = RegionContainment(scene, {}, unit_scale_regions=("case",))
    measured = checker.measure_in_frame("part", pose(), pose(), bounds)
    assert not bool(measured.contained), "The scaled box protrudes through top and bottom"
    assert float(measured.occupied_bounds_R[1, 2]) == pytest.approx(0.12)
    scene.case.spawn.scale = (2, 2, 2)
    with pytest.raises(AssertionError, match="configuration changed"):
        checker.measure_in_frame("part", pose(), pose(), bounds)
    with pytest.raises(AssertionError, match="unit spawn scale"):
        RegionContainment(scene, {}, unit_scale_regions=("case",))


@pytest.mark.parametrize("name", ("part", "case"))
@pytest.mark.parametrize(
    "field", ("variants", "collision_props", "prim_physics", "deformable_props", "usd_path", "scale")
)
def test_configuration_mutation_cannot_reuse_geometry_cache(configured, name, field):
    scene, bounds = configured
    checker = RegionContainment(scene, {}, unit_scale_regions=("case",))
    checker.measure_in_frame("part", pose(), pose(), bounds)
    value = (1.1, 1.1, 1.1) if field == "scale" else "changed"
    setattr(getattr(scene, name).spawn, field, value)
    with pytest.raises(AssertionError):
        checker.measure_in_frame("part", pose(), pose(), bounds)


def test_cache_reads_once_and_owns_bounds_and_diagnostics(configured, monkeypatch):
    scene, bounds = configured
    reader = containment.read_collision_primitives
    reads = []

    def read(*args, **kwargs):
        reads.append(args[0])
        return reader(*args, **kwargs)

    monkeypatch.setattr(containment, "read_collision_primitives", read)
    checker = RegionContainment(scene, {})
    input_bounds = torch.tensor(bounds, dtype=torch.float64)
    first = checker.measure_in_frame("part", pose(), pose(), input_bounds)
    first.occupied_bounds_R.fill_(100)
    first.raw_face_margins_m.fill_(100)
    input_bounds[1, 0] = 0.01
    assert not bool(checker.measure_in_frame("part", pose(), pose(), input_bounds).contained)
    assert bool(checker.measure_in_frame("part", pose(), pose(), bounds).contained)
    for x in (0.05, 0.2, 0):
        measured = checker.measure_in_frame("part", pose((x, 0, 0)), pose(), bounds)
        assert bool(measured.contained) == (x <= 0.05)
    assert len(reads) == 1


def test_dtype_device_shape_and_batch_contracts(configured):
    scene, bounds = configured
    checker = RegionContainment(scene, {})
    checker.measure_in_frame("part", pose(), pose(), bounds)
    for objects, regions in ((pose().float(), pose()), (pose(), pose().float()), (pose().to("meta"), pose())):
        with pytest.raises(AssertionError, match="another dtype or device"):
            checker.measure_in_frame("part", objects, regions, bounds)
    for invalid in (torch.zeros(6), torch.ones(7, dtype=torch.int64), torch.ones(7, dtype=torch.float16)):
        with pytest.raises(AssertionError):
            checker.measure_in_frame("part", invalid, pose(), bounds)
    with pytest.raises(AssertionError, match="batch dimensions"):
        checker.measure_in_frame("part", pose().repeat(2, 1), pose().repeat(3, 1), bounds)
    with pytest.raises(AssertionError, match="Bounds must match"):
        checker.measure_in_frame("part", pose(), pose(), torch.tensor(bounds, dtype=torch.float32))


def test_fixed_region_adapter_validates_and_freezes_its_definition(configured):
    scene, bounds = configured
    region = SimpleNamespace(center_xyz=(0, 0, 0), rotation_xyzw=(0, 0, 0, 1), half_extents_xyz=(0.1, 0.1, 0.1))
    checker = RegionContainment(scene, {"bin": region})
    assert bool(checker.contains("part", pose(), "bin"))
    region.center_xyz = (0.2, 0, 0)
    with pytest.raises(AssertionError, match="definition changed"):
        checker.contains("part", pose(), "bin")
    assert not bool(RegionContainment(scene, {"bin": region}).contains("part", pose(), "bin"))
    region.rotation_xyzw = (0, 0, 0, 2)
    with pytest.raises(AssertionError, match="fixed region pose"):
        RegionContainment(scene, {"bin": region}).contains("part", pose(), "bin")


@pytest.fixture
def native(monkeypatch):
    path = Path(__file__).with_name("data") / "return_to_service_body_placement.json"
    data = json.loads(path.read_text())
    shapes = {}
    records = {"case": {"affordances": {"interior_bounds": data["bounds"]}}}
    scene = SimpleNamespace(case=configuration("case.usda"))
    for name, component in data["components"].items():
        boxes = CollisionBoxes(
            torch.tensor(component["centers"], dtype=torch.float64),
            torch.tensor(component["axes"], dtype=torch.float64),
            torch.tensor(component["half_extents"], dtype=torch.float64),
            tuple(component["features"]),
        )
        shapes[name] = CollisionPrimitives(boxes, torch.tensor(component["cylinder_axes"], dtype=torch.int64))
        setattr(scene, name, configuration(f"{name}.usda"))
        records[name] = dict(zip(("bounds_min", "bounds_max"), component["authored_bounds"], strict=True))
    monkeypatch.setattr(containment, "read_collision_primitives", lambda path, **kwargs: shapes[Path(path).stem])
    poses = {name: torch.tensor(value, dtype=torch.float64) for name, value in data["poses"].items()}
    return data, scene, records, poses


def _runtime_for(scene_cfg, records, poses):
    runtime = ServiceRuntime.__new__(ServiceRuntime)
    layout = SimpleNamespace(source_records=records, case_floor_contact_allowance_m=50e-6)

    def batched(name):
        value = poses[name]
        return value[None] if value.ndim == 1 else value

    runtime.env = SimpleNamespace(cfg=SimpleNamespace(scene=scene_cfg), arena_world=SimpleNamespace(get_pose_w=batched))
    runtime.layout = layout
    runtime.region_containment = RegionContainment(scene_cfg, {}, unit_scale_regions=("case",))
    runtime._case_interior = None
    runtime._case_overlap_bounds = None
    runtime._case_region_definition = runtime._case_region_configuration()
    runtime._bounds_cache = {}
    runtime._case_names = tuple(name for name in poses if name != "case")
    runtime.update = lambda: pytest.fail("Geometry inspection must not advance evaluator state")
    return runtime


def assert_case_contents(runtime, expected):
    inside, overlaps = runtime._case_contents()
    for name, contained in expected.items():
        assert inside[name] == [contained]
        measured = runtime.measure_case_containment(
            name, runtime.env.arena_world.get_pose_w(name)[0], runtime.env.arena_world.get_pose_w("case")[0]
        )
        assert bool(measured.contained) is contained
    return inside, overlaps


def test_retained_native_envelope_rejection_and_runtime_primitive_acceptance(native):
    data, scene, records, poses = native
    runtime = _runtime_for(scene, records, poses)
    q_C_W = quat_conjugate(poses["case"][3:])
    T_C_B = torch.cat((quat_apply(q_C_W, poses["body"][:3] - poses["case"][:3]), quat_mul(q_C_W, poses["body"][3:])))
    assert not bool(
        box_contained_on_support(
            T_C_B, data["components"]["body"]["authored_bounds"], data["bounds"], floor_allowance_m=50e-6
        )
    )
    expected = dict.fromkeys(data["components"], True)
    assert_case_contents(runtime, expected)
    measured = runtime.region_containment.measure_in_frame(
        "body", poses["body"], poses["case"], data["bounds"], floor_allowance_m=50e-6
    )
    assert bool((measured.effective_face_margins_m > 0).all())
    assert float(measured.effective_face_margins_m[2]) == pytest.approx(10.671e-6, abs=0.05e-6)
    poses["body"][:3] -= quat_apply(poses["case"][3:], pose((0, 0, 20e-6))[:3])
    expected["body"] = False
    assert_case_contents(runtime, expected)


@pytest.mark.parametrize("name", ("body", "dust_cup", "filter_original"))
@pytest.mark.parametrize("face", range(6))
def test_each_assembly_member_can_protrude_through_each_face(native, name, face):
    data, scene, records, poses = native
    runtime = _runtime_for(scene, records, poses)
    measured = runtime.region_containment.measure_in_frame(
        name, poses[name], poses["case"], data["bounds"], floor_allowance_m=50e-6
    )
    _, axis = divmod(face, 3)
    displacement = poses[name].new_zeros(3)
    displacement[axis] = (-1 if face < 3 else 1) * (measured.effective_face_margins_m[face] + 5e-6)
    poses[name][:3] += quat_apply(poses["case"][3:], displacement)
    expected = dict.fromkeys(data["components"], True)
    expected[name] = False
    assert_case_contents(runtime, expected)


def test_case_measurements_follow_moving_case_and_environment_offset(native):
    data, scene, records, poses = native
    runtime = _runtime_for(scene, records, poses)
    assert_case_contents(runtime, dict.fromkeys(data["components"], True))
    parent = pose((3.2, -2.4, 0.7), Rotation.from_euler("xyz", (0.4, -0.7, 1.3)).as_quat())
    for name in poses:
        poses[name][:] = compose(parent, poses[name])
    assert_case_contents(runtime, dict.fromkeys(data["components"], True))
    poses["case"][0] += 1
    assert_case_contents(runtime, dict.fromkeys(data["components"], False))


@pytest.mark.parametrize("bad_name", ("case", "body", "dust_cup", "filter_original"))
def test_invalid_case_participant_refuses_containment_and_never_implies_absence(native, bad_name):
    data, scene, records, poses = native
    runtime = _runtime_for(scene, records, poses)
    poses[bad_name][0] = math.nan
    expected = {name: bad_name not in ("case", name) for name in data["components"]}
    inside, overlaps = assert_case_contents(runtime, expected)
    for name, valid in expected.items():
        if not valid:
            assert not inside[name][0]
            assert overlaps[name][0], "An invalid unexpected participant must prevent correct case contents"


def test_conservative_unexpected_partial_intrusion_and_scaled_envelope(configured):
    scene, bounds = configured
    records = {
        "case": {"affordances": {"interior_bounds": bounds}},
        "part": {"bounds_min": (-0.02, -0.03, -0.04), "bounds_max": (0.045, 0.03, 0.04)},
    }
    scene.part.spawn.scale = (2, 2, 2)
    poses = {"case": pose(), "part": pose((0.125, 0, 0))}
    runtime = _runtime_for(scene, records, poses)
    inside, overlap = runtime._case_contents()
    assert inside == {"part": [False]}
    assert overlap == {"part": [True]}, "Scaled bounds must preserve conservative unexpected-object exclusion"
    poses["part"][0] = 0.16
    assert runtime._case_contents() == ({"part": [False]}, {"part": [False]})
    poses["part"][6] = 0
    assert runtime._case_contents() == ({"part": [False]}, {"part": [True]})


def test_batched_and_scalar_case_measurements_agree_across_independent_environments(native):
    data, scene, records, original = native
    parent = pose((2.4, -3.1, 0.2), Rotation.from_euler("xyz", (0.3, 0.6, -0.7)).as_quat())
    poses = {name: torch.stack((value, compose(parent, value), value.clone())) for name, value in original.items()}
    poses["filter_original"][1, 2] += 0.2
    poses["dust_cup"][2, 6] = math.nan
    runtime = _runtime_for(scene, records, poses)
    inside, overlaps = runtime._case_contents()
    assert inside == {
        "body": [True, True, True],
        "dust_cup": [True, True, False],
        "filter_original": [True, False, True],
    }
    assert overlaps["dust_cup"][2], "Only the invalid environment must refuse absence"
    for env_id in range(3):
        for name in data["components"]:
            measured = runtime.measure_case_containment(name, poses[name][env_id], poses["case"][env_id])
            assert bool(measured.contained) == inside[name][env_id]


def test_runtime_rejects_changed_case_bounds_instead_of_using_stale_intrusion_cache(native):
    data, scene, records, poses = native
    runtime = _runtime_for(scene, records, poses)
    runtime._case_contents()
    records["case"]["affordances"]["interior_bounds"][1][0] += 0.05
    with pytest.raises(AssertionError, match="Case interior or floor allowance changed"):
        runtime._case_contents()


def test_conservative_intrusion_uses_normalized_accepted_live_frames(configured):
    scene, bounds = configured
    records = {
        "case": {"affordances": {"interior_bounds": bounds}},
        "part": {"bounds_min": (-0.02, -0.03, -0.04), "bounds_max": (0.045, 0.03, 0.04)},
    }
    case = pose((3, -2, 0.8), Rotation.from_euler("xyz", (0.2, 0.4, 0.8)).as_quat())
    poses = {"case": case, "part": compose(case, pose((0.1200005, 0, 0)))}
    runtime = _runtime_for(scene, records, poses)
    assert runtime._case_contents() == ({"part": [False]}, {"part": [True]})
    poses["case"][3:] *= 1 + 0.9e-4
    poses["part"][3:] *= 1 - 0.9e-4
    assert runtime._case_contents() == ({"part": [False]}, {"part": [True]})


@pytest.mark.parametrize("caller", ("measure", "contents"))
@pytest.mark.parametrize("change", ("bounds", "allowance", "negative_allowance", "nan_allowance"))
def test_every_case_caller_rejects_metadata_mutation_before_any_evaluator_tick(native, caller, change):
    _, scene, records, poses = native
    runtime = _runtime_for(scene, records, poses)
    if change == "bounds":
        records["case"]["affordances"]["interior_bounds"][1][0] += 0.05
    else:
        runtime.layout.case_floor_contact_allowance_m = {
            "allowance": 0.001,
            "negative_allowance": -1e-6,
            "nan_allowance": math.nan,
        }[change]
    with pytest.raises(AssertionError, match="Case interior or floor allowance changed"):
        if caller == "measure":
            runtime.measure_case_containment("body", poses["body"], poses["case"])
        else:
            runtime._case_contents()
