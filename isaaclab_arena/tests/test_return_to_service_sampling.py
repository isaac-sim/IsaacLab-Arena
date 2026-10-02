# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check batched workcell measurements against independent object queries on CPU."""

import math
import torch
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


class _MeasuredWorld:
    """Supply measured tensors without starting a simulation or a GPU context."""

    def __init__(self, names: tuple[str, ...], num_envs: int):
        self.poses = {}
        self.linear = {}
        self.angular = {}
        self.velocity_reads = 0
        for name in names:
            self.poses[name] = torch.tensor((0, 0, 0, 0, 0, 0, 1), dtype=torch.float64).repeat(num_envs, 1)
            self.linear[name] = torch.zeros((num_envs, 3), dtype=torch.float64)
            self.angular[name] = torch.zeros((num_envs, 3), dtype=torch.float64)

    def get_pose_w(self, name):
        return self.poses[name]

    def get_pose_e(self, name):
        return self.poses[name]

    def get_root_linear_velocity_w(self, name):
        self.velocity_reads += 1
        return self.linear[name]

    def get_root_angular_velocity_w(self, name):
        self.velocity_reads += 1
        return self.angular[name]


def _runtime(num_envs: int, asset_directory):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.geometry.containment import RegionContainment
    from isaaclab_arena_environments.return_to_service.runtime import ServiceRuntime
    from isaaclab_arena_environments.return_to_service.scene import ServiceLayout

    names = ("dust_cup", "obstruction", "debris_0", "debris_1", "debris_2")
    world = _MeasuredWorld((*names, "case"), num_envs)
    records = {}
    for index, name in enumerate(names):
        # Deliberately heterogeneous, noncentered bounds expose incorrect broadcasting.
        records[name] = {
            "bounds_min": (-0.005, -0.008, -0.01),
            "bounds_max": (0.012 + index * 0.004, 0.009, 0.013),
            "affordances": {},
        }
    records["case"] = {"affordances": {"interior_bounds": ((-0.2, -0.2, -0.1), (0.2, 0.2, 0.1))}}
    records["dust_cup"]["affordances"] = {
        "inlet_bounds": ((-0.02, -0.02, -0.02), (0.02, 0.02, 0.02)),
        "debris_cavity_cylinder": {"x_range": (-0.03, 0.03), "center_yz": (0, 0), "radius": 0.04},
    }
    scene_cfg = SimpleNamespace()
    for name in names:
        path = asset_directory / f"{name}.usda"
        stage = Usd.Stage.CreateNew(str(path))
        root = UsdGeom.Xform.Define(stage, "/Asset")
        stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        cube = UsdGeom.Cube.Define(stage, "/Asset/Shape")
        lower = torch.tensor(records[name]["bounds_min"], dtype=torch.float64)
        upper = torch.tensor(records[name]["bounds_max"], dtype=torch.float64)
        cube.AddTranslateOp().Set(Gf.Vec3d(*((lower + upper) / 2).tolist()))
        cube.AddScaleOp().Set(Gf.Vec3d(*((upper - lower) / 2).tolist()))
        cube.GetPrim().SetCustomDataByKey("arena:feature", "solid")
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        stage.GetRootLayer().Save()
        setattr(scene_cfg, name, SimpleNamespace(spawn=SimpleNamespace(usd_path=str(path), scale=None)))
    runtime = object.__new__(ServiceRuntime)
    runtime.env = SimpleNamespace(
        arena_world=world, num_envs=num_envs, device="cpu", cfg=SimpleNamespace(scene=scene_cfg)
    )
    runtime.layout = SimpleNamespace(
        source_records=records, case_floor_contact_allowance_m=ServiceLayout.case_floor_contact_allowance_m
    )
    runtime._bounds_cache = {}
    runtime._case_interior = None
    runtime._case_overlap_bounds = None
    runtime._case_region_definition = runtime._case_region_configuration()
    runtime.region_containment = RegionContainment(scene_cfg, {})
    runtime._cup_inlet = None
    runtime._settled_cache = None
    runtime._velocity_names = names
    runtime._case_names = names
    return runtime


@pytest.mark.parametrize("num_envs", (1, 5))
def test_batched_case_and_cup_match_individual_measurements(num_envs, tmp_path):
    from isaaclab_arena.geometry.measurements import box_contained, box_overlaps, sphere_overlaps_cylinder
    from isaaclab_arena_environments.return_to_service.connectors import relative_pose

    runtime = _runtime(num_envs, tmp_path)
    world = runtime.env.arena_world
    for env_id in range(num_envs):
        world.poses["case"][env_id, 3:] = torch.tensor((0, 0, math.sin(0.2 * env_id), math.cos(0.2 * env_id)))
        world.poses["dust_cup"][env_id, 3:] = torch.tensor((math.sin(0.3 * env_id), 0, 0, math.cos(0.3 * env_id)))
        world.poses["obstruction"][env_id, 0] = env_id * 0.04
        for index in range(3):
            world.poses[f"debris_{index}"][env_id, :3] = torch.tensor(
                (0.04 * (env_id + index), 0.06 * env_id, 0.08 * index)
            )
    inside, overlap = runtime._case_contents()
    interior = runtime.layout.source_records["case"]["affordances"]["interior_bounds"]
    contact_interior = torch.tensor(interior, dtype=torch.float64)
    contact_interior[0, 2] -= runtime.layout.case_floor_contact_allowance_m
    for name in runtime._case_names:
        pose = relative_pose(world.get_pose_w("case"), world.get_pose_w(name))
        assert inside[name] == box_contained(pose, runtime._bounds(name), contact_interior).tolist()
        assert overlap[name] == box_overlaps(pose, runtime._bounds(name), interior).tolist()
    assert any(inside["dust_cup"])
    assert not any(inside["debris_2"])

    cup = world.get_pose_w("dust_cup")
    geometry = runtime.layout.source_records["dust_cup"]["affordances"]
    blocked = box_overlaps(
        relative_pose(cup, world.get_pose_w("obstruction")), runtime._bounds("obstruction"), geometry["inlet_bounds"]
    )
    cavity = geometry["debris_cavity_cylinder"]
    debris = torch.zeros(num_envs, dtype=torch.bool)
    for name in ("debris_0", "debris_1", "debris_2"):
        debris |= sphere_overlaps_cylinder(
            relative_pose(cup, world.get_pose_w(name)),
            runtime._bounds(name),
            cavity["x_range"],
            cavity["center_yz"],
            cavity["radius"],
        )
    assert runtime._cup_contents() == (blocked.tolist(), debris.tolist())
    assert blocked[0] and debris[0]
    if num_envs > 1:
        assert not blocked[-1] and not debris[-1]


def test_batched_case_floor_contact_keeps_each_environment_boundary_separate(tmp_path):
    runtime = _runtime(2, tmp_path)
    floor = runtime.layout.source_records["case"]["affordances"]["interior_bounds"][0][2]
    bottom = runtime.layout.source_records["obstruction"]["bounds_min"][2]
    poses = runtime.env.arena_world.poses["obstruction"]
    poses[:, 2] = torch.tensor((floor - bottom - 49e-6, floor - bottom - 52e-6))
    inside, overlap = runtime._case_contents()
    assert inside["obstruction"] == [True, False]
    assert overlap["obstruction"] == [True, True]


def test_settled_cache_refreshes_after_partial_reset_without_resetting_other_episode(tmp_path):
    runtime = _runtime(2, tmp_path)
    world = runtime.env.arena_world
    world.linear["debris_0"][0, 0] = 0.03
    world.angular["debris_1"][1, 2] = 0.2
    assert runtime._settled("debris_0") == [False, True]
    assert runtime._settled("debris_1") == [True, False]
    reads = world.velocity_reads
    world.linear["debris_0"][0, 0] = 0
    world.angular["debris_1"][1, 2] = 0
    assert runtime._settled("debris_0") == [False, True]
    assert world.velocity_reads == reads, "One update must reuse the same velocity evidence."

    runtime._case_contents()
    cached_bounds = runtime._bounds_cache[runtime._case_names]
    runtime.cfg = SimpleNamespace(scenario_names=("combined",))
    runtime.scenario_names = ["combined", "combined"]
    runtime.layout.buttons = {}
    runtime.sockets = {}
    case = SimpleNamespace(
        data=SimpleNamespace(default_joint_pos=SimpleNamespace(torch=torch.zeros((2, 2)))),
        write_joint_state_to_sim_index=Mock(),
    )
    runtime.env.scene = {"case": case}
    runtime.models = [object(), object()]
    runtime.statuses = [object(), object()]
    runtime.snapshots = [object(), object()]
    runtime._last_steps = [12, 18]
    runtime._power_remaining = [0.3, 0.7]
    runtime._previous_flow_pressed = [True, True]
    runtime._case_locks = [Mock(), Mock()]
    runtime._show_instrument = Mock()
    untouched = (runtime.models[1], runtime.statuses[1], runtime.snapshots[1])

    runtime.reset([0])
    assert (runtime.models[1], runtime.statuses[1], runtime.snapshots[1]) == untouched
    assert runtime._last_steps == [-1, 18]
    assert runtime._power_remaining == [0, 0.7]
    assert runtime._previous_flow_pressed == [False, True]
    runtime._case_locks[1].GetJointEnabledAttr.assert_not_called()
    assert runtime._settled("debris_0") == [True, True]
    assert runtime._settled("debris_1") == [True, True]
    assert world.velocity_reads == 2 * reads
    assert runtime._bounds_cache[runtime._case_names] is cached_bounds


def test_geometry_caches_keep_reading_current_object_poses(tmp_path):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.geometry.containment import BoxRegion, RegionContainment

    runtime = _runtime(2, tmp_path)
    runtime.layout.regions = {
        "waste": SimpleNamespace(
            center_xyz=(0.1, 0, 0),
            half_extents_xyz=(0.02, 0.15, 0.1),
            rotation_xyzw=(0, 0, math.sqrt(0.5), math.sqrt(0.5)),
        )
    }
    path = tmp_path / "component.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    cube = UsdGeom.Cube.Define(stage, "/Asset/Shape")
    lower, upper = torch.tensor(runtime._bounds("debris_0"), dtype=torch.float64)
    cube.AddTranslateOp().Set(Gf.Vec3d(*((lower + upper) / 2).tolist()))
    cube.AddScaleOp().Set(Gf.Vec3f(*((upper - lower) / 2).tolist()))
    cube.GetPrim().SetCustomDataByKey("arena:feature", "solid")
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    stage.GetRootLayer().Save()
    scene_cfg = runtime.env.cfg.scene
    scene_cfg.debris_0.spawn.usd_path = str(path)
    runtime.region_containment = RegionContainment(scene_cfg)
    scene_cfg.waste = SimpleNamespace(spawn=SimpleNamespace(usd_path=str(path), scale=None))
    runtime.regions = {"waste": BoxRegion("waste", ((-0.02, -0.15, -0.1), (0.02, 0.15, 0.1)))}
    runtime.env.arena_world.poses["waste"] = runtime.env.arena_world.poses["case"].clone()
    runtime.env.arena_world.poses["waste"][:] = torch.tensor(
        [0.1, 0, 0, 0, 0, math.sqrt(0.5), math.sqrt(0.5)], dtype=torch.float64
    )
    assert runtime._case_contents()[0]["obstruction"] == [True, True]
    assert runtime._cup_contents() == ([True, True], [True, True])
    assert runtime._in_region("debris_0", "waste") == [True, True]
    cached_interior = runtime._case_interior
    cached_inlet = runtime._cup_inlet
    cached_region = runtime.regions["waste"]
    cached_geometry = runtime.region_containment._geometry["debris_0"]
    for name in ("obstruction", "debris_0", "debris_1", "debris_2"):
        runtime.env.arena_world.poses[name][1, 0] = 2

    assert runtime._case_contents()[0]["obstruction"] == [True, False]
    assert runtime._cup_contents() == ([True, False], [True, False])
    assert runtime._in_region("debris_0", "waste") == [True, False]
    assert runtime._case_interior is cached_interior
    assert runtime._cup_inlet is cached_inlet
    assert runtime.regions["waste"] is cached_region
    assert runtime.region_containment._geometry["debris_0"] is cached_geometry

    runtime.env.arena_world.poses["waste"][1, 0] += 2
    assert runtime._in_region("debris_0", "waste") == [True, True], "Containment must follow the live fixture."
