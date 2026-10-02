# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise placement composition with real primitive geometry and synthetic measured state."""

import math
import torch
from types import SimpleNamespace

import pytest

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.geometry.containment import BoxRegion, RegionContainment
from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
from isaaclab_arena.tasks.place_in_region_task import PlaceInRegionTask


@pytest.fixture
def measured(tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics

    path = tmp_path / "part.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    cube = UsdGeom.Cube.Define(stage, "/Asset/collider")
    cube.CreateSizeAttr(0.04)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    stage.GetRootLayer().Save()
    poses = {name: torch.tensor([[0, 0, 0, 0, 0, 0, 1]] * 2, dtype=torch.float64) for name in ("part", "bin")}
    velocity = torch.zeros((2, 3), dtype=torch.float64)
    scene_cfg = SimpleNamespace(
        **{name: SimpleNamespace(spawn=SimpleNamespace(usd_path=str(path), scale=None)) for name in poses}
    )
    world = SimpleNamespace(
        get_pose_w=lambda name: poses[name],
        get_root_linear_velocity_w=lambda name: velocity,
        get_root_angular_velocity_w=lambda name: torch.zeros_like(velocity),
        gripper_width=torch.tensor([0.06, 0.06]),
    )
    env = SimpleNamespace(
        cfg=SimpleNamespace(scene=scene_cfg),
        arena_world=world,
        scene=SimpleNamespace(deformable_objects={}),
        num_envs=2,
        device="cpu",
    )
    return env, poses, velocity


def test_live_offset_region_moves_rotates_and_fails_closed_independently(measured):
    from isaaclab.utils.math import quat_apply, quat_mul

    env, poses, _ = measured
    region = BoxRegion(
        "bin", ((-0.05, -0.06, -0.07), (0.05, 0.06, 0.07)), (0.12, 0.03, 0.01), (0, 0, math.sqrt(0.5), math.sqrt(0.5))
    )
    checker = RegionContainment(env.cfg.scene)
    poses["bin"][:] = torch.tensor((3, -2, 0.5, math.sqrt(0.5), 0, 0, math.sqrt(0.5)))
    poses["part"][:, :3] = poses["bin"][:, :3] + quat_apply(
        poses["bin"][:, 3:], poses["bin"].new_tensor(region.position_xyz).expand(2, 3)
    )
    poses["part"][:, 3:] = quat_mul(poses["bin"][:, 3:], poses["bin"].new_tensor(region.rotation_xyzw).expand(2, 4))
    assert checker.measure_region("part", poses["part"], poses["bin"], region).contained.tolist() == [True, True]
    poses["bin"][1, 0] += 0.2
    assert checker.measure_region("part", poses["part"], poses["bin"], region).contained.tolist() == [True, False]
    poses["bin"][1, 3:] *= 2
    poses["part"][1, 3:] *= 0.5
    result = checker.measure_region("part", poses["part"], poses["bin"], region)
    assert result.valid_pose.tolist() == [True, False]
    assert torch.isnan(result.occupied_bounds_R[1]).all()


def test_live_region_rejects_changed_parent_spawn_geometry(measured):
    env, poses, _ = measured
    region = BoxRegion("bin", ((-1, -1, -1), (1, 1, 1)))
    checker = RegionContainment(env.cfg.scene)
    checker.measure_region("part", poses["part"], poses["bin"], region)
    env.cfg.scene.bin.spawn.usd_path = "changed.usda"
    with pytest.raises(AssertionError, match="Region configuration changed"):
        checker.measure_region("part", poses["part"], poses["bin"], region)


class _Gripper:
    def get_opening_width_m(self, world):
        return world.gripper_width


@pytest.mark.parametrize("gripper_name", ("PandaGripper", "RobotiqGripper"))
def test_release_configuration_accepts_immutable_embodiment_grippers(gripper_name):
    from copy import deepcopy

    from isaaclab_arena.embodiments import gripper

    subject = Asset("part")
    subject.object_type = ObjectType.RIGID
    task = PlaceInRegionTask(subject, Asset("bin"), ((-1, -1, -1), (1, 1, 1)), grasp_width_m=0.04)
    task.configure_for_embodiment(SimpleNamespace(gripper=getattr(gripper, gripper_name)()))
    cfg = deepcopy(task.get_termination_cfg())
    assert len(cfg.success) == 1


def test_full_shape_settling_and_release_share_one_interruptible_hold(measured):
    env, poses, velocity = measured
    subject = Asset("part")
    subject.object_type = ObjectType.RIGID
    task = PlaceInRegionTask(
        subject,
        Asset("bin"),
        ((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)),
        grasp_width_m=0.04,
        placement_consecutive_steps=3,
    )
    task.configure_for_embodiment(SimpleNamespace(gripper=_Gripper()))
    cfg = CompositeTaskBase([task], desired_subtask_success_state=[True]).get_termination_cfg()
    tracker = ProgressTracker(
        cfg.success, 2, "cpu", env=env, desired_subtask_success_state=cfg.desired_subtask_success_state
    )
    assert task.get_scene_cfg() is None and task.get_events_cfg() is None
    poses["part"][1, 0] = 0.09  # Center is inside, but the collision shape protrudes.
    for step in range(3):
        tracker.step(env, torch.full((2,), step))
    assert tracker.is_complete().tolist() == [True, False]
    poses["part"][1, 0] = 0
    env.arena_world.gripper_width[0] = 0.03
    tracker.step(env, torch.full((2,), 3))
    assert tracker.is_complete().tolist() == [False, False]
    env.arena_world.gripper_width[0] = 0.06
    velocity[1, 0] = 0.04
    tracker.step(env, torch.full((2,), 4))
    velocity[1] = 0
    tracker.step(env, torch.full((2,), 5))
    assert tracker.is_complete().tolist() == [False, False]
    tracker.step(env, torch.full((2,), 6))
    assert tracker.is_complete().tolist() == [True, False]
    tracker.step(env, torch.full((2,), 7))
    assert tracker.is_complete().tolist() == [True, True]
    tracker.reset([0])
    assert tracker.is_complete().tolist() == [False, True]


@pytest.mark.parametrize(
    "field,value",
    (
        ("floor_allowance_m", -0.1),
        ("rotation_xyzw", (0, 0, 0, 0)),
        ("bounds", ((0, 0, 0), (0, 1, 1))),
        ("position_xyz", (math.nan, 0, 0)),
    ),
)
def test_region_rejects_invalid_static_contract(field, value):
    args = {"parent_name": "bin", "bounds": ((-1, -1, -1), (1, 1, 1)), field: value}
    with pytest.raises(AssertionError):
        BoxRegion(**args)


def test_region_owns_immutable_copies_of_serialized_lists():
    bounds, position = [[-1, -1, -1], [1, 1, 1]], [0, 0, 0]
    region = BoxRegion("bin", bounds, position)
    bounds[0][0] = 10
    position[0] = 10
    assert region.bounds[0][0] == -1 and region.position_xyz[0] == 0


@pytest.mark.parametrize("object_type", [ObjectType.BASE, ObjectType.ARTICULATION, ObjectType.DEFORMABLE])
def test_region_task_rejects_nonrigid_subject_before_runtime(object_type):
    subject = Asset("part")
    subject.object_type = object_type
    with pytest.raises(AssertionError, match="subject must be a rigid object"):
        PlaceInRegionTask(subject, Asset("bin"), ((-1, -1, -1), (1, 1, 1)))
