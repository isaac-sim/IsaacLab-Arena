# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""BURVIK seating, physical support, task ordering, and per-environment reset checks."""

from types import SimpleNamespace

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _fixture():
    import math
    import torch

    from isaaclab.utils.math import combine_frame_transforms
    from pxr import Usd, UsdGeom

    from isaaclab_arena_environments.burvik_assembly.geometry import ASSEMBLY_ORDER, PART_NAMES, attachment_pose

    class Scene(dict):
        num_envs = 2
        device = "cpu"
        deformable_objects = {}
        env_prim_paths = ["/World/envs/env_0", "/World/envs/env_1"]

    class World:
        def get_pose_w(self, name):
            return poses[name]

        def get_pose_e(self, name):
            return poses[name]

        def get_root_linear_velocity_w(self, name):
            return velocity[name][:, :3]

        def get_root_angular_velocity_w(self, name):
            return velocity[name][:, 3:]

    stage = Usd.Stage.CreateInMemory()
    scene = Scene()
    for path in scene.env_prim_paths:
        for name in PART_NAMES:
            UsdGeom.Xform.Define(stage, f"{path}/{name}")
    poses = {"tray": torch.tensor([[0.0, 0.0, 1.17, 0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)}
    velocity = {name: torch.zeros(2, 6) for name in PART_NAMES}
    spec = {"sockets": []}
    for index in range(4):
        angle = index * math.pi / 2
        spec["sockets"].append({
            "seat": [0.162 * math.cos(angle), 0.162 * math.sin(angle), 0.004],
            "axis": [-0.089 * math.cos(angle), -0.089 * math.sin(angle), 0.996],
        })
    attachments = {}
    for name in ASSEMBLY_ORDER:
        p, r = attachment_pose(name, spec)
        attachments[name] = {"position": p.tolist(), "quaternion": r.as_quat().tolist()}
        position, quaternion = combine_frame_transforms(
            poses["tray"][:, :3],
            poses["tray"][:, 3:],
            torch.tensor(p, dtype=torch.float32).expand(2, -1),
            torch.tensor(r.as_quat(), dtype=torch.float32).expand(2, -1),
        )
        poses[name] = torch.cat((position, quaternion), dim=-1)
    for index in range(1, 5):
        force = torch.tensor([[[[0.0, 0.0, 6.2]]]]).repeat(2, 1, 1, 1)
        scene[f"leg_{index}_bench_contact"] = SimpleNamespace(
            data=SimpleNamespace(normal_force_matrix_w=SimpleNamespace(torch=force))
        )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene=scene,
        sim=SimpleNamespace(stage=stage),
        arena_world=World(),
        episode_length_buf=torch.zeros(2, dtype=torch.long),
        step_dt=1.0 / 240.0,
    )
    return env, attachments, poses, velocity


def _test_seating_and_support(_app):
    from isaaclab_arena_environments.burvik_assembly.connections import (
        attachment_enabled,
        clear_connections,
        engage_seated_connections,
    )
    from isaaclab_arena_environments.burvik_assembly.task import installed_part

    env, attachments, poses, velocities = _fixture()
    # Mere pose alignment must never count as retained insertion.
    assert not installed_part(env, "leg_1", **attachments["leg_1"]).any()
    poses["leg_1"][1, 0] += 0.003
    poses["leg_2"][1, 3:] = poses["leg_2"][1, 3:].roll(1)
    engage_seated_connections(env, None, attachments)
    assert attachment_enabled(env, "leg_1").tolist() == [True, False]
    assert attachment_enabled(env, "leg_2").tolist() == [True, False]
    assert not installed_part(env, "handle", **attachments["handle"], require_standing=True).any()
    env.episode_length_buf += 1
    assert installed_part(env, "handle", **attachments["handle"], require_standing=True).tolist() == [True, True]
    # A supported but moving table, or a stationary hovering table, must fail.
    velocities["tray"][0, 0] = 0.1
    for index in range(1, 5):
        env.scene[f"leg_{index}_bench_contact"].data.normal_force_matrix_w.torch[1] = 0
    assert not installed_part(env, "handle", **attachments["handle"], require_standing=True).any()
    clear_connections(env, [0], attachments)
    assert attachment_enabled(env, "handle").tolist() == [False, True]
    return True


def _test_round_pin_symmetry(_app):
    import torch

    from isaaclab.utils.math import quat_mul

    from isaaclab_arena_environments.burvik_assembly.connections import attachment_errors

    env, attachments, poses, _ = _fixture()
    half_turn = torch.tensor([[0.0, 0.0, 1.0, 0.0]]).expand(2, -1)
    for name in ("leg_1", "handle"):
        poses[name][:, 3:] = quat_mul(poses[name][:, 3:], half_turn)
        distance, angle = attachment_errors(env, name, **attachments[name])
        assert (distance < 1e-6).all()
        if name == "leg_1":
            assert (angle < 0.001).all(), "An unthreaded circular pin must allow axial rotation"
        else:
            assert (angle > 3.0).all(), "The two-pin handle still requires its complete orientation"
    return True


def test_round_pin_symmetry():
    assert run_function_with_persistent_simulation_app(_test_round_pin_symmetry)


def _test_order_holds_and_partial_reset(_app):
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
    from isaaclab_arena_environments.burvik_assembly.connections import clear_connections, engage_seated_connections
    from isaaclab_arena_environments.burvik_assembly.task import InstallBurvikPartTask

    env, attachments, poses, _ = _fixture()
    tasks = [InstallBurvikPartTask(name, transform, final=name == "handle") for name, transform in attachments.items()]
    termination = CompositeTaskBase(
        tasks, subtasks_are_sequential=True, desired_subtask_success_state=[True] * 5
    ).get_termination_cfg()
    tracker = ProgressTracker(
        termination.success, 2, "cpu", env=env, subtasks_are_sequential=True, desired_subtask_success_state=[True] * 5
    )

    def step(count):
        for _ in range(count):
            env.episode_length_buf += 1
            tracker.step(env, step_index=env.episode_length_buf)

    # Complete later physical connections first; progress must wait for leg 1.
    poses["leg_1"][:, 0] += 0.01
    engage_seated_connections(env, None, attachments)
    step(30)
    assert not tracker.get_subtask_completion().any()
    poses["leg_1"][:, 0] -= 0.01
    engage_seated_connections(env, None, attachments)
    step(12)
    poses["leg_1"][0, 0] += 0.005
    step(1)
    poses["leg_1"][0, 0] -= 0.005
    step(12)
    assert tracker.get_subtask_completion()[:, 0].tolist() == [False, True]
    step(240)
    assert tracker.is_complete().tolist() == [True, True]
    assert [event.criteria_name for event in tracker.get_events()[0]] == [
        criteria.name for criteria in termination.success
    ]
    # Current final conditions are checked even after a milestone has completed.
    poses["leg_2"][0, 0] += 0.005
    step(1)
    assert tracker.is_complete().tolist() == [False, True]
    clear_connections(env, [0], attachments)
    tracker.reset([0])
    assert not tracker.get_subtask_completion()[0].any()
    assert tracker.get_subtask_completion()[1].all()
    assert tracker.get_events()[0] == []
    return True


def test_seating_and_support():
    assert run_function_with_persistent_simulation_app(_test_seating_and_support)


def test_order_holds_and_partial_reset():
    assert run_function_with_persistent_simulation_app(_test_order_holds_and_partial_reset)


def _test_observed_motion(_app):
    import torch

    from isaaclab.utils.math import quat_from_angle_axis, quat_mul

    from isaaclab_arena_environments.burvik_assembly.task import parts_settled

    env, _, poses, velocities = _fixture()
    assert not parts_settled(env).any(), "A single pose cannot establish rest"
    env.episode_length_buf += 1
    assert parts_settled(env).all()
    # Solver angular residuals alone must not reject unchanged geometry.
    velocities["leg_1"][:, 5] = 0.3
    env.episode_length_buf += 1
    assert parts_settled(env).all()
    env.episode_length_buf += 1
    poses["leg_1"][0, 0] += 0.03 * env.step_dt
    rotation = quat_from_angle_axis(torch.tensor([0.1 * env.step_dt]), torch.tensor([[0.0, 0.0, 1.0]]))
    poses["leg_2"][1:2, 3:] = quat_mul(rotation, poses["leg_2"][1:2, 3:])
    assert not parts_settled(env).any(), "Actual translation and axial rotation both prevent rest"
    assert not parts_settled(env).any(), "Repeated predicate calls must not erase measured motion"
    env.episode_length_buf += 1
    assert parts_settled(env).all()
    poses["handle"][:, 3:] *= -1
    env.episode_length_buf += 1
    assert parts_settled(env).all(), "Quaternion sign is not physical motion"
    env.burvik_motion_measurement.reset([0])
    assert parts_settled(env).tolist() == [False, True]
    env.episode_length_buf += 1
    assert parts_settled(env).all()
    env.episode_length_buf[0] = 0
    assert parts_settled(env).tolist() == [False, True], "An episode restart invalidates its prior sample"
    return True


def test_observed_motion():
    assert run_function_with_persistent_simulation_app(_test_observed_motion)
