# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate managed predicate composition and partial-reset lifecycle."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_composite_predicate_lifecycle(_simulation_app) -> bool:
    import torch
    from functools import partial
    from types import SimpleNamespace

    from isaaclab.managers import ManagerTermBase, TerminationManager, TerminationTermCfg

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.composite import CompositePredicate
    from isaaclab_arena.tasks.predicates.consecutive import ConsecutivePredicate
    from isaaclab_arena.tasks.predicates.object_settling import (
        ObjectInitialRestPoseRecorder,
        ObjectsSettledForConsecutiveSteps,
    )
    from isaaclab_arena.tasks.predicates.spatial import depth_in_range, lateral_in_proximity, tilt_axis_aligned
    from isaaclab_arena.tasks.terminations import SuccessMode

    class _PlayingSimulation:
        def is_playing(self) -> bool:
            return True

    class _ResetTrackingPredicate(ManagerTermBase):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.reset_calls = []

        def __call__(self, env):
            return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

        def reset(self, env_ids=None):
            self.reset_calls.append(env_ids)

    class _ArenaWorld:
        def __init__(self):
            self.poses = {
                "receiver": torch.tensor([
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]),
                "subject": torch.tensor([
                    [0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [1.005, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]),
            }
            self.linear_velocity = torch.tensor([
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
            ])
            self.angular_velocity = torch.zeros((2, 3))

        def get_pose_w(self, name: str) -> torch.Tensor:
            return self.poses[name]

        def get_position_w(self, name: str) -> torch.Tensor:
            return self.poses[name][:, :3]

        def get_root_linear_velocity_w(self, _name: str) -> torch.Tensor:
            return self.linear_velocity

        def get_root_angular_velocity_w(self, _name: str) -> torch.Tensor:
            return self.angular_velocity

    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        extras={},
        scene=SimpleNamespace(deformable_objects={}),
        sim=_PlayingSimulation(),
        arena_world=_ArenaWorld(),
        object_initial_rest_pose_recorder=ObjectInitialRestPoseRecorder(num_envs=2, device="cpu"),
    )
    settled_cfg = TerminationTermCfg(
        func=ObjectsSettledForConsecutiveSteps,
        params={
            "object_names": ["subject"],
            "lin_vel_threshold": 0.05,
            "ang_vel_threshold": 0.1,
            "consecutive_steps": 2,
        },
    )
    group_cfg = TerminationTermCfg(
        func=CompositePredicate,
        params={
            "predicates": [
                TerminationTermCfg(
                    func=lateral_in_proximity,
                    params={
                        "subject_name": "subject",
                        "receiver_name": "receiver",
                        "target_offset_xyz": (0.0, 0.0, 0.0),
                        "tolerance_lateral": 0.01,
                    },
                ),
                TerminationTermCfg(
                    func=depth_in_range,
                    params={
                        "subject_name": "subject",
                        "receiver_name": "receiver",
                        "target_offset_xyz": (0.0, 0.0, 0.0),
                        "depth_min": -0.01,
                        "depth_max": 0.01,
                    },
                ),
                TerminationTermCfg(
                    func=tilt_axis_aligned,
                    params={
                        "subject_name": "subject",
                        "receiver_name": "receiver",
                        "max_tilt_rad": 0.1,
                    },
                ),
                settled_cfg,
            ],
            "mode": SuccessMode.ALL,
        },
    )
    manager = TerminationManager({"success": group_cfg}, env)
    resolved_group = manager.get_term_cfg("success").func
    resolved_settled = resolved_group.predicates[-1].func
    assert isinstance(resolved_group, CompositePredicate)
    assert isinstance(resolved_settled, ConsecutivePredicate)

    assert manager.compute().tolist() == [False, False]
    env.arena_world.linear_velocity[:] = 0.0
    assert manager.compute().tolist() == [True, False]

    # Spatial gates remain current; they do not reset the motion counter.
    env.arena_world.poses["subject"][0, 0] = 0.02
    assert manager.compute().tolist() == [False, True]

    # TerminationManager forwards a partial reset through CompositePredicate.
    manager.reset(env_ids=[0])
    env.arena_world.poses["subject"][0, 0] = 0.005
    assert resolved_settled.consecutive_true_steps.tolist() == [0, 2]
    assert manager.compute().tolist() == [False, True]

    # The consecutive window applies to the combined result and resets when any child fails.
    env.first_gate = torch.tensor([True, True])
    env.second_gate = torch.tensor([True, True])

    def _first_gate(env):
        return env.first_gate

    def _second_gate(env):
        return env.second_gate

    combined_manager = TerminationManager(
        {
            "success": TerminationTermCfg(
                func=CompositePredicate,
                params={
                    "predicates": [
                        TerminationTermCfg(func=_first_gate),
                        TerminationTermCfg(func=_second_gate),
                    ],
                    "mode": SuccessMode.ALL,
                    "consecutive_steps": 2,
                },
            )
        },
        env,
    )
    assert combined_manager.compute().tolist() == [False, False]
    env.first_gate[0] = False
    assert combined_manager.compute().tolist() == [False, True]

    # Shared reset logic unwraps configs and partials, resets identities once, and preserves partial env IDs.
    reset_predicate = _ResetTrackingPredicate(TerminationTermCfg(func=_ResetTrackingPredicate), env)
    reset_predicate_partial = partial(reset_predicate)
    reset_group = CompositePredicate(
        TerminationTermCfg(
            func=CompositePredicate,
            params={
                "predicates": [
                    TerminationTermCfg(func=reset_predicate_partial),
                    TerminationTermCfg(func=reset_predicate_partial),
                ]
            },
        ),
        env,
    )
    resolved_reset_predicate = reset_group.predicates[0].func.func
    assert reset_group.predicates[1].func.func is resolved_reset_predicate
    reset_group.reset(torch.tensor([1]))
    assert len(resolved_reset_predicate.reset_calls) == 1
    assert torch.equal(resolved_reset_predicate.reset_calls[0], torch.tensor([1]))

    reset_predicate.reset_calls.clear()
    reset_tracker = ProgressTracker(
        progress_objectives=[
            ProgressObjective(
                name="shared_reset",
                predicate_sequences={"first": [reset_predicate_partial], "second": [reset_predicate_partial]},
            )
        ],
        num_envs=env.num_envs,
        device=env.device,
        env=env,
    )
    reset_tracker.reset(torch.tensor([0]))
    assert len(reset_predicate.reset_calls) == 1
    assert torch.equal(reset_predicate.reset_calls[0], torch.tensor([0]))

    # Progress resolves nested managed configs without changing the reusable task definition.
    nested_group_cfg = TerminationTermCfg(
        func=CompositePredicate,
        params={"predicates": [TerminationTermCfg(func=_first_gate)]},
    )
    nested_tracker = ProgressTracker(
        progress_objectives=[ProgressObjective(name="nested", predicate_sequence=[nested_group_cfg])],
        num_envs=env.num_envs,
        device=env.device,
        env=env,
    )
    assert isinstance(nested_tracker.get_predicate("nested"), CompositePredicate)
    nested_tracker.step(env, step_index=torch.tensor([1, 1]))
    assert nested_tracker.is_complete().tolist() == [False, True]
    assert nested_group_cfg.func is CompositePredicate
    env.first_gate[0] = True
    nested_tracker.step(env, step_index=torch.tensor([2, 2]))
    assert nested_tracker.is_complete().tolist() == [True, True]
    assert combined_manager.compute().tolist() == [False, True]

    # A nested settling counter only advances where its containing sequence is active.
    env.ready = torch.tensor([True, False])

    def _is_ready(env):
        return env.ready

    delayed_tracker = ProgressTracker(
        progress_objectives=[
            ProgressObjective(
                name="delayed_nested_settling",
                predicate_sequence=[
                    _is_ready,
                    TerminationTermCfg(func=CompositePredicate, params={"predicates": [group_cfg]}),
                ],
            )
        ],
        num_envs=env.num_envs,
        device=env.device,
        env=env,
    )
    delayed_composite = delayed_tracker.get_predicate("delayed_nested_settling", predicate_index=1)
    nested_settled = delayed_composite.predicates[0].func.predicates[-1].func
    assert isinstance(nested_settled, ObjectsSettledForConsecutiveSteps)
    delayed_tracker.step(env, step_index=torch.tensor([1, 1]))
    delayed_tracker.step(env, step_index=torch.tensor([2, 2]))
    assert nested_settled.consecutive_true_steps.tolist() == [1, 0]

    env.ready[1] = True
    delayed_tracker.step(env, step_index=torch.tensor([3, 3]))
    assert nested_settled.consecutive_true_steps.tolist() == [2, 0]
    assert delayed_tracker.is_complete().tolist() == [True, False]
    delayed_tracker.step(env, step_index=torch.tensor([4, 4]))
    assert nested_settled.consecutive_true_steps.tolist() == [2, 1]
    delayed_tracker.step(env, step_index=torch.tensor([5, 5]))
    assert delayed_tracker.is_complete().tolist() == [True, True]
    delayed_tracker.reset(torch.tensor([0]))
    assert nested_settled.consecutive_true_steps.tolist() == [0, 2]
    assert delayed_composite.consecutive_true_steps.tolist() == [0, 1]
    assert delayed_tracker.is_complete().tolist() == [False, True]

    # ManagerBase deep-copies configs, so task-build configuration stays declarative.
    assert settled_cfg.func is ObjectsSettledForConsecutiveSteps
    return True


def test_composite_predicate_lifecycle():
    assert run_function_with_persistent_simulation_app(_test_composite_predicate_lifecycle)


def _test_gear_insertion_success_and_diagnostics(_simulation_app) -> bool:
    import torch
    from types import SimpleNamespace

    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.tasks.predicates.composite import CompositePredicate
    from isaaclab_arena.tasks.predicates.spatial import velocity_below_threshold
    from isaaclab_arena_environments.isaac_cap.gear_insertion.task.metrics import _terminal_diagnostics
    from isaaclab_arena_environments.isaac_cap.gear_insertion.task.predicates import GearIsSupported
    from isaaclab_arena_environments.isaac_cap.gear_insertion.task.task import GearInsertionTask

    try:
        GearInsertionTask(
            plate=Asset("plate"),
            gears=[Asset("gear")],
            target_offsets_xyz=[(0.0, 0.0, 0.0)],
            upright_axis_threshold_deg=181.0,
        )
    except ValueError as error:
        assert "must be in (0, 180]" in str(error)
    else:
        raise AssertionError("GearInsertionTask should reject orientation thresholds above 180 degrees.")

    task = GearInsertionTask(
        plate=Asset("plate"),
        gears=[Asset("gear_a"), Asset("gear_b")],
        target_offsets_xyz=[(0.1, 0.0, 0.2), (-0.1, 0.0, 0.2)],
        consecutive_success_steps=3,
    )
    success_objectives = task.get_termination_cfg().success
    assert len(success_objectives) == 1
    assert success_objectives[0].name == "gear_insertion"
    success_cfg = success_objectives[0].predicate_sequence[0]
    assert success_cfg.func is CompositePredicate
    assert success_cfg.params["consecutive_steps"] == 3
    gear_predicates = success_cfg.params["predicates"]
    assert len(gear_predicates) == 2
    assert all(predicate.func is CompositePredicate for predicate in gear_predicates)
    for gear_name, predicate in zip(("gear_a", "gear_b"), gear_predicates, strict=True):
        support = predicate.params["predicates"][-2]
        assert support.func is GearIsSupported
        velocity = predicate.params["predicates"][-1]
        assert velocity.func is velocity_below_threshold
        assert velocity.params == {
            "subject_name": gear_name,
            "linear_velocity_threshold": 0.05,
            "angular_velocity_threshold": 0.5,
        }

    gear_a_gates = torch.tensor([
        [True, False, True],
        [False, True, True],
        [True, True, False],
        [False, False, True],
        [True, False, False],
    ])
    gear_b_gates = torch.tensor([
        [False, True, True],
        [True, False, True],
        [False, True, False],
        [True, True, False],
        [False, False, True],
    ])
    resolved_success = SimpleNamespace(
        results=torch.tensor([[True, False, True], [False, True, True]]),
        predicates=[
            SimpleNamespace(func=SimpleNamespace(results=gear_a_gates)),
            SimpleNamespace(func=SimpleNamespace(results=gear_b_gates)),
        ],
    )
    diagnostics = _terminal_diagnostics(resolved_success, torch.tensor([2, 0]), ("gear_a", "gear_b"))
    assert diagnostics == [
        {
            "gear_a": {
                "success": True,
                "xy": True,
                "z": True,
                "upright": False,
                "support": True,
                "velocity": False,
            },
            "gear_b": {
                "success": True,
                "xy": True,
                "z": True,
                "upright": False,
                "support": False,
                "velocity": True,
            },
        },
        {
            "gear_a": {
                "success": True,
                "xy": True,
                "z": False,
                "upright": True,
                "support": False,
                "velocity": True,
            },
            "gear_b": {
                "success": False,
                "xy": False,
                "z": True,
                "upright": False,
                "support": True,
                "velocity": False,
            },
        },
    ]

    resolved_success.predicates[0].func.results = torch.zeros((4, 3), dtype=torch.bool)
    try:
        _terminal_diagnostics(resolved_success, torch.tensor([0]), ("gear_a", "gear_b"))
    except AssertionError as error:
        assert "expected (5, 3)" in str(error)
    else:
        raise AssertionError("Gear diagnostics should reject malformed child result shapes.")

    resolved_success.predicates[0].func.results = gear_a_gates
    resolved_success.results = torch.zeros((1, 3), dtype=torch.bool)
    try:
        _terminal_diagnostics(resolved_success, torch.tensor([0]), ("gear_a", "gear_b"))
    except AssertionError as error:
        assert "exposes 1 gears" in str(error)
    else:
        raise AssertionError("Gear diagnostics should reject result/name count mismatches.")
    return True


def test_gear_insertion_success_and_diagnostics():
    assert run_function_with_persistent_simulation_app(_test_gear_insertion_success_and_diagnostics)
