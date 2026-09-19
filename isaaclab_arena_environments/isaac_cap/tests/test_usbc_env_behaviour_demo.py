# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check generated USB-C mating poses and the demo's real success/reset path."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

pytestmark = pytest.mark.isaac_cap


def _test_usbc_demo_geometry(_simulation_app) -> bool:
    import torch
    from types import SimpleNamespace

    from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

    from isaaclab_arena.tasks.predicates.spatial import (
        _relative_axial_distances,
        depth_in_range,
        lateral_in_proximity,
        velocity_below_threshold,
    )
    from isaaclab_arena_environments.isaac_cap.usbc_insertion.usbc_env_behaviour_demo import (
        _MATING_ROTATIONS_XYZW,
        _build_demo_environment,
        _plug_pose_at_depth,
    )

    angles = torch.tensor([0.0, 0.7])
    T_W_R = torch.cat(
        (torch.tensor([[0.0, 0.0, 0.8], [1.5, -0.3, 0.9]]), quat_from_euler_xyz(angles, -angles, angles)), dim=-1
    )
    for variant in ("easy", "medium"):
        task = _build_demo_environment(variant).task
        assert task.plug.scale == (1.0, 1.0, 1.0)
        predicates = task.get_termination_cfg().success[0].predicate_sequence[0].params["predicates"]
        geometry_predicates = [
            predicate
            for predicate in predicates
            if predicate.func in (depth_in_range, lateral_in_proximity, velocity_below_threshold)
        ]
        assert [predicate.func for predicate in geometry_predicates] == [
            depth_in_range,
            lateral_in_proximity,
            velocity_below_threshold,
        ]
        mating = predicates[0].params
        geometry = {key: value for key, value in mating.items() if key not in ("depth_min", "depth_max")}
        q_R_P = torch.tensor([_MATING_ROTATIONS_XYZW[variant]]).expand(2, -1)
        wide_axis_R = quat_apply(q_R_P, torch.tensor([[1.0, 0.0, 0.0]]).expand(2, -1))
        expected_wide_axis_R = torch.tensor([0.0, 0.0, 1.0])
        assert torch.allclose(wide_axis_R, expected_wide_axis_R.expand(2, -1), atol=1.0e-6)
        insertion_axis_R = quat_apply(q_R_P, torch.tensor([[0.0, 0.0, 1.0]]).expand(2, -1))
        assert torch.allclose(insertion_axis_R, torch.tensor(mating["receiver_axis"]).expand(2, -1), atol=1.0e-6)
        target = (mating["depth_min"] + mating["depth_max"]) / 2 if mating["depth_max"] else mating["depth_min"] + 0.001
        for depth in (-0.004, target):
            T_W_P = _plug_pose_at_depth(T_W_R, q_R_P, mating, depth)
            poses = {task.plug.name: T_W_P, task.receiver.name: T_W_R}
            world = SimpleNamespace(
                get_pose_w=poses.__getitem__, get_root_linear_velocity_w=lambda _name: torch.zeros((2, 3))
            )
            env = SimpleNamespace(num_envs=2, arena_world=world)
            measured_depth, lateral = _relative_axial_distances(env, **geometry)
            assert torch.allclose(measured_depth, torch.full((2,), depth), atol=1.0e-6)
            assert torch.allclose(lateral, torch.zeros(2), atol=1.0e-6)
            results = [predicate.func(env, **predicate.params) for predicate in geometry_predicates]
            assert bool(torch.stack(results).all().item()) == (depth == target)
            assert all(bool(result.all().item()) for result in results[1:])
    return True


def test_usbc_demo_geometry() -> None:
    assert run_function_with_persistent_simulation_app(_test_usbc_demo_geometry)


def _test_usbc_demo_success_reset(simulation_app, variant: str) -> bool:
    from isaaclab_arena_environments.isaac_cap.usbc_insertion.usbc_env_behaviour_demo import run_demo

    run_demo(
        simulation_app,
        variant=variant,
        cycles=2,
        pause_steps=1,
        real_time=False,
    )
    return True


@pytest.mark.parametrize("variant", ("medium", "easy"))
def test_usbc_demo_success_reset(variant: str) -> None:
    assert run_function_with_persistent_simulation_app(_test_usbc_demo_success_reset, variant=variant)


def _test_usbc_demo_cli_visualizers(_simulation_app) -> bool:
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    from isaaclab_arena.utils.isaaclab_utils import simulation_app as simulation_app_utils
    from isaaclab_arena_environments.isaac_cap.usbc_insertion import usbc_env_behaviour_demo as demo_module

    for arguments, expected_kit in (
        ([], True),
        (["--viz", "none"], False),
    ):
        app_context = SimpleNamespace(app_launcher=SimpleNamespace(has_window=True))
        context_factory = Mock(return_value=nullcontext(app_context))
        visualizer_factory = Mock()
        with (
            patch.object(sys, "argv", ["usbc_env_behaviour_demo.py", *arguments]),
            patch.object(simulation_app_utils, "SimulationAppContext", context_factory),
            patch.object(demo_module, "run_demo") as run_demo_mock,
            patch.dict(sys.modules, {"isaaclab_visualizers.kit": SimpleNamespace(KitVisualizerCfg=visualizer_factory)}),
        ):
            demo_module.main()
        parsed_args = context_factory.call_args.args[0]
        assert not hasattr(parsed_args, "headless")
        assert visualizer_factory.called == expected_kit
        run_demo_mock.assert_called_once()
        assert "teleop" not in run_demo_mock.call_args.kwargs
        assert "control" not in run_demo_mock.call_args.kwargs
        assert run_demo_mock.call_args.kwargs["visualizer_cfg"] == (
            visualizer_factory.return_value if expected_kit else None
        )
    return True


def test_usbc_demo_cli_visualizers() -> None:
    assert run_function_with_persistent_simulation_app(_test_usbc_demo_cli_visualizers)


@pytest.mark.with_subprocess
@pytest.mark.parametrize("variant", ("easy", "medium"))
def test_usbc_demo_headless_cli(variant: str) -> None:
    from isaaclab_arena.tests.utils.constants import TestConstants
    from isaaclab_arena.tests.utils.subprocess import run_subprocess

    result = run_subprocess(
        [
            TestConstants.python_path,
            f"{TestConstants.arena_environments_dir}/isaac_cap/usbc_insertion/usbc_env_behaviour_demo.py",
            variant,
            "--cycles",
            "1",
            "--pause-steps",
            "1",
            "--no-real-time",
            "--viz",
            "none",
        ],
        capture_output=True,
    )
    for predicate_name in (
        "depth_in_range",
        "lateral_in_proximity",
        "velocity_below_threshold",
        "gripper_released",
        "gripper_distance_from_object_exceeds_threshold",
    ):
        assert predicate_name in result.stdout, result.stdout
    assert "success reset observed" in result.stdout, result.stdout
