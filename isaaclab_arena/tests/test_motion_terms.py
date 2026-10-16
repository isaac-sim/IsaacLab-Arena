# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test the built-in episode recorder terms for object and joint motion."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _velocity_env(velocities):
    """Return a stand-in env whose object velocity is read from the mutable list ``velocities``."""
    import torch
    from types import SimpleNamespace

    return SimpleNamespace(
        num_envs=2,
        device="cpu",
        arena_world=SimpleNamespace(get_mean_linear_velocity_w=lambda name: torch.tensor(velocities[0])),
    )


def _test_object_peak_speed_tracks_the_highest_speed(simulation_app):
    import torch

    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg
    from isaaclab_arena.recording.motion_terms import ObjectPeakSpeed

    velocities = [[[3.0, 4.0, 0.0], [0.0, 0.0, 1.0]]]
    env = _velocity_env(velocities)
    term = ObjectPeakSpeed(EpisodeRecorderTermCfg(func=ObjectPeakSpeed, params={"object_name": "box"}), env)
    term.post_step(env)
    velocities[0] = [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    term.post_step(env)

    assert term(env, 0, object_name="box") == {"object": "box", "peak_linear_speed": 5.0}
    assert term(env, 1, object_name="box") == {"object": "box", "peak_linear_speed": 2.0}
    term.reset(torch.tensor([0]))
    assert term(env, 0, object_name="box")["peak_linear_speed"] == 0.0
    assert term(env, 1, object_name="box")["peak_linear_speed"] == 2.0
    return True


def test_object_peak_speed_tracks_the_highest_speed():
    assert run_function_with_persistent_simulation_app(_test_object_peak_speed_tracks_the_highest_speed)


def _test_object_peak_speed_survives_inference_mode(simulation_app):
    import torch

    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg
    from isaaclab_arena.recording.motion_terms import ObjectPeakSpeed

    env = _velocity_env([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    term = ObjectPeakSpeed(EpisodeRecorderTermCfg(func=ObjectPeakSpeed, params={"object_name": "box"}), env)
    term.reset(None)
    with torch.inference_mode():
        term.post_step(env)
    # rollout_policy() resets outside inference mode after stepping inside it.
    term.reset(torch.tensor([0]))
    assert term(env, 1, object_name="box")["peak_linear_speed"] == 1.0
    return True


def test_object_peak_speed_survives_inference_mode():
    assert run_function_with_persistent_simulation_app(_test_object_peak_speed_survives_inference_mode)


def _test_joint_openness_range_tracks_min_and_max(simulation_app):
    import torch
    from types import SimpleNamespace

    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg
    from isaaclab_arena.recording.motion_terms import JointOpennessRange

    # Values exactly representable in float32, so the recorded floats compare equal.
    openness = [[0.25, 0.75]]
    door = SimpleNamespace(name="microwave", get_openness=lambda env: torch.tensor(openness[0]))
    env = SimpleNamespace(num_envs=2, device="cpu")
    term = JointOpennessRange(EpisodeRecorderTermCfg(func=JointOpennessRange, params={"openable_object": door}), env)
    term.reset(None)
    term.post_step(env)
    openness[0] = [0.5, 0.125]
    term.post_step(env)

    assert term(env, 0, openable_object=door) == {"object": "microwave", "min_openness": 0.25, "max_openness": 0.5}
    assert term(env, 1, openable_object=door) == {"object": "microwave", "min_openness": 0.125, "max_openness": 0.75}
    term.reset(torch.tensor([1]))
    assert term(env, 1, openable_object=door) is None
    return True


def test_joint_openness_range_tracks_min_and_max():
    assert run_function_with_persistent_simulation_app(_test_joint_openness_range_tracks_min_and_max)


def _test_non_finite_motion_values_are_recorded_as_null(simulation_app):
    import json
    import torch
    from types import SimpleNamespace

    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg
    from isaaclab_arena.recording.motion_terms import JointOpennessRange, ObjectPeakSpeed

    nan = float("nan")

    # Env 0 sees a NaN velocity, env 1 stays finite.
    velocities = [[[nan, 0.0, 0.0], [0.0, 2.0, 0.0]]]
    speed_env = _velocity_env(velocities)
    speed_term = ObjectPeakSpeed(EpisodeRecorderTermCfg(func=ObjectPeakSpeed, params={"object_name": "box"}), speed_env)
    speed_term.post_step(speed_env)
    nan_speed_output = speed_term(speed_env, 0, object_name="box")
    assert nan_speed_output == {"object": "box", "peak_linear_speed": None}
    assert speed_term(speed_env, 1, object_name="box") == {"object": "box", "peak_linear_speed": 2.0}
    json.dumps(nan_speed_output, allow_nan=False)

    # Env 0 sees a NaN openness, env 1 stays finite.
    openness = [[nan, 0.5]]
    door = SimpleNamespace(name="microwave", get_openness=lambda env: torch.tensor(openness[0]))
    joint_env = SimpleNamespace(num_envs=2, device="cpu")
    joint_term = JointOpennessRange(
        EpisodeRecorderTermCfg(func=JointOpennessRange, params={"openable_object": door}), joint_env
    )
    joint_term.post_step(joint_env)
    nan_openness_output = joint_term(joint_env, 0, openable_object=door)
    assert nan_openness_output == {"object": "microwave", "min_openness": None, "max_openness": None}
    assert joint_term(joint_env, 1, openable_object=door) == {
        "object": "microwave",
        "min_openness": 0.5,
        "max_openness": 0.5,
    }
    json.dumps(nan_openness_output, allow_nan=False)
    return True


def test_non_finite_motion_values_are_recorded_as_null():
    assert run_function_with_persistent_simulation_app(_test_non_finite_motion_values_are_recorded_as_null)


def _test_pick_then_close_door_suffixes_motion_terms(simulation_app):
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.tasks.close_door_task import CloseDoorTask
    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("kitchen_with_open_drawer")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()
    microwave = asset_registry.get_asset_by_name("microwave")()
    destination = ObjectReference(
        name="destination_location",
        prim_path="{ENV_REGEX_NS}/kitchen_with_open_drawer/Cabinet_B_02",
        parent_asset=background,
    )
    task = CompositeTaskBase(
        [PickAndPlaceTask(cracker_box, destination, background), CloseDoorTask(microwave)],
        subtasks_are_sequential=True,
    )

    assert list(task.get_episode_recorder_terms()) == ["object_motion_subtask_0", "joint_motion_subtask_1"]
    return True


def test_pick_then_close_door_suffixes_motion_terms():
    assert run_function_with_persistent_simulation_app(_test_pick_then_close_door_suffixes_motion_terms)
