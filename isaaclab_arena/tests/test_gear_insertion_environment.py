# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Newton integration tests for gear insertion assets, placement, and episode resets."""

from contextlib import contextmanager

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

GEAR_NAME = "gear_insertion_medium_gear"
BASE_NAME = "gear_insertion_base"
TARGET_NAME = "medium_gear_target"
CAMERA_NAMES = ("external_camera", "external_camera_2", "wrist_camera")


@contextmanager
def _gear_environment(*, position_randomization_m=None, enable_cameras=False):
    """Build the registered environment with its real assets and deterministic placement seed."""
    import argparse

    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.cli import (
        add_environment_cli_args,
        build_environment_from_cli,
        ensure_environments_registered,
    )

    ensure_environments_registered()
    factory_type = EnvironmentRegistry().get_component_by_name("gear_insertion")
    parser = argparse.ArgumentParser()
    add_environment_cli_args(parser, factory_type)
    argv = [] if position_randomization_m is None else ["--position_randomization_m", str(position_randomization_m)]
    args = parser.parse_args(argv)
    args.enable_cameras = enable_cameras
    args.hdr = None
    arena = build_environment_from_cli(factory_type, args)
    builder = ArenaEnvBuilder(arena, ArenaEnvBuilderCfg(num_envs=2, seed=42, placement_seed=42))
    cfg, kwargs = builder.compose_manager_cfg()
    # No dataset export; the environment still installs its progress-tracking recorder.
    cfg.recorders = {}
    if enable_cameras:
        for name in CAMERA_NAMES:
            camera_cfg = getattr(cfg.scene, name)
            camera_cfg.width = 128
            camera_cfg.height = 96
    env = builder.make_registered(cfg, kwargs)
    try:
        yield env, arena, args.position_randomization_m
    finally:
        env.close()


def _test_gear_insertion_placement_and_partial_reset(simulation_app, position_randomization_m):
    import torch

    from isaaclab.utils.math import subtract_frame_transforms

    from isaaclab_arena_environments.gear_insertion_environment import GEAR_BASE_XY, GEAR_INITIAL_XY

    with _gear_environment(position_randomization_m=position_randomization_m) as (env, arena, offset):
        base = env.unwrapped
        nominal = torch.tensor((GEAR_BASE_XY, GEAR_INITIAL_XY), device=base.device)
        # AtPosition is optimized iteratively and retains small solver residuals.
        position_tolerance = 0.002 if offset == 0 else 1e-4
        samples = []
        target_local = arena.task.insertion_target.initial_pose_relative_to_parent
        expected_offset = torch.tensor(target_local.position_xyz, device=base.device).expand(2, -1)
        with torch.inference_mode():
            # Cross a placement-pool refill as well as the initial layouts.
            for _ in range(7):
                env.reset()
                poses = torch.stack([base.arena_world.get_pose_e(name).clone() for name in (BASE_NAME, GEAR_NAME)])
                assert ((poses[:, :, :2] - nominal[:, None, :]).abs() <= offset + position_tolerance).all()
                samples.append(poses[:, :, :2].clone())
                target = base.arena_world.get_pose_e(TARGET_NAME)
                relative_position, _ = subtract_frame_transforms(
                    poses[0, :, :3], poses[0, :, 3:], target[:, :3], target[:, 3:]
                )
                torch.testing.assert_close(relative_position, expected_offset, atol=1e-5, rtol=0)

                initial_height = poses[1, :, 2].clone()
                action = torch.zeros(env.action_space.shape, device=base.device)
                for _ in range(10):
                    _, _, terminated, truncated, _ = env.step(action)
                    assert not terminated.any() and not truncated.any(), "Initial placement must not end the task."
                gear_pose = base.arena_world.get_pose_e(GEAR_NAME)
                assert torch.isfinite(gear_pose).all()
                assert (gear_pose[:, 2] - initial_height).abs().max() < 0.005, "The gear must remain on the table."

            samples = torch.stack(samples)
            if offset > 0:
                assert ((samples.amax(dim=0) - samples.amin(dim=0)) > offset * 0.25).all()
                offsets = samples - nominal[None, :, None, :]
                assert not torch.allclose(offsets[:, 0], offsets[:, 1]), "Sample each asset independently."
            else:
                torch.testing.assert_close(
                    samples, nominal[None, :, None, :].expand_as(samples), atol=position_tolerance, rtol=0
                )

            previous = {name: base.arena_world.get_pose_e(name).clone() for name in (BASE_NAME, GEAR_NAME, TARGET_NAME)}
            base.reset(env_ids=[0])
            for name, pose in previous.items():
                torch.testing.assert_close(base.arena_world.get_pose_e(name)[1], pose[1], atol=1e-6, rtol=0)
    return True


@pytest.mark.with_newton
@pytest.mark.parametrize("position_randomization_m", [None, 0.0], ids=["randomized", "fixed"])
def test_gear_insertion_placement_and_partial_reset(position_randomization_m):
    assert run_function_with_persistent_simulation_app(
        _test_gear_insertion_placement_and_partial_reset,
        position_randomization_m=position_randomization_m,
    )


def _test_gear_insertion_colliders_and_episode_completion(simulation_app):
    import numpy as np
    import torch

    import omni.usd
    from isaaclab_newton.physics import NewtonManager
    from pxr import Usd, UsdGeom, UsdPhysics

    with _gear_environment() as (env, arena, _):
        base = env.unwrapped
        with torch.inference_mode():
            env.reset()
            stage = omni.usd.get_context().get_stage()
            model = NewtonManager.get_model()
            for env_index in range(base.num_envs):
                root = f"/World/envs/env_{env_index}"
                pad_shapes = [
                    label for label in model.shape_label if label.startswith(root + "/") and "fingertipsstep" in label
                ]
                assert len(pad_shapes) == 2, "Each fingertip must compile to one collider, not a hull decomposition."
                pads = []
                for prim in Usd.PrimRange(stage.GetPrimAtPath(f"{root}/Robot")):
                    if prim.IsA(UsdGeom.Mesh) and "fingertipsstep" in prim.GetName():
                        pads.append(prim)
                        assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "convexHull"
                        assert prim.GetAttribute("mjc:priority").Get() == 2
                        np.testing.assert_allclose(prim.GetAttribute("mjc:solref").Get(), (0.01, 1.0))
                assert len(pads) == 2
                for asset_name, mesh_name in ((BASE_NAME, "base"), (GEAR_NAME, "medium")):
                    prim = stage.GetPrimAtPath(f"{root}/{asset_name}/collisions/factory_gear_{mesh_name}_collision")
                    assert prim.GetAttribute("newton:hydroelasticEnabled").Get() is False
                    assert prim.GetAttribute("mjc:priority").Get() == 1
                    np.testing.assert_allclose(prim.GetAttribute("mjc:solref").Get(), (0.005, 1.0))

            # Seat one gear while leaving the other at its randomized source position.
            gear = base.scene[GEAR_NAME]
            first_env = torch.tensor([0], device=base.device)
            target_pose = base.arena_world.get_pose_w(TARGET_NAME)[[0]].clone()
            gear.write_root_pose_to_sim_index(root_pose=target_pose, env_ids=first_env)
            gear.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros(1, 6, device=base.device), env_ids=first_env
            )
            action = torch.zeros(env.action_space.shape, device=base.device)
            success = False
            required_steps = arena.task.success_criteria.consecutive_success_steps
            for step in range(required_steps + 30):
                _, _, terminated, truncated, _ = env.step(action)
                assert not terminated[1] and not truncated.any()
                if terminated[0]:
                    assert step + 1 >= required_steps, "Insertion must settle before completing."
                    assert base.termination_manager.get_term("success")[0]
                    success = True
                    break
            assert success, "A gear seated on the target peg must complete the task."

            # Automatic reset must clear progress for the new episode.
            assert not base.progress_tracker.is_complete().any()
            assert (
                torch.linalg.vector_norm(
                    base.arena_world.get_pose_w(GEAR_NAME)[0, :2] - base.arena_world.get_pose_w(TARGET_NAME)[0, :2]
                )
                > 0.1
            )

            # A dropped gear must fail independently of the successful insertion path.
            second_env = torch.tensor([1], device=base.device)
            dropped_pose = base.arena_world.get_pose_w(GEAR_NAME)[[1]].clone()
            dropped_pose[:, 2] = arena.task.background_scene.object_min_z - 0.2
            gear.write_root_pose_to_sim_index(root_pose=dropped_pose, env_ids=second_env)
            gear.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros(1, 6, device=base.device), env_ids=second_env
            )
            _, _, terminated, truncated, _ = env.step(action)
            assert terminated.tolist() == [False, True]
            assert not truncated.any()
            assert base.termination_manager.get_term("gear_dropped").tolist() == [False, True]
            assert not base.termination_manager.get_term("success").any()
    return True


@pytest.mark.with_newton
def test_gear_insertion_colliders_and_episode_completion():
    assert run_function_with_persistent_simulation_app(_test_gear_insertion_colliders_and_episode_completion)


def _test_gear_insertion_cameras_survive_episode_reset(simulation_app):
    import torch

    with _gear_environment(enable_cameras=True) as (env, _, _):
        base = env.unwrapped
        with torch.inference_mode():
            for _ in range(3):
                env.reset()
                for _ in range(3):
                    env.step(torch.zeros(env.action_space.shape, device=base.device))
                for name in CAMERA_NAMES:
                    rgb = base.scene[name].data.output["rgb"].torch
                    assert rgb.shape[:3] == (2, 96, 128)
                    assert rgb.shape[-1] in (3, 4)
                    assert torch.isfinite(rgb).all()
                    for frame in rgb:
                        assert frame[..., :3].max() > frame[..., :3].min(), f"Camera {name} returned a blank frame."
    return True


@pytest.mark.with_newton
@pytest.mark.with_cameras
def test_gear_insertion_cameras_survive_episode_reset():
    assert run_function_with_persistent_simulation_app(
        _test_gear_insertion_cameras_survive_episode_reset, enable_cameras=True
    )
