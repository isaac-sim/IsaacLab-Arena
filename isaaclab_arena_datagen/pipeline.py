# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Shared data-collection pipeline building blocks.

This module holds the per-step recording logic and the setup container used by
**both** entry points:

* the standalone scene generator (:mod:`isaaclab_arena_datagen.run_datagen`),
  which steps the environment with zero actions, and
* the policy-rollout collector
  (:mod:`isaaclab_arena_datagen.collection.collector`), which records the same
  modalities while a policy drives the environment.

Keeping :func:`record_camera_step` / :func:`save_dynamic_objects` here means the
two entry points cannot drift in what they capture.

These functions import Isaac Sim-dependent modules (camera handler, etc.), so
they are only importable once a ``SimulationApp`` is running.
"""

from __future__ import annotations

import dataclasses
import torch
import tqdm
import warnings
from typing import Any

from isaaclab_arena_datagen.camera_handler import IsaacLabArenaCameraHandler, create_static_camera
from isaaclab_arena_datagen.camera_trajectory import CameraViewTrajectory
from isaaclab_arena_datagen.dynamic_object_tracker import DynamicObjectTracker
from isaaclab_arena_datagen.io.hdf5_writer import DatagenHDF5Writer, camera_id_from_index
from isaaclab_arena_datagen.object_registry import ObjectInstanceRegistry
from isaaclab_arena_datagen.utils.camera_utils import DEFAULT_CAMERA, resolve_coord, validate_camera_configs


@dataclasses.dataclass
class CameraSetup:
    """Groups a camera handler with its configuration and folder ID."""

    handler: IsaacLabArenaCameraHandler
    camera_id: str
    trajectory: CameraViewTrajectory


def resolve_cameras(env_class: Any, num_steps: int) -> list[CameraViewTrajectory]:
    """Return the camera trajectories for *env_class*, validated against *num_steps*.

    Falls back to :data:`DEFAULT_CAMERA` if the environment class does not define
    ``get_default_cameras``.

    Args:
        env_class: The datagen environment class (carries ``get_default_cameras``).
        num_steps: Number of simulation steps the trajectories must cover.

    Returns:
        A validated list of :class:`CameraViewTrajectory`.
    """
    if hasattr(env_class, "get_default_cameras"):
        cameras = env_class.get_default_cameras(num_steps)
    else:
        warnings.warn(
            f"Environment {getattr(env_class, 'name', env_class)!r} has no "
            "get_default_cameras(); falling back to DEFAULT_CAMERA.",
            stacklevel=2,
        )
        cameras = [DEFAULT_CAMERA]
    validate_camera_configs(cameras, num_steps)
    return cameras


def build_camera_setups(
    cameras: list[CameraViewTrajectory],
    width: int,
    height: int,
    instance_registry: ObjectInstanceRegistry,
    prim_path_prefix: str = "/World/DatagenCamera",
) -> list[CameraSetup]:
    """Spawn one standalone Isaac Lab camera sensor per trajectory.

    Must be called *after* the environment has been built and reset so the USD
    stage is ready. The cameras are independent of the environment's own
    observation cameras.

    Args:
        cameras: Camera trajectories (static or dynamic).
        width: Image width in pixels, shared by all cameras.
        height: Image height in pixels, shared by all cameras.
        instance_registry: Shared registry so object IDs/colors are consistent
            across all cameras.
        prim_path_prefix: USD prim path prefix; camera ``i`` is spawned at
            ``f"{prim_path_prefix}_{i}"``.

    Returns:
        One :class:`CameraSetup` per trajectory.
    """
    camera_setups: list[CameraSetup] = []
    for cam_idx, cam_cfg in enumerate(cameras):
        handler = create_static_camera(
            position=resolve_coord(cam_cfg.position, 0),
            target=resolve_coord(cam_cfg.target, 0),
            width=width,
            height=height,
            focal_length=cam_cfg.focal_length_mm,
            prim_path=f"{prim_path_prefix}_{cam_idx}",
            instance_registry=instance_registry,
        )
        camera_setups.append(
            CameraSetup(
                handler=handler,
                camera_id=camera_id_from_index(cam_idx),
                trajectory=cam_cfg,
            )
        )
    return camera_setups


@dataclasses.dataclass
class SimDataCollectionSetup:
    """Groups all objects produced by the data-collection setup phase.

    Use the :meth:`from_config` classmethod to build an instance from high-level
    configuration values instead of constructing each object manually.

    Attributes:
        env: IsaacLab environment instance (already reset).
        camera_setups: One :class:`CameraSetup` per camera.
        writer: Dataset writer used to persist all modalities.
        dynamic_tracker: Tracker that accumulates per-step object poses.
    """

    env: Any
    camera_setups: list[CameraSetup]
    writer: DatagenHDF5Writer
    dynamic_tracker: DynamicObjectTracker

    @classmethod
    def from_config(
        cls,
        scene_name: str,
        output_dir: str,
        num_steps: int,
        width: int = 640,
        height: int = 480,
    ) -> SimDataCollectionSetup:
        """Build the full data-collection setup from configuration values.

        Registers the datagen environments, creates the Isaac Lab Arena scene,
        spawns camera sensors, and initialises the dataset writer and
        dynamic-object tracker.

        Args:
            scene_name: Datagen environment name (e.g. ``"ball_box_robot"``).
            output_dir: Directory where ``dataset.h5`` is written.
            num_steps: Number of simulation steps to run.
            width: Image width in pixels, shared by all cameras.
            height: Image height in pixels, shared by all cameras.

        Returns:
            A fully initialised :class:`SimDataCollectionSetup`.
        """
        from isaaclab_arena_datagen.environments import DATAGEN_ENVIRONMENTS, register_datagen_environments
        from isaaclab_arena_environments.cli import (
            get_arena_builder_from_cli,
            get_isaaclab_arena_environments_cli_parser,
        )

        if scene_name not in DATAGEN_ENVIRONMENTS:
            raise ValueError(f"Unknown scene {scene_name!r}. Available: {sorted(DATAGEN_ENVIRONMENTS)}")
        register_datagen_environments()

        arena_parser = get_isaaclab_arena_environments_cli_parser()
        args_cli = arena_parser.parse_args([scene_name])

        arena_builder = get_arena_builder_from_cli(args_cli)
        env = arena_builder.make_registered()
        env.reset()

        cameras = resolve_cameras(DATAGEN_ENVIRONMENTS[scene_name], num_steps)

        shared_registry = ObjectInstanceRegistry()
        camera_setups = build_camera_setups(cameras, width, height, shared_registry)

        writer = DatagenHDF5Writer(
            output_dir=output_dir,
            sequence_index=0,
            cameras=[(cam.camera_id, height, width) for cam in camera_setups],
            num_frames=num_steps,
        )
        dynamic_tracker = DynamicObjectTracker(shared_registry, num_steps=num_steps)

        return cls(
            env=env,
            camera_setups=camera_setups,
            writer=writer,
            dynamic_tracker=dynamic_tracker,
        )


# ---------------------------------------------------------------------------
# Per-step recording (shared by both entry points)
# ---------------------------------------------------------------------------


def record_camera_step(
    handler: IsaacLabArenaCameraHandler,
    writer: DatagenHDF5Writer,
    dynamic_tracker: DynamicObjectTracker,
    env: Any,
    cam_id: str,
    cam_cfg: CameraViewTrajectory,
    step_idx: int,
) -> None:
    """Record all modalities for one camera at one simulation step.

    Writes RGB, depth, intrinsics, extrinsics, normals, and semantic
    segmentation for the current frame, and computes adjacent-frame flow when
    applicable.

    Args:
        handler: Camera handler that provides sensor data and flow computation.
        writer: Dataset writer used to persist each modality to disk.
        dynamic_tracker: Tracker that accumulates per-step object visibility.
        env: IsaacLab environment instance.
        cam_id: String identifier for the camera (e.g. ``"cam0"``).
        cam_cfg: Camera configuration (static or dynamic trajectory).
        step_idx: Zero-based simulation step index.
    """
    # Re-pose only cameras with a dynamic (per-step) position or target.
    if not isinstance(cam_cfg.position, tuple) or not isinstance(cam_cfg.target, tuple):
        handler.set_world_pose(resolve_coord(cam_cfg.position, step_idx), resolve_coord(cam_cfg.target, step_idx))
    # Triggers a render to refresh camera sensor buffers; does not step physics
    # (that would desynchronize multiple cameras).
    handler.update(env.unwrapped.step_dt)

    writer.write_rgb(handler.get_rgb(), cam_id, step_idx)
    writer.write_depth(handler.get_depth(), cam_id, step_idx)
    writer.write_intrinsics(handler.get_intrinsics(), cam_id, step_idx)
    writer.write_extrinsics(handler.get_T_W_from_C(), cam_id, step_idx)
    writer.write_normals(handler.get_normals(), cam_id, step_idx)
    seg_rgba_hw4, semantic_info = handler.get_object_instance_segmentation(env)
    writer.write_semantic_segmentation(seg_rgba_hw4, semantic_info, cam_id, step_idx)
    dynamic_tracker.register_visible_objects(semantic_info)

    _write_adjacent_flow(handler, writer, env, cam_id, step_idx)

    handler.cache_scene_flow_frame(env)


def _write_adjacent_flow(
    handler: IsaacLabArenaCameraHandler,
    writer: DatagenHDF5Writer,
    env: Any,
    cam_id: str,
    step_idx: int,
) -> None:
    """Compute and write adjacent-frame scene flow and optical flow.

    Flow is a delta, so there is nothing to write for the first frame
    (``step_idx == 0``); the function returns early in that case.
    """
    if step_idx == 0:
        return
    prev_idx = step_idx - 1
    flow_result = handler.compute_exact_scene_flow(env)
    scene_flow_W_hw3 = flow_result.scene_flow_W_hw3 if flow_result is not None else None
    optical_flow_hw2 = handler.compute_true_optical_flow(scene_flow_W_hw3)
    if optical_flow_hw2 is not None:
        writer.write_optical_flow(optical_flow_hw2, cam_id, prev_idx)
    if flow_result is not None:
        scene_flow_C_hw3 = handler.convert_scene_flow_W_to_C(flow_result.scene_flow_W_hw3)
        writer.write_scene_flow_3d(
            scene_flow_C_hw3 if scene_flow_C_hw3 is not None else flow_result.scene_flow_W_hw3,
            cam_id,
            prev_idx,
        )
        if flow_result.scene_flow_track_type_hw is not None:
            writer.write_scene_flow_track_type(flow_result.scene_flow_track_type_hw, cam_id, prev_idx)


def run_simulation_loop(
    env: Any,
    camera_setups: list[CameraSetup],
    writer: DatagenHDF5Writer,
    dynamic_tracker: DynamicObjectTracker,
    num_steps: int,
) -> None:
    """Step the environment with zero actions and record every camera each step.

    Used by the standalone scene generator. (The policy-rollout collector drives
    stepping itself and calls :func:`record_camera_step` directly.)

    Args:
        env: IsaacLab environment instance (already reset).
        camera_setups: One :class:`CameraSetup` per camera.
        writer: Dataset writer used to persist all modalities.
        dynamic_tracker: Tracker that accumulates per-step object poses.
        num_steps: Total number of simulation steps to execute.
    """
    dt = env.unwrapped.step_dt
    with torch.inference_mode():
        env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))
        for cam in camera_setups:
            cam.handler.update(dt)

    for step_idx in tqdm.tqdm(range(num_steps)):
        with torch.inference_mode():
            env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))

            for cam in camera_setups:
                record_camera_step(
                    cam.handler,
                    writer,
                    dynamic_tracker,
                    env,
                    cam.camera_id,
                    cam.trajectory,
                    step_idx,
                )

            dynamic_tracker.record_step_poses(env, step_idx)


def save_dynamic_objects(
    env: Any,
    writer: DatagenHDF5Writer,
    dynamic_tracker: DynamicObjectTracker,
    translation_eps_m: float,
    rotation_eps_rad: float,
    mesh_spacing_m: float,
) -> None:
    """Extract and persist dynamic-object poses and mesh samples.

    Identifies objects whose per-step translation exceeds *translation_eps_m* or
    whose per-step rotation exceeds *rotation_eps_rad*, writes their per-step
    poses, and samples their mesh surfaces.

    Args:
        env: IsaacLab environment instance.
        writer: Dataset writer used to persist poses and mesh samples.
        dynamic_tracker: Tracker that has accumulated per-step object poses.
        translation_eps_m: Per-step translation threshold (metres).
        rotation_eps_rad: Per-step rotation threshold (radians).
        mesh_spacing_m: Mesh surface sample spacing in metres.
    """
    result = dynamic_tracker.filter_and_collect_moving_object_poses(
        translation_eps_m=translation_eps_m,
        rotation_eps_rad=rotation_eps_rad,
    )
    writer.write_dynamic_object_poses(result)
    mesh_samples = dynamic_tracker.sample_dynamic_object_meshes(
        env,
        result,
        spacing_m=mesh_spacing_m,
        translation_eps_m=translation_eps_m,
        rotation_eps_rad=rotation_eps_rad,
    )
    writer.write_mesh_samples(mesh_samples)
