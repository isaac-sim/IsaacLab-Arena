# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Collect datagen-format data while a policy drives the environment.

:class:`DatagenCollector` plugs into ``isaaclab_arena.evaluation.policy_runner``'s
``rollout_policy`` loop via an opt-in ``collector`` argument. After each
environment step it records the same modalities the standalone generator
produces (RGB, depth, normals, semantics, optical/scene flow, dynamic-object
poses) into a single ``dataset.h5`` in the SyntheticScene schema -- using
**dedicated** static cameras that are independent of the policy's own
observation cameras.

It reuses :func:`isaaclab_arena_datagen.pipeline.record_camera_step` and
:func:`~isaaclab_arena_datagen.pipeline.save_dynamic_objects`, so policy-driven
collection and standalone generation capture identical data.

Requirements / limitations:

* The ``SimulationApp`` must be launched with cameras enabled
  (``--enable_cameras``); sensor rendering is impossible otherwise.
* The collector pre-allocates ``num_frames`` datasets up front, so it needs a
  fixed horizon. In ``--num_steps`` mode that is the step count; in
  ``--num_episodes`` mode the policy runner passes the worst case
  (``num_episodes * max_episode_length``) and frames are recorded contiguously
  across episode resets (early-finishing episodes leave trailing frames unused).
* Single environment only (``num_envs == 1``), matching the camera handler.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from isaaclab_arena_datagen.camera_trajectory import CameraViewTrajectory
from isaaclab_arena_datagen.dynamic_object_tracker import DynamicObjectTracker
from isaaclab_arena_datagen.io.hdf5_writer import DatagenHDF5Writer
from isaaclab_arena_datagen.object_registry import ObjectInstanceRegistry
from isaaclab_arena_datagen.pipeline import (
    CameraSetup,
    build_camera_setups,
    record_camera_step,
    resolve_cameras,
    save_dynamic_objects,
)
from isaaclab_arena_datagen.utils.constants import DEFAULT_ROTATION_EPS_RAD, DEFAULT_TRANSLATION_EPS_M


@dataclasses.dataclass
class DatagenCollectorConfig:
    """Configuration for policy-rollout data collection.

    Attributes:
        output_dir: Directory where ``dataset.h5`` is written.
        cameras: Explicit camera trajectories. When ``None`` (default), the
            collector uses the environment class's ``get_default_cameras`` if it
            has one, else falls back to a single default view.
        width: Image width in pixels, shared by all cameras.
        height: Image height in pixels, shared by all cameras.
        dynamic_translation_eps: Per-step translation threshold (metres).
        dynamic_rotation_eps: Per-step rotation threshold (radians).
        mesh_sample_spacing: Mesh surface sample spacing (metres).
    """

    output_dir: str
    cameras: list[CameraViewTrajectory] | None = None
    width: int = 640
    height: int = 480
    dynamic_translation_eps: float = DEFAULT_TRANSLATION_EPS_M
    dynamic_rotation_eps: float = DEFAULT_ROTATION_EPS_RAD
    mesh_sample_spacing: float = 0.01


def _resolve_env_class(env_name: str | None) -> Any | None:
    """Look up a registered environment class by name (``None`` if unavailable)."""
    if env_name is None:
        return None
    from isaaclab_arena.assets.registries import EnvironmentRegistry

    registry = EnvironmentRegistry()
    if registry.is_registered(env_name, ensure_loaded=False):
        return registry.get_component_by_name(env_name)
    return None


class DatagenCollector:
    """Records datagen-format data each step of a policy rollout.

    Build via :meth:`from_env` after the environment is created and reset. Pass
    the instance to ``rollout_policy(..., collector=collector)``; the loop calls
    :meth:`on_step` after every ``env.step`` and :meth:`finalize` once at the end.
    """

    def __init__(
        self,
        camera_setups: list[CameraSetup],
        writer: DatagenHDF5Writer,
        dynamic_tracker: DynamicObjectTracker,
        cfg: DatagenCollectorConfig,
        num_steps: int,
    ) -> None:
        self._camera_setups = camera_setups
        self._writer = writer
        self._dynamic_tracker = dynamic_tracker
        self._cfg = cfg
        self._num_steps = num_steps
        self._step = 0
        self._finalized = False
        self._last_env: Any = None

    @classmethod
    def from_env(
        cls,
        env: Any,
        cfg: DatagenCollectorConfig,
        num_steps: int,
        env_name: str | None = None,
    ) -> DatagenCollector:
        """Spawn dedicated datagen cameras and open the writer for *env*.

        Args:
            env: A built, reset Isaac Lab Arena environment (gym-wrapped).
            cfg: Collector configuration.
            num_steps: Fixed number of steps to record (writer pre-allocation).
            env_name: ``example_environment`` name, used to look up the scene's
                ``get_default_cameras`` when ``cfg.cameras`` is ``None``.

        Returns:
            A ready-to-use :class:`DatagenCollector`.
        """
        assert num_steps is not None and num_steps > 1, "DatagenCollector requires a fixed horizon > 1 frame."

        if cfg.cameras is not None:
            from isaaclab_arena_datagen.utils.camera_utils import validate_camera_configs

            cameras = cfg.cameras
            validate_camera_configs(cameras, num_steps)
        else:
            cameras = resolve_cameras(_resolve_env_class(env_name), num_steps)

        shared_registry = ObjectInstanceRegistry()
        camera_setups = build_camera_setups(cameras, cfg.width, cfg.height, shared_registry)

        writer = DatagenHDF5Writer(
            output_dir=cfg.output_dir,
            sequence_index=0,
            cameras=[(cam.camera_id, cfg.height, cfg.width) for cam in camera_setups],
            num_frames=num_steps,
        )
        dynamic_tracker = DynamicObjectTracker(shared_registry, num_steps=num_steps)

        return cls(camera_setups, writer, dynamic_tracker, cfg, num_steps)

    def on_step(self, env: Any, obs: Any, actions: Any, step_idx: int) -> None:
        """Record all cameras + object poses for the current step.

        Uses an internal contiguous counter rather than *step_idx* so indices
        stay aligned with the pre-allocated datasets. Extra steps beyond
        ``num_steps`` (e.g. if the rollout overruns) are ignored.

        Args:
            env: IsaacLab environment instance.
            obs: Observation dict from ``env.step`` (unused; cameras are dedicated).
            actions: Action tensor (unused; recorded scene state is read from sim).
            step_idx: Rollout step counter (informational only).
        """
        self._last_env = env
        if self._finalized or self._step >= self._num_steps:
            return
        idx = self._step
        for cam in self._camera_setups:
            record_camera_step(
                cam.handler,
                self._writer,
                self._dynamic_tracker,
                env,
                cam.camera_id,
                cam.trajectory,
                idx,
            )
        self._dynamic_tracker.record_step_poses(env, idx)
        self._step += 1

    def finalize(self, env: Any | None = None) -> None:
        """Persist dynamic-object poses + mesh samples and close the file.

        Idempotent: safe to call more than once.

        Args:
            env: IsaacLab environment instance. May be ``None`` if the rollout
                already supplied it to :meth:`on_step`; in that case the last
                seen env is reused.
        """
        if self._finalized:
            return
        self._finalized = True
        save_dynamic_objects(
            env if env is not None else self._last_env,
            self._writer,
            self._dynamic_tracker,
            self._cfg.dynamic_translation_eps,
            self._cfg.dynamic_rotation_eps,
            self._cfg.mesh_sample_spacing,
        )
        self._writer.close()
