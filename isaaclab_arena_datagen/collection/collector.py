# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Collect datagen-format data while a policy drives the environment.

:class:`DatagenCollector` plugs into ``rollout_policy`` (used by both
``policy_runner`` and ``eval_runner``) via an opt-in ``collector`` argument.
After each environment step it records the same modalities the standalone
generator produces (RGB, depth, normals, semantics, optical/scene flow,
dynamic-object poses + mesh samples) into nested per-episode ``dataset.h5`` files
in the SyntheticScene schema -- using **dedicated** static cameras that are
independent of the policy's own observation cameras.

**One HDF5 file per episode.** The collector splits the rollout at episode
boundaries and writes ``episode_0000/dataset.h5``,
``episode_0001/dataset.h5``, ... under ``cfg.output_dir``, each trimmed to that
episode's exact frame count. Isaac Lab resets a done env *within* ``step()``
(and re-renders), so the frame observed on a ``done`` step is already the *next*
episode's first frame; the collector
accounts for this so each file contains exactly one episode's frames, with scene
flow reset at each boundary.

It reuses :func:`isaaclab_arena_datagen.pipeline.record_camera_step` and
:func:`~isaaclab_arena_datagen.pipeline.save_dynamic_objects`, so policy-driven
and standalone collection capture identical data.

Requirements / limitations:

* The ``SimulationApp`` must be launched with cameras enabled (``--enable_cameras``).
* Single environment only (``num_envs == 1``), matching the camera handler.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from isaaclab_arena_datagen.camera_trajectory import CameraViewTrajectory
from isaaclab_arena_datagen.dynamic_object_tracker import DynamicObjectTracker
from isaaclab_arena_datagen.io.hdf5_writer import DatagenHDF5Writer, episode_output_dir
from isaaclab_arena_datagen.object_registry import ObjectInstanceRegistry
from isaaclab_arena_datagen.pipeline import (
    CameraSetup,
    build_camera_setups,
    record_camera_step,
    resolve_cameras,
    save_dynamic_objects,
)
from isaaclab_arena_datagen.utils.constants import DEFAULT_ROTATION_EPS_RAD, DEFAULT_TRANSLATION_EPS_M

# Extra capacity over max_episode_length when pre-allocating an episode file,
# guarding against off-by-one between the env's horizon and recorded frames.
_CAPACITY_MARGIN = 2


@dataclasses.dataclass
class DatagenCollectorConfig:
    """Configuration for policy-rollout data collection.

    Attributes:
        output_dir: Directory where per-episode ``episode_NNNN/dataset.h5`` files are written.
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
    """Records datagen-format data each step of a policy rollout, one file per episode.

    Build via :meth:`from_env` after the environment is created. Pass the
    instance to ``rollout_policy(..., collector=collector)``; the loop calls
    :meth:`on_step` after every ``env.step``, :meth:`end_episode` at each episode
    boundary (before its explicit reset), and :meth:`finalize`/:meth:`close` at
    the end.
    """

    def __init__(
        self,
        camera_setups: list[CameraSetup],
        registry: ObjectInstanceRegistry,
        cfg: DatagenCollectorConfig,
        capacity: int,
    ) -> None:
        self._camera_setups = camera_setups
        self._registry = registry
        self._cfg = cfg
        self._capacity = capacity

        self._episode_idx = 0
        self._local = 0  # frames recorded in the current episode
        self._episode_open = False
        self._closed = False
        self._writer: DatagenHDF5Writer | None = None
        self._tracker: DynamicObjectTracker | None = None
        self._last_env: Any = None

    @classmethod
    def from_env(
        cls,
        env: Any,
        cfg: DatagenCollectorConfig,
        env_name: str | None = None,
    ) -> DatagenCollector:
        """Spawn the dedicated datagen cameras for *env* (writers open per episode).

        Args:
            env: A built Isaac Lab Arena environment (gym-wrapped).
            cfg: Collector configuration.
            env_name: ``example_environment`` name, used to look up the scene's
                ``get_default_cameras`` when ``cfg.cameras`` is ``None``.

        Returns:
            A ready-to-use :class:`DatagenCollector`.
        """
        # Per-episode files are sized to the env's max episode length (the upper
        # bound on episode frames) and trimmed to the actual length at close.
        capacity = int(env.unwrapped.max_episode_length) + _CAPACITY_MARGIN

        if cfg.cameras is not None:
            from isaaclab_arena_datagen.utils.camera_utils import validate_camera_configs

            cameras = cfg.cameras
            validate_camera_configs(cameras, capacity)
        else:
            cameras = resolve_cameras(_resolve_env_class(env_name), capacity)

        registry = ObjectInstanceRegistry()
        camera_setups = build_camera_setups(cameras, cfg.width, cfg.height, registry)
        return cls(camera_setups, registry, cfg, capacity)

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def _start_episode(self) -> None:
        """Open a fresh writer + tracker for a new episode and reset flow caches."""
        self._writer = DatagenHDF5Writer(
            output_dir=episode_output_dir(self._cfg.output_dir, self._episode_idx),
            sequence_index=0,
            cameras=[(cam.camera_id, self._cfg.height, self._cfg.width) for cam in self._camera_setups],
            num_frames=self._capacity,
        )
        self._tracker = DynamicObjectTracker(self._registry, num_steps=self._capacity)
        for cam in self._camera_setups:
            cam.handler.reset_scene_flow()
        self._local = 0
        self._episode_open = True

    def _end_episode(self, env: Any) -> None:
        """Trim, write dynamic objects, and close the current episode file."""
        assert self._writer is not None and self._tracker is not None
        if self._local == 0:
            # Nothing recorded (e.g. rollout ended immediately); drop the empty file.
            self._writer.close()
        else:
            self._writer.trim(self._local)
            self._tracker.trim(self._local)
            save_dynamic_objects(
                env,
                self._writer,
                self._tracker,
                self._cfg.dynamic_translation_eps,
                self._cfg.dynamic_rotation_eps,
                self._cfg.mesh_sample_spacing,
            )
            self._writer.close()
        self._episode_open = False
        self._episode_idx += 1

    # ------------------------------------------------------------------
    # Hooks called by rollout_policy
    # ------------------------------------------------------------------

    def on_step(self, env: Any, obs: Any, actions: Any, step_idx: int) -> None:
        """Record one frame of the current episode.

        The rollout loop drives episode boundaries explicitly: for datagen it
        disables the env's in-``step()`` auto-reset and calls :meth:`end_episode`
        immediately before each explicit ``env.reset()``. As a result this method
        only ever sees frames from a single, fully-settled episode, so the
        previous episode's final render can no longer leak into a new episode's
        first frame.

        Args:
            env: IsaacLab environment instance.
            obs: Observation dict from ``env.step`` (unused; cameras are dedicated).
            actions: Action tensor (unused; recorded scene state is read from sim).
            step_idx: Rollout step counter (informational only).
        """
        self._last_env = env
        if self._closed:
            return

        if not self._episode_open:
            self._start_episode()

        if self._local < self._capacity:
            for cam in self._camera_setups:
                record_camera_step(
                    cam.handler,
                    self._writer,
                    self._tracker,
                    env,
                    cam.camera_id,
                    cam.trajectory,
                    self._local,
                )
            self._tracker.record_step_poses(env, self._local)
            self._local += 1

    def end_episode(self, env: Any) -> None:
        """Flush the in-progress episode file (idempotent).

        The rollout loop calls this at an episode boundary, right before its
        explicit ``env.reset()``, so each episode is closed from a settled scene.
        """
        if self._closed or not self._episode_open:
            return
        self._end_episode(env)

    def finalize(self, env: Any | None = None) -> None:
        """Flush the in-progress episode and stop recording. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._episode_open:
            self._end_episode(env if env is not None else self._last_env)

    def close(self, env: Any | None = None) -> None:
        """Finalize, then detach and remove the dedicated datagen cameras.

        Call once the collector is done (e.g. between eval-runner jobs) so its
        cameras' replicator annotators do not leak into the next job. Idempotent.
        """
        try:
            self.finalize(env)
        except Exception as exc:  # pragma: no cover - best-effort during cleanup
            print(f"[datagen] Warning: failed to finalize datagen episode: {exc}")
        for cam in self._camera_setups:
            cam.handler.close()
        self._camera_setups = []
