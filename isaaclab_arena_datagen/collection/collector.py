# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Collect SyntheticScene datagen data while a policy drives the environment.

:class:`DatagenCollector` plugs into ``rollout_policy`` (policy_runner /
eval_runner) via the opt-in ``collector`` argument, recording the standalone
generator's modalities (RGB, depth, normals, semantics, optical/scene flow,
dynamic-object poses + mesh samples) from dedicated cameras -- independent of the
policy's own observation cameras, and reusing
:mod:`~isaaclab_arena_datagen.pipeline` so the data matches.

One HDF5 file per episode (``episode_NNNN/dataset.h5`` under ``cfg.output_dir``,
trimmed to its frame count). The rollout loop owns episode boundaries: it
disables the env's in-``step()`` auto-reset and calls :meth:`end_episode` before
each explicit reset, so every recorded frame belongs to one settled episode.

Requirements: cameras enabled (``--enable_cameras``); single env (``num_envs == 1``).
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable
from typing import Any

from isaaclab_arena_datagen.camera_trajectory import CameraViewTrajectory
from isaaclab_arena_datagen.dynamic_object_tracker import DynamicObjectTracker
from isaaclab_arena_datagen.io.hdf5_writer import DatagenHDF5Writer, StoredFloatType, episode_output_dir
from isaaclab_arena_datagen.object_registry import ObjectInstanceRegistry
from isaaclab_arena_datagen.pipeline import (
    CameraSetup,
    build_camera_setups,
    record_camera_step,
    resolve_cameras,
    save_dynamic_objects,
)
from isaaclab_arena_datagen.utils.camera_utils import resolve_coord
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
        camera_sampler: Optional callable returning a fresh fixed-length
            :class:`CameraViewTrajectory` list. When set, the cameras are re-aimed
            in place (``set_world_pose``) to a new layout every episode rather than
            staying fixed for the whole job.
        store_normals: Write the normals dataset. Off by default to shrink datasets.
        store_flow3d: Write the 3D scene-flow datasets. Off by default to shrink datasets.
        store_float_type: Storage precision for depth and 2D flow.
        frame_stride: Record every n-th simulated frame (1 keeps every frame). The
            skipped frames still advance physics, so flow spans the full gap.
        color_resolution: Stored (width, height) for color; None keeps the render
            size. depth/flow2d/semantic resolution behave the same for their datasets.
    """

    output_dir: str
    cameras: list[CameraViewTrajectory] | None = None
    width: int = 640
    height: int = 480
    dynamic_translation_eps: float = DEFAULT_TRANSLATION_EPS_M
    dynamic_rotation_eps: float = DEFAULT_ROTATION_EPS_RAD
    mesh_sample_spacing: float = 0.01
    camera_sampler: Callable[[], list[CameraViewTrajectory]] | None = None
    store_normals: bool = False
    store_flow3d: bool = False
    store_float_type: StoredFloatType = StoredFloatType.FLOAT32
    frame_stride: int = 1
    color_resolution: tuple[int, int] | None = None
    depth_resolution: tuple[int, int] | None = None
    flow2d_resolution: tuple[int, int] | None = None
    semantic_resolution: tuple[int, int] | None = None


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
        assert cfg.frame_stride >= 1, f"frame_stride must be >= 1, got {cfg.frame_stride}"
        self._camera_setups = camera_setups
        self._registry = registry
        self._cfg = cfg
        self._capacity = capacity

        self._episode_idx = 0
        self._local = 0  # frames recorded in the current episode
        self._substep = 0  # simulated steps seen in the current episode (for frame_stride)
        self._episode_open = False
        self._closed = False
        self._writer: DatagenHDF5Writer | None = None
        self._tracker: DynamicObjectTracker | None = None
        self._last_env: Any = None
        self.sequences: list[dict] = []
        """Plain-dict summary of each closed episode (consumed by the manifest writer)."""

    @property
    def config(self) -> DatagenCollectorConfig:
        """The configuration this collector was built with."""
        return self._cfg

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

        if cfg.camera_sampler is not None:
            # Per-episode randomisation: the sampler defines the camera count/lens;
            # this initial layout is re-sampled and re-posed at every episode start.
            cameras = cfg.camera_sampler()
        elif cfg.cameras is not None:
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
        """Open a fresh writer + tracker and reset flow for a new episode.

        Cameras are re-aimed by :meth:`resample_cameras` before the episode's reset, not
        here, so the reset's RTX rerenders flush the new poses into the camera buffers.
        """
        self._writer = DatagenHDF5Writer(
            output_dir=episode_output_dir(self._cfg.output_dir, self._episode_idx),
            sequence_index=0,
            cameras=[(cam.camera_id, self._cfg.height, self._cfg.width) for cam in self._camera_setups],
            num_frames=self._capacity,
            store_normals=self._cfg.store_normals,
            store_flow3d=self._cfg.store_flow3d,
            store_float_type=self._cfg.store_float_type,
            color_resolution=self._cfg.color_resolution,
            depth_resolution=self._cfg.depth_resolution,
            flow2d_resolution=self._cfg.flow2d_resolution,
            semantic_resolution=self._cfg.semantic_resolution,
        )
        self._tracker = DynamicObjectTracker(self._registry, num_steps=self._capacity)
        for cam in self._camera_setups:
            cam.handler.reset_scene_flow()
        self._local = 0
        self._episode_open = True

    def _reaim_cameras(self, trajectories: list[CameraViewTrajectory]) -> None:
        """Re-pose the existing camera sensors to *trajectories* (look-at) in place."""
        assert len(trajectories) == len(self._camera_setups), (
            f"camera_sampler returned {len(trajectories)} cameras but "
            f"{len(self._camera_setups)} were spawned; the count must stay fixed."
        )
        for cam, traj in zip(self._camera_setups, trajectories):
            cam.handler.set_world_pose(resolve_coord(traj.position, 0), resolve_coord(traj.target, 0))
            cam.trajectory = traj

    def resample_cameras(self) -> None:
        """Re-aim the cameras to a fresh sampled layout for the next episode.

        The rollout loop calls this right before the episode's ``env.reset()`` so the
        reset's RTX rerenders flush the new poses; otherwise the first recorded frame is
        rendered from the previous layout. No-op when no camera_sampler is configured
        (fixed cameras need no re-aim).
        """
        if self._cfg.camera_sampler is not None:
            self._reaim_cameras(self._cfg.camera_sampler())

    def _end_episode(self, env: Any, outcome: str) -> None:
        """Trim, write dynamic objects, append a sequence record, and close the file."""
        assert self._writer is not None and self._tracker is not None
        episode_dir = episode_output_dir(self._cfg.output_dir, self._episode_idx)
        if self._local == 0:
            # Nothing recorded (e.g. rollout ended immediately); drop the empty file.
            self._writer.close()
        else:
            self._writer.trim(self._local)
            self._tracker.trim(self._local)
            result = save_dynamic_objects(
                env,
                self._writer,
                self._tracker,
                self._cfg.dynamic_translation_eps,
                self._cfg.dynamic_rotation_eps,
                self._cfg.mesh_sample_spacing,
            )
            self._writer.close()
            self.sequences.append({
                "episode_index": self._episode_idx,
                "path": os.path.join(episode_dir, "dataset.h5"),
                "num_frames": self._local,
                "camera_ids": [cam.camera_id for cam in self._camera_setups],
                "dynamic_object_names": sorted(result.objects_metadata.keys()),
                "outcome": outcome,
            })
        self._episode_open = False
        self._episode_idx += 1
        self._substep = 0

    # ------------------------------------------------------------------
    # Hooks called by rollout_policy
    # ------------------------------------------------------------------

    def on_step(self, env: Any, obs: Any, actions: Any, step_idx: int) -> None:
        """Record one frame of the current episode (opening it on first use).

        Args:
            env: IsaacLab environment instance.
            obs: Observation dict (unused; cameras are dedicated).
            actions: Action tensor (unused; scene state is read from sim).
            step_idx: Rollout step counter (informational only).
        """
        self._last_env = env
        if self._closed:
            return

        # Only every frame_stride-th simulated step is recorded; the rest still
        # stepped physics, so the next recorded frame's flow spans the whole gap.
        record_this_step = self._substep % self._cfg.frame_stride == 0
        self._substep += 1
        if not record_this_step:
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

    def end_episode(self, env: Any, outcome: str = "timeout") -> None:
        """Flush the in-progress episode file (idempotent).

        Args:
            outcome: Outcome label for the episode, one of ``"success"`` |
                ``"failure"`` | ``"timeout"``. The rollout loop passes the
                classified outcome; the default suits a partial episode flushed
                by :meth:`finalize`.
        """
        if self._closed or not self._episode_open:
            return
        self._end_episode(env, outcome)

    def finalize(self, env: Any | None = None) -> None:
        """Flush the in-progress episode and stop recording. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._episode_open:
            self._end_episode(env if env is not None else self._last_env, "timeout")

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
