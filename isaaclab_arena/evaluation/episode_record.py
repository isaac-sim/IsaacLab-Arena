# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Episode record schema for the Agentic Autograder (Phase 1).

Phase 1 writes one EpisodeRecord per eval job. Future phases will produce
per-episode records once individual episode tracking is wired into the
rollout loop.

ModelOutput, SubtaskPhase, and EpisodeTrace are defined here as stubs so
code in later phases can import them from a stable location.
"""

import dataclasses
import json
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Any

SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FailureCategory(str, Enum):
    """Root-cause taxonomy for episode failures."""

    GRASP_MISS = "grasp_miss"
    GRASP_SLIP = "grasp_slip"
    WRONG_OBJECT = "wrong_object"
    WRONG_DESTINATION = "wrong_destination"
    COLLISION = "collision"
    TIMEOUT = "timeout"
    APPROACH_FAILURE = "approach_failure"
    STAGNATION = "stagnation"
    RECOVERY_FAILED = "recovery_failed"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Sub-dataclasses (stubs populated by Phase 3+ components)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SubtaskPhase:
    """A single phase within a task episode."""

    name: str
    start_timestep: float
    end_timestep: float
    completed: bool
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SubtaskPhase":
        return cls(**data)


@dataclasses.dataclass
class ModelOutput:
    """Output from one VLM analysis component on one episode.

    All fields are optional so Phase 1 records can coexist with fully
    populated Phase 3+ records.
    """

    model_id: str
    model_version: str = ""
    prompt_version: str = ""
    progress_score: float | None = None
    progress_label: int | None = None
    progress_per_frame: list[float] | None = None
    success_judgment: bool | None = None
    failure_detected: bool | None = None
    failure_timestep: float | None = None
    failure_description: str | None = None
    failure_category: str | None = None
    subtask_phases: list[SubtaskPhase] = dataclasses.field(default_factory=list)
    key_frame_timestamps: list[float] = dataclasses.field(default_factory=list)
    confidence: float | None = None
    uncertainty_notes: str | None = None
    raw_output: dict = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["subtask_phases"] = [p.to_dict() for p in self.subtask_phases]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ModelOutput":
        data = dict(data)  # don't mutate the caller's dict
        phases_raw = data.pop("subtask_phases", [])
        known = {f.name for f in dataclasses.fields(cls)}
        obj = cls(**{k: v for k, v in data.items() if k in known})
        obj.subtask_phases = [SubtaskPhase.from_dict(p) for p in phases_raw]
        return obj


# ---------------------------------------------------------------------------
# EpisodeRecord
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class EpisodeRecord:
    """Structured record for one eval job run.

    In Phase 1 this is one record per job. Future phases will split this
    into one record per individual episode (episode_id will then be
    {job_name}_{seed}_{env_idx}).
    """

    # Identity
    episode_id: str
    job_name: str
    policy_type: str
    policy_config: dict
    arena_env_args: list[str]
    num_envs: int

    # Simulation length
    num_steps: int | None
    num_episodes: int | None

    # Task
    language_instruction: str | None

    # Artifacts
    video_paths: list[str]
    hdf5_path: str | None

    # Outcome
    status: str
    success: bool | None
    scalar_metrics: dict[str, Any]

    # Timing
    wall_time_seconds: float | None

    # Metadata
    created_at: str
    schema_version: str = SCHEMA_VERSION

    # Extended task identity (None for records written before these fields were added)
    task_name: str | None = None
    embodiment: str | None = None
    env_params: dict | None = None
    seed: int | None = None
    sensitivity_sweep_params: dict | None = None

    # Extended timing
    episode_length_steps: int | None = None
    step_dt: float | None = None

    # Per-episode boundaries within the job video (one entry per completed episode).
    # Each dict: {env_idx: int, start_step: int, end_step: int} (0-indexed, inclusive).
    # Frame index in CameraObsVideoRecorder output equals step index, so these can be
    # used to slice the job video into per-episode clips.
    episode_boundaries: list[dict] = dataclasses.field(default_factory=list)

    # ---------------------------------------------------------------------------

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodeRecord":
        data = dict(data)
        data.setdefault("schema_version", SCHEMA_VERSION)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "EpisodeRecord":
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# EpisodeTrace (stub for Phase 3+)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class EpisodeTrace:
    """Merged per-episode record combining EpisodeRecord with VLM model outputs.

    Populated by the Autograder dispatcher in Phase 3+.
    """

    episode_record: EpisodeRecord
    model_outputs: list[ModelOutput] = dataclasses.field(default_factory=list)
    consensus_progress: float | None = None
    consensus_success: bool | None = None
    consensus_failure_category: str | None = None
    disagreements: list[dict] = dataclasses.field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    created_at: str = dataclasses.field(default_factory=lambda: _utc_now())

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_video_paths(video_dir: str | None, job_name: str) -> list[str]:
    """Return paths to MP4 files written for this job, sorted."""
    if not video_dir:
        return []
    job_video_dir = os.path.join(video_dir, job_name)
    if not os.path.isdir(job_video_dir):
        return []
    return sorted(
        os.path.join(job_video_dir, f)
        for f in os.listdir(job_video_dir)
        if f.endswith(".mp4")
    )


def _get_hdf5_path(env) -> str | None:
    """Extract HDF5 dataset path from the environment config if available."""
    try:
        import pathlib

        cfg = env.unwrapped.cfg
        if not hasattr(cfg, "recorders") or cfg.recorders is None:
            return None
        return str(
            pathlib.Path(cfg.recorders.dataset_export_dir_path)
            / pathlib.Path(cfg.recorders.dataset_filename + ".hdf5")
        )
    except Exception:
        return None


def build_episode_record(
    job,
    env,
    metrics: dict[str, Any] | None,
    status: str,
    video_dir: str | None = None,
    seed: int | None = None,
    episode_boundaries: list[dict] | None = None,
) -> EpisodeRecord:
    """Build an EpisodeRecord from a completed eval job.

    Args:
        job: Completed Job instance.
        env: The gymnasium environment (may be wrapped with RecordVideo).
        metrics: Scalar metrics returned by rollout_policy(), or None.
        status: Job status string ("completed" or "failed").
        video_dir: Root video directory (args_cli.video_dir) if recording was enabled.
        seed: RNG seed used for this job, if set.

    Returns:
        Populated EpisodeRecord.
    """
    scalar_metrics = dict(metrics) if metrics else {}

    success: bool | None = None
    if "success_rate" in scalar_metrics:
        success = float(scalar_metrics["success_rate"]) > 0.0

    wall_time: float | None = None
    if job.start_time is not None and job.end_time is not None:
        wall_time = round(job.end_time - job.start_time, 3)

    language_instruction = job.language_instruction or getattr(env.unwrapped.cfg, "task_description", None)

    step_dt: float | None = None
    try:
        step_dt = float(env.unwrapped.step_dt)
    except Exception:
        pass

    episode_length_steps: int | None = job.num_steps
    if episode_length_steps is None:
        try:
            episode_length_steps = int(env.unwrapped.max_episode_length)
        except Exception:
            pass

    return EpisodeRecord(
        episode_id=job.name,
        job_name=job.name,
        policy_type=job.policy_type,
        policy_config=dict(job.policy_config_dict),
        arena_env_args=list(job.arena_env_args),
        num_envs=job.num_envs,
        num_steps=job.num_steps,
        num_episodes=job.num_episodes,
        language_instruction=language_instruction,
        video_paths=_find_video_paths(video_dir, job.name),
        hdf5_path=_get_hdf5_path(env),
        status=status,
        success=success,
        scalar_metrics=scalar_metrics,
        wall_time_seconds=wall_time,
        created_at=_utc_now(),
        task_name=job.task_name,
        embodiment=job.embodiment,
        env_params=dict(job.env_params) if job.env_params else None,
        seed=seed,
        episode_length_steps=episode_length_steps,
        step_dt=step_dt,
        episode_boundaries=episode_boundaries if episode_boundaries is not None else [],
    )


def write_episode_record(record: EpisodeRecord, output_dir: str) -> str:
    """Write an EpisodeRecord to a JSON file.

    Args:
        record: The episode record to write.
        output_dir: Directory to write into.

    Returns:
        Absolute path of the written file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{record.episode_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(record.to_json())
    return path
