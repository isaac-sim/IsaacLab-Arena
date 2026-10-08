# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Load and validate recorded variation samples.

A variation record groups samples for a set of variations at one lifecycle scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from isaaclab_arena.recording.episode_results import read_episode_records


@dataclass(frozen=True)
class EpisodeVariationRecord:
    """Record containing samples for a set of variations in one episode."""

    runtime_samples: dict[str, Any]
    """Recorded run-time samples keyed by ``host.variation``."""


@dataclass
class RebuildVariationRecord:
    """Record containing build-time samples and per-episode variation records for one rebuild."""

    build_time_samples: dict[str, Any]
    """Recorded build-time samples keyed by ``host.variation``."""

    episode_records: list[EpisodeVariationRecord]
    """Ordered records, each containing samples for a set of variations in one episode."""

    @property
    def num_recorded_episodes(self) -> int:
        """Return the number of episode variation records."""
        return len(self.episode_records)


def load_rebuild_variation_record(
    path: str | Path,
    *,
    build_time_variation_keys: set[str],
) -> RebuildVariationRecord:
    """Load recorded variation samples from one episode-result JSONL.

    Args:
        path: Episode-result JSONL path.
        build_time_variation_keys: Variation keys that must remain constant across all rows.
            Every recorded key not in this set is treated as a run-time variation.

    Returns:
        One rebuild record containing build-time samples and ordered episode records.
    """
    path = Path(path)
    assert path.suffix.lower() == ".jsonl", f"Recorded variation samples must be loaded from JSONL: {path}"
    build_time_samples: dict[str, Any] = {}
    build_time_counts = dict.fromkeys(build_time_variation_keys, 0)
    samples_per_record: list[dict[str, Any]] = []
    for record in read_episode_records(path):
        record_samples = record.get("variations", {})
        assert isinstance(record_samples, dict), "variations must be a mapping when present"
        samples_per_record.append(record_samples)
        for key, value in record_samples.items():
            if key not in build_time_variation_keys:
                continue
            if key in build_time_samples:
                assert value == build_time_samples[key], f"Build-time variation {key!r} changes across episodes"
            else:
                build_time_samples[key] = value
            build_time_counts[key] += 1

    assert samples_per_record, f"No episode records found in {path}"
    for key in sorted(build_time_variation_keys):
        count = build_time_counts[key]
        assert count == 0 or count == len(
            samples_per_record
        ), f"Build-time variation {key!r} is missing from some episode records"

    episode_records: list[EpisodeVariationRecord] = []
    for record_samples in samples_per_record:
        for key in build_time_samples:
            del record_samples[key]
        episode_records.append(EpisodeVariationRecord(runtime_samples=record_samples))
    return RebuildVariationRecord(
        build_time_samples=build_time_samples,
        episode_records=episode_records,
    )


def load_runtime_variation_samples(path: str | Path, variation_key: str) -> list[Any] | None:
    """Return one runtime sample per record when every record contains ``variation_key``.

    Returns ``None`` when no record contains the variation. Partial presence is
    rejected because a run-time variation must be replayed for every record or
    remain live-sampled for every record.
    """
    variation_record = load_rebuild_variation_record(path, build_time_variation_keys=set())
    presence = [variation_key in episode_record.runtime_samples for episode_record in variation_record.episode_records]
    assert not any(presence) or all(
        presence
    ), f"Run-time variation {variation_key!r} must be present in every source record or none."
    if not any(presence):
        return None
    return [episode_record.runtime_samples[variation_key] for episode_record in variation_record.episode_records]


def validate_recorded_variation_sample_keys(
    samples: RebuildVariationRecord,
    enabled_record_keys: set[str],
) -> None:
    """Validate that recorded sample keys match enabled variations.

    Args:
        samples: Recorded variation samples to validate.
        enabled_record_keys: Enabled ``host.variation`` keys for the environment build.
    """
    unknown_build = set(samples.build_time_samples) - enabled_record_keys
    assert (
        not unknown_build
    ), f"Recorded build-time samples have no enabled variation on this build: {sorted(unknown_build)}"
    runtime_keys: set[str] = set()
    for index, episode_record in enumerate(samples.episode_records):
        runtime_keys.update(episode_record.runtime_samples)
        unknown_runtime = set(episode_record.runtime_samples) - enabled_record_keys
        assert (
            not unknown_runtime
        ), f"Recorded episode at index {index} has samples with no enabled variation: {sorted(unknown_runtime)}"
    overlapping_keys = set(samples.build_time_samples) & runtime_keys
    assert (
        not overlapping_keys
    ), f"Recorded variation keys appear as both build-time and run-time samples: {sorted(overlapping_keys)}"
