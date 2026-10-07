# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Load and extract variation condition overlays."""

from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EpisodeCondition:
    """One ordered episode draw within rebuild conditions."""

    condition_id: str
    """Stable identifier for the source episode."""

    runtime_variations: dict[str, Any]
    """Recorded run-time variation values keyed by ``host.variation``."""


@dataclass
class RebuildConditions:
    """Build-time values and ordered episode conditions for one environment rebuild."""

    build_time_variations: dict[str, Any]
    """Recorded build-time variation values keyed by ``host.variation``."""

    episodes: list[EpisodeCondition]
    """Ordered run-time conditions sourced from episode-result rows."""

    @property
    def num_conditions(self) -> int:
        """Return the number of episode conditions."""
        return len(self.episodes)


def load_episode_conditions_overlay(
    path: str | Path,
    *,
    build_time_variation_keys: Collection[str] = (),
) -> RebuildConditions:
    """Load one episode-result JSONL as rebuild conditions.

    Args:
        path: Episode-result JSONL path.
        build_time_variation_keys: Variation keys that must remain constant across all rows.

    Returns:
        Rebuild-scoped build-time values and ordered episode conditions.
    """
    path = Path(path)
    assert path.suffix.lower() == ".jsonl", f"Episode conditions must be loaded from JSONL: {path}"
    requested_build_time_keys = set(build_time_variation_keys)
    build_time_variations: dict[str, Any] = {}
    build_time_counts = dict.fromkeys(requested_build_time_keys, 0)
    episodes: list[EpisodeCondition] = []
    with path.open(encoding="utf-8") as episode_results:
        for line_number, raw_line in enumerate(episode_results, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record = json.loads(raw_line, object_pairs_hook=_unique_json_mapping)
            assert isinstance(record, dict), f"Line {line_number} in {path} is not a JSON object"
            variations = record.get("variations", {})
            assert isinstance(variations, dict), "variations must be a mapping when present"
            for key, value in variations.items():
                if key not in requested_build_time_keys:
                    continue
                if key in build_time_variations:
                    assert (
                        value == build_time_variations[key]
                    ), f"Build-time variation {key!r} changes across episode records"
                else:
                    build_time_variations[key] = value
                build_time_counts[key] += 1
            episodes.append(
                EpisodeCondition(
                    condition_id=f"condition_{len(episodes):06d}",
                    runtime_variations={
                        key: value for key, value in variations.items() if key not in requested_build_time_keys
                    },
                )
            )

    assert episodes, f"No episode records found in {path}"
    for key in sorted(requested_build_time_keys):
        count = build_time_counts[key]
        assert count == 0 or count == len(
            episodes
        ), f"Build-time variation {key!r} is missing from some episode records"

    return RebuildConditions(
        build_time_variations=build_time_variations,
        episodes=episodes,
    )


def validate_overlay_variation_keys(
    overlay: RebuildConditions,
    enabled_record_keys: set[str],
) -> None:
    """Validate that recorded keys match enabled variations.

    Args:
        overlay: Rebuild conditions to validate.
        enabled_record_keys: Enabled ``host.variation`` keys for the environment build.
    """
    unknown_build = set(overlay.build_time_variations) - enabled_record_keys
    assert (
        not unknown_build
    ), f"Condition overlay lists build-time keys with no enabled variation on this build: {sorted(unknown_build)}"
    runtime_keys: set[str] = set()
    for episode in overlay.episodes:
        runtime_keys.update(episode.runtime_variations)
        unknown_runtime = set(episode.runtime_variations) - enabled_record_keys
        assert (
            not unknown_runtime
        ), f"Condition {episode.condition_id!r} lists runtime keys with no enabled variation: {sorted(unknown_runtime)}"
    overlapping_keys = set(overlay.build_time_variations) & runtime_keys
    assert (
        not overlapping_keys
    ), f"Condition overlay lists variation keys as both build-time and run-time: {sorted(overlapping_keys)}"


def _unique_json_mapping(items: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys instead of silently replacing condition values."""
    result = dict(items)
    assert len(result) == len(items), "Duplicate key in episode condition record"
    return result
