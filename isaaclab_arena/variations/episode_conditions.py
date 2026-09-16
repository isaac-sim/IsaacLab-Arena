# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Load and extract episode condition overlays for variation replay."""

from __future__ import annotations

import json
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Placement layouts are not replayed in this phase; strip them when extracting conditions.
_PLACEMENT_VARIATION_KEYS = frozenset({"scene.relation_placement"})

CONDITION_OVERLAY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EpisodeCondition:
    """One ordered episode draw from a condition overlay."""

    condition_id: str
    runtime_variations: dict[str, Any]


@dataclass
class EpisodeConditionsOverlay:
    """Rebuild-scoped build-time values and ordered run-time episode conditions."""

    schema_version: int
    build_time_variations: dict[str, Any]
    episodes: list[EpisodeCondition]
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def num_conditions(self) -> int:
        return len(self.episodes)


def load_episode_conditions_overlay(path: str | Path) -> EpisodeConditionsOverlay:
    """Load a condition overlay YAML file."""
    path = Path(path)
    payload = yaml.safe_load(path.read_text())
    assert isinstance(payload, dict), f"Condition overlay must be a mapping: {path}"
    schema_version = payload.get("schema_version", CONDITION_OVERLAY_SCHEMA_VERSION)
    assert schema_version == CONDITION_OVERLAY_SCHEMA_VERSION, (
        f"Unsupported condition overlay schema {schema_version!r} in {path}; "
        f"expected {CONDITION_OVERLAY_SCHEMA_VERSION}."
    )
    build_time = payload.get("build_time_variations") or {}
    assert isinstance(build_time, dict), "build_time_variations must be a mapping"
    episodes_payload = payload.get("episodes") or []
    assert isinstance(episodes_payload, list), "episodes must be a list"
    episodes: list[EpisodeCondition] = []
    for index, entry in enumerate(episodes_payload):
        assert isinstance(entry, dict), f"episodes[{index}] must be a mapping"
        condition_id = entry.get("condition_id") or f"condition_{index:06d}"
        runtime = entry.get("runtime_variations") or {}
        assert isinstance(runtime, dict), f"episodes[{index}].runtime_variations must be a mapping"
        episodes.append(EpisodeCondition(condition_id=str(condition_id), runtime_variations=dict(runtime)))
    source = payload.get("source") or {}
    assert isinstance(source, dict), "source must be a mapping"
    return EpisodeConditionsOverlay(
        schema_version=schema_version,
        build_time_variations=dict(build_time),
        episodes=episodes,
        source=dict(source),
    )


def _collect_variations_per_line(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    per_line: list[dict[str, Any]] = []
    for record in lines:
        variations = record.get("variations") or {}
        assert isinstance(variations, dict), "variations must be a mapping when present"
        per_line.append({key: value for key, value in variations.items() if key not in _PLACEMENT_VARIATION_KEYS})
    return per_line


def _infer_build_time_variations(per_line: list[dict[str, Any]]) -> dict[str, Any]:
    """Return keys whose value is identical on every episode line."""
    if not per_line:
        return {}
    build_time: dict[str, Any] = {}
    all_keys = set().union(*per_line)
    for key in sorted(all_keys):
        values = [line[key] for line in per_line if key in line]
        if len(values) != len(per_line):
            continue
        first = values[0]
        if all(value == first for value in values[1:]):
            build_time[key] = first
    return build_time


def extract_overlay_from_episode_results(
    episode_results_path: str | Path,
    *,
    source: dict[str, Any] | None = None,
) -> EpisodeConditionsOverlay:
    """Build a condition overlay from one rebuild's episode JSONL."""
    path = Path(episode_results_path)
    lines: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        record = json.loads(raw_line)
        assert isinstance(record, dict), f"Line {line_number} in {path} is not a JSON object"
        lines.append(record)
    assert lines, f"No episode records found in {path}"

    per_line = _collect_variations_per_line(lines)
    build_time_variations = _infer_build_time_variations(per_line)
    episodes: list[EpisodeCondition] = []
    for index, line_variations in enumerate(per_line):
        runtime = {key: value for key, value in line_variations.items() if key not in build_time_variations}
        condition_id = f"condition_{index:06d}"
        episodes.append(EpisodeCondition(condition_id=condition_id, runtime_variations=runtime))

    return EpisodeConditionsOverlay(
        schema_version=CONDITION_OVERLAY_SCHEMA_VERSION,
        build_time_variations=build_time_variations,
        episodes=episodes,
        source=dict(source or {"episode_results": str(path)}),
    )


def overlay_to_yaml_dict(overlay: EpisodeConditionsOverlay) -> dict[str, Any]:
    """Serialize an overlay for YAML output."""
    return {
        "schema_version": overlay.schema_version,
        "source": overlay.source,
        "build_time_variations": overlay.build_time_variations,
        "episodes": [
            {
                "condition_id": episode.condition_id,
                "runtime_variations": episode.runtime_variations,
            }
            for episode in overlay.episodes
        ],
    }


def validate_overlay_variation_keys(
    overlay: EpisodeConditionsOverlay,
    enabled_record_keys: set[str],
) -> None:
    """Assert every overlay key matches an enabled variation on this build."""
    unknown_build = set(overlay.build_time_variations) - enabled_record_keys
    assert (
        not unknown_build
    ), f"Condition overlay lists build-time keys with no enabled variation on this build: {sorted(unknown_build)}"
    for episode in overlay.episodes:
        unknown_runtime = set(episode.runtime_variations) - enabled_record_keys
        assert (
            not unknown_runtime
        ), f"Condition {episode.condition_id!r} lists runtime keys with no enabled variation: {sorted(unknown_runtime)}"
