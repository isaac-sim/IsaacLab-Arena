# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Strictly read episode-result JSONL records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_episode_records(path: str | Path) -> list[dict[str, Any]]:
    """Read nonempty JSON object records, rejecting malformed input and duplicate keys."""
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line, object_pairs_hook=_unique_json_mapping)
                assert isinstance(record, dict), "Expected a JSON object"
            except (AssertionError, json.JSONDecodeError) as error:
                raise AssertionError(f"{path}, line {line_number}: {error}") from error
            records.append(record)
    return records


def _unique_json_mapping(items: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys instead of silently replacing values."""
    result = dict(items)
    assert len(result) == len(items), "Duplicate key in JSON object"
    return result
