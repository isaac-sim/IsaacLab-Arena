# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Locate and read the per-episode results files written by the episode recorder.

Kept free of simulation dependencies so reporting and analysis tools can read recorded results
without a running SimulationApp.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Matches the per-episode results filename written by EpisodeRecorderManager.write. The Experiment
# Runner writes one file per rebuild (``episode_results_rebuild<R>.jsonl``); the policy runner writes
# one per rank (``episode_results_rank<N>.jsonl``, which carries no rebuild and so maps to rebuild 0).
EPISODE_RESULTS_FILENAME_PATTERN = re.compile(r"^episode_results(?:_rebuild(?P<rebuild>\d+))?(?:_rank\d+)?\.jsonl$")


def parse_episode_results_rebuild_index(filename: str) -> int | None:
    """Return the rebuild index encoded in ``filename``, or ``None`` when it is not a results file.

    Files that carry no rebuild number map to rebuild 0.

    Args:
        filename: Bare filename to parse.
    """
    match = EPISODE_RESULTS_FILENAME_PATTERN.match(filename)
    if match is None:
        return None
    rebuild = match.group("rebuild")
    return 0 if rebuild is None else int(rebuild)


def find_episode_results_files(root: str | Path) -> list[Path]:
    """Return every per-episode results file under ``root``, sorted by path.

    Args:
        root: Directory to search recursively.
    """
    return sorted(
        path for path in Path(root).rglob("*.jsonl") if parse_episode_results_rebuild_index(path.name) is not None
    )


def read_episode_results(path: str | Path) -> list[dict]:
    """Return the records in one per-episode results file, skipping blank lines.

    Args:
        path: Results file to read.
    """
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records
