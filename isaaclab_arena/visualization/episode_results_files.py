# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Locate and parse evaluation result artifacts without importing the runtime stack."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

REPORT_DIRNAME = "report"

# Matches the per-episode results filename written by EpisodeRecorderManager.write.
EPISODE_RESULTS_FILENAME_PATTERN = re.compile(
    r"^episode_results(?:_rebuild(?P<rebuild>\d+))?(?:_rank(?P<rank>\d+))?\.jsonl$"
)

# Matches CameraObsVideoRecorder filenames without importing that module's torch/moviepy deps.
EPISODE_VIDEO_FILENAME_PATTERN = re.compile(
    r"^(?P<prefix>.+?)(?:-rebuild(?P<rebuild>\d+))?-env(?P<env>\d+)-(?P<camera>.+)-episode-(?P<episode>\d+)\.mp4$"
)


@dataclass(frozen=True)
class DataIssue:
    """One recoverable problem found while reading result artifacts."""

    path: str
    message: str


@dataclass(frozen=True)
class ParsedEpisodeResultsName:
    rebuild_index: int
    rank_index: int | None


@dataclass(frozen=True)
class ParsedEpisodeVideoName:
    prefix: str
    rebuild_index: int
    env_index: int
    camera_name: str
    episode_index: int


def parse_episode_results_filename(filename: str) -> ParsedEpisodeResultsName | None:
    """Parse an episode results filename, or return ``None`` when it is not one."""
    match = EPISODE_RESULTS_FILENAME_PATTERN.match(filename)
    if match is None:
        return None
    rebuild = match.group("rebuild")
    rank = match.group("rank")
    return ParsedEpisodeResultsName(
        rebuild_index=0 if rebuild is None else int(rebuild),
        rank_index=None if rank is None else int(rank),
    )


def parse_episode_video_filename(filename: str) -> ParsedEpisodeVideoName | None:
    """Parse a recorder mp4 filename, or return ``None`` if it does not match the recorder format."""
    match = EPISODE_VIDEO_FILENAME_PATTERN.match(filename)
    if match is None:
        return None
    rebuild = match.group("rebuild")
    return ParsedEpisodeVideoName(
        prefix=match.group("prefix"),
        rebuild_index=0 if rebuild is None else int(rebuild),
        env_index=int(match.group("env")),
        camera_name=match.group("camera"),
        episode_index=int(match.group("episode")),
    )


def format_episode_video_filename(name_prefix: str, env_index: int, camera_name: str, episode_index: int) -> str:
    """Build the mp4 filename for one ``(env, camera, episode)`` in tests and small utilities."""
    safe_camera_name = camera_name.replace("/", "_").replace("\\", "_")
    return f"{name_prefix}-env{env_index}-{safe_camera_name}-episode-{episode_index}.mp4"


def _is_under_generated_report(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    return bool(relative.parts) and relative.parts[0] == REPORT_DIRNAME


def find_episode_results_files(root: str | Path) -> list[Path]:
    """Return every per-episode results file under ``root``, excluding generated report pages."""
    root = Path(root)
    return sorted(
        path
        for path in root.rglob("*.jsonl")
        if not _is_under_generated_report(path, root) and parse_episode_results_filename(path.name) is not None
    )


def find_episode_video_files(root: str | Path) -> list[Path]:
    """Return every recorder mp4 under ``root``, excluding generated report pages."""
    root = Path(root)
    return sorted(
        path
        for path in root.rglob("*.mp4")
        if not _is_under_generated_report(path, root) and parse_episode_video_filename(path.name) is not None
    )


def read_episode_results(path: str | Path, root: str | Path | None = None) -> tuple[list[dict], list[DataIssue]]:
    """Return valid JSON object records from one results file plus recoverable read issues."""
    path = Path(path)
    display_path = _display_path(path, root)
    records: list[dict] = []
    issues: list[DataIssue] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], [DataIssue(display_path, f"could not read results file: {exc}")]

    for line_number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(DataIssue(display_path, f"line {line_number}: invalid JSON ({exc.msg})"))
            continue
        if not isinstance(record, dict):
            issues.append(DataIssue(display_path, f"line {line_number}: expected a JSON object"))
            continue
        records.append(record)
    return records, issues


def _display_path(path: Path, root: str | Path | None) -> str:
    if root is None:
        return str(path)
    try:
        return str(path.relative_to(Path(root)))
    except ValueError:
        return str(path)
