# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The filename contract for the recorder's per-episode mp4s.

Kept free of simulation and encoding dependencies, and re-exported from
``camera_observation_video_recorder``, so reporting and analysis tools can identify recorded videos
without importing the recorder itself.

Filename format: ``<name_prefix>-env<N>-<camera_name>-episode-<E>.mp4``, where ``name_prefix`` may
carry a ``-rebuild<R>`` segment.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

# Regular expression to parse the filename of an episode video.
_EPISODE_VIDEO_FILENAME_PATTERN = re.compile(
    r"^(?P<prefix>.+?)(?:-rebuild(?P<rebuild>\d+))?-env(?P<env>\d+)-(?P<camera>.+)-episode-(?P<episode>\d+)\.mp4$"
)


@dataclass
class ParsedEpisodeVideoName:
    """The fields recovered from a recorder mp4 filename by ``parse_episode_video_filename``."""

    prefix: str
    env_index: int
    camera_name: str
    episode_index: int
    rebuild_index: int | None
    """The rebuild this video belongs to, or ``None`` when the prefix carried no ``-rebuild`` segment."""


def sanitize_camera_name(camera_name: str) -> str:
    """Strip path separators so a camera name can't escape video_folder."""
    return camera_name.replace("/", "_").replace(os.sep, "_")


def format_episode_video_filename(name_prefix: str, env_index: int, camera_name: str, episode_index: int) -> str:
    """Build the mp4 filename for one (env, camera, episode). Inverse of ``parse_episode_video_filename``."""
    return f"{name_prefix}-env{env_index}-{sanitize_camera_name(camera_name)}-episode-{episode_index}.mp4"


def parse_episode_video_filename(filename: str) -> ParsedEpisodeVideoName | None:
    """Parse a recorder mp4 filename, or return ``None`` if it does not match the recorder's format."""
    match = _EPISODE_VIDEO_FILENAME_PATTERN.match(filename)
    if match is None:
        return None
    rebuild = match.group("rebuild")
    return ParsedEpisodeVideoName(
        prefix=match.group("prefix"),
        env_index=int(match.group("env")),
        camera_name=match.group("camera"),
        episode_index=int(match.group("episode")),
        rebuild_index=int(rebuild) if rebuild is not None else None,
    )
