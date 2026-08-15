# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Extract the middle frame from RoboLab exact-scene capture videos."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

_CAMERA_VIDEO_PATTERN = re.compile(r"^env(?P<env>\d+)_(?P<camera>.+)\.mp4$")


def output_filename_for_video(video_path: Path) -> str:
    """Return the collision-free PNG filename represented by a capture video path."""
    task_name = video_path.parent.name
    if video_path.name == "viewport.mp4":
        return f"{task_name}_viewport.png"
    match = _CAMERA_VIDEO_PATTERN.fullmatch(video_path.name)
    assert match is not None, f"Unrecognized capture video filename: '{video_path}'"
    return f"{task_name}_env{match.group('env')}_{match.group('camera')}.png"


def _probe_frame_count(video_path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    assert len(streams) == 1, f"Expected one video stream in '{video_path}', got {len(streams)}"
    value = streams[0].get("nb_read_frames")
    assert isinstance(value, str) and value.isdigit(), f"Could not count frames in '{video_path}'"
    frame_count = int(value)
    assert frame_count > 0, f"Video contains no frames: '{video_path}'"
    return frame_count


def extract_middle_frame(video_path: Path, output_path: Path) -> int:
    """Extract and return the zero-based middle-frame index from one video."""
    frame_index = _probe_frame_count(video_path) // 2
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"select=eq(n\\,{frame_index})",
            "-frames:v",
            "1",
            str(output_path),
        ],
        check=True,
    )
    assert output_path.is_file() and output_path.stat().st_size > 0, f"ffmpeg did not produce '{output_path}'"
    return frame_index


def extract_capture_directory(input_dir: Path, output_dir: Path) -> list[Path]:
    """Extract all capture videos below ``input_dir`` into ``output_dir``."""
    video_paths = sorted(input_dir.rglob("*.mp4"))
    assert video_paths, f"No MP4 videos found below '{input_dir}'"
    destinations = [output_dir / output_filename_for_video(path) for path in video_paths]
    duplicate_destinations = sorted({path for path in destinations if destinations.count(path) > 1})
    assert not duplicate_destinations, f"Multiple videos map to the same PNGs: {duplicate_destinations}"
    existing = [path for path in destinations if path.exists()]
    assert not existing, f"Refusing to overwrite existing PNGs: {existing}"

    output_dir.mkdir(parents=True, exist_ok=True)
    for video_path, output_path in zip(video_paths, destinations, strict=True):
        frame_index = extract_middle_frame(video_path, output_path)
        print(f"{video_path} -> {output_path} (frame {frame_index})")
    return destinations


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Capture root containing one directory per task")
    parser.add_argument("output_dir", type=Path, help="Destination directory for extracted PNGs")
    return parser


def main() -> None:
    """Run middle-frame extraction from command-line arguments."""
    args = _create_argument_parser().parse_args()
    extract_capture_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
