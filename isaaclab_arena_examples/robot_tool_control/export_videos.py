# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Export camera videos with a pause on the saved terminal observation."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path


def export_episode_videos(session_dir: Path, episode_index: int = 0, hold_seconds: float = 2.0) -> dict:
    """Append the saved terminal image as a pause in each exported camera video.

    Args:
        session_dir: Completed robot command session directory.
        episode_index: Episode whose videos and terminal observation are exported.
        hold_seconds: Display duration of the terminal image, in seconds.

    Returns:
        Export metadata including the output video paths.
    """
    assert math.isfinite(hold_seconds) and hold_seconds > 0, "hold_seconds must be finite and positive"
    assert episode_index >= 0, "episode_index must be nonnegative"
    ffmpeg_executable = shutil.which("ffmpeg")
    assert ffmpeg_executable is not None, "ffmpeg is required to export videos"
    session_dir = session_dir.expanduser().resolve()
    terminal_dir = session_dir / f"episode_{episode_index:03d}_terminal"
    observation = json.loads((terminal_dir / "observation.json").read_text())
    frames_per_second = round(1.0 / observation["step_dt"])
    assert frames_per_second > 0, "Camera frame rate must be positive"
    hold_frames = max(1, math.ceil(hold_seconds * frames_per_second))
    output_dir = session_dir / "presentation_videos"
    output_dir.mkdir(exist_ok=True)
    exported_videos = {}
    for camera_name, image_path in observation["images"].items():
        terminal_image = terminal_dir / Path(image_path).name
        source_video = session_dir / "videos" / f"robot-cam-env0-{camera_name}-episode-{episode_index}.mp4"
        assert source_video.is_file(), f"Missing completed video: {source_video}"
        assert terminal_image.is_file(), f"Missing terminal image: {terminal_image}"
        output_video = output_dir / f"{source_video.stem}-with-final-frame.mp4"
        temporary_video = output_video.with_suffix(".tmp.mp4")
        filters = (
            f"[0:v]fps={frames_per_second},format=yuv420p,setsar=1,setpts=PTS-STARTPTS[episode];"
            f"[1:v]fps={frames_per_second},format=yuv420p,setsar=1,trim=end_frame={hold_frames},"
            "setpts=PTS-STARTPTS[terminal];"
            "[episode][terminal]concat=n=2:v=1:a=0[video]"
        )
        try:
            subprocess.run(
                [
                    ffmpeg_executable,
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(source_video),
                    "-loop",
                    "1",
                    "-framerate",
                    str(frames_per_second),
                    "-i",
                    str(terminal_image),
                    "-filter_complex",
                    filters,
                    "-map",
                    "[video]",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-threads",
                    "2",
                    "-movflags",
                    "+faststart",
                    str(temporary_video),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            temporary_video.replace(output_video)
        finally:
            temporary_video.unlink(missing_ok=True)
        exported_videos[camera_name] = str(output_video)
    metadata = {
        "episode": episode_index,
        "simulation_steps": observation["simulation_steps"],
        "simulation_duration_s": observation["simulation_steps"] * observation["step_dt"],
        "termination": observation["termination"],
        "terminal_image_display_s": hold_frames / frames_per_second,
        "videos": exported_videos,
    }
    (output_dir / f"episode_{episode_index:03d}_export.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    """Export a completed episode without running the simulator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session_dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--hold_seconds", type=float, default=2.0)
    args = parser.parse_args()
    print(json.dumps(export_episode_videos(args.session_dir, args.episode, args.hold_seconds), indent=2))


if __name__ == "__main__":
    main()
