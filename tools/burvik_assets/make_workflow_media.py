# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Create the workflow's PNGs and small GIFs on the host using FFmpeg."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "artifacts/ikea_burvik"
MAX_BYTES = 2_000_000


def probe(path: Path) -> dict:
    """Return the first video stream's metadata and container duration."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def fingerprint(path: Path) -> dict:
    """Return byte size and SHA-256 without loading an entire video into memory."""
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def ffmpeg(arguments: list[str]) -> None:
    """Run a bounded-thread FFmpeg conversion and fail on encoding errors."""
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "2", *arguments], check=True)


def make_gif(source: Path, destination: Path, speed: int) -> dict:
    """Convert the complete clip to a looping time-lapse below the commit limit."""
    for width, colors, fps in ((640, 64, 4), (480, 48, 4), (400, 32, 4), (400, 32, 3)):
        filters = (
            f"setpts=(PTS-STARTPTS)/{speed},fps={fps},scale={width}:-2:flags=lanczos,split[a][b];"
            f"[a]palettegen=max_colors={colors}:stats_mode=diff[p];"
            "[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle"
        )
        ffmpeg([
            "-i",
            str(source),
            "-filter_complex_threads",
            "1",
            "-filter_complex",
            filters,
            "-an",
            "-loop",
            "0",
            str(destination),
        ])
        if destination.stat().st_size < MAX_BYTES:
            break
    assert destination.stat().st_size < MAX_BYTES, f"GIF exceeds {MAX_BYTES} bytes: {destination}"
    return {"speedup": speed, "target_fps": fps, "palette_colors": colors, **probe(destination)}


def main() -> None:
    """Convert the three recorded checkpoints and write their media provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standalone", type=Path, default=ARTIFACTS / "results")
    parser.add_argument("--arena", type=Path, default=ARTIFACTS / "arena_camera_recording")
    parser.add_argument("--robot", type=Path, default=ARTIFACTS / "yam_ik_assembly_v10")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs/images/burvik_workflow")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"maximum_file_bytes": MAX_BYTES, "videos": {}, "stills": {}}
    videos = (
        (args.standalone / "burvik_assembly.mp4", "standalone.gif", 4),
        (args.arena / "burvik_arena_viewport.mp4", "arena.gif", 4),
        (args.robot / "yam_top.mp4", "robot_ik.gif", 8),
    )
    for source, name, speed in videos:
        destination = args.output / name
        metadata = make_gif(source, destination, speed)
        manifest["videos"][name] = {
            "source": f"{source.parent.name}/{source.name}",
            "source_file": fingerprint(source),
            "source_metadata": probe(source),
            "output_file": fingerprint(destination),
            "output_metadata": metadata,
        }
        print(f"{name}: {destination.stat().st_size:,} bytes, {speed}x playback")
    stills = (
        (args.standalone / "loose_parts.png", "standalone_loose.png"),
        (args.standalone / "assembled.png", "standalone_assembled.png"),
        (args.arena / "assembled_viewport.png", "arena_assembled.png"),
        (args.robot / "final_top.png", "robot_assembled.png"),
    )
    for source, name in stills:
        destination = args.output / name
        shutil.copyfile(source, destination)
        assert destination.stat().st_size < MAX_BYTES, f"PNG exceeds commit limit: {source}"
        manifest["stills"][name] = {
            "source": f"{source.parent.name}/{source.name}",
            **fingerprint(destination),
        }
    carry_frame = args.output / "robot_turn.png"
    ffmpeg(["-ss", "135", "-i", str(args.robot / "yam_top.mp4"), "-frames:v", "1", str(carry_frame)])
    assert carry_frame.stat().st_size < MAX_BYTES, "Robot turning frame exceeds commit limit"
    manifest["stills"][carry_frame.name] = {"source_video_time_s": 135, **fingerprint(carry_frame)}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
