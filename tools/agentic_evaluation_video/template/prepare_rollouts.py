# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Select evaluated episodes, requiring an explicit flag to include failures."""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("evaluations", nargs="+", type=Path)
parser.add_argument("--speed", type=int, default=6)
parser.add_argument("--include-failures", action="store_true")
parser.add_argument("--episodes", nargs="+", type=int)
parser.add_argument("--limit", type=int, default=3)
parser.add_argument("--frames-dir", default="rollout_frames")
args = parser.parse_args()
project = json.loads((ROOT / "project.json").read_text())
assert args.speed > 0
selected = []
for evaluation in args.evaluations:
    for results in sorted(evaluation.glob("*/episode_results_rebuild*.jsonl")):
        rebuild = results.stem.split("rebuild")[-1]
        for line in results.read_text().splitlines():
            episode = json.loads(line)
            if args.episodes is not None and episode["episode_in_env"] not in args.episodes:
                continue
            if not episode["success"] and not args.include_failures:
                continue
            criteria = episode["progress"]["criteria_by_name"]
            assert len(criteria) == project["expected_subtasks"]
            if episode["success"]:
                assert episode["progress"]["all_complete"]
                assert all(value["is_complete"] for value in criteria.values())
            source = (
                results.parent
                / f"robot-cam-rebuild{rebuild}-env{episode['env_id']}-external_camera_2_rgb-episode-{episode['episode_in_env']}.mp4"
            )
            selected.append((source, results, episode))
assert selected, "No matching episodes; failures require --include-failures"
assert len(selected[: args.limit]) == project["rollout_count"], "The edit needs three recorded episodes"
manifest = []
for number, (source, results, episode) in enumerate(selected[: args.limit], 1):
    metadata = json.loads(
        subprocess.check_output([
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(source),
        ])
    )["streams"][0]
    numerator, denominator = map(int, metadata["r_frame_rate"].split("/"))
    source_fps = numerator / denominator
    folder = ROOT / args.frames_dir / f"rollout_{number:02d}"
    assert not folder.exists(), f"Preserve existing frames: {folder}"
    folder.mkdir(parents=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vf",
            f"setpts=PTS/{args.speed},fps=30",
            "-q:v",
            "2",
            "-start_number",
            "0",
            str(folder / "frame_%05d.jpg"),
        ],
        check=True,
    )
    placement_events = [event for event in episode["progress"]["events"] if event["predicate_index"] == 2]
    if episode["success"]:
        assert len(placement_events) == project["expected_subtasks"]
    completion_step = episode["episode_length"] if episode["success"] else None
    lift_events = [event for event in episode["progress"]["events"] if event["predicate_index"] == 1]
    manifest.append({
        "source": str(source.resolve().relative_to(ROOT)),
        "results": str(results.resolve().relative_to(ROOT)),
        "episode": episode["episode_in_env"],
        "env_id": episode["env_id"],
        "success": episode["success"],
        "robot_side": project["rollout_sides"][number - 1],
        "robot_edge": "front" if project["require_front_left"] else None,
        "episode_length": episode["episode_length"],
        "source_fps": source_fps,
        "source_frames": int(metadata["nb_frames"]),
        "speed": args.speed,
        "frames": str(folder.relative_to(ROOT)),
        "frame_count": len(list(folder.glob("frame_*.jpg"))),
        "completion_step": completion_step,
        "lift_step": lift_events[0]["step"] if lift_events else None,
    })
(ROOT / "rollouts.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
