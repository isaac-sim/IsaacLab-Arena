# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Record workbench object-control recipes and convert successful videos to documentation GIFs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

TASKS = ("close_drawer", "close_toolcase", "bin_shelf", "turn_off_scope", "full_cleanup")
REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_DIR = REPO_ROOT / "isaaclab_arena_environments/workbench_cleanup"


def write_demo_environment(name: str, output: Path) -> Path:
    """Save the task's complete scene with a screen-facing view for oscilloscope demonstrations."""
    import yaml

    from isaaclab_arena.environment_spec.arena_env_graph_yaml_loader import load_env_graph_spec_dict

    spec = load_env_graph_spec_dict(ENV_DIR / f"task_{name}.yaml")
    if name in ("turn_off_scope", "full_cleanup"):
        viewer = spec["env_cfg_override"]["viewer"]
        # The shared scene's opposite view hides the oscilloscope screen and power button.
        viewer["eye"] = [1.2, -1.35, 1.35] if name == "turn_off_scope" else [2.05, -1.6, 2.0]
        viewer["lookat"] = [0.03, -0.32, 0.8] if name == "turn_off_scope" else [-0.05, 0.0, 0.95]
    destination = output / "environment.yaml"
    destination.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return destination


def convert_video_to_gif(video: Path, destination: Path) -> None:
    """Write an 800-pixel-wide, 10 FPS GIF at twice the video's playback speed."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-filter_complex",
            (
                "setpts=PTS/2,fps=10,scale=800:-2:flags=lanczos,split[a][b];"
                "[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=3"
            ),
            "-loop",
            "0",
            str(destination),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Mounted directory for recordings and reports.")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--placement-seed", type=int, default=42)
    parser.add_argument("--gif-dir", type=Path, default=REPO_ROOT / "docs/images/workbench_cleanup_assets")
    parser.add_argument("--convert-only", action="store_true", help="Convert existing successful recordings.")
    args = parser.parse_args()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in args.tasks:
        output = output_root / name
        if not args.convert_only:
            assert not output.exists(), f"Output already exists: {output}. Choose a new --output-dir."
            output.mkdir()
            env_spec = write_demo_environment(name, output)
            recipe = output / "commands.jsonl"
            shutil.copyfile(ENV_DIR / "object_control" / f"{name}.jsonl", recipe)
            command = [
                sys.executable,
                "-u",
                str(REPO_ROOT / "isaaclab_arena/agentic_tools/object_control_runner.py"),
                "--env_spec",
                str(env_spec),
                "--commands",
                str(recipe),
                "--output",
                str(output),
                "--placement_seed",
                str(args.placement_seed),
                "--seed",
                "42",
                "--visualizer",
                "kit",
            ]
            print(f"Recording {name}; log: {output / 'run.log'}", flush=True)
            with (output / "run.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((output / "validation.json").read_text(encoding="utf-8"))
        assert report["success"] == [True], f"Task failed: {output / 'validation.json'}"
        gif = args.gif_dir.resolve() / f"{name}.gif"
        convert_video_to_gif(output / "demo.mp4", gif)
        summary[name] = {
            "success": report["success"],
            "subtasks": report["subtasks"],
            "video": str(output / "demo.mp4"),
            "gif": str(gif),
            "report": str(output / "validation.json"),
        }
        print(f"Completed {name}: {gif}", flush=True)
    (output_root / "demo_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
