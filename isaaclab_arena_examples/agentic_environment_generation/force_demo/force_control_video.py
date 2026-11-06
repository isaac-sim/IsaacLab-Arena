# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Annotate Lab-recorded video and capture force-recipe boundary images."""

import json
from pathlib import Path


def configure_video(env_cfg, output: Path) -> None:
    """Record bounded clips at the environment's control rate using Isaac Lab."""
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    step_dt = env_cfg.sim.dt * env_cfg.decimation
    clip_steps = max(1, round(5.0 / step_dt))
    env_cfg.video_recorders = [
        VideoRecorderCfg(
            source="visualizer:kit",
            output_dir=str(output / "raw_video"),
            output_filename_prefix="viewport",
            fps=1.0 / step_dt,
            video_length=clip_steps,
            video_interval=clip_steps,
        )
    ]


def overlay_progress_banner(frame, progress: dict):
    """Draw recipe progress without changing pixels below the banner."""
    import numpy as np

    from PIL import Image, ImageDraw, ImageFont

    completion = progress["completion"]
    active = progress["index"]
    status_labels = []
    for index, complete in enumerate(completion):
        status = "done" if complete else "active" if index == active else "pending"
        status_labels.append(f"{index + 1}:{status}")
    image = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, 0, image.width - 1, 75), fill=(8, 12, 18, 205))
    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 24)
        progress_font = ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        title_font = ImageFont.load_default()
        progress_font = title_font
    title = progress["title"]
    if active is not None:
        title = f"Subtask {active + 1}/{len(completion)}  |  {title}"
    draw.text((18, 10), title, fill=(255, 255, 255, 255), font=title_font)
    draw.text((18, 45), "Subtasks  " + "  ".join(status_labels), fill=(136, 220, 170, 255), font=progress_font)
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


class ForceControlVideo:
    """Keep progress metadata alongside Lab's clips, including on failed recipes."""

    def __init__(self, env, output: Path):
        self.env = env
        self.output = output
        self.raw_dir = output / "raw_video"
        self.previous_clips = set(self.raw_dir.glob("viewport_*.mp4"))
        self.timeline = []
        self.frames = 0
        self.closed = False
        self.visualizer = next(v for v in env.sim.visualizers if v.cfg.visualizer_type == "kit")

    def record_progress(self, simulation_time_s: float, progress: dict) -> None:
        """Save changes to the title or completion state without advancing predicates."""
        if not self.timeline or progress != self.timeline[-1]["progress"]:
            self.timeline.append({"simulation_time_s": simulation_time_s, "progress": progress})

    def snapshot(self, filename: str, progress: dict) -> None:
        """Capture a boundary image through the same visualizer used by Lab's recorder."""
        import imageio.v3 as iio

        self.env.sim.render()
        frame = self.visualizer.render_rgb_array()
        assert frame is not None, "The Kit visualizer did not return a snapshot"
        iio.imwrite(self.output / filename, overlay_progress_banner(frame, progress))

    def close(self) -> None:
        """Flush Lab's recorder and stitch annotated clips into demo.mp4."""
        import imageio.v2 as imageio

        if self.closed:
            return
        self.closed = True
        for recorder in self.env.video_recorders:
            recorder.close()
        (self.output / "video_progress.json").write_text(json.dumps(self.timeline, indent=2))
        clips = sorted(set(self.raw_dir.glob("viewport_*.mp4")) - self.previous_clips)
        assert clips, "Lab's video recorder produced no clips; see the simulation log"
        timeline_index = 0
        with imageio.get_writer(
            str(self.output / "demo.mp4"), fps=1.0 / self.env.step_dt, codec="libx264", quality=8
        ) as writer:
            for clip in clips:
                with imageio.get_reader(str(clip)) as reader:
                    for frame in reader:
                        simulation_time_s = (self.frames + 1) * self.env.step_dt
                        while (
                            timeline_index + 1 < len(self.timeline)
                            and self.timeline[timeline_index + 1]["simulation_time_s"] <= simulation_time_s + 1e-9
                        ):
                            timeline_index += 1
                        writer.append_data(overlay_progress_banner(frame, self.timeline[timeline_index]["progress"]))
                        self.frames += 1
