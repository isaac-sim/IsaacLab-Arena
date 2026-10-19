# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Render recorded dual-arm contact with an occupied, partly leaning book row."""

import argparse
import json
import math
from pathlib import Path

from isaaclab_arena_examples.library.render_video import LibraryRenderer


class CrowdedRenderer(LibraryRenderer):
    """Keep both physical hands and the occupied shelf visible from a fixed camera."""

    def __init__(self, scene, width, height):
        super().__init__(scene, width, height)
        self.camera_eye = [0.70, -2.20, 1.55]
        self.camera_target = [0.0, 0.38, 1.30]
        self.fov = 48.0
        self.set_camera()

    def shadow_projection(self, positions):
        camera = (positions - self.shadow_center) @ self.shadow_rotation.T
        return self.torch.stack((camera[..., 0] / 1.8, camera[..., 1] / 1.8, -camera[..., 2] / 3), dim=-1)


def main():
    """Write verified motion footage and state previews without implying failed runs succeeded."""
    import numpy as np

    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()
    if min(args.width, args.height) <= 0 or args.width % 2 or args.height % 2:
        parser.error("Dimensions must be positive even integers.")
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("FPS must be positive and finite.")
    output = args.output or args.capture_dir / "crowded-shelf.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    result_path = args.capture_dir / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 20) if font_path.exists() else ImageFont.load_default()
    labels = {
        "helper_spread_gap": "Open a gap by pushing adjacent books",
        "align_return": "Hold the neighboring books, align the returned book",
        "insert_return": "Insert the returned book into the opened gap",
        "seat_return": "Set the returned book on the shelf",
        "helper_support_lean": "Support the leaning neighbor before extraction",
        "loan_grasp": "Reach the requested book",
        "extract_loan": "Withdraw requested book while supporting its neighbor",
        "loan_box_lower": "Place the requested book in the checkout box",
        "done": "Return and loan completed",
    }

    def caption(pixels, text):
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, args.height - 64, args.width, args.height), fill=(22, 29, 34))
        draw.multiline_text((12, args.height - 59), text, font=font, fill=(240, 244, 247), spacing=4)
        return np.asarray(image)

    with np.load(args.capture_dir / "scene.npz", allow_pickle=False) as scene:
        poses, times = scene["poses"], scene["times"]
        renderer = CrowdedRenderer(scene, args.width, args.height)

        def frame(index):
            phase = str(scene["phases"][index])
            extra = json.loads(str(scene["extras_json"][index])) if "extras_json" in scene.files else {}
            gap = extra.get("gap_m")
            detail = (
                f"Measured gap: {gap*1000:.1f} mm | t={times[index]:.1f}s"
                if gap is not None
                else f"t={times[index]:.1f}s"
            )
            return caption(renderer.render(poses[index]), f'{labels.get(phase,phase.replace("_"," "))}\n{detail}')

        first = frame(0)
        last = frame(len(poses) - 1)
        if not result.get("success"):
            last = caption(
                renderer.render(poses[-1]),
                "Trial state | completion not verified\n" + str(result.get("reason", "probe"))[:100],
            )
        Image.fromarray(first).save(output.parent / "crowded-first.png")
        Image.fromarray(last).save(output.parent / "crowded-last.png")
        if args.preview_only:
            return
        writer = imageio_ffmpeg.write_frames(
            str(output),
            (args.width, args.height),
            fps=args.fps,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            quality=8,
            macro_block_size=1,
            ffmpeg_log_level="error",
            output_params=["-movflags", "+faststart"],
        )
        writer.send(None)
        count = 0
        try:
            for _ in range(round(2 * args.fps)):
                writer.send(first)
                count += 1
            for i in range(len(poses)):
                writer.send(frame(i))
                count += 1
                if i % 100 == 0:
                    print(f"Rendered {i+1}/{len(poses)}", flush=True)
            for _ in range(round(2 * args.fps)):
                writer.send(last)
                count += 1
        finally:
            writer.close()
        metadata = {
            "output": str(output),
            "frames": count,
            "fps": args.fps,
            "width": args.width,
            "height": args.height,
            "duration_seconds": count / args.fps,
            "simulation_success": result.get("success", False),
            "validation_method": result.get("validation", {}).get("method", "online simulation validation"),
            "source_time_end": float(times[-1]),
            "renderer": "recorded USD geometry and PhysX poses, nvdiffrast CUDA",
        }
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
