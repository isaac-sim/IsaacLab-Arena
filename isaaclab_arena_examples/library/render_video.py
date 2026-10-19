# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Render a library introduction and recorded book handling as an H.264 movie."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab_arena_examples.bento.render_video import CudaRenderer


class LibraryRenderer(CudaRenderer):
    """Frame either the whole library or the book handling station."""

    def __init__(self, scene, width, height, room=False):
        super().__init__(scene, width, height)
        self.room = room
        # The captured 100 m ground plane overlaps the library floor at z=0.
        # Exclude that duplicate surface to avoid camera-dependent depth fighting.
        triangles = self.vertices[self.triangles.long()]
        static = (self.body_ids[self.triangles.long()] == 0).all(-1)
        ground = static & (triangles[..., 2].abs() < 1e-5).all(-1)
        oversized = (triangles[..., :2].amax(dim=1) - triangles[..., :2].amin(dim=1)).amax(dim=1) > 10.0
        self.triangles = self.triangles[~(ground & oversized)].contiguous()
        # Only movable bodies cast shadows; walls and decorative shelves must not
        # project clipped room-sized bands across the floor. Keep the light fixed.
        dynamic = (self.body_ids[self.triangles.long()] > 0).all(-1)
        self.shadow_triangles = self.triangles[dynamic].contiguous()
        self.shadow_center = self.torch.tensor([0.0, 0.2, 1.0], device="cuda")
        self.shadow_resolution = 1024
        paths = scene["body_paths"].tolist() if "body_paths" in scene.files else []
        self.mobile_index = paths.index("/World/MobileBase") if "/World/MobileBase" in paths else None
        self.camera_eye = [0.0, -7.0, 4.8] if room else [1.65, -2.25, 2.45]
        self.camera_target = [0.0, 0.5, 0.9] if room else [0.36, 0.30, 1.02]
        self.fov = 55.0 if room else 50.0
        self.set_camera()

    def set_camera(self, offset=None):
        """Update the camera from a measured mobile base displacement."""
        torch = self.torch
        offset = torch.zeros(3, device="cuda") if offset is None else offset
        self.eye = torch.tensor(self.camera_eye, dtype=torch.float32, device="cuda") + offset
        target = torch.tensor(self.camera_target, dtype=torch.float32, device="cuda") + offset
        forward = torch.nn.functional.normalize(target - self.eye, dim=0)
        right = torch.nn.functional.normalize(
            torch.linalg.cross(forward, torch.tensor([0.0, 0.0, 1.0], device="cuda")), dim=0
        )
        self.view_rotation = torch.stack((right, torch.linalg.cross(right, forward), -forward))
        self.focal = 1.0 / math.tan(math.radians(self.fov) / 2)

    def shadow_projection(self, positions):
        """Use a fixed light volume covering the full mobile route and bookcase."""
        camera = (positions - self.shadow_center) @ self.shadow_rotation.T
        return self.torch.stack((camera[..., 0] / 4.5, camera[..., 1] / 4.5, -camera[..., 2] / 6.0), dim=-1)

    def render(self, poses):
        """Follow the measured base while retaining the shelf and both boxes in view."""
        if not self.room and self.mobile_index is not None:
            offset = self.torch.as_tensor(poses[self.mobile_index, :3], device="cuda").clone()
            offset[2] = 0.0
            self.set_camera(offset)
        return super().render(poses)


def main():
    """Encode measured book handling with a room introduction and phase captions."""
    import numpy as np

    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--output", type=Path, help="Movie destination; defaults to library-mobile.mp4 in capture_dir.")
    args = parser.parse_args()
    if min(args.width, args.height) <= 0 or args.width % 2 or args.height % 2:
        parser.error("Dimensions must be positive even integers.")
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("FPS must be positive and finite.")
    directory = args.capture_dir
    output = args.output or directory / "library-mobile.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 22) if font_path.exists() else ImageFont.load_default()

    def caption(pixels, text):
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)
        lines = text.split("\n")
        band = 18 + 29 * len(lines)
        draw.rectangle((0, args.height - band, args.width, args.height), fill=(22, 29, 34))
        draw.multiline_text((16, args.height - band + 8), text, fill=(240, 244, 247), font=font, spacing=4)
        return np.asarray(image)

    with np.load(directory / "scene.npz", allow_pickle=False) as scene:
        poses, times = scene["poses"], scene["times"]
        renderer = LibraryRenderer(scene, args.width, args.height)
        room_renderer = LibraryRenderer(scene, args.width, args.height, room=True)
        room = caption(room_renderer.render(poses[0]), "Library | catalogued shelf slots, return box and checkout box")
        del room_renderer
        first = caption(renderer.render(poses[0]), "Mobile NHC12 | return box -> assigned shelf slot")
        result_path = directory / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        final_caption = (
            "Complete | LIB-001 -> A-01; LIB-002 -> A-02\nLIB-003 found at A-03 -> checkout box"
            if result.get("success")
            else f"Recorded final state | {result.get('reason', 'completion not verified')}"
        )
        last = caption(renderer.render(poses[-1]), final_caption)
        for name, pixels in (
            ("library-mobile-room.png", room),
            ("library-mobile-first.png", first),
            ("library-mobile-last.png", last),
        ):
            Image.fromarray(pixels).save(output.parent / name)
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
        encoded = 0
        try:
            for _ in range(round(3 * args.fps)):
                writer.send(room)
                encoded += 1
            names = scene["books"] if "books" in scene.files else scene["foods"]
            labels = {
                "return_top": "LIB-001 | return box -> shelf A-01",
                "return_bottom": "LIB-002 | return box -> shelf A-02",
                "retrieve": "LIB-003 | find shelf A-03 -> checkout box",
                "all": "All assigned books delivered",
            }
            phases = {
                "navigate_pick": "Drive to pickup station",
                "navigate_place": "Drive to delivery station",
                "navigate_drop": "Drive to destination",
                "search_catalog": "Look up requested ID and shelf slot",
                "approach_pick": "Approach pickup from the front",
                "lower_pick": "Align with upper book grasp region",
                "insert_pick": "Enter shelf / return box",
                "raise_pick": "Raise book clear of its support",
                "extract": "Withdraw book from shelf",
                "stow": "Hold book safely for mobile transport",
                "approach_drop": "Approach assigned shelf / checkout box",
                "lower_drop": "Align above shelf floor",
                "insert_drop": "Slide book horizontally into assigned slot",
                "seat_drop": "Set book onto shelf / checkout box",
                "release": "Release book at destination",
                "withdraw": "Withdraw gripper from shelf",
                "approach": "Approach selected book",
                "descend": "Lower gripper",
                "close": "Grip book",
                "lift": "Lift book clear",
                "transfer": "Move to assigned destination",
                "place": "Insert / lower book",
                "open": "Release book",
                "retreat": "Withdraw gripper",
                "done": "Complete",
            }
            for index, pose in enumerate(poses):
                label = labels.get(str(names[index]), str(names[index]))
                phase = str(scene["phases"][index])
                pixels = caption(
                    renderer.render(pose),
                    f"{label}\n{phases.get(phase, phase)} | simulation t={times[index]:.1f}s",
                )
                writer.send(pixels)
                encoded += 1
                if index % 100 == 0:
                    print(f"Rendered {index + 1}/{len(poses)}", flush=True)
            for _ in range(round(2 * args.fps)):
                writer.send(last)
                encoded += 1
        finally:
            writer.close()
        metadata = {
            "output": str(output),
            "frames": encoded,
            "fps": args.fps,
            "width": args.width,
            "height": args.height,
            "duration_seconds": encoded / args.fps,
            "source_time_end": float(times[-1]),
            "playback_speed": args.fps * float(np.median(np.diff(times))),
            "renderer": "nvdiffrast CUDA, recorded USD geometry and PhysX poses; not RTX footage",
            "shadows": "world-fixed light volume; robot, books and mobile base cast shadows",
            "floor": "duplicate coplanar 100 m ground plane omitted to prevent depth fighting",
        }
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
