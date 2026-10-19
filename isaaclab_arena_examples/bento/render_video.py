# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Render recorded USD geometry and PhysX poses with a CUDA rasterizer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse the capture directory and movie settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--camera", choices=("overview", "bento", "robot"), default="overview")
    parser.add_argument("--save-frames", action="store_true", help="Save every encoded frame as a PNG.")
    parser.add_argument("--preview-only", action="store_true", help="Render the first and last captured states.")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2:
        parser.error("--width and --height must be positive even integers.")
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be positive and finite.")
    if args.camera == "robot" and not args.preview_only:
        parser.error("--camera robot requires --preview-only.")
    return args


class CudaRenderer:
    """Rasterize recorded body meshes from a fixed oblique camera."""

    def __init__(self, scene, width: int, height: int, camera_mode: str = "overview"):
        """Upload the recorded geometry and initialize the CUDA camera.

        Args:
            scene: Loaded capture arrays with body-local geometry and world poses.
            width: Output image width in pixels.
            height: Output image height in pixels.
            camera_mode: Whole workcell overview or food-focused close view.
        """
        try:
            import torch

            import nvdiffrast.torch as dr
        except ImportError as exc:
            raise RuntimeError(
                "Video rendering requires PyTorch and NVIDIA nvdiffrast in this Python environment."
            ) from exc
        assert torch.cuda.is_available(), "Video rendering requires a CUDA GPU."
        self.torch, self.dr = torch, dr
        self.ctx = dr.RasterizeCudaContext(device="cuda:0")
        self.width, self.height = width, height
        self.vertices = torch.as_tensor(scene["vertices"], dtype=torch.float32, device="cuda")
        self.normals = torch.as_tensor(scene["normals"], dtype=torch.float32, device="cuda")
        self.colors = torch.as_tensor(scene["colors"], dtype=torch.float32, device="cuda")
        self.body_ids = torch.as_tensor(scene["vertex_body_ids"], dtype=torch.int64, device="cuda")
        self.triangles = torch.as_tensor(scene["triangles"], dtype=torch.int32, device="cuda").contiguous()
        triangle_vertices = self.vertices[self.triangles.long()]
        ground = (self.body_ids[self.triangles.long()] == 0).all(-1) & (triangle_vertices[..., 2].abs() < 0.1).all(-1)
        self.shadow_triangles = self.triangles[~ground].contiguous()
        assert self.vertices.shape == self.normals.shape == self.colors.shape, "Geometry attributes must match."
        assert (
            self.vertices.ndim == 2 and self.vertices.shape[1] == 3 and len(self.triangles)
        ), "Empty capture geometry."
        assert self.body_ids.shape == (len(self.vertices),), "Each vertex must have a body ID."
        self.camera_mode = camera_mode
        cameras = {
            "overview": ([1.4, -1.45, 2.25], [0.35, 0.03, 0.98], 45.0),
            "bento": ([1.02, -0.7, 1.82], [0.50, 0.13, 0.80], 45.0),
            "plated": ([0.90, -0.31, 1.41], [0.57, 0.24, 0.80], 35.0),
            "robot": ([1.4, -1.45, 2.25], [0.35, 0.03, 1.4], 35.0),
        }
        self.camera_eye, self.camera_target, self.fov = cameras[camera_mode]
        if camera_mode == "robot":
            self.focus_robot_logo(scene["poses"][0])
        self.eye = torch.tensor(self.camera_eye, dtype=torch.float32, device="cuda")
        target = torch.tensor(self.camera_target, dtype=torch.float32, device="cuda")
        forward = torch.nn.functional.normalize(target - self.eye, dim=0)
        right = torch.nn.functional.normalize(
            torch.linalg.cross(forward, torch.tensor([0.0, 0.0, 1.0], device="cuda")), dim=0
        )
        up = torch.linalg.cross(right, forward)
        self.view_rotation = torch.stack((right, up, -forward))
        self.focal = 1.0 / math.tan(math.radians(self.fov) / 2)
        self.light = torch.nn.functional.normalize(torch.tensor([0.4, -0.6, 1.0], device="cuda"), dim=0)
        self.background = torch.tensor([0.82, 0.86, 0.89], device="cuda")
        self.shadow_center = torch.tensor([0.35, 0.03, 1.0], device="cuda")
        shadow_right = torch.nn.functional.normalize(
            torch.linalg.cross(-self.light, torch.tensor([0.0, 0.0, 1.0], device="cuda")), dim=0
        )
        self.shadow_rotation = torch.stack((shadow_right, torch.linalg.cross(self.light, shadow_right), self.light))
        self.shadow_resolution = 1024

    def focus_robot_logo(self, poses):
        """Frame the recorded robot's blue material geometry from its outward surface."""
        torch = self.torch
        blue = (
            (self.colors[:, 2] > self.colors[:, 0] * 1.8)
            & (self.colors[:, 2] > self.colors[:, 1] * 1.35)
            & (self.colors[:, 2] > 0.2)
            & (self.body_ids > 0)
            & (self.body_ids < 8)
        )
        assert blue.any().item(), "The capture contains no blue robot logo material."
        # The base uses the same blue paint; frame lettering on the forearm instead.
        forearm_blue = blue & (self.body_ids >= 4)
        if forearm_blue.any().item():
            blue = forearm_blue
        body_id = int(torch.bincount(self.body_ids[blue]).argmax().item())
        selection = blue & (self.body_ids == body_id)
        body_poses = torch.as_tensor(poses, dtype=torch.float32, device="cuda")[self.body_ids[selection]]
        quaternions = torch.nn.functional.normalize(body_poses[:, 3:], dim=-1)
        positions = self.rotate(self.vertices[selection], quaternions) + body_poses[:, :3]
        normals = self.rotate(self.normals[selection], quaternions)
        preferred = torch.nn.functional.normalize(
            torch.tensor(self.camera_eye, device="cuda") - positions.mean(dim=0), dim=0
        )
        facing = (normals * preferred).sum(-1) > 0.20
        if facing.any().item():
            positions, normals = positions[facing], normals[facing]
        target = positions.mean(dim=0)
        outward = normals.mean(dim=0)
        outward = preferred if outward.norm().item() < 0.1 else torch.nn.functional.normalize(outward, dim=0)
        distance = max(float((positions.max(dim=0).values - positions.min(dim=0).values).norm().item()) * 2.5, 0.16)
        eye = target + outward * distance + torch.tensor([0.0, 0.0, 0.01], device="cuda")
        self.camera_eye, self.camera_target = eye.cpu().tolist(), target.cpu().tolist()
        self.logo_body_id = body_id

    def shadow_projection(self, positions):
        """Project world points into an orthographic directional-light camera."""
        camera = (positions - self.shadow_center) @ self.shadow_rotation.T
        return self.torch.stack((camera[..., 0] / 1.3, camera[..., 1] / 1.3, -camera[..., 2] / 3.0), dim=-1)

    def visibility(self, world, position, normal):
        """Compute a soft shadow from the current recorded mesh positions."""
        torch, dr = self.torch, self.dr
        projected = self.shadow_projection(world)
        clip = torch.cat((projected, torch.ones_like(projected[:, :1])), dim=-1).unsqueeze(0).contiguous()
        rast, _ = dr.rasterize(self.ctx, clip, self.shadow_triangles, resolution=[self.shadow_resolution] * 2)
        depth = torch.where(rast[..., 3:4] > 0, rast[..., 2:3], torch.ones_like(rast[..., 2:3]))
        query = self.shadow_projection(position)
        uv = ((query[..., :2] + 1) / 2).contiguous()
        bias = 0.0004 + 0.0010 * (1 - (normal * self.light).sum(-1, keepdim=True).clamp(0, 1))
        lit = torch.zeros_like(query[..., 2:3])
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                offset = torch.tensor([x, y], dtype=torch.float32, device="cuda") * (1.5 / self.shadow_resolution)
                sampled = dr.texture(
                    depth.contiguous(), (uv + offset).contiguous(), filter_mode="linear", boundary_mode="clamp"
                )
                lit += (query[..., 2:3] - bias <= sampled).to(torch.float32) / 9
        inside = (query[..., :2].abs() <= 1).all(-1, keepdim=True)
        return torch.where(inside, lit, torch.ones_like(lit))

    def rotate(self, values, quaternions):
        """Rotate body-local vectors with XYZW quaternions."""
        qvec = quaternions[:, :3]
        twice_cross = 2.0 * self.torch.linalg.cross(qvec, values)
        return values + quaternions[:, 3:] * twice_cross + self.torch.linalg.cross(qvec, twice_cross)

    def render(self, poses):
        """Return one uint8 RGB image for recorded world poses.

        Args:
            poses: Body poses in world XYZ and XYZW quaternion order.

        Returns:
            Image array in top-to-bottom row order.
        """
        torch, dr = self.torch, self.dr
        with torch.no_grad():
            body_poses = torch.as_tensor(poses, dtype=torch.float32, device="cuda")[self.body_ids]
            quaternions = torch.nn.functional.normalize(body_poses[:, 3:], dim=-1)
            world = self.rotate(self.vertices, quaternions) + body_poses[:, :3]
            normals = self.rotate(self.normals, quaternions)
            camera = (world - self.eye) @ self.view_rotation.T
            near, far = 0.03, 20.0
            clip = (
                torch.stack(
                    (
                        camera[:, 0] * self.focal / (self.width / self.height),
                        camera[:, 1] * self.focal,
                        -(far + near) / (far - near) * camera[:, 2] - 2 * far * near / (far - near),
                        -camera[:, 2],
                    ),
                    dim=-1,
                )
                .unsqueeze(0)
                .contiguous()
            )
            rast, _ = dr.rasterize(self.ctx, clip, self.triangles, resolution=[self.height, self.width])
            attributes = torch.cat((self.colors, normals, world), dim=-1).unsqueeze(0).contiguous()
            interpolated, _ = dr.interpolate(attributes, rast, self.triangles)
            color, normal, position = interpolated[..., :3], interpolated[..., 3:6], interpolated[..., 6:9]
            normal = torch.nn.functional.normalize(normal, dim=-1)
            view = torch.nn.functional.normalize(self.eye - position, dim=-1)
            # Shade both sides: USD assets may contain open meshes with mixed winding.
            normal = torch.where((normal * view).sum(-1, keepdim=True) < 0, -normal, normal)
            diffuse = (normal * self.light).sum(-1, keepdim=True).clamp(0, 1)
            visibility = self.visibility(world, position, normal)
            halfway = torch.nn.functional.normalize(view + self.light, dim=-1)
            specular = (normal * halfway).sum(-1, keepdim=True).clamp(0, 1).pow(32) * 0.12
            sky = 0.38 + 0.08 * normal[..., 2:3].clamp(0, 1)
            shaded = color * (sky + 0.65 * diffuse * visibility) + specular * visibility
            output = torch.where(rast[..., 3:4] > 0, shaded, self.background).contiguous()
            output = dr.antialias(output, rast, clip, self.triangles)
            # USD diffuse colors are linear; export standard sRGB for PNG and MP4.
            linear = output.clamp(0, 1)
            srgb = torch.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear.pow(1 / 2.4) - 0.055)
            return (srgb[0].flip(0) * 255).round().to(torch.uint8).cpu().numpy()


def run(args: argparse.Namespace) -> None:
    """Render capture previews, optional frame images, and an H.264 movie.

    Args:
        args: Parsed capture path and output settings.
    """
    import numpy as np

    from PIL import Image

    scene_path = args.capture_dir / "scene.npz"
    assert scene_path.is_file(), f"Capture geometry is missing: {scene_path}"
    with np.load(scene_path, allow_pickle=False) as scene:
        poses = scene["poses"]
        times = scene["times"]
        assert len(poses) > 0 and len(poses) == len(times), "Capture poses and times must match."
        renderer = CudaRenderer(scene, args.width, args.height, args.camera)
        first = renderer.render(poses[0])
        if args.camera == "robot":
            image_path = args.capture_dir / "preview-robot-logo.png"
            Image.fromarray(first).save(image_path)
            logo_metadata = {
                "source_frame": 0,
                "source_time": float(times[0]),
                "logo_body_id": renderer.logo_body_id,
                "camera_eye": renderer.camera_eye,
                "camera_target": renderer.camera_target,
                "fov_y_degrees": renderer.fov,
                "logo_source": "recorded USD blue robot material geometry",
            }
            (args.capture_dir / "robot-logo.json").write_text(
                json.dumps(logo_metadata, indent=2) + "\n", encoding="utf-8"
            )
            print(f"Saved {image_path}", flush=True)
            return
        last = renderer.render(poses[-1])
        preview_prefix = "preview" if args.camera == "overview" else "preview-bento"
        Image.fromarray(first).save(args.capture_dir / f"{preview_prefix}-first.png")
        Image.fromarray(last).save(args.capture_dir / f"{preview_prefix}-last.png")
        plated_renderer = CudaRenderer(scene, args.width, args.height, "plated")
        Image.fromarray(plated_renderer.render(poses[-1])).save(args.capture_dir / "plated-bento.png")
        del plated_renderer
        if args.preview_only:
            print(f"Saved first/last previews in {args.capture_dir}", flush=True)
            return

        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise RuntimeError("MP4 encoding requires imageio-ffmpeg in this Python environment.") from exc
        output = args.output or args.capture_dir / ("bento.mp4" if args.camera == "overview" else "bento-closeup.mp4")
        output.parent.mkdir(parents=True, exist_ok=True)
        frame_dir = args.capture_dir / ("frames" if args.camera == "overview" else "frames-bento")
        if args.save_frames:
            frame_dir.mkdir(parents=True, exist_ok=True)
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

        def append(pixels):
            nonlocal encoded
            if args.save_frames:
                Image.fromarray(pixels).save(frame_dir / f"frame_{encoded:06d}.png")
            writer.send(pixels)
            encoded += 1

        first_hold, last_hold = round(1.5 * args.fps), round(2 * args.fps)
        try:
            for _ in range(first_hold):
                append(first)
            for index, pose in enumerate(poses):
                append(first if index == 0 else last if index == len(poses) - 1 else renderer.render(pose))
                if index % 100 == 0 or index == len(poses) - 1:
                    print(f"Rendered {index + 1}/{len(poses)} captured states", flush=True)
            for _ in range(last_hold):
                append(last)
        finally:
            writer.close()
        capture_path = args.capture_dir / "capture.json"
        capture = json.loads(capture_path.read_text()) if capture_path.is_file() else {}
        interval = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
        sampling_fps = capture.get("sampling_fps", 1 / interval if interval > 0 else None)
        metadata = {
            "output": str(output),
            "renderer": "nvdiffrast_cuda",
            "geometry_source": "USD meshes and primitives",
            "state_source": "recorded PhysX body poses",
            "fps": args.fps,
            "frames": encoded,
            "width": args.width,
            "height": args.height,
            "duration_seconds": encoded / args.fps,
            "source_frames": len(poses),
            "source_sampling_fps": sampling_fps,
            "source_time_start": float(times[0]),
            "source_time_end": float(times[-1]),
            "playback_speed": args.fps / sampling_fps if sampling_fps else None,
            "first_hold_seconds": first_hold / args.fps,
            "last_hold_seconds": last_hold / args.fps,
            "camera_mode": args.camera,
            "camera_eye": renderer.camera_eye,
            "camera_target": renderer.camera_target,
            "fov_y_degrees": renderer.fov,
            "shading": "Lambert diffuse, Blinn-Phong specular, and filtered directional shadow maps; no texture maps",
        }
        metadata_name = "video.json" if args.camera == "overview" else "video-bento.json"
        (args.capture_dir / metadata_name).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {output} ({encoded} frames)", flush=True)


if __name__ == "__main__":
    run(parse_args())
