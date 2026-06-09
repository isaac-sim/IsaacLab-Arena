# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Synchronized visualization for parallel Arena environments.

Produces two synced views of a running simulation:

- Global view: read via env.render(); requires render_mode="rgb_array". env.render()'s
  render product is pointed at a dedicated world camera we pose (cfg.viewer.cam_prim_path),
  because the default GUI viewport camera cannot be posed headless. One render product,
  no extra RTX sensor.
- Per-env grid view: a scene-registered TiledCamera, one panel per env tiled into
  a grid. Add it to the scene before build (build_env_camera_cfg + Scene.add_sensor)
  so Isaac Lab updates it as part of env.step.

Placement follows the scene, not hardcoded world coords: the per-env camera is
offset from each env origin and the global view is framed from the env-origin
bounding box. GlobalView takes world-frame eye/target (or None to auto-frame);
EnvView takes eye/target offsets relative to each env origin.

env.render() retrieves what env.step already rendered, so the two views stay in
sync without an extra physics step.

Usage:

    scene.add_sensor(name, SynchronizedVisualizer.build_env_camera_cfg(env_view, name))
    # compose cfg, set cfg.viewer.resolution, build with render_mode="rgb_array"
    viz = SynchronizedVisualizer(env, env_view=env_view, global_view=GlobalView())
    env.reset()
    viz.initialize()
    for _ in range(num_steps):
        env.step(actions)
        viz.capture()
    viz.save("/workspaces/isaaclab_arena/outputs/sync_viz")
"""

from __future__ import annotations

import math
import numpy as np
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from isaaclab_arena.utils import video_io

# Isaac Lab cameras and torch are imported lazily inside methods so this module
# can be imported before the Isaac Sim SimulationApp is launched. video_io above
# is safe to import eagerly (numpy only at module scope; moviepy/Pillow deferred).
if TYPE_CHECKING:
    import gymnasium

    from isaaclab.sensors.camera import Camera

# Horizontal aperture [mm] shared by the per-env and global pinhole cameras. Paired
# with focal_length to set the horizontal FOV; keep both cameras on one value so
# their framing matches.
HORIZONTAL_APERTURE_MM = 20.955


@dataclass
class EnvView:
    """Per-env camera view, given as offsets from each env origin.

    Position and target are env_origin + offset for every env, so each grid panel
    gets the same relative framing regardless of where its env sits in the world.
    """

    eye_offset: tuple[float, float, float] = (0.9, 0.0, 1.0)
    """Camera position relative to the env origin [m]."""

    lookat_offset: tuple[float, float, float] = (0.0, 0.0, 0.1)
    """Camera target (look-at point) relative to the env origin [m]."""

    width: int = 480
    """Per-env render width [px]."""

    height: int = 360
    """Per-env render height [px]."""

    focal_length: float = 24.0
    """Pinhole focal length [mm]."""

    def __post_init__(self) -> None:
        # A zero eye->target vector yields a degenerate (NaN) view rotation in
        # _lookat_offset; catch it here rather than deep in camera setup.
        assert tuple(self.eye_offset) != tuple(self.lookat_offset), (
            "EnvView.eye_offset and lookat_offset must differ (got both "
            f"{self.eye_offset}); the camera has no direction to look."
        )
        assert self.width > 0 and self.height > 0, "EnvView width/height must be positive."
        # A non-positive focal length is silently accepted by Isaac Sim's USD prim
        # setup and yields a broken projection; reject it here.
        assert self.focal_length > 0, "EnvView.focal_length must be positive."


@dataclass
class GlobalView:
    """Whole-simulation camera view, given in world coordinates.

    With eye and lookat both None, the camera auto-frames from the env-origin
    bounding box using distance_scale, azimuth_deg, and elevation_deg.
    """

    eye: tuple[float, float, float] | None = None
    """Camera position in world coordinates [m]. None = auto-frame."""

    lookat: tuple[float, float, float] | None = None
    """Camera target in world coordinates [m]. None = origins centroid."""

    width: int = 1280
    """Global render width [px]. Must be mirrored onto env_cfg.viewer.resolution before build."""

    height: int = 720
    """Global render height [px]."""

    distance_scale: float = 1.6
    """Auto-frame distance as a multiple of the env-origin span. Ignored when eye is set."""

    azimuth_deg: float = -45.0
    """Auto-frame horizontal angle around the centroid [deg]. Ignored when eye is set."""

    elevation_deg: float = 35.0
    """Auto-frame camera elevation above the ground plane [deg]. Ignored when eye is set."""

    margin: float = 1.5
    """Extra padding on the env-origin span when auto-framing [m]. Ignored when eye is set."""

    focal_length: float = 18.147
    """Global camera pinhole focal length [mm]; with HORIZONTAL_APERTURE_MM gives ~60 deg
    horizontal FOV, which the default distance_scale is tuned against."""

    def __post_init__(self) -> None:
        assert self.width > 0 and self.height > 0, "GlobalView width/height must be positive."
        assert self.focal_length > 0, "GlobalView.focal_length must be positive."
        # Auto-frame params are unused with an explicit eye; guard only then, so a bad
        # value fails here instead of deep in posing after a full rollout.
        if self.eye is None:
            assert self.distance_scale > 0, "GlobalView.distance_scale must be positive when auto-framing."
            assert (
                -90.0 < self.elevation_deg < 90.0
            ), f"GlobalView.elevation_deg must be in (-90, 90) when auto-framing, got {self.elevation_deg}."


@dataclass
class _CaptureBuffers:
    """Per-view frame buffers.

    capture() always appends to grid_frames; global_frames may be shorter if
    env.render() returns None; per_env_frames is only populated when
    capture_per_env=True. global_frames and grid_frames are not guaranteed to be
    the same length and must not be zipped by index.
    """

    global_frames: list[np.ndarray] = field(default_factory=list)
    """Composed whole-simulation frames, one per capture()."""

    grid_frames: list[np.ndarray] = field(default_factory=list)
    """Tiled per-env grid frames, one per capture()."""

    per_env_frames: dict[int, list[np.ndarray]] = field(default_factory=dict)
    """Keyed by env index in [0, num_envs); one list of frames per env."""


class SynchronizedVisualizer:
    """Capture a global viewport view and a per-env tiled-camera grid from a built env.

    The per-env TiledCamera must be registered on the scene before build (see
    build_env_camera_cfg); the global view reuses the env viewport camera via
    env.render() (render_mode="rgb_array").

    Args:
        env: A built, gym-wrapped Arena env (render_mode="rgb_array"). env.unwrapped
            must expose sim, device, and scene (with env_origins and num_envs).
        env_view: Per-env view configuration.
        global_view: Whole-simulation view configuration. Defaults to auto-framed.
        grid_cols: Columns in the per-env grid. Defaults to ceil(sqrt(num_envs)).
        max_grid_width: Cap on composed grid width [px]; downscaled to fit.
        env_cam_name: Scene-sensor key of the registered tiled camera. Must match
            the name used with Scene.add_sensor. Defaults to ENV_CAM_NAME.
        capture_per_env: Retain one frame stream per env for save(save_per_env=True).
            Off by default; the grid already shows every env and retaining N streams
            is memory-heavy on long rollouts.
    """

    ENV_CAM_NAME = "sync_viz_env_cam"
    """Scene-sensor key for the per-env tiled camera."""

    GLOBAL_CAM_PRIM_PATH = "/World/SyncVizGlobalCam"
    """USD path of the world camera the global view renders through."""

    @staticmethod
    def _check_optional_deps() -> None:
        """Verify the video/image encoders are importable, raising early if not."""
        missing = []
        try:
            import moviepy.video.io.ImageSequenceClip  # noqa: F401
        except ImportError:
            missing.append("moviepy")
        try:
            import PIL  # noqa: F401
        except ImportError:
            missing.append("Pillow")
        if missing:
            raise ImportError(
                f"SynchronizedVisualizer requires {', '.join(missing)} for video/image output. "
                f"Install with: pip install {' '.join(missing)}"
            )

    @staticmethod
    def _lookat_offset(
        eye_offset: tuple[float, float, float],
        lookat_offset: tuple[float, float, float],
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Return a (pos, quat_wxyz) offset looking from eye toward target.

        The quaternion matches Camera.set_world_poses_from_view (OpenGL convention),
        so it pairs with an OffsetCfg(convention="opengl").
        """
        import torch

        from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

        eyes = torch.tensor([eye_offset], dtype=torch.float32)
        targets = torch.tensor([lookat_offset], dtype=torch.float32)
        rot_matrix = create_rotation_matrix_from_view(eyes, targets, up_axis="Z", device="cpu")
        quat = quat_from_matrix(rot_matrix)[0].tolist()
        pos = (float(eye_offset[0]), float(eye_offset[1]), float(eye_offset[2]))
        quat_wxyz = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        return pos, quat_wxyz

    @staticmethod
    def build_env_camera_cfg(env_view: EnvView | None = None, name: str | None = None):
        """Build the per-env TiledCameraCfg to register on the scene.

        Pass the result to Scene.add_sensor(name, cfg) before build so Isaac Lab
        sets up tiled rendering once and updates it as part of env.step. Adding a
        TiledCamera after build is unreliable; registering it pre-build is the only
        supported path.

        A scene-managed camera recomputes its pose from parent x offset each step, so
        the eye/target framing is baked into OffsetCfg here. The env prim is static
        with identity rotation, so the offset equals the pose relative to the origin.

        Args:
            env_view: Per-env view config (resolution, focal length, eye/target).
            name: Scene-sensor key (the prim leaf name). Defaults to ENV_CAM_NAME.
                Use the same value for Scene.add_sensor and the env_cam_name arg.

        Returns:
            A TiledCameraCfg anchored per env namespace.
        """
        import isaaclab.sim as sim_utils
        from isaaclab.sensors.camera import TiledCameraCfg

        view = env_view if env_view is not None else EnvView()
        key = name if name is not None else SynchronizedVisualizer.ENV_CAM_NAME

        pos, quat_wxyz = SynchronizedVisualizer._lookat_offset(view.eye_offset, view.lookat_offset)
        return TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/" + key,
            update_period=0,
            width=view.width,
            height=view.height,
            data_types=["rgb"],
            offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=quat_wxyz, convention="opengl"),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=view.focal_length,
                focus_distance=400.0,
                horizontal_aperture=HORIZONTAL_APERTURE_MM,
                clipping_range=(0.1, 1.0e5),
            ),
        )

    def __init__(
        self,
        env: gymnasium.Env,
        env_view: EnvView | None = None,
        global_view: GlobalView | None = None,
        grid_cols: int | None = None,
        max_grid_width: int | None = 1920,
        env_cam_name: str | None = None,
        capture_per_env: bool = False,
    ):
        # Fail fast on missing optional encoders: save() / _downscale_to_width need
        # moviepy and Pillow, and discovering they are absent only after a full
        # (expensive) rollout would discard every captured frame.
        self._check_optional_deps()

        self.env = env
        self.unwrapped = env.unwrapped
        self.device = self.unwrapped.device
        self.num_envs = int(self.unwrapped.scene.num_envs)

        self.env_view = env_view if env_view is not None else EnvView()
        self.global_view = global_view if global_view is not None else GlobalView()
        # The global frame IS the viewport, whose resolution comes from
        # cfg.viewer.resolution, not GlobalView. If the caller did not mirror them,
        # GlobalView.width/height would be silently ignored; fail loudly instead.
        viewer_res = tuple(self.unwrapped.cfg.viewer.resolution)
        assert viewer_res == (self.global_view.width, self.global_view.height), (
            f"cfg.viewer.resolution {viewer_res} must match GlobalView "
            f"({self.global_view.width}, {self.global_view.height}); the global view reuses the "
            "viewport, so set env_cfg.viewer.resolution = (global_view.width, global_view.height) before build."
        )
        # ceil(sqrt(N)) keeps the grid roughly square for any env count (4 -> 2x2, 36 -> 6x6).
        self.grid_cols = grid_cols if grid_cols is not None else math.ceil(math.sqrt(self.num_envs))
        assert self.grid_cols > 0, f"grid_cols must be positive, got {self.grid_cols}."
        # Cap on composed grid width [px]; downscaled to fit so a large env count
        # (e.g. 6x6x480px = 2880px) stays a reasonable video size.
        self.max_grid_width = max_grid_width
        self._env_cam_name = env_cam_name if env_cam_name is not None else self.ENV_CAM_NAME
        self.capture_per_env = capture_per_env

        # The global view reads the viewport via env.render(), which only returns
        # frames under render_mode="rgb_array".
        assert self.env.render_mode == "rgb_array", (
            "SynchronizedVisualizer needs the env built with render_mode='rgb_array' "
            f"for the global viewport view; got render_mode={self.env.render_mode!r}."
        )

        self._env_camera: Camera | None = None
        self._buffers = _CaptureBuffers()
        self._initialized = False
        self._closed = False
        # Warn-once sentinels so a degenerate global frame doesn't flood stderr
        # once per capture() during warm-up or on a persistent misconfiguration.
        self._warned_global_black = False
        self._warned_global_none = False
        self._resolve_env_camera()
        self._setup_global_camera_prim()

    def _resolve_env_camera(self) -> None:
        """Look up the scene-registered per-env tiled camera.

        It must have been added to the scene before build (see build_env_camera_cfg)
        and is updated by env.step, so the visualizer never renders or updates it.
        """
        scene_sensors = self.unwrapped.scene.sensors
        assert self._env_cam_name in scene_sensors, (
            f"No scene sensor named {self._env_cam_name!r} found on the env "
            f"(have: {sorted(scene_sensors.keys())}). Register it before build via "
            "Scene.add_sensor(name, SynchronizedVisualizer.build_env_camera_cfg(env_view, name))."
        )
        self._env_camera = scene_sensors[self._env_cam_name]

    def _setup_global_camera_prim(self) -> None:
        """Create a world camera prim and set cfg.viewer.cam_prim_path to point at it.

        env.render() lazily binds its render product to cfg.viewer.cam_prim_path on the
        first call. The default GUI viewport camera is absent headless (the capture mode)
        and not programmatically posable there, so we point that single render product at
        our own world camera instead: no extra RTX sensor.

        Runs in __init__ so cam_prim_path is set before the first env.render().
        """
        from pxr import Gf, UsdGeom

        stage = self.unwrapped.sim.stage
        cam = UsdGeom.Camera.Define(stage, self.GLOBAL_CAM_PRIM_PATH)
        cam.CreateFocalLengthAttr(float(self.global_view.focal_length))
        cam.CreateHorizontalApertureAttr(HORIZONTAL_APERTURE_MM)
        # Match vertical aperture to the output aspect so pixels stay square.
        cam.CreateVerticalApertureAttr(HORIZONTAL_APERTURE_MM * self.global_view.height / self.global_view.width)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1.0e6))
        self.unwrapped.cfg.viewer.cam_prim_path = self.GLOBAL_CAM_PRIM_PATH

    def initialize(self) -> None:
        """Pose the global viewport camera. Call once after env.reset(), before capture().

        The per-env tiled camera needs no setup here: it is a scene sensor posed by
        its baked OffsetCfg and updated by env.step.
        """
        assert not self._closed, "initialize() called after close(); the visualizer is not reusable."
        assert not self._initialized, "initialize() already called; it must run exactly once."
        self._position_global_camera()
        self._initialized = True

    def reposition(self) -> None:
        """Re-apply the global viewport pose. Call after an env.reset() that moved env origins.

        Only the global view needs this; the per-env tiled camera is restored
        automatically from its baked OffsetCfg on reset.
        """
        assert not self._closed, "reposition() called after close(); the visualizer is not reusable."
        self._position_global_camera()

    def _position_global_camera(self) -> None:
        """Pose the global camera prim to the resolved eye/target.

        Authors the global camera prim's world transform directly via USD xform ops,
        which is headless-safe.
        """
        origins = self.unwrapped.scene.env_origins.to(self.device)  # (N, 3)
        eye, target = self._resolve_global_pose(origins)
        self._pose_camera_prim(self.GLOBAL_CAM_PRIM_PATH, eye.detach().cpu().numpy(), target.detach().cpu().numpy())

    def _pose_camera_prim(self, prim_path: str, eye, target) -> None:
        """Author a USD camera prim's world transform to look from eye toward target.

        USD cameras look down local -Z with +Y up, so we build the camera-to-world
        basis directly and write it as a single matrix xform op (USD's row-vector
        convention: rows are the camera's local axes in world coords).
        """
        import numpy as np

        from pxr import Gf, UsdGeom

        eye = np.asarray(eye, dtype=float).reshape(3)
        target = np.asarray(target, dtype=float).reshape(3)
        forward = target - eye
        assert np.linalg.norm(forward) > 1e-6, "Global eye and target coincide; the camera has no direction to look."
        forward = forward / np.linalg.norm(forward)
        back = -forward  # camera local +Z points away from the target
        right = np.cross(np.array([0.0, 0.0, 1.0]), back)
        norm = np.linalg.norm(right)
        # Near-vertical view: world up is parallel to the view direction, leaving
        # "right" undefined; fall back to world +X so the basis stays orthonormal.
        right = right / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
        cam_up = np.cross(back, right)

        cam_to_world = Gf.Matrix4d(
            right[0],
            right[1],
            right[2],
            0.0,
            cam_up[0],
            cam_up[1],
            cam_up[2],
            0.0,
            back[0],
            back[1],
            back[2],
            0.0,
            eye[0],
            eye[1],
            eye[2],
            1.0,
        )
        prim = self.unwrapped.sim.stage.GetPrimAtPath(prim_path)
        assert prim.IsValid(), f"Viewport camera prim {prim_path!r} not found; cannot pose the global view."
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.MakeMatrixXform().Set(cam_to_world)

    def _resolve_global_pose(self, origins):
        """Compute the global camera (eye, target).

        Auto-frames from the env-origin bounding box when GlobalView.eye is unset.
        """
        import torch

        centroid = origins.mean(dim=0)
        target = (
            torch.tensor(self.global_view.lookat, device=self.device, dtype=torch.float32)
            if self.global_view.lookat is not None
            else centroid
        )

        if self.global_view.eye is not None:
            eye = torch.tensor(self.global_view.eye, device=self.device, dtype=torch.float32)
            return eye, target

        # Auto-frame: place the camera at an azimuth/elevation around the
        # centroid, at a distance proportional to the env-origin span.
        span_xyz = origins.max(dim=0).values - origins.min(dim=0).values
        span = float(torch.max(span_xyz[:2]).item()) + 2.0 * self.global_view.margin
        span = max(span, 1.0)
        distance = span * self.global_view.distance_scale

        az = math.radians(self.global_view.azimuth_deg)
        el = math.radians(self.global_view.elevation_deg)
        horizontal = distance * math.cos(el)
        offset = torch.tensor(
            [horizontal * math.cos(az), horizontal * math.sin(az), distance * math.sin(el)],
            device=self.device,
            dtype=torch.float32,
        )
        eye = centroid + offset
        return eye, target

    def capture(self) -> None:
        """Append one frame per view (per-env grid + global) to the internal buffers.
        Call once per env.step.

        Warns once (not per frame) if env.render() returns None, or if the global frame
        is all-zeros (annotator not warmed up); an all-zeros frame is still encoded.
        """
        assert not self._closed, "capture() called after close(); create a new SynchronizedVisualizer."
        assert self._initialized, "Call initialize() before capture()."
        assert self._env_camera is not None, "Per-env camera unresolved; _resolve_env_camera() did not run."

        env_rgb = self._read_rgb(self._env_camera, "per-env")  # (N, H, W, 3)
        per_env = []
        for i in range(self.num_envs):
            img = video_io.to_uint8(env_rgb[i])
            per_env.append(img)
            # Only retain per-env streams when asked; the grid is built from per_env
            # either way (see _tile below).
            if self.capture_per_env:
                self._buffers.per_env_frames.setdefault(i, []).append(img)
        grid = self._tile(per_env, self.grid_cols)
        if self.max_grid_width is not None and grid.shape[1] > self.max_grid_width:
            grid = self._downscale_to_width(grid, self.max_grid_width)
        self._buffers.grid_frames.append(grid)

        global_rgb = self.env.render()
        if global_rgb is None:
            # __init__ enforces render_mode="rgb_array", so this only happens on a
            # misconfiguration/version mismatch; warn once rather than every step.
            if not self._warned_global_none:
                print(
                    "Warning: env.render() returned None; no global frames will be recorded. "
                    "Build the env with render_mode='rgb_array' and run with --enable_cameras."
                )
                self._warned_global_none = True
            return
        global_frame = video_io.to_uint8(global_rgb)
        if not global_frame.any() and not self._warned_global_black:
            print(
                "Warning: captured an all-zeros global frame; the viewport annotator may not have "
                "warmed up. Warm up with a few env.step() calls before capture(). The frame is still encoded."
            )
            self._warned_global_black = True
        self._buffers.global_frames.append(global_frame)

    @staticmethod
    def _read_rgb(camera, label: str):
        """Return the camera's rgb output, with a clear error if not yet ready."""
        output = camera.data.output
        if "rgb" not in output:
            raise RuntimeError(
                f"{label} camera has no 'rgb' output yet (available: {list(output.keys())}). "
                "The annotator has not delivered a frame; ensure env.step() has been called "
                "and warm up with a few steps before capturing."
            )
        return output["rgb"]

    @staticmethod
    def _downscale_to_width(img: np.ndarray, max_width: int) -> np.ndarray:
        """Downscale img to max_width, keeping aspect and even dims for h264."""
        from PIL import Image

        h, w = img.shape[:2]
        new_w = max_width - (max_width % 2)
        new_h = max(2, round(h * new_w / w))
        new_h -= new_h % 2
        return np.asarray(Image.fromarray(img).resize((new_w, new_h), Image.Resampling.BILINEAR))

    @staticmethod
    def _tile(images: list[np.ndarray], num_cols: int) -> np.ndarray:
        assert len(images) > 0, "_tile() called with zero images."
        ref_shape = images[0].shape[:2]
        assert all(img.shape[:2] == ref_shape for img in images), (
            "_tile() requires all per-env images to share the same (H, W); "
            f"got mixed shapes {sorted({img.shape[:2] for img in images})}."
        )
        n = len(images)
        num_rows = math.ceil(n / num_cols)
        h, w = ref_shape
        grid = np.zeros((num_rows * h, num_cols * w, 3), dtype=np.uint8)
        for idx, img in enumerate(images):
            r, c = divmod(idx, num_cols)
            grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = img[:, :, :3]
        return grid

    def save(
        self,
        output_dir: str,
        fps: int = 30,
        name_prefix: str = "sync_viz",
        save_per_env: bool = False,
        also_gif: bool = False,
    ) -> dict[str, str]:
        """Write the captured frames to output_dir as mp4 (and optional gif).

        Args:
            output_dir: Directory to write outputs into (created if missing).
            fps: Frames per second for the encoded videos.
            name_prefix: Prefix for output filenames.
            save_per_env: Also write one mp4 per environment. Requires the
                visualizer to have been built with capture_per_env=True.
            also_gif: Also write animated gifs alongside the mp4s.

        Returns:
            Mapping of logical output name to written file path.
        """
        # name_prefix joins into a path; an absolute or separator-bearing value would
        # escape output_dir (os.path.join drops earlier parts on an absolute arg).
        assert (
            "/" not in name_prefix and "\\" not in name_prefix
        ), f"name_prefix must not contain path separators, got {name_prefix!r}."
        if save_per_env and not self._buffers.per_env_frames:
            print(
                "Warning: save_per_env=True but no per-env frames were captured; build the "
                "visualizer with capture_per_env=True to record them. Writing global/grid only."
            )
        os.makedirs(output_dir, exist_ok=True)
        written: dict[str, str] = {}

        if self._buffers.global_frames:
            path = os.path.join(output_dir, f"{name_prefix}_global.mp4")
            video_io.write_video(self._buffers.global_frames, path, fps)
            written["global"] = path
            if also_gif:
                gif_path = os.path.join(output_dir, f"{name_prefix}_global.gif")
                video_io.write_gif(self._buffers.global_frames, gif_path, fps)
                written["global_gif"] = gif_path

        if self._buffers.grid_frames:
            path = os.path.join(output_dir, f"{name_prefix}_grid.mp4")
            video_io.write_video(self._buffers.grid_frames, path, fps)
            written["grid"] = path
            if also_gif:
                gif_path = os.path.join(output_dir, f"{name_prefix}_grid.gif")
                video_io.write_gif(self._buffers.grid_frames, gif_path, fps)
                written["grid_gif"] = gif_path

        if save_per_env:
            for env_idx, frames in self._buffers.per_env_frames.items():
                if not frames:
                    continue
                path = os.path.join(output_dir, f"{name_prefix}_env{env_idx:03d}.mp4")
                video_io.write_video(frames, path, fps)
                written[f"env{env_idx:03d}"] = path
                if also_gif:
                    gif_path = os.path.join(output_dir, f"{name_prefix}_env{env_idx:03d}.gif")
                    video_io.write_gif(frames, gif_path, fps)
                    written[f"env{env_idx:03d}_gif"] = gif_path

        if not written:
            print(
                "Warning: save() wrote nothing: no frames were captured. "
                "Call capture() (after initialize() and at least one env.step()) before save()."
            )
        return written

    def close(self) -> None:
        """Release the sensor references and free the captured frame buffers.

        Idempotent: safe to call more than once. The visualizer is not reusable
        afterwards (initialize() and capture() assert on _closed).
        """
        self._env_camera = None
        self._initialized = False
        self._closed = True
        # Drop accumulated frames (potentially GBs for long, many-env rollouts).
        self._buffers = _CaptureBuffers()
