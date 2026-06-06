# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Synchronized, general-purpose visualization for parallel Arena environments.

Produces two complementary views of a running simulation:

* **Global view** — reuses the env's viewport camera (``cfg.viewer.cam_prim_path``)
  via ``env.render()``; no extra camera sensor is spawned. The env must be built
  with ``render_mode="rgb_array"``. Its pose is driven by the framing logic here.
* **Per-env grid view** — a scene-registered :class:`~isaaclab.sensors.TiledCamera`
  (one panel per env, tiled into a grid). The camera must be added to the scene
  *before* build via :meth:`SynchronizedVisualizer.build_env_camera_cfg` +
  ``Scene.add_sensor`` so Isaac Lab updates it as part of ``env.step``.

The per-env camera is anchored to each env base (pose relative to
``scene.env_origins``) and the global view is framed from the env-origin
bounding box, so placement follows the actual scene rather than hardcoded
world coordinates. The caller specifies:

* :class:`GlobalView` — eye/target in world coordinates (or ``None`` to
  auto-frame from the env-origin bounding box).
* :class:`EnvView` — eye/target offsets expressed *relative to each env origin*.

Both views read whatever ``env.step`` already rendered — no separate
``sim.render()`` / ``camera.update()`` pass — so frames stay in sync.

Typical usage::

    scene.add_sensor(name, SynchronizedVisualizer.build_env_camera_cfg(env_view, name))
    # ... compose cfg, set cfg.viewer.resolution, build with render_mode="rgb_array" ...
    viz = SynchronizedVisualizer(env, env_view=env_view, global_view=GlobalView())
    env.reset()
    viz.initialize()  # poses the global viewport camera
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
# can be imported before the Isaac Sim ``SimulationApp`` is launched. ``video_io``
# above is safe to import eagerly (numpy only at module scope; moviepy/Pillow are
# themselves deferred).
if TYPE_CHECKING:
    import gymnasium

    from isaaclab.sensors.camera import Camera


@dataclass
class EnvView:
    """Per-environment camera view, specified relative to each env origin.

    The camera position and target are ``env_origin + offset`` for every
    environment, so the same relative framing is reproduced in every panel of
    the grid regardless of where each env sits in the world.
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
        # ``_lookat_offset``; catch it here rather than deep in camera setup.
        assert tuple(self.eye_offset) != tuple(self.lookat_offset), (
            "EnvView.eye_offset and lookat_offset must differ (got both "
            f"{self.eye_offset}); the camera has no direction to look."
        )
        assert self.width > 0 and self.height > 0, "EnvView width/height must be positive."


@dataclass
class GlobalView:
    """Whole-simulation camera view, specified in world coordinates.

    If :attr:`eye` and :attr:`lookat` are both ``None`` the camera is
    auto-framed from the bounding box of all env origins, using
    :attr:`distance_scale`, :attr:`azimuth_deg`, and :attr:`elevation_deg`.
    """

    eye: tuple[float, float, float] | None = None
    """Camera position in world coordinates [m]. ``None`` = auto-frame."""

    lookat: tuple[float, float, float] | None = None
    """Camera target in world coordinates [m]. ``None`` = origins centroid."""

    width: int = 1280
    """Global render width [px]."""

    height: int = 720
    """Global render height [px]."""

    distance_scale: float = 1.6
    """Auto-frame distance as a multiple of the env-origin span."""

    azimuth_deg: float = -45.0
    """Auto-frame horizontal angle around the centroid [deg]."""

    elevation_deg: float = 35.0
    """Auto-frame camera elevation above the ground plane [deg]."""

    margin: float = 1.5
    """Extra padding added to the env-origin span when auto-framing [m]."""

    def __post_init__(self) -> None:
        # ``elevation_deg = ±90`` collapses the horizontal framing distance to 0
        # (camera directly overhead with no standoff), which auto-frames to a
        # degenerate pose. Keep it strictly within the open interval.
        assert (
            -90.0 < self.elevation_deg < 90.0
        ), f"GlobalView.elevation_deg must be in (-90, 90), got {self.elevation_deg}."
        assert self.width > 0 and self.height > 0, "GlobalView width/height must be positive."


@dataclass
class _CaptureBuffers:
    global_frames: list[np.ndarray] = field(default_factory=list)
    """Composed whole-simulation frames, one per :meth:`SynchronizedVisualizer.capture`."""

    grid_frames: list[np.ndarray] = field(default_factory=list)
    """Tiled per-env grid frames, one per :meth:`SynchronizedVisualizer.capture`."""

    per_env_frames: dict[int, list[np.ndarray]] = field(default_factory=dict)
    """Keyed by env index in ``[0, num_envs)``; one list of frames per environment."""


class SynchronizedVisualizer:
    """Capture a global viewport view and a per-env tiled-camera grid from a built env.

    The per-env :class:`~isaaclab.sensors.TiledCamera` must be registered on the
    scene before build (see :meth:`build_env_camera_cfg`); the global view reuses
    the env's viewport camera via ``env.render()`` (``render_mode="rgb_array"``).

    Args:
        env: A built (gym-wrapped) Arena environment, built with
            ``render_mode="rgb_array"``. ``env.unwrapped`` must expose ``sim``,
            ``device``, and ``scene`` (with ``env_origins`` and ``num_envs``).
        env_view: Per-environment view configuration.
        global_view: Whole-simulation view configuration. Defaults to an
            auto-framed :class:`GlobalView`.
        grid_cols: Number of columns in the per-env grid. Defaults to
            ``ceil(sqrt(num_envs))`` (roughly square).
        max_grid_width: Cap on the composed grid width [px]; downscaled to fit.
        env_cam_name: Scene-sensor key of the registered tiled camera. Must match
            the name used with ``Scene.add_sensor``. Defaults to :attr:`ENV_CAM_NAME`.
    """

    ENV_CAM_NAME = "sync_viz_env_cam"
    """Scene-sensor key for the per-env :class:`~isaaclab.sensors.TiledCamera`."""

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
    def _lookat_offset(eye_offset, lookat_offset):
        """Return an ``(pos, quat_wxyz)`` offset that looks from ``eye`` toward ``target``.

        The quaternion is computed exactly like
        :meth:`~isaaclab.sensors.Camera.set_world_poses_from_view` (OpenGL
        convention), so it pairs with an ``OffsetCfg(convention="opengl")``.
        """
        import torch

        from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

        eyes = torch.tensor([eye_offset], dtype=torch.float32)
        targets = torch.tensor([lookat_offset], dtype=torch.float32)
        rot_matrix = create_rotation_matrix_from_view(eyes, targets, up_axis="Z", device="cpu")
        q = quat_from_matrix(rot_matrix)[0].tolist()
        pos = (float(eye_offset[0]), float(eye_offset[1]), float(eye_offset[2]))
        quat_wxyz = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        return pos, quat_wxyz

    @staticmethod
    def build_env_camera_cfg(env_view: EnvView | None = None, name: str | None = None):
        """Build the per-env :class:`~isaaclab.sensors.TiledCameraCfg` to register on the scene.

        Pass the result to ``Scene.add_sensor(name, cfg)`` *before* build so Isaac
        Lab sets up tiled rendering once and updates it as part of ``env.step`` —
        avoiding the all-white frames a second render product causes once the sim
        is playing, and any out-of-sync manual ``update()`` pass.

        A scene-managed camera recomputes its pose from ``parent × offset`` each
        step, so the eye/target framing is baked into ``OffsetCfg`` here. The env
        prim is static with identity rotation, so the offset equals the desired
        pose relative to the env origin.

        Args:
            env_view: Per-env view config (resolution, focal length, eye/target).
            name: Scene-sensor key (the prim leaf name). Defaults to
                :attr:`ENV_CAM_NAME`. Use the same value for ``Scene.add_sensor``
                and the :class:`SynchronizedVisualizer` ``env_cam_name`` arg.

        Returns:
            A :class:`~isaaclab.sensors.TiledCameraCfg` anchored per env namespace.
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
                horizontal_aperture=20.955,
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
    ):
        # Fail fast on missing optional encoders: ``save()`` / ``_downscale_to_width``
        # need moviepy and Pillow, and discovering they are absent only *after* a
        # full (expensive) rollout would discard every captured frame.
        self._check_optional_deps()

        self.env = env
        self.unwrapped = env.unwrapped
        self.device = self.unwrapped.device
        self.num_envs = int(self.unwrapped.scene.num_envs)

        self.env_view = env_view if env_view is not None else EnvView()
        self.global_view = global_view if global_view is not None else GlobalView()
        # ``ceil(sqrt(N))`` keeps the tiled grid roughly square for any env count
        # (e.g. 4 -> 2x2, 36 -> 6x6). Each tile uses the same env-relative framing
        # (offset and distance), so every panel is composed consistently regardless
        # of env count.
        self.grid_cols = grid_cols if grid_cols is not None else math.ceil(math.sqrt(self.num_envs))
        # Cap on the composed grid width [px]; the grid is downscaled to fit so a
        # large env count (e.g. 6x6x480px = 2880px) stays a reasonable video size.
        self.max_grid_width = max_grid_width
        # Scene-sensor key to look up the registered tiled camera. Must match the
        # name used with ``Scene.add_sensor`` (defaults to :attr:`ENV_CAM_NAME`).
        self._env_cam_name = env_cam_name if env_cam_name is not None else self.ENV_CAM_NAME

        # The global view reads the viewport via ``env.render()``, which only
        # returns frames under ``render_mode="rgb_array"``.
        assert self.env.render_mode == "rgb_array", (
            "SynchronizedVisualizer needs the env built with render_mode='rgb_array' "
            f"for the global viewport view; got render_mode={self.env.render_mode!r}."
        )

        self._env_camera: Camera | None = None
        self._buffers = _CaptureBuffers()
        self._initialized = False

        self._resolve_env_camera()

    def _resolve_env_camera(self) -> None:
        """Look up the scene-registered per-env :class:`~isaaclab.sensors.TiledCamera`.

        The camera must have been added to the scene before build (see
        :meth:`build_env_camera_cfg`); it is updated by ``env.step``, so the
        visualizer never has to render or update it itself.
        """
        scene_sensors = self.unwrapped.scene.sensors
        assert self._env_cam_name in scene_sensors, (
            f"No scene sensor named {self._env_cam_name!r} found on the env "
            f"(have: {sorted(scene_sensors.keys())}). Register it before build via "
            "Scene.add_sensor(name, SynchronizedVisualizer.build_env_camera_cfg(env_view, name))."
        )
        self._env_camera = scene_sensors[self._env_cam_name]

    def initialize(self) -> None:
        """Pose the global viewport camera. Call once after ``env.reset()``, before :meth:`capture`.

        The per-env tiled camera needs no setup here: it is a scene sensor posed
        by its baked ``OffsetCfg`` and updated by ``env.step``.
        """
        assert not self._initialized, "initialize() already called; it must run exactly once."
        self._position_global_camera()
        self._initialized = True

    def reposition(self) -> None:
        """Re-apply the global viewport pose. Call after an ``env.reset()`` that moved env origins.

        Only the global view needs this; the per-env tiled camera is restored
        automatically from its baked ``OffsetCfg`` on reset.
        """
        self._position_global_camera()

    def _position_global_camera(self) -> None:
        """Drive the env's viewport camera to the resolved global eye/target.

        Uses the env's ``viewport_camera_controller`` (the canonical Arena path,
        also used by ``reapply_viewer_cfg``), subtracting its ``viewer_origin`` so
        our world-frame pose lands correctly regardless of ``ViewerCfg.origin_type``.
        """
        origins = self.unwrapped.scene.env_origins.to(self.device)  # (N, 3)
        eye, target = self._resolve_global_pose(origins)

        vcc = getattr(self.unwrapped, "viewport_camera_controller", None)
        assert vcc is not None, (
            "Env has no viewport_camera_controller, so the global viewport view cannot be posed. "
            "Build with render_mode='rgb_array' and run with --enable_cameras."
        )
        viewer_origin = vcc.viewer_origin.detach().cpu().numpy().reshape(3)
        eye_np = eye.detach().cpu().numpy() - viewer_origin
        target_np = target.detach().cpu().numpy() - viewer_origin
        vcc.update_view_location(eye=eye_np.tolist(), lookat=target_np.tolist())

    def _resolve_global_pose(self, origins):
        """Compute the global camera (eye, target), auto-framing from the env-origin bounding box when ``GlobalView.eye`` is unset."""
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
        """Append one frame per view to the internal buffers. Call once per ``env.step``.

        The tiled camera is read from its scene-sensor buffer. The global view
        calls ``env.render()``, which is a cheap annotator read here *because* the
        scene-registered RTX tiled camera makes ``has_rtx_sensors`` true — so
        ``env.step`` already rendered and ``render()`` skips its own ``sim.render()``.
        (Without an RTX scene sensor, ``env.render()`` would trigger a render pass.)
        No manual ``camera.update()`` is needed, keeping both views in sync.
        """
        assert self._initialized, "Call initialize() before capture()."
        assert self._env_camera is not None

        env_rgb = self._read_rgb(self._env_camera, "per-env")  # (N, H, W, 3)
        per_env = []
        for i in range(self.num_envs):
            img = video_io.to_uint8(env_rgb[i])
            per_env.append(img)
            self._buffers.per_env_frames.setdefault(i, []).append(img)
        grid = self._tile(per_env, self.grid_cols)
        if self.max_grid_width is not None and grid.shape[1] > self.max_grid_width:
            grid = self._downscale_to_width(grid, self.max_grid_width)
        self._buffers.grid_frames.append(grid)

        global_rgb = self.env.render()
        if global_rgb is not None:
            self._buffers.global_frames.append(video_io.to_uint8(global_rgb))

    @staticmethod
    def _read_rgb(camera, label: str):
        """Return the camera's ``rgb`` output, with a clear error if not yet ready."""
        output = camera.data.output
        if "rgb" not in output:
            raise RuntimeError(
                f"{label} camera has no 'rgb' output yet (available: {list(output.keys())}). "
                "The annotator has not delivered a frame — ensure env.step() has been called "
                "and warm up with a few steps before capturing."
            )
        return output["rgb"]

    @staticmethod
    def _downscale_to_width(img: np.ndarray, max_width: int) -> np.ndarray:
        """Downscale ``img`` to ``max_width`` (keeping aspect, even dims for h264)."""
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
        """Write the captured frames to ``output_dir`` as mp4 (and optional gif).

        Args:
            output_dir: Directory to write outputs into (created if missing).
            fps: Frames per second for the encoded videos.
            name_prefix: Prefix for output filenames.
            save_per_env: Also write one mp4 per environment.
            also_gif: Also write animated gifs alongside the mp4s.

        Returns:
            Mapping of logical output name to written file path.
        """
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

        return written

    def close(self) -> None:
        """Release the sensor references and free the captured frame buffers."""
        self._env_camera = None
        self._initialized = False
        # Drop accumulated frames (potentially GBs for long, many-env rollouts).
        self._buffers = _CaptureBuffers()
