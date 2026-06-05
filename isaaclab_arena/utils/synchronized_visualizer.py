# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Synchronized, general-purpose visualization for parallel Arena environments.

Produces two complementary views of a running simulation:

* **Global view** — a single camera that frames the *whole* simulation
  (all parallel envs at once), like a bird's-eye / isometric grid shot.
* **Per-env grid view** — one camera per environment, tiled into a grid so each
  env gets its own panel. By default this is a scene-registered
  :class:`~isaaclab.sensors.TiledCamera` (injected pre-build via
  :meth:`SynchronizedVisualizer.add_env_camera_to_cfg`), with a standalone
  multi-prim :class:`~isaaclab.sensors.Camera` fallback.

The per-env camera is anchored to each env base (its pose is relative to
``scene.env_origins``) and the global camera is positioned relative to the
env-origin bounding box, so placement follows the actual scene rather than
hardcoded world coordinates. The caller specifies:

* :class:`GlobalView` — eye/target in world coordinates (or ``None`` to
  auto-frame from the env-origin bounding box).
* :class:`EnvView` — eye/target offsets expressed *relative to each env origin*.

Typical usage (inside a rollout, after ``sim.render()``)::

    viz = SynchronizedVisualizer(
        env,
        env_view=EnvView(eye_offset=(0.9, 0.0, 1.0), lookat_offset=(0.0, 0.0, 0.1)),
        global_view=GlobalView(),  # auto-frame
    )
    viz.initialize()
    for _ in range(num_steps):
        env.step(actions)
        env.unwrapped.sim.render()
        viz.capture()
    viz.save("/workspaces/isaaclab_arena/outputs/sync_viz")
"""

from __future__ import annotations

import logging
import math
import numpy as np
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# Cameras and torch are imported lazily inside methods so this module can be
# imported before the Isaac Sim ``SimulationApp`` is launched.
if TYPE_CHECKING:
    import gymnasium

    from isaaclab.sensors.camera import Camera

    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg

logger = logging.getLogger(__name__)


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

    use_tiled_camera: bool = True
    """Render the per-env grid with a scene-registered :class:`~isaaclab.sensors.TiledCamera`.

    This is the efficient, canonical path: all per-env cameras render into one
    tiled buffer. It requires injecting the camera into the env's scene config
    *before* the env is built (see :meth:`SynchronizedVisualizer.add_env_camera_to_cfg`).
    When ``False`` (or when no injected camera is found on the built env), the
    visualizer falls back to a standalone multi-prim :class:`~isaaclab.sensors.Camera`
    created post-build.
    """

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

    focal_length: float = 24.0
    """Pinhole focal length [mm]."""

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
    """Attach a global camera and a per-env tiled camera to a built Arena env.

    Args:
        env: A built (gym-wrapped) Arena environment. ``env.unwrapped`` must
            expose ``sim``, ``device``, ``scene`` (with ``env_origins`` and
            ``num_envs``), and ``step_dt``.
        env_view: Per-environment view configuration.
        global_view: Whole-simulation view configuration. Defaults to an
            auto-framed :class:`GlobalView`.
        grid_cols: Number of columns in the per-env grid. Defaults to
            ``ceil(sqrt(num_envs))`` (roughly square).
        prim_root: USD scope under which the visualizer creates its camera
            prims.
    """

    ENV_CAM_NAME = "sync_viz_env_cam"
    """Scene-sensor key for the injected per-env :class:`~isaaclab.sensors.TiledCamera`."""

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
    def add_env_camera_to_cfg(
        env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
        env_view: EnvView | None = None,
        name: str | None = None,
    ) -> str:
        """Inject a per-env :class:`~isaaclab.sensors.TiledCamera` into a scene cfg.

        Call *before* the env is built (between ``builder.compose_manager_cfg()``
        and ``builder.make_registered(env_cfg=cfg)``). Registering as a scene
        sensor sets up tiled rendering once — avoiding the all-white frames a
        second render product causes once the sim is already playing.

        The prim is created per env namespace (anchored to the env base). A
        scene-managed camera recomputes its pose from ``parent × offset`` every
        step, overriding runtime ``set_world_poses_from_view``, so the eye/target
        framing is baked into ``OffsetCfg`` here. The env prim is static with
        identity rotation, so the offset equals the desired pose relative to the
        env origin.

        Args:
            env_cfg: A composed manager-based env cfg whose ``scene`` will be
                extended with the camera sensor.
            env_view: Per-env view config (resolution, focal length, eye/target).
            name: Scene-sensor key. Defaults to :attr:`ENV_CAM_NAME`. If you pass
                a custom name here, pass the same value as ``env_cam_name`` to
                :class:`SynchronizedVisualizer` so it can find the camera;
                otherwise the visualizer falls back to a standalone Camera.

        Returns:
            The scene-sensor key the camera was registered under.
        """
        import isaaclab.sim as sim_utils
        from isaaclab.sensors.camera import TiledCameraCfg

        from isaaclab_arena.utils.configclass import combine_configclass_instances, make_configclass

        view = env_view if env_view is not None else EnvView()
        key = name if name is not None else SynchronizedVisualizer.ENV_CAM_NAME

        pos, quat_wxyz = SynchronizedVisualizer._lookat_offset(view.eye_offset, view.lookat_offset)
        camera_cfg = TiledCameraCfg(
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
        holder = make_configclass(f"SyncVizCameraCfg_{key}", [(key, TiledCameraCfg, camera_cfg)])()
        # combine_configclass_instances takes configclass *instances* (it calls
        # ``type(i)`` internally); its ``*args: type`` annotation is imprecise.
        env_cfg.scene = combine_configclass_instances("SceneCfg", env_cfg.scene, holder)  # pyright: ignore
        return key

    def __init__(
        self,
        env: gymnasium.Env,
        env_view: EnvView | None = None,
        global_view: GlobalView | None = None,
        grid_cols: int | None = None,
        prim_root: str = "/World/SyncViz",
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
        self.prim_root = prim_root
        # Cap on the composed grid width [px]; the grid is downscaled to fit so a
        # large env count (e.g. 6x6x480px = 2880px) stays a reasonable video size.
        self.max_grid_width = max_grid_width
        # Scene-sensor key to look up the injected tiled camera. Must match the
        # ``name`` passed to :meth:`add_env_camera_to_cfg` (defaults to
        # :attr:`ENV_CAM_NAME` on both sides).
        self._env_cam_name = env_cam_name if env_cam_name is not None else self.ENV_CAM_NAME

        self._env_camera: Camera | None = None
        self._global_camera: Camera | None = None
        self._buffers = _CaptureBuffers()
        self._initialized = False

        self._build_sensors()

    def _build_sensors(self) -> None:
        """Resolve the per-env camera and create the standalone global camera.

        Per-env: prefer the scene-registered :class:`~isaaclab.sensors.TiledCamera`
        (from :meth:`add_env_camera_to_cfg`), else fall back to a standalone
        multi-prim :class:`~isaaclab.sensors.Camera`. The global camera is always a
        single standalone Camera, which does not conflict with tiled rendering.
        """
        import isaaclab.sim as sim_utils
        from isaaclab.sensors.camera import Camera, CameraCfg

        def _pinhole(focal_length: float) -> sim_utils.PinholeCameraCfg:
            return sim_utils.PinholeCameraCfg(
                focal_length=focal_length,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            )

        self._using_tiled = False
        scene_sensors = self.unwrapped.scene.sensors
        if self.env_view.use_tiled_camera and self._env_cam_name in scene_sensors:
            # Efficient path: the tiled camera was injected into the scene cfg
            # pre-build, so Isaac Lab already set up its render product correctly.
            self._env_camera = scene_sensors[self._env_cam_name]
            self._using_tiled = True
        else:
            if self.env_view.use_tiled_camera:
                # Misconfiguration (use_tiled_camera=True but the camera was never
                # injected, or under a different name) — fall back gracefully but
                # surface it loudly so the setup mistake is visible.
                logger.warning(
                    "No scene tiled camera %r found; falling back to a standalone multi-prim "
                    "Camera. Call SynchronizedVisualizer.add_env_camera_to_cfg(env_cfg, name=%r) "
                    "before building to use the efficient TiledCamera path.",
                    self._env_cam_name,
                    self._env_cam_name,
                )
            # Fallback: one mount Xform per env, matched by a single regex prim
            # path. A multi-prim Camera returns a batched [num_envs, H, W, C] buffer.
            for i in range(self.num_envs):
                sim_utils.create_prim(f"{self.prim_root}/EnvCam_{i:03d}", "Xform")
            self._env_camera = Camera(
                cfg=CameraCfg(
                    prim_path=f"{self.prim_root}/EnvCam_.*/CameraSensor",
                    update_period=0,
                    width=self.env_view.width,
                    height=self.env_view.height,
                    data_types=["rgb"],
                    spawn=_pinhole(self.env_view.focal_length),
                )
            )

        sim_utils.create_prim(f"{self.prim_root}/GlobalCam", "Xform")
        self._global_camera = Camera(
            cfg=CameraCfg(
                prim_path=f"{self.prim_root}/GlobalCam/CameraSensor",
                update_period=0,
                width=self.global_view.width,
                height=self.global_view.height,
                data_types=["rgb"],
                spawn=_pinhole(self.global_view.focal_length),
            )
        )

    def initialize(self) -> None:
        """Reset the sim so the new sensors initialize, then position cameras.

        Must be called once after construction and before :meth:`capture`.

        .. warning::
            This calls ``sim.reset()``, which re-runs scene initialization and
            **discards any placement / randomization state** applied between
            building the env and calling this method. Set up any custom initial
            state (e.g. via ``env.reset()``) *after* :meth:`initialize`, not
            before.
        """
        assert self._env_camera is not None and self._global_camera is not None
        sim = self.unwrapped.sim
        sim.reset()
        self._env_camera.update(dt=0.0)
        self._global_camera.update(dt=0.0)
        self._position_cameras()
        self._initialized = True

    def reposition(self) -> None:
        """Re-apply the global camera pose. Call after every ``env.reset()``.

        Only the standalone global camera is re-posed here. When the per-env grid
        uses the scene-registered :class:`~isaaclab.sensors.TiledCamera`, its pose
        is governed entirely by the baked ``OffsetCfg`` (see
        :meth:`add_env_camera_to_cfg`) and is restored automatically on reset, so
        :meth:`_position_cameras` skips it. In the standalone-fallback case the
        per-env camera *is* re-posed here as well.
        """
        self._position_cameras()

    def _position_cameras(self) -> None:
        import torch

        assert self._env_camera is not None and self._global_camera is not None
        origins = self.unwrapped.scene.env_origins.to(self.device)  # (N, 3)

        # The scene tiled camera is posed via its baked ``OffsetCfg`` (see
        # ``add_env_camera_to_cfg``); only the standalone fallback needs an
        # explicit eye/target here.
        if not self._using_tiled:
            eye_off = torch.tensor(self.env_view.eye_offset, device=self.device, dtype=torch.float32)
            look_off = torch.tensor(self.env_view.lookat_offset, device=self.device, dtype=torch.float32)
            self._env_camera.set_world_poses_from_view(origins + eye_off, origins + look_off)

        global_eye, global_target = self._resolve_global_pose(origins)
        self._global_camera.set_world_poses_from_view(
            global_eye.unsqueeze(0),
            global_target.unsqueeze(0),
        )

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
        """Grab one frame from each camera and append to the internal buffers.

        Call after ``sim.render()`` for the current step. Under
        ``render_mode=None`` the env is not rendered automatically by
        ``env.step``, so an explicit ``sim.render()`` is required first. Cameras
        are updated here.
        """
        assert self._initialized, "Call initialize() before capture()."
        assert self._env_camera is not None and self._global_camera is not None
        dt = float(getattr(self.unwrapped, "step_dt", 0.0))

        self._env_camera.update(dt=dt)
        self._global_camera.update(dt=dt)

        env_rgb = self._read_rgb(self._env_camera, "per-env")  # (N, H, W, 3)
        per_env = []
        for i in range(self.num_envs):
            img = self._to_uint8(env_rgb[i])
            per_env.append(img)
            self._buffers.per_env_frames.setdefault(i, []).append(img)
        grid = self._tile(per_env, self.grid_cols)
        if self.max_grid_width is not None and grid.shape[1] > self.max_grid_width:
            grid = self._downscale_to_width(grid, self.max_grid_width)
        self._buffers.grid_frames.append(grid)

        global_rgb = self._read_rgb(self._global_camera, "global")[0]
        self._buffers.global_frames.append(self._to_uint8(global_rgb))

    @staticmethod
    def _read_rgb(camera, label: str):
        """Return the camera's ``rgb`` output, with a clear error if not yet ready."""
        output = camera.data.output
        if "rgb" not in output:
            raise RuntimeError(
                f"{label} camera has no 'rgb' output yet (available: {list(output.keys())}). "
                "The annotator has not delivered a frame — ensure the scene has been rendered "
                "(sim.render()) and warm up with a few steps before calling capture()."
            )
        return output["rgb"]

    @staticmethod
    def _to_uint8(frame) -> np.ndarray:
        arr = frame.detach().cpu().numpy()
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr[:, :, :3]

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
            self._write_video(self._buffers.global_frames, path, fps)
            written["global"] = path
            if also_gif:
                gif_path = os.path.join(output_dir, f"{name_prefix}_global.gif")
                self._write_gif(self._buffers.global_frames, gif_path, fps)
                written["global_gif"] = gif_path

        if self._buffers.grid_frames:
            path = os.path.join(output_dir, f"{name_prefix}_grid.mp4")
            self._write_video(self._buffers.grid_frames, path, fps)
            written["grid"] = path
            if also_gif:
                gif_path = os.path.join(output_dir, f"{name_prefix}_grid.gif")
                self._write_gif(self._buffers.grid_frames, gif_path, fps)
                written["grid_gif"] = gif_path

        if save_per_env:
            for env_idx, frames in self._buffers.per_env_frames.items():
                if not frames:
                    continue
                path = os.path.join(output_dir, f"{name_prefix}_env{env_idx:03d}.mp4")
                self._write_video(frames, path, fps)
                written[f"env{env_idx:03d}"] = path

        return written

    @staticmethod
    def _write_video(frames: list[np.ndarray], path: str, fps: int) -> None:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(path, logger=None, audio=False)
        del clip

    @staticmethod
    def _write_gif(frames: list[np.ndarray], path: str, fps: int) -> None:
        from PIL import Image

        pil_frames = [Image.fromarray(f) for f in frames]
        duration_ms = int(1000 / max(fps, 1))
        pil_frames[0].save(
            path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
        )

    def close(self) -> None:
        """Release references to the underlying sensors."""
        self._env_camera = None
        self._global_camera = None
        self._initialized = False
