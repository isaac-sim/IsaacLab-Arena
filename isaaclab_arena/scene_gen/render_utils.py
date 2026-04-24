"""Render utilities for scene visualization.

Adapted from RoboLab's render_utils.py for AppLauncher (instead of SimulationApp).

Usage:
    render_scene_to_image(usd_path, output_path)
    render_all_scenes(scene_folder, output_folder)
"""

from __future__ import annotations

import os
from pathlib import Path


def render_scene_to_image(
    usd_path: str,
    output_dir: str,
    resolution: tuple[int, int] = (640, 480),
    skip_frames: int = 100,
    settle_frames: int = 300,
    add_lighting: bool = True,
    hide_franka_table: bool = True,
    camera_position: tuple[float, float, float] = (-0.3, 0.3, 0.7),
    camera_target: tuple[float, float, float] = (0.5, 0.0, 0.0),
) -> str | None:
    """Render a USD scene to a PNG image.

    Adapted from RoboLab's render_stage_frame(). Requires non-headless mode.

    Args:
        settle_frames: Number of physics frames to run before capture (~5s at 60Hz = 300).
                       Objects drop onto the table and settle into stable poses.
        hide_franka_table: If True, make the franka_table invisible during render.
    """
    import omni.usd
    import omni.kit.app
    from pxr import Gf, UsdGeom, UsdLux

    os.makedirs(output_dir, exist_ok=True)
    usd_path = os.path.abspath(str(usd_path))

    # Open the stage
    ctx = omni.usd.get_context()
    ctx.open_stage(usd_path)

    for _ in range(30):
        omni.kit.app.get_app().update()

    stage = ctx.get_stage()
    if stage is None:
        print(f"[Render] Failed to open: {usd_path}")
        return None

    # Hide franka_table (robot mount) so it doesn't clutter the scene render
    if hide_franka_table:
        for prim in stage.Traverse():
            if prim.GetName() == "franka_table":
                UsdGeom.Imageable(prim).MakeInvisible()

    # Disable physics debug visualization (hides joint markers / green dots)
    try:
        import carb.settings
        settings = carb.settings.get_settings()
        settings.set("/physics/visualizationDisplayJoints", False)
        settings.set("/physics/visualizationDisplayJointsDefault", False)
        settings.set("/persistent/physics/visualizationDisplayJoints", False)
    except Exception:
        pass

    # Physics settling: let objects drop onto table and reach stable poses
    if settle_frames > 0:
        try:
            import omni.timeline
            timeline = omni.timeline.get_timeline_interface()
            timeline.play()
            for _ in range(settle_frames):
                omni.kit.app.get_app().update()
            timeline.pause()
            print(f"[Render] Physics settled for {settle_frames} frames")
        except Exception as e:
            print(f"[Render] Physics settling skipped: {e}")

    # Create a new render camera (modifying /OmniverseKit_Persp doesn't work)
    cam_path = "/World/RenderCam"
    if not stage.GetPrimAtPath(cam_path).IsValid():
        cam = UsdGeom.Camera.Define(stage, cam_path)
        cam.CreateFocalLengthAttr(24.0)
        cam.CreateHorizontalApertureAttr(20.955)

    # Set camera look-at transform
    eye = Gf.Vec3d(*camera_position)
    target = Gf.Vec3d(*camera_target)
    up = Gf.Vec3d(0, 0, 1)
    forward = (target - eye).GetNormalized()
    right = Gf.Cross(forward, up).GetNormalized()
    actual_up = Gf.Cross(right, forward)

    mat = Gf.Matrix4d(1)
    mat.SetRow(0, Gf.Vec4d(right[0], right[1], right[2], 0))
    mat.SetRow(1, Gf.Vec4d(actual_up[0], actual_up[1], actual_up[2], 0))
    mat.SetRow(2, Gf.Vec4d(-forward[0], -forward[1], -forward[2], 0))
    mat.SetRow(3, Gf.Vec4d(eye[0], eye[1], eye[2], 1))

    cam_xform = UsdGeom.Xformable(stage.GetPrimAtPath(cam_path))
    cam_xform.ClearXformOpOrder()
    cam_xform.AddTransformOp().Set(mat)

    for _ in range(10):
        omni.kit.app.get_app().update()

    # Switch viewport to our camera
    from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
    vp = get_active_viewport()
    if vp:
        vp.set_active_camera(cam_path)

    for _ in range(10):
        omni.kit.app.get_app().update()

    # Add lighting via USD (exposure attributes for RTX brightness)
    if add_lighting:
        from pxr import Sdf
        dist = UsdLux.DistantLight.Define(stage, "/World/render_distant_light")
        dist.CreateIntensityAttr(1.0)
        dist.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        dist.GetPrim().CreateAttribute("inputs:exposure", Sdf.ValueTypeNames.Float).Set(10.0)
        dist.GetPrim().CreateAttribute("inputs:angle", Sdf.ValueTypeNames.Float).Set(30.0)

        dome = UsdLux.DomeLight.Define(stage, "/World/render_dome_light")
        dome.CreateIntensityAttr(1.0)
        dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        dome.GetPrim().CreateAttribute("inputs:exposure", Sdf.ValueTypeNames.Float).Set(9.0)

    # Render frames for stabilization
    for _ in range(skip_frames):
        omni.kit.app.get_app().update()

    # Capture viewport
    output_name = os.path.splitext(os.path.basename(usd_path))[0]
    output_path = os.path.join(output_dir, f"{output_name}.png")

    if vp:
        capture_viewport_to_file(vp, output_path)
        for _ in range(10):
            omni.kit.app.get_app().update()
        if os.path.exists(output_path):
            print(f"[Render] Saved: {output_path}")
            return output_path

    print(f"[Render] Capture failed for {usd_path}")
    return None


def render_all_scenes(
    scene_folder: str,
    output_folder: str,
    resolution: tuple[int, int] = (640, 480),
    view: str = "angled",
    ignore_files: list[str] | None = None,
) -> list[str]:
    """Render all .usda scenes in a folder to PNG images."""
    views = {
        "front": ((1.5, 0.0, 1.5), (0.5, 0.0, 0.0)),
        "angled": ((-0.5, 0.3, 1.5), (0.5, 0.0, 0.0)),  # RoboLab screenshot view
        "top": ((0.5, 0.0, 1.5), (0.5, 0.0, 0.0)),
    }
    cam_pos, cam_target = views.get(view, views["angled"])

    scene_folder = os.path.abspath(scene_folder)
    usd_files = []
    for f in sorted(os.listdir(scene_folder)):
        if f.endswith((".usda", ".usd", ".usdc")):
            if ignore_files and f in ignore_files:
                continue
            usd_files.append(os.path.join(scene_folder, f))

    print(f"[Render] Found {len(usd_files)} scenes in {scene_folder}")

    rendered = []
    for usd_file in usd_files:
        result = render_scene_to_image(
            usd_file, output_folder,
            resolution=resolution,
            camera_position=cam_pos,
            camera_target=cam_target,
        )
        if result:
            rendered.append(result)

    print(f"[Render] Rendered {len(rendered)}/{len(usd_files)} scenes")
    return rendered