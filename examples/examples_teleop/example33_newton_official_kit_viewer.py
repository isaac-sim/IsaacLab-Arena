"""Live Isaac Sim / Kit mirror of Newton's official Franka cube-stacking demo.

This is a hybrid viewer:

* Newton runs the real simulation, IK, contact, and grasping logic.
* Isaac Sim / Kit renders a lightweight mirror of the Newton bodies in real time.

It deliberately does not use IsaacLab-Arena's ManagerBased env or action bridge,
so the physics path stays close to example31 / Newton's official
``example_ik_cube_stacking.py``.

Run inside the IsaacLab-Arena Docker container:

    cd /workspaces/isaaclab_arena
    /isaac-sim/python.sh examples/examples_teleop/example33_newton_official_kit_viewer.py
"""

from __future__ import annotations

import argparse
import contextlib
import time

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror Newton official cube stacking into Isaac Sim / Kit.")
    parser.add_argument("--world-count", type=int, default=1, help="Number of Newton worlds.")
    parser.add_argument("--num-frames", type=int, default=1800, help="Maximum rendered frames.")
    parser.add_argument("--load-visuals", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-mujoco-contacts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--newton-verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true", default=True)
    parser.add_argument("--mirror-robot-bodies", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--robot-body-size", type=float, default=0.035)
    parser.add_argument("--sleep", action=argparse.BooleanOptionalAction, default=True)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=False, enable_cameras=False)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _set_xform(prim, pos, quat_xyzw) -> None:
    from pxr import Gf, UsdGeom

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    transform = Gf.Matrix4d()
    transform.SetRotate(Gf.Quatd(float(quat_xyzw[3]), Gf.Vec3d(*[float(v) for v in quat_xyzw[:3]])))
    transform.SetTranslateOnly(Gf.Vec3d(*[float(v) for v in pos]))
    xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(transform)


def _create_cube(stage, path: str, size: float, color) -> object:
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(size)

    material_path = f"{path}_material"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(material)
    return cube.GetPrim()


def _create_kit_scene(example):
    from pxr import Gf, UsdGeom, UsdLux
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    stage.DefinePrim("/World", "Xform")

    light = UsdLux.DomeLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(1200.0)

    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    _set_xform(camera.GetPrim(), (0.8, -1.4, 0.75), (0.55, 0.0, 0.0, 0.84))

    table = _create_cube(stage, "/World/NewtonMirror/Table", 1.0, (0.25, 0.25, 0.25))
    _set_xform(table, tuple(example.table_pos), (0.0, 0.0, 0.0, 1.0))
    UsdGeom.Xformable(table).AddScaleOp().Set(Gf.Vec3d(0.8, 0.8, example.table_height))

    body_prims = []
    if args_cli.mirror_robot_bodies:
        for body_id in range(example.robot_body_count):
            prim = _create_cube(
                stage,
                f"/World/NewtonMirror/RobotBody_{body_id:02d}",
                args_cli.robot_body_size,
                (0.9, 0.75, 0.18),
            )
            body_prims.append(prim)

    cube_prims = []
    for cube_id in range(example.cube_count):
        prim = _create_cube(
            stage,
            f"/World/NewtonMirror/Cube_{cube_id}",
            example.cube_size,
            ((0.8, 0.2, 0.2), (0.2, 0.8, 0.2), (0.2, 0.2, 0.8))[cube_id % 3],
        )
        cube_prims.append(prim)

    return body_prims, cube_prims


def _update_kit_scene(example, body_prims, cube_prims) -> None:
    body_q = example.state_0.body_q.numpy()
    num_bodies_per_world = example.num_bodies_per_world

    if body_prims:
        for body_id, prim in enumerate(body_prims):
            pose = body_q[body_id]
            _set_xform(prim, pose[:3], pose[3:])

    for cube_id, prim in enumerate(cube_prims):
        body_id = example.robot_body_count + cube_id
        pose = body_q[body_id]
        _set_xform(prim, pose[:3], pose[3:])

    if args_cli.world_count > 1:
        print(
            f"[WARN] Kit mirror currently displays world 0 only; Newton is simulating {args_cli.world_count} worlds.",
            flush=True,
        )


def main() -> None:
    import warp as wp
    import newton.viewer

    from example_newton_official_cube_stacking_utils import ArenaDockerCubeStackingExample

    if args_cli.quiet:
        wp.config.quiet = True

    viewer = newton.viewer.ViewerNull(num_frames=args_cli.num_frames)
    newton_args = argparse.Namespace(
        world_count=args_cli.world_count,
        headless=True,
        verbose=args_cli.newton_verbose,
        use_mujoco_contacts=args_cli.use_mujoco_contacts,
        load_visuals=args_cli.load_visuals,
    )
    example = ArenaDockerCubeStackingExample(viewer, newton_args)
    body_prims, cube_prims = _create_kit_scene(example)

    print("[INFO] Isaac Sim Kit mirror started. Newton is running the official IK cube-stacking physics.", flush=True)

    frame = 0
    while simulation_app.is_running() and frame < args_cli.num_frames:
        example.step()
        _update_kit_scene(example, body_prims, cube_prims)
        simulation_app.update()
        frame += 1
        if args_cli.sleep:
            time.sleep(example.frame_dt)

    print(f"[INFO] Finished {frame} frames.", flush=True)


if __name__ == "__main__":
    try:
        with contextlib.suppress(KeyboardInterrupt):
            main()
    finally:
        simulation_app.close()
