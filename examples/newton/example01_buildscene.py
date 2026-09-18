# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""最小 Newton 刚体仿真：一个立方体落到静态台面上，并用 Newton viewer 显示。

使用方法：

    .venv/bin/python examples/newton/example01_buildscene.py --num_steps 500 --keep_open
    .venv/bin/python examples/newton/example01_buildscene.py --viz kit --num_steps 500
    .venv/bin/python examples/newton/example01_buildscene.py --viz none --num_steps 500
"""

import argparse
import contextlib
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Newton 最小刚体场景示例。")
parser.add_argument("--num_steps", type=int, default=500, help="要执行的物理步数。")
parser.add_argument(
    "--keep_open",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="仿真结束后保持 Newton viewer 窗口打开。",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=["newton"], device="cuda:0")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    """创建 Newton 场景并执行固定数量的物理步。"""
    import newton
    import newton.solvers

    # IsaacLab Newton 0.13.6 imports the pre-1.5 name. Newton 1.5 moved the
    # same bit-mask enum to ``newton.ModelFlags``. This must run after Kit
    # starts so Newton does not preload a conflicting pxr binding.
    if not hasattr(newton.solvers, "SolverNotifyFlags"):
        newton.solvers.SolverNotifyFlags = newton.ModelFlags

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from isaaclab.sim import SimulationCfg, build_simulation_context

    physics_cfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(solver="newton", integrator="implicitfast"),
        num_substeps=1,
    )
    sim_cfg = SimulationCfg(device=args_cli.device, dt=1.0 / 1200.0, physics=physics_cfg)
    visualizers = [] if args_cli.visualizer == ["none"] else args_cli.visualizer

    with build_simulation_context(
        sim_cfg=sim_cfg,
        add_lighting=True,
        visualizers=visualizers,
    ) as sim:
        sim.set_camera_view(eye=(2.0, -2.0, 1.5), target=(0.0, 0.0, 0.3))

        table = RigidObject(
            RigidObjectCfg(
                prim_path="/World/Table",
                spawn=sim_utils.CuboidCfg(
                    size=(1.2, 1.2, 0.1),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
            )
        )
        cube = RigidObject(
            RigidObjectCfg(
                prim_path="/World/Cube",
                spawn=sim_utils.CuboidCfg(
                    size=(0.15, 0.15, 0.15),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.8)),
            )
        )

        sim.reset()
        table.reset()
        cube.reset()
        print(f"[INFO] Newton 场景已就绪，开始执行 {args_cli.num_steps} 个物理步。", flush=True)

        for step in range(args_cli.num_steps):
            if sim.visualizers and not any(
                visualizer.is_running() and not visualizer.is_closed for visualizer in sim.visualizers
            ):
                print(f"[INFO] visualizer 在第 {step} 步前关闭。", flush=True)
                break
            table.write_data_to_sim()
            cube.write_data_to_sim()
            sim.step()
            table.update(sim.get_physics_dt())
            cube.update(sim.get_physics_dt())
            if step == 0:
                print("[INFO] 已完成第 1 个 Newton 物理步。", flush=True)
            time.sleep(sim.get_physics_dt())

        print("[INFO] Newton 仿真结束。", flush=True)
        if args_cli.keep_open:
            print("关闭 visualizer 窗口或按 Ctrl+C 退出。", flush=True)
            while sim.visualizers and any(
                visualizer.is_running() and not visualizer.is_closed for visualizer in sim.visualizers
            ):
                table.write_data_to_sim()
                cube.write_data_to_sim()
                sim.step()
                table.update(sim.get_physics_dt())
                cube.update(sim.get_physics_dt())
                time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    try:
        with contextlib.suppress(KeyboardInterrupt):
            main()
    finally:
        simulation_app.close()
