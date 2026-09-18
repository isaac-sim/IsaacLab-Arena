# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""IsaacLab 场景中的 Newton SDF Hydroelastic 碰撞示例。

IsaacLab 负责启动 Kit、创建 USD 场景并同步刚体状态；物理步进、网格 SDF 查询和
Hydroelastic 接触求解均由 Newton 完成。SDF 在 Newton ``MODEL_INIT`` 回调中烘焙：
此时 IsaacLab 已经从 USD 创建 ``ModelBuilder``，但模型尚未定型。

运行方法：

    .venv/bin/python examples/sdf/sdf_isaaclab_newton.py
    .venv/bin/python examples/sdf/sdf_isaaclab_newton.py --viz none --num-steps 480
"""

import argparse
import contextlib
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


TABLE_HALF_EXTENTS = (1.6, 1.6, 0.05)
TABLE_POSITION = (0.0, 0.0, -0.05)
JUG_USD_PATH = Path(
    "/home/magengyu/IsaacLab-Arena/scene/fstylejug_a01/fstylejug_a01_inst_physx.usd"
)
CONTAINER_USD_PATH = Path(
    "/home/magengyu/IsaacLab-Arena/scene/open_cardboard_box_hydroelastic.usda"
)
JUG_POSITION = (0.45, -0.25, 0.01)
CONTAINER_POSITION = (0.45, 0.22, 0.02)
MESH_SDF_MAX_RESOLUTION = 256
MESH_SDF_NARROW_BAND_RANGE = (-0.04, 0.04)
MESH_SDF_CACHE_DIR = Path(__file__).resolve().parents[2] / "scene" / "tmp"
TABLE_SHAPE_MARGIN = 0.005
TABLE_COLLISION_GAP = 0.010
JUG_SHAPE_MARGIN = 0.002
JUG_COLLISION_GAP = 0.002
CONTAINER_SHAPE_MARGIN = 0.002
CONTAINER_COLLISION_GAP = 0.010
CONTACT_STIFFNESS = 250000.0
CONTACT_DAMPING = 1000.0
HYDROELASTIC_CONTACT_STIFFNESS = 1.0e11
RIGID_CONTACT_MAX = 4096
FRAME_RATE_HZ = 480
SIM_SUBSTEPS = 8


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the IsaacLab Newton SDF example."""
    parser = argparse.ArgumentParser(
        description="IsaacLab + Newton 网格 SDF 碰撞示例。"
    )
    parser.add_argument(
        "--num-steps", type=int, default=2400, help="运行的 IsaacLab 物理步数。"
    )
    parser.add_argument(
        "--log-every", type=int, default=240, help="打印水壶位置的步数间隔。"
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(visualizer=["newton"], device="cuda:0")
    args = parser.parse_args()
    assert args.num_steps > 0, "--num-steps 必须为正整数。"
    assert args.log_every > 0, "--log-every 必须为正整数。"
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _contact_properties_for_label(label: str) -> tuple[float, float]:
    """Return the margin and contact gap assigned to one IsaacLab shape label."""
    if "/World/FStyleJug" in label:
        return JUG_SHAPE_MARGIN, JUG_COLLISION_GAP
    if "/World/TargetContainer" in label:
        return CONTAINER_SHAPE_MARGIN, CONTAINER_COLLISION_GAP
    return TABLE_SHAPE_MARGIN, TABLE_COLLISION_GAP


def _assert_watertight_hydroelastic_mesh(geometry, label: str) -> None:
    """Assert that a mesh has no boundary, non-manifold, or degenerate edges."""
    vertices = np.asarray(geometry.vertices)
    indices = np.asarray(geometry.indices, dtype=np.int64)
    assert indices.size % 3 == 0, f"{label} 的索引数不是三角形数的整数倍。"
    assert indices.size == 0 or indices.max() < len(
        vertices
    ), f"{label} 的索引超出顶点范围。"
    _, welded_indices = np.unique(vertices, axis=0, return_inverse=True)
    triangles = welded_indices[indices].reshape(-1, 3)
    degenerate = np.logical_or.reduce(
        (
            triangles[:, 0] == triangles[:, 1],
            triangles[:, 1] == triangles[:, 2],
            triangles[:, 2] == triangles[:, 0],
        )
    )
    triangles = triangles[~degenerate]
    edges = triangles[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2)
    edges.sort(axis=1)
    _, edge_use_counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edge_count = int(np.count_nonzero(edge_use_counts == 1))
    non_manifold_edge_count = int(np.count_nonzero(edge_use_counts > 2))
    degenerate_count = int(np.count_nonzero(degenerate))
    assert (
        boundary_edge_count == 0
        and non_manifold_edge_count == 0
        and degenerate_count == 0
    ), (
        f"{label} 不是水密碰撞网格：边界边={boundary_edge_count}，"
        f"非流形边={non_manifold_edge_count}，退化三角形={degenerate_count}。"
        "Hydroelastic 仅支持封闭体。"
    )


def _configure_newton_sdf(_payload) -> None:
    """Bake SDFs and enable Hydroelastic on imported closed collision meshes."""
    import newton

    from isaaclab_newton.physics import NewtonManager

    builder = NewtonManager._builder
    assert builder is not None, "Newton MODEL_INIT 时未找到 ModelBuilder。"
    builder.rigid_gap = TABLE_COLLISION_GAP
    MESH_SDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    sdf_shape_count = 0
    for shape_index, geometry in enumerate(builder.shape_source):
        label = str(builder.shape_label[shape_index])
        margin, gap = _contact_properties_for_label(label)
        builder.shape_margin[shape_index] = margin
        builder.shape_gap[shape_index] = gap
        builder.shape_material_ke[shape_index] = CONTACT_STIFFNESS
        builder.shape_material_kd[shape_index] = CONTACT_DAMPING
        builder.shape_material_restitution[shape_index] = 0.0

        if not hasattr(geometry, "build_sdf"):
            continue

        geometry.build_sdf(
            max_resolution=MESH_SDF_MAX_RESOLUTION,
            narrow_band_range=MESH_SDF_NARROW_BAND_RANGE,
            margin=gap,
            scale=tuple(float(value) for value in builder.shape_scale[shape_index]),
            cache_dir=MESH_SDF_CACHE_DIR,
        )
        assert geometry.sdf is not None, f"{label} 的 SDF 构造失败。"
        builder.shape_force_sdf[shape_index] = True
        if "/World/FStyleJug" in label or "/World/TargetContainer" in label:
            _assert_watertight_hydroelastic_mesh(geometry, label)
            builder.shape_flags[shape_index] |= int(newton.ShapeFlags.HYDROELASTIC)
            builder.shape_material_kh[shape_index] = HYDROELASTIC_CONTACT_STIFFNESS
        sdf_shape_count += 1

    assert sdf_shape_count > 0, "IsaacLab 场景未导入任何可构造 SDF 的网格 shape。"
    print(
        f"[INFO] Newton SDF Hydroelastic 已就绪：{sdf_shape_count} 个网格 shape，"
        f"resolution={MESH_SDF_MAX_RESOLUTION}，"
        f"narrow_band={MESH_SDF_NARROW_BAND_RANGE}",
        flush=True,
    )


def _get_newton_picking_viewer(sim):
    """Return the interactive Newton viewer owned by IsaacLab, when available."""
    for visualizer in sim.visualizers:
        # IsaacLab's NewtonVisualizer currently exposes its ViewerGL as _viewer.
        # Keep this lookup defensive so --viz none and other visualizers still work.
        viewer = getattr(visualizer, "_viewer", None)
        if callable(getattr(viewer, "apply_forces", None)):
            viewer.picking_enabled = True
            return viewer
    return None


def main() -> None:
    """Create the IsaacLab scene and step it with Newton's SDF contact pipeline."""
    import newton.solvers

    # IsaacLab Newton 0.13.6 still imports the pre-1.5 enum name.
    if not hasattr(newton.solvers, "SolverNotifyFlags"):
        import newton

        newton.solvers.SolverNotifyFlags = newton.ModelFlags

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBase, AssetBaseCfg, RigidObject, RigidObjectCfg
    from isaaclab.physics import PhysicsEvent
    from isaaclab_newton.physics import (
        MJWarpSolverCfg,
        NewtonCfg,
        NewtonCollisionPipelineCfg,
        NewtonManager,
        NewtonShapeCfg,
        HydroelasticSDFCfg,
    )
    from isaaclab.sim import SimulationCfg, build_simulation_context

    class StaticCollisionAsset(AssetBase):
        """Keep a static USD collision asset in the IsaacLab stage for Newton."""

        @property
        def num_instances(self) -> int:
            """Return the single static USD instance."""
            return 1

        @property
        def data(self):
            """Return no dynamic state because Newton owns this static collider."""
            return None

        def reset(self, env_ids=None) -> None:
            """Reset no state for the static collision asset."""

        def write_data_to_sim(self) -> None:
            """Write no dynamic state for the static collision asset."""

        def update(self, dt: float) -> None:
            """Update no dynamic state for the static collision asset."""

        def _initialize_impl(self) -> None:
            """Initialize no physics view; Newton imports this as a static shape."""

    physics_cfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=RIGID_CONTACT_MAX,
            nconmax=RIGID_CONTACT_MAX,
            iterations=100,
            ls_iterations=50,
            ccd_iterations=35,
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(
            broad_phase="sap",
            reduce_contacts=True,
            rigid_contact_max=RIGID_CONTACT_MAX,
            sdf_hydroelastic_config=HydroelasticSDFCfg(
                anchor_contact=True,
                output_contact_surface=True,
            ),
        ),
        default_shape_cfg=NewtonShapeCfg(
            margin=TABLE_SHAPE_MARGIN,
            gap=TABLE_COLLISION_GAP,
        ),
        num_substeps=SIM_SUBSTEPS,
    )
    sim_cfg = SimulationCfg(
        device=args_cli.device,
        dt=1.0 / FRAME_RATE_HZ,
        physics=physics_cfg,
    )

    with build_simulation_context(
        sim_cfg=sim_cfg,
        add_lighting=True,
        visualizers=[] if args_cli.visualizer == ["none"] else args_cli.visualizer,
    ) as sim:
        collision_props = sim_utils.CollisionPropertiesCfg(
            contact_offset=TABLE_COLLISION_GAP,
            rest_offset=0.0,
        )
        contact_material = sim_utils.PhysxRigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            compliant_contact_stiffness=CONTACT_STIFFNESS,
            compliant_contact_damping=CONTACT_DAMPING,
        )

        table = RigidObject(
            RigidObjectCfg(
                prim_path="/World/Table",
                spawn=sim_utils.CuboidCfg(
                    size=tuple(2.0 * value for value in TABLE_HALF_EXTENTS),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True
                    ),
                    collision_props=collision_props,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.2, 0.2, 0.2)
                    ),
                    physics_material=contact_material,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=TABLE_POSITION),
            )
        )
        jug = RigidObject(
            RigidObjectCfg(
                prim_path="/World/FStyleJug",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(JUG_USD_PATH),
                    collision_props=collision_props,
                    physics_material=contact_material,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=JUG_POSITION),
            )
        )
        container = StaticCollisionAsset(
            AssetBaseCfg(
                prim_path="/World/TargetContainer",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(CONTAINER_USD_PATH),
                    collision_props=collision_props,
                    physics_material=contact_material,
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=CONTAINER_POSITION),
            )
        )

        # The callback runs after USD ingestion and before ModelBuilder.finalize().
        NewtonManager.register_callback(
            _configure_newton_sdf,
            PhysicsEvent.MODEL_INIT,
            name="configure_newton_sdf",
        )
        sim.reset()
        table.reset()
        jug.reset()
        container.reset()
        sim.set_camera_view(eye=(2.0, -2.0, 1.5), target=(0.45, 0.0, 0.1))
        picking_viewer = _get_newton_picking_viewer(sim)
        if picking_viewer is None:
            print(
                "[INFO] 未找到可交互的 Newton visualizer；"
                "请使用默认 --viz newton 以启用右键拖拽。",
                flush=True,
            )
        else:
            print("[INFO] Newton viewer 右键拾取和拖拽已启用。", flush=True)

        for step in range(1, args_cli.num_steps + 1):
            if not simulation_app.is_running():
                break
            table.write_data_to_sim()
            jug.write_data_to_sim()
            container.write_data_to_sim()
            state = NewtonManager.get_state_0()
            state.clear_forces()
            if picking_viewer is not None:
                picking_viewer.apply_forces(state)
            sim.step()
            table.update(sim.get_physics_dt())
            jug.update(sim.get_physics_dt())
            container.update(sim.get_physics_dt())
            if step % args_cli.log_every == 0:
                position = jug.data.root_pos_w.torch[0].detach().cpu().tolist()
                hydroelastic_sdf = NewtonManager._collision_pipeline.hydroelastic_sdf
                surface = hydroelastic_sdf.get_contact_surface()
                face_count = (
                    0 if surface is None else int(surface.face_contact_count.numpy()[0])
                )
                print(
                    f"[INFO] step={step:04d}, 水壶世界坐标: "
                    f"x={position[0]:+.5f}, y={position[1]:+.5f}, z={position[2]:+.5f}, "
                    f"Hydroelastic faces: {face_count}",
                    flush=True,
                )


if __name__ == "__main__":
    try:
        with contextlib.suppress(KeyboardInterrupt):
            main()
    finally:
        simulation_app.close()
