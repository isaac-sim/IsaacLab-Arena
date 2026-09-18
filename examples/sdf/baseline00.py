# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Newton 原生场景：运动学台面、水壶和目标容器。"""

import tempfile
import warnings
from pathlib import Path

import warp as wp

import newton
import newton.examples
from newton.usd import SchemaResolverNewton, SchemaResolverPhysx
from pxr import Gf, Usd, UsdGeom


TABLE_HALF_EXTENTS = (1.6, 1.6, 0.05)
TABLE_POSITION = (0.0, 0.0, -0.05)
JUG_USD_PATH = Path("/home/magengyu/IsaacLab-Arena/scene/fstylejug_a01/fstylejug_a01_inst_physx.usd")
CONTAINER_USD_PATH = Path(
    "/home/magengyu/IsaacLab-Arena/scene/Container_B04_40x30x12cm/"
    "Container_B04_40x30x12cm_PR_V_NVD_01.usd"
)
JUG_POSITION = (0.45, -0.25, 0.01)
CONTAINER_POSITION = (0.45, 0.22, 0.02)
MESH_SDF_MAX_RESOLUTION = 256
MESH_SDF_NARROW_BAND_RANGE = (-0.04, 0.04)
MESH_SDF_CACHE_DIR = Path(tempfile.gettempdir()) / "newton_sdf_cache"
TABLE_SHAPE_MARGIN = 0.005
TABLE_COLLISION_GAP = 0.01
# 水壶最小包围盒尺度约为 55 mm，2 mm 约占其 3.6%。
JUG_SHAPE_MARGIN = 0.002
JUG_COLLISION_GAP = 0.002
# 纸箱经厘米到米换算后的尺寸约为 403 x 301 x 120 mm。
CONTAINER_SHAPE_MARGIN = 0.002
CONTAINER_COLLISION_GAP = 0.010
CONTACT_STIFFNESS = 250000.0
CONTACT_DAMPING = 1000.0
RIGID_CONTACT_MAX = 4096
SIMULATION_FREQUENCY_HZ = 960
SDF_SOLVER_ITERATIONS = 20


def _get_container_stage_in_metres() -> Usd.Stage:
    """Open the centimetre-authored container with an explicit metre scale."""
    stage = Usd.Stage.Open(str(CONTAINER_USD_PATH))
    assert stage is not None, f"无法打开目标容器 USD：{CONTAINER_USD_PATH}"
    root_prim = stage.GetDefaultPrim()
    assert root_prim, f"目标容器 USD 没有默认 Prim：{CONTAINER_USD_PATH}"
    UsdGeom.Xformable(root_prim).AddScaleOp().Set(Gf.Vec3f(0.01, 0.01, 0.01))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    return stage


def _build_sdf_collision_models(builder: newton.ModelBuilder, asset_name: str, shape_indices: list[int]) -> None:
    """Cook cached SDFs for imported mesh colliders and force Newton to use them."""
    assert shape_indices, f"{asset_name} USD 未导入任何碰撞 shape。"
    for shape_index in shape_indices:
        geometry = builder.shape_source[shape_index]
        assert hasattr(geometry, "build_sdf"), f"{asset_name} shape={shape_index} 不是可构造 SDF 的网格。"
        geometry.build_sdf(
            max_resolution=MESH_SDF_MAX_RESOLUTION,
            narrow_band_range=MESH_SDF_NARROW_BAND_RANGE,
            margin=builder.shape_gap[shape_index],
            scale=tuple(float(value) for value in builder.shape_scale[shape_index]),
            cache_dir=MESH_SDF_CACHE_DIR,
        )
        assert geometry.sdf is not None, f"{asset_name} shape={shape_index} 的 SDF 构造失败。"
        builder.shape_force_sdf[shape_index] = True


def _set_shape_contact_properties(
    builder: newton.ModelBuilder,
    shape_indices: list[int],
    margin: float,
    gap: float,
) -> None:
    """Set per-shape contact distance, stiffness, damping, and restitution."""
    for shape_index in shape_indices:
        builder.shape_margin[shape_index] = margin
        builder.shape_gap[shape_index] = gap
        builder.shape_material_ke[shape_index] = CONTACT_STIFFNESS
        builder.shape_material_kd[shape_index] = CONTACT_DAMPING
        builder.shape_material_restitution[shape_index] = 0.0


class Example:
    """Build and simulate the native Newton scene."""

    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.log_every = args.log_every
        self.step_count = 0
        self.frame_dt = 1.0 / 120.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.rigid_gap = TABLE_COLLISION_GAP
        builder.default_shape_cfg.mu = 1.0
        table_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(*TABLE_POSITION), q=wp.quat_identity()),
            is_kinematic=True,
            label="table",
        )
        table_shape = builder.add_shape_box(
            body=table_body,
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            color=wp.vec3(0.2, 0.2, 0.2),
            label="table",
        )
        builder.shape_margin[table_shape] = TABLE_SHAPE_MARGIN
        builder.shape_gap[table_shape] = TABLE_COLLISION_GAP
        builder.shape_material_ke[table_shape] = CONTACT_STIFFNESS
        builder.shape_material_kd[table_shape] = CONTACT_DAMPING
        builder.shape_material_restitution[table_shape] = 0.0

        jug_result = builder.add_usd(
            str(JUG_USD_PATH),
            xform=wp.transform(p=wp.vec3(*JUG_POSITION), q=wp.quat_identity()),
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
        )
        jug_bodies = list(jug_result["path_body_map"].values())
        assert len(jug_bodies) == 1, "水壶 Physics USD 应当只包含一个刚体。"
        self.jug_body = jug_bodies[0]
        jug_shape_indices = list(jug_result["path_shape_map"].values())
        _set_shape_contact_properties(builder, jug_shape_indices, JUG_SHAPE_MARGIN, JUG_COLLISION_GAP)
        _build_sdf_collision_models(builder, "水壶", jug_shape_indices)

        container_result = builder.add_usd(
            _get_container_stage_in_metres(),
            xform=wp.transform(p=wp.vec3(*CONTAINER_POSITION), q=wp.quat_identity()),
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
        )
        container_shape_indices = list(container_result["path_shape_map"].values())
        _set_shape_contact_properties(
            builder,
            container_shape_indices,
            CONTAINER_SHAPE_MARGIN,
            CONTAINER_COLLISION_GAP,
        )
        _build_sdf_collision_models(builder, "目标容器", container_shape_indices)

        self.model = builder.finalize()
        self.model.rigid_contact_max = RIGID_CONTACT_MAX
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            rigid_contact_max=RIGID_CONTACT_MAX,
            broad_phase="sap",
        )
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            njmax=RIGID_CONTACT_MAX,
            nconmax=RIGID_CONTACT_MAX,
            iterations=100,
            ls_iterations=50,
            ccd_iterations=35,
            sdf_iterations=SDF_SOLVER_ITERATIONS,
            impratio=1.0,
            cone="pyramidal",
            tolerance=1.0e-6,
            use_mujoco_cpu=False,
            use_mujoco_contacts=False,
            update_data_interval=1,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.collision_pipeline.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(2.0, -2.0, 1.5), pitch=-15.0, yaw=135.0)
        self.graph = self._capture()

    def _simulate(self) -> None:
        """Advance one displayed frame using MuJoCo-Warp substeps."""
        for _ in range(self.sim_substeps):
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _capture(self):
        """Capture the simulation step when the active device supports it."""
        if not wp.get_device().is_cuda:
            return None
        try:
            with wp.ScopedCapture() as capture:
                self._simulate()
            return capture.graph
        except Exception as exc:
            warnings.warn(f"CUDA graph capture failed; using eager stepping: {exc}", stacklevel=2)
            return None

    def step(self) -> None:
        """Advance the simulation and print the jug world position."""
        if self.graph is None:
            self._simulate()
        else:
            wp.capture_launch(self.graph)
        self.step_count += 1
        self.sim_time += self.frame_dt
        if self.step_count % self.log_every == 0:
            position = self.state_0.body_q.numpy()[self.jug_body, :3]
            print(f"[INFO] step={self.step_count:04d}, 水壶世界坐标: {position}", flush=True)

    def render(self) -> None:
        """Render the current Newton state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--log-every", type=int, default=1000, help="每隔多少帧打印一次水壶位置。")
    viewer, args = newton.examples.init(parser)
    assert args.log_every > 0, "--log-every 必须为正整数。"
    newton.examples.run(Example(viewer, args), args)
