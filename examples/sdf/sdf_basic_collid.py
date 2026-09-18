# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Newton 原生 SDF 碰撞示例：运动学台面、水壶和目标容器。

使用方法：

    .venv/bin/python examples/sdf/sdf_basic_collid.py

水壶使用原始瓶身和瓶盖作为视觉模型，使用 Blender 生成的单一实心水密网格
构建物理 SDF。右侧控制面板可分别显示碰撞源网格（灰色）和烘焙后的 SDF
等值面（黄色）；内置 ``Show Collision`` 是两者的总开关。还可显示离散 SDF
接触点和法向力代理颜色，便于检查水壶、台面与容器之间的接触。

基础 SDF 接触只输出离散接触点，不包含 Hydroelastic 接触面积。因此调试视图中的
“压力”是法向力代理值 ``CONTACT_STIFFNESS * penetration``，用于比较接触强弱，
并非真实单位为 Pa 的面压力。
"""

import warnings
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.usd import SchemaResolverNewton, SchemaResolverPhysx


TABLE_HALF_EXTENTS = (1.6, 1.6, 0.05)
TABLE_POSITION = (0.0, 0.0, -0.05)
JUG_USD_PATH = Path(
    "/home/magengyu/IsaacLab-Arena/scene/fstylejug_a01/fstylejug_a01_visual_solid_physx.usda"
)
CONTAINER_USD_PATH = Path(
    "/home/magengyu/IsaacLab-Arena/scene/open_cardboard_box_hydroelastic.usda"
)
JUG_POSITION = (0.45, -0.25, 0.01)
CONTAINER_POSITION = (0.45, 0.22, 0.02)
MESH_SDF_MAX_RESOLUTION = 128
MESH_SDF_NARROW_BAND_RANGE = (-0.04, 0.04)
MESH_SDF_CACHE_DIR = Path(__file__).resolve().parents[2] / "scene" / "tmp"
TABLE_SHAPE_MARGIN = 0.005
TABLE_COLLISION_GAP = 0.01
# 水壶最小包围盒尺度约为 55 mm，2 mm 约占其 3.6%。
JUG_SHAPE_MARGIN = 0.002
JUG_COLLISION_GAP = 0.002
# 纸箱尺寸为 800 x 600 x 200 mm。
CONTAINER_SHAPE_MARGIN = 0.002
CONTAINER_COLLISION_GAP = 0.010
CONTACT_STIFFNESS = 250000.0
CONTACT_DAMPING = 1000.0
RIGID_CONTACT_MAX = 4096
SDF_SOLVER_ITERATIONS = 20
SDF_CONTACT_POINT_RADIUS = 0.006
FRAME_RATE_HZ = 120
SIM_SUBSTEPS = 4


def _build_sdf_collision_models(
    builder: newton.ModelBuilder, asset_name: str, shape_indices: list[int]
) -> None:
    """Cook cached SDFs for imported mesh colliders and force Newton to use them."""
    assert shape_indices, f"{asset_name} USD 未导入任何碰撞 shape。"
    MESH_SDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for shape_index in shape_indices:
        geometry = builder.shape_source[shape_index]
        assert hasattr(
            geometry, "build_sdf"
        ), f"{asset_name} shape={shape_index} 不是可构造 SDF 的网格。"
        geometry.build_sdf(
            max_resolution=MESH_SDF_MAX_RESOLUTION,
            narrow_band_range=MESH_SDF_NARROW_BAND_RANGE,
            margin=builder.shape_gap[shape_index],
            scale=tuple(float(value) for value in builder.shape_scale[shape_index]),
            cache_dir=MESH_SDF_CACHE_DIR,
        )
        assert (
            geometry.sdf is not None
        ), f"{asset_name} shape={shape_index} 的 SDF 构造失败。"
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
        self.show_jug_collision_source_mesh = True
        self.show_jug_sdf_isosurface = True
        self.show_sdf_contact_points = hasattr(viewer, "renderer")
        self.show_sdf_contact_pressure = False
        self.frame_dt = 1.0 / FRAME_RATE_HZ
        self.sim_substeps = SIM_SUBSTEPS
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
        self.jug_visual_shape_indices = []
        self.jug_collision_shape_indices = []
        for prim_path, shape_index in jug_result["path_shape_map"].items():
            if prim_path.endswith("/Collision"):
                self.jug_collision_shape_indices.append(shape_index)
            else:
                self.jug_visual_shape_indices.append(shape_index)
                # Newton 仅将 collider Mesh 默认标记为可见。视觉 Mesh 不参与
                # 碰撞，故需在导入后显式打开其渲染 flag。
                builder.shape_flags[shape_index] |= int(newton.ShapeFlags.VISIBLE)
        assert len(self.jug_visual_shape_indices) == 2, "水壶应包含瓶身和瓶盖视觉 Mesh。"
        assert (
            len(self.jug_collision_shape_indices) == 1
        ), "水壶应只使用一个实心 Collision Mesh。"
        _set_shape_contact_properties(
            builder,
            self.jug_collision_shape_indices,
            JUG_SHAPE_MARGIN,
            JUG_COLLISION_GAP,
        )
        _build_sdf_collision_models(builder, "水壶", self.jug_collision_shape_indices)

        container_result = builder.add_usd(
            str(CONTAINER_USD_PATH),
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
        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state_0
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(2.0, -2.0, 1.5), pitch=-15.0, yaw=135.0)
        self.graph = self._capture()

    def _simulate(self) -> None:
        """Advance one displayed frame using MuJoCo-Warp substeps."""
        for _ in range(self.sim_substeps):
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
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
            warnings.warn(
                f"CUDA graph capture failed; using eager stepping: {exc}", stacklevel=2
            )
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
            print(
                f"[INFO] step={self.step_count:04d}, 水壶世界坐标: {position}",
                flush=True,
            )

    def render(self) -> None:
        """Render the current Newton state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self._apply_jug_collision_debug_visibility()
        self._render_sdf_contact_debug_data()
        self.viewer.end_frame()

    def _render_sdf_contact_debug_data(self) -> None:
        """Render optional discrete SDF contact points and normal-force proxies."""
        if not hasattr(self.viewer, "renderer"):
            return

        points, force_proxy = self._get_sdf_contact_debug_data()
        self._log_sdf_contact_points(
            "points",
            points,
            np.tile(np.array((0.2, 1.0, 0.2), dtype=np.float32), (len(points), 1)),
            self.show_sdf_contact_points,
            SDF_CONTACT_POINT_RADIUS,
        )
        self._log_sdf_contact_points(
            "pressure",
            points,
            self._pseudo_color(force_proxy),
            self.show_sdf_contact_pressure,
            1.5 * SDF_CONTACT_POINT_RADIUS,
        )
        self.sdf_contact_count = len(points)
        self.sdf_max_force_proxy = float(force_proxy.max(initial=0.0))

    def _apply_jug_collision_debug_visibility(self) -> None:
        """Independently hide the source mesh and cooked SDF isosurface in ViewerGL."""
        objects = getattr(self.viewer, "objects", None)
        shape_to_batch = getattr(self.viewer, "_shape_to_batch", None)
        sdf_batches = getattr(self.viewer, "_sdf_isomesh_instances", None)
        if objects is None or shape_to_batch is None or sdf_batches is None:
            return

        master_visible = bool(self.viewer.show_collision)
        source_visible = master_visible and self.show_jug_collision_source_mesh
        for shape_index in self.jug_collision_shape_indices:
            batch = shape_to_batch[shape_index]
            instance = objects.get(batch.name) if batch is not None else None
            if instance is not None:
                instance.hidden = not source_visible

        sdf_visible = master_visible and self.show_jug_sdf_isosurface
        collision_indices = set(self.jug_collision_shape_indices)
        for batch in sdf_batches.values():
            if not collision_indices.intersection(batch.model_shapes):
                continue
            instance = objects.get(batch.name)
            if instance is not None:
                instance.hidden = not sdf_visible

    def _get_sdf_contact_debug_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return contact midpoints and stiffness-times-penetration force proxies."""
        contact_count = min(
            int(self.contacts.rigid_contact_count.numpy()[0]), RIGID_CONTACT_MAX
        )
        if contact_count == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.float32)

        shape0 = self.contacts.rigid_contact_shape0.numpy()[:contact_count]
        shape1 = self.contacts.rigid_contact_shape1.numpy()[:contact_count]
        point0 = self.contacts.rigid_contact_point0.numpy()[:contact_count]
        point1 = self.contacts.rigid_contact_point1.numpy()[:contact_count]
        offset0 = self.contacts.rigid_contact_offset0.numpy()[:contact_count]
        offset1 = self.contacts.rigid_contact_offset1.numpy()[:contact_count]
        normal = self.contacts.rigid_contact_normal.numpy()[:contact_count]
        shape_body = self.model.shape_body.numpy()
        body_q = self.state_0.body_q.numpy()

        anchor0 = self._transform_contact_points(
            point0 + offset0, shape0, shape_body, body_q
        )
        anchor1 = self._transform_contact_points(
            point1 + offset1, shape1, shape_body, body_q
        )
        signed_separation = np.einsum("ij,ij->i", anchor1 - anchor0, normal)
        penetration = np.maximum(-signed_separation, 0.0)
        return 0.5 * (anchor0 + anchor1), CONTACT_STIFFNESS * penetration

    @staticmethod
    def _transform_contact_points(
        points: np.ndarray,
        shape_indices: np.ndarray,
        shape_body: np.ndarray,
        body_q: np.ndarray,
    ) -> np.ndarray:
        """Transform body-frame contact points to world space."""
        world_points = points.copy()
        body_indices = shape_body[shape_indices]
        dynamic = body_indices >= 0
        if not np.any(dynamic):
            return world_points

        transforms = body_q[body_indices[dynamic]]
        positions = transforms[:, :3]
        quaternions = transforms[:, 3:]
        vectors = world_points[dynamic]
        quaternion_vectors = quaternions[:, :3]
        twice_cross = 2.0 * np.cross(quaternion_vectors, vectors)
        world_points[dynamic] = (
            vectors
            + quaternions[:, 3:4] * twice_cross
            + np.cross(quaternion_vectors, twice_cross)
        )
        world_points[dynamic] += positions
        return world_points

    @staticmethod
    def _pseudo_color(values: np.ndarray) -> np.ndarray:
        """Map force proxies to a quantile-scaled continuous pseudocolor ramp."""
        finite_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        stops = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        normalized = np.zeros_like(finite_values, dtype=np.float32)
        positive = finite_values > 0.0
        positive_values = finite_values[positive]
        if len(positive_values) == 1:
            normalized[positive] = 1.0
        elif len(positive_values) > 1:
            quantiles = np.quantile(positive_values, stops)
            breakpoints, first_indices = np.unique(quantiles, return_index=True)
            breakpoint_colors = stops[first_indices]
            normalized[positive] = np.interp(
                positive_values, breakpoints, breakpoint_colors
            )
        ramp = np.array(
            (
                (0.0, 0.0, 1.0),  # blue
                (0.0, 1.0, 1.0),  # cyan
                (0.0, 1.0, 0.0),  # green
                (1.0, 1.0, 0.0),  # yellow
                (1.0, 0.0, 0.0),  # red
            ),
            dtype=np.float32,
        )
        colors = np.column_stack(
            [np.interp(normalized, stops, ramp[:, channel]) for channel in range(3)]
        ).astype(np.float32)
        return colors

    def _log_sdf_contact_points(
        self,
        name: str,
        points: np.ndarray,
        colors: np.ndarray,
        is_enabled: bool,
        radius: float,
    ) -> None:
        """Log a contact point cloud, clearing it when disabled or empty."""
        self.viewer.log_points(
            f"/sdf_contact_debug/{name}",
            (
                wp.array(points, dtype=wp.vec3, device=self.model.device)
                if is_enabled and len(points)
                else None
            ),
            radii=radius,
            colors=(
                wp.array(colors, dtype=wp.vec3, device=self.model.device)
                if is_enabled and len(points)
                else None
            ),
        )

    def gui(self, imgui) -> None:
        """Render controls and statistics for basic SDF contact diagnostics."""
        _, self.show_jug_collision_source_mesh = imgui.checkbox(
            "Show jug collision source mesh (gray)",
            self.show_jug_collision_source_mesh,
        )
        _, self.show_jug_sdf_isosurface = imgui.checkbox(
            "Show jug SDF isosurface (yellow)", self.show_jug_sdf_isosurface
        )
        imgui.text("Show Collision is the master switch for both collision layers")
        _, self.show_sdf_contact_points = imgui.checkbox(
            "Show SDF contact points", self.show_sdf_contact_points
        )
        _, self.show_sdf_contact_pressure = imgui.checkbox(
            "Show normal-force proxy", self.show_sdf_contact_pressure
        )
        imgui.text("Relative force: blue, cyan, green, yellow, red (high)")
        imgui.text(f"SDF contacts: {getattr(self, 'sdf_contact_count', 0)}")
        imgui.text(
            f"Max normal-force proxy: {getattr(self, 'sdf_max_force_proxy', 0.0):.3e} N"
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--log-every", type=int, default=1000, help="每隔多少帧打印一次水壶位置。"
    )
    viewer, args = newton.examples.init(parser)
    assert args.log_every > 0, "--log-every 必须为正整数。"
    newton.examples.run(Example(viewer, args), args)
