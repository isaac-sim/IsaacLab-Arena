# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Newton Hydroelastic 场景：运动学台面、水壶和目标容器。

水壶保留原始瓶身、瓶盖及材质作为视觉模型，仅使用独立的实心水密
``Collision`` 网格构建 SDF 和 Hydroelastic 碰撞。右侧控制面板可分别显示
碰撞源网格（灰色）和烘焙后的 SDF 等值面（黄色）；内置 ``Show Collision``
是这两层调试显示的总开关。接触面、侵入深度、压力和法向也可独立显示。

| 类别 | 参数 | 配置值 | 说明 |
| --- | --- | --- | --- |
| 场景 | ``TABLE_HALF_EXTENTS`` | ``(1.6, 1.6, 0.05) m`` | 运动学台面的半尺寸。 |
| 场景 | ``TABLE_POSITION`` | ``(0, 0, -0.05) m`` | 台面中心位置，使台面上表面位于 ``z=0``。 |
| 场景 | ``JUG_POSITION`` | ``(0.45, 0.22, 0.25) m`` | 水壶初始位置，位于纸箱正上方。 |
| 场景 | ``JUG_MASS`` | ``5.0 kg`` | 水壶质量；惯量张量按导入质量同比例缩放。 |
| 场景 | ``CONTAINER_POSITION`` | ``(0.45, 0.22, 0.02) m`` | 纸箱初始位置。 |
| SDF | ``MESH_SDF_MAX_RESOLUTION`` | ``256`` | 最长轴的最大稀疏 SDF 分辨率。 |
| SDF | ``MESH_SDF_NARROW_BAND_RANGE`` | ``(-0.04, 0.04) m`` | SDF 窄带内、外距离。 |
| SDF | ``MESH_SDF_CACHE_DIR`` | 项目目录 ``scene/tmp`` | 烘焙 SDF 的磁盘缓存位置。 |
| 接触 | ``*_SHAPE_MARGIN`` | 台面 ``5 mm``；水壶、纸箱 ``2 mm`` | 接触面偏移距离。 |
| 接触 | ``*_COLLISION_GAP`` | 台面、纸箱 ``10 mm``；水壶 ``2 mm`` | 开始生成接触的额外距离。 |
| 接触 | ``TABLE_CONTACT_STIFFNESS`` | ``2.5e5`` | 台面的常规点接触刚度。 |
| 接触 | ``JUG_CONTACT_STIFFNESS`` | ``5.0e4`` | 水壶的常规点接触刚度。 |
| 接触 | ``BOX_CONTACT_STIFFNESS`` | ``2.5e5`` | box 的常规点接触刚度。 |
| 接触 | ``CONTACT_DAMPING`` | ``1.0e3`` | 三类碰撞体共用的接触阻尼。 |
| Hydroelastic | ``JUG_HYDROELASTIC_CONTACT_STIFFNESS`` | ``2.0e9`` | 水壶的 SDF 深度到压力系数。 |
| Hydroelastic | ``BOX_HYDROELASTIC_CONTACT_STIFFNESS`` | ``1.0e11`` | box 的 SDF 深度到压力系数。 |
| Hydroelastic | ``mc_edge_clamp_min`` | ``0.0`` | 不额外偏置 marching-cubes 接触面边缘。 |
| 可视化 | ``PRESSURE_DIFFUSION_ITERATIONS`` / ``PRESSURE_DIFFUSION_RATE`` | ``8`` / ``0.85`` | 压力沿零水平集网格的扩散轮数和比例。 |
| 可视化 | ``HYDROELASTIC_DEBUG_VECTOR_MAX_LENGTH`` | ``50 mm`` | 法向箭头的显示长度。 |
| 求解 | ``FRAME_RATE_HZ`` / ``SIM_SUBSTEPS`` | ``480 Hz`` / ``4`` | 显示帧率与每帧积分子步数。 |
| 求解 | ``frame_dt`` / ``sim_dt`` | ``1/480 s`` / ``1/1920 s`` | 由显示帧率和子步数计算。 |
| 求解 | ``iterations`` / ``ls_iterations`` | ``100`` / ``50`` | Newton 求解与线搜索迭代上限。 |
| 求解 | ``ccd_iterations`` | ``35`` | 连续碰撞检测迭代上限。 |
| 容量 | ``RIGID_CONTACT_MAX`` | ``4096`` | 碰撞管线与求解器允许的最大刚体接触数。 |
"""

import warnings
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.geometry import HydroelasticSDF
from newton.usd import SchemaResolverNewton, SchemaResolverPhysx


TABLE_HALF_EXTENTS = (1.6, 1.6, 0.05)
TABLE_POSITION = (0.0, 0.0, -0.05)
JUG_USD_PATH = Path(
    "/home/magengyu/IsaacLab-Arena/scene/fstylejug_a01/fstylejug_a01_visual_solid_physx.usda"
)
CONTAINER_USD_PATH = Path(
    "/home/magengyu/IsaacLab-Arena/scene/open_cardboard_box_hydroelastic.usda"
)
JUG_POSITION = (0.45, 0.22, 0.25)
JUG_MASS = 5.0
CONTAINER_POSITION = (0.45, 0.22, 0.02)
MESH_SDF_MAX_RESOLUTION = 256
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
TABLE_CONTACT_STIFFNESS = 2.5e5
JUG_CONTACT_STIFFNESS = 5.0e4
BOX_CONTACT_STIFFNESS = 2.5e5
CONTACT_DAMPING = 1.0e3
# Hydroelastic contact uses pressure per signed-distance depth, independently
# of the point-contact stiffness used by the table's conventional collider.
JUG_HYDROELASTIC_CONTACT_STIFFNESS = 2.0e9
BOX_HYDROELASTIC_CONTACT_STIFFNESS = 1.0e11
PRESSURE_DIFFUSION_ITERATIONS = 8
PRESSURE_DIFFUSION_RATE = 0.85
PRESSURE_VERTEX_WELD_TOLERANCE = 1.0e-6
HYDROELASTIC_DEBUG_VECTOR_MAX_LENGTH = 0.05
RIGID_CONTACT_MAX = 4096
FRAME_RATE_HZ = 120
SIM_SUBSTEPS = 4


def _build_sdf_collision_models(
    builder: newton.ModelBuilder,
    asset_name: str,
    shape_indices: list[int],
    *,
    is_hydroelastic: bool,
    hydroelastic_stiffness: float,
) -> None:
    """Cook cached SDFs and optionally enable hydroelastic contact for closed meshes."""
    assert shape_indices, f"{asset_name} USD 未导入任何碰撞 shape。"
    MESH_SDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for shape_index in shape_indices:
        geometry = builder.shape_source[shape_index]
        assert hasattr(
            geometry, "build_sdf"
        ), f"{asset_name} shape={shape_index} 不是可构造 SDF 的网格。"
        _exit_if_mesh_is_not_watertight(geometry, asset_name, shape_index)
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
        if is_hydroelastic:
            builder.shape_flags[shape_index] |= int(newton.ShapeFlags.HYDROELASTIC)
            builder.shape_material_kh[shape_index] = hydroelastic_stiffness


def _exit_if_mesh_is_not_watertight(
    geometry: newton.Mesh, asset_name: str, shape_index: int
) -> None:
    """Exit when a collision mesh has boundary or non-manifold edges."""
    vertices = np.asarray(geometry.vertices)
    indices = np.asarray(geometry.indices, dtype=np.int64)
    assert (
        indices.size % 3 == 0
    ), f"{asset_name} shape={shape_index} 的索引数不是三角形数的整数倍。"
    assert indices.size == 0 or indices.max() < len(
        vertices
    ), f"{asset_name} shape={shape_index} 的索引超出顶点范围。"
    _, welded_vertex_indices = np.unique(vertices, axis=0, return_inverse=True)
    triangles = welded_vertex_indices[indices].reshape(-1, 3)
    is_degenerate = np.logical_or.reduce(
        (
            triangles[:, 0] == triangles[:, 1],
            triangles[:, 1] == triangles[:, 2],
            triangles[:, 2] == triangles[:, 0],
        )
    )
    degenerate_triangle_count = int(np.count_nonzero(is_degenerate))
    triangles = triangles[~is_degenerate]
    edges = triangles[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2)
    edges.sort(axis=1)
    _, edge_use_counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edge_count = int(np.count_nonzero(edge_use_counts == 1))
    non_manifold_edge_count = int(np.count_nonzero(edge_use_counts > 2))
    assert (
        boundary_edge_count == 0
        and non_manifold_edge_count == 0
        and degenerate_triangle_count == 0
    ), (
        f"{asset_name} shape={shape_index} 不是水密碰撞网格：边界边={boundary_edge_count}，"
        f"非流形边={non_manifold_edge_count}，退化三角形={degenerate_triangle_count}。"
        "Hydroelastic 仅支持封闭体；请提供水密碰撞代理后重试。"
    )


def _set_shape_contact_properties(
    builder: newton.ModelBuilder,
    shape_indices: list[int],
    margin: float,
    gap: float,
    stiffness: float,
) -> None:
    """Set per-shape contact distance, stiffness, damping, and restitution."""
    for shape_index in shape_indices:
        builder.shape_margin[shape_index] = margin
        builder.shape_gap[shape_index] = gap
        builder.shape_material_ke[shape_index] = stiffness
        builder.shape_material_kd[shape_index] = CONTACT_DAMPING
        builder.shape_material_restitution[shape_index] = 0.0


def _set_body_mass(
    builder: newton.ModelBuilder, body_index: int, target_mass: float
) -> None:
    """Set body mass while preserving the imported mass distribution."""
    imported_mass = float(builder.body_mass[body_index])
    assert imported_mass > 0.0, "只能缩放具有正质量的动态刚体。"
    assert target_mass > 0.0, "目标质量必须为正数。"
    mass_scale = target_mass / imported_mass
    builder.body_mass[body_index] = target_mass
    builder.body_inertia[body_index] = builder.body_inertia[body_index] * mass_scale


class Example:
    """Build and simulate the native Newton scene."""

    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        if hasattr(self.viewer, "picking_enabled"):
            self.viewer.picking_enabled = True
        self.log_every = args.log_every
        self.step_count = 0
        self.show_jug_collision_source_mesh = True
        self.show_jug_sdf_isosurface = True
        self.show_hydroelastic_surface = hasattr(viewer, "renderer")
        self.show_hydroelastic_depth = False
        self.show_hydroelastic_pressure = False
        self.show_hydroelastic_normal = False
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
        builder.shape_material_ke[table_shape] = TABLE_CONTACT_STIFFNESS
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
                # 非碰撞视觉 Mesh 由 Newton 导入后可能没有 VISIBLE 标志。
                builder.shape_flags[shape_index] |= int(newton.ShapeFlags.VISIBLE)
        assert len(self.jug_visual_shape_indices) == 2, "水壶应包含瓶身和瓶盖视觉 Mesh。"
        assert (
            len(self.jug_collision_shape_indices) == 1
        ), "水壶应只使用一个实心水密 Collision Mesh。"
        _set_body_mass(builder, self.jug_body, JUG_MASS)
        _set_shape_contact_properties(
            builder,
            self.jug_collision_shape_indices,
            JUG_SHAPE_MARGIN,
            JUG_COLLISION_GAP,
            JUG_CONTACT_STIFFNESS,
        )
        _build_sdf_collision_models(
            builder,
            "水壶",
            self.jug_collision_shape_indices,
            is_hydroelastic=True,
            hydroelastic_stiffness=JUG_HYDROELASTIC_CONTACT_STIFFNESS,
        )

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
            BOX_CONTACT_STIFFNESS,
        )
        _build_sdf_collision_models(
            builder,
            "目标容器",
            container_shape_indices,
            is_hydroelastic=True,
            hydroelastic_stiffness=BOX_HYDROELASTIC_CONTACT_STIFFNESS,
        )

        self.model = builder.finalize()
        self.model.rigid_contact_max = RIGID_CONTACT_MAX
        sdf_hydroelastic_config = HydroelasticSDF.Config(
            mc_edge_clamp_min=0.0,
            output_contact_surface=hasattr(viewer, "renderer"),
            anchor_contact=True,
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            rigid_contact_max=RIGID_CONTACT_MAX,
            broad_phase="sap",
            sdf_hydroelastic_config=sdf_hydroelastic_config,
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
        self.graph = (
            None if getattr(self.viewer, "picking_enabled", False) else self._capture()
        )

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
            hydroelastic_face_count = self._get_hydroelastic_face_count()
            print(
                f"[INFO] step={self.step_count:04d}, 水壶世界坐标: {position}, "
                f"Hydroelastic faces: {hydroelastic_face_count}",
                flush=True,
            )

    def render(self) -> None:
        """Render the current Newton state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self._apply_jug_collision_debug_visibility()
        self._render_hydroelastic_debug_data()
        self.viewer.end_frame()

    def _apply_jug_collision_debug_visibility(self) -> None:
        """Independently hide the jug source mesh and cooked SDF isosurface."""
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

    def _render_hydroelastic_debug_data(self) -> None:
        """Render optional hydroelastic contact-surface diagnostics."""
        if not hasattr(self.viewer, "renderer"):
            return

        surface = (
            self.collision_pipeline.hydroelastic_sdf.get_contact_surface()
            if self.collision_pipeline.hydroelastic_sdf is not None
            else None
        )
        if surface is None:
            self.hydroelastic_face_count = 0
            self.hydroelastic_max_penetration = 0.0
            self.hydroelastic_max_pressure = 0.0
            self.hydroelastic_max_diffused_pressure = 0.0
            self._log_hydroelastic_lines(
                "surface", None, None, None, self.show_hydroelastic_surface
            )
            self._log_hydroelastic_lines(
                "depth", None, None, None, self.show_hydroelastic_depth
            )
            self._log_hydroelastic_lines(
                "pressure", None, None, None, self.show_hydroelastic_pressure
            )
            self._log_hydroelastic_arrows("normal", None, None, None)
            return

        face_count = self._get_hydroelastic_face_count()
        if face_count == 0:
            self.hydroelastic_face_count = 0
            self.hydroelastic_max_penetration = 0.0
            self.hydroelastic_max_pressure = 0.0
            self.hydroelastic_max_diffused_pressure = 0.0
            self._log_hydroelastic_lines(
                "surface", None, None, None, self.show_hydroelastic_surface
            )
            self._log_hydroelastic_lines(
                "depth", None, None, None, self.show_hydroelastic_depth
            )
            self._log_hydroelastic_lines(
                "pressure", None, None, None, self.show_hydroelastic_pressure
            )
            self._log_hydroelastic_arrows("normal", None, None, None)
            return

        triangles = surface.contact_surface_point.numpy()[: 3 * face_count].reshape(
            face_count, 3, 3
        )
        depths = surface.contact_surface_depth.numpy()[:face_count]
        shape_pairs = surface.contact_surface_shape_pair.numpy()[:face_count]
        centers = triangles.mean(axis=1)
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        normal_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(
            normals,
            normal_lengths,
            out=np.zeros_like(normals),
            where=normal_lengths > 0.0,
        )

        surface_starts = triangles[:, (0, 1, 2)].reshape(-1, 3)
        surface_ends = triangles[:, (1, 2, 0)].reshape(-1, 3)
        # These three visualizations use the exact same triangle edges.  Keep
        # them mutually exclusive so the gray surface cannot cover pseudocolor
        # depth/pressure lines through coplanar depth fighting.
        show_pressure = self.show_hydroelastic_pressure
        show_depth = self.show_hydroelastic_depth and not show_pressure
        show_surface = self.show_hydroelastic_surface and not (
            show_depth or show_pressure
        )
        self._log_hydroelastic_lines(
            "surface",
            surface_starts,
            surface_ends,
            np.tile(np.array((0.9, 0.9, 0.9), dtype=np.float32), (3 * face_count, 1)),
            show_surface,
        )

        penetration = np.maximum(-depths, 0.0)
        self._log_hydroelastic_lines(
            "depth",
            surface_starts,
            surface_ends,
            np.repeat(self._pseudo_color(penetration), 3, axis=0),
            show_depth,
        )
        shape_stiffness = self.model.shape_material_kh.numpy()
        pressure = shape_stiffness[shape_pairs[:, 1]] * penetration
        diffused_pressure = self._diffuse_surface_values(triangles, pressure)
        self._log_hydroelastic_lines(
            "pressure",
            surface_starts,
            surface_ends,
            np.repeat(self._pseudo_color(diffused_pressure), 3, axis=0),
            show_pressure,
        )
        self._log_hydroelastic_arrows(
            "normal",
            centers,
            centers + normals * HYDROELASTIC_DEBUG_VECTOR_MAX_LENGTH,
            np.tile(np.array((0.2, 1.0, 0.2), dtype=np.float32), (face_count, 1)),
        )

        self.hydroelastic_face_count = face_count
        self.hydroelastic_max_penetration = float(penetration.max(initial=0.0))
        self.hydroelastic_max_pressure = float(pressure.max(initial=0.0))
        self.hydroelastic_max_diffused_pressure = float(
            diffused_pressure.max(initial=0.0)
        )

    @staticmethod
    def _diffuse_surface_values(
        triangles: np.ndarray, face_values: np.ndarray
    ) -> np.ndarray:
        """Diffuse face values through shared vertices of the SDF zero-level mesh."""
        if len(face_values) == 0 or PRESSURE_DIFFUSION_ITERATIONS == 0:
            return face_values.astype(np.float32, copy=True)

        quantized_vertices = np.rint(
            triangles.reshape(-1, 3) / PRESSURE_VERTEX_WELD_TOLERANCE
        ).astype(np.int64)
        _, vertex_ids = np.unique(
            quantized_vertices, axis=0, return_inverse=True
        )
        face_vertex_ids = vertex_ids.reshape(-1, 3)
        flat_vertex_ids = face_vertex_ids.reshape(-1)
        vertex_count = int(flat_vertex_ids.max(initial=-1)) + 1
        vertex_face_count = np.bincount(
            flat_vertex_ids, minlength=vertex_count
        ).astype(np.float32)

        source = np.nan_to_num(
            face_values, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)
        diffused = source.copy()
        for _ in range(PRESSURE_DIFFUSION_ITERATIONS):
            vertex_sum = np.bincount(
                flat_vertex_ids,
                weights=np.repeat(diffused, 3),
                minlength=vertex_count,
            )
            vertex_average = np.divide(
                vertex_sum,
                vertex_face_count,
                out=np.zeros(vertex_count, dtype=np.float64),
                where=vertex_face_count > 0.0,
            )
            neighbor_average = vertex_average[face_vertex_ids].mean(axis=1)
            diffused = (
                (1.0 - PRESSURE_DIFFUSION_RATE) * source
                + PRESSURE_DIFFUSION_RATE * neighbor_average
            ).astype(np.float32)
        return diffused

    def _get_hydroelastic_face_count(self) -> int:
        """Return the active marching-cubes hydroelastic contact-face count."""
        hydroelastic_sdf = self.collision_pipeline.hydroelastic_sdf
        if hydroelastic_sdf is None:
            return 0
        surface = hydroelastic_sdf.get_contact_surface()
        if surface is None:
            return 0
        return int(surface.face_contact_count.numpy()[0])

    @staticmethod
    def _pseudo_color(values: np.ndarray) -> np.ndarray:
        """Map values to a quantile-scaled continuous pseudocolor ramp."""
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
                (0.0, 0.0, 1.0),
                (0.0, 1.0, 1.0),
                (0.0, 1.0, 0.0),
                (1.0, 1.0, 0.0),
                (1.0, 0.0, 0.0),
            ),
            dtype=np.float32,
        )
        colors = np.column_stack(
            [np.interp(normalized, stops, ramp[:, channel]) for channel in range(3)]
        ).astype(np.float32)
        return colors

    def _log_hydroelastic_lines(
        self,
        name: str,
        starts: np.ndarray | None,
        ends: np.ndarray | None,
        colors: np.ndarray | None,
        is_enabled: bool,
    ) -> None:
        """Log a hydroelastic line batch, clearing it when its toggle is disabled."""
        self.viewer.log_lines(
            f"/hydroelastic_debug/{name}",
            (
                wp.array(starts, dtype=wp.vec3, device=self.model.device)
                if is_enabled and starts is not None
                else None
            ),
            (
                wp.array(ends, dtype=wp.vec3, device=self.model.device)
                if is_enabled and ends is not None
                else None
            ),
            (
                wp.array(colors, dtype=wp.vec3, device=self.model.device)
                if is_enabled and colors is not None
                else None
            ),
        )

    def _log_hydroelastic_arrows(
        self,
        name: str,
        starts: np.ndarray | None,
        ends: np.ndarray | None,
        colors: np.ndarray | None,
    ) -> None:
        """Log a hydroelastic arrow batch, clearing it when its toggle is disabled."""
        is_enabled = getattr(self, f"show_hydroelastic_{name}")
        self.viewer.log_arrows(
            f"/hydroelastic_debug/{name}",
            (
                wp.array(starts, dtype=wp.vec3, device=self.model.device)
                if is_enabled and starts is not None
                else None
            ),
            (
                wp.array(ends, dtype=wp.vec3, device=self.model.device)
                if is_enabled and ends is not None
                else None
            ),
            (
                wp.array(colors, dtype=wp.vec3, device=self.model.device)
                if is_enabled and colors is not None
                else None
            ),
        )

    def gui(self, imgui) -> None:
        """Render controls and statistics for hydroelastic contact diagnostics."""
        _, self.show_jug_collision_source_mesh = imgui.checkbox(
            "Show jug collision source mesh (gray)",
            self.show_jug_collision_source_mesh,
        )
        _, self.show_jug_sdf_isosurface = imgui.checkbox(
            "Show jug SDF isosurface (yellow)", self.show_jug_sdf_isosurface
        )
        imgui.text("Show Collision is the master switch for both collision layers")
        surface_changed, self.show_hydroelastic_surface = imgui.checkbox(
            "Show hydroelastic contact surface", self.show_hydroelastic_surface
        )
        if surface_changed and self.show_hydroelastic_surface:
            self.show_hydroelastic_depth = False
            self.show_hydroelastic_pressure = False
        depth_changed, self.show_hydroelastic_depth = imgui.checkbox(
            "Show penetration depth", self.show_hydroelastic_depth
        )
        if depth_changed and self.show_hydroelastic_depth:
            self.show_hydroelastic_surface = False
            self.show_hydroelastic_pressure = False
        pressure_changed, self.show_hydroelastic_pressure = imgui.checkbox(
            "Show diffused contact pressure", self.show_hydroelastic_pressure
        )
        if pressure_changed and self.show_hydroelastic_pressure:
            self.show_hydroelastic_surface = False
            self.show_hydroelastic_depth = False
        _, self.show_hydroelastic_normal = imgui.checkbox(
            "Show contact normals", self.show_hydroelastic_normal
        )
        imgui.text("Surface/depth/pressure are mutually exclusive display modes")
        imgui.text("Depth/pressure: blue, cyan, green, yellow, red (high)")
        imgui.text(
            f"Pressure diffusion on SDF surface: {PRESSURE_DIFFUSION_ITERATIONS} iterations, "
            f"rate {PRESSURE_DIFFUSION_RATE:.2f}"
        )
        imgui.text("Contact normal: green")
        imgui.text(f"Hydroelastic faces: {getattr(self, 'hydroelastic_face_count', 0)}")
        imgui.text(
            f"Max penetration: {1.0e3 * getattr(self, 'hydroelastic_max_penetration', 0.0):.3f} mm"
        )
        imgui.text(
            f"Max physical pressure: {getattr(self, 'hydroelastic_max_pressure', 0.0):.3e} Pa"
        )
        imgui.text(
            "Max diffused display pressure: "
            f"{getattr(self, 'hydroelastic_max_diffused_pressure', 0.0):.3e} Pa"
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--log-every", type=int, default=1000, help="每隔多少帧打印一次水壶位置。"
    )
    viewer, args = newton.examples.init(parser)
    assert args.log_every > 0, "--log-every 必须为正整数。"
    newton.examples.run(Example(viewer, args), args)
