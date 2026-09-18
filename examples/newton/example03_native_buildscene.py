# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Newton 原生场景：运动学台面、水壶和目标容器。

该场景不经过 IsaacLab，但使用与 ``example02_buildscene.py`` 相同的运动学
台面和本地 USD 资产，用于隔离 IsaacLab 集成层的影响。

运行方法：

    .venv/bin/python examples/newton/example03_native_buildscene.py
    .venv/bin/python examples/newton/example03_native_buildscene.py --log-every 10
"""

import tempfile
import warnings
from pathlib import Path

import numpy as np
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
CONTAINER_POSITION = (0.45, 0.22, 0.0)
MESH_SDF_MAX_RESOLUTION = 512
MESH_SDF_NARROW_BAND_RANGE = (-0.02, 0.02)
MESH_SDF_CACHE_DIR = Path(tempfile.gettempdir()) / "newton_sdf_cache"
RIGID_CONTACT_GAP = 0.005
RIGID_CONTACT_MAX = 4096
SDF_GRID_LINE_STRIDE = 8


@wp.kernel(enable_backward=False)
def _transform_shape_local_lines(
    local_starts: wp.array[wp.vec3],
    local_ends: wp.array[wp.vec3],
    line_shape: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_world: wp.array[wp.int32],
    world_offsets: wp.array[wp.vec3],
    world_starts: wp.array[wp.vec3],
    world_ends: wp.array[wp.vec3],
):
    """Transform shape-local SDF grid lines into viewer world space."""
    tid = wp.tid()
    shape_idx = line_shape[tid]
    body_idx = shape_body[shape_idx]
    X_ws = shape_transform[shape_idx]
    if body_idx >= 0:
        X_ws = wp.transform_multiply(body_q[body_idx], X_ws)

    offset = wp.vec3(0.0, 0.0, 0.0)
    world_idx = shape_world[shape_idx]
    if world_idx >= 0 and world_idx < world_offsets.shape[0]:
        offset = world_offsets[world_idx]

    world_starts[tid] = wp.transform_point(X_ws, local_starts[tid]) + offset
    world_ends[tid] = wp.transform_point(X_ws, local_ends[tid]) + offset


def _get_container_stage_in_metres() -> Usd.Stage:
    """Open the centimetre-authored container with an explicit metre scale."""
    stage = Usd.Stage.Open(str(CONTAINER_USD_PATH))
    assert stage is not None, f"无法打开目标容器 USD：{CONTAINER_USD_PATH}"
    root_prim = stage.GetDefaultPrim()
    assert root_prim, f"目标容器 USD 没有默认 Prim：{CONTAINER_USD_PATH}"
    UsdGeom.Xformable(root_prim).AddScaleOp().Set(Gf.Vec3f(0.01, 0.01, 0.01))
    # Newton does not apply a non-unit metersPerUnit value itself. The explicit
    # root scale above performs the conversion, so neutralize the metadata too.
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    return stage


def _report_collision_models(builder: newton.ModelBuilder, asset_name: str, shape_indices: list[int]) -> None:
    """Print the collision geometry Newton actually created for one imported asset."""
    assert shape_indices, f"{asset_name} USD 未导入任何碰撞 shape。"
    for shape_index in shape_indices:
        geometry_type = getattr(builder.shape_type[shape_index], "name", str(builder.shape_type[shape_index]))
        geometry = builder.shape_source[shape_index]
        if getattr(geometry, "sdf", None) is not None:
            collision_model = "SDF"
        elif geometry_type == "MESH":
            collision_model = "三角网格（未生成 SDF，求解器可能使用凸包近似）"
        else:
            collision_model = geometry_type
        watertight = getattr(geometry, "is_watertight", "n/a")
        force_sdf = builder.shape_force_sdf[shape_index]
        print(
            f"[INFO] {asset_name} shape={shape_index}: 最终碰撞模型={collision_model} "
            f"(geometry_type={geometry_type}, watertight={watertight}, force_sdf={force_sdf})",
            flush=True,
        )


def _build_sdf_collision_models(builder: newton.ModelBuilder, asset_name: str, shape_indices: list[int]) -> None:
    """Cook cached SDFs for imported mesh colliders and force Newton to use them."""
    assert shape_indices, f"{asset_name} USD 未导入任何碰撞 shape。"
    for shape_index in shape_indices:
        geometry = builder.shape_source[shape_index]
        assert hasattr(geometry, "build_sdf"), f"{asset_name} shape={shape_index} 不是可构造 SDF 的网格。"
        shape_scale = tuple(float(value) for value in builder.shape_scale[shape_index])
        print(
            f"[INFO] 正在为 {asset_name} shape={shape_index} 构造或读取缓存 SDF "
            f"(baked_scale={shape_scale})...",
            flush=True,
        )
        geometry.build_sdf(
            max_resolution=MESH_SDF_MAX_RESOLUTION,
            narrow_band_range=MESH_SDF_NARROW_BAND_RANGE,
            margin=builder.shape_gap[shape_index],
            scale=shape_scale,
            cache_dir=MESH_SDF_CACHE_DIR,
        )
        assert geometry.sdf is not None, f"{asset_name} shape={shape_index} 的 SDF 构造失败。"
        builder.shape_force_sdf[shape_index] = True


class Example:
    """Build and simulate the IsaacLab-equivalent scene with native Newton APIs."""

    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True

        self.viewer = viewer
        self.log_every = args.log_every
        self.show_sdf_bounds = hasattr(viewer, "renderer")
        self.show_sdf_coarse = hasattr(viewer, "renderer")
        self.show_sdf_narrow_band = hasattr(viewer, "renderer")
        self.show_sdf_linear_cells = False
        self.sdf_grid_line_stride = args.sdf_grid_stride
        assert self.sdf_grid_line_stride >= 1, "--sdf-grid-stride 必须至少为 1。"
        self.step_count = 0
        self.frame_dt = 1.0 / 120.0
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        # Newton's 0.1 m default gap is excessive for these decimetre-scale
        # assets and is also used as SDF AABB padding by _build_sdf_collision_models.
        builder.rigid_gap = RIGID_CONTACT_GAP
        builder.default_shape_cfg.mu = 1.0

        # Match example02: the table is a kinematic rigid body rather than a world shape.
        table_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(*TABLE_POSITION), q=wp.quat_identity()),
            is_kinematic=True,
            label="table",
        )
        builder.add_shape_box(
            body=table_body,
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            color=wp.vec3(0.2, 0.2, 0.2),
            label="table",
        )

        jug_result = builder.add_usd(
            str(JUG_USD_PATH),
            xform=wp.transform(p=wp.vec3(*JUG_POSITION), q=wp.quat_identity()),
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
        )
        jug_bodies = list(jug_result["path_body_map"].values())
        assert len(jug_bodies) == 1, "水壶 Physics USD 应当只包含一个刚体。"
        self.jug_body = jug_bodies[0]
        jug_shape_indices = list(jug_result["path_shape_map"].values())
        _build_sdf_collision_models(builder, "水壶", jug_shape_indices)
        _report_collision_models(builder, "水壶", jug_shape_indices)

        # Newton ignores non-unit metersPerUnit metadata. The helper converts
        # this centimetre-authored 40x30x12 asset to 0.4x0.3x0.12 m explicitly.
        # It has collision geometry but no rigid body, so Newton imports it as
        # a static target.
        container_result = builder.add_usd(
            _get_container_stage_in_metres(),
            xform=wp.transform(p=wp.vec3(*CONTAINER_POSITION), q=wp.quat_identity()),
            # floating=False,
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
        )
        assert container_result["path_shape_map"], "目标容器 USD 未提供可导入的几何。"
        container_shape_indices = list(container_result["path_shape_map"].values())
        _build_sdf_collision_models(builder, "目标容器", container_shape_indices)
        _report_collision_models(builder, "目标容器", container_shape_indices)

        self.model = builder.finalize()

        # Do not use MuJoCo's internal mesh-contact path here: it can replace a
        # mesh by a convex approximation.  The Newton collision pipeline uses
        # the SDFs built above and supplies those contacts to the solver.
        self.rigid_contact_max = RIGID_CONTACT_MAX
        self.model.rigid_contact_max = self.rigid_contact_max
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            rigid_contact_max=self.rigid_contact_max,
            broad_phase="sap",
        )
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            njmax=self.rigid_contact_max,
            nconmax=self.rigid_contact_max,
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
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(2.0, -2.0, 1.5),
            pitch=-15.0,
            yaw=135.0,
        )
        self._init_sdf_grid_visualization()
        self.graph = self._capture()

    def _init_sdf_grid_visualization(self) -> None:
        """Build shape-local lines for SDF bounds, coarse grids, and narrow-band grids."""
        bounds_starts, bounds_ends, bounds_shapes = [], [], []
        coarse_starts, coarse_ends, coarse_shapes = [], [], []
        narrow_starts, narrow_ends, narrow_shapes = [], [], []
        linear_starts, linear_ends, linear_shapes = [], [], []
        self.sdf_grid_info = []
        seen_sdfs = set()

        def append_line(starts, ends, shapes, start, end, shape_idx):
            starts.append(start)
            ends.append(end)
            shapes.append(shape_idx)

        def append_box(starts, ends, shapes, lower, upper, shape_idx):
            corners = [
                np.array((x, y, z), dtype=np.float32)
                for x in (lower[0], upper[0])
                for y in (lower[1], upper[1])
                for z in (lower[2], upper[2])
            ]
            box_edges = (
                (0, 1),
                (0, 2),
                (0, 4),
                (1, 3),
                (1, 5),
                (2, 3),
                (2, 6),
                (3, 7),
                (4, 5),
                (4, 6),
                (5, 7),
                (6, 7),
            )
            for a, b in box_edges:
                append_line(starts, ends, shapes, corners[a], corners[b], shape_idx)

        def sample_axis(lower, upper, cell_count, stride):
            indices = list(range(0, int(cell_count) + 1, stride))
            if indices[-1] != int(cell_count):
                indices.append(int(cell_count))
            return [lower + (upper - lower) * i / int(cell_count) for i in indices]

        def append_lattice(starts, ends, shapes, lower, upper, cell_counts, stride, shape_idx):
            xs = sample_axis(lower[0], upper[0], cell_counts[0], stride)
            ys = sample_axis(lower[1], upper[1], cell_counts[1], stride)
            zs = sample_axis(lower[2], upper[2], cell_counts[2], stride)
            for y in ys:
                for z in zs:
                    append_line(starts, ends, shapes, (lower[0], y, z), (upper[0], y, z), shape_idx)
            for x in xs:
                for z in zs:
                    append_line(starts, ends, shapes, (x, lower[1], z), (x, upper[1], z), shape_idx)
            for x in xs:
                for y in ys:
                    append_line(starts, ends, shapes, (x, y, lower[2]), (x, y, upper[2]), shape_idx)

        for shape_idx, source in enumerate(self.model.shape_source):
            sdf = getattr(source, "sdf", None)
            texture_data = sdf.to_texture_kernel_data() if sdf is not None else None
            if texture_data is None:
                continue

            lower = np.asarray(texture_data.sdf_box_lower, dtype=np.float32)
            upper = np.asarray(texture_data.sdf_box_upper, dtype=np.float32)
            voxel_size = np.asarray(texture_data.voxel_size, dtype=np.float32)
            subgrid_size = int(texture_data.subgrid_size)
            slots = texture_data.subgrid_start_slots.numpy()
            coarse_resolution = np.asarray(slots.shape, dtype=np.int32)
            fine_resolution = coarse_resolution * subgrid_size

            sdf_key = id(sdf)
            if sdf_key not in seen_sdfs:
                seen_sdfs.add(sdf_key)
                slot_empty = np.iinfo(np.uint32).max
                slot_linear = slot_empty - 1
                self.sdf_grid_info.append(
                    (
                        self.model.shape_label[shape_idx],
                        coarse_resolution,
                        fine_resolution,
                        voxel_size,
                        int(np.count_nonzero(slots < slot_linear)),
                        int(np.count_nonzero(slots == slot_linear)),
                        int(np.count_nonzero(slots == slot_empty)),
                    )
                )

            append_box(bounds_starts, bounds_ends, bounds_shapes, lower, upper, shape_idx)
            append_lattice(coarse_starts, coarse_ends, coarse_shapes, lower, upper, coarse_resolution, 1, shape_idx)

            slot_empty = np.iinfo(np.uint32).max
            slot_linear = slot_empty - 1
            block_size = voxel_size * subgrid_size
            for block_idx in np.ndindex(slots.shape):
                slot = slots[block_idx]
                if slot == slot_empty:
                    continue
                block_lower = lower + np.asarray(block_idx, dtype=np.float32) * block_size
                block_upper = np.minimum(block_lower + block_size, upper)
                if slot == slot_linear:
                    append_box(linear_starts, linear_ends, linear_shapes, block_lower, block_upper, shape_idx)
                else:
                    append_lattice(
                        narrow_starts,
                        narrow_ends,
                        narrow_shapes,
                        block_lower,
                        block_upper,
                        (subgrid_size, subgrid_size, subgrid_size),
                        self.sdf_grid_line_stride,
                        shape_idx,
                    )

        device = self.model.device

        def make_line_set(starts, ends, shapes):
            local_starts = wp.array(starts, dtype=wp.vec3, device=device)
            local_ends = wp.array(ends, dtype=wp.vec3, device=device)
            return (
                local_starts,
                local_ends,
                wp.array(shapes, dtype=wp.int32, device=device),
                wp.empty_like(local_starts),
                wp.empty_like(local_ends),
            )

        self.sdf_bounds_lines = make_line_set(bounds_starts, bounds_ends, bounds_shapes)
        self.sdf_coarse_lines = make_line_set(coarse_starts, coarse_ends, coarse_shapes)
        self.sdf_narrow_lines = make_line_set(narrow_starts, narrow_ends, narrow_shapes)
        self.sdf_linear_lines = make_line_set(linear_starts, linear_ends, linear_shapes)

    def _render_sdf_grid_visualization(self) -> None:
        """Render the SDF grids using the current rigid-body transforms."""
        line_sets = (
            ("/sdf/bounds", self.show_sdf_bounds, self.sdf_bounds_lines, (1.0, 0.55, 0.05)),
            (
                "/sdf/coarse",
                self.show_sdf_coarse,
                self.sdf_coarse_lines,
                (0.45, 0.45, 0.45),
            ),
            (
                "/sdf/narrow_band",
                self.show_sdf_narrow_band,
                self.sdf_narrow_lines,
                (0.0, 0.8, 1.0),
            ),
            (
                "/sdf/linear_cells",
                self.show_sdf_linear_cells,
                self.sdf_linear_lines,
                (1.0, 0.9, 0.1),
            ),
        )
        for name, visible, lines, color in line_sets:
            local_starts, local_ends, shapes, world_starts, world_ends = lines
            if not visible or len(local_starts) == 0:
                self.viewer.log_lines(name, None, None, None)
                continue
            wp.launch(
                _transform_shape_local_lines,
                dim=len(local_starts),
                inputs=[
                    local_starts,
                    local_ends,
                    shapes,
                    self.model.shape_transform,
                    self.model.shape_body,
                    self.state_0.body_q,
                    self.model.shape_world,
                    self.viewer.world_offsets,
                ],
                outputs=[world_starts, world_ends],
                device=self.model.device,
            )
            self.viewer.log_lines(name, world_starts, world_ends, color)

    def _simulate(self) -> None:
        """Advance one displayed frame using MuJoCo-Warp substeps."""
        # Refresh SDF contacts every substep. Reusing contacts for all substeps
        # can leave fast-moving or mouse-dragged bodies using stale normals and
        # contact positions long enough to cross a thin container wall.
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
            print(
                f"[INFO] step={self.step_count:04d}, 水壶世界坐标: "
                f"x={position[0]:+.5f}, y={position[1]:+.5f}, z={position[2]:+.5f}",
                flush=True,
            )

    def render(self) -> None:
        """Render the current Newton state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self._render_sdf_grid_visualization()
        self.viewer.end_frame()

    def gui(self, imgui) -> None:
        """Render controls and metadata for the SDF grid visualization."""
        _, self.show_sdf_bounds = imgui.checkbox("Show SDF bounds", self.show_sdf_bounds)
        _, self.show_sdf_coarse = imgui.checkbox("Show coarse grid", self.show_sdf_coarse)
        _, self.show_sdf_narrow_band = imgui.checkbox("Show narrow-band grid", self.show_sdf_narrow_band)
        _, self.show_sdf_linear_cells = imgui.checkbox(
            "Show narrow-band coarse fallback", self.show_sdf_linear_cells
        )
        imgui.text(f"Narrow-band lines: every {self.sdf_grid_line_stride} fine voxels")
        imgui.text("Orange: bounds, gray: coarse, cyan: fine, yellow: coarse fallback")
        for label, coarse, fine, voxel, fine_count, linear_count, empty_count in self.sdf_grid_info:
            imgui.text(
                f"{label}: coarse {coarse[0]}x{coarse[1]}x{coarse[2]}, "
                f"logical fine {fine[0]}x{fine[1]}x{fine[2]}"
            )
            imgui.text(
                f"voxel {1.0e3 * voxel[0]:.3f}x{1.0e3 * voxel[1]:.3f}x"
                f"{1.0e3 * voxel[2]:.3f} mm"
            )
            imgui.text(f"blocks: fine {fine_count}, coarse fallback {linear_count}, empty {empty_count}")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--log-every", type=int, default=1000, help="每隔多少帧打印一次水壶位置。")
    parser.add_argument(
        "--sdf-grid-stride",
        type=int,
        default=SDF_GRID_LINE_STRIDE,
        help="窄带块中每隔 N 个精细体素绘制一条网格线。",
    )
    viewer, args = newton.examples.init(parser)
    assert args.log_every > 0, "--log-every 必须为正整数。"
    newton.examples.run(Example(viewer, args), args)
