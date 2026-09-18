# Copyright (c) 2026, Isaac Lab-Arena Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Visualize all allocated SDF grids as pseudo-colored point clouds."""

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.geometry.sdf_texture import TextureSDFData, texture_sample_sdf
from newton.usd import SchemaResolverNewton, SchemaResolverPhysx


SDF_MAX_RESOLUTION = 256
SDF_NARROW_BAND_RANGE = (-0.02, 0.02)
SDF_CACHE_DIR = Path(__file__).resolve().parents[2] / "scene" / "tmp"
SDF_POINT_STRIDE = 2
SDF_COLOR_RANGE = 0.02


@wp.kernel(enable_backward=False)
def _sample_sdf_points(
    sdf_data: wp.array(dtype=TextureSDFData),
    point_sdf: wp.array(dtype=wp.int32),
    local_points: wp.array(dtype=wp.vec3),
    color_range: float,
    colors: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    distance = texture_sample_sdf(sdf_data[point_sdf[tid]], local_points[tid])
    normalized_distance = wp.min(wp.abs(distance) / color_range, 1.0)

    # Negative distance: blue -> white. Positive distance: white -> red.
    if distance < 0.0:
        colors[tid] = wp.vec3(1.0 - normalized_distance, 1.0 - normalized_distance, 1.0)
    else:
        colors[tid] = wp.vec3(1.0, 1.0 - normalized_distance, 1.0 - normalized_distance)


@wp.kernel(enable_backward=False)
def _transform_points(
    local_points: wp.array(dtype=wp.vec3),
    point_shapes: wp.array(dtype=wp.int32),
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    world_points: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    shape_index = point_shapes[tid]
    body_index = shape_body[shape_index]
    body_transform = wp.transform_identity()
    if body_index >= 0:
        body_transform = body_q[body_index]
    world_points[tid] = wp.transform_point(
        body_transform * shape_transform[shape_index], local_points[tid]
    )


def _grid_points(
    lower: np.ndarray, upper: np.ndarray, cells: np.ndarray, stride: int
) -> np.ndarray:
    """Return regularly spaced grid nodes, retaining the final node on every axis."""
    axes = []
    for axis in range(3):
        count = int(cells[axis])
        indices = np.arange(0, count + 1, stride, dtype=np.int32)
        if indices[-1] != count:
            indices = np.append(indices, count)
        axes.append(lower[axis] + (upper[axis] - lower[axis]) * indices / count)
    coordinates = np.meshgrid(*axes, indexing="ij")
    return np.stack(coordinates, axis=-1).reshape(-1, 3).astype(np.float32)


class Example:
    """Visualize an imported USD model's coarse, narrow, and fallback SDF grids."""

    def __init__(self, viewer, args):
        assert args.usd.is_file(), f"USD 文件不存在：{args.usd}"
        assert args.sdf_point_stride >= 1, "--sdf-point-stride 必须不小于 1"
        assert args.sdf_color_range > 0.0, "--sdf-color-range 必须大于 0"

        self.viewer = viewer
        # Newton 在 ``show_collision`` 时从 shape 的 SDF 做 marching cubes，
        # 显示的正是距离为 0 的碰撞等值面，而不是 USD 的原始渲染网格。
        self.viewer.show_collision = True
        self.show_coarse = True
        self.show_narrow = True
        self.show_fallback = True
        self.color_range = args.sdf_color_range

        builder = newton.ModelBuilder()
        result = builder.add_usd(
            str(args.usd),
            xform=wp.transform_identity(),
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
        )
        self.shape_indices = sorted(set(result["path_shape_map"].values()))
        assert self.shape_indices, f"USD 中未导入任何碰撞形状：{args.usd}"

        for shape_index in self.shape_indices:
            geometry = builder.shape_source[shape_index]
            assert hasattr(
                geometry, "build_sdf"
            ), f"形状 {shape_index} 不是三角网格，无法构建 SDF"
            geometry.build_sdf(
                max_resolution=SDF_MAX_RESOLUTION,
                narrow_band_range=SDF_NARROW_BAND_RANGE,
                margin=builder.shape_gap[shape_index],
                scale=tuple(float(value) for value in builder.shape_scale[shape_index]),
                cache_dir=str(SDF_CACHE_DIR),
            )
            assert geometry.sdf is not None, f"形状 {shape_index} 的 SDF 构建失败"
            builder.shape_force_sdf[shape_index] = True
            # 合并工具导出的碰撞代理会携带 CollisionAPI。这里再显式设置一次，
            # 使旧的纯 Mesh USD 也能被 Show Collision 作为 SDF 等值面显示。
            builder.shape_flags[shape_index] |= int(newton.ShapeFlags.COLLIDE_SHAPES)

        self.model = builder.finalize()
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=(1.2, -1.2, 0.9), pitch=-25.0, yaw=135.0)

        self.sdf_data = wp.array(
            [
                self.model.shape_source[index].sdf.to_texture_kernel_data()
                for index in self.shape_indices
            ],
            dtype=TextureSDFData,
            device=self.model.device,
        )
        self.shape_sdf_indices = {
            shape_index: sdf_index
            for sdf_index, shape_index in enumerate(self.shape_indices)
        }
        self.layers, min_voxel_size = self._create_layers(args.sdf_point_stride)
        self.point_radius = max(0.00025, min_voxel_size * 0.30)

    def _create_layers(self, stride: int):
        records = {"coarse": [], "narrow": [], "fallback": []}
        min_voxel_size = float("inf")

        for shape_index in self.shape_indices:
            sdf = self.model.shape_source[shape_index].sdf
            data = sdf.to_texture_kernel_data()
            lower = np.asarray(data.sdf_box_lower, dtype=np.float32)
            upper = np.asarray(data.sdf_box_upper, dtype=np.float32)
            slots = np.asarray(data.subgrid_start_slots.numpy(), dtype=np.uint32)
            coarse_cells = np.asarray(slots.shape, dtype=np.int32)
            subgrid_size = int(data.subgrid_size)
            voxel_size = np.asarray(data.voxel_size, dtype=np.float32)
            min_voxel_size = min(min_voxel_size, float(np.min(voxel_size)))

            records["coarse"].append(
                (shape_index, _grid_points(lower, upper, coarse_cells, 1))
            )
            empty_slot = np.iinfo(np.uint32).max
            linear_slot = empty_slot - 1
            for block_index in np.ndindex(*coarse_cells):
                slot = int(slots[block_index])
                if slot == empty_slot:
                    continue
                block_index_array = np.asarray(block_index, dtype=np.float32)
                block_lower = lower + (upper - lower) * block_index_array / coarse_cells
                block_upper = (
                    lower + (upper - lower) * (block_index_array + 1.0) / coarse_cells
                )
                if slot == linear_slot:
                    records["fallback"].append(
                        (
                            shape_index,
                            _grid_points(
                                block_lower, block_upper, np.ones(3, dtype=np.int32), 1
                            ),
                        )
                    )
                else:
                    records["narrow"].append(
                        (
                            shape_index,
                            _grid_points(
                                block_lower,
                                block_upper,
                                np.full(3, subgrid_size, dtype=np.int32),
                                stride,
                            ),
                        )
                    )

        return {
            name: self._make_layer(layer_records)
            for name, layer_records in records.items()
        }, min_voxel_size

    def _make_layer(self, records):
        if not records:
            empty_points = wp.empty(0, dtype=wp.vec3, device=self.model.device)
            empty_indices = wp.empty(0, dtype=wp.int32, device=self.model.device)
            return (
                empty_points,
                empty_indices,
                empty_indices,
                empty_points,
                empty_points,
            )

        point_blocks = [points for _, points in records]
        points = np.concatenate(point_blocks, axis=0)
        point_shapes = np.concatenate(
            [
                np.full(len(points), shape_index, dtype=np.int32)
                for shape_index, points in records
            ]
        )
        point_sdf = np.concatenate(
            [
                np.full(
                    len(points), self.shape_sdf_indices[shape_index], dtype=np.int32
                )
                for shape_index, points in records
            ]
        )
        local_points = wp.array(points, dtype=wp.vec3, device=self.model.device)
        color = wp.empty(len(points), dtype=wp.vec3, device=self.model.device)
        world_points = wp.empty_like(local_points)
        point_sdf_array = wp.array(point_sdf, dtype=wp.int32, device=self.model.device)
        wp.launch(
            _sample_sdf_points,
            dim=len(points),
            inputs=[
                self.sdf_data,
                point_sdf_array,
                local_points,
                self.color_range,
                color,
            ],
            device=self.model.device,
        )
        return (
            local_points,
            wp.array(point_shapes, dtype=wp.int32, device=self.model.device),
            point_sdf_array,
            color,
            world_points,
        )

    def update(self):
        for name, visible in (
            ("coarse", self.show_coarse),
            ("narrow", self.show_narrow),
            ("fallback", self.show_fallback),
        ):
            local_points, point_shapes, _, colors, world_points = self.layers[name]
            if len(local_points) == 0:
                self.viewer.log_points(f"/sdf/{name}", None, hidden=True)
                continue
            wp.launch(
                _transform_points,
                dim=len(local_points),
                inputs=[
                    local_points,
                    point_shapes,
                    self.model.shape_transform,
                    self.model.shape_body,
                    self.state.body_q,
                    world_points,
                ],
                device=self.model.device,
            )
            self.viewer.log_points(
                f"/sdf/{name}",
                world_points,
                radii=self.point_radius,
                colors=colors,
                hidden=not visible,
            )

    def render(self):
        self.viewer.begin_frame(0.0)
        self.viewer.log_state(self.state)
        self.update()
        self.viewer.end_frame()

    def step(self):
        """Keep the static visualization compatible with Newton's example loop."""

    def gui(self, imgui):
        """Render point-cloud visibility controls."""
        imgui.text("Show Collision：显示 Newton SDF 的 0 等值面。")
        _, self.show_coarse = imgui.checkbox("显示 coarse 点", self.show_coarse)
        _, self.show_narrow = imgui.checkbox("显示 narrow 点", self.show_narrow)
        _, self.show_fallback = imgui.checkbox("显示 fallback 点", self.show_fallback)
        imgui.text("颜色：负距离为蓝色，零距离为白色，正距离为红色。")


def main():
    parser = newton.examples.create_parser()
    parser.add_argument("usd", type=Path, help="要可视化的 USD 模型路径")
    parser.add_argument(
        "--sdf-point-stride",
        type=int,
        default=SDF_POINT_STRIDE,
        help="narrow SDF 子网格的点采样步长；数值越大，显示越稀疏。",
    )
    parser.add_argument(
        "--sdf-color-range",
        type=float,
        default=SDF_COLOR_RANGE,
        help="伪彩色距离饱和范围（米）。",
    )
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()
