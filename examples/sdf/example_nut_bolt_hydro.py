# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example SDF Mesh Collision
#
# Demonstrates nut/bolt mesh collision using hydroelastic contacts.
#
# Command: python -m newton.examples nut_bolt_hydro
#
###########################################################################

import argparse
import tempfile
from pathlib import Path

import numpy as np
import trimesh
import warp as wp

import newton
import newton.examples
from newton.geometry import HydroelasticSDF

# Assembly type for the nut and bolt
ASSEMBLY_STR = "m20_loose"

ISAACGYM_ENVS_REPO_URL = "https://github.com/isaac-sim/IsaacGymEnvs.git"
ISAACGYM_NUT_BOLT_FOLDER = "assets/factory/mesh/factory_nut_bolt"

SDF_MAX_RESOLUTION = 128
SDF_NARROW_BAND_RANGE = (-0.005, 0.005)
# The viewer maps 0.5 mm of penetration to the red end of its hydroelastic
# surface colormap. The UI converts that depth to pressure using SHAPE_CFG.kh.
PRESSURE_COLORMAP_MAX_DEPTH = 0.0005
PRESSURE_ZERO_COLOR = (0.0, 0.0, 1.0)
SDF_GRID_LINE_STRIDE = 8
# Persist cooked SDFs across runs so the (slow) cook only happens once.
# Entries are content-addressed, so leftovers from older runs are harmless.
MESH_SDF_CACHE_DIR = Path(tempfile.gettempdir()) / "newton_sdf_cache"

SHAPE_CFG = newton.ModelBuilder.ShapeConfig(
    margin=0.0,
    mu=0.01,
    # Hydroelastic supplies the per-contact stiffness for the nut/bolt pair, so
    # ``ke``/``kd`` reach only the non-hydroelastic contacts. At the 1e10 ``kh``
    # default the thread contacts resolve over a ~95 ms solref time constant --
    # roughly 45 substeps -- and the nut sits visibly cocked under MuJoCo. XPBD
    # projects positions and is insensitive to this either way.
    kh=1e10,  # Hydroelastic contact stiffness
    ke=1e7,
    kd=1e4,
    gap=0.005,
    density=8000.0,
    mu_torsional=0.0,
    mu_rolling=0.0,
    is_hydroelastic=True,
)


# Demonstrate the user-facing pressure-callback API for the hydroelastic
# solver. The contact patch is defined as the iso-pressure surface
# ``p_a == p_b``; users supply a Warp ``@wp.func`` that maps a signed depth
# to a pressure value, plus a ``@wp.struct`` carrying any per-shape state it
# needs. The callback must be finite for any signed depth (positive or
# negative) and monotone non-increasing in ``signed_depth`` so the marching-
# cubes interpolation stays continuous across the patch boundary. Do not clip
# the non-contact side to zero pressure; with different shape stiffnesses, the
# iso-pressure surface can pass through that thin outside region.
#
# Here we re-implement the default linear law ``pressure = -kh * signed_depth``
# explicitly so the example exercises the user pathway. Nonlinear laws should
# similarly extend into ``signed_depth >= 0`` instead of flattening there.
@wp.struct
class LinearPressureData:
    shape_kh: wp.array[wp.float32]


@wp.func
def linear_pressure(signed_depth: wp.float32, shape_idx: wp.int32, data: LinearPressureData) -> wp.float32:
    return -data.shape_kh[shape_idx] * signed_depth


@wp.kernel(enable_backward=False)
def transform_shape_local_lines(
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
    """Transform shape-local visualization lines into viewer world space."""
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


def add_mesh_object(
    builder: newton.ModelBuilder,
    mesh: newton.Mesh,
    transform: wp.transform,
    shape_cfg: newton.ModelBuilder.ShapeConfig | None = None,
    key: str | None = None,
    center_vec: wp.vec3 | None = None,
    scale: float = 1.0,
) -> int:
    """Add a mesh shape on a new body.

    Args:
        builder: Model builder that receives the body and shape.
        mesh: Mesh geometry with SDF data.
        transform: Body transform with position [m] and orientation.
        shape_cfg: Optional shape configuration.
        key: Optional body label.
        center_vec: Optional mesh center offset in local coordinates [m].
        scale: Uniform mesh scale [unitless].

    Returns:
        Created body index.
    """
    if center_vec is not None:
        center_world = wp.quat_rotate(transform.q, center_vec)
        transform = wp.transform(transform.p + center_world, transform.q)

    body = builder.add_body(label=key, xform=transform)
    builder.add_shape_mesh(body, mesh=mesh, scale=(scale, scale, scale), cfg=shape_cfg)
    return body


def load_mesh_with_sdf(
    mesh_file: str,
    shape_cfg: newton.ModelBuilder.ShapeConfig | None = None,
    scale: float = 1.0,
    center_origin: bool = True,
) -> tuple[newton.Mesh, wp.vec3]:
    """Load a triangle mesh and build an SDF.

    Args:
        mesh_file: Mesh file path.
        shape_cfg: Optional shape configuration used for contact margin [m].
        scale: Uniform mesh scale [unitless].
        center_origin: Whether to recenter mesh vertices about the AABB center.

    Returns:
        Tuple of ``(mesh, center_vec)`` where ``center_vec`` is the recenter offset [m].
    """
    mesh_data = trimesh.load(mesh_file, force="mesh")
    vertices = np.array(mesh_data.vertices, dtype=np.float32)
    indices = np.array(mesh_data.faces.flatten(), dtype=np.int32)
    center_vec = wp.vec3(0.0, 0.0, 0.0)

    if center_origin:
        min_extent = vertices.min(axis=0)
        max_extent = vertices.max(axis=0)
        center = (min_extent + max_extent) / 2
        vertices = vertices - center
        center_vec = wp.vec3(center) * float(scale)

    mesh = newton.Mesh(vertices, indices)
    mesh.build_sdf(
        max_resolution=SDF_MAX_RESOLUTION,
        narrow_band_range=SDF_NARROW_BAND_RANGE,
        margin=shape_cfg.gap if shape_cfg and shape_cfg.gap is not None else 0.005,
        scale=(scale, scale, scale),
        cache_dir=MESH_SDF_CACHE_DIR,
    )
    return mesh, center_vec


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.collide_every = 2 if args.solver == "mujoco" else 1  # re-collide every K substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.world_count = args.world_count
        self.viewer = viewer
        self.show_pressure_surface = hasattr(viewer, "renderer")
        self.show_sdf_bounds = hasattr(viewer, "renderer")
        self.show_sdf_coarse = hasattr(viewer, "renderer")
        self.show_sdf_narrow_band = hasattr(viewer, "renderer")
        self.show_sdf_linear_cells = False
        self.sdf_grid_line_stride = args.sdf_grid_stride
        if self.sdf_grid_line_stride < 1:
            raise ValueError("--sdf-grid-stride must be at least 1")
        self.solver_type = args.solver
        self.test_mode = args.test
        self.deterministic = args.deterministic
        self.deterministic_solver = args.deterministic_solver

        # XPBD contact correction (0.0 = no correction, 1.0 = full correction)
        self.xpbd_contact_relaxation = 0.8

        # Scene scaling factor (1.0 = original size)
        self.scene_scale = 1.0

        # Ground plane offset (negative = below origin)
        self.ground_plane_offset = -0.01

        # Grid dimensions for nut/bolt scene (number of assemblies in X and Y)
        self.num_per_world = args.num_per_world
        self.grid_x = int(np.ceil(np.sqrt(self.num_per_world)))
        self.grid_y = int(np.ceil(self.num_per_world / self.grid_x))

        # Contact budget per world. MuJoCo's njmax/nconmax are per-world limits
        # while CollisionPipeline's rigid_contact_max covers every world, so both
        # derive from this rather than from a fixed whole-scene pool. A threading
        # assembly peaks near 115 contacts, so this leaves ~9x headroom.
        self.contacts_per_world = 1024 * self.num_per_world

        # Maximum number of rigid contacts to allocate (limits memory usage)
        self.rigid_contact_max = self.contacts_per_world * self.world_count

        # Broad phase mode: NXN (O(N²)), SAP (O(N log N)), EXPLICIT (precomputed pairs)
        self.broad_phase_mode = "sap"

        world_builder = self._build_nut_bolt_scene()

        main_scene = newton.ModelBuilder()
        main_scene.default_shape_cfg.gap = 0.001 * self.scene_scale
        # Add ground plane at z = ground_plane_offset.
        # For plane equation n·x + d = 0, with n=(0,0,1): z + d = 0, so z = -d.
        # Therefore d is the negative offset, and z = offset uses d = -offset.
        main_scene.add_shape_plane(
            plane=(0.0, 0.0, 1.0, -self.ground_plane_offset),
            width=0.0,
            length=0.0,
            label="ground_plane",
        )
        main_scene.replicate(world_builder, world_count=self.world_count)

        self.model = main_scene.finalize()

        # Configure the hydroelastic pipeline with our custom (still linear)
        # pressure callback. ``shape_kh`` reuses the per-shape stiffness already
        # stored on the model. Disable the marching-cubes edge clamp because
        # threading dynamics on the M20 helix are sensitive to the contact-
        # surface vertex bias the clamp introduces.
        pressure_data = LinearPressureData()
        pressure_data.shape_kh = self.model.shape_material_kh
        sdf_hydroelastic_config = HydroelasticSDF.Config(
            pressure_func=linear_pressure,
            pressure_data=pressure_data,
            mc_edge_clamp_min=0.0,
            output_contact_surface=self.show_pressure_surface,
        )

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            rigid_contact_max=self.rigid_contact_max,
            broad_phase=self.broad_phase_mode,
            sdf_hydroelastic_config=sdf_hydroelastic_config,
            deterministic=self.deterministic,
        )

        # Create solver based on user choice
        if self.solver_type == "xpbd":
            self.solver = newton.solvers.SolverXPBD(
                self.model,
                iterations=10,
                rigid_contact_relaxation=self.xpbd_contact_relaxation,
                deterministic=wp.DeterministicMode.RUN_TO_RUN
                if self.deterministic_solver
                else wp.DeterministicMode.NOT_GUARANTEED,
            )
        elif self.solver_type == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                use_mujoco_contacts=False,
                # The scene defines no sensors, and MuJoCo's tactile sensor kernel
                # mixes max/add reductions, which deterministic mode rejects.
                disable_sensors=True,
                solver="newton",
                integrator="implicitfast",
                cone="elliptic",
                njmax=self.contacts_per_world,
                nconmax=self.contacts_per_world,
                iterations=15,
                ls_iterations=100,
                impratio=1.0,
                deterministic=wp.DeterministicMode.RUN_TO_RUN
                if self.deterministic_solver
                else wp.DeterministicMode.NOT_GUARANTEED,
            )
        else:
            raise ValueError(f"Unknown solver '{self.solver_type}'")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.contacts = self.collision_pipeline.contacts()
        self.collision_pipeline.collide(self.state_0, self.contacts)

        self.viewer.set_model(self.model)
        self.viewer.show_hydro_contact_surface = self.show_pressure_surface
        self.viewer.show_visual = not self.show_pressure_surface

        offset = 0.15 * self.scene_scale
        self.viewer.set_world_offsets((offset, offset, 0.0))
        camera_offset = np.sqrt(self.world_count) * offset * 1.25
        self.viewer.set_camera(pos=wp.vec3(camera_offset, -camera_offset, 0.5 * camera_offset), pitch=-15.0, yaw=135.0)
        self._init_pressure_mesh_visualization()
        self._init_sdf_grid_visualization()

        # Initialize test tracking data (only in test mode for nut_bolt scene)
        self._init_test_tracking()

        self.capture()

    def _build_nut_bolt_scene(self) -> newton.ModelBuilder:
        print("Downloading nut/bolt assets...")
        asset_path = newton.examples.download_external_git_folder(ISAACGYM_ENVS_REPO_URL, ISAACGYM_NUT_BOLT_FOLDER)
        print(f"Assets downloaded to: {asset_path}")

        world_builder = newton.ModelBuilder()
        world_builder.default_shape_cfg.gap = 0.001 * self.scene_scale

        bolt_file = str(asset_path / f"factory_bolt_{ASSEMBLY_STR}.obj")
        nut_file = str(asset_path / f"factory_nut_{ASSEMBLY_STR}_subdiv_3x.obj")
        bolt_mesh, bolt_center = load_mesh_with_sdf(
            bolt_file, shape_cfg=SHAPE_CFG, scale=self.scene_scale, center_origin=True
        )
        nut_mesh, nut_center = load_mesh_with_sdf(
            nut_file, shape_cfg=SHAPE_CFG, scale=self.scene_scale, center_origin=True
        )

        # Spacing between assemblies in the grid
        spacing = 0.1 * self.scene_scale

        # Create grid of nut/bolt assemblies
        count = 0
        for i in range(self.grid_x):
            if count >= self.num_per_world:
                break
            for j in range(self.grid_y):
                if count >= self.num_per_world:
                    break
                # Center the grid around origin
                x_offset = (i - (self.grid_x - 1) / 2.0) * spacing
                y_offset = (j - (self.grid_y - 1) / 2.0) * spacing

                # Add bolt at grid position
                bolt_xform = wp.transform(wp.vec3(x_offset, y_offset, 0.0 * self.scene_scale), wp.quat_identity())
                add_mesh_object(
                    world_builder,
                    bolt_mesh,
                    bolt_xform,
                    SHAPE_CFG,
                    key=f"bolt_{i}_{j}",
                    center_vec=bolt_center,
                    scale=self.scene_scale,
                )

                # Add nut above bolt at grid position
                nut_xform = wp.transform(
                    wp.vec3(x_offset, y_offset, 0.041 * self.scene_scale),
                    wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 8),
                )
                add_mesh_object(
                    world_builder,
                    nut_mesh,
                    nut_xform,
                    SHAPE_CFG,
                    key=f"nut_{i}_{j}",
                    center_vec=nut_center,
                    scale=self.scene_scale,
                )
                count += 1

        return world_builder

    def capture(self):
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        for sub in range(self.sim_substeps):
            # Refresh contacts every K substeps so contact normals stay
            # aligned with the threading rotation.
            if sub % self.collide_every == 0:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.state_0.clear_forces()

            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

        # Track transforms for test validation
        self._track_test_data()

    def _init_pressure_mesh_visualization(self):
        """Build the complete hydroelastic mesh used as the zero-pressure layer."""
        starts = []
        ends = []
        line_shapes = []
        shape_flags = self.model.shape_flags.numpy()
        shape_scales = self.model.shape_scale.numpy()

        for shape_idx, source in enumerate(self.model.shape_source):
            if not shape_flags[shape_idx] & int(newton.ShapeFlags.HYDROELASTIC):
                continue
            if not isinstance(source, newton.Mesh):
                continue

            vertices = np.asarray(source.vertices, dtype=np.float32) * shape_scales[shape_idx]
            triangles = np.asarray(source.indices, dtype=np.int32).reshape(-1, 3)
            edges = np.concatenate(
                (triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)]),
                axis=0,
            )
            edges.sort(axis=1)
            edges = np.unique(edges, axis=0)
            starts.append(vertices[edges[:, 0]])
            ends.append(vertices[edges[:, 1]])
            line_shapes.append(np.full(len(edges), shape_idx, dtype=np.int32))

        if starts:
            starts = np.concatenate(starts, axis=0)
            ends = np.concatenate(ends, axis=0)
            line_shapes = np.concatenate(line_shapes)
        else:
            starts = np.empty((0, 3), dtype=np.float32)
            ends = np.empty((0, 3), dtype=np.float32)
            line_shapes = np.empty(0, dtype=np.int32)

        device = self.model.device
        self.pressure_mesh_local_starts = wp.array(starts, dtype=wp.vec3, device=device)
        self.pressure_mesh_local_ends = wp.array(ends, dtype=wp.vec3, device=device)
        self.pressure_mesh_shapes = wp.array(line_shapes, dtype=wp.int32, device=device)
        self.pressure_mesh_world_starts = wp.empty_like(self.pressure_mesh_local_starts)
        self.pressure_mesh_world_ends = wp.empty_like(self.pressure_mesh_local_ends)

    def _render_pressure_mesh_visualization(self):
        """Render every mesh edge blue so zero-pressure regions remain visible."""
        if not self.show_pressure_surface or len(self.pressure_mesh_local_starts) == 0:
            self.viewer.log_lines("/pressure/zero_mesh", None, None, None)
            return

        wp.launch(
            transform_shape_local_lines,
            dim=len(self.pressure_mesh_local_starts),
            inputs=[
                self.pressure_mesh_local_starts,
                self.pressure_mesh_local_ends,
                self.pressure_mesh_shapes,
                self.model.shape_transform,
                self.model.shape_body,
                self.state_0.body_q,
                self.model.shape_world,
                self.viewer.world_offsets,
            ],
            outputs=[self.pressure_mesh_world_starts, self.pressure_mesh_world_ends],
            device=self.model.device,
        )
        self.viewer.log_lines(
            "/pressure/zero_mesh",
            self.pressure_mesh_world_starts,
            self.pressure_mesh_world_ends,
            PRESSURE_ZERO_COLOR,
        )

    def _init_sdf_grid_visualization(self):
        """Build shape-local lines for coarse and sparse narrow-band SDF grids."""
        bounds_starts = []
        bounds_ends = []
        bounds_shapes = []
        coarse_starts = []
        coarse_ends = []
        coarse_shapes = []
        narrow_starts = []
        narrow_ends = []
        narrow_shapes = []
        linear_starts = []
        linear_ends = []
        linear_shapes = []
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
            append_lattice(
                coarse_starts,
                coarse_ends,
                coarse_shapes,
                lower,
                upper,
                coarse_resolution,
                1,
                shape_idx,
            )

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
                    append_box(
                        linear_starts,
                        linear_ends,
                        linear_shapes,
                        block_lower,
                        block_upper,
                        shape_idx,
                    )
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
        self.sdf_bounds_local_starts = wp.array(bounds_starts, dtype=wp.vec3, device=device)
        self.sdf_bounds_local_ends = wp.array(bounds_ends, dtype=wp.vec3, device=device)
        self.sdf_bounds_shapes = wp.array(bounds_shapes, dtype=wp.int32, device=device)
        self.sdf_bounds_world_starts = wp.empty_like(self.sdf_bounds_local_starts)
        self.sdf_bounds_world_ends = wp.empty_like(self.sdf_bounds_local_ends)

        self.sdf_coarse_local_starts = wp.array(coarse_starts, dtype=wp.vec3, device=device)
        self.sdf_coarse_local_ends = wp.array(coarse_ends, dtype=wp.vec3, device=device)
        self.sdf_coarse_shapes = wp.array(coarse_shapes, dtype=wp.int32, device=device)
        self.sdf_coarse_world_starts = wp.empty_like(self.sdf_coarse_local_starts)
        self.sdf_coarse_world_ends = wp.empty_like(self.sdf_coarse_local_ends)

        self.sdf_narrow_local_starts = wp.array(narrow_starts, dtype=wp.vec3, device=device)
        self.sdf_narrow_local_ends = wp.array(narrow_ends, dtype=wp.vec3, device=device)
        self.sdf_narrow_shapes = wp.array(narrow_shapes, dtype=wp.int32, device=device)
        self.sdf_narrow_world_starts = wp.empty_like(self.sdf_narrow_local_starts)
        self.sdf_narrow_world_ends = wp.empty_like(self.sdf_narrow_local_ends)

        self.sdf_linear_local_starts = wp.array(linear_starts, dtype=wp.vec3, device=device)
        self.sdf_linear_local_ends = wp.array(linear_ends, dtype=wp.vec3, device=device)
        self.sdf_linear_shapes = wp.array(linear_shapes, dtype=wp.int32, device=device)
        self.sdf_linear_world_starts = wp.empty_like(self.sdf_linear_local_starts)
        self.sdf_linear_world_ends = wp.empty_like(self.sdf_linear_local_ends)

    def _render_sdf_grid_visualization(self):
        """Render moving SDF bounds, coarse cells, and narrow-band subgrids."""
        line_sets = (
            (
                "/sdf/bounds",
                self.show_sdf_bounds,
                self.sdf_bounds_local_starts,
                self.sdf_bounds_local_ends,
                self.sdf_bounds_shapes,
                self.sdf_bounds_world_starts,
                self.sdf_bounds_world_ends,
                (1.0, 0.55, 0.05),
            ),
            (
                "/sdf/coarse",
                self.show_sdf_coarse,
                self.sdf_coarse_local_starts,
                self.sdf_coarse_local_ends,
                self.sdf_coarse_shapes,
                self.sdf_coarse_world_starts,
                self.sdf_coarse_world_ends,
                (0.45, 0.45, 0.45),
            ),
            (
                "/sdf/narrow_band",
                self.show_sdf_narrow_band,
                self.sdf_narrow_local_starts,
                self.sdf_narrow_local_ends,
                self.sdf_narrow_shapes,
                self.sdf_narrow_world_starts,
                self.sdf_narrow_world_ends,
                (0.0, 0.8, 1.0),
            ),
            (
                "/sdf/linear_cells",
                self.show_sdf_linear_cells,
                self.sdf_linear_local_starts,
                self.sdf_linear_local_ends,
                self.sdf_linear_shapes,
                self.sdf_linear_world_starts,
                self.sdf_linear_world_ends,
                (1.0, 0.9, 0.1),
            ),
        )
        for name, visible, local_starts, local_ends, shapes, world_starts, world_ends, color in line_sets:
            if not visible or len(local_starts) == 0:
                self.viewer.log_lines(name, None, None, None)
                continue
            wp.launch(
                transform_shape_local_lines,
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

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self._render_pressure_mesh_visualization()
        self._render_sdf_grid_visualization()
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.log_hydro_contact_surface(
            (
                self.collision_pipeline.hydroelastic_sdf.get_contact_surface()
                if self.collision_pipeline.hydroelastic_sdf is not None
                else None
            ),
            penetrating_only=True,
        )
        self.viewer.end_frame()

    def gui(self, imgui):
        """Render controls for the hydroelastic pressure visualization."""
        changed, self.show_pressure_surface = imgui.checkbox("Show mesh pressure", self.show_pressure_surface)
        if changed:
            self.viewer.show_hydro_contact_surface = self.show_pressure_surface
            self.viewer.show_visual = not self.show_pressure_surface

        pressure_max_mpa = SHAPE_CFG.kh * PRESSURE_COLORMAP_MAX_DEPTH / 1.0e6
        imgui.text(f"Pressure: blue = 0 MPa, red >= {pressure_max_mpa:g} MPa")
        imgui.text("Blue full mesh: zero pressure; colored patch: loaded region")
        _, self.show_sdf_bounds = imgui.checkbox("Show SDF bounds", self.show_sdf_bounds)
        _, self.show_sdf_coarse = imgui.checkbox("Show coarse grid", self.show_sdf_coarse)
        _, self.show_sdf_narrow_band = imgui.checkbox("Show narrow-band grid", self.show_sdf_narrow_band)
        _, self.show_sdf_linear_cells = imgui.checkbox(
            "Show narrow-band coarse fallback", self.show_sdf_linear_cells
        )
        imgui.text(f"Narrow-band lines: every {self.sdf_grid_line_stride} fine voxels")
        imgui.text("Orange: bounds, gray: coarse, cyan: fine, yellow: coarse fallback")
        for (
            label,
            coarse_resolution,
            fine_resolution,
            voxel_size,
            fine_count,
            linear_count,
            empty_count,
        ) in self.sdf_grid_info:
            imgui.text(
                f"{label}: coarse {coarse_resolution[0]}x{coarse_resolution[1]}x{coarse_resolution[2]}, "
                f"logical fine {fine_resolution[0]}x{fine_resolution[1]}x{fine_resolution[2]}"
            )
            imgui.text(
                f"voxel {1.0e3 * voxel_size[0]:.3f}x{1.0e3 * voxel_size[1]:.3f}x"
                f"{1.0e3 * voxel_size[2]:.3f} mm"
            )
            imgui.text(f"blocks: fine {fine_count}, coarse fallback {linear_count}, empty {empty_count}")

    def _init_test_tracking(self):
        """Initialize tracking data for test validation."""
        if not self.test_mode:
            self.bolt_body_indices = None
            self.nut_body_indices = None
            return

        # Find bolt and nut body indices by key
        self.bolt_body_indices = []
        self.nut_body_indices = []

        for i in range(self.grid_x):
            for j in range(self.grid_y):
                bolt_key = f"bolt_{i}_{j}"
                nut_key = f"nut_{i}_{j}"

                if bolt_key in self.model.body_label:
                    self.bolt_body_indices.append(self.model.body_label.index(bolt_key))
                if nut_key in self.model.body_label:
                    self.nut_body_indices.append(self.model.body_label.index(nut_key))

        # Store initial transforms
        body_q = self.state_0.body_q.numpy()
        self.bolt_initial_transforms = [body_q[idx].copy() for idx in self.bolt_body_indices]
        self.nut_initial_transforms = [body_q[idx].copy() for idx in self.nut_body_indices]

        # Track maximum rotation change and z displacement for nuts
        self.nut_max_rotation_change = [0.0] * len(self.nut_body_indices)
        self.nut_min_z = [body_q[idx][2] for idx in self.nut_body_indices]

    def _track_test_data(self):
        """Track transforms for test validation (called each step in test mode)."""
        if not self.test_mode:
            return

        body_q = self.state_0.body_q.numpy()

        # Track nut rotation and z position
        for i, nut_idx in enumerate(self.nut_body_indices):
            current_q = body_q[nut_idx]
            initial_q = self.nut_initial_transforms[i]

            # Compute rotation change using quaternion dot product
            # |q1 · q2| = cos(theta/2), where theta is the angle between orientations
            q_current = current_q[3:7]  # quaternion part (x, y, z, w)
            q_initial = initial_q[3:7]
            dot = abs(np.dot(q_current, q_initial))
            dot = min(dot, 1.0)  # Clamp for numerical stability
            rotation_angle = 2.0 * np.arccos(dot)
            self.nut_max_rotation_change[i] = max(self.nut_max_rotation_change[i], rotation_angle)

            # Track minimum z (nuts should move down)
            self.nut_min_z[i] = min(self.nut_min_z[i], current_q[2])

    def test_final(self):
        """Verify simulation state after example completes.

        - Bolts should stay approximately in place (limited displacement)
        - Nuts should rotate (thread engagement) and move slightly downward
        """
        body_q = self.state_0.body_q.numpy()

        # Check bolts stayed in place
        max_bolt_displacement = 0.02  # 2 cm tolerance
        for i, bolt_idx in enumerate(self.bolt_body_indices):
            current_pos = body_q[bolt_idx][:3]
            initial_pos = self.bolt_initial_transforms[i][:3]
            displacement = np.linalg.norm(current_pos - initial_pos)
            assert displacement < max_bolt_displacement, (
                f"Bolt {i}: displaced too much. "
                f"Displacement={displacement:.4f} (max allowed={max_bolt_displacement:.4f})"
            )

        # The 45 degree threshold catches stalled thread engagement while
        # allowing observed solver/contact-count variation.
        min_rotation_threshold = np.radians(45.0)
        min_descent = 0.005
        for i in range(len(self.nut_body_indices)):
            # Check rotation occurred
            max_rotation = self.nut_max_rotation_change[i]
            assert max_rotation > min_rotation_threshold, (
                f"Nut {i}: did not rotate enough. "
                f"Max rotation={np.degrees(max_rotation):.2f} degrees "
                f"(expected > {np.degrees(min_rotation_threshold):.2f} degrees)"
            )

            # Check nut moved downward (min_z should be less than initial z)
            initial_z = self.nut_initial_transforms[i][2]
            min_z = self.nut_min_z[i]
            descent = initial_z - min_z
            assert descent > min_descent, (
                f"Nut {i}: did not move down enough. Descent={descent:.4f} (expected > {min_descent:.4f})"
            )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=20)
        parser.add_argument(
            "--deterministic",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                "Make contact generation and ordering reproducible across runs on the same GPU. "
                "Costs a few percent of step time. Needs a world count small enough to keep the "
                "hydroelastic face-contact buffer under the 2^20 deterministic contact-id limit."
            ),
        )
        parser.add_argument(
            "--deterministic-solver",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                "Additionally make the solver bit-exact. Separate from --deterministic because "
                "it instruments every atomic scatter in the solver and costs ~7x step time, "
                "while contact ordering is what varies between runs."
            ),
        )
        parser.add_argument(
            "--solver",
            type=str,
            choices=["xpbd", "mujoco"],
            default="mujoco",
            help="Solver to use: 'xpbd' or 'mujoco'.",
        )
        parser.add_argument(
            "--num-per-world",
            type=int,
            default=1,
            help="Number of assemblies per world.",
        )
        parser.add_argument(
            "--sdf-grid-stride",
            type=int,
            default=SDF_GRID_LINE_STRIDE,
            help="Draw one line every N fine voxels inside allocated narrow-band blocks.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
