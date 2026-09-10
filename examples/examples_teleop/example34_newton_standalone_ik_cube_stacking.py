"""Standalone Newton Franka IK cube stacking for the Arena Docker image.

This is a self-contained copy of the official Newton cube-stacking logic,
adapted so it does not import ``newton.examples`` or
``newton.examples.ik.example_ik_cube_stacking`` at runtime.

Run inside the IsaacLab-Arena Docker container:

    cd /workspaces/isaaclab_arena
    /isaac-sim/python.sh examples/examples_teleop/example34_newton_standalone_ik_cube_stacking.py \
        --viewer gl --world-count 1 --num-frames 1800

Collision-only run, useful for fast headless tests:

    /isaac-sim/python.sh examples/examples_teleop/example34_newton_standalone_ik_cube_stacking.py \
        --viewer null --world-count 1 --num-frames 1800 --test --quiet --no-load-visuals

Fast no-window check:

    /isaac-sim/python.sh examples/examples_teleop/example34_newton_standalone_ik_cube_stacking.py \
        --viewer null --world-count 1 --num-frames 1800 --test --quiet
"""

from __future__ import annotations

import argparse
import ast
import enum
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.ik as ik
import newton.viewer


class TaskType(enum.IntEnum):
    APPROACH = 0
    REFINE_APPROACH = 1
    GRASP = 2
    LIFT = 3
    MOVE_TO_DROP_OFF = 4
    REFINE_DROP_OFF = 5
    RELEASE = 6
    RETRACT = 7
    HOME = 8


@wp.kernel(enable_backward=False)
def set_target_pose_kernel(
    task_schedule: wp.array[wp.int32],
    task_time_soft_limits: wp.array[float],
    task_object: wp.array[int],
    task_idx: wp.array[int],
    task_time_elapsed: wp.array[float],
    task_dt: float,
    task_offset_approach: wp.vec3,
    task_offset_lift: wp.vec3,
    task_offset_retract: wp.vec3,
    task_drop_off_pos: wp.vec3,
    cube_size: float,
    home_pos: wp.vec3,
    task_init_body_q: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    ee_index: int,
    robot_body_count: int,
    num_bodies_per_world: int,
    ee_pos_target: wp.array[wp.vec3],
    ee_pos_target_interpolated: wp.array[wp.vec3],
    ee_rot_target: wp.array[wp.vec4],
    ee_rot_target_interpolated: wp.array[wp.vec4],
    gripper_target: wp.array2d[wp.float32],
):
    tid = wp.tid()

    idx = task_idx[tid]
    task = task_schedule[idx]
    task_time_soft_limit = task_time_soft_limits[idx]
    cube_body_index = task_object[idx]
    cube_index = cube_body_index - robot_body_count

    task_time_elapsed[tid] += task_dt
    t = wp.min(1.0, task_time_elapsed[tid] / task_time_soft_limit)

    ee_body_id = tid * num_bodies_per_world + ee_index
    ee_pos_prev = wp.transform_get_translation(task_init_body_q[ee_body_id])
    ee_quat_prev = wp.transform_get_rotation(task_init_body_q[ee_body_id])
    ee_quat_target = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi)

    obj_body_id = tid * num_bodies_per_world + cube_body_index
    obj_pos_current = wp.transform_get_translation(body_q[obj_body_id])
    obj_quat_current = wp.transform_get_rotation(body_q[obj_body_id])
    cube_offset = wp.float(cube_index) * cube_size * wp.vec3(0.0, 0.0, 1.0)

    t_gripper = 0.0

    if task == TaskType.APPROACH.value:
        ee_pos_target[tid] = obj_pos_current + task_offset_approach
        ee_quat_target = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi) * wp.quat_inverse(obj_quat_current)
    elif task == TaskType.REFINE_APPROACH.value:
        ee_pos_target[tid] = obj_pos_current
        ee_quat_target = ee_quat_prev
    elif task == TaskType.GRASP.value:
        ee_pos_target[tid] = ee_pos_prev
        ee_quat_target = ee_quat_prev
        t_gripper = t
    elif task == TaskType.LIFT.value:
        ee_pos_target[tid] = ee_pos_prev + task_offset_lift
        ee_quat_target = ee_quat_prev
        t_gripper = 1.0
    elif task == TaskType.MOVE_TO_DROP_OFF.value:
        ee_pos_target[tid] = task_drop_off_pos + cube_offset + task_offset_approach
        t_gripper = 1.0
    elif task == TaskType.REFINE_DROP_OFF.value:
        ee_pos_target[tid] = task_drop_off_pos + cube_offset
        t_gripper = 1.0
    elif task == TaskType.RELEASE.value:
        ee_pos_target[tid] = task_drop_off_pos + cube_offset
        t_gripper = 1.0 - t
    elif task == TaskType.RETRACT.value:
        ee_pos_target[tid] = task_drop_off_pos + cube_offset + task_offset_retract
    elif task == TaskType.HOME.value:
        ee_pos_target[tid] = home_pos
    else:
        ee_pos_target[tid] = home_pos

    ee_pos_target_interpolated[tid] = ee_pos_prev * (1.0 - t) + ee_pos_target[tid] * t
    ee_quat_interpolated = wp.quat_slerp(ee_quat_prev, ee_quat_target, t)

    ee_rot_target[tid] = ee_quat_target[:4]
    ee_rot_target_interpolated[tid] = ee_quat_interpolated[:4]

    gripper_pos = 0.06 * (1.0 - t_gripper)
    gripper_target[tid, 0] = gripper_pos
    gripper_target[tid, 1] = gripper_pos


@wp.kernel(enable_backward=False)
def advance_task_kernel(
    task_time_soft_limits: wp.array[float],
    ee_pos_target: wp.array[wp.vec3],
    ee_rot_target: wp.array[wp.vec4],
    body_q: wp.array[wp.transform],
    num_bodies_per_world: int,
    ee_index: int,
    task_idx: wp.array[int],
    task_time_elapsed: wp.array[float],
    task_init_body_q: wp.array[wp.transform],
):
    tid = wp.tid()
    idx = task_idx[tid]
    task_time_soft_limit = task_time_soft_limits[idx]

    ee_body_id = tid * num_bodies_per_world + ee_index
    ee_pos_current = wp.transform_get_translation(body_q[ee_body_id])
    ee_quat_current = wp.transform_get_rotation(body_q[ee_body_id])

    pos_err = wp.length(ee_pos_target[tid] - ee_pos_current)
    ee_quat_target = wp.quaternion(ee_rot_target[tid][:3], ee_rot_target[tid][3])
    quat_rel = ee_quat_current * wp.quat_inverse(ee_quat_target)
    rot_err = wp.abs(wp.degrees(2.0 * wp.atan2(wp.length(quat_rel[:3]), quat_rel[3])))

    if (
        task_time_elapsed[tid] >= task_time_soft_limit
        and pos_err < 0.001
        and rot_err < 0.5
        and task_idx[tid] < wp.len(task_time_soft_limits) - 1
    ):
        task_idx[tid] += 1
        task_time_elapsed[tid] = 0.0

        body_id_start = tid * num_bodies_per_world
        for i in range(num_bodies_per_world):
            body_id = body_id_start + i
            task_init_body_q[body_id] = body_q[body_id]


def make_collision_only_urdf(urdf_path: Path) -> Path:
    cache_dir = Path.cwd() / ".tmp_newton_examples"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / f"franka_collision_only_example34_{os.getuid()}_{os.getpid()}.urdf"
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for link in root.findall("link"):
        for visual in list(link.findall("visual")):
            link.remove(visual)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


class StandaloneCubeStackingExample:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.collide_substeps = False
        self.world_count = args.world_count
        self.headless = args.headless
        self.verbose = args.verbose
        self.load_visuals = args.load_visuals
        self.viewer = viewer

        self.cube_count = 3
        self.cube_size = 0.05

        self.table_height = 0.1
        self.table_pos = wp.vec3(0.0, -0.5, 0.5 * self.table_height)
        self.table_top_center = self.table_pos + wp.vec3(0.0, 0.0, 0.5 * self.table_height)
        self.robot_base_pos = self.table_top_center + wp.vec3(-0.5, 0.0, 0.0)

        self.task_offset_approach = wp.vec3(0.0, 0.0, 1.0 * self.cube_size)
        self.task_offset_lift = wp.vec3(0.0, 0.0, 4.0 * self.cube_size)
        self.task_offset_retract = wp.vec3(0.0, 0.0, 2.0 * self.cube_size)
        self.task_drop_off_pos = self.table_top_center + wp.vec3(0.0, -0.15, 0.5 * self.cube_size)

        self.use_mujoco_contacts = args.use_mujoco_contacts
        franka_with_table = self.build_franka_with_table()
        scene = self.build_scene(franka_with_table)
        self.robot_body_count = franka_with_table.body_count

        self.model_single = franka_with_table.finalize()
        self.model = scene.finalize()
        self.num_bodies_per_world = self.model.body_count // self.world_count

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            iterations=20,
            ls_iterations=100,
            nconmax=1000,
            njmax=2000,
            cone="elliptic",
            impratio=1000.0,
            use_mujoco_contacts=self.use_mujoco_contacts,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        wp.copy(self.control.joint_target_pos[:9], self.model.joint_q[:9])

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.contacts = self.model.contacts()

        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.setup_ik()
        self.setup_tasks()

        if self.headless:
            self.viewer = newton.viewer.ViewerNull(num_frames=args.num_frames)

        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = False
        if hasattr(self.viewer, "renderer"):
            self.viewer.set_world_offsets(wp.vec3(1.5, 1.5, 0.0))

        self.capture()
        self.episode_steps = 0

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

        self.graph_ik = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)
            self.graph_ik = capture.graph

    def simulate(self):
        if not self.collide_substeps:
            self.model.collide(self.state_0, self.contacts)

        for _ in range(self.sim_substeps):
            if self.collide_substeps:
                self.model.collide(self.state_0, self.contacts)

            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.episode_steps == 1:
            self.start_time = time.perf_counter()

        self.set_joint_targets()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        if self.episode_steps > 1:
            self.sim_time += self.frame_dt

        if self.verbose and self.episode_steps > 0:
            tock = time.perf_counter()
            print(f"Step {self.episode_steps} time: {tock - self.start_time:.2f}, sim time: {self.sim_time:.2f}")
            print(f"RT factor: {self.world_count * self.sim_time / (tock - self.start_time):.2f}")
            print("_" * 100)

        self.episode_steps += 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def build_franka_with_table(self):
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        franka_urdf = newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"
        urdf_path = franka_urdf if self.load_visuals else make_collision_only_urdf(franka_urdf)
        builder.add_urdf(
            urdf_path,
            xform=wp.transform(self.robot_base_pos, wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            hide_visuals=not self.load_visuals,
            parse_visuals_as_colliders=False,
        )

        builder.joint_q[:9] = [
            -3.6802115e-03,
            2.3901723e-02,
            3.6804110e-03,
            -2.3683236e00,
            -1.2918962e-04,
            2.3922248e00,
            7.8549200e-01,
            0.05,
            0.05,
        ]
        builder.joint_target_pos[:9] = [
            -3.6802115e-03,
            2.3901723e-02,
            3.6804110e-03,
            -2.3683236e00,
            -1.2918962e-04,
            2.3922248e00,
            7.8549200e-01,
            1.0,
            1.0,
        ]
        builder.joint_target_ke[:9] = [4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]
        builder.joint_target_kd[:9] = [450, 450, 350, 350, 200, 200, 200, 10, 10]
        builder.joint_effort_limit[:9] = [87, 87, 87, 87, 12, 12, 12, 100, 100]
        builder.joint_armature[:9] = [0.3] * 4 + [0.11] * 3 + [0.15] * 2

        gravcomp_attr = builder.custom_attributes["mujoco:jnt_actgravcomp"]
        if gravcomp_attr.values is None:
            gravcomp_attr.values = {}
        for dof_idx in range(7):
            gravcomp_attr.values[dof_idx] = True

        gravcomp_body = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp_body.values is None:
            gravcomp_body.values = {}
        for body_idx in range(2, 14):
            gravcomp_body.values[body_idx] = 1.0

        shape_cfg = newton.ModelBuilder.ShapeConfig(margin=0.0, density=1000.0)
        shape_cfg.ke = 5.0e4
        shape_cfg.kd = 5.0e2
        shape_cfg.kf = 1.0e3
        shape_cfg.mu = 0.75

        builder.add_shape_box(
            body=-1,
            hx=0.4,
            hy=0.4,
            hz=0.5 * self.table_height,
            xform=wp.transform(self.table_pos, wp.quat_identity()),
            cfg=shape_cfg,
        )

        if self.use_mujoco_contacts:
            condim_attr = builder.custom_attributes["mujoco:condim"]
            if condim_attr.values is None:
                condim_attr.values = {}
            for shape_idx in range(builder.shape_count):
                if builder.shape_body[shape_idx] in (12, 13):
                    condim_attr.values[shape_idx] = 4

        return builder

    def build_scene(self, franka_with_table: newton.ModelBuilder):
        rng = np.random.default_rng(42)
        density_range = [300.0, 500.0]
        x_range = [-0.1, 0.1]
        y_range = [-0.1, 0.1]
        theta_range = [-0.9 * np.pi, 0.9 * np.pi]

        default_cube_offset = wp.transform(wp.vec3(0.0, 0.15, 0.5 * self.cube_size))
        sampling_region_origin = wp.transform(self.table_top_center) * default_cube_offset
        min_distance = np.sqrt(2) * self.cube_size + 0.04

        scene = newton.ModelBuilder()
        for world_id in range(self.world_count):
            scene.begin_world()
            scene.add_builder(franka_with_table)
            self.add_cubes(
                scene,
                world_id,
                density_range,
                x_range,
                y_range,
                theta_range,
                sampling_region_origin,
                min_distance,
                rng,
            )
            scene.end_world()

        scene.add_ground_plane()
        return scene

    def add_cubes(
        self,
        scene: newton.ModelBuilder,
        world_id: int,
        density_range: list[float],
        x_range: list[float],
        y_range: list[float],
        theta_range: list[float],
        sampling_region_origin: wp.transform,
        min_distance: float,
        rng: np.random.Generator,
    ):
        density = rng.uniform(density_range[0], density_range[1])
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=density, margin=0.0)

        def get_random_pos():
            random_x = rng.uniform(x_range[0], x_range[1])
            random_y = rng.uniform(y_range[0], y_range[1])
            return wp.vec3(random_x, random_y, 0.0)

        cube_pos = []
        for i in range(self.cube_count):
            key = f"world_{world_id}/cube_{i}"
            new_pos = get_random_pos()
            if len(cube_pos) > 0:
                l2_dists_too_close = [wp.norm_l2(new_pos - pos) < min_distance for pos in cube_pos]
                attempts = 0
                while any(l2_dists_too_close):
                    new_pos = get_random_pos()
                    l2_dists_too_close = [wp.norm_l2(new_pos - pos) < min_distance for pos in cube_pos]
                    attempts += 1
                    if attempts >= 1000:
                        raise RuntimeError(f"Failed to place cube {i} after 1000 attempts")

            cube_pos.append(new_pos)
            random_theta = rng.uniform(theta_range[0], theta_range[1])
            random_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), random_theta)
            body_xform = sampling_region_origin * wp.transform(cube_pos[-1], random_rot)
            mesh_body = scene.add_body(xform=body_xform)

            half_size = 0.5 * self.cube_size
            cube_shape_idx = scene.shape_count
            cube_color = [[0.8, 0.2, 0.2], [0.2, 0.8, 0.2], [0.2, 0.2, 0.8]][i]

            scene.add_shape_box(
                body=mesh_body,
                hx=half_size,
                hy=half_size,
                hz=half_size,
                cfg=shape_cfg,
                label=key,
                color=cube_color,
            )

            if self.use_mujoco_contacts:
                condim_attr = scene.custom_attributes["mujoco:condim"]
                if condim_attr.values is None:
                    condim_attr.values = {}
                condim_attr.values[cube_shape_idx] = 4

    def setup_ik(self):
        self.ee_index = 11
        body_q_np = self.state.body_q.numpy()
        self.ee_tf = wp.transform(*body_q_np[self.ee_index])
        self.home_pos = wp.vec3(body_q_np[self.ee_index][:3])

        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([self.home_pos] * self.world_count, dtype=wp.vec3),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.transform_get_rotation(self.ee_tf)[:4]] * self.world_count, dtype=wp.vec4),
        )

        ik_dofs = self.model_single.joint_coord_count
        self.joint_limit_lower = wp.clone(self.model.joint_limit_lower.reshape((self.world_count, -1))[:, :ik_dofs])
        self.joint_limit_upper = wp.clone(self.model.joint_limit_upper.reshape((self.world_count, -1))[:, :ik_dofs])
        self.obj_joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.joint_limit_lower.flatten(),
            joint_limit_upper=self.joint_limit_upper.flatten(),
        )
        self.joint_q_ik = wp.clone(self.model.joint_q.reshape((self.world_count, -1))[:, :ik_dofs])

        self.ik_iters = 24
        self.ik_solver = ik.IKSolver(
            model=self.model_single,
            n_problems=self.world_count,
            objectives=[self.pos_obj, self.rot_obj, self.obj_joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def setup_tasks(self):
        task_per_object = 9
        task_schedule = []
        for _ in range(self.cube_count):
            task_schedule.extend(
                [
                    TaskType.APPROACH,
                    TaskType.REFINE_APPROACH,
                    TaskType.GRASP,
                    TaskType.LIFT,
                    TaskType.MOVE_TO_DROP_OFF,
                    TaskType.REFINE_DROP_OFF,
                    TaskType.RELEASE,
                    TaskType.RETRACT,
                    TaskType.HOME,
                ]
            )

        self.task_counter = len(task_schedule)
        self.task_schedule = wp.array(task_schedule, shape=(self.task_counter), dtype=wp.int32)
        self.task_time_soft_limits = wp.array([1.0] * self.task_counter, dtype=float)

        task_object = []
        for i in range(self.cube_count):
            task_object.extend([self.robot_body_count + i] * task_per_object)
        self.task_object = wp.array(task_object, shape=(self.task_counter), dtype=wp.int32)

        self.task_init_body_q = wp.clone(self.state_0.body_q)
        self.task_idx = wp.zeros(self.world_count, dtype=wp.int32)
        self.task_dt = self.frame_dt
        self.task_time_elapsed = wp.zeros(self.world_count, dtype=wp.float32)

        self.ee_pos_target = wp.zeros(self.world_count, dtype=wp.vec3)
        self.ee_pos_target_interpolated = wp.zeros(self.world_count, dtype=wp.vec3)
        self.ee_rot_target = wp.zeros(self.world_count, dtype=wp.vec4)
        self.ee_rot_target_interpolated = wp.zeros(self.world_count, dtype=wp.vec4)
        self.gripper_target_interpolated = wp.zeros(shape=(self.world_count, 2), dtype=wp.float32)

    def set_joint_targets(self):
        wp.launch(
            set_target_pose_kernel,
            dim=self.world_count,
            inputs=[
                self.task_schedule,
                self.task_time_soft_limits,
                self.task_object,
                self.task_idx,
                self.task_time_elapsed,
                self.task_dt,
                self.task_offset_approach,
                self.task_offset_lift,
                self.task_offset_retract,
                self.task_drop_off_pos,
                self.cube_size,
                self.home_pos,
                self.task_init_body_q,
                self.state_0.body_q,
                self.ee_index,
                self.robot_body_count,
                self.num_bodies_per_world,
            ],
            outputs=[
                self.ee_pos_target,
                self.ee_pos_target_interpolated,
                self.ee_rot_target,
                self.ee_rot_target_interpolated,
                self.gripper_target_interpolated,
            ],
        )

        self.pos_obj.set_target_positions(self.ee_pos_target_interpolated)
        self.rot_obj.set_target_rotations(self.ee_rot_target_interpolated)

        if self.graph_ik is not None:
            wp.capture_launch(self.graph_ik)
        else:
            self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)

        joint_target_pos_view = self.control.joint_target_pos.reshape((self.world_count, -1))
        wp.copy(dest=joint_target_pos_view[:, :7], src=self.joint_q_ik[:, :7])
        wp.copy(dest=joint_target_pos_view[:, 7:9], src=self.gripper_target_interpolated[:, :2])

        wp.launch(
            advance_task_kernel,
            dim=self.world_count,
            inputs=[
                self.task_time_soft_limits,
                self.ee_pos_target,
                self.ee_rot_target,
                self.state_0.body_q,
                self.num_bodies_per_world,
                self.ee_index,
            ],
            outputs=[
                self.task_idx,
                self.task_time_elapsed,
                self.task_init_body_q,
            ],
        )

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        world_success = [True] * self.world_count
        target_rot_inv = wp.quat_inverse(wp.quat_identity())

        for world_id in range(self.world_count):
            for cube_id in range(self.cube_count):
                drop_off_pos = np.array(self.task_drop_off_pos) + np.array([0.0, 0.0, self.cube_size * cube_id])
                cube_body_id = world_id * self.num_bodies_per_world + self.robot_body_count + cube_id
                cube_pos = body_q[cube_body_id][:3]
                cube_rot = body_q[cube_body_id][3:]

                pos_error = cube_pos - drop_off_pos
                pos_error_xy = np.linalg.norm(pos_error[:2])
                pos_error_z = np.abs(pos_error[2])
                if np.isnan(pos_error_xy) or np.isnan(pos_error_z) or pos_error_xy > 0.02 or pos_error_z > 0.01:
                    world_success[world_id] = False
                    break

                quat_rel = wp.quat(cube_rot) * target_rot_inv
                quat_rel_np = np.array(quat_rel)
                rot_err = np.abs(np.degrees(2.0 * np.arctan2(np.linalg.norm(quat_rel_np[:3]), quat_rel_np[3])))
                if np.isnan(rot_err) or rot_err > 5.0:
                    world_success[world_id] = False
                    break

        success_rate = np.mean(world_success)
        if success_rate < 0.7:
            raise ValueError(f"World success rate is {success_rate}, expected 0.7 or higher")
        print(f"World success rate: {success_rate}")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--viewer", choices=["gl", "usd", "null", "viser"], default="gl", help="Viewer backend.")
    parser.add_argument("--output-path", type=str, default="output.usd", help="USD output path for --viewer usd.")
    parser.add_argument("--num-frames", type=int, default=1800, help="Total number of frames.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False, help="Headless GL viewer.")
    parser.add_argument("--test", action=argparse.BooleanOptionalAction, default=False, help="Run final success check.")
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=False, help="Suppress Warp messages.")
    parser.add_argument("--warp-config", action="append", default=[], metavar="KEY=VALUE", help="Override warp.config.")
    parser.add_argument("--world-count", type=int, default=1, help="Number of simulation worlds.")
    parser.add_argument(
        "--use-mujoco-contacts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use MuJoCo native contacts instead of Newton contacts.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument(
        "--load-visuals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load Franka URDF visual meshes. Requires pycollada in the Isaac Python environment.",
    )
    return parser


def init_viewer(args):
    for entry in args.warp_config:
        if "=" not in entry:
            raise ValueError(f"Invalid --warp-config format {entry!r}; expected KEY=VALUE")
        key, value_str = entry.split("=", 1)
        if not hasattr(wp.config, key):
            raise ValueError(f"Invalid --warp-config key {key!r}")
        try:
            value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            value = value_str
        setattr(wp.config, key, value)

    if args.quiet:
        wp.config.quiet = True
    if args.device:
        wp.set_device(args.device)

    if args.viewer == "gl":
        return newton.viewer.ViewerGL(headless=args.headless)
    if args.viewer == "usd":
        return newton.viewer.ViewerUSD(output_path=args.output_path, num_frames=args.num_frames)
    if args.viewer == "null":
        return newton.viewer.ViewerNull(num_frames=args.num_frames)
    if args.viewer == "viser":
        return newton.viewer.ViewerViser()
    raise ValueError(f"Invalid viewer: {args.viewer}")


def run(example, args):
    while example.viewer.is_running():
        if example.viewer.should_step():
            example.step()
        example.render()

    if args.test:
        example.test_final()

    example.viewer.close()


if __name__ == "__main__":
    parsed_args = create_parser().parse_args()
    parsed_viewer = init_viewer(parsed_args)
    run(StandaloneCubeStackingExample(parsed_viewer, parsed_args), parsed_args)
