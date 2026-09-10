"""ABB + CAD-derived three-finger continuum research gripper, contact-only grasp.

MuJoCo integrates the six-axis arm with bounded drives. VBD integrates ALL
finger material points and the free rigid bottle, with two-way contact. Only
massless mounting anchors track the measured wrist; finite root springs send
their opposite wrench back to the arm with one-substep partitioning latency.
No object pose writes, fixed bottle joint, rigid finger collider or grasp assist.

The IGES outer-envelope FEM and active reference curvature are research
approximations, NOT a calibrated SRT silicone/chamber/pressure model.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
from example_newton_viewer_utils import arena_root, init_viewer
from example39_newton_standalone_abb_irb1200_pgc140_cube_stacking import PGC140CubeStackingExample, named_index

ASSETS = arena_root() / "examples/examples_teleop/assets/abb_irb1200_sfg_n4049"


@wp.kernel
def move_anchors(
    ids: wp.array[int], local: wp.array[wp.vec3], body_q: wp.array[wp.transform],
    wrist: int, dt: float, q: wp.array[wp.vec3], qd: wp.array[wp.vec3],
):
    i = wp.tid()
    target = wp.transform_point(body_q[wrist], local[i])
    p = ids[i]
    qd[p] = (target - q[p]) / dt
    q[p] = target


@wp.kernel
def root_reaction(
    roots: wp.array[int], anchors: wp.array[int], q: wp.array[wp.vec3],
    body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3],
    wrist: int, stiffness: float, wrench: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    # Opposite of VBD's zero-rest-length, undamped root spring force.
    force = stiffness * (q[roots[i]] - q[anchors[i]])
    com = wp.transform_point(body_q[wrist], body_com[wrist])
    torque = wp.cross(q[anchors[i]] - com, force)
    wp.atomic_add(wrench, wrist, wp.spatial_vector(force, torque))


class SFGSoftGraspExample(PGC140CubeStackingExample):
    """Reuse only ABB IK helpers; the gripper/task/solver are independent."""

    def __init__(self, viewer, args):
        assert args.mass > 0 and 0 <= args.activation <= 1 and args.substeps > 0
        assert args.mu >= 0 and args.iterations > 0 and args.shear > 0
        assert args.root_ke > 0 and args.contact_ke > 0 and 0 <= args.curvature <= 30
        assert args.num_frames > 0 and args.ik_iters > 0
        assert args.no_graph or not wp.get_device().is_cuda or args.substeps % 2 == 0, "CUDA graph requires even substeps for ping-pong buffers"
        assert not args.expect_grasp_failure or args.activation <= .1, "Use low actuation for the negative control"
        assert args.snapshot_dir is None or args.viewer == "gl", "Snapshots require the GL viewer"
        self.args, self.viewer = args, viewer
        self.frame_dt = 1 / 60
        self.sim_dt = self.frame_dt / args.substeps
        self.sim_time, self.episode_steps = 0., 0
        self.home_rot = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), math.pi / 2)
        self.mount_rot = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), math.pi / 2)
        self.tcp_offset = wp.vec3(.125, 0, 0)
        self.pick = np.array([.14 + args.offset_x, .07 + args.offset_y, .16])
        self.drop = self.pick + [0, -.14, 0]
        self.grasp_tcp = self.pick + [0, 0, .016]
        self.initial_tcp = self.grasp_tcp + [0, 0, .16]
        b = self.build_arm()
        self.robot_model = b.finalize()
        self.setup_ik()
        self.solve_ik(self.initial_tcp, 160)
        initial_q = self.joint_q_ik.numpy()[0]
        b.joint_q[:] = initial_q.tolist()
        b.joint_target_pos[:] = initial_q.tolist()
        self.arm_model = b.finalize()
        self.arm_0, self.arm_1 = self.arm_model.state(), self.arm_model.state()
        self.arm_control = self.arm_model.control()
        newton.eval_fk(self.arm_model, self.arm_model.joint_q, self.arm_model.joint_qd, self.arm_0)
        assert np.linalg.norm(self.tcp_position(self.arm_0) - self.initial_tcp) < .002, "Initial IK failed"
        self.arm_solver = newton.solvers.SolverMuJoCo(
            self.arm_model, use_mujoco_contacts=True, solver="newton", integrator="implicitfast",
            iterations=30, ls_iterations=20, nconmax=128, njmax=512,
        )
        self.arm_contacts = self.arm_model.contacts()
        self.arm_count = b.body_count
        # VBD receives the *dynamic* arm's actual poses, not IK targets. It does
        # not integrate the arm again. The bottle remains DYNAMIC in this model.
        for i in range(b.body_count):
            b.body_flags[i] = int(newton.BodyFlags.KINEMATIC)
        b.joint_enabled[:] = [False] * b.joint_count
        b.joint_target_ke[:] = [0.] * b.joint_dof_count
        b.joint_target_kd[:] = [0.] * b.joint_dof_count
        wrist_tf = wp.transform(*self.arm_0.body_q.numpy()[self.ee_index])
        self.initial_mount = wp.transform_multiply(wrist_tf, wp.transform(wp.vec3(0), self.mount_rot))
        self.initial_rotation = np.array(wp.quat_to_matrix(wp.transform_get_rotation(self.initial_mount))).reshape(3, 3)
        self.add_fingers(b)
        self.add_bottle(b)
        assert self.bottle >= self.arm_count, "Never include bottle in arm pose-copy range"
        b.color()
        self.model = b.finalize()
        self.model.soft_contact_ke = args.contact_ke
        self.model.soft_contact_kd = 1e-6
        self.model.soft_contact_mu = args.mu
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.control = self.model.control()
        self.pipeline = newton.CollisionPipeline(self.model, soft_contact_margin=.006)
        self.contacts = self.pipeline.contacts()
        self.solver = newton.solvers.SolverVBD(
            self.model, iterations=args.iterations, friction_epsilon=.001,
            particle_enable_self_contact=False, rigid_body_particle_contact_buffer_size=4096,
            rigid_body_contact_buffer_size=128,
        )
        self.reaction = wp.zeros(self.arm_count, dtype=wp.spatial_vector)
        self.root_ids = wp.array(self.roots, dtype=int)
        self.anchor_ids = wp.array(self.anchors, dtype=int)
        self.anchor_local = wp.array(self.anchor_local_np, dtype=wp.vec3)
        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = False
        self.viewer.set_camera(wp.vec3(1.0, 1.1, .85), -24., -140.)
        self.sequence = self.make_sequence()
        self.phase_index, self.phase_elapsed = 0, 0.
        self.phase_start = self.initial_tcp.copy()
        self.phase_activation = 0.
        self.command_tcp = self.initial_tcp.copy()
        self.finished = False
        self.max_lift, self.airborne_transport, self.max_tracking_error = 0., 0., 0.
        self.max_root_error, self.max_reaction, self.max_speed, self.max_tip_bend = 0., 0., 0., 0.
        self.min_tet_volume_ratio = 1.
        self.tet_ids_np = self.model.tet_indices.numpy()
        self.reference_volumes = 1. / np.linalg.det(self.model.tet_poses.numpy())
        self.hold_min_lift = float("inf")
        self.hold_frames, self.contact_frames = 0, 0
        self.history = []
        self.graph = None
        # Warm kernels/buffers without advancing either state.
        if wp.get_device().is_cuda and not args.no_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        print(f"[MODEL] {self.model.tet_count} tets; {self.finger_particle_count} dynamic finger points; "
              f"{len(self.roots)} finite root springs; bottle={self.model.body_mass.numpy()[self.bottle]:.6f}kg", flush=True)
        print("[BOUNDARY] CAD-envelope + assumed active curvature; not factory pneumatic calibration; "
              "MuJoCo arm <-> root springs (1-substep lag), VBD fingers <-> free bottle", flush=True)

    def build_arm(self):
        b = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        stage = Usd.Stage.Open(str(self.args.robot_usd))
        assert stage, "Run tools/convert_sfg_n4049_igs_to_abb_usd.py first"
        ignored = [str(p.GetPath()) for p in stage.Traverse()
                   if p.GetAttribute("srt:deformableVisual").Get() or p.GetName() == "FEM"]
        assert len(ignored) == 4, ignored
        mount = next(p for p in stage.Traverse() if p.GetName() == "sfg_mount")
        mount_matrix = UsdGeom.Xformable(mount).GetLocalTransformation()
        axis = np.asarray(mount_matrix.TransformDir((0, 0, 1)))
        np.testing.assert_allclose(axis, [1, 0, 0], atol=1e-6)
        b.add_usd(stage, xform=wp.transform(wp.vec3(-.30, 0, .1), wp.quat_identity()),
                  floating=False, enable_self_collisions=False, collapse_fixed_joints=False,
                  ignore_paths=ignored, load_visual_shapes=True)
        assert b.joint_dof_count == 6
        self.ee_index = named_index(b.body_label, "link_6")
        b.joint_q[:6] = [0, -.55, .95, 0, .85, 0]
        b.joint_target_mode[:6] = [newton.JointTargetMode.POSITION] * 6
        b.joint_target_ke[:6] = [3500, 3500, 3000, 1800, 1600, 1200]
        b.joint_target_kd[:6] = [350, 350, 300, 180, 160, 120]
        b.joint_effort_limit[:6] = [300.] * 6
        b.joint_armature[:6] = [.05] * 6
        # Assumed 0.43kg rigid housing lumped at the wrist. Dynamic soft-finger
        # weight is transmitted through spring reactions, not double-counted.
        b.body_mass[self.ee_index] += .43
        b.body_inertia[self.ee_index] += wp.mat33(.0008, 0, 0, 0, .002, 0, 0, 0, .002)
        b.custom_attributes["mujoco:gravcomp"].values = {i: 1. for i in range(b.body_count)}
        return b

    def add_fingers(self, b):
        self.local_points, self.local_tets, self.ranges, self.roots = [], [], [], []
        data = np.load(self.args.robot_usd.resolve().parent / "soft_fingers.npz")
        root_local = []
        self.tip_ids = []
        for i in range(3):
            p, t, roots = data[f"points_{i}"], data[f"tets_{i}"], data[f"roots_{i}"]
            start = b.particle_count
            self.local_points.append(p.copy())
            self.local_tets.append(t.copy())
            self.ranges.append((start, start + len(p)))
            self.tip_ids.extend((start + np.flatnonzero(p[:, 2] > p[:, 2].max() - .0041)).tolist())
            b.add_soft_mesh(pos=wp.transform_get_translation(self.initial_mount),
                            rot=wp.transform_get_rotation(self.initial_mount), scale=1., vel=wp.vec3(0),
                            mesh=newton.TetMesh(p, t), density=1050., k_mu=self.args.shear,
                            k_lambda=self.args.shear * 10, k_damp=1e-4, particle_radius=.002,
                            add_surface_mesh_edges=False)
            self.roots.extend((start + roots).tolist())
            root_local.extend([np.array(wp.quat_rotate(self.mount_rot, wp.vec3(*x))) for x in p[roots]])
        self.finger_particle_count = b.particle_count
        self.anchor_local_np = np.asarray(root_local)
        self.anchors = []
        for root in self.roots:
            anchor = b.add_particle(b.particle_q[root], wp.vec3(0), mass=0., radius=0., flags=0)
            self.anchors.append(anchor)
            b.add_spring(root, anchor, ke=self.args.root_ke, kd=0., control=0.)
        self.tip_ids = np.asarray(self.tip_ids)

    def add_bottle(self, b):
        cfg = newton.ModelBuilder.ShapeConfig(mu=self.args.mu, ke=self.args.contact_ke, kd=1e-6, margin=0., gap=0.)
        b.add_shape_box(-1, wp.transform(wp.vec3(.0, .0, .05), wp.quat_identity()),
                        hx=.60, hy=.4, hz=.05, cfg=cfg, label="table", color=(.38, .41, .44))
        self.bottle = b.add_body(wp.transform(wp.vec3(*self.pick), wp.quat_identity()), label="bottle_1kg")
        radius, half_height = .032, .06
        cfg.density = self.args.mass / (math.pi * radius**2 * half_height * 2)
        self.bottle_shape = b.add_shape_cylinder(self.bottle, radius=radius, half_height=half_height,
                                                cfg=cfg, label="bottle_body", color=(.90, .68, .15))
        cap = newton.ModelBuilder.ShapeConfig(density=0, mu=self.args.mu, ke=self.args.contact_ke, kd=1e-6, margin=0., gap=0.)
        b.add_shape_cylinder(self.bottle, wp.transform(wp.vec3(0, 0, .067), wp.quat_identity()),
                             radius=.013, half_height=.007, cfg=cap, label="bottle_cap", color=(.15, .20, .23))
        b.add_ground_plane()

    def make_sequence(self):
        g = self.grasp_tcp
        drop = self.drop + [0, 0, .016]
        high = np.array([0, 0, .14])
        return [
            ("approach", 1., self.initial_tcp, 0.),
            ("descend", 2., g, 0.),
            ("close", 2., g, self.args.activation),
            ("lift", 2., g + high, self.args.activation),
            ("hold", 1.5, g + high, self.args.activation),
            ("transport", 2., drop + high, self.args.activation),
            ("place", 2., drop, self.args.activation),
            ("release", 1.5, drop, 0.),
            ("retract", 2., drop + high, 0.),
            ("settle", 1., drop + high, 0.),
        ]

    def update_active_reference(self, activation):
        """Stress-free active curvature: changes elastic energy, NEVER state positions."""
        # VBD ignores tet_activations; it DOES read tet_poses each iteration.
        # Keep the material reference orientation fixed: isotropic FEM energy
        # is objective under rigid rotation. Do not rotate the reference with
        # the live wrist or infer a pneumatic pressure from this input.
        matrices = []
        curvature = self.args.curvature * activation
        for p, t in zip(self.local_points, self.local_tets):
            deformed = p.copy().astype(np.float64)
            root_z = float(p[:, 2].min()) + .008
            radial = p[p[:, 2] <= root_z, :2].mean(0)
            unit = radial / np.linalg.norm(radial)
            s = np.maximum(0., p[:, 2] - root_z)
            offset = (p[:, :2] - radial) @ unit
            if abs(curvature) > 1e-8:
                angle = curvature * s
                shift = -(1 - np.cos(angle)) / curvature + offset * (np.cos(angle) - 1)
                deformed[:, :2] += shift[:, None] * unit
                deformed[:, 2] += np.sin(angle) / curvature - s + offset * np.sin(angle)
            xyz = (deformed @ self.initial_rotation.T)[t]
            dm = np.stack([xyz[:, j] - xyz[:, 0] for j in (1, 2, 3)], axis=-1)
            assert np.linalg.det(dm).min() > 1e-12, "Active rest shape inverted; reduce curvature"
            matrices.append(np.linalg.inv(dm))
        self.model.tet_poses.assign(np.concatenate(matrices).astype(np.float32))

    def simulate(self):
        for _ in range(self.args.substeps):
            self.arm_0.clear_forces()
            wp.copy(self.arm_0.body_f, self.reaction)
            self.arm_solver.step(self.arm_0, self.arm_1, self.arm_control, self.arm_contacts, self.sim_dt)
            self.arm_0, self.arm_1 = self.arm_1, self.arm_0
            self.state_0.clear_forces()
            # Only arm body entries copied; bottle body is never overwritten.
            wp.copy(self.state_0.body_q, self.arm_0.body_q, count=self.arm_count)
            wp.copy(self.state_0.body_qd, self.arm_0.body_qd, count=self.arm_count)
            wp.launch(move_anchors, len(self.anchors), inputs=[self.anchor_ids, self.anchor_local,
                      self.arm_0.body_q, self.ee_index, self.sim_dt, self.state_0.particle_q, self.state_0.particle_qd])
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.reaction.zero_()
            wp.launch(root_reaction, len(self.roots), inputs=[self.root_ids, self.anchor_ids,
                      self.state_0.particle_q, self.arm_0.body_q, self.arm_model.body_com,
                      self.ee_index, self.args.root_ke, self.reaction])

    def step(self):
        name, duration, target, activation = self.sequence[self.phase_index]
        self.phase_elapsed += self.frame_dt
        a = min(self.phase_elapsed / duration, 1.)
        a = a * a * (3 - 2 * a)
        self.command_tcp = (1 - a) * self.phase_start + a * target
        act = (1 - a) * self.phase_activation + a * activation
        self.solve_ik(self.command_tcp)
        self.arm_control.joint_target_pos.assign(self.joint_q_ik.numpy()[0])
        self.update_active_reference(act)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.episode_steps += 1
        self.measure(name)
        if not self.finished and self.phase_elapsed >= duration:
            print(f"[PHASE] {name}: lift={self.max_lift:.4f}m; "
                  f"bottle={self.state_0.body_q.numpy()[self.bottle, :3].round(4)}; "
                  f"root_error={self.max_root_error:.5f}m", flush=True)
            self.phase_start, self.phase_activation = target.copy(), activation
            self.phase_elapsed = 0.
            if self.phase_index == len(self.sequence) - 1:
                self.finished = True
            else:
                self.phase_index += 1

    def measure(self, phase):
        poses = self.state_0.body_q.numpy()
        p = self.state_0.particle_q.numpy()
        assert np.isfinite(p).all() and np.isfinite(poses).all(), "Nonfinite dynamics"
        position = poses[self.bottle, :3]
        lift = float(position[2] - self.pick[2])
        self.max_lift = max(self.max_lift, lift)
        if lift > .08:
            self.airborne_transport = max(self.airborne_transport, float(np.linalg.norm(position[:2] - self.pick[:2])))
        if phase == "hold":
            self.hold_min_lift = min(self.hold_min_lift, lift)
            self.hold_frames += 1
        self.max_tracking_error = max(self.max_tracking_error, float(np.linalg.norm(self.tcp_position(self.arm_0) - self.command_tcp)))
        self.max_root_error = max(self.max_root_error, float(np.linalg.norm(p[self.roots] - p[self.anchors], axis=1).max()))
        self.max_reaction = max(self.max_reaction, float(np.linalg.norm(self.reaction.numpy()[self.ee_index, :3])))
        speed = float(np.linalg.norm(self.state_0.body_qd.numpy()[self.bottle, :3]))
        self.max_speed = max(self.max_speed, speed)
        if int(self.contacts.soft_contact_count.numpy()[0]) > 0:
            self.contact_frames += 1
        # Radial tip displacement in the *measured* mount frame proves nonrigid bending.
        mount = wp.transform_multiply(wp.transform(*poses[self.ee_index]), wp.transform(wp.vec3(0), self.mount_rot))
        inv = wp.transform_inverse(mount)
        local = np.array([wp.transform_point(inv, wp.vec3(*v)) for v in p[self.tip_ids]])
        rest = np.concatenate(self.local_points)[self.tip_ids]
        bend = np.linalg.norm(rest[:, :2], axis=1) - np.linalg.norm(local[:, :2], axis=1)
        self.max_tip_bend = max(self.max_tip_bend, float(np.mean(bend)))
        if self.episode_steps % 60 == 0:
            xyz = p[self.tet_ids_np]
            volumes = np.linalg.det(np.stack([xyz[:, j] - xyz[:, 0] for j in (1, 2, 3)], axis=-1))
            self.min_tet_volume_ratio = min(self.min_tet_volume_ratio, float(np.min(volumes / self.reference_volumes)))
            assert self.min_tet_volume_ratio > 0., "Inverted simulated tetrahedron"
            sample = {"t": self.sim_time, "phase": phase, "bottle": position.tolist(),
                      "root_wrench": self.reaction.numpy()[self.ee_index].tolist(), "tip_bend_m": float(bend.mean())}
            self.history.append(sample)
            if self.args.verbose:
                print("[STATE] " + json.dumps(sample), flush=True)

    def test_final(self):
        position = self.state_0.body_q.numpy()[self.bottle, :3]
        error = np.linalg.norm(position - self.drop)
        stable = (self.max_root_error < .004 and self.max_speed < 2. and self.max_tracking_error < .02
              and self.min_tet_volume_ratio > .1)
        lifted = self.max_lift > .10
        held = self.hold_frames >= 60 and self.hold_min_lift > .09
        transported = self.airborne_transport > .11
        placed = error < .025
        success = bool(self.finished and stable and lifted and held and transported and placed)
        negative_ok = bool(self.finished and stable and self.max_lift < .03 and not held)
        report = dict(success=success, expected_failure=self.args.expect_grasp_failure,
                      negative_control_passed=negative_ok if self.args.expect_grasp_failure else None,
                      finished=self.finished, stable=stable, lifted=lifted, held=held,
                      transported=transported, placed=bool(placed), mass_kg=self.args.mass,
                      max_lift_m=self.max_lift, hold_min_lift_m=self.hold_min_lift if self.hold_frames else None,
                      airborne_transport_m=self.airborne_transport, final_position_m=position.tolist(),
                      placement_error_m=float(error), max_speed_m_s=self.max_speed,
                      max_root_error_m=self.max_root_error, max_root_reaction_N=self.max_reaction,
                      max_tcp_error_m=self.max_tracking_error, max_tip_bend_m=self.max_tip_bend,
                      min_sampled_tet_volume_ratio=self.min_tet_volume_ratio,
                      activation=self.args.activation, curvature_per_m=self.args.curvature, friction=self.args.mu,
                      shear_Pa=self.args.shear, tet_count=self.model.tet_count, assistance=False,
                      model_boundary="CAD convex-envelope FEM + active reference curvature, not SRT pressure calibration",
                      coupling="MuJoCo arm; VBD fingers + dynamic bottle; finite root springs, wrench feedback with one substep lag",
                      newton_version=newton.__version__, warp_version=wp.__version__,
                      frames=self.episode_steps, substeps=self.args.substeps, iterations=self.args.iterations, history=self.history)
        self.args.report.parent.mkdir(parents=True, exist_ok=True)
        self.args.report.write_text(json.dumps(report, indent=2) + "\n")
        print("[RESULT] " + json.dumps({k: v for k, v in report.items() if k != "history"}), flush=True)
        if self.args.test:
            assert negative_ok if self.args.expect_grasp_failure else success, "Grasp regression failed; see report"


def create_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--viewer", choices=["gl", "null", "usd", "viser"], default="gl")
    p.add_argument("--device", default=None)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--warp-config", action="append", default=[])
    p.add_argument("--output-path", default="sfg_soft_grasp.usd")
    p.add_argument("--robot-usd", type=Path, default=ASSETS / "abb_irb1200_sfg_n4049.usda")
    p.add_argument("--num-frames", type=int, default=1100)
    p.add_argument("--substeps", type=int, default=40)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--ik-iters", type=int, default=20)
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--mass", type=float, default=1.)
    p.add_argument("--activation", type=float, default=1., help="Dimensionless assumed actuation; NOT pressure")
    p.add_argument("--curvature", type=float, default=24., help="Active reference curvature, 1/m")
    p.add_argument("--shear", type=float, default=3e5, help="Assumed effective shear modulus, Pa")
    p.add_argument("--mu", type=float, default=.8)
    p.add_argument("--root-ke", type=float, default=1000., help="Per-root attachment spring N/m")
    p.add_argument("--contact-ke", type=float, default=3e5)
    p.add_argument("--offset-x", type=float, default=0.)
    p.add_argument("--offset-y", type=float, default=0.)
    p.add_argument("--test", action="store_true")
    p.add_argument("--expect-grasp-failure", action="store_true")
    p.add_argument("--snapshot-dir", type=Path, default=None, help="Save GL images at phase boundaries")
    p.add_argument("--report", type=Path, default=ASSETS / "validation/soft_grasp.json")
    return p


if __name__ == "__main__":
    args = create_parser().parse_args()
    viewer = init_viewer(args)
    try:
        example = SFGSoftGraspExample(viewer, args)
        while viewer.is_running() and example.episode_steps < args.num_frames and not example.finished:
            phase = example.phase_index
            if viewer.should_step():
                example.step()
            example.render()
            if args.snapshot_dir and (phase != example.phase_index or example.finished):
                from PIL import Image
                args.snapshot_dir.mkdir(parents=True, exist_ok=True)
                image = viewer.get_frame(render_ui=False).numpy()
                Image.fromarray(image).save(args.snapshot_dir / f"{phase:02d}_{example.sequence[phase][0]}.png")
        example.test_final()
    finally:
        viewer.close()
