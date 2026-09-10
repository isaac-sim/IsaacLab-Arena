"""ABB IRB1200 + vendor PGC140 with offset fingertips: contact-only cube stacking.

Reuses example35's viewer and IK + joint-drive + task-sequence pattern,
but calibrates the new tool from its USD and sets POSITION actuation explicitly.
No attach, cube-pose assignment, wrist-state clamping, or hidden grasp helper.
Only initial robot placement uses FK; all subsequent motion uses dynamics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.ik as ik
from example_newton_viewer_utils import (
    arena_root,
    init_viewer,
)


def named_index(labels, name):
    """Resolve a unique body/joint by leaf name instead of hardcoded indices."""
    matches = [index for index, label in enumerate(labels) if label.rsplit("/", 1)[-1] == name]
    assert len(matches) == 1, (name, matches, labels)
    return matches[0]


def rotation_error_degrees(quaternion, target):
    dot = abs(float(np.dot(quaternion, target)))
    return math.degrees(2 * math.acos(np.clip(dot, 0, 1)))


class PGC140CubeStackingExample:
    """Single-world dynamic pick/lift/hold/transport/place regression."""

    def __init__(self, viewer, args):
        assert args.world_count == 1, "This asset regression supports one world; run separate seeds for trials."
        assert 0 < args.cube_mass and 0 < args.grip_effort <= 140 and 0 < args.grip_mu
        assert args.grip_ke > 0 and args.grip_kd >= 0 and args.ik_iters > 0 and args.num_frames > 0
        self.args = args
        self.viewer = viewer
        self.frame_dt = 1 / 60
        self.sim_dt = self.frame_dt / 10
        self.sim_time = 0.0
        self.episode_steps = 0
        self.cube_size = 0.05
        self.table_height = 0.1
        self.home_rot = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), math.pi / 2)
        self.tcp_offset = wp.vec3(0.130, 0, 0)
        self.pick_positions = np.array([[0.16, 0.09, 0.125], [0.24, -0.015, 0.125], [0.075, -0.065, 0.125]])
        self.pick_positions[:, :2] += [args.cube_offset_x, args.cube_offset_y]
        self.drop_position = np.array([0.12, -0.18, 0.125])
        self.initial_tcp = self.pick_positions[0] + [0, 0, 0.22]

        robot = self.build_robot()
        self.robot_model = robot.finalize()
        self.setup_ik()
        # Solve a collision-free pregrasp BEFORE starting the episode.
        self.solve_ik(self.initial_tcp, iterations=150)
        initial_q = self.joint_q_ik.numpy()[0]
        probe = self.robot_model.state()
        newton.eval_fk(self.robot_model, wp.array(initial_q, dtype=float), self.robot_model.joint_qd, probe)
        initial_error = np.linalg.norm(self.tcp_position(probe) - self.initial_tcp)
        assert initial_error < 0.002, f"Initial IK error {initial_error:.6f} m: inspect mount/reachability"
        robot.joint_q[:] = initial_q.tolist()
        robot.joint_target_pos[:] = initial_q.tolist()

        self.cube_indices = []
        self.build_table_and_cubes(robot)
        self.model = robot.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        # Free cubes add generalized coordinates; copy robot targets only.
        targets = self.control.joint_target_pos.numpy()
        targets[:len(initial_q)] = initial_q
        self.control.joint_target_pos.assign(targets)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, use_mujoco_cpu=args.mujoco_cpu, use_mujoco_contacts=True,
            solver="newton", integrator="implicitfast", iterations=50, ls_iterations=30,
            cone="elliptic", impratio=10.0, nconmax=512, njmax=2000,
        )
        self.contacts = self.model.contacts()
        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = False

        self.sequence = self.make_sequence()
        self.phase_index = 0
        self.phase_elapsed = 0.0
        self.phase_start_tcp = self.initial_tcp.copy()
        self.phase_start_grip = 0.0
        self.command_tcp = self.initial_tcp.copy()
        self.command_grip = 0.0
        self.max_lift = np.zeros(args.cube_count)
        self.max_transport = np.zeros(args.cube_count)
        self.max_speed = np.zeros(args.cube_count)
        self.max_tracking_error = 0.0
        self.finished = False
        print(f"[MODEL] cube masses = {self.model.body_mass.numpy()[self.cube_indices].tolist()} kg", flush=True)
        print(f"[MODEL] POSITION drives; {args.grip_effort} N cap/finger; mu={args.grip_mu}; no assistance", flush=True)
        print(f"[PHASE] {self.sequence[0]['name']}", flush=True)

    def build_robot(self):
        robot = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(robot)
        usd_path = self.args.robot_usd.resolve()
        assert usd_path.is_file(), f"Generate USD first with tools/convert_pgc140_to_abb_irb1200_usd.py: {usd_path}"
        robot.add_usd(
            str(usd_path), xform=wp.transform(wp.vec3(-0.30, 0, self.table_height), wp.quat_identity()),
            floating=False, enable_self_collisions=False, collapse_fixed_joints=False,
            load_visual_shapes=self.args.load_visuals, force_show_colliders=self.args.show_colliders,
        )
        assert robot.joint_dof_count == 8, robot.joint_dof_count
        self.ee_index = named_index(robot.body_label, "link_6")
        self.pad_indices = [named_index(robot.body_label, f"pgc140_pad{i}") for i in (1, 2)]
        self.arm_dofs = []
        for number in range(1, 7):
            # The fixed root synthesized by Newton can also be called joint_1.
            matches = [i for i, label in enumerate(robot.joint_label)
                       if label.endswith(f"/joint_{number}")]
            assert len(matches) == 1, matches
            self.arm_dofs.append(robot.joint_qd_start[matches[0]])
        self.gripper_dofs = [robot.joint_qd_start[named_index(robot.joint_label, f"pgc140_finger{i}_joint")]
                             for i in (1, 2)]
        assert self.arm_dofs == list(range(6)), self.arm_dofs
        for dof in range(8):
            # URDF importer creates EFFORT drives if gains are unspecified.
            # Updating ke/kd without this mode switch does NOT enable a servo.
            robot.joint_target_mode[dof] = newton.JointTargetMode.POSITION
        robot.joint_q[:6] = [0, -0.55, 0.95, 0, 0.85, 0]
        robot.joint_target_ke[:6] = [3500, 3500, 3000, 1800, 1600, 1200]
        robot.joint_target_kd[:6] = [350, 350, 300, 180, 160, 120]
        robot.joint_effort_limit[:6] = [300] * 6
        robot.joint_armature[:6] = [0.05] * 6
        for dof in self.gripper_dofs:
            robot.joint_q[dof] = 0.0  # vendor joints: zero=open, +25mm=closed
            robot.joint_target_ke[dof] = self.args.grip_ke
            robot.joint_target_kd[dof] = self.args.grip_kd
            robot.joint_effort_limit[dof] = self.args.grip_effort
            robot.joint_armature[dof] = 0.01
        # Controller gravity compensation applies only to the robot, never cubes.
        robot.custom_attributes["mujoco:gravcomp"].values = {i: 1.0 for i in range(robot.body_count)}
        for shape_id in range(robot.shape_count):
            robot.shape_material_mu[shape_id] = self.args.grip_mu
            robot.shape_material_mu_torsional[shape_id] = 0.00002
            robot.shape_material_mu_rolling[shape_id] = 0.00002
            robot.shape_margin[shape_id] = 0.0
            robot.shape_gap[shape_id] = 0.0
        # Unlike old examples, retain the imported arm, housing AND finger
        # collisions. The new pads already live in the USD, not runtime proxies.
        print(f"[MODEL] TCP link={self.ee_index}; pad bodies={self.pad_indices}; gripper DOFs={self.gripper_dofs}")
        return robot

    def setup_ik(self):
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_index, link_offset=self.tcp_offset,
            target_positions=wp.array([self.initial_tcp], dtype=wp.vec3),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_index, link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([self.home_rot], dtype=wp.vec4), weight=1.0,
        )
        limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.robot_model.joint_limit_lower,
            joint_limit_upper=self.robot_model.joint_limit_upper,
        )
        self.joint_q_ik = wp.clone(self.robot_model.joint_q.reshape((1, -1)))
        self.ik_solver = ik.IKSolver(
            model=self.robot_model, n_problems=1, objectives=[self.pos_obj, self.rot_obj, limits],
            jacobian_mode=ik.IKJacobianType.ANALYTIC, lambda_initial=0.01,
        )

    def solve_ik(self, position, iterations=None):
        self.pos_obj.set_target_positions(wp.array([position], dtype=wp.vec3))
        self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=iterations or self.args.ik_iters)

    def build_table_and_cubes(self, builder):
        table_cfg = newton.ModelBuilder.ShapeConfig(mu=0.6, margin=0, gap=0)
        builder.add_shape_box(body=-1, hx=0.6, hy=0.4, hz=0.05,
                              xform=wp.transform(wp.vec3(0.05, 0, 0.05), wp.quat_identity()), cfg=table_cfg,
                              label="table", color=(0.42, 0.46, 0.50))
        cfg = newton.ModelBuilder.ShapeConfig(density=self.args.cube_mass / self.cube_size**3,
                                              mu=self.args.grip_mu, margin=0, gap=0,
                                              mu_torsional=0.00002, mu_rolling=0.00002)
        for i in range(self.args.cube_count):
            index = builder.add_body(xform=wp.transform(wp.vec3(*self.pick_positions[i]), wp.quat_identity()),
                                     label=f"cube_{i}")
            self.cube_indices.append(index)
            builder.add_shape_box(body=index, hx=0.025, hy=0.025, hz=0.025, cfg=cfg,
                                  label=f"cube_{i}_collision", color=[(0.85, 0.2, 0.16), (0.2, 0.72, 0.3), (0.2, 0.3, 0.85)][i])
        builder.add_ground_plane()

    def make_sequence(self):
        sequence = []
        for index in range(self.args.cube_count):
            # Extra 3mm avoids table contact at the bottoms of the 50mm pads.
            grasp = self.pick_positions[index] + [0, 0, 0.003]
            drop = self.drop_position + [0, 0, self.cube_size * index + 0.003]
            high = np.array([0, 0, 0.20])
            for name, target, grip, seconds in [
                ("approach", grasp + high, 0, 2.0),
                ("descend", grasp, 0, 2.0),
                ("close", grasp, 0.025, 1.5),
                ("lift", grasp + high, 0.025, 2.0),
                ("hold", grasp + high, 0.025, 1.0),
                ("transport", drop + high, 0.025, 2.5),
                ("place", drop, 0.025, 2.0),
                ("release", drop, 0, 1.0),
                ("retract", drop + high, 0, 1.5),
                ("settle", drop + high, 0, 1.0),
            ]:
                sequence.append(dict(name=f"cube{index}/{name}", target=target, grip=grip,
                                     seconds=seconds, cube=index, kind=name))
        return sequence

    def tcp_position(self, state):
        tf = wp.transform(*state.body_q.numpy()[self.ee_index])
        return np.array(wp.transform_point(tf, self.tcp_offset))

    def step(self):
        phase = self.sequence[self.phase_index]
        if not self.finished:
            self.phase_elapsed += self.frame_dt
            fraction = min(self.phase_elapsed / phase["seconds"], 1.0)
            alpha = fraction * fraction * (3 - 2 * fraction)
            self.command_tcp = self.phase_start_tcp * (1 - alpha) + phase["target"] * alpha
            self.command_grip = self.phase_start_grip * (1 - alpha) + phase["grip"] * alpha
        self.solve_ik(self.command_tcp)
        q = self.joint_q_ik.numpy()[0]
        target = self.control.joint_target_pos.numpy()
        target[:6] = q[:6]
        target[self.gripper_dofs] = self.command_grip
        self.control.joint_target_pos.assign(target)
        for _ in range(10):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt
        self.episode_steps += 1
        self.measure()

        if not self.finished and self.phase_elapsed >= phase["seconds"]:
            error = np.linalg.norm(self.tcp_position(self.state_0) - phase["target"])
            rotation = self.state_0.body_q.numpy()[self.ee_index, 3:]
            rot_error = rotation_error_degrees(rotation, np.array(self.home_rot))
            if error > 0.003 or rot_error > 3.0:
                if self.phase_elapsed > phase["seconds"] + 3.0:
                    raise RuntimeError(f"{phase['name']}: TCP not reached ({error:.4f}m, {rot_error:.2f}deg); refusing blind close/advance")
                return
            print(f"[PHASE] {phase['name']} done: TCP err={error:.5f}m; lift={self.max_lift.round(4).tolist()}", flush=True)
            self.phase_start_tcp = phase["target"].copy()
            self.phase_start_grip = phase["grip"]
            self.phase_elapsed = 0
            if self.phase_index == len(self.sequence) - 1:
                self.finished = True
            else:
                self.phase_index += 1

    def measure(self):
        poses = self.state_0.body_q.numpy()
        assert np.isfinite(poses).all(), "Non-finite body state"
        positions = poses[self.cube_indices, :3]
        rises = positions[:, 2] - self.pick_positions[:self.args.cube_count, 2]
        self.max_lift = np.maximum(self.max_lift, rises)
        distances = np.linalg.norm(positions[:, :2] - self.pick_positions[:self.args.cube_count, :2], axis=1)
        self.max_transport = np.maximum(self.max_transport, np.where(rises > 0.08, distances, 0))
        velocities = self.state_0.body_qd.numpy()[self.cube_indices, :3]
        self.max_speed = np.maximum(self.max_speed, np.linalg.norm(velocities, axis=1))
        error = np.linalg.norm(self.tcp_position(self.state_0) - self.command_tcp)
        self.max_tracking_error = max(self.max_tracking_error, float(error))
        if self.args.verbose and self.episode_steps % 120 == 0:
            print(f"[STATE] t={self.sim_time:.2f}; cubes={positions.round(4).tolist()}; TCP err={error:.5f}m", flush=True)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        poses = self.state_0.body_q.numpy()
        results = []
        for index, body in enumerate(self.cube_indices):
            target = self.drop_position + [0, 0, self.cube_size * index]
            position_error = poses[body, :3] - target
            # Cube yaw is symmetric under quarter-turns; use |quaternion dot|
            # so q and -q also represent the same orientation.
            angle_error = min(rotation_error_degrees(poses[body, 3:], np.array(
                wp.quat_from_axis_angle(wp.vec3(0, 0, 1), turn * math.pi / 2))) for turn in range(4))
            lifted = bool(self.max_lift[index] > 0.10)
            # The third cube's planned path is only 12.3cm long. A fixed 15cm
            # threshold falsely rejects a correctly transported/stacked cube.
            required_transport = 0.8 * float(np.linalg.norm(target[:2] - self.pick_positions[index, :2]))
            transported = bool(self.max_transport[index] >= required_transport)
            placed = bool(np.linalg.norm(position_error[:2]) < 0.02 and abs(position_error[2]) < 0.01 and angle_error < 5)
            results.append(dict(cube=index, mass_kg=self.args.cube_mass, lifted=lifted, transported=transported,
                                placed=placed, max_lift_m=float(self.max_lift[index]),
                                required_airborne_transport_m=required_transport,
                                airborne_transport_m=float(self.max_transport[index]),
                                final_position_m=poses[body, :3].tolist(), position_error_m=position_error.tolist(),
                                orientation_error_degrees=angle_error, max_speed_m_s=float(self.max_speed[index])))
        success = self.finished and all(r["lifted"] and r["transported"] and r["placed"] for r in results)
        report = dict(success=success, finished=self.finished, frames=self.episode_steps, simulated_seconds=self.sim_time,
                      contact_backend="MuJoCo native", mujoco_cpu=self.args.mujoco_cpu, assistance=False,
                      newton_version=newton.__version__, warp_version=wp.__version__,
                      grip_effort_per_finger_N=self.args.grip_effort, friction=self.args.grip_mu,
                      max_tcp_tracking_error_m=self.max_tracking_error, cubes=results)
        print("[RESULT] " + json.dumps(report, ensure_ascii=False), flush=True)
        if self.args.report:
            self.args.report.parent.mkdir(parents=True, exist_ok=True)
            self.args.report.write_text(json.dumps(report, indent=2) + "\n")
        assert success, "Contact-only lift/transport/place regression failed; see separate result fields."


def create_parser():
    # Do not inherit old tuning options which this controller does not use.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", default=None)
    parser.add_argument("--viewer", choices=["gl", "usd", "null", "viser"], default="gl")
    parser.add_argument("--output-path", default="pgc140_recording.usd")
    parser.add_argument("--num-frames", type=int, default=1200)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--warp-config", action="append", default=[])
    parser.add_argument("--world-count", type=int, choices=[1], default=1)
    parser.add_argument("--cube-count", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--grip-ke", type=float, default=20000.0, help="Position-drive stiffness, N/m.")
    parser.add_argument("--grip-kd", type=float, default=200.0, help="Position-drive damping, N s/m.")
    parser.add_argument("--grip-effort", type=float, default=100.0, help="Per-finger effort limit, N (<=140).")
    parser.add_argument("--grip-mu", type=float, default=0.8, help="Assumed dry friction; not a bottle material calibration.")
    parser.add_argument("--ik-iters", type=int, default=20)
    parser.add_argument("--load-visuals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-colliders", action="store_true")
    parser.add_argument("--robot-usd", type=Path, default=arena_root() / "examples/examples_teleop/assets/abb_irb1200_pgc140/combined/abb_irb1200_pgc140.usda")
    parser.add_argument("--cube-mass", type=float, default=1.0, help="Mass per 5cm test cube, not a bottle model.")
    parser.add_argument("--cube-offset-x", type=float, default=0.0, help="Regression offset in metres.")
    parser.add_argument("--cube-offset-y", type=float, default=0.0, help="Regression offset in metres.")
    parser.add_argument("--mujoco-cpu", action="store_true", help="Use CPU MuJoCo instead of default mujoco_warp backend.")
    parser.add_argument("--report", type=Path, default=None, help="Write separate lift/transport/place metrics with --test.")
    return parser


if __name__ == "__main__":
    args = create_parser().parse_args()
    viewer = init_viewer(args)
    try:
        example = PGC140CubeStackingExample(viewer, args)
        # ViewerNull enforces num_frames itself; ViewerGL does not. Bound the
        # runner too so GUI and headless use identical validation durations.
        while viewer.is_running() and example.episode_steps < args.num_frames:
            if viewer.should_step():
                example.step()
            example.render()
        if args.test:
            example.test_final()
        else:
            print(f"[INFO] Finished {example.episode_steps} frames.")
    finally:
        viewer.close()
