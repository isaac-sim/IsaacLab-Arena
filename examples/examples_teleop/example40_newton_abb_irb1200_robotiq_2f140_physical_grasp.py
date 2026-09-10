"""Stock ROS-Industrial Robotiq 2F-140 + local ABB USD: single-actuator physical grasp.

Retains all six revolute joints and five native mimic equality constraints.
No follower motors, cube attach, cube pose writes or robot state clamping.
Reuses example39's table, trajectories and contact-only stepping; never its
PGC140 asset or two-prismatic drive model. Run in the existing Arena Docker.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.ik as ik
from example39_newton_standalone_abb_irb1200_pgc140_cube_stacking import (
    PGC140CubeStackingExample,
    arena_root,
    create_parser as create_common_parser,
    init_viewer,
    named_index,
    rotation_error_degrees,
)

ASSETS = arena_root() / "examples/examples_teleop/assets/abb_irb1200_robotiq_2f140_urdf"
CONSTRAINT_SETTINGS = {
    "eq_solref": [0.0035, 1.0],
    "eq_solimp": [0.9999, 0.9999, 0.001, 0.5, 2.0],
    "geom_solref": [0.005, 1.0],
    "geom_solimp": [0.99, 0.999, 0.001, 0.5, 2.0],
}


def register_newton_usd_schema():
    """Load the installed schema before add_usd; unregistered HasAPI is false."""
    from pxr import Plug, Usd

    # Register BEFORE constructing SchemaRegistry: USD caches schema definitions
    # on first use and cannot discover this late-loaded plugin afterwards.
    plugin = Path("/isaac-sim/exts/omni.usd.schema.newton/usd/schema/newton/newton_usd_schemas")
    if plugin.is_dir():
        Plug.Registry().RegisterPlugins(str(plugin))
    assert Usd.SchemaRegistry().FindAppliedAPIPrimDefinition("NewtonMimicAPI"), "Newton schema unavailable"


def configure_contact_solver(solver):
    """Set physical constraint softness once, without altering dynamic state.

    Newton 1.2.1 emits mimic equalities using MuJoCo defaults and does not
    forward custom equality solref/solimp for mimics. Configure the compiled
    CPU model and its Warp copy explicitly; no external package is patched.
    Contact and mimic settings are BOTH needed for these light stock links.
    Positive time constants remain above twice the 1/600s physics step.
    """
    assert solver.mj_model.neq == 5, "Only the five native mimic constraints are expected"
    assert np.all(solver.mj_model.eq_type == 2), "Expected MuJoCo JOINT equalities"
    if solver.mjc_eq_to_newton_mimic is not None:
        mapping = solver.mjc_eq_to_newton_mimic.numpy()
        assert len(np.unique(mapping[mapping >= 0])) == 5, "Solver dropped native mimic constraints"
    for name, value in CONSTRAINT_SETTINGS.items():
        host = getattr(solver.mj_model, name)
        host[:] = value
        if solver.mjw_model is not None:
            getattr(solver.mjw_model, name).assign(host[None, ...])


class Robotiq2F140PhysicalExample(PGC140CubeStackingExample):
    """Use the upstream linkage and native constraints, not software follower PD."""

    def __init__(self, viewer, args):
        assert args.world_count == 1
        assert args.cube_mass > 0 and 0 < args.grip_effort <= 10
        assert args.grip_ke > 0 and args.grip_kd >= 0 and args.grip_mu > 0
        assert args.ik_iters > 0 and args.num_frames > 0
        assert not args.expect_grasp_failure or args.test, "Negative control requires --test"
        self.args, self.viewer = args, viewer
        self.frame_dt, self.sim_dt = 1 / 60, 1 / 600
        self.sim_time, self.episode_steps = 0.0, 0
        self.cube_size, self.table_height = 0.05, 0.1
        self.home_rot = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), math.pi / 2)
        self.pick_positions = np.array([[0.16, 0.09, 0.125], [0.24, -0.015, 0.125], [0.075, -0.065, 0.125]])
        self.pick_positions[:, :2] += [args.cube_offset_x, args.cube_offset_y]
        self.drop_position = np.array([0.12, -0.18, 0.125])
        self.initial_tcp = self.pick_positions[0] + [0, 0, 0.22]
        self.max_mimic_error = 0.0
        robot = self.build_robot()
        self.robot_model = robot.finalize()
        self.setup_ik()
        self.solve_ik(self.initial_tcp, iterations=150)
        initial_q = self.joint_q_ik.numpy()[0]
        probe = self.robot_model.state()
        newton.eval_fk(self.robot_model, wp.array(initial_q, dtype=float), self.robot_model.joint_qd, probe)
        assert np.linalg.norm(self.tcp_position(probe) - self.initial_tcp) < 0.002, "Initial pregrasp IK failed"
        robot.joint_q[:] = initial_q.tolist()
        robot.joint_target_pos[:] = initial_q.tolist()
        self.cube_indices = []
        self.build_table_and_cubes(robot)
        self.model = robot.finalize()
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        targets = self.control.joint_target_pos.numpy()
        targets[:len(initial_q)] = initial_q
        self.control.joint_target_pos.assign(targets)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, use_mujoco_cpu=args.mujoco_cpu, use_mujoco_contacts=True,
            solver="newton", integrator="implicitfast", iterations=80, ls_iterations=40,
            cone="elliptic", impratio=10, nconmax=1024, njmax=3000,
        )
        configure_contact_solver(self.solver)
        self.contacts = self.model.contacts()
        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = False
        self.sequence = self.make_sequence()
        self.phase_index, self.phase_elapsed = 0, 0.0
        self.phase_start_tcp, self.command_tcp = self.initial_tcp.copy(), self.initial_tcp.copy()
        self.phase_start_grip, self.command_grip = 0.0, 0.0
        self.max_lift = np.zeros(args.cube_count)
        self.max_transport = np.zeros(args.cube_count)
        self.max_speed = np.zeros(args.cube_count)
        self.max_tracking_error, self.finished = 0.0, False
        print(f"[MODEL] 6 ABB DOFs + 6 revolute gripper DOFs; 5 native mimic equalities", flush=True)
        print(f"[MODEL] Single gripper actuator, {args.grip_effort} Nm torque cap; followers passive; no assistance", flush=True)
        print(f"[MODEL] cube masses={self.model.body_mass.numpy()[self.cube_indices].tolist()} kg", flush=True)

    def build_robot(self):
        register_newton_usd_schema()
        robot = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(robot)
        assert self.args.robot_usd.is_file(), "Run tools/convert_robotiq_2f140_urdf_to_abb_usd.py first"
        robot.add_usd(str(self.args.robot_usd.resolve()),
                      xform=wp.transform(wp.vec3(-0.30, 0, self.table_height), wp.quat_identity()),
                      floating=False, enable_self_collisions=False, collapse_fixed_joints=False,
                      load_visual_shapes=self.args.load_visuals, force_show_colliders=self.args.show_colliders)
        assert robot.joint_dof_count == 12, robot.joint_dof_count
        self.ee_index = named_index(robot.body_label, "link_6")
        self.pad_indices = [named_index(robot.body_label, f"{side}_inner_finger_pad") for side in ("left", "right")]
        arm_joints = [next(i for i, label in enumerate(robot.joint_label) if label.endswith(f"/joint_{j}")) for j in range(1, 7)]
        self.arm_dofs = [robot.joint_qd_start[j] for j in arm_joints]
        assert self.arm_dofs == list(range(6)), self.arm_dofs
        master = named_index(robot.joint_label, "finger_joint")
        self.master_dof = robot.joint_qd_start[master]
        self.gripper_dofs = [self.master_dof]  # only this joint gets commanded by inherited step()
        manifest = json.loads((ASSETS / "source_manifest.json").read_text())
        relations = manifest["mimic"]
        self.followers = []
        # Validate the actual imported constraints against the downloaded URDF.
        mimic_count = len(robot.constraint_mimic_joint0)
        assert mimic_count == 5, mimic_count
        for name, relation in relations.items():
            follower = named_index(robot.joint_label, name)
            leader = named_index(robot.joint_label, relation["leader"])
            matches = [i for i in range(mimic_count)
                       if robot.constraint_mimic_joint0[i] == follower and robot.constraint_mimic_joint1[i] == leader]
            assert len(matches) == 1, (name, matches)
            constraint = matches[0]
            assert np.isclose(robot.constraint_mimic_coef1[constraint], relation["multiplier"]), (name, robot.constraint_mimic_coef1)
            assert np.isclose(robot.constraint_mimic_coef0[constraint], relation["offset"])
            self.followers.append((robot.joint_qd_start[follower], relation["multiplier"], relation["offset"]))
        for dof in range(12):
            robot.joint_target_mode[dof] = newton.JointTargetMode.NONE
            robot.joint_target_ke[dof] = 0.0
            robot.joint_target_kd[dof] = 0.0
        robot.joint_q[:6] = [0, -0.55, 0.95, 0, 0.85, 0]
        robot.joint_q[6:] = [0.0] * 6
        for dof in self.arm_dofs:
            robot.joint_target_mode[dof] = newton.JointTargetMode.POSITION
        robot.joint_target_ke[:6] = [3500, 3500, 3000, 1800, 1600, 1200]
        robot.joint_target_kd[:6] = [350, 350, 300, 180, 160, 120]
        robot.joint_effort_limit[:6] = [300] * 6
        robot.joint_armature[:6] = [0.05] * 6
        robot.joint_target_mode[self.master_dof] = newton.JointTargetMode.POSITION
        robot.joint_target_ke[self.master_dof] = self.args.grip_ke
        robot.joint_target_kd[self.master_dof] = self.args.grip_kd
        robot.joint_effort_limit[self.master_dof] = self.args.grip_effort
        robot.joint_armature[self.master_dof] = 0.001
        robot.custom_attributes["mujoco:gravcomp"].values = {i: 1.0 for i in range(robot.body_count)}
        for shape in range(robot.shape_count):
            robot.shape_material_mu[shape] = self.args.grip_mu
            robot.shape_material_mu_torsional[shape] = 0.00002
            robot.shape_material_mu_rolling[shape] = 0.00002
            robot.shape_margin[shape] = 0.0
            robot.shape_gap[shape] = 0.0
        print(f"[MODEL] master={self.master_dof}, followers={self.followers}; pad bodies={self.pad_indices}", flush=True)
        return robot

    def setup_ik(self):
        # Unlike a parallel slider, the 2F140 finger midpoint shifts in Z while
        # opening. Solve for the flange and compensate using MEASURED pad FK,
        # not a fixed TCP or the unachieved closing target angle.
        self.pos_obj = ik.IKObjectivePosition(link_index=self.ee_index, link_offset=wp.vec3(0),
                                              target_positions=wp.array([self.initial_tcp], dtype=wp.vec3))
        self.rot_obj = ik.IKObjectiveRotation(link_index=self.ee_index, link_offset_rotation=wp.quat_identity(),
                                              target_rotations=wp.array([self.home_rot], dtype=wp.vec4), weight=1.0)
        limits = ik.IKObjectiveJointLimit(joint_limit_lower=self.robot_model.joint_limit_lower,
                                         joint_limit_upper=self.robot_model.joint_limit_upper)
        self.joint_q_ik = wp.clone(self.robot_model.joint_q.reshape((1, -1)))
        self.ik_solver = ik.IKSolver(model=self.robot_model, n_problems=1,
                                   objectives=[self.pos_obj, self.rot_obj, limits],
                                   jacobian_mode=ik.IKJacobianType.ANALYTIC, lambda_initial=0.01)
        self.initial_probe = self.robot_model.state()
        newton.eval_fk(self.robot_model, self.robot_model.joint_q, self.robot_model.joint_qd, self.initial_probe)

    def solve_ik(self, position, iterations=None):
        state = self.state_0 if hasattr(self, "state_0") else self.initial_probe
        poses = state.body_q.numpy()
        midpoint = poses[self.pad_indices, :3].mean(axis=0)
        offset = wp.transform_point(wp.transform_inverse(wp.transform(*poses[self.ee_index])), wp.vec3(*midpoint))
        flange_target = np.asarray(position) - np.array(wp.quat_rotate(self.home_rot, offset))
        self.pos_obj.set_target_positions(wp.array([flange_target], dtype=wp.vec3))
        self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=iterations or self.args.ik_iters)

    def tcp_position(self, state):
        return state.body_q.numpy()[self.pad_indices, :3].mean(axis=0)

    def make_sequence(self):
        sequence = super().make_sequence()
        for phase in sequence:
            # Stock pads are 70mm tall, versus the 50mm PGC tooling: keep
            # their lower edge above the table, without replacing the geometry.
            phase["target"] = phase["target"] + [0, 0, 0.011]
            if phase["grip"] > 0:
                phase["grip"] = 0.7  # Upstream 2F140: q=0 open, q=0.7 closed.
            if phase["kind"] in ("close", "release"):
                phase["seconds"] = 2.0
        return sequence

    def measure(self):
        super().measure()
        q = self.state_0.joint_q.numpy()
        residual = max(abs(q[dof] - multiplier * q[self.master_dof] - offset)
                       for dof, multiplier, offset in self.followers)
        self.max_mimic_error = max(self.max_mimic_error, float(residual))
        assert np.isfinite(q).all(), "Nonfinite joint state"
        if self.args.verbose and self.episode_steps % 120 == 0:
            print(f"[GRIP] q={q[6:12].round(4).tolist()}; mimic residual={residual:.6f}rad", flush=True)

    def test_final(self):
        poses = self.state_0.body_q.numpy()
        cubes = []
        for i, body in enumerate(self.cube_indices):
            goal = self.drop_position + [0, 0, i * self.cube_size]
            error = poses[body, :3] - goal
            rotation = min(rotation_error_degrees(poses[body, 3:], np.array(
                wp.quat_from_axis_angle(wp.vec3(0, 0, 1), j * math.pi / 2))) for j in range(4))
            transport = 0.8 * float(np.linalg.norm(goal[:2] - self.pick_positions[i, :2]))
            cubes.append(dict(cube=i, mass_kg=self.args.cube_mass, max_lift_m=float(self.max_lift[i]),
                              airborne_transport_m=float(self.max_transport[i]),
                              lifted=bool(self.max_lift[i] > 0.10), transported=bool(self.max_transport[i] > transport),
                              placed=bool(np.linalg.norm(error[:2]) < 0.02 and abs(error[2]) < 0.01 and rotation < 5),
                              position_error_m=error.tolist(), orientation_error_degrees=rotation))
        success = self.finished and all(c["lifted"] and c["transported"] and c["placed"] for c in cubes)
        success = success and self.max_mimic_error < 0.01
        negative_passed = (not success and self.finished and self.max_mimic_error < 0.01
                   and all(c["max_lift_m"] < 0.02 and c["airborne_transport_m"] < 0.02 for c in cubes))
        validation_passed = negative_passed if self.args.expect_grasp_failure else success
        report = dict(success=success, assistance=False, gripper="ROS-Industrial Robotiq 2F-140",
                  expected_grasp_failure=self.args.expect_grasp_failure, validation_passed=validation_passed,
                      native_mimic_count=5, active_gripper_motors=1, follower_motors=0,
                      master_torque_limit_Nm=self.args.grip_effort, friction=self.args.grip_mu,
                  constraint_settings=CONSTRAINT_SETTINGS, metric_sample_hz=60,
                  contact_backend="MuJoCo native", simulated_seconds=self.sim_time,
                      max_mimic_error_rad=self.max_mimic_error, max_tcp_tracking_error_m=self.max_tracking_error,
                      frames=self.episode_steps, finished=self.finished, cubes=cubes,
                      newton_version=newton.__version__, warp_version=wp.__version__, mujoco_cpu=self.args.mujoco_cpu)
        print("[RESULT] " + json.dumps(report), flush=True)
        if self.args.report:
            self.args.report.parent.mkdir(parents=True, exist_ok=True)
            self.args.report.write_text(json.dumps(report, indent=2) + "\n")
        assert validation_passed, "Physical grasp, native mimic or expected negative-control regression failed"


def create_parser():
    parser = create_common_parser()
    parser.description = __doc__
    parser.set_defaults(robot_usd=ASSETS / "abb_irb1200_robotiq_2f140.usda", cube_mass=1.0,
                        output_path="robotiq_2f140_recording.usd", num_frames=1200,
                        grip_ke=80.0, grip_kd=2.0, grip_effort=5.0, ik_iters=30)
    parser.add_argument("--expect-grasp-failure", action="store_true",
                        help="With --test, require completed motion but <2cm lift/transport (low-force control).")
    # Reused viewer/common flags only; replace prismatic-drive unit descriptions.
    for action in parser._actions:
        if action.dest == "grip_effort":
            action.help = "Single master revolute torque cap in Nm (NOT finger force in N), <=10."
        elif action.dest == "grip_ke":
            action.help = "Master angular stiffness, Nm/rad. Followers have no drive."
        elif action.dest == "grip_kd":
            action.help = "Master angular damping, Nm s/rad."
    return parser


if __name__ == "__main__":
    args = create_parser().parse_args()
    viewer = init_viewer(args)
    try:
        example = Robotiq2F140PhysicalExample(viewer, args)
        while viewer.is_running() and example.episode_steps < args.num_frames:
            if viewer.should_step():
                example.step()
            example.render()
        if args.test:
            example.test_final()
    finally:
        viewer.close()