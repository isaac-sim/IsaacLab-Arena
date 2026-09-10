"""Asset provenance, real linkage FK and passive-follower dynamics regressions.

Run with /isaac-sim/python.sh in the existing Arena container, not host Python.
The FK test assigns configuration ONLY to a scratch state, never a live grasp.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples/examples_teleop"))

import numpy as np
import warp as wp
import newton
from pxr import Usd, UsdPhysics, UsdUtils

from example40_newton_abb_irb1200_robotiq_2f140_physical_grasp import (
    ASSETS, Robotiq2F140PhysicalExample, configure_contact_solver, create_parser, register_newton_usd_schema,
)


class RobotiqAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_newton_usd_schema()
        cls.fixture = Robotiq2F140PhysicalExample.__new__(Robotiq2F140PhysicalExample)
        cls.fixture.args = create_parser().parse_args(["--viewer", "null"])
        cls.fixture.table_height = 0.1
        cls.builder = cls.fixture.build_robot()
        cls.model = cls.builder.finalize()

    def test_vendor_provenance_and_urdf(self):
        manifest = json.loads((ASSETS / "source_manifest.json").read_text())
        self.assertEqual(manifest["commit"], "45196f6558fe8ba9d89bc8a105396c68c3e7e892")
        self.assertEqual(len(manifest["files"]), 17)
        for filename, record in manifest["files"].items():
            self.assertEqual(hashlib.sha256((ASSETS / "vendor" / filename).read_bytes()).hexdigest(), record["sha256"])
            self.assertIn(manifest["commit"], record["url"])
        urdf = ET.parse(ASSETS / "robotiq_2f140.urdf").getroot()
        self.assertEqual(len(urdf.findall("joint[@type='revolute']")), 6)
        self.assertEqual(len(urdf.findall("joint/mimic")), 5)
        self.assertEqual(len(urdf.findall("joint[@type='prismatic']")), 0)
        self.assertEqual(len(urdf.findall(".//collision/geometry/box")), 2)
        for box in urdf.findall(".//collision/geometry/box"):
            np.testing.assert_allclose(np.fromstring(box.get("size"), sep=" "), [0.03, 0.07, 0.0075])

    def test_usd_dependencies_and_mount(self):
        for relative, revolutes in [("gripper/robotiq_2f140.usda", 6), ("assembly.usda", 12),
                                    ("abb_irb1200_robotiq_2f140.usda", 12)]:
            path = ASSETS / relative
            layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
            self.assertFalse(unresolved, unresolved)
            self.assertTrue(layers)
            stage = Usd.Stage.Open(str(path))
            self.assertEqual(sum(p.IsA(UsdPhysics.RevoluteJoint) for p in stage.Traverse()), revolutes)
            self.assertFalse(any(p.IsA(UsdPhysics.PrismaticJoint) for p in stage.Traverse()))
            self.assertEqual(sum(p.HasAPI("NewtonMimicAPI") for p in stage.Traverse()), 5)
            if revolutes == 12:
                self.assertEqual(sum(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in stage.Traverse()), 1)
                mount = UsdPhysics.FixedJoint(stage.GetPrimAtPath("/abb_irb1200_2f140/Physics/robotiq_mount_joint"))
                self.assertTrue(mount)
                self.assertEqual(mount.GetBody0Rel().GetTargets()[0].name, "link_6")
                self.assertEqual(mount.GetBody1Rel().GetTargets()[0].name, "robotiq_arg2f_base_link")
        assembly = Usd.Stage.Open(str(ASSETS / "assembly.usda"))
        references = assembly.GetDefaultPrim().GetMetadata("references").GetAddedOrExplicitItems()
        expected = ROOT / "isaaclab_arena/assets/robots/abb/irb1200_7_70/irb1200_7_70.usda"
        self.assertEqual((ASSETS / references[0].assetPath).resolve(), expected.resolve())

    def test_native_constraints_and_single_actuator(self):
        b = self.builder
        self.assertEqual(b.joint_dof_count, 12)
        self.assertEqual(len(b.constraint_mimic_joint0), 5)
        self.assertTrue(all(b.constraint_mimic_enabled))
        self.assertEqual(sum(mode == newton.JointTargetMode.POSITION for mode in b.joint_target_mode), 7)
        for dof, _, _ in self.fixture.followers:
            self.assertEqual(b.joint_target_mode[dof], newton.JointTargetMode.NONE)
            self.assertEqual(b.joint_target_ke[dof], 0)
            self.assertEqual(b.joint_target_kd[dof], 0)
        self.assertTrue(all(m > 0 for m in b.body_mass))

    def test_fk_aperture_and_variable_tcp(self):
        e, m = self.fixture, self.model
        state = m.state()
        samples = []
        for angle in [0, 0.25, 0.5, 0.7]:
            q = m.joint_q.numpy()
            q[:6], q[e.master_dof] = 0, angle
            for dof, multiplier, offset in e.followers:
                q[dof] = multiplier * angle + offset
            newton.eval_fk(m, wp.array(q, dtype=float), m.joint_qd, state)
            poses = state.body_q.numpy()
            inverse = wp.transform_inverse(wp.transform(*poses[e.ee_index]))
            pads = np.array([wp.transform_point(inverse, wp.vec3(*poses[p, :3])) for p in e.pad_indices])
            tcp = pads.mean(axis=0)
            np.testing.assert_allclose(tcp[1:], 0, atol=1e-6)
            samples.append(dict(master_rad=angle, inner_aperture_m=float(np.linalg.norm(pads[0]-pads[1]) - 0.0075),
                                flange_local_tcp_m=tcp.tolist()))
        np.testing.assert_allclose([s["inner_aperture_m"] for s in samples],
                                   [0.12853467, 0.08676069, 0.03954643, -0.00017608], atol=1e-5)
        np.testing.assert_allclose([s["flange_local_tcp_m"][0] for s in samples],
                                   [0.17696314, 0.19058245, 0.19861083, 0.20066006], atol=1e-5)
        report = dict(samples=samples, imported_gripper_mass_kg=float(sum(self.builder.body_mass[7:])),
                      note="Stock visualization inertias retained; not a calibrated 1kg product mass model.")
        destination = ASSETS / "validation/fk.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")

    def test_single_actuator_unloaded_open_close(self):
        e, m = self.fixture, self.model
        state, next_state, control = m.state(), m.state(), m.control()
        newton.eval_fk(m, m.joint_q, m.joint_qd, state)
        solver = newton.solvers.SolverMuJoCo(m, use_mujoco_cpu=True, use_mujoco_contacts=True,
                                           solver="newton", integrator="implicitfast", iterations=80,
                                           ls_iterations=40, cone="elliptic", impratio=10)
        configure_contact_solver(solver)
        # Each physical motor may have position and velocity actuators in
        # MuJoCo; count unique actuated JOINTS, not the number of PD components.
        motor_joints = np.unique(solver.mj_model.actuator_trnid[:, 0])
        self.assertEqual(len(motor_joints), 7)
        actuated_dofs = solver.mj_model.jnt_dofadr[motor_joints]
        np.testing.assert_array_equal(np.sort(actuated_dofs), list(range(7)))
        contacts = m.contacts()
        max_residual, previous, samples = 0.0, 0.0, []
        for target in [0.65, 0.0]:
            for step in range(1200):
                fraction = min((step + 1) / 800, 1)
                alpha = fraction * fraction * (3 - 2 * fraction)
                targets = control.joint_target_pos.numpy()
                targets[:6] = m.joint_q.numpy()[:6]
                targets[e.master_dof] = previous * (1-alpha) + target * alpha
                control.joint_target_pos.assign(targets)
                state.clear_forces()
                solver.step(state, next_state, control, contacts, 1/600)
                state, next_state = next_state, state
                q = state.joint_q.numpy()
                residual = max(abs(float(q[d] - c*q[e.master_dof] - o)) for d, c, o in e.followers)
                max_residual = max(max_residual, residual)
            self.assertLess(abs(float(q[e.master_dof]) - target), 0.002)
            samples.append(dict(target_rad=target, measured_master_rad=float(q[e.master_dof]),
                                follower_angles_rad=[float(q[d]) for d, _, _ in e.followers]))
            previous = target
        self.assertLess(max_residual, 0.002)
        report = dict(success=True, backend="CPU MuJoCo", active_gripper_motors=1, follower_motors=0,
                      max_mimic_error_rad=max_residual, metric_sample_hz=600, samples=samples)
        destination = ASSETS / "validation/unloaded_open_close.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)