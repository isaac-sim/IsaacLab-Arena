#!/usr/bin/env python3
"""Fast, headless PGC140 asset/geometry regressions (no Kit application)."""

from __future__ import annotations

import hashlib
import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp
from pxr import Usd, UsdPhysics, UsdUtils

import newton

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "examples/examples_teleop/assets/abb_irb1200_pgc140"


class PGC140AssetTests(unittest.TestCase):
    def test_vendor_files_match_manifest(self):
        manifest = json.loads((ASSETS / "source_manifest.json").read_text())
        self.assertEqual(manifest["commit"], "f59f9c2f4bc8eb116448b1d798791424bf64e337")
        for name, metadata in manifest["files"].items():
            self.assertEqual(hashlib.sha256((ASSETS / "vendor" / name).read_bytes()).hexdigest(), metadata["sha256"])
        vendor = ET.parse(ASSETS / "vendor/urdf/dh_pgc140_urdf.urdf").getroot()
        self.assertEqual(len(vendor.findall(".//mimic")), 1)
        normalized = ET.parse(ASSETS / "abb_irb1200_pgc140.urdf").getroot()
        self.assertEqual(len(normalized.findall(".//mimic")), 0)
        self.assertEqual(len(normalized.findall("joint[@type='revolute']")), 6)
        self.assertEqual(len(normalized.findall("joint[@type='prismatic']")), 2)

    def test_both_usd_files_are_resolvable(self):
        for relative in ("gripper/pgc140_offset_fingers.usda", "combined/abb_irb1200_pgc140.usda"):
            path = ASSETS / relative
            stage = Usd.Stage.Open(str(path))
            self.assertTrue(stage.GetDefaultPrim().IsValid())
            joints = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.PrismaticJoint)]
            self.assertEqual(len(joints), 2)
            _, _, unresolved = UsdUtils.ComputeAllDependencies(str(path))
            self.assertEqual(unresolved, [])

    def test_fk_aperture_tcp_and_mass(self):
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(str(ASSETS / "combined/abb_irb1200_pgc140.usda"), floating=False,
                        enable_self_collisions=False, collapse_fixed_joints=False, load_visual_shapes=False)
        model = builder.finalize()
        self.assertEqual(model.joint_dof_count, 8)
        self.assertEqual(model.constraint_mimic_count, 0)
        self.assertTrue(np.isfinite(model.body_mass.numpy()).all())
        labels = [name.rsplit("/", 1)[-1] for name in model.body_label]
        flange = labels.index("link_6")
        pads = [labels.index("pgc140_pad1"), labels.index("pgc140_pad2")]
        gripper_bodies = [i for i, name in enumerate(labels) if name.startswith("pgc140_")]
        self.assertAlmostEqual(float(model.body_mass.numpy()[gripper_bodies].sum()), 1.08, places=5)
        state = model.state()
        for displacement, expected_width in ((0.0, 0.090), (0.0125, 0.065), (0.025, 0.040)):
            q = model.joint_q.numpy()
            q[-2:] = displacement
            newton.eval_fk(model, wp.array(q, dtype=float), model.joint_qd, state)
            poses = state.body_q.numpy()
            inverse = wp.transform_inverse(wp.transform(*poses[flange]))
            local = np.array([wp.transform_point(inverse, wp.vec3(*poses[pad, :3])) for pad in pads])
            np.testing.assert_allclose(local.mean(axis=0), [0.130, 0, 0], atol=1e-6)
            self.assertAlmostEqual(float(abs(local[0, 1] - local[1, 1]) - 0.008), expected_width, places=6)
        # The fingertips used by the controller really are authored USD boxes.
        for pad in pads:
            colliders = [i for i, body in enumerate(builder.shape_body)
                         if body == pad and builder.shape_flags[i] & int(newton.ShapeFlags.COLLIDE_SHAPES)]
            self.assertEqual(len(colliders), 2)  # rubber + physical bracket
            self.assertTrue(all(builder.shape_type[i] == newton.GeoType.BOX for i in colliders))


if __name__ == "__main__":
    unittest.main(verbosity=2)