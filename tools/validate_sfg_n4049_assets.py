"""Validate CAD conversion and the isolated continuous soft-finger actuator."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from pxr import Usd, UsdPhysics
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples/examples_teleop"
ASSETS = EXAMPLES / "assets/abb_irb1200_sfg_n4049"


def validate_assets():
    manifest = json.loads((ASSETS / "source_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        source = EXAMPLES / "SFG-FNC3-N4049" / name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected["sha256"]
    stage = Usd.Stage.Open(str(ASSETS / "abb_irb1200_sfg_n4049.usda"))
    assert stage.GetMetadata("metersPerUnit") == 1.
    assert len([p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]) == 6
    assert not [p for p in stage.Traverse() if p.IsA(UsdPhysics.PrismaticJoint)]
    fingers = [p for p in stage.Traverse() if p.GetAttribute("srt:deformableVisual").Get()]
    assert len(fingers) == 3
    assert all(not p.HasAPI(UsdPhysics.CollisionAPI) and not p.HasAPI(UsdPhysics.RigidBodyAPI) for p in fingers)
    mesh = np.load(ASSETS / "soft_fingers.npz")
    cad_stage = Usd.Stage.Open(str(ASSETS / "sfg_n4049_cad.usda"))
    results = []
    for i in range(3):
        points, tets, roots = mesh[f"points_{i}"], mesh[f"tets_{i}"], mesh[f"roots_{i}"]
        assert len(roots) > 3 and len(roots) < len(points) / 4
        assert tets.min() >= 0 and tets.max() < len(points)
        xyz = points[tets]
        volumes = np.linalg.det(np.stack([xyz[:, j] - xyz[:, 0] for j in (1, 2, 3)], axis=-1)) / 6
        assert np.all(volumes > 0)
        assert np.ptp(points[:, 2]) > .05
        faces = np.concatenate([tets[:, ids] for ids in ([0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3])])
        faces, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
        surface = points[np.unique(faces[counts == 1])]
        cad = np.asarray(cad_stage.GetPrimAtPath(f"/SFG/soft_finger_{i}").GetAttribute("points").Get())
        cad_to_fem = cKDTree(surface).query(cad)[0]
        fem_to_cad = cKDTree(cad).query(surface)[0]
        # Vertex sampling distances are NOT a continuous-surface Hausdorff
        # bound; internal grooves and uneven tessellation affect this audit.
        results.append(dict(finger=i, particles=len(points), tets=len(tets), volume_m3=float(volumes.sum()),
                    sampled_cad_to_fem_vertex_max_m=float(cad_to_fem.max()),
                    sampled_cad_to_fem_vertex_p95_m=float(np.percentile(cad_to_fem, 95)),
                    sampled_fem_to_cad_vertex_max_m=float(fem_to_cad.max()),
                    sampled_fem_to_cad_vertex_p95_m=float(np.percentile(fem_to_cad, 95))))
    result = {"passed": True, "revolute_joints": 6, "rigid_finger_colliders": 0, "fingers": results}
    (ASSETS / "validation").mkdir(exist_ok=True)
    (ASSETS / "validation/assets.json").write_text(json.dumps(result, indent=2) + "\n")
    print("[ASSETS] " + json.dumps(result), flush=True)


def validate_actuator(substeps, iterations):
    sys.path.insert(0, str(EXAMPLES))
    import warp as wp
    from example41_newton_abb_irb1200_sfg_n4049_soft_grasp import SFGSoftGraspExample, create_parser, init_viewer

    class FreeBend(SFGSoftGraspExample):
        """Keep wrist high above bottle; actuate and recover without object contact."""

        def make_sequence(self):
            return [("rest", .5, self.initial_tcp, 0.), ("bend", 1.5, self.initial_tcp, 1.),
                    ("hold_bend", .5, self.initial_tcp, 1.), ("recover", 1.5, self.initial_tcp, 0.),
                    ("settle", 1., self.initial_tcp, 0.)]

    args = create_parser().parse_args(["--viewer", "null", "--quiet", "--num-frames", "400",
                                     "--substeps", str(substeps), "--iterations", str(iterations)])
    viewer = init_viewer(args)
    try:
        e = FreeBend(viewer, args)
        reference = e.model.tet_poses.numpy().copy()
        e.update_active_reference(0.)
        assert np.max(np.abs(reference - e.model.tet_poses.numpy())) < .01
        assert e.model.body_inv_mass.numpy()[e.bottle] > 0
        assert np.all(e.model.particle_inv_mass.numpy()[:e.finger_particle_count] > 0)
        # Validate exactly opposite spring wrench in an artificial 1 mm offset.
        p = e.state_0.particle_q.numpy()
        p[e.roots[0]] += [0, 0, .001]
        temp = wp.array(p, dtype=wp.vec3)
        from example41_newton_abb_irb1200_sfg_n4049_soft_grasp import root_reaction
        wrench = wp.zeros(e.arm_count, dtype=wp.spatial_vector)
        wp.launch(root_reaction, len(e.roots), inputs=[e.root_ids, e.anchor_ids, temp,
                  e.arm_0.body_q, e.arm_model.body_com, e.ee_index, args.root_ke, wrench])
        np.testing.assert_allclose(wrench.numpy()[e.ee_index, :3], [0, 0, args.root_ke * .001], atol=5e-5)
        while not e.finished:
            e.step()
        q = e.state_0.particle_q.numpy()
        mount = wp.transform_multiply(wp.transform(*e.arm_0.body_q.numpy()[e.ee_index]), wp.transform(wp.vec3(0), e.mount_rot))
        inv = wp.transform_inverse(mount)
        actual = np.array([wp.transform_point(inv, wp.vec3(*v)) for v in q[:e.finger_particle_count]])
        recovered = float(np.max(np.linalg.norm(actual - np.concatenate(e.local_points), axis=1)))
        passed = e.max_tip_bend > .008 and recovered < .004 and e.max_root_error < .004
        result = dict(passed=bool(passed), free_tip_bend_m=e.max_tip_bend, recovery_max_error_m=recovered,
                      max_root_error_m=e.max_root_error, final_root_wrench=e.reaction.numpy()[e.ee_index].tolist(),
                      expected_soft_weight_N=float(e.model.particle_mass.numpy().sum() * 9.81),
                      substeps=substeps, iterations=iterations)
        (ASSETS / "validation/actuator.json").write_text(json.dumps(result, indent=2) + "\n")
        print("[ACTUATOR] " + json.dumps(result), flush=True)
        assert passed, result
    finally:
        viewer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actuator", action="store_true")
    parser.add_argument("--substeps", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    validate_assets()
    if args.actuator:
        validate_actuator(args.substeps, args.iterations)