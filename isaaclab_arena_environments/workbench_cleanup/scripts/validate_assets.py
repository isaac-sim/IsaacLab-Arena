# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate generated packages and write scene copies pointing to the new assets."""

import argparse
import copy
import json
import numpy as np
import yaml
from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab_arena_environments.workbench_cleanup.scripts.asset_io import bounds_m, file_sha256, validate_package


def check_scene(scene, packages):
    """Return a scene copy with generated USD paths after checking its object references and joint names."""
    scene = copy.deepcopy(scene)
    nodes = {}
    for node in [scene.get("background", {}), *scene.get("objects", [])]:
        params = node.get("params", {})
        if "usd_path" not in params:
            continue
        filename = Path(params["usd_path"]).name
        assert filename in packages, f"No generated package for {filename}"
        path = packages[filename]
        stage = Usd.Stage.Open(str(path))
        params["usd_path"] = str(path)
        nodes[node["id"]] = (stage, stage.GetDefaultPrim(), params.get("scale", [1, 1, 1]))
    pending = list(scene.get("object_references", []))
    while pending:
        remaining = []
        for node in pending:
            if node["parent_id"] not in nodes:
                remaining.append(node)
                continue
            stage, parent, scale = nodes[node["parent_id"]]
            prim = stage.GetPrimAtPath(parent.GetPath().AppendPath(node["prim_path"]))
            assert prim and prim.IsActive(), f"Missing scene reference: {node['id']} ({node['prim_path']})"
            if node["object_type"] == "rigid":
                assert prim.HasAPI(UsdPhysics.RigidBodyAPI), f"Not a rigid body: {prim.GetPath()}"
            if node["object_type"] == "articulation":
                assert prim.HasAPI(UsdPhysics.ArticulationRootAPI), f"Not an articulation: {prim.GetPath()}"
            nodes[node["id"]] = (stage, prim, scale)
        assert len(remaining) < len(pending), f"Unresolved or cyclic object references: {remaining}"
        pending = remaining
    for node in [*scene.get("objects", []), *scene.get("object_references", [])]:
        if node["id"] not in nodes:
            continue
        _, prim, scale = nodes[node["id"]]
        params = node.get("params", {})
        joints = {}
        for child in Usd.PrimRange(prim):
            if child.IsA(UsdPhysics.PrismaticJoint) or child.IsA(UsdPhysics.RevoluteJoint):
                joints[child.GetName()] = child
        for field in ("openable_joint_name", "pressable_joint_name"):
            if field in params:
                assert params[field] in joints, f"Missing {field}: {params[field]}"
        for name, value in params.get("initial_joint_positions", {}).items():
            assert name in joints, f"Missing initial joint: {name}"
            joint = joints[name]
            if joint.IsA(UsdPhysics.PrismaticJoint):
                assert np.allclose(scale, scale[0]), "Prismatic validation requires uniform scene scale"
                factor = scale[0] * UsdGeom.GetStageMetersPerUnit(joint.GetStage())
            else:
                factor = np.pi / 180.0
            lower = joint.GetAttribute("physics:lowerLimit").Get() * factor
            upper = joint.GetAttribute("physics:upperLimit").Get() * factor
            assert lower - 1e-6 <= value <= upper + 1e-6, f"Initial joint {name}={value} outside [{lower}, {upper}]"
    return scene, len(nodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, help="Optional read-only directory of reference assets.")
    parser.add_argument("--scene", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for reports and scene copies.")
    args = parser.parse_args()
    assert not args.output_dir.exists(), "Choose a new validation output directory"
    packages = {}
    report = {"assets": {}, "scenes": {}}
    for manifest_path in sorted(args.assets_dir.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        path = (manifest_path.parent / manifest["output"]).resolve()
        assert path.name not in packages, f"Ambiguous output filename: {path.name}"
        packages[path.name] = path
        assert file_sha256(path) == manifest["sha256"], f"Generated USD changed: {path}"
        checks = validate_package(path)
        if args.reference_dir:
            reference = args.reference_dir / path.name
            assert reference.is_file(), f"Missing reference: {reference}"
            before = file_sha256(reference)
            checks["reference_bounds_m"] = bounds_m(Usd.Stage.Open(str(reference)))
            np.testing.assert_allclose(checks["bounds_m"], checks["reference_bounds_m"], atol=2e-5, rtol=0)
            assert file_sha256(reference) == before, f"Reference changed: {reference}"
            checks["reference_sha256"] = before
        report["assets"][manifest["id"]] = checks
    assert packages, "No generated manifests found"
    scenes = {}
    for path in args.scene:
        assert path.name not in scenes, f"Duplicate scene filename: {path.name}"
        scenes[path.name], count = check_scene(yaml.safe_load(path.read_text()), packages)
        report["scenes"][path.name] = {"source": str(path), "checked_asset_nodes": count}
    args.output_dir.mkdir(parents=True)
    for name, scene in scenes.items():
        (args.output_dir / name).write_text(yaml.safe_dump(scene, sort_keys=False))
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Validated {len(packages)} packages and {len(scenes)} scene specifications: {args.output_dir}")


if __name__ == "__main__":
    main()
