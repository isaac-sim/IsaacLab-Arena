# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Inspect a recipe's source or intermediate geometry to choose component IDs for new assets."""

import argparse
import json
import yaml
from pathlib import Path

from isaaclab_arena_environments.workbench_cleanup.scripts.asset_io import bounds_m, open_asset, source_file
from isaaclab_arena_environments.workbench_cleanup.scripts.asset_operations import apply_operations
from isaaclab_arena_environments.workbench_cleanup.scripts.mesh_partition import inspect_mesh_components


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--after", type=int, default=0, help="Apply the first N operations before inspecting.")
    parser.add_argument("--mesh", help="Inspect connected components of this asset-relative mesh.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), f"Inspection output already exists: {args.output}"
    spec = yaml.safe_load(args.spec.read_text())
    assert 0 <= args.after <= len(spec.get("operations", [])), "Invalid operation count"
    asset = spec["asset"]
    source = source_file(asset["usd_path"], args.spec.resolve().parent)
    stage, _ = open_asset(source, asset.get("variants", {}))
    apply_operations(stage, spec.get("operations", [])[: args.after])
    report = {"bounds_m": bounds_m(stage), "prims": []}
    for prim in stage.Traverse():
        report["prims"].append(
            {"path": str(prim.GetPath()), "type": prim.GetTypeName(), "schemas": prim.GetAppliedSchemas()}
        )
    if args.mesh:
        report["mesh"] = inspect_mesh_components(stage, args.mesh)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
