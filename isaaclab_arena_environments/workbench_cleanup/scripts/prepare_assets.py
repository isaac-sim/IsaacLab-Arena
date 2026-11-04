# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build portable USD assets from individual YAML recipes without editing their sources."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import yaml
from pathlib import Path

from pxr import Tf

from isaaclab_arena_environments.workbench_cleanup.scripts.asset_io import (
    copy_dependencies,
    file_sha256,
    open_asset,
    source_file,
    validate_package,
)
from isaaclab_arena_environments.workbench_cleanup.scripts.asset_operations import apply_operations, asset_path


def build_asset(spec_path: Path, output_dir: Path) -> dict:
    """Build one YAML recipe into a new directory named after its asset ID.

    Args:
        spec_path: One asset specification, with asset, operations, output and optional validation fields.
        output_dir: Parent directory for generated asset packages. Existing packages are never replaced.

    Returns:
        Provenance, operation results, hashes, and portable-package validation results.
    """
    spec_path = spec_path.expanduser().resolve()
    spec = yaml.safe_load(spec_path.read_text())
    assert isinstance(spec, dict) and set(spec) <= {
        "asset",
        "operations",
        "output",
        "validation",
    }, "Invalid recipe fields"
    asset = spec["asset"]
    assert set(asset) <= {"id", "usd_path", "variants"}, "Invalid asset fields"
    name = asset["id"]
    assert isinstance(name, str) and Tf.IsValidIdentifier(name), "Asset ID must be a USD identifier"
    filename = spec["output"]
    assert Path(filename).name == filename and Path(filename).suffix in (
        ".usd",
        ".usda",
        ".usdc",
    ), "Output must be a USD filename"
    source = source_file(asset["usd_path"], spec_path.parent)
    output_dir = output_dir.expanduser().resolve()
    destination = output_dir / name
    assert not destination.exists(), f"Output package already exists: {destination}; choose a new output directory"
    assert not source.resolve().is_relative_to(destination), "Output must be separate from the source package"
    stage, source_hashes = open_asset(source, asset.get("variants", {}))
    reports = apply_operations(stage, spec.get("operations", []))
    checks = spec.get("validation", {})
    assert set(checks) <= {"required_prims"}, "Unknown validation fields"
    for path in checks.get("required_prims", []):
        prim = stage.GetPrimAtPath(asset_path(stage, path))
        assert prim and prim.IsActive(), f"Missing required scene prim: {path}"
    for prim in stage.TraverseAll():
        # These original identifiers describe provenance, not a runtime file dependency.
        info = prim.GetAssetInfo()
        if "identifier" in info:
            del info["identifier"]
            prim.SetAssetInfo(info)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=output_dir) as temporary:
        folder = Path(temporary)
        layer = stage.Flatten()
        dependencies = copy_dependencies(layer, folder)
        output = folder / filename
        assert layer.Export(str(output)), f"Failed to export {output}"
        validation = validate_package(output)
        assert all(
            file_sha256(Path(path)) == digest for path, digest in source_hashes.items()
        ), "A source layer changed"
        report = {
            "id": name,
            "source": asset["usd_path"],
            "source_layers_sha256": source_hashes,
            "spec_sha256": file_sha256(spec_path),
            "output": filename,
            "sha256": file_sha256(output),
            "operations": reports,
            "dependencies": dependencies,
            "validation": validation,
        }
        (folder / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        shutil.copy2(spec_path, folder / "asset_spec.yaml")
        # Publish only a complete, validated package; never overwrite an existing directory.
        assert not destination.exists(), f"Output package was created concurrently: {destination}"
        folder.rename(destination)
    print(f"Generated {destination / filename}", flush=True)
    return report


def main() -> None:
    """Build one or more individual asset specifications."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for spec in args.spec:
        build_asset(spec, args.output_dir)


if __name__ == "__main__":
    main()
