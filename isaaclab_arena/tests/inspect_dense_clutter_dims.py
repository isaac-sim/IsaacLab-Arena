#!/usr/bin/env python3
"""Inspect bounding box & mesh dimensions for objects in the dense_clutter environment.

Run inside the Docker container:
  /isaac-sim/python.sh isaaclab_arena/tests/inspect_dense_clutter_dims.py
"""

import math
import sys

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import torch
from isaaclab_arena.utils.usd_helpers import compute_local_bounding_box_from_usd, extract_mesh_from_usd
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


def quat_from_pitch(pitch_rad: float) -> tuple[float, float, float, float]:
    """Pitch rotation -> quaternion (x, y, z, w)."""
    half = pitch_rad / 2.0
    return (0.0, math.sin(half), 0.0, math.cos(half))


def main():
    from isaaclab_arena.assets.asset_registry import AssetRegistry
    registry = AssetRegistry()

    # Objects and their pitch angles
    objects_info = [
        ("red_container", (0.5, 0.5, 0.5), 0.0),
        ("red_container", (0.5, 0.5, 2.0), 0.0),  # taller container
        ("red_container", (0.5, 0.5, 3.0), 0.0),  # even taller
        ("mustard_bottle", (1.0, 1.0, 1.0), math.pi / 6),
        ("beer_bottle", (1.2, 1.2, 1.2), -math.pi / 5),
        ("tomato_soup_can", (1.0, 1.0, 1.0), math.pi / 7),
        ("cracker_box", (1.0, 1.0, 1.0), 0.0),
        ("sugar_box", (1.0, 1.0, 1.0), 0.0),
    ]

    print("=" * 110)
    print(f"{'Name':<25} {'Scale':<18} {'AABB (x, y, z) m':<30} {'Rotated AABB (x, y, z) m':<30} {'Pitch':<8}")
    print("=" * 110)

    for name, scale, pitch_rad in objects_info:
        asset_cls = registry.get_asset_by_name(name)
        usd_path = asset_cls.usd_path
        bbox = compute_local_bounding_box_from_usd(usd_path, scale)
        sz = bbox.size[0].tolist()
        sz_str = f"({sz[0]:.4f}, {sz[1]:.4f}, {sz[2]:.4f})"
        scale_str = f"({scale[0]}, {scale[1]}, {scale[2]})"

        if pitch_rad != 0.0:
            quat = quat_from_pitch(pitch_rad)
            rbbox = bbox.rotated(quat)
            rsz = rbbox.size[0].tolist()
            rsz_str = f"({rsz[0]:.4f}, {rsz[1]:.4f}, {rsz[2]:.4f})"
        else:
            rsz_str = "—"

        pitch_str = f"{math.degrees(pitch_rad):.0f}°"
        print(f"{name:<25} {scale_str:<18} {sz_str:<30} {rsz_str:<30} {pitch_str:<8}")

    # Mesh extents
    print("\n" + "=" * 110)
    print("Mesh extents (actual geometry):")
    print("=" * 110)
    for name, scale, _ in objects_info:
        asset_cls = registry.get_asset_by_name(name)
        usd_path = asset_cls.usd_path
        try:
            mesh = extract_mesh_from_usd(usd_path, scale)
            bounds = mesh.bounds  # (2, 3) array: [[xmin, ymin, zmin], [xmax, ymax, zmax]]
            sz = bounds[1] - bounds[0]
            scale_str = f"({scale[0]}, {scale[1]}, {scale[2]})"
            print(f"  {name:<25} scale={scale_str:<18} mesh size: ({sz[0]:.4f}, {sz[1]:.4f}, {sz[2]:.4f})")
        except Exception as e:
            print(f"  {name:<25} mesh error: {e}")

    # Analysis: can a tilted beer bottle sit at the edge of the container without mesh collision?
    print("\n" + "=" * 110)
    print("ANALYSIS: Beer bottle tilted -36° next to red_container at various Z scales")
    print("  Beer bottle width (X): 0.048m -> rotated AABB X: 0.159m")
    print("  If bottle center is at container X-edge, rotated AABB extends ~0.08m into container")
    print("  But the actual cylinder (4.8cm wide) tilted 36° from vertical won't reach a short container")
    print("=" * 110)

    app.close()


if __name__ == "__main__":
    main()
