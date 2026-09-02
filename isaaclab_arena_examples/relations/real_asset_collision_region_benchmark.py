# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare mesh, AABB, and max-XY-radius feasible regions for real assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numpy as np
import platform
import scipy
from dataclasses import asdict, dataclass
from pathlib import Path
from scipy.signal import fftconvolve

from isaaclab_arena.relations.benchmark.provenance import collect_source_revision
from isaaclab_arena_examples.relations.obstacle_layout_distribution_benchmark import (
    OBSTACLES,
    TABLE_BOUNDS,
    _source_revision,
)

DEFAULT_ASSETS = ("beer_bottle", "sweet_potato", "jug")
DEFAULT_YAWS_DEG = (0.0, 45.0, 90.0)
REPRESENTATIONS = ("mesh", "aabb", "max-xy-radius-circle")


@dataclass(frozen=True)
class RegionMeasurement:
    """One representation's feasible-region agreement with projected mesh."""

    asset: str
    yaw_deg: float
    representation: str
    feasible_area_fraction: float
    mesh_feasible_recall: float
    representation_feasible_precision: float
    mesh_region_iou: float
    mesh_feasible_missed_fraction: float
    representation_feasible_invalid_fraction: float


def _parse_strings(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    assert values
    return values


def _parse_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(","))
    assert values
    return values


def _rotate_mesh(mesh, yaw_deg: float):
    import trimesh

    rotated = mesh.copy()
    rotated.apply_transform(trimesh.transformations.rotation_matrix(math.radians(yaw_deg), (0.0, 0.0, 1.0)))
    return rotated


def _offset_footprint(offsets_xy: np.ndarray, pitch_m: float) -> np.ndarray:
    """Rasterize footprint points around their AABB center."""
    center = (offsets_xy.min(axis=0) + offsets_xy.max(axis=0)) / 2
    indices = np.floor((offsets_xy - center) / pitch_m + 0.5).astype(int)
    radius_x = int(np.max(np.abs(indices[:, 0])))
    radius_y = int(np.max(np.abs(indices[:, 1])))
    footprint = np.zeros((2 * radius_y + 1, 2 * radius_x + 1), dtype=bool)
    footprint[indices[:, 1] + radius_y, indices[:, 0] + radius_x] = True
    return footprint


def _mesh_footprint(mesh, pitch_m: float) -> np.ndarray:
    voxels = mesh.voxelized(pitch_m).fill()
    return _offset_footprint(np.asarray(voxels.points)[:, :2], pitch_m)


def _primitive_footprint(representation: str, width: float, depth: float, yaw_deg: float, pitch_m: float) -> np.ndarray:
    if representation == "aabb":
        yaw = math.radians(yaw_deg)
        projected_width = abs(width * math.cos(yaw)) + abs(depth * math.sin(yaw))
        projected_depth = abs(width * math.sin(yaw)) + abs(depth * math.cos(yaw))
        radius_x = max(0, round(projected_width / (2 * pitch_m)))
        radius_y = max(0, round(projected_depth / (2 * pitch_m)))
        return np.ones((2 * radius_y + 1, 2 * radius_x + 1), dtype=bool)

    assert representation == "max-xy-radius-circle"
    radius = max(width, depth) / 2
    pixel_radius = max(0, round(radius / pitch_m))
    y, x = np.mgrid[-pixel_radius : pixel_radius + 1, -pixel_radius : pixel_radius + 1]
    return x**2 + y**2 <= (radius / pitch_m) ** 2


def _obstacle_mask(pitch_m: float) -> np.ndarray:
    xmin, xmax, ymin, ymax = TABLE_BOUNDS
    x = np.arange(xmin + pitch_m / 2, xmax, pitch_m)
    y = np.arange(ymin + pitch_m / 2, ymax, pitch_m)
    xx, yy = np.meshgrid(x, y)
    occupied = np.zeros_like(xx, dtype=bool)
    for obstacle in OBSTACLES:
        occupied |= (np.abs(xx - obstacle.x) <= obstacle.width / 2) & (np.abs(yy - obstacle.y) <= obstacle.depth / 2)
    return occupied


def _feasible_centers(footprint: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    """Compute collision-free center cells, including table containment."""
    overlap = fftconvolve(occupied.astype(np.float32), footprint[::-1, ::-1].astype(np.float32), mode="same")
    feasible = overlap < 0.5
    rows, columns = np.nonzero(footprint)
    center_y, center_x = np.asarray(footprint.shape) // 2
    min_y, max_y = int(rows.min() - center_y), int(rows.max() - center_y)
    min_x, max_x = int(columns.min() - center_x), int(columns.max() - center_x)
    contained = np.zeros_like(feasible)
    contained[-min_y : feasible.shape[0] - max_y, -min_x : feasible.shape[1] - max_x] = True
    return feasible & contained


def _footprints(mesh, width: float, depth: float, yaw_deg: float, pitch_m: float) -> dict[str, np.ndarray]:
    return {
        "mesh": _mesh_footprint(_rotate_mesh(mesh, yaw_deg), pitch_m),
        "aabb": _primitive_footprint("aabb", width, depth, yaw_deg, pitch_m),
        "max-xy-radius-circle": _primitive_footprint("max-xy-radius-circle", width, depth, yaw_deg, pitch_m),
    }


def _measurement(
    asset_name: str,
    yaw_deg: float,
    representation: str,
    region: np.ndarray,
    mesh_region: np.ndarray,
) -> RegionMeasurement:
    intersection = np.count_nonzero(region & mesh_region)
    union = np.count_nonzero(region | mesh_region)
    region_count = np.count_nonzero(region)
    mesh_count = np.count_nonzero(mesh_region)
    false_exclusion = np.count_nonzero(mesh_region & ~region)
    false_acceptance = np.count_nonzero(region & ~mesh_region)
    return RegionMeasurement(
        asset=asset_name,
        yaw_deg=yaw_deg,
        representation=representation,
        feasible_area_fraction=region_count / region.size,
        mesh_feasible_recall=intersection / mesh_count,
        representation_feasible_precision=intersection / region_count,
        mesh_region_iou=intersection / union,
        mesh_feasible_missed_fraction=false_exclusion / mesh_count,
        representation_feasible_invalid_fraction=false_acceptance / region_count,
    )


def _mesh_sha256(mesh) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(mesh.vertices).tobytes())
    digest.update(np.ascontiguousarray(mesh.faces).tobytes())
    return digest.hexdigest()


def _file_sha256(path: str) -> str | None:
    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_plot(
    masks: dict[tuple[str, float, str], np.ndarray],
    assets: tuple[str, ...],
    yaw_deg: float,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(assets), len(REPRESENTATIONS), figsize=(9, 2.7 * len(assets)), squeeze=False)
    for row, asset_name in enumerate(assets):
        for column, representation in enumerate(REPRESENTATIONS):
            axis = axes[row, column]
            axis.imshow(
                masks[(asset_name, yaw_deg, representation)],
                origin="lower",
                extent=TABLE_BOUNDS,
                vmin=0,
                vmax=1,
                cmap="Blues",
            )
            if row == 0:
                axis.set_title(representation.replace("-", " "))
            if column == 0:
                axis.set_ylabel(f"{asset_name}\nY position (m)")
            if row == len(assets) - 1:
                axis.set_xlabel("X position (m)")
    figure.suptitle(f"Collision-free center regions at {yaw_deg:g}° yaw")
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=_parse_strings, default=DEFAULT_ASSETS)
    parser.add_argument("--yaws-deg", type=_parse_floats, default=DEFAULT_YAWS_DEG)
    parser.add_argument("--pitch-m", type=float, default=0.003)
    parser.add_argument("--plot-yaw-deg", type=float, default=45.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    assert args.pitch_m > 0.0
    assert args.plot_yaw_deg in args.yaws_deg

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.relations.warp_mesh_manager import WarpMeshAndSphereCache

    registry = AssetRegistry()
    mesh_cache = WarpMeshAndSphereCache(device="cpu")
    occupied = _obstacle_mask(args.pitch_m)
    masks = {}
    measurements = []
    asset_metadata = []
    for asset_name in args.assets:
        asset = registry.get_asset_by_name(asset_name)()
        mesh = mesh_cache.get_collision_mesh_or_raise(asset)
        bounds = np.asarray(mesh.bounds)
        width, depth = bounds[1, :2] - bounds[0, :2]
        usd_path = str(asset.usd_path)
        asset_metadata.append({
            "name": asset_name,
            "usd_path": usd_path,
            "usd_sha256": _file_sha256(usd_path),
            "extracted_mesh_sha256": _mesh_sha256(mesh),
            "mesh_vertices": len(mesh.vertices),
            "mesh_faces": len(mesh.faces),
            "mesh_watertight": bool(mesh.is_watertight),
            "xy_aabb_m": [float(width), float(depth)],
        })
        for yaw_deg in args.yaws_deg:
            footprints = _footprints(mesh, width, depth, yaw_deg, args.pitch_m)
            regions = {
                representation: _feasible_centers(footprint, occupied)
                for representation, footprint in footprints.items()
            }
            mesh_region = regions["mesh"]
            for representation, region in regions.items():
                masks[(asset_name, yaw_deg, representation)] = region
                measurements.append(_measurement(asset_name, yaw_deg, representation, region, mesh_region))

    benchmark_root = Path(__file__).resolve().parents[2]
    payload = {
        "schema_version": 1,
        "experiment": "real-asset-collision-region-coverage",
        "benchmark_revision": collect_source_revision(benchmark_root),
        "arena_source_revision": _source_revision(benchmark_root),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "trimesh": __import__("trimesh").__version__,
        },
        "table_bounds": TABLE_BOUNDS,
        "obstacles": [asdict(obstacle) for obstacle in OBSTACLES],
        "pitch_m": args.pitch_m,
        "yaws_deg": args.yaws_deg,
        "representations": REPRESENTATIONS,
        "mesh_reference": "projected filled mesh voxelization",
        "coordinate_definition": "translation of the footprint AABB center, not the USD root pose",
        "grid_phase": "nearest center with half-up tie breaking",
        "robolab_circle_formula": "radius=max(xy_aabb_width, xy_aabb_depth)/2",
        "assets": asset_metadata,
        "measurements": [asdict(measurement) for measurement in measurements],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.plot is not None:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        _write_plot(masks, args.assets, args.plot_yaw_deg, args.plot)
    for measurement in measurements:
        if measurement.representation != "mesh":
            print(
                f"{measurement.asset:14} yaw={measurement.yaw_deg:5.1f} "
                f"{measurement.representation:20} recall={measurement.mesh_feasible_recall:.3f} "
                f"precision={measurement.representation_feasible_precision:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
