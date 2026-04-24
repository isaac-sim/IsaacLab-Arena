"""Generate metadata for all scene .usda files.

Adapted from RoboLab's generate_scene_metadata.py.
Scans scene USDs, extracts object info, saves centralized metadata.

Usage (inside Isaac Sim):
    python isaaclab_arena/scene_gen/generate_scene_metadata.py \
        --scene-folder /path/to/scenes \
        --output-folder /path/to/output \
        --generate-images
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any


def count_objects_in_usd(usda_path: str) -> tuple[int, list[str]]:
    """Count objects in a .usda file by parsing the text (no Isaac Sim needed)."""
    content = Path(usda_path).read_text()
    # Find all def "name" prims under /world
    objects = re.findall(r'def\s+(?:Xform\s+)?"([\w_]+)"', content)
    # Filter out infrastructure prims
    skip = {'world', 'groundplane', 'physicsscene', 'physicsmaterial',
            'collisionmesh', 'collisionplane', 'looks', 'physicsmaterials',
            'table', 'franka_table'}
    objects = [o for o in objects if o.lower() not in skip]
    return len(objects), objects


def check_ground_planes(usda_path: str) -> int:
    """Count GroundPlane prims (payloads can inject duplicates)."""
    content = Path(usda_path).read_text()
    return len(re.findall(r'def\s+Xform\s+"GroundPlane"', content))


def generate_metadata_for_folder(
    scene_folder: str,
    output_folder: str,
    ignore_files: list[str] | None = None,
) -> dict[str, Any]:
    """Generate metadata for all .usda files in a folder.

    Args:
        scene_folder: Directory with .usda scene files.
        output_folder: Where to save metadata JSON/CSV.
        ignore_files: Filenames to skip.

    Returns:
        Dict of {filename: {objects, ground_planes, file_size}}.
    """
    os.makedirs(output_folder, exist_ok=True)

    usd_files = sorted([
        os.path.join(scene_folder, f)
        for f in os.listdir(scene_folder)
        if f.endswith((".usda", ".usd", ".usdc"))
    ])

    if ignore_files:
        usd_files = [f for f in usd_files if os.path.basename(f) not in ignore_files]

    print(f"[Metadata] Found {len(usd_files)} USD files in {scene_folder}")

    results = {}

    for usd_file in usd_files:
        filename = os.path.basename(usd_file)
        obj_count, obj_names = count_objects_in_usd(usd_file)
        gp_count = check_ground_planes(usd_file)
        file_size = os.path.getsize(usd_file)

        results[filename] = {
            "num_objects": obj_count,
            "objects": obj_names,
            "ground_planes": gp_count,
            "file_size_kb": file_size // 1024,
        }

        gp_status = "OK" if gp_count <= 1 else f"WARN({gp_count})"
        print(f"  {filename}: {obj_count} objects, GP={gp_status}, {file_size//1024}KB")

    # Save JSON
    json_path = os.path.join(output_folder, "scene_metadata.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Metadata] JSON saved: {json_path}")

    # Save CSV
    csv_path = os.path.join(output_folder, "scene_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "num_objects", "objects", "ground_planes", "size_kb"])
        for filename, info in results.items():
            writer.writerow([
                filename,
                info["num_objects"],
                ", ".join(info["objects"]),
                info["ground_planes"],
                info["file_size_kb"],
            ])
    print(f"[Metadata] CSV saved: {csv_path}")

    # Save statistics
    if results:
        stats = {
            "total_scenes": len(results),
            "total_objects": sum(r["num_objects"] for r in results.values()),
            "average_objects_per_scene": sum(r["num_objects"] for r in results.values()) / len(results),
            "total_unique_objects": len(set(
                obj for r in results.values() for obj in r["objects"]
            )),
            "scenes_with_ground_plane_issues": sum(
                1 for r in results.values() if r["ground_planes"] > 1
            ),
        }
        stats_path = os.path.join(output_folder, "scene_statistics.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[Metadata] Statistics: {stats}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate metadata for scene USD files")
    parser.add_argument("--scene-folder", required=True, help="Folder with .usda files")
    parser.add_argument("--output-folder", required=True, help="Folder for metadata output")
    parser.add_argument("--generate-images", action="store_true", help="Also render screenshots")
    parser.add_argument("--ignore-files", nargs="*", default=["base_empty.usda"])
    args = parser.parse_args()

    # Metadata extraction first (text parsing, no Isaac Sim needed)
    generate_metadata_for_folder(
        args.scene_folder, args.output_folder,
        ignore_files=args.ignore_files,
    )

    if args.generate_images:
        # Non-headless required — viewport capture needs an active renderer
        from isaaclab.app import AppLauncher
        app_launcher = AppLauncher(headless=False)

        from isaaclab_arena.scene_gen.render_utils import render_all_scenes
        images_dir = os.path.join(args.scene_folder, "images")
        render_all_scenes(args.scene_folder, images_dir, view = 'front', ignore_files=args.ignore_files)

        app_launcher.app.close()
