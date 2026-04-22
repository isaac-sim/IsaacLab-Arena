# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate ``object_catalog.json`` from USD files in Arena's asset tree.

Scans Arena's registered USD assets (S3-hosted and Lightwheel cache) and
extracts metadata (dimensions, physics properties, semantic labels) into a
JSON catalog that ``scene_gen`` can use for LLM-driven scene generation.

Usage (inside the Arena Docker container)::

    # Regenerate full catalog from all Arena-registered assets
    /isaac-sim/python.sh isaaclab_arena/scene_gen/generate_catalog.py

    # Scan a specific directory of USD files
    /isaac-sim/python.sh isaaclab_arena/scene_gen/generate_catalog.py \\
        --objects /path/to/usd/dir

    # List all semantic labels in an existing catalog
    /isaac-sim/python.sh isaaclab_arena/scene_gen/generate_catalog.py \\
        --list-classes

    # Labels grouped by dataset
    /isaac-sim/python.sh isaaclab_arena/scene_gen/generate_catalog.py \\
        --list-classes --by-dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from isaaclab_arena.scene_gen.catalog_utils import (
    ARENA_ROOT,
    OBJECT_CATALOG_PATH,
    get_dataset_from_path,
    get_usd_rigid_body_info,
    iter_object_files,
    load_catalog,
    print_object_info,
)


def _collect_registered_usd_paths() -> list[Path]:
    """Collect USD paths from Arena's AssetRegistry (all registered objects)."""
    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.assets.object_base import ObjectType

    registry = AssetRegistry()
    paths: list[Path] = []

    for name in registry.get_all_keys():
        cls = registry.get_asset_by_name(name)
        tags = getattr(cls, "tags", [])
        obj_type = getattr(cls, "object_type", None)
        usd_path = getattr(cls, "usd_path", None)

        if "object" not in tags:
            continue
        if obj_type == ObjectType.SPAWNER:
            continue
        if usd_path and Path(usd_path).exists():
            paths.append(Path(usd_path))

    return sorted(set(paths))


def generate_catalog(
    objects_dir: Path | None = None,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Generate object catalog by scanning USD files.

    If *objects_dir* is ``None``, asset paths are gathered from Arena's
    ``AssetRegistry`` (which includes S3-hosted assets, Lightwheel cache,
    and any manually registered objects).

    Args:
        objects_dir: Explicit directory (or single file) to scan.
                     Falls back to ``AssetRegistry`` if ``None``.
        verbose: Print per-object details.

    Returns:
        List of object info dictionaries.
    """
    if objects_dir is not None:
        objects_dir = Path(objects_dir)
        if objects_dir.is_dir():
            usd_files = iter_object_files(objects_dir)
        else:
            usd_files = [objects_dir]
    else:
        usd_files = _collect_registered_usd_paths()

    print(f"Scanning {len(usd_files)} USD files...")

    catalog: list[dict[str, Any]] = []
    for usd in usd_files:
        try:
            info = get_usd_rigid_body_info(str(usd))
        except Exception as e:
            print(f"  [SKIP] {usd.name}: {e}")
            continue

        info["dataset"] = get_dataset_from_path(str(usd))

        abs_path = info.get("usd_path", "")
        arena_root_str = str(ARENA_ROOT)
        if abs_path and arena_root_str in abs_path:
            info["usd_path"] = abs_path.replace(arena_root_str + "/", "")

        if verbose:
            print_object_info(info, usd)

        catalog.append(info)

    return catalog


def save_catalog(
    catalog: list[dict[str, Any]],
    output_path: Path = OBJECT_CATALOG_PATH,
) -> None:
    """Save catalog to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(catalog, indent=2))
    print(f"Saved catalog with {len(catalog)} objects to: {output_path}")


def list_classes(
    catalog: list[dict[str, Any]],
    by_dataset: bool = False,
    verbose: bool = False,
) -> None:
    """Print all unique semantic labels (classes) from the catalog."""

    def _display_path(usd_path: str) -> str:
        prefix = "assets/objects/"
        return usd_path[len(prefix) :] if usd_path.startswith(prefix) else usd_path

    no_class: dict[str, list[tuple[str, str]]] = defaultdict(list)

    if by_dataset:
        by_ds_cls: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
        for obj in catalog:
            ds = obj.get("dataset", "unknown")
            cls = obj.get("class", "").strip()
            name = obj.get("name", "")
            path = _display_path(obj.get("usd_path", ""))
            if cls:
                by_ds_cls[ds][cls].append((name, path))
            else:
                no_class[ds].append((name, path))

        print("\nSemantic Labels by Dataset:")
        print("=" * 60)
        for ds in sorted(by_ds_cls):
            classes = by_ds_cls[ds]
            print(f"\n{ds} ({len(classes)} classes):")
            for cls in sorted(classes):
                items = sorted(classes[cls])
                if verbose:
                    print(f"  - {cls} ({len(items)} objects):")
                    for n, p in items:
                        print(f"      {n} ({p})")
                else:
                    print(f"  - {cls}")
    else:
        by_cls: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for obj in catalog:
            cls = obj.get("class", "").strip()
            name = obj.get("name", "")
            path = _display_path(obj.get("usd_path", ""))
            ds = obj.get("dataset", "unknown")
            if cls:
                by_cls[cls].append((name, path))
            else:
                no_class[ds].append((name, path))

        print("\nAll Semantic Labels:")
        print("=" * 60)
        for cls in sorted(by_cls):
            items = sorted(by_cls[cls])
            if verbose:
                print(f"\n{cls} ({len(items)} objects):")
                for n, p in items:
                    print(f"    {n} ({p})")
            else:
                print(f"  {cls}: {len(items)} objects")
        print(f"\nTotal: {len(by_cls)} unique classes")

    total_no_class = sum(len(v) for v in no_class.values())
    if total_no_class > 0:
        print(f"\n\nObjects WITHOUT a class attribute ({total_no_class} total):")
        print("=" * 60)
        for ds in sorted(no_class):
            items = sorted(no_class[ds])
            if verbose:
                print(f"\n{ds} ({len(items)} objects):")
                for n, p in items:
                    print(f"    {n} ({p})")
            else:
                print(f"  {ds}: {len(items)} objects")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate object_catalog.json from Arena's USD assets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--objects",
        type=Path,
        default=None,
        help=(
            "Path to objects directory or single USD file. "
            "If omitted, scans all objects registered in Arena's AssetRegistry."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OBJECT_CATALOG_PATH,
        help="Output path for catalog JSON",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-object details",
    )
    parser.add_argument(
        "--list-classes",
        action="store_true",
        help="List all semantic labels instead of generating catalog",
    )
    parser.add_argument(
        "--by-dataset",
        action="store_true",
        help="Group classes by dataset (use with --list-classes)",
    )

    args = parser.parse_args()

    if args.list_classes:
        if not args.output.exists():
            print(f"Error: Catalog not found at {args.output}")
            print("Run without --list-classes first to generate the catalog.")
            return 1

        catalog = load_catalog(args.output)
        list_classes(catalog, by_dataset=args.by_dataset, verbose=args.verbose)
    else:
        catalog = generate_catalog(objects_dir=args.objects, verbose=args.verbose)
        save_catalog(catalog, output_path=args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
