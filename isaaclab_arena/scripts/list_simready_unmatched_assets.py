# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""List SimReady GA props with no plausible match in the Arena asset registry.

Usage (inside the dev container)::

    /isaac-sim/python.sh isaaclab_arena/scripts/list_simready_unmatched_assets.py \\
        --out isaaclab_arena_environments/agent_generated/simready_unmatched_assets.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from isaaclab_arena.assets.registries import AssetRegistry, ensure_assets_registered
from isaaclab_arena.assets.simready_object_library import ISAAC_SIMREADY_GA_S3_URL

_STOPWORDS = {
    "a01",
    "a02",
    "a03",
    "b01",
    "robolab",
    "ycb",
    "hot3d",
    "handal",
    "physics",
    "axis",
    "aligned",
    "instanceable",
    "sm",
    "usd",
    "usda",
    "usdz",
    "isaac",
    "simready",
    "assets",
    "object",
    "objects",
    "library",
    "arena",
    "srl",
    "props",
    "industrial",
    "residential",
    "equipment",
    "warehouse",
    "kitchen",
    "food",
    "dish",
    "machine",
    "machines",
    "hardware",
    "tools",
    "vehicle",
    "component",
    "components",
}


@dataclass(frozen=True)
class ArenaObjectRecord:
    """One registered Arena pick-up object."""

    registry_name: str
    usd_path: str
    basename: str
    tokens: frozenset[str]


@dataclass(frozen=True)
class SimReadyRecord:
    """One SimReady GA prop."""

    usd_path: str
    rel_path: str
    label: str
    tokens: frozenset[str]


def _tokenize(text: str) -> set[str]:
    text = unquote(text).lower()
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    parts = re.split(r"[^a-z0-9]+", text)
    return {part for part in parts if len(part) >= 3 and part not in _STOPWORDS}


def _basename(path: str) -> str:
    return Path(unquote(path)).name.lower()


def _resolve_usd_path(cls: type) -> str | None:
    usd_path = getattr(cls, "usd_path", None)
    if usd_path is None:
        return None
    if isinstance(usd_path, str):
        return usd_path
    if hasattr(usd_path, "get_path"):
        try:
            return str(usd_path.get_path())
        except Exception:
            return str(usd_path)
    return str(usd_path)


def _collect_arena_objects() -> list[ArenaObjectRecord]:
    ensure_assets_registered()
    registry = AssetRegistry()
    records: list[ArenaObjectRecord] = []
    for name in registry.get_all_keys():
        cls = registry.get_asset_by_name(name)
        tags = getattr(cls, "tags", None) or []
        if "object" not in tags:
            continue
        usd_path = _resolve_usd_path(cls)
        if not usd_path:
            continue
        basename = _basename(usd_path)
        tokens = frozenset(_tokenize(f"{name} {usd_path} {basename}"))
        records.append(
            ArenaObjectRecord(
                registry_name=name,
                usd_path=usd_path,
                basename=basename,
                tokens=tokens,
            )
        )
    return records


def _simready_label(rel_path: str) -> str:
    path = Path(rel_path)
    if path.suffix.lower() in {".usd", ".usda", ".usdz"}:
        parent = path.parent.name
        if parent and parent not in {".", "/"}:
            return parent
    return path.stem


_REGISTRY_SUFFIX_RE = re.compile(r"(_(?:robolab|ycb_robolab|hot3d_robolab|handal_robolab|fruits_veggies_robolab))+$")


def _normalize_stem(name: str) -> str:
    stem = Path(name).stem.lower()
    stem = re.sub(r"^sm_", "", stem)
    stem = re.sub(r"_[ab][0-9]{2}(_[0-9]{2})?$", "", stem)
    stem = re.sub(r"_[0-9]+$", "", stem)
    return stem


def _registry_core_name(registry_name: str) -> str:
    core = _REGISTRY_SUFFIX_RE.sub("", registry_name.lower())
    core = re.sub(r"^[0-9]{3}_", "", core)
    core = re.sub(r"[0-9]+$", "", core)
    return core.strip("_")


def _path_contains_token(path: str, token: str) -> bool:
    if len(token) < 4:
        return False
    pattern = rf"(^|/|_|-){re.escape(token)}($|/|_|-|\.)"
    return re.search(pattern, path.lower()) is not None


def _has_arena_match(simready: SimReadyRecord, arena_objects: list[ArenaObjectRecord]) -> tuple[bool, str | None]:
    """Return whether a SimReady asset plausibly matches any Arena registry object."""
    simready_basename = _basename(simready.usd_path)
    simready_stem = _normalize_stem(simready_basename)
    simready_rel_lower = simready.rel_path.lower()

    for arena in arena_objects:
        if arena.basename == simready_basename:
            return True, arena.registry_name

        arena_stem = _normalize_stem(arena.basename)
        if arena_stem and arena_stem == simready_stem:
            return True, arena.registry_name

        if arena_stem and len(arena_stem) >= 5:
            if arena_stem in simready_basename or arena_stem in simready_rel_lower:
                return True, arena.registry_name

        core_name = _registry_core_name(arena.registry_name)
        if len(core_name) >= 5 and _path_contains_token(simready_rel_lower, core_name):
            return True, arena.registry_name

        ycb_match = re.search(r"^[0-9]{3}_(.+)$", arena_stem)
        if ycb_match:
            ycb_object = ycb_match.group(1)
            if ycb_object in simready_basename or ycb_object in simready_rel_lower:
                return True, arena.registry_name

    return False, None


async def _collect_simready_assets() -> list[SimReadyRecord]:
    from simready.search import AssetLibrary, SearchFilterPathContains

    library = AssetLibrary()
    await library.add_s3_source(ISAAC_SIMREADY_GA_S3_URL)
    matches = library.search(include_all=[SearchFilterPathContains("SimReady")])
    records: list[SimReadyRecord] = []
    seen: set[str] = set()
    prefix = "/Assets/Isaac/6.0/Isaac/SimReady/"
    for match in matches:
        usd_path = str(match.asset_path)
        if usd_path in seen:
            continue
        seen.add(usd_path)
        rel_path = usd_path.split(prefix, 1)[-1] if prefix in usd_path else usd_path
        label = _simready_label(rel_path)
        tokens = frozenset(_tokenize(f"{rel_path} {label} {usd_path}"))
        records.append(
            SimReadyRecord(
                usd_path=usd_path,
                rel_path=rel_path,
                label=label,
                tokens=tokens,
            )
        )
    records.sort(key=lambda record: record.rel_path.lower())
    return records


async def main_async(out: Path) -> dict[str, Any]:
    arena_objects = _collect_arena_objects()
    simready_assets = await _collect_simready_assets()
    unmatched: list[dict[str, Any]] = []
    matched_examples: list[dict[str, str]] = []
    for simready in simready_assets:
        matched, registry_name = _has_arena_match(simready, arena_objects)
        if matched:
            if len(matched_examples) < 25:
                matched_examples.append({
                    "simready_label": simready.label,
                    "simready_rel_path": simready.rel_path,
                    "arena_registry_name": registry_name or "",
                })
            continue
        unmatched.append({
            "label": simready.label,
            "rel_path": simready.rel_path,
            "usd_path": simready.usd_path,
        })
    payload = {
        "simready_source": ISAAC_SIMREADY_GA_S3_URL,
        "arena_object_count": len(arena_objects),
        "simready_asset_count": len(simready_assets),
        "matched_count": len(simready_assets) - len(unmatched),
        "unmatched_count": len(unmatched),
        "matching_notes": (
            "A SimReady asset is 'matched' only on strong signals: identical USD basename/stem, "
            "Arena USD stem substring in the SimReady path, registry core name token in the "
            "SimReady relative path, or YCB-style numbered object stem overlap. "
            "Review borderline cases manually."
        ),
        "matched_examples": matched_examples,
        "unmatched_assets": unmatched,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    txt_path = out.with_suffix(".txt")
    txt_lines = [f"{row['label']}\t{row['rel_path']}" for row in unmatched]
    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("isaaclab_arena_environments/agent_generated/simready_unmatched_assets.json"),
        help="Output JSON path (a .txt tab-separated list is written alongside).",
    )
    args = parser.parse_args()
    payload = asyncio.run(main_async(args.out))
    print(
        f"Wrote {payload['unmatched_count']} unmatched / {payload['simready_asset_count']} SimReady assets "
        f"({payload['matched_count']} matched Arena registry objects) → {args.out}"
    )


if __name__ == "__main__":
    main()
