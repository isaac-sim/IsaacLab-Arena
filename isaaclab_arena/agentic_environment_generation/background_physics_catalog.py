# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Resolve background USDs and enumerate physics-enabled subprims."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from isaaclab_arena.assets.registries import AssetRegistry


@dataclass(frozen=True)
class PhysicsPrimEntry:
    """One rigid-body and/or articulation prim inside an asset USD."""

    usd_prim_path: str
    physics_kinds: frozenset[str]
    revolute_joint_names: frozenset[str] = frozenset()

    @property
    def leaf_name(self) -> str:
        return self.usd_prim_path.rsplit("/", 1)[-1]


def resolve_background_usd_path(registry: AssetRegistry, background_name: str) -> str:
    """Instantiate or inspect ``background_name`` and return its USD path."""
    asset_cls = registry.get_asset_by_name(background_name)
    if background_name == "lightwheel_robocasa_kitchen":
        return str(asset_cls(layout_id=1, style_id=1).usd_path)

    class_usd = getattr(asset_cls, "usd_path", None)
    if class_usd is not None:
        return str(class_usd)

    return str(asset_cls().usd_path)


def list_physics_prim_entries(usd_path: str | Path) -> list[PhysicsPrimEntry]:
    """Return physics prims in ``usd_path`` with optional revolute-joint names."""
    from pxr import UsdPhysics

    from isaaclab_arena.utils.usd_helpers import is_articulation_root, is_rigid_body, open_stage

    usd_path = str(usd_path)
    entries: list[PhysicsPrimEntry] = []
    with open_stage(usd_path) as stage:
        articulation_paths: dict[str, set[str]] = {}
        for prim in stage.Traverse():
            if prim.IsPseudoRoot() or not is_articulation_root(prim):
                continue
            root_path = str(prim.GetPath())
            joint_names: set[str] = set()
            for child in stage.Traverse():
                child_path = str(child.GetPath())
                if child_path.startswith(root_path) and child.IsA(UsdPhysics.RevoluteJoint):
                    joint_names.add(child.GetName())
            articulation_paths[root_path] = joint_names

        for prim in stage.Traverse():
            if prim.IsPseudoRoot():
                continue
            kinds: list[str] = []
            if is_articulation_root(prim):
                kinds.append("articulation")
            if is_rigid_body(prim):
                kinds.append("rigid_body")
            if kinds:
                path = str(prim.GetPath())
                entries.append(
                    PhysicsPrimEntry(
                        usd_prim_path=path,
                        physics_kinds=frozenset(kinds),
                        revolute_joint_names=frozenset(articulation_paths.get(path, set())),
                    )
                )
    return sorted(entries, key=lambda entry: entry.usd_prim_path)


def format_physics_prim_catalog(entries: list[PhysicsPrimEntry], *, usd_path: str | Path) -> str:
    """Format physics prim entries for an LLM prompt or debug message."""
    lines = [f"USD: {usd_path}", f"Physics prims: {len(entries)}", ""]
    for entry in entries:
        kind_text = ",".join(sorted(entry.physics_kinds))
        line = f"[{kind_text}] {entry.usd_prim_path}"
        if entry.revolute_joint_names:
            line += "  revolute_joints=" + ", ".join(sorted(entry.revolute_joint_names))
        lines.append(line)
    return "\n".join(lines)


def isaaclab_prim_path_for_background_reference(
    background_node_id: str,
    usd_prim_path: str,
    usd_path: str | Path,
) -> str:
    """Build an Isaac Lab prim path for a subprim inside ``background_node_id``."""
    if usd_prim_path.startswith("{ENV_REGEX_NS}/"):
        return usd_prim_path

    from isaaclab_arena.utils.usd_helpers import open_stage

    with open_stage(str(usd_path)) as stage:
        default_prim = stage.GetDefaultPrim()
        default_path = str(default_prim.GetPath())
        assert usd_prim_path.startswith(
            default_path
        ), f"prim {usd_prim_path!r} is not under default prim {default_path!r} in {usd_path!r}"
        suffix = usd_prim_path[len(default_path) :]
    return f"{{ENV_REGEX_NS}}/{background_node_id}{suffix}"
