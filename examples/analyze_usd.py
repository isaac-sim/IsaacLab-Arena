# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Inspect the geometry, physics, and collision definitions in a USD asset.

Usage:

    .venv/bin/python examples/analyze_usd.py /path/to/asset.usd
    .venv/bin/python examples/analyze_usd.py /path/to/asset.usd --json report.json
"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pxr import Kind, Usd, UsdGeom, UsdPhysics


def _json_value(value: Any, attribute_name: str | None = None, is_mesh: bool = False) -> Any:
    """Convert USD values to JSON-compatible data without dumping mesh arrays."""
    if is_mesh and not isinstance(value, str | bytes) and hasattr(value, "__len__"):
        return {"count": len(value)}
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _prim_details(prim: Usd.Prim) -> dict[str, Any]:
    """Return all authored metadata, relationships, and attributes for one prim."""
    is_mesh = prim.IsA(UsdGeom.Mesh)
    attributes = {}
    for attribute in prim.GetAttributes():
        value = attribute.Get()
        attributes[attribute.GetName()] = {
            "type": str(attribute.GetTypeName()),
            "custom": attribute.IsCustom(),
            "authored": attribute.HasAuthoredValueOpinion(),
            "value": _json_value(value, attribute.GetName(), is_mesh),
        }

    relationships = {
        relationship.GetName(): _paths(relationship.GetTargets()) for relationship in prim.GetRelationships()
    }
    variants = prim.GetVariantSets()
    return {
        "path": str(prim.GetPath()),
        "name": prim.GetName(),
        "type": prim.GetTypeName() or "untyped",
        "active": prim.IsActive(),
        "defined": prim.IsDefined(),
        "instance": prim.IsInstance(),
        "applied_schemas": list(prim.GetAppliedSchemas()),
        "metadata": _json_value(prim.GetAllAuthoredMetadata()),
        "variant_sets": {
            name: variants.GetVariantSet(name).GetVariantSelection() for name in variants.GetNames()
        },
        "relationships": relationships,
        "attributes": attributes,
    }


def _paths(targets) -> list[str]:
    """Return relationship targets as strings."""
    return [str(target) for target in targets]


def _attribute_value(schema: Any, attribute_getter: str) -> Any:
    """Read a schema attribute when the schema version provides it."""
    getter = getattr(schema, attribute_getter, None)
    if getter is None:
        return None
    attribute = getter()
    return attribute.Get() if attribute and attribute.HasAuthoredValueOpinion() else None


def analyze_usd(usd_path: Path) -> dict[str, Any]:
    """Return a structured summary of a USD stage.

    Args:
        usd_path: Local USD, USDA, or USDC file to open.

    Returns:
        Asset metadata plus lists of geometry, physics, collision, and joint prims.
    """
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"无法打开 USD 文件：{usd_path}")

    prim_type_counts: Counter[str] = Counter()
    meshes: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    rigid_bodies: list[dict[str, Any]] = []
    collisions: list[dict[str, Any]] = []
    joints: list[dict[str, Any]] = []
    articulation_roots: list[str] = []
    physics_schemas: list[dict[str, Any]] = []
    prims: list[dict[str, Any]] = []

    for prim in stage.TraverseAll():
        if not prim.IsActive():
            continue

        prims.append(_prim_details(prim))
        prim_type_counts[prim.GetTypeName() or "untyped"] += 1
        path = str(prim.GetPath())
        applied_schemas = list(prim.GetAppliedSchemas())

        kind = Usd.ModelAPI(prim).GetKind()
        if kind and Kind.Registry.IsA(kind, "model"):
            models.append({"path": path, "kind": str(kind), "type": prim.GetTypeName()})

        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get() or []
            face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get() or []
            meshes.append(
                {
                    "path": path,
                    "points": len(points),
                    "faces": len(face_vertex_counts),
                    "double_sided": mesh.GetDoubleSidedAttr().Get(),
                }
            )

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body = UsdPhysics.RigidBodyAPI(prim)
            mass = UsdPhysics.MassAPI(prim) if prim.HasAPI(UsdPhysics.MassAPI) else None
            rigid_bodies.append(
                {
                    "path": path,
                    "kinematic_enabled": _attribute_value(rigid_body, "GetKinematicEnabledAttr"),
                    "rigid_body_enabled": _attribute_value(rigid_body, "GetRigidBodyEnabledAttr"),
                    "mass": _attribute_value(mass, "GetMassAttr") if mass else None,
                }
            )

        if prim.HasAPI(UsdPhysics.CollisionAPI):
            mesh_collision = UsdPhysics.MeshCollisionAPI(prim) if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else None
            collisions.append(
                {
                    "path": path,
                    "type": prim.GetTypeName(),
                    "mesh_approximation": _attribute_value(mesh_collision, "GetApproximationAttr") if mesh_collision else None,
                }
            )

        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(path)

        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            joints.append(
                {
                    "path": path,
                    "type": prim.GetTypeName(),
                    "body0": _paths(joint.GetBody0Rel().GetTargets()),
                    "body1": _paths(joint.GetBody1Rel().GetTargets()),
                }
            )

        physics_related = [schema for schema in applied_schemas if "Physics" in schema or "Physx" in schema]
        if physics_related:
            physics_schemas.append({"path": path, "schemas": physics_related})

    root_layer = stage.GetRootLayer()
    return {
        "file": str(usd_path.resolve()),
        "root_layer": root_layer.realPath or root_layer.identifier,
        "up_axis": UsdGeom.GetStageUpAxis(stage),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
        "prim_type_counts": dict(sorted(prim_type_counts.items())),
        "models": models,
        "meshes": meshes,
        "rigid_bodies": rigid_bodies,
        "collisions": collisions,
        "articulation_roots": articulation_roots,
        "joints": joints,
        "physics_schemas": physics_schemas,
        "prims": prims,
    }


def _print_report(report: dict[str, Any]) -> None:
    """Print a concise human-readable USD report."""
    print(f"文件：{report['file']}")
    print(f"根图层：{report['root_layer']}")
    print(f"坐标轴：{report['up_axis']}，单位：{report['meters_per_unit']} m/unit")
    print(f"默认 Prim：{report['default_prim'] or '未设置'}")
    print(f"Prim 类型统计：{report['prim_type_counts']}")

    for title, entries in (
        ("模型", report["models"]),
        ("网格", report["meshes"]),
        ("刚体", report["rigid_bodies"]),
        ("碰撞体", report["collisions"]),
        ("关节", report["joints"]),
        ("Articulation 根", [{"path": path} for path in report["articulation_roots"]]),
    ):
        print(f"\n{title}（{len(entries)}）")
        for entry in entries:
            print(f"  {entry}")

    print("\n所有 Prim 明细（网格数组仅显示数量）")
    print(json.dumps(report["prims"], ensure_ascii=False, indent=2))


def _html_tree(value: Any, label: str, expanded: bool = False) -> str:
    """Render a JSON-compatible value as an expandable HTML tree node."""
    escaped_label = html.escape(label)
    if isinstance(value, dict):
        children = "".join(_html_tree(item, str(key)) for key, item in value.items())
        return (
            f"<li><details{' open' if expanded else ''}><summary>{escaped_label}"
            f" <span class=\"count\">({len(value)})</span></summary><ul>{children}</ul></details></li>"
        )
    if isinstance(value, list):
        children = "".join(_html_tree(item, f"[{index}]") for index, item in enumerate(value))
        return (
            f"<li><details{' open' if expanded else ''}><summary>{escaped_label}"
            f" <span class=\"count\">({len(value)})</span></summary><ul>{children}</ul></details></li>"
        )
    rendered_value = html.escape(json.dumps(value, ensure_ascii=False))
    return f"<li><span class=\"key\">{escaped_label}</span>: <span class=\"value\">{rendered_value}</span></li>"


def _write_html_report(report: dict[str, Any], output_path: Path) -> None:
    """Write an offline, expandable HTML representation of a USD report."""
    tree = _html_tree(report, "USD 报告", expanded=True)
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>USD 分析报告</title>
  <style>
    body {{ margin: 2rem; color: #1f2937; font: 14px/1.5 system-ui, sans-serif; background: #f8fafc; }}
    main {{ max-width: 1100px; padding: 1.5rem 2rem; background: #fff; border-radius: 10px; box-shadow: 0 1px 4px #0002; }}
    button {{ margin-right: .5rem; padding: .35rem .7rem; cursor: pointer; }}
    ul {{ margin: .2rem 0 .2rem 1.1rem; padding-left: 1rem; border-left: 1px solid #dbe3ee; list-style: none; }}
    li {{ margin: .12rem 0; }} summary {{ cursor: pointer; color: #0f4c81; }}
    .key {{ color: #7c3aed; }} .value {{ color: #166534; white-space: pre-wrap; }} .count {{ color: #64748b; }}
  </style>
</head>
<body>
  <main>
    <h1>USD 分析报告</h1>
    <p><button onclick="setOpen(true)">全部展开</button><button onclick="setOpen(false)">全部折叠</button></p>
    <ul>{tree}</ul>
  </main>
  <script>function setOpen(open) {{ document.querySelectorAll('details').forEach((item) => item.open = open); }}</script>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")


def main() -> None:
    """Parse arguments and print or save a USD asset report."""
    parser = argparse.ArgumentParser(description="分析 USD 的模型、物理与碰撞信息。")
    parser.add_argument("usd_path", type=Path, help="待分析的本地 USD 文件")
    parser.add_argument("--json", type=Path, metavar="OUTPUT", help="将完整报告写入 JSON 文件")
    parser.add_argument("--html", type=Path, metavar="OUTPUT", help="将可展开的树状报告写入 HTML 文件")
    args = parser.parse_args()

    if not args.usd_path.is_file():
        parser.error(f"文件不存在：{args.usd_path}")

    report = analyze_usd(args.usd_path)
    _print_report(report)
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n完整 JSON 报告已写入：{args.json}")
    if args.html:
        _write_html_report(report, args.html)
        print(f"可展开的 HTML 报告已写入：{args.html}")


if __name__ == "__main__":
    main()
