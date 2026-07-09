# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Kit viewport PNG capture for review GUI node thumbnails (SimApp subprocess only)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import omni.usd
from omni.kit.viewport.utility import frame_viewport_prims, get_active_viewport
from pxr import Gf, Sdf, UsdGeom, UsdLux

from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.asset_usd import (
    AabbDimensionsM,
    ObjectReferenceUsdTarget,
    object_reference_cache_key,
    resolve_node_aabb_dimensions_m,
    resolve_node_usd_paths,
    resolve_object_reference_usd_targets,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.kit_viewport import (
    PRE_CAPTURE_UPDATES,
    capture_viewport_png,
    thumbnail_cache_dir,
    wait_for_stage_load,
)


def _usd_cache_key(usd_path: str) -> str:
    """Return a stable short hash for caching thumbnails keyed by USD path."""
    return hashlib.sha1(usd_path.encode("utf-8")).hexdigest()[:16]


def render_thumbnails_with_app(app, spec: ArenaEnvGraphSpec) -> tuple[dict[str, Path], dict[str, AabbDimensionsM]]:
    """Render cache-missed node thumbnails and return png paths plus AABB sizes in meters."""
    asset_paths = resolve_node_usd_paths(spec)
    reference_targets = resolve_object_reference_usd_targets(spec)
    if not asset_paths and not reference_targets:
        print("[thumbnail_capture] no asset USD paths resolved; skipping thumbnail rendering.", file=sys.stderr)
        return {}, {}

    cache_dir = thumbnail_cache_dir()

    resolved: dict[str, Path] = {}
    to_render: dict[str, tuple[str, Path]] = {}
    for node_id, usd_path in asset_paths.items():
        cache_path = cache_dir / f"{_usd_cache_key(usd_path)}.png"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            resolved[node_id] = cache_path
        else:
            to_render[node_id] = (usd_path, cache_path)

    reference_to_render: dict[str, tuple[ObjectReferenceUsdTarget, Path]] = {}
    for node_id, target in reference_targets.items():
        cache_path = cache_dir / f"{object_reference_cache_key(target.usd_path, target.relative_prim_path)}.png"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            resolved[node_id] = cache_path
        else:
            reference_to_render[node_id] = (target, cache_path)

    if to_render:
        print(
            f"[thumbnail_capture] rendering {len(to_render)} new thumbnail(s) "
            f"(reusing {len(resolved)} from cache at {cache_dir})...",
            file=sys.stderr,
        )
        captured = _capture_usd_thumbnails(app, to_render)
        for node_id, (_usd_path, cache_path) in to_render.items():
            if node_id in captured and cache_path.exists() and cache_path.stat().st_size > 0:
                resolved[node_id] = cache_path
    elif reference_to_render:
        print(
            f"[thumbnail_capture] rendering {len(reference_to_render)} object_reference thumbnail(s) "
            f"(reusing {len(resolved)} from cache at {cache_dir})...",
            file=sys.stderr,
        )
    else:
        print(f"[thumbnail_capture] all {len(resolved)} thumbnail(s) served from cache.", file=sys.stderr)

    if reference_to_render:
        captured_refs = _capture_object_reference_thumbnails(app, reference_to_render)
        for node_id, (_target, cache_path) in reference_to_render.items():
            if node_id in captured_refs and cache_path.exists() and cache_path.stat().st_size > 0:
                resolved[node_id] = cache_path

    return resolved, resolve_node_aabb_dimensions_m(spec)


def _capture_usd_thumbnails(app, to_render: dict[str, tuple[str, Path]]) -> dict[str, bytes]:
    """Capture queued USDs under one booted ``SimulationApp``, deduplicated by path."""
    out: dict[str, bytes] = {}

    path_to_node_ids: dict[str, list[str]] = {}
    path_to_cache: dict[str, Path] = {}
    for node_id, (usd_path, cache_path) in to_render.items():
        path_to_node_ids.setdefault(usd_path, []).append(node_id)
        path_to_cache[usd_path] = cache_path

    for usd_path, node_ids in path_to_node_ids.items():
        cache_path = path_to_cache[usd_path]
        try:
            png_bytes = _render_one_usd(app, usd_path, cache_path)
        except Exception as exc:
            print(f"[thumbnail_capture]   render failed for {usd_path}: {exc}", file=sys.stderr)
            continue
        if png_bytes:
            for node_id in node_ids:
                out[node_id] = png_bytes

    return out


def _capture_object_reference_thumbnails(
    app,
    to_render: dict[str, tuple[ObjectReferenceUsdTarget, Path]],
) -> dict[str, bytes]:
    """Capture object_reference snapshots with collision meshes highlighted."""
    out: dict[str, bytes] = {}
    for node_id, (target, cache_path) in to_render.items():
        try:
            png_bytes = _render_object_reference(app, target, cache_path)
        except Exception as exc:
            print(
                f"[thumbnail_capture]   object_reference render failed for {target.relative_prim_path}: {exc}",
                file=sys.stderr,
            )
            continue
        if png_bytes:
            out[node_id] = png_bytes
    return out


def _render_one_usd(app, usd_path: str, cache_path: Path) -> bytes | None:
    """Open ``usd_path`` as the stage root, frame the default prim, capture PNG."""
    ctx = omni.usd.get_context()
    if not ctx.open_stage(usd_path):
        print(f"[thumbnail_capture]   open_stage failed: {usd_path}", file=sys.stderr)
        return None
    stage = ctx.get_stage()

    wait_for_stage_load(app, ctx)
    _ensure_default_lighting(stage)

    target_prim = stage.GetDefaultPrim()
    if not target_prim or not target_prim.IsValid():
        target_prim = stage.GetPrimAtPath(Sdf.Path("/"))

    viewport = get_active_viewport()
    framed = frame_viewport_prims(viewport, prims=[str(target_prim.GetPath())])
    if not framed:
        print(f"[thumbnail_capture]   warning: frame_viewport_prims failed for {usd_path}", file=sys.stderr)

    png_bytes = capture_viewport_png(app, cache_path)
    if png_bytes is not None:
        return png_bytes
    print(f"[thumbnail_capture]   capture produced no file: {cache_path}", file=sys.stderr)
    return None


def _absolute_prim_path(stage, relative_suffix: str) -> str:
    """Join a default-prim-relative suffix to the stage default prim."""
    default_prim = stage.GetDefaultPrim()
    assert default_prim and default_prim.IsValid(), "USD stage has no default prim"
    base = str(default_prim.GetPath())
    if not relative_suffix:
        return base
    return f"{base}/{relative_suffix.lstrip('/')}"


def _collect_collision_prim_paths(stage, root_path: str) -> list[str]:
    """Return collision/physics prim paths under ``root_path`` (including the root)."""
    from pxr import Usd

    from isaaclab_arena.utils.usd_helpers import has_physics_or_collision

    root = stage.GetPrimAtPath(root_path)
    assert root and root.IsValid(), f"Prim not found: {root_path}"
    paths = [root_path]
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        if path != root_path and has_physics_or_collision(prim):
            paths.append(path)
    return paths


def _enable_collision_mesh_visualization(selected_prim_paths: list[str]) -> None:
    """Match the viewport Show By Type → Physics → Colliders → Selected workflow."""
    import carb.settings
    import omni.usd

    settings = carb.settings.get_settings()
    settings.set_bool("/persistent/physics/visualizationCollisionMesh", True)
    settings.set_int("/persistent/physics/visualizationDisplayColliders", 2)
    if selected_prim_paths:
        omni.usd.get_context().get_selection().set_selected_prim_paths(selected_prim_paths, True)


def _render_object_reference(app, target: ObjectReferenceUsdTarget, cache_path: Path) -> bytes | None:
    """Open the parent USD, frame the referenced prim, and capture collision meshes."""
    from omni.kit.viewport.utility import frame_viewport_prims, get_active_viewport

    ctx = omni.usd.get_context()
    if not ctx.open_stage(target.usd_path):
        print(f"[thumbnail_capture]   open_stage failed: {target.usd_path}", file=sys.stderr)
        return None
    stage = ctx.get_stage()

    wait_for_stage_load(app, ctx)
    _ensure_default_lighting(stage)

    root_path = _absolute_prim_path(stage, target.relative_prim_path)
    collision_paths = _collect_collision_prim_paths(stage, root_path)
    _enable_collision_mesh_visualization(collision_paths)

    viewport = get_active_viewport()
    framed = frame_viewport_prims(viewport, prims=collision_paths)
    if not framed:
        print(
            f"[thumbnail_capture]   warning: frame_viewport_prims failed for {target.relative_prim_path}",
            file=sys.stderr,
        )

    png_bytes = capture_viewport_png(app, cache_path, pre_capture_updates=PRE_CAPTURE_UPDATES + 3)
    if png_bytes is not None:
        return png_bytes
    print(f"[thumbnail_capture]   capture produced no file: {cache_path}", file=sys.stderr)
    return None


def _ensure_default_lighting(stage) -> None:
    """Add dome + key lights when the stage has none (standalone object USDs)."""
    for prim in stage.Traverse():
        if (
            prim.HasAPI(UsdLux.LightAPI)
            or prim.IsA(UsdLux.BoundableLightBase)
            or prim.IsA(UsdLux.NonboundableLightBase)
        ):
            return

    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/_ReviewDomeLight"))
    dome.CreateIntensityAttr(800.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    key = UsdLux.DistantLight.Define(stage, Sdf.Path("/_ReviewKeyLight"))
    key.CreateIntensityAttr(2500.0)
    key.CreateAngleAttr(2.0)
    key_xformable = UsdGeom.Xformable(key.GetPrim())
    key_xformable.ClearXformOpOrder()
    rot = key_xformable.AddRotateXYZOp()
    rot.Set(Gf.Vec3f(-45.0, 30.0, 0.0))
