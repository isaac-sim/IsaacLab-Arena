# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""USD viewport thumbnail rendering for the review GUI SimApp server."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvInitialGraphSpec
from isaaclab_arena.environments.arena_env_graph_types import ArenaEnvGraphNodeType

THUMBNAIL_CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache" / "llm_env_gen_thumbnails"

# Registry-backed nodes with a root USD. ``object_reference`` nodes point at a prim
# inside a parent background and need parent-stage framing — not supported yet.
_RENDERABLE_NODE_TYPES = frozenset({
    ArenaEnvGraphNodeType.EMBODIMENT,
    ArenaEnvGraphNodeType.BACKGROUND,
    ArenaEnvGraphNodeType.OBJECT,
})


AabbDimensionsM = tuple[float, float, float]


def resolve_node_aabb_dimensions_m(spec: ArenaEnvInitialGraphSpec) -> dict[str, AabbDimensionsM]:
    """Return axis-aligned bounding box sizes in meters for each node with a resolvable USD."""
    asset_paths = _resolve_node_usd_paths(spec)
    dimensions: dict[str, AabbDimensionsM] = {}
    for node_id, usd_path in asset_paths.items():
        dims = _aabb_dimensions_from_usd(usd_path)
        if dims is not None:
            dimensions[node_id] = dims
    return dimensions


def _render_thumbnails_with_app(
    app, spec: ArenaEnvInitialGraphSpec
) -> tuple[dict[str, Path], dict[str, AabbDimensionsM]]:
    """Render cache-missed node thumbnails and return png paths plus AABB sizes in meters."""
    asset_paths = _resolve_node_usd_paths(spec)
    if not asset_paths:
        print("[thumbnail_render] no asset USD paths resolved; skipping thumbnail rendering.", file=sys.stderr)
        return {}, {}

    THUMBNAIL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    resolved: dict[str, Path] = {}
    to_render: dict[str, tuple[str, Path]] = {}
    for node_id, usd_path in asset_paths.items():
        cache_path = THUMBNAIL_CACHE_DIR / f"{_usd_cache_key(usd_path)}.png"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            resolved[node_id] = cache_path
        else:
            to_render[node_id] = (usd_path, cache_path)

    if to_render:
        print(
            f"[thumbnail_render] rendering {len(to_render)} new thumbnail(s) "
            f"(reusing {len(resolved)} from cache at {THUMBNAIL_CACHE_DIR})...",
            file=sys.stderr,
        )
        captured = _capture_usd_thumbnails(app, to_render)
        for node_id, (_usd_path, cache_path) in to_render.items():
            if node_id in captured and cache_path.exists() and cache_path.stat().st_size > 0:
                resolved[node_id] = cache_path
    else:
        print(f"[thumbnail_render] all {len(resolved)} thumbnail(s) served from cache.", file=sys.stderr)

    return resolved, resolve_node_aabb_dimensions_m(spec)


def _resolve_node_usd_paths(spec: ArenaEnvInitialGraphSpec) -> dict[str, str]:
    """Map ``node.id → usd_path`` via :class:`AssetRegistry`."""
    try:
        from isaaclab_arena.assets.registries import AssetRegistry  # noqa: PLC0415
    except Exception as exc:
        print(f"[thumbnail_render] AssetRegistry import failed: {exc}", file=sys.stderr)
        return {}

    registry = AssetRegistry()
    paths: dict[str, str] = {}
    for node in spec.nodes:
        if node.type not in _RENDERABLE_NODE_TYPES:
            continue
        try:
            if not registry.is_registered(node.name):
                print(f"[thumbnail_render]   {node.id}: asset '{node.name}' not registered, skipping.", file=sys.stderr)
                continue
            cls = registry.get_asset_by_name(node.name)
            usd_path = _extract_usd_path(cls)
            if not usd_path:
                print(f"[thumbnail_render]   {node.id}: '{node.name}' has no usd_path, skipping.", file=sys.stderr)
                continue
            paths[node.id] = usd_path
        except Exception as exc:
            print(f"[thumbnail_render]   {node.id}: lookup failed for '{node.name}': {exc}", file=sys.stderr)
    return paths


def _extract_usd_path(cls) -> str | None:
    """Return the asset's root USD path, or ``None`` if not extractable."""
    usd_path = getattr(cls, "usd_path", None)
    if usd_path:
        return usd_path

    try:
        instance = cls()
    except Exception:
        return None
    scene_config = getattr(instance, "scene_config", None)
    robot = getattr(scene_config, "robot", None) if scene_config is not None else None
    spawn = getattr(robot, "spawn", None) if robot is not None else None
    return getattr(spawn, "usd_path", None) if spawn is not None else None


def _usd_cache_key(usd_path: str) -> str:
    return hashlib.sha1(usd_path.encode("utf-8")).hexdigest()[:16]


def _aabb_dimensions_from_usd(usd_path: str) -> AabbDimensionsM | None:
    """Return local axis-aligned bounding box size (x, y, z) in meters for a USD asset."""
    try:
        from isaaclab_arena.utils.usd_helpers import compute_local_bounding_box_from_usd  # noqa: PLC0415

        bbox = compute_local_bounding_box_from_usd(usd_path)
        size = bbox.size[0]
        return (float(size[0]), float(size[1]), float(size[2]))
    except Exception as exc:
        print(f"[thumbnail_render]   bbox failed for {usd_path}: {exc}", file=sys.stderr)
        return None


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
            print(f"[thumbnail_render]   render failed for {usd_path}: {exc}", file=sys.stderr)
            continue
        if png_bytes:
            for node_id in node_ids:
                out[node_id] = png_bytes

    return out


def _render_one_usd(app, usd_path: str, cache_path: Path) -> bytes | None:
    """Open ``usd_path`` as the stage root, frame the default prim, capture PNG."""
    import omni.usd  # noqa: PLC0415
    from omni.kit.viewport.utility import (  # noqa: PLC0415
        capture_viewport_to_file,
        frame_viewport_prims,
        get_active_viewport,
    )
    from pxr import Sdf  # noqa: PLC0415

    ctx = omni.usd.get_context()
    if not ctx.open_stage(usd_path):
        print(f"[thumbnail_render]   open_stage failed: {usd_path}", file=sys.stderr)
        return None
    stage = ctx.get_stage()

    _wait_for_stage_load(app, ctx)
    _ensure_default_lighting(stage)

    target_prim = stage.GetDefaultPrim()
    if not target_prim or not target_prim.IsValid():
        target_prim = stage.GetPrimAtPath(Sdf.Path("/"))

    viewport = get_active_viewport()
    framed = frame_viewport_prims(viewport, prims=[str(target_prim.GetPath())])
    if not framed:
        print(f"[thumbnail_render]   warning: frame_viewport_prims failed for {usd_path}", file=sys.stderr)

    for _ in range(30):
        app.update()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    capture_obj = capture_viewport_to_file(viewport, str(cache_path))
    _wait_for_capture(app, capture_obj, cache_path, max_updates=600)

    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_bytes()
    print(f"[thumbnail_render]   capture produced no file: {cache_path}", file=sys.stderr)
    return None


def _wait_for_stage_load(app, usd_context, max_updates: int = 600) -> None:
    """Pump frames until stage loading settles (plus a short post-settle tail)."""
    settled = 0
    for _ in range(max_updates):
        app.update()
        try:
            _msg, loading_count, loaded_count = usd_context.get_stage_loading_status()
        except Exception:
            return
        if loading_count == 0 and loaded_count == 0:
            settled += 1
            if settled > 15:
                return
        else:
            settled = 0


def _wait_for_capture(app, capture_obj, cache_path: Path, max_updates: int = 600) -> None:
    """Pump ``app.update()`` until the capture PNG exists or the budget expires."""
    if capture_obj is None:
        for _ in range(max_updates):
            app.update()
        return

    future = (
        getattr(capture_obj, "_Capture__future", None)
        or getattr(capture_obj, "_RenderCapture__future", None)
        or getattr(capture_obj, "future", None)
    )

    for _ in range(max_updates):
        app.update()
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return
        if future is not None and future.done():
            for _ in range(15):
                app.update()
                if cache_path.exists() and cache_path.stat().st_size > 0:
                    return
            return


def _ensure_default_lighting(stage) -> None:
    """Add dome + key lights when the stage has none (standalone object USDs)."""
    from pxr import Gf, Sdf, UsdGeom, UsdLux  # noqa: PLC0415

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
