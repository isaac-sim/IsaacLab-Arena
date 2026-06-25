# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Kit viewport PNG capture for review GUI node thumbnails (SimApp subprocess only)."""

from __future__ import annotations

import sys
from pathlib import Path

from isaaclab_arena.assets.asset_cache import get_review_gui_thumbnail_cache_dir
from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvInitialGraphSpec
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.asset_usd import (
    AabbDimensionsM,
    resolve_node_aabb_dimensions_m,
    resolve_node_usd_paths,
    usd_cache_key,
)


def _thumbnail_cache_dir() -> Path:
    return get_review_gui_thumbnail_cache_dir()


def render_thumbnails_with_app(
    app, spec: ArenaEnvInitialGraphSpec
) -> tuple[dict[str, Path], dict[str, AabbDimensionsM]]:
    """Render cache-missed node thumbnails and return png paths plus AABB sizes in meters."""
    asset_paths = resolve_node_usd_paths(spec)
    if not asset_paths:
        print("[thumbnail_capture] no asset USD paths resolved; skipping thumbnail rendering.", file=sys.stderr)
        return {}, {}

    thumbnail_cache_dir = _thumbnail_cache_dir()

    resolved: dict[str, Path] = {}
    to_render: dict[str, tuple[str, Path]] = {}
    for node_id, usd_path in asset_paths.items():
        cache_path = thumbnail_cache_dir / f"{usd_cache_key(usd_path)}.png"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            resolved[node_id] = cache_path
        else:
            to_render[node_id] = (usd_path, cache_path)

    if to_render:
        print(
            f"[thumbnail_capture] rendering {len(to_render)} new thumbnail(s) "
            f"(reusing {len(resolved)} from cache at {thumbnail_cache_dir})...",
            file=sys.stderr,
        )
        captured = _capture_usd_thumbnails(app, to_render)
        for node_id, (_usd_path, cache_path) in to_render.items():
            if node_id in captured and cache_path.exists() and cache_path.stat().st_size > 0:
                resolved[node_id] = cache_path
    else:
        print(f"[thumbnail_capture] all {len(resolved)} thumbnail(s) served from cache.", file=sys.stderr)

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
        print(f"[thumbnail_capture]   open_stage failed: {usd_path}", file=sys.stderr)
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
        print(f"[thumbnail_capture]   warning: frame_viewport_prims failed for {usd_path}", file=sys.stderr)

    for _ in range(30):
        app.update()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    capture_obj = capture_viewport_to_file(viewport, str(cache_path))
    _wait_for_capture(app, capture_obj, cache_path, max_updates=600)

    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_bytes()
    print(f"[thumbnail_capture]   capture produced no file: {cache_path}", file=sys.stderr)
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
            if cache_path.exists() and cache_path.stat().st_size > 0:
                return
        return

    # Kit has no stable capture-completion API; ``future`` is best-effort. File existence is
    # the reliable exit condition (see the loop below).
    future = getattr(capture_obj, "future", None)

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
