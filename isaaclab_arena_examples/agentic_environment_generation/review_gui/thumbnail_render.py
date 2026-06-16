# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""USD viewport thumbnail rendering for the review GUI SimApp sidecar."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvInitialGraphSpec

THUMBNAIL_CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache" / "llm_env_gen_thumbnails"


def _render_thumbnails_with_app(app, spec: ArenaEnvInitialGraphSpec) -> dict[str, Path]:
    """Resolve each node's USD via ``AssetRegistry``, render cache-misses, return PNG paths.

    ``app`` must already be a booted ``SimulationApp``. The caller owns the
    lifecycle (Kit may turn ``app.close()`` into ``os._exit(0)`` — that's why
    the sidecar holds the only reference and closes it inside its ``finally``).

    Returns ``{node.id: png_path}`` for nodes whose asset USD could be located
    *and* whose PNG exists on disk (either from the persistent cache under
    ``THUMBNAIL_CACHE_DIR`` or freshly rendered into the cache by
    :func:`_capture_usd_thumbnails`). Missing entries fall through to the
    placeholder in :func:`_render_node_thumbnail`, so a partial failure (one
    bad asset) never breaks the rest of the page.

    We return ``Path`` rather than ``bytes`` so the sidecar protocol can ship
    just the filenames over its stdin/stdout pipe (a few hundred bytes of JSON
    instead of multiple MB of base64 PNG data). The parent reads the bytes
    itself off the shared filesystem cache.

    Ordering matters: ``SimulationApp`` MUST be launched before any
    ``AssetRegistry`` access, because ``ensure_assets_registered()`` imports
    isaaclab asset modules which transitively load ``pxr``. ``pxr`` loaded
    before ``AppLauncher`` puts Kit's extension manager into an unrecoverable
    state ("extension class wrapper for base class ... has not been created
    yet"). This is the same root cause we fixed for the pytest suite.
    """
    asset_paths = _resolve_node_usd_paths(spec)
    if not asset_paths:
        print("[thumbnail_render] no asset USD paths resolved; skipping thumbnail rendering.", file=sys.stderr)
        return {}

    THUMBNAIL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Split into cache-hits vs to-render. Cache key is sha1(usd_path) so
    # the same USD across multiple envs / nodes hits the same PNG.
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
        # ``_capture_usd_thumbnails`` still returns ``{node_id: bytes}``, but
        # we only use it as a presence signal here — the same call also wrote
        # the PNG to ``cache_path`` as a side effect, which is what we return.
        captured = _capture_usd_thumbnails(app, to_render)
        for node_id, (_usd_path, cache_path) in to_render.items():
            if node_id in captured and cache_path.exists() and cache_path.stat().st_size > 0:
                resolved[node_id] = cache_path
    else:
        print(f"[thumbnail_render] all {len(resolved)} thumbnail(s) served from cache.", file=sys.stderr)

    return resolved


def _sidecar_launch_args() -> argparse.Namespace:
    """AppLauncher args for the review GUI sidecar (Kit UI + viewport capture)."""
    return argparse.Namespace(visualizer=["kit"], enable_cameras=True, livestream=-1)


def _launch_simulation_app():
    """Boot Isaac Sim's ``SimulationApp`` with the Kit visualizer, or ``None`` on failure.

    Kept as a tiny helper so the call site can lazy-import inside this
    function — module-level import of ``simulation_app`` would drag Kit
    into every invocation, including ``--help``.
    """
    try:
        # Lazy-import: keeps the default ``review_graph`` invocation Kit-free.
        from isaaclab_arena.utils.isaaclab_utils.simulation_app import get_app_launcher  # noqa: PLC0415

        return get_app_launcher(_sidecar_launch_args()).app
    except Exception as exc:
        print(f"[thumbnail_render] SimulationApp launch failed: {exc}", file=sys.stderr)
        return None


def _resolve_node_usd_paths(spec: ArenaEnvInitialGraphSpec) -> dict[str, str]:
    """Map ``node.id → usd_path`` via :class:`AssetRegistry`, skipping unresolvable nodes.

    Tries two lookup strategies in order:

    1. Class-attribute ``cls.usd_path`` — the convention every ``LibraryObject``
       subclass in ``object_library.py`` follows. No instantiation, cheap.

    2. ``cls().scene_config.robot.spawn.usd_path`` — the convention every
       :class:`EmbodimentBase` subclass uses. Requires instantiating the
       embodiment because the Franka embodiments populate ``scene_config.robot``
       inside ``__init__`` rather than as a class default. Embodiment
       ``__init__`` is light (no Kit / sim required) — it only constructs
       configclass objects.

    This function MUST be called only after ``SimulationApp`` has booted — see
    the docstring of :func:`_render_thumbnails_with_app` for why.
    """
    try:
        from isaaclab_arena.assets.registries import AssetRegistry  # noqa: PLC0415
    except Exception as exc:
        print(f"[thumbnail_render] AssetRegistry import failed: {exc}", file=sys.stderr)
        return {}

    registry = AssetRegistry()
    paths: dict[str, str] = {}
    for node in spec.nodes:
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
    """Return the asset's root USD path, or ``None`` if not extractable.

    See :func:`_resolve_node_usd_paths` for the two strategies tried in order.
    """
    # Strategy 1: ``LibraryObject`` convention.
    usd_path = getattr(cls, "usd_path", None)
    if usd_path:
        return usd_path

    # Strategy 2: ``EmbodimentBase`` convention. Walk
    # ``instance.scene_config.robot.spawn.usd_path``. We instantiate with no
    # args; every embodiment ``__init__`` defaults all parameters.
    # NoEmbodiment legitimately has no robot — its instance.scene_config
    # exists but ``.robot`` is absent / None, so the getattr chain returns
    # None and we silently fall through.
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


def _capture_usd_thumbnails(app, to_render: dict[str, tuple[str, Path]]) -> dict[str, bytes]:
    """Capture all queued USDs under one already-booted ``SimulationApp``.

    Deduplicates by ``usd_path`` so the same USD shared by multiple nodes is
    only rendered once and the bytes are fanned back out.
    """
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
    """Open ``usd_path`` directly as the stage, frame the camera, capture PNG.

    Opening the USD as the stage root (rather than ``new_stage`` + reference
    wrapper) is what makes viewport capture actually produce a file in
    headless mode — Kit's viewport machinery binds to the just-opened stage
    cleanly, whereas a referenced sub-stage left the render product empty in
    every test we tried. The trade-off is that we lose isolation between
    captures (each call replaces the stage), but Kit handles that fine
    because we call ``open_stage`` again on the next asset.
    """
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

    # Wait for textures / payloads / Nucleus fetches to settle before framing.
    _wait_for_stage_load(app, ctx)

    # Standalone object USDs (avocado, bowl, ...) ship no lights, so a viewport
    # capture renders them as a near-black silhouette against the dark skybox
    # — that's the "blank thumbnail" symptom. Complete scene USDs (maple table)
    # already include their own lighting, so this is a no-op for them.
    _ensure_default_lighting(stage)

    # Use the default prim if present, otherwise the pseudo-root, for framing.
    target_prim = stage.GetDefaultPrim()
    if not target_prim or not target_prim.IsValid():
        target_prim = stage.GetPrimAtPath(Sdf.Path("/"))

    viewport = get_active_viewport()

    # Use Kit's own ``frame_viewport_prims`` (the "F"-key equivalent / ``FramePrimsCommand``)
    # so we go through the viewport camera controller. Manually editing the
    # ``/OmniverseKit_Persp`` xform op directly worked sometimes but Kit's
    # camera controller treats /OmniverseKit_Persp as an internal state and
    # silently overrode our edits for small assets — that's why avocado / bowl
    # captured as tiny specks even with the right math. Letting Kit do the
    # framing is both correct and avoids us re-implementing the math.
    framed = frame_viewport_prims(viewport, prims=[str(target_prim.GetPath())])
    if not framed:
        print(f"[thumbnail_render]   warning: frame_viewport_prims failed for {usd_path}", file=sys.stderr)

    # Settle Hydra after camera change so the captured frame matches the new pose.
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
    """Pump frames until ``usd_context.get_stage_loading_status()`` reports nothing pending.

    Returns after stage load completes or after the budget is exhausted. We
    also need a few extra frames after the count goes to zero so material
    binding / texture upload finishes — they don't show up in the load count.
    """
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
    """Pump ``app.update()`` until the capture PNG lands on disk (or we time out).

    Kit's capture future is fulfilled inside its async loop during
    ``app.update()``, but future completion doesn't always coincide with the
    file being flushed — checking the file directly is the most reliable
    completion signal. We also keep the future-based fast path so a
    successful capture doesn't have to wait for the file system to settle.
    """
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
            # Future is done but file might still be flushing — give it a few frames.
            for _ in range(15):
                app.update()
                if cache_path.exists() and cache_path.stat().st_size > 0:
                    return
            return


def _ensure_default_lighting(stage) -> None:
    """Add a dome + key distant light if the stage has none.

    Without this, standalone object USDs (which don't ship their own lights)
    render as a near-black silhouette. We skip the addition if any
    ``UsdLuxLight``-derived prim already exists on the stage to avoid
    double-lighting scenes like the maple table that bake in their own rig.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdLux  # noqa: PLC0415

    for prim in stage.Traverse():
        if (
            prim.HasAPI(UsdLux.LightAPI)
            or prim.IsA(UsdLux.BoundableLightBase)
            or prim.IsA(UsdLux.NonboundableLightBase)
        ):
            return

    # Soft hemispherical fill so the asset is visible from any angle, plus a
    # weak directional key for shape definition. Intensities are tuned for
    # OmniPBR / RTX defaults; tweak if asset libraries adopt darker materials.
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/_ReviewDomeLight"))
    dome.CreateIntensityAttr(800.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    key = UsdLux.DistantLight.Define(stage, Sdf.Path("/_ReviewKeyLight"))
    key.CreateIntensityAttr(2500.0)
    key.CreateAngleAttr(2.0)
    # Aim the key roughly from the camera's 3/4 angle so the lit side faces
    # the viewport.
    key_xformable = UsdGeom.Xformable(key.GetPrim())
    key_xformable.ClearXformOpOrder()
    rot = key_xformable.AddRotateXYZOp()
    rot.Set(Gf.Vec3f(-45.0, 30.0, 0.0))
