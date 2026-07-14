# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Kit viewport PNG capture for review GUI node thumbnails (SimApp subprocess only)."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import omni.usd
from omni.kit.viewport.utility import frame_viewport_prims, get_active_viewport
from pxr import Gf, Sdf, UsdGeom, UsdLux

from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.asset_usd import (
    AabbDimensionsM,
    ObjectReferenceUsdTarget,
    object_reference_cache_key,
    resolve_background_viewer_cfgs,
    resolve_node_aabb_dimensions_m,
    resolve_node_usd_paths,
    resolve_object_reference_usd_targets,
    viewer_cfg_for_asset_spec,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.kit_viewport import (
    PRE_CAPTURE_UPDATES,
    capture_viewport_png,
    pump_app,
    thumbnail_cache_dir,
    wait_for_stage_load,
)

COLLIDER_SELECTION_PUMP_UPDATES = PRE_CAPTURE_UPDATES + 3


@dataclass
class _UsdSnapshotJob:
    """Queued captures that share one opened parent USD stage."""

    usd_path: str
    viewer_cfg: object | None = None
    asset_captures: list[tuple[str, Path]] = field(default_factory=list)
    ref_captures: list[tuple[str, ObjectReferenceUsdTarget, Path]] = field(default_factory=list)


def _usd_cache_key(usd_path: str) -> str:
    """Return a stable short hash for caching thumbnails keyed by USD path."""
    return hashlib.sha1(usd_path.encode("utf-8")).hexdigest()[:16]


def render_thumbnails_with_app(app, spec: ArenaEnvGraphSpec) -> tuple[dict[str, Path], dict[str, AabbDimensionsM]]:
    """Render cache-missed node thumbnails and return png paths plus AABB sizes in meters."""
    asset_paths = resolve_node_usd_paths(spec)
    reference_targets = resolve_object_reference_usd_targets(spec)
    viewer_cfgs = resolve_background_viewer_cfgs(spec)
    if not asset_paths and not reference_targets:
        print("[thumbnail_capture] no asset USD paths resolved; skipping thumbnail rendering.", file=sys.stderr)
        return {}, {}

    cache_dir = thumbnail_cache_dir()

    resolved: dict[str, Path] = {}
    to_render: dict[str, tuple[str, Path, object | None]] = {}
    for node_id, usd_path in asset_paths.items():
        cache_path = cache_dir / f"{_usd_cache_key(usd_path)}.png"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            resolved[node_id] = cache_path
        else:
            to_render[node_id] = (usd_path, cache_path, viewer_cfgs.get(node_id))

    reference_to_render: dict[str, tuple[ObjectReferenceUsdTarget, Path]] = {}
    for node_id, target in reference_targets.items():
        cache_path = cache_dir / f"{object_reference_cache_key(target.usd_path, target.relative_prim_path)}.png"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            resolved[node_id] = cache_path
        else:
            reference_to_render[node_id] = (target, cache_path)

    if to_render or reference_to_render:
        print(
            f"[thumbnail_capture] rendering {len(to_render)} asset and "
            f"{len(reference_to_render)} object_reference thumbnail(s) "
            f"(reusing {len(resolved)} from cache)...",
            file=sys.stderr,
        )
        jobs = _build_usd_snapshot_jobs(spec, to_render, reference_to_render)
        captured = _capture_usd_snapshot_jobs(app, jobs)
        for node_id, cache_path in [
            *((nid, cp) for nid, (_usd, cp, _cfg) in to_render.items()),
            *((nid, cp) for nid, (_tgt, cp) in reference_to_render.items()),
        ]:
            if node_id in captured and cache_path.exists() and cache_path.stat().st_size > 0:
                resolved[node_id] = cache_path
    else:
        print(f"[thumbnail_capture] all {len(resolved)} thumbnail(s) served from cache.", file=sys.stderr)

    return resolved, resolve_node_aabb_dimensions_m(spec)


def _build_usd_snapshot_jobs(
    spec: ArenaEnvGraphSpec,
    to_render: dict[str, tuple[str, Path, object | None]],
    reference_to_render: dict[str, tuple[ObjectReferenceUsdTarget, Path]],
) -> list[_UsdSnapshotJob]:
    """Group asset and object_reference captures by parent USD path."""
    jobs_by_usd: dict[str, _UsdSnapshotJob] = {}

    for node_id, (usd_path, cache_path, viewer_cfg) in to_render.items():
        job = jobs_by_usd.setdefault(usd_path, _UsdSnapshotJob(usd_path=usd_path))
        if viewer_cfg is not None:
            job.viewer_cfg = viewer_cfg
        job.asset_captures.append((node_id, cache_path))

    ref_by_id = {ref.id: ref for ref in (spec.object_references or [])}
    for node_id, (target, cache_path) in reference_to_render.items():
        job = jobs_by_usd.setdefault(target.usd_path, _UsdSnapshotJob(usd_path=target.usd_path))
        job.ref_captures.append((node_id, target, cache_path))
        ref = ref_by_id.get(node_id)
        if ref is not None and job.viewer_cfg is None:
            try:
                parent_spec = spec._asset_by_id(ref.parent_id)
                job.viewer_cfg = viewer_cfg_for_asset_spec(parent_spec)
            except KeyError:
                pass

    return list(jobs_by_usd.values())


def _capture_usd_snapshot_jobs(app, jobs: list[_UsdSnapshotJob]) -> dict[str, bytes]:
    """Open each parent USD once and capture queued asset plus object_reference snapshots."""
    out: dict[str, bytes] = {}
    for job in jobs:
        try:
            out.update(_capture_usd_snapshot_job(app, job))
        except Exception as exc:
            print(f"[thumbnail_capture]   render failed for {job.usd_path}: {exc}", file=sys.stderr)
    return out


def _capture_usd_snapshot_job(app, job: _UsdSnapshotJob) -> dict[str, bytes]:
    """Capture all queued snapshots for one opened parent USD."""
    out: dict[str, bytes] = {}
    ctx = omni.usd.get_context()
    if not ctx.open_stage(job.usd_path):
        print(f"[thumbnail_capture]   open_stage failed: {job.usd_path}", file=sys.stderr)
        return out
    stage = ctx.get_stage()
    wait_for_stage_load(app, ctx)
    _ensure_default_lighting(stage)

    if job.asset_captures:
        cache_path = job.asset_captures[0][1]
        png_bytes = _capture_stage_snapshot(
            app,
            stage,
            cache_path,
            viewer_cfg=job.viewer_cfg,
            collision_paths=None,
        )
        if png_bytes:
            for node_id, _cache_path in job.asset_captures:
                out[node_id] = png_bytes

    if job.ref_captures:
        # Object references are framed tightly on their own prim via frame_viewport_prims;
        # the background's viewer_cfg camera (e.g. a kitchen eye/lookat aimed at a wall)
        # must not be applied here or the capture points away from the target.
        _configure_collision_mesh_visualization()

        for node_id, target, cache_path in job.ref_captures:
            root_path = _absolute_prim_path(stage, target.relative_prim_path)
            collision_paths = _collect_collision_prim_paths(stage, root_path)
            _select_collision_prims(app, collision_paths)
            viewport = get_active_viewport()
            framed = frame_viewport_prims(viewport, prims=[root_path])
            if not framed:
                print(
                    f"[thumbnail_capture]   warning: frame_viewport_prims failed for {root_path}",
                    file=sys.stderr,
                )
            png_bytes = capture_viewport_png(app, cache_path, pre_capture_updates=PRE_CAPTURE_UPDATES)
            if png_bytes:
                out[node_id] = png_bytes
            else:
                print(
                    f"[thumbnail_capture]   capture produced no file for {target.relative_prim_path}: {cache_path}",
                    file=sys.stderr,
                )

    return out


def _capture_stage_snapshot(
    app,
    stage,
    cache_path: Path,
    *,
    viewer_cfg,
    collision_paths: list[str] | None,
) -> bytes | None:
    """Capture the active viewport for an already-open stage."""
    if viewer_cfg is not None:
        _apply_viewer_cfg(app, viewer_cfg)
    else:
        target_prim = stage.GetDefaultPrim()
        if not target_prim or not target_prim.IsValid():
            target_prim = stage.GetPrimAtPath(Sdf.Path("/"))
        viewport = get_active_viewport()
        framed = frame_viewport_prims(viewport, prims=[str(target_prim.GetPath())])
        if not framed:
            print(
                f"[thumbnail_capture]   warning: frame_viewport_prims failed for {target_prim.GetPath()}",
                file=sys.stderr,
            )

    if collision_paths:
        _configure_collision_mesh_visualization()
        _select_collision_prims(app, collision_paths)

    png_bytes = capture_viewport_png(app, cache_path)
    if png_bytes is not None:
        return png_bytes
    print(f"[thumbnail_capture]   capture produced no file: {cache_path}", file=sys.stderr)
    return None


def _apply_viewer_cfg(app, viewer_cfg) -> None:
    """Point the active viewport camera at ``viewer_cfg`` eye/lookat (world frame)."""
    from isaacsim.core.utils.viewports import set_camera_view

    set_camera_view(list(viewer_cfg.eye), list(viewer_cfg.lookat))
    pump_app(app, count=PRE_CAPTURE_UPDATES)


def _render_one_usd(app, usd_path: str, cache_path: Path, *, viewer_cfg=None) -> bytes | None:
    """Open ``usd_path`` and capture one asset snapshot (debug helper)."""
    job = _UsdSnapshotJob(usd_path=usd_path, viewer_cfg=viewer_cfg, asset_captures=[("asset", cache_path)])
    captured = _capture_usd_snapshot_job(app, job)
    return captured.get("asset")


def _render_object_reference(
    app,
    target: ObjectReferenceUsdTarget,
    cache_path: Path,
    *,
    viewer_cfg=None,
) -> bytes | None:
    """Open the parent USD and capture one object_reference snapshot (debug helper)."""
    job = _UsdSnapshotJob(
        usd_path=target.usd_path,
        viewer_cfg=viewer_cfg,
        ref_captures=[("ref", target, cache_path)],
    )
    captured = _capture_usd_snapshot_job(app, job)
    return captured.get("ref")


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


def _configure_collision_mesh_visualization() -> None:
    """Enable viewport Show By Type → Physics → Colliders → Selected."""
    import carb.settings

    # VisualizerMode: 0=None, 1=Selected, 2=All. Use Selected so only the picked
    # object_reference subtree shows collider wireframes, not the whole scene.
    settings = carb.settings.get_settings()
    settings.set_bool("/persistent/physics/visualizationCollisionMesh", True)
    settings.set_int("/persistent/physics/visualizationDisplayColliders", 1)


def _select_collision_prims(app, selected_prim_paths: list[str]) -> None:
    """Select collision prims and pump Kit so the viewport overlay updates."""
    if selected_prim_paths:
        omni.usd.get_context().get_selection().set_selected_prim_paths(selected_prim_paths, True)
    pump_app(app, count=COLLIDER_SELECTION_PUMP_UPDATES)


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
