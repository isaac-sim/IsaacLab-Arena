# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Kit viewport PNG capture for review GUI node thumbnails (SimApp subprocess only)."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import omni.usd
from omni.kit.viewport.utility import frame_viewport_prims, get_active_viewport
from pxr import Gf, Sdf, UsdGeom, UsdLux

from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import instantiate_assets_from_spec
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.asset_usd import (
    AabbDimensionsM,
    absolute_prim_path,
    object_reference_cache_key,
    resolve_aabb_dimensions_m,
    resolve_node_usd_paths,
    usd_cache_key,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.kit_viewport import (
    PRE_CAPTURE_UPDATES,
    capture_viewport_png,
    pump_app,
    thumbnail_cache_dir,
    wait_for_stage_load,
)


@dataclass
class _UsdSnapshotJob:
    """Queued captures that share one opened parent USD stage."""

    usd_path: str
    viewer_cfg: object | None = None
    asset_captures: list[tuple[str, Path]] = field(default_factory=list)
    ref_captures: list[tuple[str, str, Path]] = field(default_factory=list)


def render_thumbnails_with_app(app, spec: ArenaEnvGraphSpec) -> tuple[dict[str, Path], dict[str, AabbDimensionsM]]:
    """Render cache-missed node thumbnails and return png paths plus AABB sizes in meters."""
    assets_by_node_id = instantiate_assets_from_spec(spec, AssetRegistry())
    # Exclude embodiment from thumbnail rendering.
    assets_by_node_id.pop(spec.embodiment.id)
    asset_node_ids = [spec.background.id, *(obj.id for obj in spec.objects)]
    asset_paths = resolve_node_usd_paths(assets_by_node_id, asset_node_ids)
    background_viewer_cfg = assets_by_node_id[spec.background.id].get_viewer_cfg()

    cache_dir = thumbnail_cache_dir()

    thumbnail_paths: dict[str, Path] = {}
    jobs_by_usd: dict[str, _UsdSnapshotJob] = {}
    asset_render_count = 0
    ref_render_count = 0
    for node_id, usd_path in asset_paths.items():
        cache_path = cache_dir / f"{usd_cache_key(usd_path)}.png"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            thumbnail_paths[node_id] = cache_path
        else:
            job = jobs_by_usd.setdefault(usd_path, _UsdSnapshotJob(usd_path=usd_path))
            if node_id == spec.background.id:
                job.viewer_cfg = background_viewer_cfg
            job.asset_captures.append((node_id, cache_path))
            asset_render_count += 1

    for ref in spec.object_references or []:
        if ref.prim_path is None:
            continue
        usd_path = asset_paths.get(ref.parent_id)
        if not usd_path:
            continue
        relative_prim_path = ref.prim_path.lstrip("/")
        cache_path = cache_dir / f"{object_reference_cache_key(usd_path, relative_prim_path)}.png"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            thumbnail_paths[ref.id] = cache_path
        else:
            job = jobs_by_usd.setdefault(usd_path, _UsdSnapshotJob(usd_path=usd_path))
            if ref.parent_id == spec.background.id:
                job.viewer_cfg = background_viewer_cfg
            job.ref_captures.append((ref.id, relative_prim_path, cache_path))
            ref_render_count += 1

    jobs = list(jobs_by_usd.values())
    if not asset_paths and not jobs and not thumbnail_paths:
        print("[thumbnail_capture] no asset USD paths resolved; skipping thumbnail rendering.", file=sys.stderr)
        return {}, {}

    if jobs:
        print(
            f"[thumbnail_capture] rendering {asset_render_count} asset and "
            f"{ref_render_count} object_reference thumbnail(s) "
            f"(reusing {len(thumbnail_paths)} from cache)...",
            file=sys.stderr,
        )
        captured = _capture_usd_snapshot_jobs(app, jobs)
        for node_id, cache_path in [
            *((nid, cp) for job in jobs for nid, cp in job.asset_captures),
            *((nid, cp) for job in jobs for nid, _rel, cp in job.ref_captures),
        ]:
            if node_id in captured and cache_path.exists() and cache_path.stat().st_size > 0:
                thumbnail_paths[node_id] = cache_path
    else:
        print(f"[thumbnail_capture] all {len(thumbnail_paths)} thumbnail(s) served from cache.", file=sys.stderr)

    aabb_dimensions_m = resolve_aabb_dimensions_m(assets_by_node_id)
    return thumbnail_paths, aabb_dimensions_m


# Capture orchestration — open each queued USD stage once and write PNGs.


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
        )
        if png_bytes:
            for node_id, _cache_path in job.asset_captures:
                out[node_id] = png_bytes

    if job.ref_captures:
        # Seed the camera with the background viewer_cfg orientation before framing.
        # frame_viewport_prims preserves the incoming view direction.
        if job.viewer_cfg is not None:
            _apply_viewer_cfg(app, job.viewer_cfg)
        _set_collision_mesh_visualization(enabled=True)
        try:
            for node_id, relative_prim_path, cache_path in job.ref_captures:
                root_path = absolute_prim_path(stage, relative_prim_path)
                # Selecting the subtree root is enough; Kit shows colliders for all prims below it.
                omni.usd.get_context().get_selection().set_selected_prim_paths([root_path], True)
                pump_app(app, count=PRE_CAPTURE_UPDATES)
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
                        f"[thumbnail_capture]   capture produced no file for {relative_prim_path}: {cache_path}",
                        file=sys.stderr,
                    )
        finally:
            # The collider-viz carb settings are persistent, so disable them and clear the
            # selection to avoid leaking collider wireframes into later asset captures.
            omni.usd.get_context().get_selection().clear_selected_prim_paths()
            _set_collision_mesh_visualization(enabled=False)

    return out


def _capture_stage_snapshot(
    app,
    stage,
    cache_path: Path,
    *,
    viewer_cfg,
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

    png_bytes = capture_viewport_png(app, cache_path)
    if png_bytes is not None:
        return png_bytes
    print(f"[thumbnail_capture]   capture produced no file: {cache_path}", file=sys.stderr)
    return None


# Viewport and stage setup — camera, lighting, and collision-mesh overlay.


def _apply_viewer_cfg(app, viewer_cfg) -> None:
    """Point the active viewport camera at ``viewer_cfg`` eye/lookat (world frame)."""
    from isaacsim.core.utils.viewports import set_camera_view

    set_camera_view(list(viewer_cfg.eye), list(viewer_cfg.lookat))
    pump_app(app, count=PRE_CAPTURE_UPDATES)


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


def _set_collision_mesh_visualization(*, enabled: bool) -> None:
    """Toggle viewport Show By Type → Physics → Colliders → Selected."""
    import carb.settings

    # VisualizerMode: 0=None, 1=Selected, 2=All. Use Selected so only the picked
    # object_reference subtree shows collider wireframes, not the whole scene.
    settings = carb.settings.get_settings()
    settings.set_bool("/persistent/physics/visualizationCollisionMesh", enabled)
    settings.set_int("/persistent/physics/visualizationDisplayColliders", 1 if enabled else 0)
