# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Kit app pump and viewport PNG capture helpers (SimApp subprocess only)."""

from __future__ import annotations

from pathlib import Path

from isaaclab_arena.assets.asset_cache import get_arena_asset_cache_dir

PRE_CAPTURE_UPDATES = 5
CAPTURE_DONE_TAIL_UPDATES = 3
CAPTURE_WAIT_MAX_UPDATES = 10
THUMBNAIL_CACHE_SUBDIR = "agentic_env_gen_thumbnails"
SIM_PREVIEW_CACHE_SUBDIR = "agentic_env_gen_sim_preview"


def review_gui_cache_dir(subdir: str) -> Path:
    """Return a mkdir'd cache directory for review GUI SimApp viewport PNGs."""
    cache_dir = get_arena_asset_cache_dir().parent / subdir
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def thumbnail_cache_dir() -> Path:
    """Cache directory for per-node USD thumbnail PNGs."""
    return review_gui_cache_dir(THUMBNAIL_CACHE_SUBDIR)


def sim_preview_cache_dir() -> Path:
    """Cache directory for sim-preview rollout viewport frames."""
    return review_gui_cache_dir(SIM_PREVIEW_CACHE_SUBDIR)


def pump_app(app, *, count: int = 1) -> None:
    """Pump Kit render/UI updates without advancing physics simulation."""
    import carb.settings

    settings = carb.settings.get_settings()
    prev_play = settings.get("/app/player/playSimulations")
    settings.set_bool("/app/player/playSimulations", False)
    for _ in range(count):
        app.update()
    if prev_play is not None:
        settings.set_bool("/app/player/playSimulations", bool(prev_play))
    else:
        settings.set_bool("/app/player/playSimulations", True)


def wait_for_stage_load(app, usd_context, max_updates: int = 600) -> None:
    """Pump frames until stage loading settles (plus a short post-settle tail)."""
    settled = 0
    for _ in range(max_updates):
        pump_app(app)
        try:
            _msg, loading_count, loaded_count = usd_context.get_stage_loading_status()
        except Exception:
            return
        if loading_count == 0 and loaded_count == 0:
            settled += 1
            if settled >= CAPTURE_DONE_TAIL_UPDATES:
                return
        else:
            settled = 0


def wait_for_capture(app, capture_obj, cache_path: Path, max_updates: int = CAPTURE_WAIT_MAX_UPDATES) -> None:
    """Pump render updates until the capture PNG exists or the budget expires."""
    if capture_obj is None:
        for _ in range(max_updates):
            pump_app(app)
            if cache_path.exists() and cache_path.stat().st_size > 0:
                return
        return

    # File existence is the reliable capture-completion signal.
    future = getattr(capture_obj, "future", None)

    for _ in range(max_updates):
        pump_app(app)
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return
        if future is not None and future.done():
            for _ in range(CAPTURE_DONE_TAIL_UPDATES):
                pump_app(app)
                if cache_path.exists() and cache_path.stat().st_size > 0:
                    return
            return


def capture_viewport_png(
    app,
    cache_path: Path,
    *,
    max_updates: int = CAPTURE_WAIT_MAX_UPDATES,
    pre_capture_updates: int = PRE_CAPTURE_UPDATES,
) -> bytes | None:
    """Capture the active Kit viewport to ``cache_path`` and return PNG bytes."""
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    viewport = get_active_viewport()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pump_app(app, count=pre_capture_updates)
    capture_obj = capture_viewport_to_file(viewport, str(cache_path))
    wait_for_capture(app, capture_obj, cache_path, max_updates=max_updates)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_bytes()
    return None
