# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import html as html_lib


def format_aabb_dimensions_m(dims: tuple[float, float, float]) -> str:
    """Format axis-aligned bounding box size as ``x × y × z m``."""
    x, y, z = dims
    return f"{x:.3f} × {y:.3f} × {z:.3f} m"


def _render_aabb_dimensions(aabb_dimensions_m: tuple[float, float, float] | None) -> str:
    if aabb_dimensions_m is None:
        return ""
    return f'<span class="thumb-dims">AABB {html_lib.escape(format_aabb_dimensions_m(aabb_dimensions_m))}</span>'


def _render_unsupported_thumbnail(registry_name: str) -> str:
    return f"""<div class="thumb-wrap">
  <div class="thumb thumb-unsupported">
    <span class="thumb-initial">PR</span>
    <span class="thumb-name">{html_lib.escape(registry_name)}</span>
    <span class="thumb-note">Resolve prim_path to enable collision-mesh snapshot</span>
  </div>
</div>"""


def render_asset_thumbnail(
    registry_name: str,
    png_bytes: bytes | None = None,
    aabb_dimensions_m: tuple[float, float, float] | None = None,
    *,
    is_object_reference: bool = False,
) -> str:
    """Per-asset thumbnail: USD capture if available, else two-letter placeholder."""
    if is_object_reference and png_bytes is None:
        return _render_unsupported_thumbnail(registry_name)

    dims_html = _render_aabb_dimensions(aabb_dimensions_m)
    collision_note = (
        '<span class="thumb-note">Collision mesh preview</span>' if is_object_reference and png_bytes else ""
    )
    if png_bytes:
        b64 = base64.b64encode(png_bytes).decode("ascii")
        title = html_lib.escape(registry_name, quote=True)
        src = f"data:image/png;base64,{b64}"
        return (
            '<div class="thumb-wrap">'
            '<div class="thumb thumb-rendered thumb-zoomable" role="button" tabindex="0" '
            f'aria-label="Zoom {title} snapshot" data-zoom-src="{src}" data-zoom-title="{title}">'
            f'<img src="{src}" alt="{title} thumbnail">'
            f'<span class="thumb-name">{html_lib.escape(registry_name)}</span>'
            f"{collision_note}"
            "</div>"
            f"{dims_html}"
            "</div>"
        )
    initial = (registry_name[:2] if registry_name else "?").upper()
    return f"""<div class="thumb-wrap">
  <div class="thumb">
    <span class="thumb-initial">{html_lib.escape(initial)}</span>
    <span class="thumb-name">{html_lib.escape(registry_name)}</span>
  </div>
  {dims_html}
</div>"""
