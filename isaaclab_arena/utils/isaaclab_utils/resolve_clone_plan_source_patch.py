# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime patch for ``isaaclab.cloner.cloner_utils.resolve_clone_plan_source``.

The pinned Isaac Lab raises when a prim path is owned by two clone-plan destination
templates, even when one is nested in the other. ObjectSets trigger it: they force a
per-cfg clone plan, so a camera under the robot matches both ``/World/envs/env_{}/Robot``
and its own template. Ports upstream #5929, which prefers the nearest owner.
"""

from __future__ import annotations

import contextlib
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.cloner import ClonePlan

_PATCHED_MARKER = "_isaaclab_arena_nested_clone_template_patch"


def _resolve_clone_plan_source(path_expr: str, plan: ClonePlan) -> tuple[str, str, str] | None:
    """Resolve a destination path expression to its owning source path. E.g. for camera at
    /World/envs/env_{}/Robot/panda_link0/external_camera, return its exact source path instead of /World/envs/env_{}/Robot.

    Args:
        path_expr: Destination-side path expression.
        plan: Active clone plan.

    Returns:
        A ``(source_path, destination_glob, asset_suffix)`` tuple, or None when no source path owns
        ``path_expr`` so callers fall back to direct stage resolution.
    """
    from isaaclab.cloner.cloner_utils import get_suffix

    # Collect all candidates that own the path expression.
    candidates: list[tuple[str, str, int]] = []
    for source_index, destination_template in enumerate(plan.destinations):
        if "{}" in destination_template:
            suffix = get_suffix(path_expr, destination_template)
            if suffix is not None:
                candidates.append((destination_template, suffix, source_index))

    if not candidates:
        return None
    # Pick the path with the shortest suffix.
    min_suffix_len = min(len(suffix) for _, suffix, _ in candidates)
    owning_templates = {template for template, suffix, _ in candidates if len(suffix) == min_suffix_len}
    # It shall have only one owning template.
    if len(owning_templates) > 1:
        raise ValueError(f"path_expr {path_expr!r}: matches multiple destination templates {sorted(owning_templates)}.")

    matching_template = next(iter(owning_templates))
    matching_rows = [index for template, _, index in candidates if template == matching_template]
    matching_suffix = next(suffix for template, suffix, _ in candidates if template == matching_template)
    # It shall cover all envs.
    if not plan.clone_mask[matching_rows].any(dim=0).all():
        raise NotImplementedError(
            f"path_expr {path_expr!r}: partial-env heterogeneous coverage is unsupported;"
            " matching rows must collectively cover all envs."
        )
    # override the destination glob and asset suffix to the matching template and suffix.
    return plan.sources[matching_rows[0]], matching_template.replace("{}", "*"), matching_suffix or ""


def installed_resolver_handles_nesting() -> bool:
    """Return whether the installed Isaac Lab already resolves nested destination templates."""
    import torch

    from isaaclab.cloner import ClonePlan
    from isaaclab.cloner.cloner_utils import resolve_clone_plan_source

    plan = ClonePlan(
        sources=("/World/envs/env_0/Robot", "/World/envs/env_0/Robot/link/camera"),
        destinations=("/World/envs/env_{}/Robot", "/World/envs/env_{}/Robot/link/camera"),
        clone_mask=torch.ones((2, 1), dtype=torch.bool),
    )
    try:
        resolve_clone_plan_source("/World/envs/env_0/Robot/link/camera", plan)
    except ValueError:
        return False
    return True


def patch_resolve_clone_plan_source() -> bool:
    """Install the patched resolver, rebinding every module that holds the stock function.

    Returns:
        Whether this call installed the patch.
    """
    import isaaclab.cloner.cloner_utils as cloner_utils

    original = cloner_utils.resolve_clone_plan_source
    if getattr(original, _PATCHED_MARKER, False):
        return False
    if installed_resolver_handles_nesting():
        return False

    # Import the known importers so their stale bindings are visible to the sweep below.
    import isaaclab.cloner  # noqa: F401
    import isaaclab.sim.utils.queries  # noqa: F401

    setattr(_resolve_clone_plan_source, _PATCHED_MARKER, True)
    patched_modules = []
    for module in list(sys.modules.values()):
        with contextlib.suppress(Exception):
            if getattr(module, "resolve_clone_plan_source", None) is original:
                module.resolve_clone_plan_source = _resolve_clone_plan_source
                patched_modules.append(module.__name__)

    assert patched_modules, "Expected at least isaaclab.cloner.cloner_utils to hold the stock resolver."
    print(f"Patched resolve_clone_plan_source for nested clone templates in: {', '.join(sorted(patched_modules))}")
    return True
